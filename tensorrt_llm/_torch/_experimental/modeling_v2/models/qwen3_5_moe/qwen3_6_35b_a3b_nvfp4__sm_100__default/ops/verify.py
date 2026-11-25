# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convolution and recurrent verification using the runtime's replay buffers."""

import torch

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels import verify_gdn as k
from ..kernels import verify_gdn_tiled
from ..weights import BF16


class ConvolutionVerify(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("one request at width four", {"batch": 1, "width": 4, "slots": 5}),
        Cell("multiple requests at width two", {"batch": 5, "width": 2, "slots": 11}),
    )

    def __call__(self, x, weight, slots, history, windows, width):
        rows = x.shape[0]
        out = torch.empty((rows, 8192), dtype=BF16, device=x.device)
        k.convolution_verify[(rows // width, 32)](
            x,
            weight,
            slots,
            history,
            windows,
            out,
            history.stride(),
            windows.stride(),
            width,
            256,
            num_warps=4,
        )
        return out

    def reference(self, x, weight, slots, history, windows, width):
        batch = slots.shape[0]
        rows = x[:, :8192].reshape(batch, width, 8192).transpose(-1, -2)
        sequence = torch.cat((history[slots.long()], rows), dim=-1)
        frames = sequence.unfold(-1, 4, 1)
        conv = (frames.float() * weight[None, :, None].float()).sum(-1)
        windows[:batch, :width].copy_(frames[..., 1:].transpose(1, 2))
        return torch.nn.functional.silu(conv.transpose(1, 2)).to(BF16).reshape(-1, 8192)


class DeltaVerify(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("small replay, same buffer", {"batch": 1, "width": 4, "previous": (3,)}),
        Cell(
            "mixed replay, rollover, and a fresh slot",
            {"batch": 3, "width": 2, "previous": (3, 15, 0)},
        ),
        Cell(
            "fused convolution with an anchor rollover",
            {"batch": 1, "width": 4, "previous": (15,), "fused": True},
        ),
        Cell(
            "tiled wide verification with rollover",
            {"batch": 64, "width": 2, "previous": (10, 15, 0, 1) * 16},
        ),
    )

    def __call__(
        self,
        qkv,
        ba,
        a_log,
        dt,
        slots,
        replay_slots,
        state,
        delta,
        key,
        decay,
        accepted,
        buffer,
        width,
        max_width,
        conv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ):
        rows = qkv.shape[0]
        out = torch.empty((rows, 32, 128), dtype=BF16, device=qkv.device)
        batch = rows // width
        if conv is None and batch >= 32:
            qk = torch.empty((rows, 16, 256), dtype=torch.float32, device=qkv.device)
            gates = torch.empty((rows, 32, 3), dtype=torch.float32, device=qkv.device)
            verify_gdn_tiled.prepare[(rows, 16)](qkv, ba, a_log, dt, qk, gates, num_warps=4)
            tile = 64
            verify_gdn_tiled.verify[(128 // tile, 32, batch)](
                qkv,
                qk,
                gates,
                slots,
                replay_slots,
                state,
                delta,
                key,
                decay,
                accepted,
                buffer,
                out,
                state.stride(0),
                delta.stride(),
                key.stride(),
                decay.stride(),
                accepted.stride(0),
                buffer.stride(0),
                delta.shape[2],
                max_width,
                width,
                tile,
                num_warps=4,
            )
            return out
        tile = 8 if rows <= 4 else 32
        weight, history, windows = conv if conv else (None, None, None)
        k.delta_verify[(rows // width, 32, 128 // tile)](
            qkv,
            ba,
            a_log,
            dt,
            slots,
            replay_slots,
            state,
            delta,
            key,
            decay,
            accepted,
            buffer,
            out,
            state.stride(0),
            delta.stride(),
            key.stride(),
            decay.stride(),
            accepted.stride(0),
            buffer.stride(0),
            delta.shape[2],
            max_width,
            width,
            tile,
            weight,
            history,
            windows,
            history.stride() if history is not None else (),
            windows.stride() if windows is not None else (),
            conv is not None,
            num_warps=4,
        )
        return out

    def reference(
        self,
        qkv,
        ba,
        a_log,
        dt,
        slots,
        replay_slots,
        state,
        delta,
        key,
        decay,
        accepted,
        buffer,
        width,
        max_width,
        conv: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ):
        if conv is not None:
            weight, history, windows = conv
            qkv = convolution_verify.reference(qkv, weight, slots, history, windows, width)
        q, k_new, value = qkv.float().split((2048, 2048, 4096), dim=-1)
        q = q.reshape(-1, 16, 128)
        k_new = k_new.reshape(-1, 16, 128)
        value = value.reshape(-1, 32, 128)
        q = q * (q.square().sum(-1, keepdim=True) + 1.0e-6).rsqrt() * (128**-0.5)
        k_new = k_new * (k_new.square().sum(-1, keepdim=True) + 1.0e-6).rsqrt()
        beta, a = ba.float().chunk(2, dim=-1)
        beta = beta.sigmoid().to(BF16).float()
        g = -a_log.float().exp() * torch.nn.functional.softplus(a + dt.float())
        output = []
        for request, (slot, replay_slot) in enumerate(zip(slots.tolist(), replay_slots.tolist())):
            past = int(accepted[replay_slot])
            read_buffer = int(buffer[replay_slot])
            s = state[slot].clone()
            previous = torch.zeros(32, dtype=torch.float32, device=qkv.device)
            for token in range(past):
                current = decay[replay_slot, read_buffer, :, token]
                past_key = key[replay_slot, read_buffer, token].float().repeat_interleave(2, 0)
                past_delta = delta[replay_slot, read_buffer, token].float()
                s = (
                    s * (current - previous).exp()[:, None, None]
                    + past_delta[:, :, None] * past_key[:, None]
                )
                previous = current
            write_buffer, offset = read_buffer, past
            if past + max_width > delta.shape[2]:
                state[slot].copy_(s)
                previous = torch.zeros_like(previous)
                write_buffer, offset = 1 - read_buffer, 0
            for t in range(width):
                row = request * width + t
                new_key = k_new[row].repeat_interleave(2, 0)
                previous = previous + g[row]
                s = s * g[row].exp()[:, None, None]
                new_delta = (value[row] - (s * new_key[:, None]).sum(-1)) * beta[row, :, None]
                s = s + new_delta[:, :, None] * new_key[:, None]
                output.append((s * q[row].repeat_interleave(2, 0)[:, None]).sum(-1))
                delta[replay_slot, write_buffer, offset + t].copy_(new_delta)
                key[replay_slot, write_buffer, offset + t].copy_(k_new[row])
                decay[replay_slot, write_buffer, :, offset + t].copy_(previous)
        return torch.stack(output).to(BF16)


convolution_verify = ConvolutionVerify()
delta_verify = DeltaVerify()

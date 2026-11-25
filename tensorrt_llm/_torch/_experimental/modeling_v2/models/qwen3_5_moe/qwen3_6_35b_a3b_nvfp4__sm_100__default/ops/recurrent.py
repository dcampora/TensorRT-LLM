# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gated DeltaNet operations over runtime-owned recurrent slots."""

import torch
import triton
from flashinfer.gdn_prefill import chunk_gated_delta_rule

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels import gated_delta as k
from ..kernels import prefill_stencil
from ..weights import BF16, FP8


class ConvolutionDecode(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("decode with nonconsecutive pool slots", {"rows": 3, "slots": 7}),
        Cell("batched decode", {"rows": 32, "slots": 37}),
    )

    def __call__(self, x, weight, slots, history):
        out = torch.empty((x.shape[0], 8192), device=x.device, dtype=BF16)
        k.conv_decode[(x.shape[0], 32)](
            x,
            weight,
            slots,
            history,
            out,
            *history.stride(),
            256,
        )
        return out

    def reference(self, x, weight, slots, history):
        rows = slots.long()
        window = torch.cat((history[rows], x[:, :8192, None]), dim=-1)
        history[rows] = window[:, :, 1:]
        return torch.nn.functional.silu((window.float() * weight.float()).sum(-1)).to(BF16)


class DeltaDecode(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("decode with manager slot padding", {"rows": 3, "slots": 7}),
        Cell("continuous batch", {"rows": 32, "slots": 37}),
    )

    def __call__(self, qkv, ba, a_log, dt, slots, state):
        out = torch.empty((qkv.shape[0], 32, 128), device=qkv.device, dtype=BF16)
        k.gdn_decode[(qkv.shape[0], 32, 8)](
            qkv,
            ba,
            a_log,
            dt,
            slots,
            state,
            out,
            state.stride(0),
            16,
            num_warps=4,
        )
        return out

    def reference(self, qkv, ba, a_log, dt, slots, state):
        q, key, value = qkv.float().split((2048, 2048, 4096), dim=-1)
        q = q.reshape(-1, 16, 128).repeat_interleave(2, dim=1)
        key = key.reshape(-1, 16, 128).repeat_interleave(2, dim=1)
        value = value.reshape(-1, 32, 128)
        q = q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt() * (128**-0.5)
        key = key * (key.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
        b, a = ba.float().chunk(2, dim=-1)
        beta = b.sigmoid().to(BF16).float()
        decay = (-a_log.float().exp() * torch.nn.functional.softplus(a + dt.float())).exp()
        indices = slots.long()
        carry = state[indices] * decay[..., None, None]
        update = (value - (carry * key[..., None, :]).sum(-1)) * beta[..., None]
        carry = carry + update[..., None] * key[..., None, :]
        state[indices] = carry
        return (carry * q[..., None, :]).sum(-1).to(BF16)


convolution_decode = ConvolutionDecode()
delta_decode = DeltaDecode()


class ConvolutionPrefill(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("fresh contexts including a one-token prompt", {"lengths": (1, 7), "initial": False}),
        Cell("a short continuation and a fresh request", {"lengths": (2, 6), "initial": True}),
        Cell(
            "tiled fresh prefill with a one-token request", {"lengths": (1, 137), "initial": False}
        ),
    )

    def __call__(self, x, weight, rows, cu, slots, history, initial_history):
        count = x.shape[0]
        opts = dict(device=x.device, dtype=BF16)
        q = torch.empty((count, 16, 128), **opts)
        key = torch.empty_like(q)
        value = torch.empty((count, 32, 128), **opts)
        if count >= 64 and initial_history is None:
            tile = 4 if count <= 128 else 8
            prefill_stencil.conv[(triton.cdiv(count, tile), 16)](
                x,
                weight,
                rows,
                cu,
                slots,
                history,
                q,
                key,
                value,
                count,
                history.stride(),
                tile,
                512,
                num_warps=4,
            )
        else:
            k.conv_prefill[(count, 32)](
                x,
                weight,
                rows,
                cu,
                slots,
                initial_history,
                history,
                q,
                key,
                value,
                *history.stride(),
                256,
                initial_history is not None,
                num_warps=4,
            )
        return q, key, value

    def reference(self, x, weight, rows, cu, slots, history, initial_history):
        offsets = cu.tolist()
        output = []
        for request, slot in enumerate(slots.tolist()):
            chunk = x[offsets[request] : offsets[request + 1], :8192].T
            if initial_history is None:
                initial = torch.zeros_like(history[slot])
            else:
                initial = initial_history[request]
            padded = torch.cat((initial, chunk), dim=-1)
            convolved = (padded.float().unfold(-1, 4, 1) * weight.float()[:, None, :]).sum(-1)
            output.append(torch.nn.functional.silu(convolved.T).to(BF16))
            history[slot].copy_(padded[:, -3:])
        q, k, v = torch.cat(output).split((2048, 2048, 4096), dim=-1)
        return q.reshape(-1, 16, 128), k.reshape(-1, 16, 128), v.reshape(-1, 32, 128)


class DeltaGates(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("short context", {"rows": 7}),
        Cell("ordinary prefill", {"rows": 1024}),
    )

    def __call__(self, ba, a_log, dt):
        count = ba.shape[0]
        gates = torch.empty((count, 32), dtype=torch.float32, device=ba.device)
        beta = torch.empty_like(gates)
        k.gdn_gates[(count,)](ba, a_log, dt, gates, beta, False, num_warps=1)
        return gates, beta

    def reference(self, ba, a_log, dt):
        b, a = ba.float().chunk(2, dim=-1)
        decay = (-a_log.float().exp() * torch.nn.functional.softplus(a + dt.float())).exp()
        return decay, b.sigmoid().to(BF16).float()


convolution_prefill = ConvolutionPrefill()
delta_gates = DeltaGates()


def delta_prefill(x, ba, weight, a_log, dt, history, state, step):
    rows = x.shape[0]
    slots = step.slots[: step.contexts]
    cu = step.cu
    initial_history = initial_state = output_slots = None
    initial_slots = slots
    use_initial = step.has_initial is not None
    if use_initial:
        flags = step.has_initial
        initial_history = history[slots.long()].masked_fill(~flags[:, None, None], 0)
        initial_state = state
        initial_slots = torch.where(flags, slots, -1)
        output_slots = slots

    q, key, value = convolution_prefill(
        x, weight, step.request_rows[:rows], cu, slots, history, initial_history
    )
    gates, beta = delta_gates(ba, a_log, dt)
    out, _ = chunk_gated_delta_rule(
        q,
        key,
        value,
        gates,
        beta,
        initial_state=initial_state,
        output_state=state,
        output_final_state=True,
        state_indices=initial_slots,
        output_state_indices=output_slots,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=True,
        backend="flashinfer",
        use_cp=False,
        max_seqlen=step.max_context,
    )
    return out


class DeltaNorm(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("BF16 output before FP8 casting", {"rows": 3, "quantized": False}),
        Cell("fused FP8 output", {"rows": 32, "quantized": True}),
        Cell("tiled FP8 output on a large context", {"rows": 513, "quantized": True}),
    )

    def __call__(self, x, qkvz, weight, scale):
        rows = x.shape[0]
        quant = scale is not None
        out = torch.empty((rows, 4096), device=x.device, dtype=FP8 if quant else BF16)
        if rows >= 96:
            tile, warps = (4 if rows < 2048 else 8, 2) if rows >= 512 else (8, 4)
            prefill_stencil.warp_norm(x, qkvz, weight, out, scale, tile, warps)
        else:
            k.gdn_norm[(rows, 32)](x, qkvz, weight, out, qkvz.stride(0), scale, quant, num_warps=4)
        return out

    def reference(self, x, qkvz, weight, scale):
        z = qkvz[:, 8192:].reshape_as(x).float()
        y = x.float()
        y = y * (y.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        y = (y * weight.float() * torch.nn.functional.silu(z)).to(BF16).flatten(1)
        if scale is not None:
            y = (y.float() / scale).clamp(-448, 448).to(FP8)
        return y


delta_norm = DeltaNorm()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Indexed Gated DeltaNet over the runtime's convolution and recurrent slots."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import triton as tr

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels import gdn_prefill, recurrent
from ..kernels.pointwise import gated_norm


@dataclass(frozen=True)
class Context:
    starts: torch.Tensor
    chunks: torch.Tensor
    chunk_starts: torch.Tensor
    sequence_slots: torch.Tensor
    has_initial_state: torch.Tensor
    batch: int
    longest: int
    gdn_output_tile: int


def _reference_recurrence(
    x: torch.Tensor,
    ba: torch.Tensor,
    weight: torch.Tensor,
    log_decay: torch.Tensor,
    bias: torch.Tensor,
    history: torch.Tensor,
    state: torch.Tensor,
    slots: torch.Tensor,
    lengths: list[int],
    initial: list[bool],
) -> torch.Tensor:
    result = []
    offset = 0
    for slot, length, has_state in zip(slots.tolist(), lengths, initial, strict=True):
        conv = (
            history[slot].float().clone() if has_state else torch.zeros_like(history[slot]).float()
        )
        recurrent_state = (
            state[slot].float().clone() if has_state else torch.zeros_like(state[slot]).float()
        )
        for row in range(offset, offset + length):
            window = torch.cat((conv, x[row, :10240].float().unsqueeze(1)), dim=1)
            conv = window[:, 1:]
            products = (window * weight.float()).to(x.dtype).float()
            projected = F.silu(products.sum(1)).to(x.dtype).float()
            q, k, v = projected.split((2048, 2048, 6144))
            q, k = (
                (t / (t.square().sum(dim=-1, keepdim=True) + 1e-6).sqrt()).repeat_interleave(
                    3, dim=0
                )
                for t in (q.reshape(16, 128), k.reshape(16, 128))
            )
            b, a = ba[row].float().split(48)
            beta = b.sigmoid().to(x.dtype).float()
            decay = (-log_decay.float().exp() * F.softplus(a + bias.float())).exp()
            recurrent_state *= decay[:, None, None]
            v = (v.reshape(48, 128) - (recurrent_state * k[:, None, :]).sum(-1)) * beta[:, None]
            recurrent_state += v[:, :, None] * k[:, None, :]
            result.append(((recurrent_state * q[:, None, :]).sum(-1) * 128**-0.5).reshape(-1))
        offset += length
        history[slot].copy_(conv)
        state[slot].copy_(recurrent_state)
    return torch.stack(result).to(x.dtype)


class GatedDeltaDecode(OpWrapper):
    """Advance the convolution and FP32 state of one token per request."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(why="Single request", spec={"batch": 1}),
        Cell(why="Noncontiguous live slots in a mixed step", spec={"batch": 32}),
        Cell(why="Full decode graph", spec={"batch": 128}),
    )

    def __call__(
        self,
        x: torch.Tensor,
        ba: torch.Tensor,
        weight: torch.Tensor,
        log_decay: torch.Tensor,
        bias: torch.Tensor,
        history: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        rows = x.shape[0]
        convolved = x.new_empty((rows, 10240))
        recurrent.decode_conv[(rows, 40)](
            x,
            weight,
            convolved,
            history,
            slots,
            x.stride(0),
            *history.stride(),
            10240,
            256,
            num_warps=4,
        )
        output = x.new_empty((rows, 6144))
        recurrent.recurrent[(4, rows * 48)](
            convolved,
            ba,
            log_decay,
            bias,
            state,
            output,
            slots,
            state.stride(0),
            32,
            num_warps=1,
            num_stages=3,
        )
        return output

    def reference(
        self,
        x: torch.Tensor,
        ba: torch.Tensor,
        weight: torch.Tensor,
        log_decay: torch.Tensor,
        bias: torch.Tensor,
        history: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        rows = x.shape[0]
        return _reference_recurrence(
            x, ba, weight, log_decay, bias, history, state, slots, [1] * rows, [True] * rows
        )


class GatedDeltaContext(OpWrapper):
    """Causal depthwise convolution and 64-token delta-rule solve."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            why="Short oracles with independent requests",
            spec={"lengths": (15, 79), "carried": (False, False)},
        ),
        Cell(
            why="Continuation and fresh request sharing a step",
            spec={"lengths": (1, 141), "carried": (True, False)},
        ),
        Cell(why="Chunked prefill", spec={"lengths": (1024,), "carried": (True,)}),
    )

    def __call__(
        self,
        x: torch.Tensor,
        ba: torch.Tensor,
        weight: torch.Tensor,
        log_decay: torch.Tensor,
        bias: torch.Tensor,
        history: torch.Tensor,
        state: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        rows = x.shape[0]
        convolved = x.new_empty((rows, 10240))
        tile = 16 if context.longest >= 1024 else 4
        recurrent.context_conv[(context.batch, tr.cdiv(context.longest, tile), 40)](
            x,
            weight,
            convolved,
            history,
            context.starts,
            context.sequence_slots,
            context.has_initial_state,
            x.stride(0),
            *history.stride(),
            10240,
            tile,
            256,
            num_warps=4,
        )
        recurrent.context_conv_tail[(context.batch, 40)](
            x,
            history,
            context.starts,
            context.sequence_slots,
            context.has_initial_state,
            x.stride(0),
            *history.stride(),
            10240,
            256,
            num_warps=4,
        )
        g = x.new_empty((rows, 48), dtype=torch.float32)
        # The gate rounds through BF16, but the products in the solve use FP32.
        beta = torch.empty_like(g)
        recurrent.prefill_gates[(rows,)](ba, log_decay, bias, g, beta, 48, num_warps=1)
        return gdn_prefill.gdn(convolved, g, beta, context, state)

    def reference(
        self,
        x: torch.Tensor,
        ba: torch.Tensor,
        weight: torch.Tensor,
        log_decay: torch.Tensor,
        bias: torch.Tensor,
        history: torch.Tensor,
        state: torch.Tensor,
        context: Context,
    ) -> torch.Tensor:
        lengths = context.starts.diff().tolist()
        return _reference_recurrence(
            x,
            ba,
            weight,
            log_decay,
            bias,
            history,
            state,
            context.sequence_slots,
            lengths,
            context.has_initial_state.tolist(),
        )


class GatedRmsNorm(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(why="FP8 for output GEMM", spec={"rows": 1, "pack": True}),
        Cell(why="Continuing batched context", spec={"rows": 142, "pack": False}),
    )

    def __call__(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        scale: torch.Tensor | None = None,
        *,
        decode: bool = False,
    ) -> torch.Tensor:
        rows = x.shape[0]
        output = torch.empty_like(x, dtype=torch.float8_e4m3fn if scale is not None else x.dtype)
        group = 1 if decode else min(4, tr.next_power_of_2(tr.cdiv(rows * 48, 304)))
        pdl = decode and rows <= 8
        gated_norm[(tr.cdiv(rows * 48, group),)](
            x,
            z,
            weight,
            output,
            rows * 48,
            z.stride(0),
            group,
            eps,
            scale,
            scale is not None,
            pdl,
            num_warps=1,
            launch_pdl=pdl,
        )
        return output

    def reference(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        scale: torch.Tensor | None = None,
        *,
        decode: bool = False,
    ) -> torch.Tensor:
        f = x.float().view(-1, 128)
        y = (f * (f.square().mean(-1, keepdim=True) + eps).rsqrt()) * weight.float()
        y = (y.reshape(x.shape) * F.silu(z.float())).to(x.dtype)
        if scale is not None:
            y = (y.float() * scale.reciprocal()).clamp(-448, 448).to(torch.float8_e4m3fn)
        return y


gdn_decode = GatedDeltaDecode()
gdn_context = GatedDeltaContext()
gated_rmsnorm = GatedRmsNorm()

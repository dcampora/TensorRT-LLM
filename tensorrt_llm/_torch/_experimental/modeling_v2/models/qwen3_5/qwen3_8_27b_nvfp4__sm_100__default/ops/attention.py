# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Partial RoPE and gated full attention for the text checkpoint."""

from __future__ import annotations

import torch
import triton as tr

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels.pointwise import qkv_rope, sigmoid_inplace

QUERY_HEADS = 24
KV_HEADS = 4
HEAD_SIZE = 256
QUERY_WIDTH = QUERY_HEADS * HEAD_SIZE
KV_WIDTH = KV_HEADS * HEAD_SIZE


class QkvRope(OpWrapper):
    """Deinterleave the query gates and apply 64-dimensional rotary embeddings."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(why="Decode", spec={"rows": 1, "position": 4095}),
        Cell(why="Prefill", spec={"rows": 128, "position": 0}),
    )

    def __call__(
        self,
        projection: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        rope: torch.Tensor,
        positions: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = projection.shape[0]
        q = projection.new_empty((rows, QUERY_WIDTH))
        k = projection.new_empty((rows, KV_WIDTH))
        v = torch.empty_like(k)
        gate = torch.empty_like(q)
        qkv_rope[(rows, QUERY_HEADS + KV_HEADS)](
            projection,
            q,
            k,
            v,
            gate,
            q_norm,
            k_norm,
            rope,
            positions,
            None,
            projection.stride(0),
            QUERY_HEADS,
            KV_HEADS,
            0,
            0,
            False,
            eps,
            num_warps=4,
        )
        return q, k, v, gate

    def reference(
        self,
        projection: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        rope: torch.Tensor,
        positions: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qg, k, v = projection.split((QUERY_WIDTH * 2, KV_WIDTH, KV_WIDTH), dim=-1)
        q, gate = qg.reshape(-1, QUERY_HEADS, HEAD_SIZE * 2).chunk(2, dim=-1)
        cos, sin = rope[positions].unsqueeze(1).chunk(2, dim=-1)

        def rotate(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            f = x.float()
            y = ((f * (f.square().mean(-1, keepdim=True) + eps).rsqrt()) * (weight.float() + 1)).to(
                projection.dtype
            )
            lo, hi = y[..., :64].float().chunk(2, dim=-1)
            return torch.cat(
                ((lo * cos - hi * sin).to(y.dtype), (hi * cos + lo * sin).to(y.dtype), y[..., 64:]),
                dim=-1,
            ).flatten(1)

        return (
            rotate(q, q_norm),
            rotate(k.reshape(-1, KV_HEADS, HEAD_SIZE), k_norm),
            v,
            gate.flatten(1),
        )


class AttentionGate(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(why="Single generation token", spec={"rows": 1}),
        Cell(why="Full prefill", spec={"rows": 1024}),
    )

    def __call__(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        sigmoid_inplace[(tr.cdiv(x.numel(), 1024),)](x, gate, x.numel(), 1024)
        return x

    def reference(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return x.copy_(x.float() * gate.float().sigmoid())


prepare_qkv = QkvRope()
apply_gate = AttentionGate()

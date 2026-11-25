# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Partial RoPE and gated full attention for the text checkpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import triton as tr

from .....catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from ..kernels.fp8_tma import qkv_parts
from ..kernels.pointwise import qkv_rope, sigmoid_inplace

if TYPE_CHECKING:
    from .paged import Cache

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
        Cell(why="Direct paged store crosses a page boundary", spec={"width": 4}),
        Cell(why="Direct paged store combines FP8 partitions", spec={"width": 2, "parts": True}),
        Cell(why="Padded FP8 partition stride", spec={"width": 2, "parts": True, "pad_rows": 7}),
    )

    def __call__(
        self,
        projection: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        rope: torch.Tensor,
        positions: torch.Tensor,
        eps: float,
        *,
        scales: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache: Cache | None = None,
        table: torch.Tensor | None = None,
        rows_per_seq: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = projection.shape[-2]
        q = projection.new_empty((rows, QUERY_WIDTH), dtype=torch.bfloat16)
        gate = torch.empty_like(q)
        if cache is None:
            k = q.new_empty((rows, KV_WIDTH))
            v = torch.empty_like(k)
            page_size = page_stride = divisor = table_stride = 0
        else:
            k, v = cache.key, cache.value
            page_size = k.shape[2]
            page_stride, divisor = k.stride(0), cache.index_divisor
            assert table is not None
            table_stride = table.stride(0)
        sx, sw = scales if scales is not None else (None, None)
        plane = projection.stride(0) if scales is not None else 0
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
            projection.stride(-2),
            QUERY_HEADS,
            KV_HEADS,
            eps,
            sx,
            sw,
            plane,
            table,
            page_size,
            page_stride,
            divisor,
            table_stride,
            rows_per_seq,
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
        *,
        scales: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache: Cache | None = None,
        table: torch.Tensor | None = None,
        rows_per_seq: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if scales is not None:
            projection = ((projection[0] + projection[1]) * (scales[0] * scales[1])).bfloat16()
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

        key = rotate(k.reshape(-1, KV_HEADS, HEAD_SIZE), k_norm)
        if cache is not None:
            from .paged import append_kv

            assert table is not None
            append_kv.reference(key, v, positions, cache, table, None, rows_per_seq)
            key, v = cache.key, cache.value
        return rotate(q, q_norm), key, v, gate.flatten(1)

    def is_valid(
        self,
        projection: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        rope: torch.Tensor,
        positions: torch.Tensor,
        eps: float,
        *,
        scales: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache: Cache | None = None,
        table: torch.Tensor | None = None,
        rows_per_seq: int = 1,
    ) -> None:
        assert projection.shape[-1] == 2 * (QUERY_WIDTH + KV_WIDTH)
        assert projection.stride(-1) == 1 and projection.shape[-2] == positions.numel()
        if scales is None:
            assert projection.ndim == 2 and projection.dtype == torch.bfloat16
        else:
            assert projection.shape[0] == 2 and projection.ndim == 3
            assert projection.dtype == torch.float32
        if cache is not None:
            assert table is not None and table.stride(-1) == 1
            assert rows_per_seq > 0 and positions.numel() % rows_per_seq == 0


class Fp8QkvPartitions(OpWrapper):
    """Project a decode activation with two FP32 K partitions."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(why="One target verify request", spec={"rows": 5}),
        Cell(why="Batched verifier", spec={"rows": 128}),
    )

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return qkv_parts(x, weight)

    def reference(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        weight = weight.squeeze(0)
        with true_fp32_matmul():
            return torch.stack(
                [
                    x[:, start:end].float() @ weight[start:end].float()
                    for start, end in ((0, 2560), (2560, 5120))
                ]
            )

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor) -> None:
        assert x.dtype == weight.dtype == torch.float8_e4m3fn
        assert x.ndim == 2 and x.shape[1] == 5120 and x.stride(-1) == 1
        assert weight.shape == (1, 5120, 14336) and weight.stride(-2) == 1


fp8_qkv_partitions = Fp8QkvPartitions()


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

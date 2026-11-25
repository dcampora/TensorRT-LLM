# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 Q/K normalization, partial RoPE, and gated attention."""

import torch

from .....catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from ..kernels import attention


class PrepareQkv(OpWrapper):
    """Unpack interleaved queries and gates, apply Q/K RMSNorm and rotate 64 features."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("Single decode row at a nonzero position", dict(rows=1, tiled=False)),
        Cell("Long prefill with the grouped-head kernel", dict(rows=256, tiled=True)),
    )

    def __call__(
        self,
        projection: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        rope: torch.Tensor,
        pos: torch.Tensor,
        tiled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return attention.qkv(projection, wq, wk, rope, pos, tiled_prefill=tiled)

    def reference(
        self,
        projection: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        rope: torch.Tensor,
        pos: torch.Tensor,
        tiled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qg, k, v = projection.split((12288, 1024, 1024), dim=-1)
        q = qg.view(-1, 24, 512)[..., :256]
        k = k.view(-1, 4, 256)
        cos, sin = rope[pos.long()].unsqueeze(1).chunk(2, dim=-1)
        results = []
        for x, weight in ((q, wq), (k, wk)):
            x = x.float()
            x = (
                (x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * (1 + weight.float()))
                .to(projection.dtype)
                .float()
            )
            first, second, rest = x.split((32, 32, 192), dim=-1)
            rotated = torch.cat(
                (first * cos - second * sin, second * cos + first * sin, rest), dim=-1
            )
            results.append(rotated.to(projection.dtype))
        return *results, v.view(-1, 4, 256)


class AttentionGate(OpWrapper):
    """Apply the query's sigmoid output gate."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = tuple(Cell(f"{rows} attention rows", dict(rows=rows)) for rows in (1, 96, 128))

    def __call__(self, value: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        return attention.prefill_gate(value, projection)

    def reference(self, value: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        gate = projection[:, :12288].reshape(-1, 24, 512)[..., 256:].float()
        return (value.view_as(gate).float() * gate.sigmoid()).flatten(1).to(projection.dtype)


class ShortCausalAttention(OpWrapper):
    """Compute causal GQA, with FP32 probabilities, on fresh short contexts."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("Two packed short contexts", dict(lengths=(12, 32))),
        Cell("Two query tiles and a partial final key tile", dict(lengths=(103,))),
    )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        projection: torch.Tensor,
        offsets: torch.Tensor,
        tiles: torch.Tensor,
    ) -> torch.Tensor:
        return attention.dense_causal(
            q, k, v, projection, offsets, tiles, fp32_p=True, value_tile=64
        )

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        projection: torch.Tensor,
        offsets: torch.Tensor,
        tiles: torch.Tensor,
    ) -> torch.Tensor:
        edges = offsets.tolist()
        out = []
        with true_fp32_matmul():
            for start, end in zip(edges, edges[1:]):
                query = q[start:end].transpose(0, 1).float()
                keys = k[start:end].transpose(0, 1).repeat_interleave(6, dim=0).float()
                values = v[start:end].transpose(0, 1).repeat_interleave(6, dim=0).float()
                mask = torch.ones(
                    (end - start, end - start), device=q.device, dtype=torch.bool
                ).triu(1)
                probs = (
                    (query @ keys.transpose(1, 2) * 0.0625)
                    .masked_fill(mask, -torch.inf)
                    .softmax(-1)
                )
                out.append((probs @ values).transpose(0, 1).bfloat16())
        gate = projection[:, :12288].view(-1, 24, 512)[..., 256:].float().sigmoid()
        return (torch.cat(out).float() * gate).flatten(1).to(projection.dtype)


prepare_qkv = PrepareQkv()
attention_gate = AttentionGate()
short_causal_attention = ShortCausalAttention()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused BF16 activation and block quantization operations."""

import torch
import torch.nn.functional as F

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels import activation_quant, norm_quant
from .projection import reference_block_quantize


class SwiGluQuantize(OpWrapper):
    """Round SwiGLU to BF16 and quantize its 128-channel activation groups."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = tuple(
        Cell(f"SwiGLU with {rows} packed rows", dict(rows=rows)) for rows in (1, 5, 32, 128, 1024)
    )

    def __call__(self, gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return activation_quant.swiglu_quantize(gate_up)

    def reference(self, gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate, up = gate_up.float().chunk(2, dim=-1)
        return reference_block_quantize((F.silu(gate) * up).bfloat16())


class GdnNormQuantize(OpWrapper):
    """Fuse the GDN headwise RMSNorm, SiLU gate and FP8 conversion."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = tuple(Cell(f"Normalize {rows} GDN rows", dict(rows=rows)) for rows in (1, 7, 32, 128))

    def __call__(
        self,
        x: torch.Tensor,
        projection: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return activation_quant.gdn_norm_quantize(x, projection, weight)

    def reference(
        self,
        x: torch.Tensor,
        projection: torch.Tensor,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = F.silu(projection[:, 10240:].float().view_as(x))
        x = x.float()
        x = x * (x.square().mean(dim=-1, keepdim=True) + 1e-6).rsqrt() * weight.float()
        return reference_block_quantize((x * gate).flatten(1).bfloat16())


class AttentionGateQuantize(OpWrapper):
    """Read the interleaved query gate and quantize the gated attention output."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = tuple(
        Cell(f"Gate and quantize {rows} attention rows", dict(rows=rows))
        for rows in (1, 7, 128, 1024)
    )

    def __call__(
        self,
        attention: torch.Tensor,
        projection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return activation_quant.prefill_attention_quantize(attention, projection, False)

    def reference(
        self,
        attention: torch.Tensor,
        projection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = projection[:, :12288].view(-1, 24, 512)[..., 256:].float()
        value = attention.view_as(gate).float() * gate.sigmoid()
        return reference_block_quantize(value.flatten(1).bfloat16())


class ResidualNormQuantize(OpWrapper):
    """Write the BF16 residual and dynamically quantize the normalized input."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("Retain BF16 for GDN gate GEMM", dict(rows=5, normalized=True)),
        Cell("Long FFN prefill with FP8 consumers", dict(rows=1024, normalized=False)),
    )

    def __call__(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        store_normalized: bool,
        programmatic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return norm_quant.residual_norm_quantize(
            x, residual, weight, store_normalized, programmatic
        )

    def reference(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        store_normalized: bool,
        programmatic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        added = x.float() + residual.float()
        inverse = (added.square().mean(dim=-1, keepdim=True) + 1e-6).rsqrt()
        normalized = (added * inverse * (1 + weight.float())).bfloat16()
        residual.copy_(added)
        if store_normalized:
            x.copy_(normalized)
        return reference_block_quantize(normalized)


class DraftInputNorm(OpWrapper):
    """Gather an embedding and normalize the two inputs of the MTP FC."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("Int32 decode IDs", dict(rows=8, wide=False)),
        Cell("Int64 prefill IDs", dict(rows=128, wide=True)),
    )

    def __call__(
        self,
        ids: torch.Tensor,
        table: torch.Tensor,
        hidden: torch.Tensor,
        token_weight: torch.Tensor,
        hidden_weight: torch.Tensor,
    ) -> torch.Tensor:
        return norm_quant.dual_input_norm(ids, table, hidden, token_weight, hidden_weight)

    def reference(
        self,
        ids: torch.Tensor,
        table: torch.Tensor,
        hidden: torch.Tensor,
        token_weight: torch.Tensor,
        hidden_weight: torch.Tensor,
    ) -> torch.Tensor:
        def gemma_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            value = x.float()
            rms = (value.square().mean(dim=-1, keepdim=True) + 1e-6).rsqrt()
            return (value * rms * (1 + weight.float())).bfloat16()

        return torch.cat(
            (
                gemma_norm(F.embedding(ids, table), token_weight),
                gemma_norm(hidden, hidden_weight),
            ),
            dim=-1,
        )


swiglu_quantize = SwiGluQuantize()
gdn_norm_quantize = GdnNormQuantize()
attention_gate_quantize = AttentionGateQuantize()
residual_norm_quantize = ResidualNormQuantize()
draft_input_norm = DraftInputNorm()

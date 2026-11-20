# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated attention and MoE epilogues."""

import torch
from torch.nn import functional as F

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels import gates


class SigmoidGate(OpWrapper):
    """Apply a per-element or per-row sigmoid gate to BF16 rows."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            why="batch-1 attention output with row-strided gates",
            spec=dict(rows=1, width=4096, shared=False),
        ),
        Cell(
            why="prefill attention crosses row and column tiles",
            spec=dict(rows=1024, width=4096, shared=False),
        ),
        Cell(
            why="a shared-expert scalar gate and routed residual per token",
            spec=dict(rows=128, width=2048, shared=True),
        ),
    )

    def __call__(self, x, gate, addend=None):
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        gates.sigmoid_gate(x, gate, addend, out)
        return out

    def reference(self, x, gate, addend=None):
        out = x.float() * gate.float().sigmoid()
        if addend is not None:
            out = out + addend.float()
        return out.to(x.dtype)


class GatedRmsNorm(OpWrapper):
    """Normalize [tokens, heads, width] and multiply by a row-strided SiLU gate."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(why="batch-1 decode", spec=dict(rows=1, heads=32, width=128)),
        Cell(why="multi-request decode", spec=dict(rows=32, heads=32, width=128)),
        Cell(why="long packed prefill", spec=dict(rows=4096, heads=32, width=128)),
    )

    def __call__(self, x, z, weight, eps):
        out = torch.empty_like(x, memory_format=torch.contiguous_format)
        gates.gated_rms_norm(x, z, weight, eps, out)
        return out

    def reference(self, x, z, weight, eps):
        normalized = F.rms_norm(x.float(), (x.shape[-1],), weight.float(), eps)
        return (normalized * F.silu(z.float())).to(x.dtype)


sigmoid_gate = SigmoidGate()
gated_rms_norm = GatedRmsNorm()

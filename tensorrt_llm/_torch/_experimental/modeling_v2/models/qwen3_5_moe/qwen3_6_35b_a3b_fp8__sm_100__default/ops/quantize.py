# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 input preparation for the Qwen3.6 projection and expert paths."""

import torch

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels.norm_quantize import add_rmsnorm_quant as _add_rmsnorm_quant
from ..kernels.quantize import quantize as _quantize


def _reference_quantize(x):
    rows, width = x.shape
    blocks = x.reshape(rows, width // 128, 128).float()
    minimum = x.new_tensor(1e-10).float()
    maximum = blocks.abs().amax(dim=-1).clamp_min(minimum)
    inverse_scale = (448.0 / maximum.double()).float()
    data = (blocks * inverse_scale.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return data.reshape_as(x), inverse_scale.double().reciprocal().float().T.contiguous()


class BlockQuantize(OpWrapper):
    """Quantize BF16 rows to E4M3 with FP32 scales stored [block, token]."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(why="one projection row with an all-zero block", spec=dict(rows=1, width=2048)),
        Cell(why="two target-verify rows", spec=dict(rows=2, width=4096)),
        Cell(why="one shared-expert intermediate", spec=dict(rows=1, width=512)),
        Cell(why="many packed context rows", spec=dict(rows=1024, width=2048)),
    )

    def __call__(self, x):
        return _quantize(x)

    def is_valid(self, x):
        if x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[1] % 128:
            raise ValueError("expected BF16 rows with width divisible by 128")

    def reference(self, x):
        return _reference_quantize(x)


class AddRmsNormQuant(OpWrapper):
    """Update a BF16 residual and normalized activation, returning its FP8 data and scales."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(why="one sequence with two verification rows", spec=dict(rows=2)),
        Cell(why="graph-sized generation batch", spec=dict(rows=64)),
        Cell(why="packed prefill with arbitrary request length", spec=dict(rows=1025)),
    )

    def __call__(self, hidden, residual, weight, eps):
        return _add_rmsnorm_quant(hidden, residual, weight, eps)

    def is_valid(self, hidden, residual, weight, eps):
        if (
            hidden.ndim != 2
            or hidden.shape[1] != 2048
            or residual.shape != hidden.shape
            or not hidden.is_contiguous()
            or not residual.is_contiguous()
        ):
            raise ValueError("expected contiguous hidden and residual rows of width 2048")
        if hidden.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
            raise ValueError("expected BF16 hidden and residual inputs")

    def reference(self, hidden, residual, weight, eps):
        summed = hidden.float() + residual.float()
        residual.copy_(summed)
        inv_rms = (summed.square().mean(dim=-1, keepdim=True) + eps).rsqrt()
        hidden.copy_(summed * inv_rms * (1 + weight.float()))
        return _reference_quantize(hidden)


block_quantize = BlockQuantize()
add_rmsnorm_quant = AddRmsNormQuant()

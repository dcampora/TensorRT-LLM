# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the BF16 and block-scaled inputs to a decoder sublayer."""

import torch
import triton
import triton.language as tl

from .quantize import _to_fp8


@triton.jit
def _add_rmsnorm_quant(
    Hidden, Residual, Weight, FP8, Scale, rows: tl.constexpr, size: tl.constexpr, eps: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, size)
    offset = row * size + col
    s = tl.load(Hidden + offset).to(tl.float32) + tl.load(Residual + offset).to(tl.float32)
    tl.store(Residual + offset, s)
    inv_rms = tl.rsqrt(tl.sum(s * s, 0) / size + eps)
    weight = tl.load(Weight + col).to(tl.float32) + 1.0
    normalized = (s * inv_rms * weight).to(tl.bfloat16)
    tl.store(Hidden + offset, normalized)

    groups: tl.constexpr = size // 128
    matrix = tl.reshape(normalized.to(tl.float32), (groups, 128))
    values, scales = _to_fp8(matrix)
    tl.store(FP8 + offset, tl.reshape(values, (size,)))
    tl.store(Scale + tl.arange(0, groups) * rows + row, scales)


def add_rmsnorm_quant(hidden, residual, weight, eps):
    rows, size = hidden.shape
    data = torch.empty((rows, size), dtype=torch.float8_e4m3fn, device=hidden.device)
    scales = torch.empty((size // 128, rows), dtype=torch.float32, device=hidden.device)
    _add_rmsnorm_quant[(rows,)](
        hidden, residual, weight, data, scales, rows, size, eps, num_warps=4
    )
    return data, scales

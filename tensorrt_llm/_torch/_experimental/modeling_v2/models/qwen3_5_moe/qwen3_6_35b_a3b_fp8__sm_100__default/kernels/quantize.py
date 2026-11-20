# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token-wise E4M3 quantization with native SM100 group-scale layout."""

import torch
import triton
import triton.language as tl


@triton.jit
def _to_fp8(x):
    # The native quantizer rounds its amax floor to BF16.
    amax_min: tl.constexpr = 1.000444171950221e-10
    amax = tl.maximum(tl.max(tl.abs(x), 1), amax_min)
    qscale = tl.div_rn(448.0, amax)
    quantized = tl.clamp(x * qscale[:, None], -448.0, 448.0)
    return quantized, tl.div_rn(1.0, qscale)


@triton.jit
def _quantize_kernel(
    X,
    Out,
    Scale,
    rows: tl.constexpr,
    width: tl.constexpr,
    stride: tl.constexpr,
    groups_per_cta: tl.constexpr,
):
    blocks: tl.constexpr = width // 128
    group = tl.program_id(0) * groups_per_cta + tl.arange(0, groups_per_cta)
    row = group // blocks
    block = group % blocks
    col = block[:, None] * 128 + tl.arange(0, 128)[None, :]
    mask = row < rows
    x = tl.load(X + row[:, None] * stride + col, mask[:, None], 0).to(tl.float32)
    quantized, scale = _to_fp8(x)
    tl.store(Out + row[:, None] * width + col, quantized, mask[:, None])
    tl.store(Scale + block * rows + row, scale, mask)


def quantize(x):
    """Return FP8 rows and contiguous [group, token] FP32 scales."""
    rows, width = x.shape
    blocks = width // 128
    values = torch.empty((rows, width), dtype=torch.float8_e4m3fn, device=x.device)
    scales = torch.empty((blocks, rows), dtype=torch.float32, device=x.device)
    groups = 1 if rows * blocks < 128 else 4
    _quantize_kernel[(triton.cdiv(rows * blocks, groups),)](
        x,
        values,
        scales,
        rows,
        width,
        x.stride(0),
        groups,
        num_warps=groups,
    )
    return values, scales

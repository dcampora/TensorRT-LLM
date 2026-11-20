# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small-row E4M3 GEMM with the checkpoint's floating-point block scales."""

import torch
import triton
import triton.language as tl


@triton.jit
def _block_mm(
    X,
    XScale,
    Weight,
    WeightScale,
    Partial,
    m: tl.constexpr,
    n: tl.constexpr,
    k: tl.constexpr,
    cols: tl.constexpr,
    groups: tl.constexpr,
    scale_stride: tl.constexpr,
):
    col = tl.program_id(0) * cols + tl.arange(0, cols)
    split = tl.program_id(1)
    row = tl.arange(0, 16)
    block = tl.arange(0, 128)
    accum = tl.full((16, cols), 0, tl.float32)
    for i in tl.range(groups):
        group = split * groups + i
        feat = group * 128 + block
        x = tl.load(X + row[:, None] * k + feat[None, :], row[:, None] < m, 0).to(
            tl.float8e4nv, bitcast=True
        )
        w = tl.load(Weight + col[None, :] * k + feat[:, None])
        result = tl.dot(x, w)
        xs = tl.load(XScale + group * scale_stride + row, row < m, 0)
        ws = tl.load(WeightScale + (col // 128) * (k // 128) + group)
        accum = tl.fma(result * ws[None, :], xs[:, None], accum)

    tl.store(Partial + (split * m + row[:, None]) * n + col[None, :], accum, row[:, None] < m)


@triton.jit
def _sum_partials(Partial, Out, size: tl.constexpr, splits: tl.constexpr, block: tl.constexpr):
    flat = tl.program_id(0) * block + tl.arange(0, block)
    split = tl.arange(0, splits)
    partial = tl.load(Partial + split[:, None] * size + flat[None, :], flat[None, :] < size, 0)
    tl.store(Out + flat, tl.sum(partial, 0), flat < size)


def low_m_gemm(x, x_scale, weight, weight_scale, cols=64, groups=2):
    rows, k = x.shape
    n = weight.shape[0]
    splits = k // (groups * 128)
    assert splits * groups * 128 == k and n % cols == 0
    out = torch.empty((rows, n), device=x.device, dtype=torch.bfloat16)
    if splits == 1:
        partial = out
    else:
        partial = torch.empty((splits, rows, n), device=x.device, dtype=torch.float32)
    _block_mm[(n // cols, splits)](
        x.view(torch.uint8),
        x_scale,
        weight,
        weight_scale,
        partial,
        rows,
        n,
        k,
        cols,
        groups,
        x_scale.stride(0),
        num_warps=4,
        num_stages=2,
    )
    if splits > 1:
        block = 256
        _sum_partials[(triton.cdiv(rows * n, block),)](
            partial,
            out,
            rows * n,
            splits,
            block,
            num_warps=4,
        )
    return out

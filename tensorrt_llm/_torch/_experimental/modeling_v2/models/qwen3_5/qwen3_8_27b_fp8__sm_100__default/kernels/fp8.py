# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from aegis-engines qwen3.8-27b-fp8-batched, kernels.py (114cf57c911e)."""

import torch
import triton as tr
import triton.language as tl

MATCH_ROUNDING = True


@tr.jit
def _requantize_pow2(W, S, Y, T, K: tl.constexpr):
    n_group = tl.program_id(0)
    k_group = tl.program_id(1)
    n = n_group * 128 + tl.arange(0, 128)
    k = k_group * 128 + tl.arange(0, 128)
    offset = n[:, None] * K + k[None, :]
    block = n_group * (K // 128) + k_group
    s = tl.load(S + block).to(tl.float32)
    w = tl.load(W + offset).to(tl.float32)
    # The Blackwell backend converts the original dequantized block to
    # BF16 at this boundary. Keep that intermediate in registers.
    value = (w * s).to(tl.bfloat16).to(tl.float32)
    peak = tl.maximum(tl.max(tl.max(tl.abs(value), 1), 0), 1.0e-4)
    raw = (peak * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    exponent = (raw >> 23) + ((raw & 0x7FFFFF) != 0).to(tl.int32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    tl.store(T + block, scale)
    tl.store(Y + offset, tl.clamp(value / scale, -448.0, 448.0))


def requantize_pow2(weight, scales):
    n, k = weight.shape
    fp8 = torch.empty_like(weight)
    ue8m0 = torch.empty_like(scales)
    _requantize_pow2[(n // 128, k // 128)](weight, scales, fp8, ue8m0, k, num_warps=8)
    return fp8, ue8m0


@tr.jit(do_not_specialize=["M", "PADDED"])
def _quantize_aligned(
    X, Y, S, M, PADDED, K: tl.constexpr, POW2: tl.constexpr, MATCH: tl.constexpr, ROWS: tl.constexpr
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    g = tl.program_id(1)
    col = g * 128 + tl.arange(0, 128)
    offset = row[:, None] * K + col[None, :]
    x = tl.load(X + offset, row[:, None] < M, 0.0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), 1), 1.0e-10)
    s = amax * (1.0 / 448.0)
    if POW2:
        bits = s.to(tl.int32, bitcast=True)
        exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        s = (exponent << 23).to(tl.float32, bitcast=True)
    if POW2:
        # Dividing BF16-derived activations by an exact power of two is a shift.
        inverse = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
        y = x * inverse[:, None]
    elif MATCH:
        y = x * tl.div_rn(448.0, amax[:, None])
    else:
        y = tl.div_rn(x, s[:, None])
    tl.store(Y + offset, tl.clamp(y, -448.0, 448.0), row[:, None] < PADDED)
    tl.store(S + g * PADDED + row, s, row < PADDED)


def quantize_aligned(x, pow2):
    """Quantize and zero-fill to the MN-major CUTLASS alignment in one launch."""
    m, k = x.shape
    padded = tr.cdiv(m, 4) * 4
    a = torch.empty((padded, k), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((k // 128, padded), device=x.device, dtype=torch.float32)
    tile_rows = 4
    warps = 4
    if pow2:
        if m >= 256:
            tile_rows = 32
        elif m >= 128:
            tile_rows, warps = 8, 1
    _quantize_aligned[(tr.cdiv(padded, tile_rows), k // 128)](
        x,
        a,
        s,
        m,
        padded,
        k,
        pow2,
        MATCH_ROUNDING,
        tile_rows,
        num_warps=warps,
    )
    return a, s


@tr.jit
def _fp8_group_dots(
    A, W, P, K: tl.constexpr, N: tl.constexpr, ROWS: tl.constexpr, GROUPS: tl.constexpr
):
    n = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    m = tl.arange(0, 16)
    k = tl.arange(0, 128)
    g = tl.program_id(1) * GROUPS
    for u in tl.static_range(GROUPS):
        group = g + u
        a = tl.load(
            A + group * 128 + k[:, None] + m[None, :] * K,
            (group < K // 128) & (m[None, :] == 0),
            0.0,
        )
        w = tl.load(W + n[:, None] * K + group * 128 + k[None, :], group < K // 128, 0.0)
        raw = tl.dot(w, a)
        dot = tl.sum(tl.where(m[None, :] == 0, raw, 0.0), 1)
        tl.store(P + group * N + n, dot, group < K // 128)


@tr.jit
def _ordered_group_sum(
    P, SA, SW, Y, K: tl.constexpr, N: tl.constexpr, D: tl.constexpr, SCALE_STRIDE: tl.constexpr
):
    n = tl.program_id(0) * D + tl.arange(0, D)
    acc = tl.full((D,), 0.0, tl.float32)
    for group in tl.range(0, K // 128, loop_unroll_factor=4):
        dot = tl.load(P + group * N + n)
        sa = tl.load(SA + group * SCALE_STRIDE)
        sw = tl.load(SW + n // 128 * (K // 128) + group).to(tl.float32)
        acc = tl.fma(dot, sa * sw, acc)
    tl.store(Y + n, acc)


def ordered_fp8_product(a, sa, weights, scales, scale_stride):
    """Parallelize K-group FP8 products and keep their scale-add order."""
    n, k = weights.shape
    partial = torch.empty((k // 128, n), device=a.device, dtype=torch.float32)
    out = torch.empty((1, n), device=a.device, dtype=torch.bfloat16)
    groups = 4 if k >= 8192 else 2
    _fp8_group_dots[(n // 128, tr.cdiv(k, 128 * groups))](
        a,
        weights,
        partial,
        k,
        n,
        128,
        groups,
        num_warps=4,
    )
    _ordered_group_sum[(n // 128,)](
        partial,
        sa,
        scales,
        out,
        k,
        n,
        128,
        scale_stride,
        num_warps=4,
    )
    return out

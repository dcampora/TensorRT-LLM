# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 draft kernels from aegis-engines spec_kernels.py and batch_spec_kernels.py."""

import triton as tr
import triton.language as tl

from .quantized import _silu


@tr.jit
def bf16_experts(
    X, W, Ids, P, Y, GATE: tl.constexpr, BN: tl.constexpr, N: tl.constexpr, K: tl.constexpr
):
    slot = tl.program_id(1)
    e = tl.load(Ids + slot).to(tl.int64)
    xi = slot // 8 if GATE else slot
    j = tl.arange(0, K)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    w = W + e * N * K + n[:, None] * K + j[None, :]
    x = tl.load(X + xi * K + j).to(tl.float32)
    a = tl.sum(tl.load(w).to(tl.float32) * x[None, :], 1).to(tl.bfloat16).to(tl.float32)
    if GATE:
        b = (
            tl.sum(tl.load(w + N // 2 * K).to(tl.float32) * x[None, :], 1)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        a = _silu(a) * b
    else:
        a *= tl.load(P + slot).to(tl.bfloat16).to(tl.float32)
    width: tl.constexpr = N // 2 if GATE else N
    tl.store(Y + slot * width + n, a)


@tr.jit
def draft_prepare(Ids, Embed, Hidden, WE, WH, Out, D: tl.constexpr):
    row, which = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, D)
    if which == 0:
        tok = tl.load(Ids + row)
        x = tl.load(Embed + tok * D + d).to(tl.float32)
        w = tl.load(WE + d).to(tl.float32)
    else:
        x = tl.load(Hidden + row * D + d).to(tl.float32)
        w = tl.load(WH + d).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + 1e-6)
    tl.store(Out + (row * 2 + which) * D + d, (x * inv) * (1.0 + w))


@tr.jit
def draft_experts(
    X,
    W,
    Experts,
    Limit,
    Perm,
    Active,
    P,
    Y,
    GATE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    tile, block = tl.program_id(0), tl.program_id(1)
    if tile < tl.load(Active):
        expert = tl.load(Experts + tile).to(tl.int64)
        m = tile * BM + tl.arange(0, BM)
        valid = m < tl.load(Limit + tile)
        slots = tl.load(Perm + m, valid, 0)
        row = slots // 8 if GATE else m
        n = block * BN + tl.arange(0, BN)
        j = tl.arange(0, BK)
        acc = tl.full((BM, BN), 0.0, tl.float32)
        if GATE:
            other = tl.full((BM, BN), 0.0, tl.float32)
        for f in range(K // BK):
            k = f * BK + j
            x = tl.load(X + row[:, None] * K + k[None, :], valid[:, None], 0.0)
            offset = (expert * N + n[:, None]) * K + k[None, :]
            w = tl.load(W + offset)
            acc = tl.dot(x, tl.trans(w), acc)
            if GATE:
                up = tl.load(W + offset + (N // 2) * K)
                other = tl.dot(x, tl.trans(up), other)
        a = acc.to(tl.bfloat16).to(tl.float32)
        if GATE:
            b = other.to(tl.bfloat16).to(tl.float32)
            y = _silu(a) * b
            tl.store(Y + m[:, None] * (N // 2) + n[None, :], y, valid[:, None])
        else:
            p = tl.load(P + slots, valid, 0.0).to(tl.bfloat16).to(tl.float32)
            tl.store(Y + slots[:, None] * N + n[None, :], a * p[:, None], valid[:, None])

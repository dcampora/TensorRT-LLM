# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from aegis-engines kernels.py (a6c09c6502)."""

import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def fp8_cast(X, Scale, Y, T, INV: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    s = tl.load(Scale)
    if not INV:
        s = 1.0 / s
    x = tl.load(X + i, i < T, 0).to(tl.float32)
    tl.store(Y + i, tl.minimum(tl.maximum(x * s, -448.0), 448.0), i < T)


@tr.jit
def _store_normalized(Y, Q, Scale, offset, x, inv, norm, QUANT: tl.constexpr):
    result = ((x * inv) * (1.0 + norm)).to(tl.bfloat16)
    tl.store(Y + offset, result)
    if QUANT:
        value = result.to(tl.float32) * (1.0 / tl.load(Scale))
        tl.store(Q + offset, tl.minimum(tl.maximum(value, -448.0), 448.0))


@tr.jit
def norm(
    X,
    Residual,
    W,
    Y,
    NEW: tl.constexpr,
    ADD: tl.constexpr,
    D: tl.constexpr,
    Q=None,
    Scale=None,
    QUANT: tl.constexpr = False,
):
    row = tl.program_id(0)
    d = tl.arange(0, D)
    x = tl.load(X + row * D + d).to(tl.float32)
    if ADD:
        x += tl.load(Residual + row * D + d).to(tl.float32)
    if NEW:
        tl.store(Residual + row * D + d, x)
    w = tl.load(W + d).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, 0) / D + 1e-6)
    _store_normalized(Y, Q, Scale, row * D + d, x, r, w, QUANT)


@tr.jit
def _e2m1(nibble):
    bits = ((nibble.to(tl.uint16) & 7) << 9) | ((nibble.to(tl.uint16) & 8) << 12)
    return bits.to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32) * 16384.0


@tr.jit
def _w4_tile(W, S, n, k, K: tl.constexpr):
    w = tl.load(W + n[:, None] * (K // 2) + k[None, :] // 2)
    s = tl.load(S + n[:, None] * (K // 16) + k[None, :] // 16).to(tl.float32)
    return _e2m1(w >> ((k[None, :] % 2) * 4)) * s


@tr.jit
def _silu(g):
    return g / (1.0 + tl.exp(-g))


@tr.jit
def w4_gemv(
    X,
    W,
    S,
    Alpha,
    Y,
    Ids,
    Prob,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    ROUTED: tl.constexpr,
    MODE: tl.constexpr,
    INTERLEAVED: tl.constexpr = False,
):
    ni = tl.program_id(0)
    slot = tl.program_id(1)
    source = slot
    expert = 0
    if ROUTED:
        expert = tl.load(Ids + slot)
        if MODE == 1:
            source = slot // 8
    n = ni * BN + tl.arange(0, BN)
    k = tl.arange(0, K // 2)
    a0 = tl.load(X + source * K + 2 * k).to(tl.float32)
    a1 = tl.load(X + source * K + 2 * k + 1).to(tl.float32)
    wbase = W + expert * (N * K // 2)
    sbase = S + expert * (N * K // 16)
    gate_n = n // 64 * 128 + 64 + n % 64 if INTERLEAVED else n
    w0 = tl.load(wbase + gate_n[:, None] * (K // 2) + k[None, :])
    s0 = tl.load(sbase + gate_n[:, None] * (K // 16) + k[None, :] // 8).to(tl.float32)
    wlo, whi = tl.inline_asm_elementwise(
        " { .reg .b8 byte; .reg .b32 tmp; cvt.u8.u32 byte, $2; "
        "cvt.rn.f16x2.e2m1x2 tmp, byte; mov.b32 {$0, $1}, tmp; } ",
        "=h,=h,r",
        [w0.to(tl.uint32)],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )
    accum = tl.sum((wlo.to(tl.float32) * a0[None, :] + whi.to(tl.float32) * a1[None, :]) * s0, 1)
    alpha = tl.load(Alpha + expert).to(tl.float32)
    if MODE == 2 and ROUTED:
        p = tl.load(Prob + slot).to(tl.bfloat16).to(tl.float32)
        alpha = (alpha * p).to(tl.bfloat16).to(tl.float32)
    out = (accum.to(tl.bfloat16).to(tl.float32) * alpha).to(tl.bfloat16)
    if MODE == 1:
        up_n = gate_n - 64 if INTERLEAVED else n + N // 2
        w1 = tl.load(wbase + up_n[:, None] * (K // 2) + k[None, :])
        s1 = tl.load(sbase + up_n[:, None] * (K // 16) + k[None, :] // 8).to(tl.float32)
        wlo, whi = tl.inline_asm_elementwise(
            " { .reg .b8 byte; .reg .b32 tmp; cvt.u8.u32 byte, $2; "
            "cvt.rn.f16x2.e2m1x2 tmp, byte; mov.b32 {$0, $1}, tmp; } ",
            "=h,=h,r",
            [w1.to(tl.uint32)],
            dtype=(tl.float16, tl.float16),
            is_pure=True,
            pack=1,
        )
        acc1 = tl.sum((wlo.to(tl.float32) * a0[None, :] + whi.to(tl.float32) * a1[None, :]) * s1, 1)
        up = (acc1.to(tl.bfloat16).to(tl.float32) * alpha).to(tl.bfloat16)
        out = _silu(out.to(tl.float32)) * up.to(tl.float32)
    width: tl.constexpr = N // 2 if MODE == 1 else N
    tl.store(Y + slot * width + n, out)


@tr.jit
def w4_matmul(
    X,
    W,
    S,
    Alpha,
    Y,
    Sorted,
    RowStarts,
    TileStarts,
    P,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    MODE: tl.constexpr,
    ROUTED: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    INTERLEAVED: tl.constexpr = False,
):
    mt, nt = tl.program_id(0), tl.program_id(1)
    active = True
    expert = 0
    if ROUTED:
        active = mt < tl.load(TileStarts + 256)
    if active:
        i = tl.arange(0, BM)
        if ROUTED:
            lo = 0
            for step in tl.static_range(7, -1, -1):
                mid = lo + (1 << step)
                lo = tl.where(tl.load(TileStarts + mid) <= mt, mid, lo)
            expert = lo
            begin = tl.load(RowStarts + lo)
            end = tl.load(RowStarts + lo + 1)
            r = begin + (mt - tl.load(TileStarts + lo)) * BM + i
            valid = r < end
            slot = tl.load(Sorted + r, valid, 0).to(tl.int32)
            if MODE == 1:
                row = slot // 8
            else:
                row = slot
        else:
            row = mt * BM + i
            slot = row
            valid = row < M
        n = nt * BN + tl.arange(0, BN)
        if INTERLEAVED:
            n_weight = (n % (N // 2)) // 64 * 128 + n % 64 + tl.where(n < N // 2, 64, 0)
        else:
            n_weight = n
        k = tl.arange(0, BK)
        wbase = W + expert * (N * K // 2)
        sbase = S + expert * (N * K // 16)
        acc = tl.full((BM, BN), 0.0, tl.float32)
        for kb in range(K // BK):
            kk = kb * BK + k
            a = tl.load(X + row[:, None] * K + kk[None, :], valid[:, None], 0)
            w = _w4_tile(wbase, sbase, n_weight, kk, K).to(tl.bfloat16)
            acc = tl.dot(a, tl.trans(w), acc)
        alpha = tl.load(Alpha + expert).to(tl.float32)
        if MODE == 2 and ROUTED:
            p = tl.load(P + slot, valid, 0).to(tl.bfloat16).to(tl.float32)
            alpha = (p * alpha).to(tl.bfloat16).to(tl.float32)[:, None]
        val = acc.to(tl.bfloat16).to(tl.float32) * alpha
        tl.store(Y + slot[:, None] * N + n[None, :], val, valid[:, None])


@tr.jit
def swiglu(X, Y, SIZE, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    n = i // D * (2 * D) + i % D
    g = tl.load(X + n, i < SIZE, 0).to(tl.float32)
    u = tl.load(X + n + D, i < SIZE, 0).to(tl.float32)
    tl.store(Y + i, _silu(g) * u, i < SIZE)


@tr.jit
def combine(Experts, Shared, SG, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    v = tl.full((BLOCK,), 0.0, tl.float32)
    for i in tl.static_range(8):
        v += tl.load(Experts + (row * 8 + i) * D + d).to(tl.float32)
    g = tl.load(SG + row * 257 + 256).to(tl.float32)
    g = (1.0 / (1.0 + libdevice.exp(-g))).to(tl.bfloat16).to(tl.float32)
    o = tl.load(Shared + row * D + d).to(tl.float32)
    o = (o * g).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * D + d, v.to(tl.bfloat16).to(tl.float32) + o)


@tr.jit
def router_topk(
    R,
    Weights,
    Ids,
    STRIDE: tl.constexpr,
    Alpha=None,
    SharedGate=None,
    WEIGHTED: tl.constexpr = False,
):
    row = tl.program_id(0)
    e = tl.arange(0, 256)
    raw = tl.load(R + row * STRIDE + e)
    x = raw.to(tl.float32)
    raw_bits = tl.where(x == 0.0, 0, raw.to(tl.uint16, bitcast=True)).to(tl.uint32)
    order = tl.where(raw_bits >= 0x8000, raw_bits ^ 0xFFFF, raw_bits ^ 0x8000)
    packed = (order << 8) | (255 - e)
    winning = tl.topk(packed, 8)
    ix = (255 - (winning & 255)).to(tl.int32)
    x = tl.exp(x - tl.max(x, 0))
    all_prob = x / tl.sum(x, 0)
    chosen = tl.gather(all_prob, ix, 0)
    chosen = chosen / tl.sum(chosen, 0)
    if WEIGHTED:
        a = tl.load(Alpha + ix).to(tl.float32)
        chosen = (chosen.to(tl.bfloat16).to(tl.float32) * a).to(tl.bfloat16)
        shared = tl.load(R + row * STRIDE + 256).to(tl.float32)
        g = (1.0 / (1.0 + libdevice.exp(-shared))).to(tl.bfloat16)
        tl.store(SharedGate + row, g)
    slot = tl.arange(0, 8)
    tl.store(Ids + row * 8 + slot, ix)
    tl.store(Weights + row * 8 + slot, chosen)


@tr.jit
def add_shared(
    Routed,
    Shared,
    SG,
    Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    RS: tl.constexpr,
    Alpha=None,
    SCALE: tl.constexpr = False,
):
    row, block = tl.program_id(0), tl.program_id(1)
    i = row * D + block * BLOCK + tl.arange(0, BLOCK)
    r = tl.load(Routed + i).to(tl.float32)
    o = tl.load(Shared + i).to(tl.float32)
    if SCALE:
        o = (o * tl.load(Alpha).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    g = tl.load(SG + row * RS + 256).to(tl.float32)
    g = (1.0 / (1.0 + libdevice.exp(-g))).to(tl.bfloat16).to(tl.float32)
    s = (o * g).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + i, r + s)


@tr.jit
def combine_norm(Experts, Shared, Router, Residual, W, Y, D: tl.constexpr, SAVE: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, D)
    offset = row * D + d
    v = tl.full((D,), 0.0, tl.float32)
    for slot in tl.static_range(8):
        v += tl.load(Experts + (row * 8 + slot) * D + d).to(tl.float32)
    sg = tl.load(Router + row * 257 + 256).to(tl.float32)
    sg = (1.0 / (1.0 + libdevice.exp(-sg))).to(tl.bfloat16).to(tl.float32)
    shared = tl.load(Shared + offset).to(tl.float32)
    shared = (sg * shared).to(tl.bfloat16).to(tl.float32)
    h = (v.to(tl.bfloat16).to(tl.float32) + shared).to(tl.bfloat16).to(tl.float32)
    h += tl.load(Residual + offset).to(tl.float32)
    if SAVE:
        tl.store(Residual + offset, h)
    w = tl.load(W + d).to(tl.float32)
    inv = tl.rsqrt(tl.sum(h * h, 0) / D + 1e-6)
    tl.store(Y + offset, (h * inv) * (1.0 + w))


@tr.jit
def shared_swiglu(Raw, Alpha, Y):
    row = tl.program_id(0)
    d = tl.arange(0, 512)
    g = tl.load(Raw + row * 1024 + d).to(tl.float32)
    u = tl.load(Raw + row * 1024 + 512 + d).to(tl.float32)
    a = tl.load(Alpha).to(tl.float32)
    g = (g * a).to(tl.bfloat16).to(tl.float32)
    u = (u * a).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + row * 512 + d, _silu(g) * u)

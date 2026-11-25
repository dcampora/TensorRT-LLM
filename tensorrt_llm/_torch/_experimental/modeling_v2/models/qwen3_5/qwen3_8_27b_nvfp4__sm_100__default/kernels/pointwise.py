# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pointwise kernels adapted from aegis-engines cb6779a, kernels.py."""

import triton as tr
import triton.language as tl


@tr.jit
def wait_inputs():
    tl.extra.cuda.gdc_wait()


@tr.jit
def launch_dependents():
    tl.extra.cuda.gdc_launch_dependents()


@tr.jit
def static_e4m3(value, scale):
    inv = 1.0 / tl.load(scale)
    return tl.clamp(value * inv, -448.0, 448.0).to(tl.float8e4nv)


@tr.jit
def fp8_convert(x, scale, y, N: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr = False):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inverse = 1.0 / tl.load(scale)
    if PDL:
        wait_inputs()
    val = tl.load(x + i, i < N, 0).to(tl.float32)
    rounded = tl.clamp(val * inverse, -448.0, 448.0).to(tl.float8e4nv)
    tl.store(y + i, rounded, i < N)
    if PDL:
        launch_dependents()


@tr.jit
def fp8_rescale(x, src_scale, dst_scale, y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    src = tl.load(src_scale).to(tl.float16).to(tl.float32)
    dst = 1.0 / tl.load(dst_scale)
    val = tl.load(x + i, i < N, 0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    val = (val * src).to(tl.float16).to(tl.float32)
    tl.store(
        y + i,
        tl.clamp(val * dst, -448.0, 448.0).to(tl.float8e4nv).to(tl.uint8, bitcast=True),
        i < N,
    )


@tr.jit
def gated_norm(
    X,
    Z,
    W,
    Y,
    M: tl.constexpr,
    ZS: tl.constexpr,
    R: tl.constexpr,
    EPS: tl.constexpr,
    Scale=None,
    PACK: tl.constexpr = False,
    PDL: tl.constexpr = False,
):
    r = tl.program_id(0) * R + tl.arange(0, R)
    d = tl.arange(0, 128)
    if PDL:
        wait_inputs()
    x = tl.load(X + r[:, None] * 128 + d[None, :], r[:, None] < M, 0).to(tl.float32)
    z = tl.load(
        Z + (r[:, None] // 48) * ZS + (r[:, None] % 48) * 128 + d[None, :], r[:, None] < M, 0
    ).to(tl.float32)
    w = tl.load(W + d).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 1) / 128 + EPS)
    y = (x * inv[:, None]) * w[None, :]
    y *= z * tl.sigmoid(z)
    if PACK:
        y = static_e4m3(y.to(tl.bfloat16).to(tl.float32), Scale)
    tl.store(Y + r[:, None] * 128 + d[None, :], y, r[:, None] < M)
    if PDL:
        launch_dependents()


@tr.jit
def _qkv_value(P, Index, Sx, Sw, PLANE: tl.constexpr):
    value = tl.load(P + Index)
    if PLANE:
        other = tl.load(P + PLANE + Index)
        scalar = tl.load(Sx) * tl.load(Sw)
        value = ((value + other) * scalar).to(tl.bfloat16)
    return value


@tr.jit
def qkv_rope(
    P,
    Q,
    K,
    V,
    G,
    WQ,
    WK,
    Rope,
    Pos,
    Slots,
    PS: tl.constexpr,
    QS: tl.constexpr,
    NS: tl.constexpr,
    KV_BATCH: tl.constexpr,
    KV_CAP: tl.constexpr,
    WRITE_CACHE: tl.constexpr,
    EPS: tl.constexpr,
    AltK=None,
    AltV=None,
    Sx=None,
    Sw=None,
    PLANE: tl.constexpr = 0,
):
    row, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, 256)
    is_k = h >= QS
    off = h * 512
    if is_k:
        off = QS * 512 + (h - QS) * 256
    x = _qkv_value(P, row * PS + off + d, Sx, Sw, PLANE).to(tl.float32)
    w = tl.load(tl.where(is_k, WK, WQ) + d).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / 256 + EPS)
    normalized = (x * inv * (w + 1.0)).to(tl.bfloat16)
    p = tl.load(Pos + row)
    rk = tl.arange(0, 32)
    cos = tl.load(Rope + p * 64 + rk)
    sin = tl.load(Rope + p * 64 + 32 + rk)
    lo = _qkv_value(P, row * PS + off + rk, Sx, Sw, PLANE).to(tl.float32)
    hi = _qkv_value(P, row * PS + off + rk + 32, Sx, Sw, PLANE).to(tl.float32)
    wl = tl.load(tl.where(is_k, WK, WQ) + rk).to(tl.float32)
    wh = tl.load(tl.where(is_k, WK, WQ) + rk + 32).to(tl.float32)
    lo = (lo * inv * (wl + 1.0)).to(tl.bfloat16).to(tl.float32)
    hi = (hi * inv * (wh + 1.0)).to(tl.bfloat16).to(tl.float32)
    rlo = lo * cos - hi * sin
    rhi = hi * cos + lo * sin
    if is_k:
        head = h - QS
        if WRITE_CACHE:
            slot = tl.load(Slots + row).to(tl.int64)
            # Within each sequence a head owns contiguous token rows.
            target = slot * KV_BATCH + head * KV_CAP * 256 + p * 256
        else:
            target = (row * NS + head) * 256
        val = _qkv_value(P, row * PS + QS * 512 + NS * 256 + head * 256 + d, Sx, Sw, PLANE)
        if AltK is not None and row == 2:
            out = AltK + head * 256
            tl.store(AltV + head * 256 + d, val)
        else:
            out = K + target
            tl.store(V + target + d, val)
    else:
        out = Q + (row * QS + h) * 256
        g = _qkv_value(P, row * PS + off + 256 + d, Sx, Sw, PLANE)
        tl.store(G + (row * QS + h) * 256 + d, g)
    tl.store(out + d, normalized, d >= 64)
    tl.store(out + rk, rlo)
    tl.store(out + 32 + rk, rhi)


@tr.jit
def sigmoid_inplace(X, G, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    g = tl.load(G + i, i < N, 0).to(tl.float32)
    tl.store(X + i, x * tl.sigmoid(g), i < N)

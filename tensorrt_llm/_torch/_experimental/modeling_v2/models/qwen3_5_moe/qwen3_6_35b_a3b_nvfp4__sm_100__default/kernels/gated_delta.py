# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from aegis-engines kernels.py (a6c09c6502)."""

import triton as tr
import triton.language as tl

from .quantized import _silu


@tr.jit
def conv_prefill(
    X,
    W,
    Rows,
    Cu,
    Slots,
    Initial,
    Conv,
    Q,
    K,
    V,
    CS: tl.constexpr,
    CC: tl.constexpr,
    CT: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
):
    token = tl.program_id(0)
    block = tl.program_id(1)
    channels = block * BLOCK + tl.arange(0, BLOCK)
    request = tl.load(Rows + token)
    first = tl.load(Cu + request)
    end = tl.load(Cu + request + 1)
    local = token - first
    result = tl.full((BLOCK,), 0.0, tl.float32)
    for i in tl.static_range(4):
        frame = local + i - 3
        x = tl.load(X + (token + i - 3) * 12288 + channels, frame >= 0, 0)
        if HAS_INITIAL and i < 3:
            history = tl.load(Initial + (request * 8192 + channels) * 3 + frame + 3, frame < 0, 0)
            x = tl.where(frame < 0, history, x)
        w = tl.load(W + channels * 4 + i)
        result += x.to(tl.float32) * w.to(tl.float32)
    result = _silu(result)
    if block < 2048 // BLOCK:
        tl.store(Q + token * 2048 + channels, result)
    elif block < 4096 // BLOCK:
        tl.store(K + token * 2048 + channels - 2048, result)
    else:
        tl.store(V + token * 4096 + channels - 4096, result)
    # One writer per request and channel for all three history frames, including
    # the leading zero/initial frames when the chunk has fewer than three tokens.
    if token == end - 1:
        slot = tl.load(Slots + request).to(tl.int64)
        base = Conv + slot * CS + channels * CC
        for i in tl.static_range(3):
            frame = end - first + i - 3
            x = tl.load(X + (end + i - 3) * 12288 + channels, frame >= 0, 0)
            if HAS_INITIAL:
                history = tl.load(
                    Initial + (request * 8192 + channels) * 3 + frame + 3, frame < 0, 0
                )
                x = tl.where(frame < 0, history, x)
            tl.store(base + i * CT, x)


@tr.jit
def conv_decode(
    X, W, Slots, Conv, Y, CS: tl.constexpr, CC: tl.constexpr, CT: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Slots + row).to(tl.int64)
    base = Conv + slot * CS + d * CC
    x0 = tl.load(base)
    x1 = tl.load(base + CT)
    x2 = tl.load(base + 2 * CT)
    x3 = tl.load(X + row * 12288 + d)
    w0 = tl.load(W + d * 4).to(tl.float32)
    w1 = tl.load(W + d * 4 + 1).to(tl.float32)
    w2 = tl.load(W + d * 4 + 2).to(tl.float32)
    w3 = tl.load(W + d * 4 + 3).to(tl.float32)
    z = x0.to(tl.float32) * w0
    z += x1.to(tl.float32) * w1
    z += x2.to(tl.float32) * w2
    z += x3.to(tl.float32) * w3
    tl.store(base, x1)
    tl.store(base + CT, x2)
    tl.store(base + 2 * CT, x3)
    tl.store(Y + row * 8192 + d, _silu(z))


@tr.jit
def _gdn_gates(BA, ALog, Bias, row, h):
    b = tl.load(BA + row * 64 + h).to(tl.float32)
    a = tl.load(BA + row * 64 + 32 + h).to(tl.float32)
    a_log = tl.load(ALog + h).to(tl.float32)
    x = a + tl.load(Bias + h).to(tl.float32)
    sp = tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    beta = tl.sigmoid(b).to(tl.bfloat16).to(tl.float32)
    return -tl.exp(a_log) * sp, beta


@tr.jit
def gdn_gates(BA, ALog, Bias, G, B, LOG: tl.constexpr):
    row = tl.program_id(0)
    h = tl.arange(0, 32)
    g, beta = _gdn_gates(BA, ALog, Bias, row, h)
    if not LOG:
        g = tl.exp(g)
    tl.store(G + row * 32 + h, g)
    tl.store(B + row * 32 + h, beta)


@tr.jit
def gdn_decode(QKV, BA, ALog, Bias, Slots, State, Y, SS: tl.constexpr, BV: tl.constexpr):
    row, h, vi = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    k = tl.arange(0, 128)
    v = vi * BV + tl.arange(0, BV)
    x = QKV + row * 8192
    q = tl.load(x + (h // 2) * 128 + k).to(tl.float32)
    key = tl.load(x + 2048 + (h // 2) * 128 + k).to(tl.float32)
    val = tl.load(x + 4096 + h * 128 + v).to(tl.float32)
    q = (q / tl.sqrt(tl.sum(q * q, 0) + 1e-6)) * 0.08838834764831845
    key = key / tl.sqrt(tl.sum(key * key, 0) + 1e-6)
    g, beta = _gdn_gates(BA, ALog, Bias, row, h)
    slot = tl.load(Slots + row).to(tl.int64)
    off = slot * SS + (h * 128 + v[:, None]) * 128 + k[None, :]
    s = tl.load(State + off) * tl.exp(g)
    val = (val - tl.sum(s * key[None, :], 1)) * beta
    s += val[:, None] * key[None, :]
    tl.store(State + off, s)
    tl.store(Y + (row * 32 + h) * 128 + v, tl.sum(s * q[None, :], 1))


@tr.jit
def gdn_norm(X, QKVZ, W, Y, ZS: tl.constexpr, Scale=None, QUANT: tl.constexpr = False):
    ro, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, 128)
    i = (ro * 32 + h) * 128 + d
    x = tl.load(X + i).to(tl.float32)
    z = tl.load(QKVZ + ro * ZS + 8192 + h * 128 + d).to(tl.float32)
    w = tl.load(W + d).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / 128.0 + 1e-6)
    y = ((x * inv) * w) * _silu(z)
    if QUANT:
        scale = 1.0 / tl.load(Scale)
        fp8 = y.to(tl.bfloat16).to(tl.float32) * scale
        tl.store(Y + i, tl.minimum(tl.maximum(fp8, -448.0), 448.0))
    else:
        tl.store(Y + i, y)

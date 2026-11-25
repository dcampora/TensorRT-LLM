# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tiled GDN stencil and norm adapted from aegis-engines prefill_stencil.py (a6c09c6502)."""

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


@gluon.jit
def conv(
    X,
    W,
    Rows,
    CU,
    Slots,
    Initial,
    History,
    Q,
    K,
    V,
    M,
    HS: gl.constexpr,
    IS: gl.constexpr,
    R: gl.constexpr,
    C: gl.constexpr,
    HAS_INITIAL: gl.constexpr,
):
    layout: gl.constexpr = BlockedLayout([1, 4], [1, 32], [1, 4], [1, 0])
    t = gl.program_id(0) * R + gl.arange(0, R, layout=SliceLayout(1, layout))
    block = gl.program_id(1)
    channel = block * C + gl.arange(0, C, layout=SliceLayout(0, layout))
    row = gl.load(Rows + t, t < M, 0)
    start = gl.load(CU + row, t < M, 0)
    end = gl.load(CU + row + 1, t < M, 0)
    tail = (t + 1 == end) & (t < M)
    slot = gl.load(Slots + row, tail, 0).to(gl.int64)
    history = History + slot[:, None] * HS[0] + channel[None, :] * HS[1]
    value = gl.full((R, C), 0.0, gl.float32, layout)
    for lag in gl.static_range(4):
        previous = t + lag - 3
        x = gl.load(
            X + previous[:, None] * 12288 + channel[None, :],
            ((previous >= start) & (t < M))[:, None],
            0.0,
        )
        if HAS_INITIAL and lag < 3:
            frame = previous - start + 3
            use_history = (previous < start) & (t < M)
            cached = gl.load(
                Initial + (row * IS[0] + frame * IS[2])[:, None] + channel[None, :] * IS[1],
                use_history[:, None],
                0.0,
            )
            x = gl.where(use_history[:, None], cached, x)
        w = gl.load(W + channel * 4 + lag).to(gl.float32)
        value = gl.fma(x.to(gl.float32), w[None, :], value)
        if lag > 0:
            gl.store(history + (lag - 1) * HS[2], x, tail[:, None])
    result = (value / (1.0 + gl.exp(-value))).to(gl.bfloat16)
    if block < 2048 // C:
        gl.store(Q + t[:, None] * 2048 + channel[None, :], result, (t < M)[:, None])
    elif block < 4096 // C:
        gl.store(K + t[:, None] * 2048 + (channel - 2048)[None, :], result, (t < M)[:, None])
    else:
        gl.store(V + t[:, None] * 4096 + (channel - 4096)[None, :], result, (t < M)[:, None])


@gluon.jit
def _warp_norm(
    Y, QKVZ, W, Out, HEADS, Scale, R: gl.constexpr, NW: gl.constexpr, QUANT: gl.constexpr
):
    layout: gl.constexpr = BlockedLayout([1, 1], [1, 32], [NW, 1], [1, 0])
    h = gl.program_id(0) * R + gl.arange(0, R, layout=SliceLayout(1, layout))
    d = gl.arange(0, 32, layout=SliceLayout(0, layout))
    offset = h[:, None] * 128 + d[None, :]
    zbase = (h // 32 * 12288 + 8192 + h % 32 * 128)[:, None] + d[None, :]
    valid = h[:, None] < HEADS
    x0 = gl.load(Y + offset, valid, 0.0).to(gl.float32)
    x1 = gl.load(Y + offset + 32, valid, 0.0).to(gl.float32)
    x2 = gl.load(Y + offset + 64, valid, 0.0).to(gl.float32)
    x3 = gl.load(Y + offset + 96, valid, 0.0).to(gl.float32)
    # Each warp reproduces the cross-warp XOR-2, XOR-1 reduction order.
    a = gl.sum(x0 * x0, 1) + gl.sum(x2 * x2, 1)
    b = gl.sum(x1 * x1, 1) + gl.sum(x3 * x3, 1)
    inv = gl.rsqrt((a + b) / 128.0 + 1e-6)[:, None]
    for i in gl.static_range(4):
        if i == 0:
            x = x0
        elif i == 1:
            x = x1
        elif i == 2:
            x = x2
        else:
            x = x3
        z = gl.load(QKVZ + zbase + i * 32, valid, 0.0).to(gl.float32)
        w = gl.load(W + d + i * 32).to(gl.float32)
        result = ((x * inv) * w[None, :]) * (z / (1.0 + gl.exp(-z)))
        if QUANT:
            result = result.to(gl.bfloat16).to(gl.float32) * (1.0 / gl.load(Scale))
            result = gl.minimum(gl.maximum(result, -448.0), 448.0)
        gl.store(Out + offset + i * 32, result, valid)


def warp_norm(x, qkvz, weight, out, scale, rows: int, warps: int) -> None:
    heads = out.size(0) * 32
    _warp_norm[(triton.cdiv(heads, rows),)](
        x, qkvz, weight, out, heads, scale, rows, warps, scale is not None, num_warps=warps
    )

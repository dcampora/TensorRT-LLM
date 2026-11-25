# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDN convolution and decode adapted from aegis-engines cb6779a, kernels.py."""

import triton as tr
import triton.language as tl


@tr.jit
def prefill_gates(BA, A, Bias, G, Beta, N: tl.constexpr):
    t = tl.program_id(0)
    h = tl.arange(0, 64)
    b = tl.load(BA + t * (2 * N) + h, h < N, 0).to(tl.float32)
    a = tl.load(BA + t * (2 * N) + N + h, h < N, 0).to(tl.float32)
    a += tl.load(Bias + h, h < N, 0).to(tl.float32)
    exp_a = tl.exp(a)
    # log(1 + exp(a)) loses bits in the negative tail. The first two
    # terms of log1p match FP32 precision for this range.
    softplus = tl.where(
        a < -8.5, exp_a * (1.0 - 0.5 * exp_a), tl.where(a <= 20.0, tl.log(1.0 + exp_a), a)
    )
    tl.store(G + t * N + h, -softplus * tl.exp(tl.load(A + h, h < N, 0).to(tl.float32)), h < N)
    beta = tl.sigmoid(b).to(tl.bfloat16).to(tl.float32)
    tl.store(Beta + t * N + h, beta, h < N)


@tr.jit
def recurrent(QKV, BA, A, Bias, H, Y, Slots, H_STRIDE: tl.constexpr, BV: tl.constexpr):
    tile, work = tl.program_id(0), tl.program_id(1)
    seq, head = work // 48, work % 48
    group = head // 3
    k = tl.arange(0, 128)
    v = tile * BV + tl.arange(0, BV)
    qkv = QKV + seq * 10240
    q = tl.load(qkv + group * 128 + k).to(tl.float32)
    key = tl.load(qkv + 2048 + group * 128 + k).to(tl.float32)
    val = tl.load(qkv + 4096 + head * 128 + v).to(tl.float32)
    q = q / tl.sqrt(tl.sum(q * q) + 1.0e-6)
    key = key / tl.sqrt(tl.sum(key * key) + 1.0e-6)
    q *= 128**-0.5
    a = tl.load(BA + seq * 96 + 48 + head).to(tl.float32)
    b = tl.load(BA + seq * 96 + head).to(tl.float32)
    a += tl.load(Bias + head).to(tl.float32)
    g = -tl.exp(tl.load(A + head).to(tl.float32)) * tl.where(a <= 20.0, tl.log(1.0 + tl.exp(a)), a)
    beta = tl.sigmoid(b).to(tl.bfloat16).to(tl.float32)
    slot = tl.load(Slots + seq).to(tl.int64)
    state_ptr = H + slot * H_STRIDE + head * 16384 + v[:, None] * 128 + k[None, :]
    state = tl.load(state_ptr) * tl.exp(g)
    val = (val - tl.sum(state * key[None, :], 1)) * beta
    state += val[:, None] * key[None, :]
    out = tl.sum(state * q[None, :], 1)
    tl.store(state_ptr, state)
    tl.store(Y + work * 128 + v, out)


@tr.jit
def _conv_product(x, weight):
    return (x * weight).to(tl.float32)


@tr.jit
def context_conv(
    X,
    W,
    Y,
    History,
    Starts,
    Slots,
    Initial,
    XS: tl.constexpr,
    HS: tl.constexpr,
    HC: tl.constexpr,
    HT: tl.constexpr,
    C: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    seq, tile, channel = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(Starts + seq)
    end = tl.load(Starts + seq + 1)
    first = start + tile * BT
    if first < end:
        slot = tl.load(Slots + seq).to(tl.int64)
        initial = tl.load(Initial + seq)
        t = first + tl.arange(0, BT)
        c = channel * BC + tl.arange(0, BC)
        acc = tl.full((BT, BC), 0, tl.float32)
        for j in tl.static_range(4):
            pos = t + j - 3
            live = (pos[:, None] >= start) & (pos[:, None] < end)
            x = tl.load(X + pos[:, None] * XS + c[None, :], live & (c[None, :] < C), 0)
            carried = tl.load(
                History + slot * HS + c[None, :] * HC + (pos[:, None] - start + 3) * HT,
                (pos[:, None] < start) & (t[:, None] < end) & (c[None, :] < C) & initial,
                0,
            )
            w = tl.load(W + c * 4 + j, c < C, 0)
            acc += _conv_product(tl.where(pos[:, None] < start, carried, x), w[None, :])
        tl.store(
            Y + t[:, None] * C + c[None, :],
            acc / (1 + tl.exp(-acc)),
            (t[:, None] < end) & (c[None, :] < C),
        )


@tr.jit
def context_conv_tail(
    X,
    History,
    Starts,
    Slots,
    Initial,
    XS: tl.constexpr,
    HS: tl.constexpr,
    HC: tl.constexpr,
    HT: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
):
    seq, tile = tl.program_id(0), tl.program_id(1)
    c = tile * BC + tl.arange(0, BC)
    start = tl.load(Starts + seq)
    end = tl.load(Starts + seq + 1)
    slot = tl.load(Slots + seq).to(tl.int64)
    initial = tl.load(Initial + seq)
    history = History + slot * HS + c * HC
    for j in tl.static_range(3):
        t = end + j - 3
        tail = tl.load(X + t * XS + c, (t >= start) & (c < C), 0)
        old = tl.load(history + (t - start + 3) * HT, (t < start) & (c < C) & initial, 0)
        tl.store(history + j * HT, tl.where(t >= start, tail, old), c < C)


@tr.jit
def decode_conv(
    X,
    W,
    Y,
    History,
    Slots,
    XS: tl.constexpr,
    HS: tl.constexpr,
    HC: tl.constexpr,
    HT: tl.constexpr,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    slot = tl.load(Slots + row).to(tl.int64)
    c = tile * BLOCK + tl.arange(0, BLOCK)
    history = History + slot * HS + c * HC
    x0 = tl.load(history)
    x1 = tl.load(history + HT)
    x2 = tl.load(history + 2 * HT)
    x3 = tl.load(X + row * XS + c)
    tl.store(history, x1)
    tl.store(history + HT, x2)
    tl.store(history + 2 * HT, x3)
    acc = _conv_product(x0, tl.load(W + c * 4))
    acc += _conv_product(x1, tl.load(W + c * 4 + 1))
    acc += _conv_product(x2, tl.load(W + c * 4 + 2))
    acc += _conv_product(x3, tl.load(W + c * 4 + 3))
    tl.store(Y + row * C + c, acc / (1 + tl.exp(-acc)))

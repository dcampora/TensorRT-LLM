# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from aegis-engines kernels.py (a6c09c6502)."""

import triton as tr
import triton.language as tl


@tr.jit
def qkv_rope(
    X,
    QW,
    KW,
    R,
    Pos,
    Seq,
    Q,
    KV,
    Gate,
    Table,
    TS: tl.constexpr,
    RS: tl.constexpr,
    PAGE: tl.constexpr,
    PS: tl.constexpr,
):
    row, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, 256)
    p = tl.load(Pos + row)
    seq = tl.load(Seq + row)
    table = Table + seq * TS + p // PAGE
    storage = KV + (p % PAGE) * 256 + d

    if h < 18:
        if h < 16:
            i = h * 512
            norm_w = QW
        else:
            i = 8192 + (h - 16) * 256
            norm_w = KW
        base = X + row * 9216 + i
        x = tl.load(base + d).to(tl.float32)
        w = tl.load(norm_w + d).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, 0) / 256.0 + 1e-6)
        v = (x * inv * (1.0 + w)).to(tl.bfloat16).to(tl.float32)
        pair = d ^ 32
        px = tl.load(base + pair, d < 64, 0).to(tl.float32)
        pw = tl.load(norm_w + pair, d < 64, 0).to(tl.float32)
        pv = (px * inv * (1.0 + pw)).to(tl.bfloat16).to(tl.float32)
        c = tl.load(R + p * 64 + d % 32)
        s = tl.load(R + p * 64 + 32 + d % 32)
        signed = tl.where(d < 32, -s, s)
        v = tl.where(d < 64, v * c + pv * signed, v).to(tl.bfloat16).to(tl.float32)
        if h < 16:
            tl.store(Q + (row * 16 + h) * 256 + d, v)
            gate = tl.load(base + 256 + d)
            tl.store(Gate + (row * 16 + h) * 256 + d, gate)
        else:
            page = tl.load(table).to(tl.int64)
            tl.store(storage + page * PS + (h - 16) * (PAGE * 256), v)
    else:
        value = tl.load(X + row * 9216 + 8704 + (h - 18) * 256 + d)
        page = tl.load(table + RS).to(tl.int64)
        tl.store(storage + page * PS + (h - 18) * (PAGE * 256), value)


@tr.jit
def attn_gate(X, Gate, Y, SIZE, BLOCK: tl.constexpr, Scale=None, QUANT: tl.constexpr = False):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < SIZE, 0.0).to(tl.float32)
    g = tl.load(Gate + i, i < SIZE, 0).to(tl.float32)
    value = x * tl.sigmoid(g)
    if QUANT:
        value = value.to(tl.bfloat16).to(tl.float32) * (1.0 / tl.load(Scale))
        value = tl.minimum(tl.maximum(value, -448.0), 448.0)
    tl.store(Y + i, value, i < SIZE)


@tr.jit
def attention(
    Q,
    KV,
    Y,
    Stats,
    CU,
    Pos,
    Rows,
    Table,
    TS: tl.constexpr,
    RS: tl.constexpr,
    PAGE: tl.constexpr,
    PS: tl.constexpr,
    PREFILL: tl.constexpr,
    BQ: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    NS: tl.constexpr,
    PROB_SCALE: tl.constexpr,
):
    block, h, request = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qrow = tl.arange(0, BQ)
    key_col = tl.arange(0, BK)
    d = tl.arange(0, 256)
    if PREFILL:
        seq = request
        first = tl.load(CU + seq)
        length = tl.load(CU + seq + 1) - first
        qfirst = block * (BQ // 8)
        cached = tl.load(Pos + first)
        last = tl.minimum(qfirst + BQ // 8, length) + cached
        qoffset = qfirst + qrow // 8
        qpos = cached + qoffset
        qidx = (first + qoffset) * 16 + h * 8 + qrow % 8
        start = 0
        active = qfirst < length
    else:
        seq = tl.load(Rows + request)
        length = tl.load(Pos + request) + 1
        start = block * SPLIT
        last = tl.minimum(start + SPLIT, length)
        qpos = tl.full((BQ,), length - 1, tl.int32)
        qidx = request * 16 + h * 8 + (qrow % 8)
        active = start < length
    if active:
        if PREFILL:
            valid = qoffset < length
        else:
            valid = qrow < 8
        q = tl.load(Q + qidx[:, None] * 256 + d[None, :], valid[:, None], 0.0)
        m = tl.full((BQ,), -float("inf"), tl.float32)
        normalizer = tl.full((BQ,), 0.0, tl.float32)
        a = tl.full((BQ, 256), 0.0, tl.float32)
        for lo in range(start, last, BK):
            kpos = lo + key_col
            table = Table + seq * TS + kpos // PAGE
            kpage = tl.load(table, kpos < last, 0).to(tl.int64)
            vpage = tl.load(table + RS, kpos < last, 0).to(tl.int64)
            offset = h * (PAGE * 256) + (kpos % PAGE) * 256
            k = tl.load(KV + (kpage * PS + offset)[:, None] + d[None, :], kpos[:, None] < last, 0.0)
            v = tl.load(KV + (vpage * PS + offset)[:, None] + d[None, :], kpos[:, None] < last, 0.0)
            s = tl.dot(q, tl.trans(k)) * 0.09016844005556021
            s = tl.where(
                (qpos[:, None] >= kpos[None, :]) & (kpos[None, :] < last), s, -float("inf")
            )
            curr = tl.maximum(m, tl.max(s, 1))
            factor = tl.exp2(m - curr)
            p = tl.exp2(s - curr[:, None])
            normalizer = normalizer * factor + tl.sum(p, 1)
            a = a * factor[:, None]
            a = tl.dot((p * PROB_SCALE).to(tl.float8e4nv), v, a)
            m = curr
        if PREFILL:
            o = (a / PROB_SCALE) / normalizer[:, None]
            tl.store(Y + qidx[:, None] * 256 + d[None, :], o, valid[:, None])
        else:
            outrow = qidx * NS + block
            tl.store(Y + outrow[:, None] * 256 + d[None, :], a / PROB_SCALE, valid[:, None])
            tl.store(Stats + outrow * 2, m, valid)
            tl.store(Stats + outrow * 2 + 1, normalizer, valid)


@tr.jit
def attn_reduce(
    Part,
    Stats,
    Pos,
    Y,
    NS: tl.constexpr,
    SPLIT: tl.constexpr,
    BS: tl.constexpr,
    Gate=None,
    Scale=None,
    QUANT: tl.constexpr = False,
    APPLY_GATE: tl.constexpr = False,
):
    row = tl.program_id(0)
    split = tl.arange(0, BS)
    d = tl.arange(0, 256)
    count = tl.load(Pos + row // 16) // SPLIT + 1
    off = row * NS + split
    m = tl.load(Stats + off * 2, split < count, -float("inf"))
    normalizer = tl.load(Stats + off * 2 + 1, split < count, 0.0)
    a = tl.load(Part + off[:, None] * 256 + d[None, :], split[:, None] < count, 0.0)
    factor = tl.exp2(m - tl.max(m, 0))
    output = tl.sum(a * factor[:, None], 0) / tl.sum(normalizer * factor, 0)
    if QUANT or APPLY_GATE:
        gate = tl.load(Gate + row * 256 + d).to(tl.float32)
        output = output.to(tl.bfloat16).to(tl.float32) * tl.sigmoid(gate)
    if QUANT:
        fp8 = output.to(tl.bfloat16).to(tl.float32) * (1.0 / tl.load(Scale))
        output = tl.minimum(tl.maximum(fp8, -448.0), 448.0)
    tl.store(Y + row * 256 + d, output)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped causal attention adapted from aegis-engines cb6779a, kernels.py."""

import triton as tr
import triton.language as tl

from .pointwise import launch_dependents, static_e4m3, wait_inputs


@tr.jit
def causal_prefill(
    Q,
    K,
    V,
    Y,
    Starts,
    Blocks,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    EXP2: tl.constexpr,
    QUERY_START: tl.constexpr = 0,
    Positions=None,
    Table=None,
    TABLE_STRIDE: tl.constexpr = 0,
    PAGE_SIZE: tl.constexpr = 0,
    PAGE_STRIDE: tl.constexpr = 0,
    PAGE_DIVISOR: tl.constexpr = 0,
):
    work, h = tl.program_id(0), tl.program_id(1)
    seq = tl.load(Blocks + work * 2)
    tile = tl.load(Blocks + work * 2 + 1)
    start = tl.load(Starts + seq)
    end = tl.load(Starts + seq + 1) - start
    qt = tile * BM + tl.arange(0, BM)
    nt = tl.arange(0, BN)
    d = tl.arange(0, 256)
    q = tl.load(
        Q + ((start + qt[:, None] - QUERY_START) * HEADS + h) * 256 + d[None, :],
        qt[:, None] < end,
        0,
    )
    hk = h // GROUP
    key = K + (start * (HEADS // GROUP) + hk) * 256
    val = V + (start * (HEADS // GROUP) + hk) * 256
    stride = (HEADS // GROUP) * 256
    prefix = 0 if Positions is None else tl.load(Positions + start)
    mx = tl.full((BM,), -float("inf"), tl.float32)
    norm = tl.full((BM,), 0, tl.float32)
    acc = tl.full((BM, 256), 0, tl.float32)
    stop = prefix + tl.minimum(end, (tile + 1) * BM)
    for base in range(0, stop, BN):
        kt = base + nt
        if Table is None:
            kval = tl.load(key + d[:, None] + kt[None, :] * stride, kt[None, :] < stop, 0)
            vval = tl.load(val + kt[:, None] * stride + d[None, :], kt[:, None] < stop, 0)
        else:
            pages = tl.load(Table + seq * TABLE_STRIDE + kt // PAGE_SIZE, kt < stop, 0)
            offset = (pages.to(tl.int64) // PAGE_DIVISOR) * PAGE_STRIDE
            offset += (hk * PAGE_SIZE + kt % PAGE_SIZE) * 256
            kval = tl.load(K + offset[None, :] + d[:, None], kt[None, :] < stop, 0)
            vval = tl.load(V + offset[:, None] + d[None, :], kt[:, None] < stop, 0)
        scores = tl.dot(q, kval)
        if EXP2:
            scores *= 0.0625 * 1.4426950408889634
        else:
            scores *= 0.0625
        scores = tl.where(
            (prefix + qt[:, None] >= kt[None, :]) & (kt[None, :] < stop), scores, -float("inf")
        )
        next_max = tl.maximum(mx, tl.max(scores, 1))
        if EXP2:
            p = tl.exp2(scores - next_max[:, None])
            alpha = tl.exp2(mx - next_max)
        else:
            p = tl.exp(scores - next_max[:, None])
            alpha = tl.exp(mx - next_max)
        norm = norm * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), vval)
        mx = next_max
    tl.store(
        Y + ((start + qt[:, None] - QUERY_START) * HEADS + h) * 256 + d[None, :],
        acc / norm[:, None],
        qt[:, None] < end,
    )


@tr.jit
def paged_decode_part(
    Q,
    K,
    V,
    Mid,
    Lse,
    Pos,
    Table,
    TAB_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    PAGE_DIVISOR: tl.constexpr,
    HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    SPLITS: tl.constexpr,
    BN: tl.constexpr,
    BH: tl.constexpr,
    ROWS_PER_SEQ: tl.constexpr,
    REQ_START: tl.constexpr,
):
    seq, hk, part = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    n = tl.load(Pos + seq) + 1
    width = tl.cdiv(tl.cdiv(n, SPLITS), 32) * 32
    begin = part * width
    end = tl.minimum(n, begin + width)
    if begin < end:
        h = tl.arange(0, BH)
        qh = hk * GROUP + h
        d = tl.arange(0, 256)
        cols = tl.arange(0, BN)
        q = tl.load(Q + (seq * HEADS + qh[:, None]) * 256 + d[None, :], h[:, None] < GROUP, 0)
        request = seq // ROWS_PER_SEQ + REQ_START
        pages = Table + request * TAB_STRIDE
        mx = tl.full((BH,), -float("inf"), tl.float32)
        norm = tl.full((BH,), 0, tl.float32)
        acc = tl.full((BH, 256), 0, tl.float32)
        for offset in range(begin, end, BN):
            t = offset + cols
            page = tl.load(pages + t // PAGE_SIZE, t < end, 0)
            kvbase = (page.to(tl.int64) // PAGE_DIVISOR) * PAGE_STRIDE
            kvbase += (hk * PAGE_SIZE + t % PAGE_SIZE) * 256
            key = tl.load(K + kvbase[None, :] + d[:, None], t[None, :] < end, 0)
            val = tl.load(V + kvbase[:, None] + d[None, :], t[:, None] < end, 0)
            score = tl.dot(q, key) * 0.0625
            score = tl.where(t[None, :] < end, score, -float("inf"))
            nm = tl.maximum(tl.max(score, 1), mx)
            p = tl.exp(score - nm[:, None])
            a = tl.exp(mx - nm)
            acc = acc * a[:, None] + tl.dot(p.to(tl.bfloat16), val)
            norm = norm * a + tl.sum(p, 1)
            mx = nm
        index = (seq * HEADS + qh) * MAX_SPLITS + part
        tl.store(Mid + index[:, None] * 256 + d[None, :], acc / norm[:, None], h[:, None] < GROUP)
        tl.store(Lse + index, mx + tl.log(norm), h < GROUP)


@tr.jit
def decode_merge(
    Mid,
    Lse,
    Pos,
    Gate,
    Out,
    HEADS: tl.constexpr,
    MAX_SPLITS: tl.constexpr,
    SPLITS: tl.constexpr,
    Scale=None,
    PACK: tl.constexpr = False,
    PDL: tl.constexpr = False,
):
    seq, head = tl.program_id(0), tl.program_id(1)
    n = tl.load(Pos + seq) + 1
    width = tl.cdiv(tl.cdiv(n, SPLITS), 32) * 32
    parts = tl.cdiv(n, width)
    d = tl.arange(0, 256)
    index = (seq * HEADS + head) * MAX_SPLITS
    if PDL:
        wait_inputs()
    mx = -float("inf")
    norm = 0.0
    acc = tl.full((256,), 0, tl.float32)
    for part in range(parts):
        score = tl.load(Lse + index + part)
        v = tl.load(Mid + (index + part) * 256 + d)
        nm = tl.maximum(mx, score)
        a = tl.exp(mx - nm)
        b = tl.exp(score - nm)
        acc = acc * a + b * v
        norm = norm * a + b
        mx = nm
    v = (acc / norm).to(tl.bfloat16).to(tl.float32)
    gate = tl.load(Gate + (seq * HEADS + head) * 256 + d).to(tl.float32)
    y = v * tl.sigmoid(gate)
    if PACK:
        y = static_e4m3(y.to(tl.bfloat16).to(tl.float32), Scale)
    tl.store(Out + (seq * HEADS + head) * 256 + d, y)
    if PDL:
        launch_dependents()


@tr.jit
def append_paged(
    K,
    V,
    PageK,
    PageV,
    Pos,
    Table,
    Req,
    PAGE_SIZE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    PAGE_DIVISOR: tl.constexpr,
    TAB_STRIDE: tl.constexpr,
    ROWS_PER_SEQ: tl.constexpr,
    REQ_START: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    request = row // ROWS_PER_SEQ + REQ_START if Req is None else tl.load(Req + row)
    length = tl.load(Pos + row)
    block = tl.load(Table + request * TAB_STRIDE + length // PAGE_SIZE)
    d = tl.arange(0, 256)
    source = (row * 4 + head) * 256 + d
    dest = (block.to(tl.int64) // PAGE_DIVISOR) * PAGE_STRIDE
    dest += (head * PAGE_SIZE + length % PAGE_SIZE) * 256 + d
    tl.store(PageK + dest, tl.load(K + source))
    tl.store(PageV + dest, tl.load(V + source))

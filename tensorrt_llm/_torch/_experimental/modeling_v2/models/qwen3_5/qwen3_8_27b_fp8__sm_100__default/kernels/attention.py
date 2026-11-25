# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from aegis-engines qwen3.8-27b-fp8-batched, kernels.py (114cf57c911e)."""

import torch
import triton as tr
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

MATCH_ROUNDING = True
ROUND_ATTN_GATE = False


@tr.jit
def _qkv(
    X,
    WQ,
    WK,
    ROPE,
    POS,
    SLOTS,
    Q,
    K,
    V,
    CK,
    CV,
    SELECTED,
    AK,
    AV,
    L: tl.constexpr,
    DECODE: tl.constexpr,
    CACHE: tl.constexpr,
    SELECTED_COUNT: tl.constexpr,
):
    row = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, 256)
    p = tl.load(POS + row)
    query_row = row
    keep = True
    if SELECTED is not None:
        query_row = row % SELECTED_COUNT
        keep = (h >= 24) | (tl.load(SELECTED + query_row) == row)
    if keep:
        if h < 28:
            is_q = h < 24
            offset = tl.where(is_q, h * 512, 12288 + (h - 24) * 256)
            xptr = X + row * 14336 + offset
            x = tl.load(xptr + d).to(tl.float32)
            w = tl.load(tl.where(is_q, WQ, WK) + d).to(tl.float32)
            inv = tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 256.0) + 1.0e-6)
            x = (x * inv * (1.0 + w)).to(tl.bfloat16).to(tl.float32)
            dr = d ^ 32
            rot = tl.load(xptr + dr, d < 64, 0).to(tl.float32)
            wr = tl.load(tl.where(is_q, WQ, WK) + dr, d < 64, 0).to(tl.float32)
            rot = (rot * inv * (1.0 + wr)).to(tl.bfloat16).to(tl.float32)
            cos = tl.load(ROPE + p * 64 + d % 32, d < 64, 1)
            sin = tl.load(ROPE + p * 64 + 32 + d % 32, d < 64, 0)
            y = tl.where(d < 64, x * cos + tl.where(d < 32, -rot, rot) * sin, x)
            if h < 24:
                tl.store(Q + (query_row * 24 + h) * 256 + d, y)
            else:
                if not DECODE:
                    tl.store(K + (row * 4 + h - 24) * 256 + d, y)
                if CACHE:
                    slot = row if SLOTS is None else tl.load(SLOTS + row)
                    if AK is not None and row == tl.num_programs(0) - 1:
                        tl.store(AK + (h - 24) * 256 + d, y)
                    else:
                        tl.store(CK + ((slot * L + p) * 4 + h - 24) * 256 + d, y)
        else:
            v = tl.load(X + row * 14336 + 13312 + (h - 28) * 256 + d)
            if not DECODE:
                tl.store(V + (row * 4 + h - 28) * 256 + d, v)
            if CACHE:
                slot = row if SLOTS is None else tl.load(SLOTS + row)
                if AV is not None and row == tl.num_programs(0) - 1:
                    tl.store(AV + (h - 28) * 256 + d, v)
                else:
                    tl.store(CV + ((slot * L + p) * 4 + h - 28) * 256 + d, v)


@gluon.jit
def _qkv_prefill(
    X, WQ, WK, ROPE, POS, SLOTS, Q, K, V, CK, CV, L: gl.constexpr, SOURCE: gl.constexpr
):
    group, row = gl.program_id(0), gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 2], [1, 1, 32], [4, 1, 1], [2, 0, 1])
    h_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, layout))
    part_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, layout))
    d_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, layout))
    head = group * 4 + gl.arange(0, 4, layout=h_layout)
    part = gl.arange(0, 4, layout=part_layout)
    lane = gl.arange(0, 64, layout=d_layout)
    column = part[None, :, None] * 64 + lane[None, None, :]
    pos = gl.load(POS + row)
    slot = gl.load(SLOTS + row) if SLOTS is not None else row
    if group < 7:
        is_query = group < 6
        head_offset = gl.where(is_query, head * 512, 12288 + (head - 24) * 256)
        source = X + row * SOURCE + head_offset[:, None, None]
        weights = gl.where(is_query, WQ, WK)
        raw = gl.load(source + column).to(gl.float32)
        weight = gl.load(weights + column).to(gl.float32)
        # Keep the four 64-channel reductions used by the single-head kernel.
        partials = gl.sum(raw * raw, axis=2)
        inv = gl.rsqrt(gl.sum(partials, axis=1) * (1.0 / 256.0) + 1.0e-6)
        y = (raw * inv[:, None, None] * (1.0 + weight)).to(gl.bfloat16).to(gl.float32)
        rotary = column ^ 32
        rotate = gl.load(source + rotary, column < 64, 0.0).to(gl.float32)
        rotate_w = gl.load(weights + rotary, column < 64, 0.0).to(gl.float32)
        rotate = (rotate * inv[:, None, None] * (1.0 + rotate_w)).to(gl.bfloat16).to(gl.float32)
        angle = ROPE + pos * 64 + column % 32
        cosine = gl.load(angle, column < 64, 1.0)
        sine = gl.load(angle + 32, column < 64, 0.0)
        y = gl.where(column < 64, y * cosine + gl.where(column < 32, -rotate, rotate) * sine, y)
        if is_query:
            gl.store(Q + (row * 24 + head[:, None, None]) * 256 + column, y)
        else:
            key = head[:, None, None] - 24
            gl.store(K + (row * 4 + key) * 256 + column, y)
            if CK is not None:
                gl.store(CK + ((slot * L + pos) * 4 + key) * 256 + column, y)
    else:
        value = head[:, None, None] - 28
        values = gl.load(X + row * SOURCE + 13312 + value * 256 + column)
        gl.store(V + (row * 4 + value) * 256 + column, values)
        if CV is not None:
            gl.store(CV + ((slot * L + pos) * 4 + value) * 256 + column, values)


def qkv(
    x,
    wq,
    wk,
    rope,
    pos,
    cache=None,
    slots=None,
    decode=False,
    selected=None,
    alternate=None,
    tiled_prefill=False,
):
    m = x.shape[0]
    q = torch.empty(
        (m if selected is None else selected.numel(), 24, 256), dtype=x.dtype, device=x.device
    )
    k = None if decode else torch.empty((m, 4, 256), dtype=x.dtype, device=x.device)
    v = None if decode else torch.empty_like(k)
    ck, cv = cache if cache is not None else (None, None)
    cache_len = 0 if ck is None else ck.shape[1]
    ak, av = alternate if alternate is not None else (None, None)
    if tiled_prefill and not decode and m >= 128:
        _qkv_prefill[(8, m)](
            x,
            wq,
            wk,
            rope,
            pos,
            slots,
            q,
            k,
            v,
            ck,
            cv,
            cache_len,
            x.shape[1],
            num_warps=4,
            enable_fp_fusion=MATCH_ROUNDING,
        )
    else:
        _qkv[(m, 32)](
            x,
            wq,
            wk,
            rope,
            pos,
            slots,
            q,
            k,
            v,
            ck,
            cv,
            selected,
            ak,
            av,
            cache_len,
            decode,
            cache is not None,
            0 if selected is None else selected.numel(),
            num_warps=4,
            enable_fp_fusion=MATCH_ROUNDING,
        )
    return q, k, v


@tr.jit
def _prefill_gate(Y, X, OUT, M: tl.constexpr, D: tl.constexpr, ROUND: tl.constexpr):
    i = tl.program_id(0) * D + tl.arange(0, D)
    row = i // 6144
    h = (i % 6144) // 256
    d = i % 256
    y = tl.load(Y + i, row < M, 0).to(tl.float32)
    g = tl.load(X + row * 14336 + h * 512 + 256 + d, row < M, 0).to(tl.float32)
    sig = tl.sigmoid(g)
    if ROUND:
        sig = sig.to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + i, y * sig, row < M)


def prefill_gate(y, x):
    m = x.shape[0]
    out = torch.empty((m, 6144), device=x.device, dtype=x.dtype)
    _prefill_gate[(tr.cdiv(m * 6144, 256),)](y, x, out, m, 256, ROUND_ATTN_GATE, num_warps=4)
    return out


@tr.jit
def _dense_causal(
    Q,
    K,
    V,
    X,
    START,
    PAIRS,
    Y,
    BM: tl.constexpr,
    BN: tl.constexpr,
    FP32_P: tl.constexpr,
    VALUE_TILE: tl.constexpr,
):
    block = tl.program_id(0)
    h = tl.program_id(1)
    seq = tl.load(PAIRS + block * 2)
    order = tl.load(PAIRS + block * 2 + 1)
    start = tl.load(START + seq)
    size = tl.load(START + seq + 1) - start
    i = order * BM + tl.arange(0, BM)
    j = tl.arange(0, BN)
    d = tl.arange(0, 256)
    value_cols = tl.program_id(2) * VALUE_TILE + tl.arange(0, VALUE_TILE)
    q = tl.load(Q + ((start + i[:, None]) * 24 + h) * 256 + d[None, :], i[:, None] < size, 0)
    kv = (start * 4 + h // 6) * 256
    accum = tl.full((BM, VALUE_TILE), 0.0, tl.float32)
    prev = tl.full((BM,), -1.0e20, tl.float32)
    den = tl.full((BM,), 0.0, tl.float32)
    for look in range(tl.cdiv(tl.minimum((order + 1) * BM, size), BN)):
        t = look * BN + j
        kk = tl.load(K + kv + t[None, :] * 1024 + d[:, None], t[None, :] < size, 0)
        vv = tl.load(V + kv + t[:, None] * 1024 + value_cols[None, :], t[:, None] < size, 0)
        s = tl.dot(q, kk) * 0.0625
        s = tl.where((t[None, :] <= i[:, None]) & (t[None, :] < size), s, -float("inf"))
        peak = tl.maximum(prev, tl.max(s, 1))
        prob = tl.exp(s - peak[:, None])
        r = tl.exp(prev - peak)
        if FP32_P:
            dot = tl.dot(prob, vv.to(tl.float32), input_precision="tf32x3")
        else:
            dot = tl.dot(prob.to(tl.bfloat16), vv)
        accum = accum * r[:, None] + dot
        den = den * r + tl.sum(prob, 1)
        prev = peak
    attn = (accum / den[:, None]).to(tl.bfloat16).to(tl.float32)
    gate = tl.load(
        X + (start + i[:, None]) * 14336 + h * 512 + 256 + value_cols[None, :], i[:, None] < size, 0
    ).to(tl.float32)
    tl.store(
        Y + ((start + i[:, None]) * 24 + h) * 256 + value_cols[None, :],
        attn * tl.sigmoid(gate),
        i[:, None] < size,
    )


def dense_causal(q, k, v, p, offsets, pairs, fp32_p=False, value_tile=256):
    y = torch.empty((q.shape[0], 6144), dtype=p.dtype, device=p.device)
    tile = 32 if fp32_p else 64
    # The host prepares blocks of 64 query rows. With a smaller K tile the Q tile stays 64.
    _dense_causal[(pairs.shape[0], 24, 256 // value_tile)](
        q,
        k,
        v,
        p,
        offsets,
        pairs,
        y,
        64,
        tile,
        fp32_p,
        value_tile,
        num_warps=8,
        num_stages=1 if fp32_p else 3,
    )
    return y

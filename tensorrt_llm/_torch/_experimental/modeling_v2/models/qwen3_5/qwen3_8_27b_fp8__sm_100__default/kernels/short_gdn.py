# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triangular prefill adapted from aegis-engines gdn.py (114cf57c911e)."""

import torch
import triton as tr
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@tr.jit
def _unit_qk(X, Y):
    row = tl.program_id(0)
    head = tl.program_id(1)
    feature = tl.arange(0, 128)
    col = head * 128 + feature
    x = tl.load(X + row * 10240 + col).to(tl.float32)
    x = x * tl.rsqrt(tl.sum(x * x, 0) + 1.0e-6)
    tl.store(Y + row * 4096 + col, x)


@tr.jit
def _make_lower(K, G, B, START, CHUNKS, A, SCAN, T: tl.constexpr):
    chunk = tl.program_id(0)
    h = tl.program_id(1)
    seq = tl.load(CHUNKS + chunk * 2)
    order = tl.load(CHUNKS + chunk * 2 + 1)
    start = tl.load(START + seq) + order * T
    end = tl.load(START + seq + 1)
    t = tl.arange(0, T)
    k = tl.arange(0, 128)
    token = start + t
    valid = token < end
    gates = tl.load(G + token * 48 + h, valid, 0.0)
    cum = tl.cumsum(gates, 0)
    beta = tl.load(B + token * 48 + h, valid, 0.0)
    keys = tl.load(
        K + token[:, None] * 4096 + 2048 + (h // 3) * 128 + k[None, :], valid[:, None], 0
    )
    gram = tl.dot(keys, tl.trans(keys))
    causal = (t[:, None] > t[None, :]) & valid[:, None] & valid[None, :]
    g = tl.where(causal, cum[:, None] - cum[None, :], -float("inf"))
    lower = -gram * tl.exp(g) * beta[:, None]

    base = (chunk * 48 + h) * T
    tl.store(A + (base + t[:, None]) * T + t[None, :], lower)
    tl.store(SCAN + base + t, cum)


@gluon.jit
def _solve(LOWER, START, CHUNKS, A, T: gl.constexpr):
    C: gl.constexpr = 16
    chunk = gl.program_id(0)
    h = gl.program_id(1)
    part = gl.program_id(2)
    seq = gl.load(CHUNKS + chunk * 2)
    order = gl.load(CHUNKS + chunk * 2 + 1)
    start = gl.load(START + seq) + order * T
    end = gl.load(START + seq + 1)
    # Keep the original row reduction tree while assigning columns to more CTAs.
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [16, 2], [4, 1], [1, 0])
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    col_layout: gl.constexpr = gl.SliceLayout(0, layout)
    r = gl.arange(0, T, layout=row_layout)
    c = part * C + gl.arange(0, C, layout=col_layout)
    base = (chunk * 48 + h) * T
    result = gl.full((T, C), 0, gl.float32, layout)
    for cur in range(1, gl.minimum(T, end - start)):
        offset = LOWER + (base + cur) * T
        coeff = gl.load(offset + c)
        forward = gl.load(offset + r)
        row = coeff + gl.sum(forward[:, None] * result, 0)
        result = gl.where(r[:, None] == cur, row[None, :], result)
    result += (r[:, None] == c[None, :]).to(gl.float32)
    stores: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    out_r = gl.arange(0, T, layout=gl.SliceLayout(1, stores))
    out_c = part * C + gl.arange(0, C, layout=gl.SliceLayout(0, stores))
    gl.store(A + (base + out_r[:, None]) * T + out_c[None, :], gl.convert_layout(result, stores))


@tr.jit
def _apply(
    X,
    QK,
    B,
    START,
    CHUNK_START,
    A,
    SCAN,
    Y,
    S,
    SLOTS,
    STATE_STRIDE: tl.constexpr,
    T: tl.constexpr,
    BV: tl.constexpr,
    ONE: tl.constexpr,
):
    seq = tl.program_id(0)
    h = tl.program_id(1)
    v = tl.program_id(2) * BV + tl.arange(0, BV)
    k = tl.arange(0, 128)
    t = tl.arange(0, T)
    start = tl.load(START + seq)
    end = tl.load(START + seq + 1)
    chunk0 = tl.load(CHUNK_START + seq)
    state = tl.full((BV, 128), 0, tl.float32)
    for c in range(tl.cdiv(end - start, T)):
        token = start + c * T + t
        valid = token < end
        base = ((chunk0 + c) * 48 + h) * T
        inverse = tl.load(A + (base + t[:, None]) * T + t[None, :])
        cum = tl.load(SCAN + base + t)
        scale_v = tl.load(B + token * 48 + h, valid, 0)
        qk_p = QK + token[:, None] * 4096 + h // 3 * 128 + k[None, :]
        q = tl.load(qk_p, valid[:, None], 0)
        key = tl.load(qk_p + 2048, valid[:, None], 0)
        x = tl.load(X + token[:, None] * 10240 + 4096 + h * 128 + v[None, :], valid[:, None], 0).to(
            tl.float32
        )
        u = tl.dot(inverse, (x * scale_v[:, None]).to(tl.bfloat16)).to(tl.bfloat16)
        if not ONE:
            w_input = (key.to(tl.float32) * scale_v[:, None] * tl.exp(cum[:, None])).to(tl.bfloat16)
            w = tl.dot(inverse, w_input).to(tl.bfloat16)
            state_bf16 = state.to(tl.bfloat16)
            delta = u.to(tl.float32) - tl.dot(w, tl.trans(state_bf16))
            old = tl.dot(q, tl.trans(state_bf16)) * tl.exp(cum[:, None])
        else:
            delta = u.to(tl.float32)
            old = tl.full((T, BV), 0, tl.float32)

        score = tl.dot(q, tl.trans(key))
        mask = (t[:, None] >= t[None, :]) & valid[:, None] & valid[None, :]
        p = score * tl.exp(tl.where(mask, cum[:, None] - cum[None, :], -float("inf")))
        local = tl.dot(p.to(tl.bfloat16), delta.to(tl.bfloat16))
        factor: tl.constexpr = 0.08838834764831845
        out = old * factor + local * factor
        tl.store(Y + (token[:, None] * 48 + h) * 128 + v[None, :], out, valid[:, None])
        if S is not None or not ONE:
            last = end - start - c * T - 1
            glast = tl.sum(tl.where(t == tl.minimum(last, T - 1), cum, 0.0), 0)
            decayed = delta * tl.exp(tl.where(valid, glast - cum, -float("inf")))[:, None]
            state = state * tl.exp(glast) + tl.dot(tl.trans(decayed.to(tl.bfloat16)), key)
    if S is not None:
        slot = tl.load(SLOTS + seq).to(tl.int64)
        tl.store(
            S + slot * STATE_STRIDE + (h * 128 + v[:, None]) * 128 + k[None, :], state, slot >= 0
        )


def chunked_delta(conv, gates, beta, offsets, chunks, chunk_starts, max_len, state, slots):
    rows = conv.shape[0]
    qk = torch.empty((rows, 4096), dtype=torch.bfloat16, device=conv.device)
    out = torch.empty((rows, 48, 128), dtype=torch.bfloat16, device=conv.device)
    tile = 64
    inverted = torch.empty(
        (chunks.shape[0], 48, tile, tile), device=conv.device, dtype=torch.bfloat16
    )
    scan = torch.empty((chunks.shape[0], 48, tile), device=conv.device, dtype=torch.float32)
    _unit_qk[(rows, 32)](conv, qk, num_warps=1)
    lower = torch.empty_like(inverted, dtype=torch.float32)
    _make_lower[(chunks.shape[0], 48)](
        qk, gates, beta, offsets, chunks, lower, scan, tile, num_warps=4
    )
    _solve[(chunks.shape[0], 48, tile // 16)](
        lower,
        offsets,
        chunks,
        inverted,
        tile,
        num_warps=4,
    )
    _apply[(offsets.numel() - 1, 48, 4)](
        conv,
        qk,
        beta,
        offsets,
        chunk_starts,
        inverted,
        scan,
        out,
        state,
        slots,
        state.stride(0),
        tile,
        32,
        max_len <= tile,
        num_warps=4,
    )
    return out

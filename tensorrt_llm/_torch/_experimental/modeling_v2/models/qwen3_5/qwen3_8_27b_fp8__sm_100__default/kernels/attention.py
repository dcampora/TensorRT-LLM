# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed attention from aegis-engines kernels.py (114cf57c911e)."""

import torch
import triton as tr
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@tr.jit
def _pack_qkv(X, WQ, WK, ROPE, POS, QKV):
    row = tl.program_id(0)
    head = tl.program_id(1)
    channel = tl.arange(0, 256)
    position = tl.load(POS + row)
    if head < 28:
        query = head < 24
        offset = tl.where(query, head * 512, 12288 + (head - 24) * 256)
        ptr = X + row * 14336 + offset
        values = tl.load(ptr + channel).to(tl.float32)
        weights = tl.where(query, WQ, WK)
        w = tl.load(weights + channel).to(tl.float32)
        inverse = tl.rsqrt(tl.sum(values * values, 0) * (1.0 / 256.0) + 1e-6)
        y = (values * inverse * (1.0 + w)).to(tl.bfloat16).to(tl.float32)
        rotary = channel ^ 32
        alternate = tl.load(ptr + rotary, channel < 64, 0.0).to(tl.float32)
        alt_weight = tl.load(weights + rotary, channel < 64, 0.0).to(tl.float32)
        alternate = (alternate * inverse * (1.0 + alt_weight)).to(tl.bfloat16).to(tl.float32)
        cosine = tl.load(ROPE + position * 64 + channel % 32, channel < 64, 1.0)
        sine = tl.load(ROPE + position * 64 + 32 + channel % 32, channel < 64, 0.0)
        y = tl.where(
            channel < 64, y * cosine + tl.where(channel < 32, -alternate, alternate) * sine, y
        )
    else:
        y = tl.load(X + row * 14336 + 13312 + (head - 28) * 256 + channel).to(tl.float32)
    tl.store(QKV + row * 8192 + head * 256 + channel, y)


@gluon.jit
def _pack_qkv_tiled(X, WQ, WK, ROPE, POS, QKV):
    group, row = gl.program_id(0), gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 2], [1, 1, 32], [4, 1, 1], [2, 0, 1])
    head = group * 4 + gl.arange(0, 4, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    part = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    lane = gl.arange(0, 64, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    channel = part[None, :, None] * 64 + lane[None, None, :]
    pos = gl.load(POS + row)
    if group < 7:
        query = group < 6
        offset = gl.where(query, head * 512, 12288 + (head - 24) * 256)
        source = X + row * 14336 + offset[:, None, None]
        weights = gl.where(query, WQ, WK)
        values = gl.load(source + channel).to(gl.float32)
        w = gl.load(weights + channel).to(gl.float32)
        parts = gl.sum(values * values, axis=2)
        inverse = gl.rsqrt(gl.sum(parts, axis=1) * (1.0 / 256.0) + 1e-6)
        y = (values * inverse[:, None, None] * (1.0 + w)).to(gl.bfloat16).to(gl.float32)
        rotary = channel ^ 32
        alternate = gl.load(source + rotary, channel < 64, 0.0).to(gl.float32)
        alt_weight = gl.load(weights + rotary, channel < 64, 0.0).to(gl.float32)
        alternate = (
            (alternate * inverse[:, None, None] * (1.0 + alt_weight)).to(gl.bfloat16).to(gl.float32)
        )
        angle = ROPE + pos * 64 + channel % 32
        cosine = gl.load(angle, channel < 64, 1.0)
        sine = gl.load(angle + 32, channel < 64, 0.0)
        y = gl.where(
            channel < 64, y * cosine + gl.where(channel < 32, -alternate, alternate) * sine, y
        )
    else:
        y = gl.load(X + row * 14336 + 13312 + (head[:, None, None] - 28) * 256 + channel).to(
            gl.float32
        )
    gl.store(QKV + row * 8192 + head[:, None, None] * 256 + channel, y)


def packed_qkv(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    rope: torch.Tensor,
    positions: torch.Tensor,
    tiled: bool,
) -> torch.Tensor:
    """Pack normalized Q, K and unchanged V without separate staging buffers."""
    rows = x.shape[0]
    qkv = torch.empty((rows, 8192), device=x.device, dtype=x.dtype)
    if tiled and rows >= 128:
        _pack_qkv_tiled[(8, rows)](x, wq, wk, rope, positions, qkv, num_warps=4)
    else:
        _pack_qkv[(rows, 32)](x, wq, wk, rope, positions, qkv, num_warps=4)
    return qkv


ROUND_ATTN_GATE = False


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
    Q_STRIDE: tl.constexpr,
    K_STRIDE: tl.constexpr,
    V_STRIDE: tl.constexpr,
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
    q = tl.load(Q + (start + i[:, None]) * Q_STRIDE + h * 256 + d[None, :], i[:, None] < size, 0)
    base = (h // 6) * 256
    accum = tl.full((BM, VALUE_TILE), 0.0, tl.float32)
    prev = tl.full((BM,), -1.0e20, tl.float32)
    den = tl.full((BM,), 0.0, tl.float32)
    for look in range(tl.cdiv(tl.minimum((order + 1) * BM, size), BN)):
        t = look * BN + j
        kk = tl.load(K + (start + t[None, :]) * K_STRIDE + base + d[:, None], t[None, :] < size, 0)
        vv = tl.load(
            V + (start + t[:, None]) * V_STRIDE + base + value_cols[None, :], t[:, None] < size, 0
        )
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
        q.stride(0),
        k.stride(0),
        v.stride(0),
        64,
        tile,
        fp32_p,
        value_tile,
        num_warps=8,
        num_stages=1 if fp32_p else 3,
    )
    return y

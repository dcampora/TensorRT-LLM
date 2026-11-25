# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused FP8 producers adapted from aegis-engines activation_quant.py (114cf57c911e)."""

import torch
import triton as tr
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.extra.cuda import libdevice


def buffers(x, rows, width, packed_scales=False):
    padded = tr.cdiv(rows, 4) * 4
    activations = torch.empty((padded, width), device=x.device, dtype=torch.float8_e4m3fn)
    if packed_scales:
        scales = torch.empty((width // 512, padded), device=x.device, dtype=torch.int32)
    else:
        scales = torch.empty((width // 128, padded), device=x.device, dtype=torch.float32)
    return activations, scales


@tr.jit(do_not_specialize=["M", "PADDED"])
def _swiglu(G, A, S, M, PADDED, K: tl.constexpr, ROWS: tl.constexpr, PACKED: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    group = tl.program_id(1)
    k = group * 128 + tl.arange(0, 128)
    pos = row[:, None] * (2 * K) + k[None, :]
    gate = tl.load(G + pos, row[:, None] < M, 0.0).to(tl.float32)
    up = tl.load(G + pos + K, row[:, None] < M, 0.0).to(tl.float32)
    activated = gate / (1.0 + libdevice.fast_expf(-gate))
    y = (activated * up).to(tl.bfloat16).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(y), 1), 1.0e-10)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    inv = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    quantized = tl.clamp(y * inv[:, None], -448.0, 448.0)
    tl.store(A + row[:, None] * K + k[None, :], quantized, row[:, None] < PADDED)
    if PACKED:
        offset = (group // 4 * PADDED + row) * 4 + group % 4
        tl.store(S + offset, exponent.to(tl.uint8), row < PADDED)
    else:
        tl.store(S + group * PADDED + row, scale, row < PADDED)


def swiglu_quantize(x, packed_scales=False):
    m, width = x.shape
    k = width // 2
    a, s = buffers(x, m, k, packed_scales)
    padded = a.shape[0]
    tile_rows = 16 if m >= 1024 else 4
    _swiglu[(tr.cdiv(padded, tile_rows), k // 128)](
        x,
        a,
        s.view(torch.uint8) if packed_scales else s,
        m,
        padded,
        k,
        tile_rows,
        packed_scales,
        num_warps=4,
    )
    return a, s


@gluon.jit(do_not_specialize=["M", "PADDED"])
def _gdn_norm(
    X,
    QKVZ,
    W,
    A,
    S,
    M,
    PADDED,
    XS: gl.constexpr,
    H: gl.constexpr,
    PAD: gl.constexpr,
    PACKED: gl.constexpr,
):
    block = gl.program_id(0)
    row, head = block // H, block % H
    # This mapping gives RMSNorm the same FP32 reduction order as the standalone kernel.
    c = gl.arange(0, 128, layout=gl.BlockedLayout([1], [32], [4], [0]))
    x = gl.load(X + block * 128 + c).to(gl.float32)
    z = gl.load(QKVZ + row * XS + 10240 + head * 128 + c).to(gl.float32)
    w = gl.load(W + c).to(gl.float32)
    inv_rms = gl.rsqrt(gl.sum(x * x, 0) * (1.0 / 128.0) + 1.0e-6)
    y = (x * inv_rms * w * (z / (1.0 + gl.exp(-z)))).to(gl.bfloat16).to(gl.float32)
    amax = gl.maximum(gl.max(gl.abs(y), 0), 1.0e-10)
    bits = (amax * (1.0 / 448.0)).to(gl.int32, bitcast=True)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(gl.int32)
    scale = (exponent << 23).to(gl.float32, bitcast=True)
    inv = ((254 - exponent) << 23).to(gl.float32, bitcast=True)
    quantized = gl.clamp(y * inv, -448.0, 448.0)
    gl.store(A + block * 128 + c, quantized)
    if PACKED:
        scale_base = S + (head // 4 * PADDED + row) * 4 + head % 4
        gl.store(scale_base, exponent.to(gl.uint8))
    else:
        gl.store(S + head * PADDED + row, scale)
    if PAD:
        if row == M - 1:
            for j in gl.static_range(1, PAD + 1):
                gl.store(A + (block + H * j) * 128 + c, 0.0)
                if PACKED:
                    gl.store(scale_base + 4 * j, 85)
                else:
                    gl.store(S + head * PADDED + row + j, 2.2737367544323206e-13)


@gluon.jit(do_not_specialize=["M", "PADDED"])
def _gdn_norm_heads(
    X, P, W, A, S, M, PADDED, XS: gl.constexpr, HEADS: gl.constexpr, PACKED: gl.constexpr
):
    # Four warps handle four independent heads. Within each warp, reduce the
    # four consecutive channel groups separately before summing them.
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 1, 32], [4, 1, 1], [2, 0, 1])
    work_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, layout))
    group_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(2, layout))
    lane_layout: gl.constexpr = gl.SliceLayout(0, gl.SliceLayout(1, layout))
    work = gl.program_id(0) * 4 + gl.arange(0, 4, layout=work_layout)
    row, head = work // HEADS, work % HEADS
    group = gl.arange(0, 4, layout=group_layout)
    lane = gl.arange(0, 32, layout=lane_layout)
    channel = group[None, :, None] * 32 + lane[None, None, :]
    offset = work[:, None, None] * 128 + channel
    valid = row[:, None, None] < M
    y = gl.load(X + offset, valid, 0.0).to(gl.float32)
    gate_offset = row[:, None, None] * XS + 10240 + head[:, None, None] * 128 + channel
    z = gl.load(P + gate_offset, valid, 0.0).to(gl.float32)
    w = gl.load(W + channel).to(gl.float32)
    partials = gl.sum(y * y, axis=2)
    inv_rms = gl.rsqrt(gl.sum(partials, axis=1) * (1.0 / 128.0) + 1.0e-6)
    y = (y * inv_rms[:, None, None] * w * (z / (1.0 + gl.exp(-z)))).to(gl.bfloat16).to(gl.float32)
    amax = gl.maximum(gl.max(gl.max(gl.abs(y), 2), 1), 1.0e-10)
    bits = (amax * (1.0 / 448.0)).to(gl.int32, bitcast=True)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(gl.int32)
    scale = (exponent << 23).to(gl.float32, bitcast=True)
    inverse = ((254 - exponent) << 23).to(gl.float32, bitcast=True)
    quantized = gl.clamp(y * inverse[:, None, None], -448.0, 448.0)
    gl.store(A + offset, gl.where(valid, quantized, 0.0))
    if PACKED:
        scale_base = S + (head // 4 * PADDED + row) * 4 + head % 4
        gl.store(scale_base, exponent.to(gl.uint8))
    else:
        gl.store(S + head * PADDED + row, scale)


def gdn_norm_quantize(x, p, weight, packed_scales=False):
    m, heads, width = x.shape
    a, s = buffers(x, m, heads * width, packed_scales)
    output_scales = s.view(torch.uint8) if packed_scales else s
    padded = a.shape[0]
    if m >= 32:
        _gdn_norm_heads[(padded * heads // 4,)](
            x,
            p,
            weight,
            a,
            output_scales,
            m,
            padded,
            p.shape[1],
            heads,
            packed_scales,
            num_warps=4,
        )
    else:
        _gdn_norm[(m * heads,)](
            x,
            p,
            weight,
            a,
            output_scales,
            m,
            padded,
            p.shape[1],
            heads,
            padded - m,
            packed_scales,
            num_warps=4,
        )
    return a, s


@tr.jit(do_not_specialize=["M", "PADDED"])
def _prefill_attention_quant(
    Y,
    X,
    TAIL,
    QUERY_TAIL,
    A,
    S,
    M,
    PADDED,
    ROUND: tl.constexpr,
    ROWS: tl.constexpr,
    SELECT: tl.constexpr,
    SELECT_QUERY: tl.constexpr,
    PACKED: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    group = tl.program_id(1)
    lane = tl.arange(0, 128)
    layout = row[:, None] * 6144 + group * 128 + lane[None, :]
    input_row = tl.load(TAIL + row, row < M, 0) if SELECT else row
    query_row = tl.load(QUERY_TAIL + row, row < M, 0) if SELECT_QUERY else input_row
    input_layout = query_row[:, None] * 6144 + group * 128 + lane[None, :]
    gate_offset = (group // 2) * 512 + (group % 2) * 128 + 256 + lane[None, :]
    gate_layout = input_row[:, None] * 14336 + gate_offset
    y = tl.load(Y + input_layout, row[:, None] < M, 0.0).to(tl.float32)
    gate = tl.load(X + gate_layout, row[:, None] < M, 0.0).to(tl.float32)
    sig = tl.sigmoid(gate)
    if ROUND:
        sig = sig.to(tl.bfloat16).to(tl.float32)
    y = (y * sig).to(tl.bfloat16).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(y), 1), 1.0e-10)
    bits = (maximum * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    inv = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    tl.store(A + layout, tl.clamp(y * inv[:, None], -448.0, 448.0), row[:, None] < PADDED)
    if PACKED:
        offset = (group // 4 * PADDED + row) * 4 + group % 4
        tl.store(S + offset, exponent.to(tl.uint8), row < PADDED)
    else:
        tl.store(S + group * PADDED + row, scale, row < PADDED)


def prefill_attention_quantize(
    attention,
    projection,
    round_gate,
    tile_rows=None,
    tail=None,
    query_tail=None,
    packed_scales=False,
):
    m = attention.shape[0] if tail is None else tail.numel()
    if tile_rows is None:
        tile_rows = 16 if m >= 1024 else 8
    activations, scales = buffers(attention, m, 6144, packed_scales)
    padded = activations.shape[0]
    _prefill_attention_quant[(tr.cdiv(padded, tile_rows), 48)](
        attention,
        projection,
        tail,
        query_tail,
        activations,
        scales.view(torch.uint8) if packed_scales else scales,
        m,
        padded,
        round_gate,
        tile_rows,
        tail is not None,
        query_tail is not None,
        packed_scales,
        num_warps=4,
    )
    return activations, scales

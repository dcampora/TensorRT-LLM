# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from aegis-engines qwen3.8-27b-fp8-batched, kernels.py (114cf57c911e)."""

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

MATCH_ROUNDING = True
ROUND_GDN_QK = False


@tr.jit
def delta_step(
    X,
    AB,
    row,
    h,
    v,
    k,
    dt,
    al,
    state,
    BETA_BF16: tl.constexpr,
    NORM_QK: tl.constexpr,
    MATCH: tl.constexpr,
    ROUND_QK: tl.constexpr,
):
    qp = X + row * 10240 + (h // 3) * 128 + k
    q = tl.load(qp).to(tl.float32)
    key = tl.load(qp + 2048).to(tl.float32)
    val = tl.load(X + row * 10240 + 4096 + h * 128 + v).to(tl.float32)
    b = tl.load(AB + row * 96 + h).to(tl.float32)
    a = tl.load(AB + row * 96 + h + 48).to(tl.float32) + dt
    g = tl.exp(al * tl.where(a <= 20.0, tl.log(1.0 + tl.exp(a)), a))
    beta = 1.0 / (1.0 + tl.exp(-b))
    if BETA_BF16:
        beta = beta.to(tl.bfloat16).to(tl.float32)
    if NORM_QK:
        if MATCH:
            q = q / tl.sqrt(tl.sum(q * q, 0) + 1.0e-6)
            key = key / tl.sqrt(tl.sum(key * key, 0) + 1.0e-6)
        else:
            q = q * tl.rsqrt(tl.sum(q * q, 0) + 1.0e-6)
            key = key * tl.rsqrt(tl.sum(key * key, 0) + 1.0e-6)
    if ROUND_QK:
        q = q.to(tl.bfloat16).to(tl.float32)
        key = key.to(tl.bfloat16).to(tl.float32)
    q = q * 0.08838834764831845
    state = state * g
    delta = (val - tl.sum(state * key[None, :], 1)) * beta
    state = tl.fma(delta[:, None], key[None, :], state)
    out = tl.sum(state * q[None, :], 1)
    return state, out, key, delta, g


@tr.jit
def _recur(
    X,
    AB,
    AL,
    DT,
    S,
    Y,
    SLOTS,
    STATE_STRIDE: tl.constexpr,
    BV: tl.constexpr,
    NORM_QK: tl.constexpr,
    BETA_BF16: tl.constexpr,
    MATCH: tl.constexpr,
    ROUND_QK: tl.constexpr,
):
    seq = tl.program_id(0)
    h = tl.program_id(1)
    v = tl.program_id(2) * BV + tl.arange(0, BV)
    k = tl.arange(0, 128)
    slot = tl.load(SLOTS + seq).to(tl.int64)
    state_p = S + slot * STATE_STRIDE + (h * 128 + v[:, None]) * 128 + k[None, :]
    state = tl.load(state_p, slot >= 0, 0).to(tl.float32)
    dt = tl.load(DT + h).to(tl.float32)
    al = -tl.exp(tl.load(AL + h).to(tl.float32))
    state, out, _, _, _ = delta_step(
        X,
        AB,
        seq,
        h,
        v,
        k,
        dt,
        al,
        state,
        BETA_BF16,
        NORM_QK,
        MATCH,
        ROUND_QK,
    )
    tl.store(Y + (seq * 48 + h) * 128 + v, out)
    tl.store(state_p, state, slot >= 0)


def recurrence(
    x: torch.Tensor,
    ab: torch.Tensor,
    al: torch.Tensor,
    dt: torch.Tensor,
    state: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """Update the active decode slots in their FP32 value-major state pool."""
    rows = x.shape[0]
    y = torch.empty((rows, 48, 128), dtype=x.dtype, device=x.device)
    value_tile = 32
    _recur[(rows, 48, 128 // value_tile)](
        x,
        ab,
        al,
        dt,
        state,
        y,
        slots,
        state.stride(0),
        value_tile,
        True,
        True,
        MATCH_ROUNDING,
        ROUND_GDN_QK,
        num_warps=1,
    )
    return y


@tr.jit
def _gdn_gates(AB, A, DT, G, B, D: tl.constexpr, M: tl.constexpr, BETA_BF16: tl.constexpr):
    i = tl.program_id(0) * D + tl.arange(0, D)
    row, h = i // 48, i % 48
    v = tl.load(AB + row * 96 + h + 48, row < M, 0).to(tl.float32)
    b = tl.load(AB + row * 96 + h, row < M, 0).to(tl.float32)
    d = tl.load(DT + h).to(tl.float32)
    a = tl.load(A + h).to(tl.float32)
    x = v + d
    g = -tl.exp(a) * tl.where(x <= 20.0, tl.log(1.0 + tl.exp(x)), x)
    beta = 1.0 / (1.0 + tl.exp(-b))
    if BETA_BF16:
        beta = beta.to(tl.bfloat16).to(tl.float32)
    tl.store(G + i, g, row < M)
    tl.store(B + i, beta, row < M)


def gdn_gates(ab, a, dt, beta_bf16=False):
    m = ab.shape[0]
    g = torch.empty((m, 48), dtype=torch.float32, device=ab.device)
    b = torch.empty_like(g)
    _gdn_gates[(tr.cdiv(m * 48, 256),)](ab, a, dt, g, b, 256, m, beta_bf16)
    return g, b


@tr.jit
def _gdn_norm(X, QKVZ, W, Y, XS: tl.constexpr, H: tl.constexpr):
    row = tl.program_id(0)
    ch = tl.arange(0, 128)
    x = tl.load(X + row * 128 + ch).to(tl.float32)
    z = tl.load(QKVZ + (row // H) * XS + 10240 + (row % H) * 128 + ch).to(tl.float32)
    w = tl.load(W + ch).to(tl.float32)
    y = x * tl.rsqrt(tl.sum(x * x, 0) * (1.0 / 128.0) + 1.0e-6) * w
    tl.store(Y + row * 128 + ch, y * (z / (1.0 + tl.exp(-z))))


def gdn_norm(x, qkvz, w):
    m, h, d = x.shape
    y = torch.empty((m, h * d), dtype=x.dtype, device=x.device)
    _gdn_norm[(m * h,)](x, qkvz, w, y, qkvz.shape[1], h, num_warps=4)
    return y


@tr.jit
def _cache_conv(
    X,
    W,
    CONV,
    Y,
    SLOTS,
    START,
    ROW_TO_SEQ,
    HAS_INITIAL,
    N: tl.constexpr,
    XS: tl.constexpr,
    STRIDE_SLOT: tl.constexpr,
    DECODE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = channel < N
    seq = row if DECODE else tl.load(ROW_TO_SEQ + row)
    slot = tl.load(SLOTS + seq).to(tl.int64)
    carry = CONV + slot * STRIDE_SLOT + channel * 3
    if not DECODE:
        start = tl.load(START + seq)
        has_initial = tl.load(HAS_INITIAL + seq)
    current = tl.load(X + row * XS + channel, valid, 0).to(tl.float32)
    accum = tl.full((BLOCK,), 0, tl.float32)
    for tap in tl.static_range(4):
        if tap == 3:
            x = current
        elif DECODE:
            x = tl.load(carry + tap, valid & (slot >= 0), 0).to(tl.float32)
            if tap == 2:
                next_x = current
            else:
                next_x = tl.load(carry + tap + 1, valid & (slot >= 0), 0)
            tl.store(carry + tap, next_x, valid & (slot >= 0))
        else:
            source = row + tap - 3
            x = tl.load(X + source * XS + channel, valid & (source >= start), 0).to(tl.float32)
            x += tl.load(
                carry + source - start + 3,
                valid & (source < start) & (slot >= 0) & has_initial,
                0,
            ).to(tl.float32)
        w = tl.load(W + 4 * channel + tap, valid, 0).to(tl.float32)
        accum = tl.fma(x, w, accum)
    activated = tl.div_rn(accum, 1 + libdevice.exp(-accum))
    tl.store(Y + row * N + channel, activated, valid)


@tr.jit
def _commit_conv(
    X,
    CONV,
    SLOTS,
    START,
    HAS_INITIAL,
    N: tl.constexpr,
    XS: tl.constexpr,
    STRIDE_SLOT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    channel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + seq).to(tl.int64)
    mask = (channel < N) & (slot >= 0)
    start = tl.load(START + seq)
    end = tl.load(START + seq + 1)
    has_initial = tl.load(HAS_INITIAL + seq)
    carry = CONV + slot * STRIDE_SLOT + channel * 3
    for tap in tl.static_range(3):
        token = end - 3 + tap
        trailing = tl.load(X + token * XS + channel, mask & (token >= start), 0)
        existing = tl.load(carry + (token - start + 3), mask & (token < start) & has_initial, 0)
        tl.store(carry + tap, tl.where(token >= start, trailing, existing), mask)


def convolution(
    projection: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    slots: torch.Tensor,
    offsets: torch.Tensor | None = None,
    row_to_seq: torch.Tensor | None = None,
    has_initial: torch.Tensor | None = None,
) -> torch.Tensor:
    """Causal SiLU convolution with manager-owned [slot, channel, 3] history."""
    rows = projection.shape[0]
    width = weight.shape[0]
    output = projection.new_empty((rows, width))
    block = 128
    grid = tr.cdiv(width, block)
    _cache_conv[(rows, grid)](
        projection,
        weight,
        state,
        output,
        slots,
        offsets,
        row_to_seq,
        has_initial,
        width,
        projection.stride(0),
        state.stride(0),
        offsets is None,
        block,
    )
    if offsets is not None:
        _commit_conv[(slots.numel(), grid)](
            projection,
            state,
            slots,
            offsets,
            has_initial,
            width,
            projection.stride(0),
            state.stride(0),
            block,
        )
    return output


@tr.jit(do_not_specialize=["ROWS"])
def _prefill_conv_tiled(
    X,
    W,
    CONV,
    SLOTS,
    START,
    ROW_TO_SEQ,
    HAS_INITIAL,
    Q,
    K,
    V,
    ROWS,
    XS: tl.constexpr,
    SLOT_STRIDE: tl.constexpr,
    TOKENS: tl.constexpr,
    CHANNELS: tl.constexpr,
):
    row = tl.program_id(0) * TOKENS + tl.arange(0, TOKENS)
    first = tl.program_id(1) * CHANNELS
    col = first + tl.arange(0, CHANNELS)
    valid = row < ROWS
    seq = tl.load(ROW_TO_SEQ + row, valid, 0)
    slot = tl.load(SLOTS + seq, valid, -1).to(tl.int64)
    begin = tl.load(START + seq, valid, 0)
    has_initial = tl.load(HAS_INITIAL + seq, valid, 0)
    history = CONV + slot[:, None] * SLOT_STRIDE + col[None, :] * 3
    accum = tl.full((TOKENS, CHANNELS), 0, tl.float32)
    for tap in tl.static_range(4):
        source = row + tap - 3
        within = source >= begin
        value = tl.load(X + source[:, None] * XS + col[None, :], (valid & within)[:, None], 0).to(
            tl.float32
        )
        if tap < 3:
            carry = tl.load(
                history + (source - begin + 3)[:, None],
                (valid & ~within & (slot >= 0) & has_initial)[:, None],
                0,
            ).to(tl.float32)
            value = value + carry
        weight = tl.load(W + col * 4 + tap).to(tl.float32)
        accum = tl.fma(value, weight[None, :], accum)
    out = tl.div_rn(accum, 1 + libdevice.exp(-accum))
    if first < 2048:
        tl.store(Q + row[:, None] * 2048 + col[None, :], out, valid[:, None])
    elif first < 4096:
        tl.store(K + row[:, None] * 2048 + col[None, :] - 2048, out, valid[:, None])
    else:
        tl.store(V + row[:, None] * 6144 + col[None, :] - 4096, out, valid[:, None])


def prefill_convolution_qkv(
    projection: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    slots: torch.Tensor,
    offsets: torch.Tensor,
    row_to_seq: torch.Tensor,
    has_initial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a temporal convolution tile and write separate Q, K and V heads.

    Adapted from aegis-engines kernels.py (114cf57c911e). Publish history
    separately so a tile reading the initial window always sees the old state.
    """
    rows = projection.shape[0]
    q, k = (projection.new_empty((rows, 16, 128)) for _ in range(2))
    v = projection.new_empty((rows, 48, 128))
    _prefill_conv_tiled[(tr.cdiv(rows, 16), 80)](
        projection,
        weight,
        state,
        slots,
        offsets,
        row_to_seq,
        has_initial,
        q,
        k,
        v,
        rows,
        projection.stride(0),
        state.stride(0),
        16,
        128,
        num_warps=4,
    )
    _commit_conv[(slots.numel(), 80)](
        projection,
        state,
        slots,
        offsets,
        has_initial,
        10240,
        projection.stride(0),
        state.stride(0),
        128,
    )
    return q, k, v

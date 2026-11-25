# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared GDN inputs adapted from aegis-engines verify_conv_rolling.py and gdn_prepared_gluon.py (114cf57c911e)."""

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _verify_conv_inputs(
    X,
    W,
    CONV,
    QK,
    VALUE,
    GATES,
    BA,
    A_LOG,
    DT,
    WINDOWS,
    SLOTS,
    STEP_SLOTS,
    WIDTH: tl.constexpr,
    XS: tl.constexpr,
    CS: tl.constexpr,
    STEP_STRIDE: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    TOKENS: tl.constexpr,
    TILE: tl.constexpr,
):
    seq = tl.program_id(0)
    group = tl.program_id(1)
    dim = tl.arange(0, TILE)
    channel = group * TILE + dim
    slot = tl.load(SLOTS + seq).to(tl.int64)
    step_slot = tl.load(STEP_SLOTS + seq)
    valid = channel < WIDTH
    index = CONV + slot * CS + channel * 3
    old0 = tl.load(index, valid & (slot >= 0), 0).to(tl.float32)
    old1 = tl.load(index + 1, valid & (slot >= 0), 0).to(tl.float32)
    old2 = tl.load(index + 2, valid & (slot >= 0), 0).to(tl.float32)
    w0 = tl.load(W + channel * 4, valid, 0).to(tl.float32)
    w1 = tl.load(W + channel * 4 + 1, valid, 0).to(tl.float32)
    w2 = tl.load(W + channel * 4 + 2, valid, 0).to(tl.float32)
    w3 = tl.load(W + channel * 4 + 3, valid, 0).to(tl.float32)
    writable = valid & (step_slot >= 0) & (slot >= 0)
    for token in tl.static_range(TOKENS):
        row = seq * TOKENS + token
        x = tl.load(X + row * XS + channel, valid, 0).to(tl.float32)
        value = tl.fma(old1, w1, old0 * w0)
        value = tl.fma(old2, w2, value)
        value = tl.fma(x, w3, value)
        y = tl.div_rn(value, 1 + libdevice.exp(-value))
        y = y.to(tl.bfloat16)
        if group < 32:
            unit = y.to(tl.float32)
            unit = unit / tl.sqrt(tl.sum(unit * unit, 0) + 1e-6)
            unit *= tl.where(group < 16, 0.08838834764831845, 1.0)
            tl.store(QK + row * 4096 + channel, unit)
        else:
            head = group - 32
            tl.store(VALUE + row * 6144 + head * 128 + dim, y)
            a = tl.load(BA + row * 96 + 48 + head).to(tl.float32) + tl.load(DT + head).to(
                tl.float32
            )
            b = tl.load(BA + row * 96 + head).to(tl.float32)
            al = -tl.exp(tl.load(A_LOG + head).to(tl.float32))
            decay = tl.exp(al * tl.where(a <= 20.0, tl.log(1.0 + tl.exp(a)), a))
            beta = (1.0 / (1.0 + tl.exp(-b))).to(tl.bfloat16).to(tl.float32)
            tl.store(GATES + row * 96 + head, beta)
            tl.store(GATES + row * 96 + 48 + head, decay)

        window = WINDOWS + step_slot.to(tl.int64) * STEP_STRIDE + token * TOKEN_STRIDE
        tl.store(window + channel * 3, old1, writable)
        tl.store(window + channel * 3 + 1, old2, writable)
        tl.store(window + channel * 3 + 2, x, writable)
        old0, old1, old2 = old1, old2, x


@tr.jit
def _anchor_prepared(
    QK,
    VALUE,
    GATES,
    STATE_SLOTS,
    REPLAY_SLOTS,
    PREVIOUS,
    BANK,
    STATE,
    DELTAS,
    KEYS,
    DECAYS,
    Y,
    STATE_STRIDE: tl.constexpr,
    DELTA_STRIDE: tl.constexpr,
    KEY_STRIDE: tl.constexpr,
    DECAY_STRIDE: tl.constexpr,
    META_STRIDE: tl.constexpr,
    RANKS: tl.constexpr,
    DECAY_RANKS: tl.constexpr,
    TOKENS: tl.constexpr,
    BV: tl.constexpr,
):
    v_tile = tl.program_id(0)
    h = tl.program_id(1)
    seq = tl.program_id(2)
    slot = tl.load(STATE_SLOTS + seq).to(tl.int64)
    log_slot = tl.load(REPLAY_SLOTS + seq).to(tl.int64)
    valid = (slot >= 0) & (log_slot >= 0)
    v = v_tile * BV + tl.arange(0, BV)
    k = tl.arange(0, 128)
    g = h // 3
    state_ptr = STATE + slot * STATE_STRIDE + (h * 128 + v[:, None]) * 128 + k[None, :]
    state = tl.load(state_ptr, valid, 0).to(tl.float32)
    previous = tl.load(PREVIOUS + log_slot * META_STRIDE, valid, 0)
    bank = tl.load(BANK + log_slot * META_STRIDE, valid, 0)
    key_ptr = KEYS + log_slot * KEY_STRIDE + g * 128 + k
    delta_ptr = DELTAS + log_slot * DELTA_STRIDE + h * 128 + v
    decay_ptr = DECAYS + log_slot * DECAY_STRIDE + h * DECAY_RANKS

    for i in tl.range(0, previous - 1, loop_unroll_factor=1):
        rank = bank * RANKS + i
        key = tl.load(key_ptr + rank * 2048).to(tl.float32)
        delta = tl.load(delta_ptr + rank * 6144).to(tl.float32)
        decay = tl.load(decay_ptr + bank * 48 * DECAY_RANKS + i)
        state = tl.fma(delta[:, None], key[None, :], state * decay)

    row_start = seq * TOKENS
    for token in tl.static_range(TOKENS):
        row = row_start + token
        query_offset = row * 4096 + g * 128 + k
        q = tl.load(QK + query_offset)
        key = tl.load(QK + query_offset + 2048)
        value = tl.load(VALUE + row * 6144 + h * 128 + v).to(tl.float32)
        beta = tl.load(GATES + row * 96 + h)
        decay = tl.load(GATES + row * 96 + 48 + h)

        state *= decay
        delta = (value - tl.sum(state * key[None, :], 1)) * beta
        state = tl.fma(delta[:, None], key[None, :], state)
        tl.store(Y + (row * 48 + h) * 128 + v, tl.sum(state * q[None, :], 1))
        if token == 0:
            # The first verifier row represents an already accepted token.
            tl.store(state_ptr, state, valid, cache_modifier=".cs")
        else:
            next_bank = tl.where(previous > 0, 1 - bank, bank)
            rank = next_bank * RANKS + token - 1
            tl.store(delta_ptr + rank * 6144, delta, valid)
            if v_tile == 0:
                if h % 3 == 0:
                    tl.store(key_ptr + rank * 2048, key, valid)
                tl.store(decay_ptr + next_bank * 48 * DECAY_RANKS + token - 1, decay, valid)


def prepare_convolution(proj, weight, conv, windows, slots, step_slots, ba, al, dt, tokens):
    rows = proj.shape[0]
    query_key = torch.empty((rows, 4096), device=proj.device, dtype=torch.float32)
    values = torch.empty((rows, 6144), device=proj.device, dtype=torch.bfloat16)
    gates = torch.empty((rows, 96), device=proj.device, dtype=torch.float32)
    _verify_conv_inputs[(slots.numel(), 80)](
        proj,
        weight,
        conv,
        query_key,
        values,
        gates,
        ba,
        al,
        dt,
        windows,
        slots,
        step_slots,
        10240,
        proj.stride(0),
        conv.stride(0),
        windows.stride(0),
        windows.stride(1),
        tokens,
        128,
        num_warps=1,
    )
    return query_key, values, gates


def replay_prepared(
    qk, value, gates, state, slots, replay_slots, prev, bank, old_v, old_k, decay, tokens
):
    tile = 16 if slots.numel() >= 96 and tokens == 2 else 8
    rows = value.shape[0]
    y = value.new_empty((rows, 48, 128))
    deltas, keys = old_v.view(torch.float32), old_k.view(torch.float32)
    _anchor_prepared[(128 // tile, 48, slots.numel())](
        qk,
        value,
        gates,
        slots,
        replay_slots,
        prev,
        bank,
        state,
        deltas,
        keys,
        decay,
        y,
        state.stride(0),
        deltas.stride(0),
        keys.stride(0),
        decay.stride(0),
        prev.stride(0),
        old_v.shape[2] // 2,
        decay.shape[-1],
        tokens,
        tile,
        num_warps=1,
    )
    return y

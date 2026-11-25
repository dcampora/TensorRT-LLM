# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rollback-safe Gated DeltaNet verification.

Adapted from aegis-engines verify_conv_rolling.py and verify_replay.py
(114cf57c911e). TensorRT-LLM supplies the persistent pool slots, the physical
tentative-window indices, and the speculative replay lifecycle.
"""

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _verify_conv(
    X,
    W,
    CONV,
    Y,
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
    channel = tl.program_id(1) * TILE + tl.arange(0, TILE)
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
        tl.store(Y + row * WIDTH + channel, y, valid)
        window = WINDOWS + step_slot.to(tl.int64) * STEP_STRIDE + token * TOKEN_STRIDE
        tl.store(window + channel * 3, old1, writable)
        tl.store(window + channel * 3 + 1, old2, writable)
        tl.store(window + channel * 3 + 2, x, writable)
        old0, old1, old2 = old1, old2, x


def verify_convolution(
    projection: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    windows: torch.Tensor,
    slots: torch.Tensor,
    step_slots: torch.Tensor,
    tokens: int,
) -> torch.Tensor:
    """Convolve each draft chain and publish its tentative windows for promotion."""
    width = weight.shape[0]
    output = projection.new_empty((projection.shape[0], width))
    tile = 128
    _verify_conv[(slots.numel(), tr.cdiv(width, tile))](
        projection,
        weight,
        conv_state,
        output,
        windows,
        slots,
        step_slots,
        width,
        projection.stride(0),
        conv_state.stride(0),
        windows.stride(0),
        windows.stride(1),
        tokens,
        tile,
    )
    return output


@tr.jit
def _anchor(
    X,
    AB,
    A_LOG,
    DT,
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
    a_log = -tl.exp(tl.load(A_LOG + h).to(tl.float32))
    dt = tl.load(DT + h).to(tl.float32)
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
        query = tl.load(X + row * 10240 + g * 128 + k).to(tl.float32)
        key = tl.load(X + row * 10240 + 2048 + g * 128 + k).to(tl.float32)
        value = tl.load(X + row * 10240 + 4096 + h * 128 + v).to(tl.float32)
        a = tl.load(AB + row * 96 + 48 + h).to(tl.float32) + dt
        b = tl.load(AB + row * 96 + h).to(tl.float32)
        decay = tl.exp(a_log * tl.where(a <= 20.0, tl.log(1.0 + tl.exp(a)), a))
        beta = (1.0 / (1.0 + tl.exp(-b))).to(tl.bfloat16).to(tl.float32)
        key = key / tl.sqrt(tl.sum(key * key, 0) + 1e-6)
        q = query / tl.sqrt(tl.sum(query * query, 0) + 1e-6) * 0.08838834764831845

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


def replay_delta(
    convolved: torch.Tensor,
    ba: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
    slots: torch.Tensor,
    replay_slots: torch.Tensor,
    previous: torch.Tensor,
    bank: torch.Tensor,
    old_value: torch.Tensor,
    old_key: torch.Tensor,
    decay: torch.Tensor,
    tokens: int,
) -> torch.Tensor:
    """Reuse the two BF16 log banks as half-capacity banks of FP32 deltas."""
    tile = 8
    num_rows = convolved.shape[0]
    y = convolved.new_empty((num_rows, 48, 128))
    deltas, keys = old_value.view(torch.float32), old_key.view(torch.float32)
    _anchor[(128 // tile, 48, slots.numel())](
        convolved,
        ba,
        a_log,
        dt_bias,
        slots,
        replay_slots,
        previous,
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
        previous.stride(0),
        old_value.shape[2] // 2,
        decay.shape[-1],
        tokens,
        tile,
        num_warps=1,
    )
    return y


@tr.jit
def _reset_steps(SLOTS, COUNT, BANK, N: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + row, row < N, -1).to(tl.int64)
    tl.store(COUNT + slot * STRIDE, 0, slot >= 0)
    tl.store(BANK + slot * STRIDE, 0, slot >= 0)


def reset_replay(slots: torch.Tensor, count: torch.Tensor, bank: torch.Tensor) -> None:
    """Discard history after all layers have committed a direct single-token update."""
    rows = slots.numel()
    _reset_steps[(tr.cdiv(rows, 128),)](slots, count, bank, rows, count.stride(0), 128)

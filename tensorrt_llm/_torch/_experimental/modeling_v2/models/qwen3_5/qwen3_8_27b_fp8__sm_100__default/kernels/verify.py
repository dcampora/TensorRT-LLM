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
def _step(state, key, value, a, b, dt, a_log):
    key = key / tl.sqrt(tl.sum(key * key, 0) + 1e-6)
    a = a + dt
    decay = tl.exp(a_log * tl.where(a <= 20.0, tl.log(1.0 + tl.exp(a)), a))
    beta = (1.0 / (1.0 + tl.exp(-b))).to(tl.bfloat16).to(tl.float32)
    state *= decay
    delta = (value - tl.sum(state * key[None, :], 1)) * beta
    return tl.fma(delta[:, None], key[None, :], state)


@tr.jit
def _replay_verify(
    X,
    BA,
    A_LOG,
    DT,
    SLOTS,
    REPLAY_SLOTS,
    PREVIOUS,
    BANK,
    STATE,
    OLD_VALUE,
    OLD_KEY,
    OLD_GATES,
    Y,
    STATE_STRIDE: tl.constexpr,
    V_STRIDE: tl.constexpr,
    K_STRIDE: tl.constexpr,
    G_STRIDE: tl.constexpr,
    META_STRIDE: tl.constexpr,
    HISTORY: tl.constexpr,
    STEP_WIDTH: tl.constexpr,
    TOKENS: tl.constexpr,
    BV: tl.constexpr,
    DIRECT: tl.constexpr,
):
    seq = tl.program_id(0)
    h = tl.program_id(1)
    v_tile = tl.program_id(2)
    slot = tl.load(SLOTS + seq).to(tl.int64)
    history_slot = tl.load(REPLAY_SLOTS + seq).to(tl.int64)
    valid = (slot >= 0) & (history_slot >= 0)
    v = v_tile * BV + tl.arange(0, BV)
    k = tl.arange(0, 128)
    g = h // 3
    old_state = STATE + slot * STATE_STRIDE + (h * 128 + v[:, None]) * 128 + k[None, :]
    state = tl.load(old_state, valid, 0).to(tl.float32)
    previous = tl.load(PREVIOUS + history_slot * META_STRIDE, valid, 0)
    bank = tl.load(BANK + history_slot * META_STRIDE, valid, 0)
    a_log = -tl.exp(tl.load(A_LOG + h).to(tl.float32))
    dt = tl.load(DT + h).to(tl.float32)

    value_base = OLD_VALUE + history_slot * V_STRIDE + h * 128 + v
    key_base = OLD_KEY + history_slot * K_STRIDE + g * 128 + k
    gates_base = OLD_GATES + history_slot * G_STRIDE + h * HISTORY
    for token in range(previous):
        col = bank * HISTORY + token
        key = tl.load(key_base + col * 2048).to(tl.float32)
        value = tl.load(value_base + col * 6144).to(tl.float32)
        gates = tl.load(gates_base + bank * 48 * HISTORY + token)
        a = (gates >> 16).to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        b = gates.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
        state = _step(state, key, value, a, b, dt, a_log)

    checkpoint = previous + STEP_WIDTH > HISTORY
    if not DIRECT and checkpoint:
        tl.store(old_state, state, valid)
    dst_bank = tl.where(checkpoint, 1 - bank, bank)
    dst_start = tl.where(checkpoint, 0, previous)
    for token in tl.static_range(TOKENS):
        row = seq * TOKENS + token
        query = tl.load(X + row * 10240 + g * 128 + k).to(tl.float32)
        key = tl.load(X + row * 10240 + 2048 + g * 128 + k)
        value = tl.load(X + row * 10240 + 4096 + h * 128 + v)
        raw_a = tl.load(BA + row * 96 + 48 + h)
        raw_b = tl.load(BA + row * 96 + h)
        state = _step(
            state,
            key.to(tl.float32),
            value.to(tl.float32),
            raw_a.to(tl.float32),
            raw_b.to(tl.float32),
            dt,
            a_log,
        )
        q = query / tl.sqrt(tl.sum(query * query, 0) + 1e-6) * 0.08838834764831845
        tl.store(Y + (row * 48 + h) * 128 + v, tl.sum(state * q[None, :], 1))
        if not DIRECT:
            col = dst_bank * HISTORY + dst_start + token
            tl.store(value_base + col * 6144, value, valid)
            if v_tile == 0:
                if h % 3 == 0:
                    tl.store(key_base + col * 2048, key, valid)
                packed = (raw_a.to(tl.uint16, bitcast=True).to(tl.uint32) << 16) | raw_b.to(
                    tl.uint16, bitcast=True
                ).to(tl.uint32)
                tl.store(gates_base + dst_bank * 48 * HISTORY + dst_start + token, packed, valid)
    if DIRECT:
        tl.store(old_state, state, valid)


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
    old_gates: torch.Tensor,
    tokens: int,
    max_tokens: int,
    direct: bool = False,
) -> torch.Tensor:
    """Verify a linear draft chain, updating only durable history and checkpoints."""
    y = convolved.new_empty((convolved.shape[0], 48, 128))
    tile = 16
    _replay_verify[(slots.numel(), 48, 128 // tile)](
        convolved,
        ba,
        a_log,
        dt_bias,
        slots,
        replay_slots,
        previous,
        bank,
        state,
        old_value,
        old_key,
        old_gates.view(torch.int32),
        y,
        state.stride(0),
        old_value.stride(0),
        old_key.stride(0),
        old_gates.stride(0),
        previous.stride(0),
        old_value.shape[2],
        max_tokens,
        tokens,
        tile,
        direct,
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

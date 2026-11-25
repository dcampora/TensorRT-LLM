# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeltaNet verification adapted from aegis-engines batch_spec_kernels.py.

The runtime retains the committed convolution window. It advances the replay
slot's active buffer and accepted prefix after verification.
"""

import triton as tr
import triton.language as tl

from .gated_delta import _gdn_gates
from .quantized import _silu


@tr.jit
def convolution_verify(
    X,
    W,
    Slots,
    Conv,
    Windows,
    Y,
    CS: tl.constexpr,
    WS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request, block = tl.program_id(0), tl.program_id(1)
    channels = block * BLOCK + tl.arange(0, BLOCK)
    slot = tl.load(Slots + request).to(tl.int64)
    frame = Conv + slot * CS[0] + channels * CS[1]
    x0 = tl.load(frame).to(tl.float32)
    x1 = tl.load(frame + CS[2]).to(tl.float32)
    x2 = tl.load(frame + 2 * CS[2]).to(tl.float32)
    w0 = tl.load(W + channels * 4).to(tl.float32)
    w1 = tl.load(W + channels * 4 + 1).to(tl.float32)
    w2 = tl.load(W + channels * 4 + 2).to(tl.float32)
    w3 = tl.load(W + channels * 4 + 3).to(tl.float32)
    for token in tl.static_range(WIDTH):
        row = request * WIDTH + token
        x3 = tl.load(X + row * 12288 + channels).to(tl.float32)
        result = ((x0 * w0 + x1 * w1) + x2 * w2) + x3 * w3
        tl.store(Y + row * 8192 + channels, _silu(result))
        window = Windows + request * WS[0] + token * WS[1] + channels * WS[2]
        tl.store(window, x1)
        tl.store(window + WS[3], x2)
        tl.store(window + 2 * WS[3], x3)
        x0, x1, x2 = x1, x2, x3


@tr.jit
def _conv_history(Conv, W, slot, channels, CS: tl.constexpr):
    frame = Conv + slot * CS[0] + channels * CS[1]
    x0 = tl.load(frame).to(tl.float32)
    x1 = tl.load(frame + CS[2]).to(tl.float32)
    x2 = tl.load(frame + 2 * CS[2]).to(tl.float32)
    base = W + channels * 4
    weights = (
        tl.load(base).to(tl.float32),
        tl.load(base + 1).to(tl.float32),
        tl.load(base + 2).to(tl.float32),
        tl.load(base + 3).to(tl.float32),
    )
    return (x0, x1, x2), weights


@tr.jit
def _conv_token(X, window, frame, weights, save, WS: tl.constexpr):
    x0, x1, x2 = frame
    w0, w1, w2, w3 = weights
    x3 = tl.load(X).to(tl.float32)
    value = ((x0 * w0 + x1 * w1) + x2 * w2) + x3 * w3
    if save:
        tl.store(window, x1)
        tl.store(window + WS[3], x2)
        tl.store(window + 2 * WS[3], x3)
    return _silu(value).to(tl.bfloat16).to(tl.float32), (x1, x2, x3)


@tr.jit
def delta_verify(
    QKV,
    BA,
    ALog,
    Bias,
    Slots,
    ReplaySlots,
    State,
    Delta,
    Key,
    Decay,
    Accepted,
    Buffer,
    Y,
    SS: tl.constexpr,
    US: tl.constexpr,
    KS: tl.constexpr,
    GS: tl.constexpr,
    AS: tl.constexpr,
    BS: tl.constexpr,
    HISTORY: tl.constexpr,
    MAX_WIDTH: tl.constexpr,
    WIDTH: tl.constexpr,
    BV: tl.constexpr,
    ConvWeight=None,
    Conv=None,
    ConvWindows=None,
    CS: tl.constexpr = (),
    WS: tl.constexpr = (),
    FUSE_CONV: tl.constexpr = False,
):
    request, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    state_slot = tl.load(Slots + request).to(tl.int64)
    replay_slot = tl.load(ReplaySlots + request).to(tl.int64)
    accepted = tl.load(Accepted + replay_slot * AS)
    buffer = tl.load(Buffer + replay_slot * BS)
    rollover = accepted + MAX_WIDTH > HISTORY
    write_buffer = tl.where(rollover, 1 - buffer, buffer)
    write_offset = tl.where(rollover, 0, accepted)

    k = tl.arange(0, 128)
    v = tile * BV + tl.arange(0, BV)
    state_ptr = State + state_slot * SS + (head * 128 + v[:, None]) * 128 + k[None, :]
    s = tl.load(state_ptr, cache_modifier=".cg")
    history_k = Key + replay_slot * KS[0] + (head // 2) * KS[3] + k * KS[4]
    history_u = Delta + replay_slot * US[0] + head * US[3] + v * US[4]
    history_g = Decay + replay_slot * GS[0] + head * GS[2]

    previous = tl.full((), 0.0, tl.float32)
    for token in range(accepted):
        key = tl.load(history_k + buffer * KS[1] + token * KS[2]).to(tl.float32)
        u = tl.load(history_u + buffer * US[1] + token * US[2]).to(tl.float32)
        cumulative = tl.load(history_g + buffer * GS[1] + token * GS[3])
        s = tl.fma(u[:, None], key[None, :], s * tl.exp(cumulative - previous))
        previous = cumulative

    if rollover:
        tl.store(state_ptr, s, cache_modifier=".cs")
        previous = tl.full((), 0.0, tl.float32)

    query_channel = (head // 2) * 128 + k
    key_channel = query_channel + 2048
    value_channel = 4096 + head * 128 + v
    # As in aegis-engines spec_kernels.py, each state tile computes the Q/K
    # stencil locally. Only one tile writes each persistent convolution channel.
    if FUSE_CONV:
        conv_q, wq = _conv_history(Conv, ConvWeight, state_slot, query_channel, CS)
        conv_k, wk = _conv_history(Conv, ConvWeight, state_slot, key_channel, CS)
        conv_v, wv = _conv_history(Conv, ConvWeight, state_slot, value_channel, CS)
        save_qk = (tile == 0) & (head % 2 == 0)

    for token in tl.static_range(WIDTH):
        row = request * WIDTH + token
        if FUSE_CONV:
            qkv = QKV + row * 12288
            window = ConvWindows + request * WS[0] + token * WS[1]
            q, conv_q = _conv_token(
                qkv + query_channel, window + query_channel * WS[2], conv_q, wq, save_qk, WS
            )
            key, conv_k = _conv_token(
                qkv + key_channel, window + key_channel * WS[2], conv_k, wk, save_qk, WS
            )
            value, conv_v = _conv_token(
                qkv + value_channel, window + value_channel * WS[2], conv_v, wv, True, WS
            )
        else:
            qkv = QKV + row * 8192
            q = tl.load(qkv + query_channel).to(tl.float32)
            key = tl.load(qkv + key_channel).to(tl.float32)
            value = tl.load(qkv + value_channel).to(tl.float32)

        q = (q / tl.sqrt(tl.sum(q * q, 0) + 1.0e-6)) * 0.08838834764831845
        key = key / tl.sqrt(tl.sum(key * key, 0) + 1.0e-6)
        gate, beta = _gdn_gates(BA, ALog, Bias, row, head)
        previous += gate
        s *= tl.exp(gate)
        delta = (value - tl.sum(s * key[None, :], 1)) * beta
        s = tl.fma(delta[:, None], key[None, :], s)
        tl.store(Y + (row * 32 + head) * 128 + v, tl.sum(s * q[None, :], 1))

        index = write_offset + token
        tl.store(history_u + write_buffer * US[1] + index * US[2], delta)
        if tile == 0:
            tl.store(history_g + write_buffer * GS[1] + index * GS[3], previous)
            if head % 2 == 0:
                tl.store(history_k + write_buffer * KS[1] + index * KS[2], key)

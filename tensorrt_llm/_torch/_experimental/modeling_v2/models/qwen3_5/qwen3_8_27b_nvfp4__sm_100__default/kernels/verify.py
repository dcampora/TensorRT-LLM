# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDN verification adapted from aegis-engines cb6779a, spec_kernels.py."""

import triton as tr
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .pointwise import launch_dependents, wait_inputs


@tr.jit
def verify_conv(
    X,
    Weight,
    History,
    Slots,
    Output,
    Intermediate,
    X_STRIDE: tl.constexpr,
    H_SEQ: tl.constexpr,
    H_CH: tl.constexpr,
    H_TIME: tl.constexpr,
    I_SEQ: tl.constexpr,
    I_ROW: tl.constexpr,
    I_CH: tl.constexpr,
    I_TIME: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    PDL: tl.constexpr,
):
    seq, tile = tl.program_id(0), tl.program_id(1)
    channel = tile * BLOCK + tl.arange(0, BLOCK)
    if PDL:
        wait_inputs()
    slot = tl.load(Slots + seq).to(tl.int64)
    state = History + slot * H_SEQ + channel * H_CH
    source = X + seq * WIDTH * X_STRIDE + channel
    candidate = Intermediate + seq * I_SEQ + channel * I_CH
    output = Output + seq * WIDTH * 10240 + channel
    weight = Weight + channel * 4
    w0, w1, w2, w3 = (
        tl.load(weight),
        tl.load(weight + 1),
        tl.load(weight + 2),
        tl.load(weight + 3),
    )
    x0 = tl.load(state)
    x1 = tl.load(state + H_TIME)
    x2 = tl.load(state + 2 * H_TIME)
    for row in tl.static_range(WIDTH):
        x3 = tl.load(source + row * X_STRIDE)
        acc = (x0 * w0).to(tl.float32)
        acc += (x1 * w1).to(tl.float32)
        acc += (x2 * w2).to(tl.float32)
        acc += (x3 * w3).to(tl.float32)
        tl.store(output + row * 10240, acc / (1.0 + tl.exp(-acc)))
        window = candidate + row * I_ROW
        tl.store(window, x1)
        tl.store(window + I_TIME, x2)
        tl.store(window + 2 * I_TIME, x3)
        x0, x1, x2 = x1, x2, x3
    if PDL:
        launch_dependents()


@gluon.jit
def verify_prepare(
    X,
    Weight,
    History,
    Slots,
    QK,
    Value,
    BA,
    A,
    Bias,
    Gates,
    Intermediate,
    X_STRIDE: gl.constexpr,
    H_SEQ: gl.constexpr,
    H_CH: gl.constexpr,
    H_TIME: gl.constexpr,
    I_SEQ: gl.constexpr,
    I_ROW: gl.constexpr,
    I_CH: gl.constexpr,
    I_TIME: gl.constexpr,
    WIDTH: gl.constexpr,
    PARTS: gl.constexpr,
):
    # Adapted from aegis-engines cb6779a, batch_spec_kernels.py.
    group, seq = gl.program_id(0), gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [PARTS, 1], [1, 0])
    part = group * PARTS + gl.arange(0, PARTS, layout=gl.SliceLayout(1, layout))
    lane = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    channel = part[:, None] * 128 + lane[None, :]
    slot = gl.load(Slots + seq).to(gl.int64)
    state = History + slot * H_SEQ + channel * H_CH
    weight = Weight + channel * 4
    w0 = gl.load(weight)
    w1 = gl.load(weight + 1)
    w2 = gl.load(weight + 2)
    w3 = gl.load(weight + 3)
    source = X + seq * WIDTH * X_STRIDE + channel
    dest = Intermediate + seq * I_SEQ + channel * I_CH
    x0 = gl.load(state)
    x1 = gl.load(state + H_TIME)
    x2 = gl.load(state + 2 * H_TIME)
    for step in gl.static_range(WIDTH):
        row = seq * WIDTH + step
        x3 = gl.load(source + step * X_STRIDE)
        acc = (x0 * w0).to(gl.float32)
        acc += (x1 * w1).to(gl.float32)
        acc += (x2 * w2).to(gl.float32)
        acc += (x3 * w3).to(gl.float32)
        convolved = (acc / (1.0 + gl.exp(-acc))).to(gl.bfloat16)
        window = dest + step * I_ROW
        gl.store(window, x1)
        gl.store(window + I_TIME, x2)
        gl.store(window + 2 * I_TIME, x3)
        if group * PARTS < 32:
            value = convolved.to(gl.float32)
            value = value / gl.sqrt(gl.sum(value * value, 1) + 1e-6)[:, None]
            if group * PARTS < 16:
                value *= 128**-0.5
            gl.store(QK + row * 4096 + channel, value)
        else:
            gl.store(Value + row * 6144 + channel - 4096, convolved)
            head = part - 32
            log_a = gl.exp(gl.load(A + head).to(gl.float32))
            bias = gl.load(Bias + head).to(gl.float32)
            a = gl.load(BA + row * 96 + 48 + head).to(gl.float32) + bias
            b = gl.load(BA + row * 96 + head).to(gl.float32)
            decay = gl.exp(-log_a * gl.where(a <= 20.0, gl.log(1 + gl.exp(a)), a))
            beta = (1.0 / (1.0 + gl.exp(-b))).to(gl.bfloat16).to(gl.float32)
            gl.store(Gates + row * 96 + head, decay)
            gl.store(Gates + row * 96 + 48 + head, beta)
        x0, x1, x2 = x1, x2, x3


@tr.jit
def _prepared_candidate(QK, V, Gates, row, head, kc, vc):
    channel = head // 3 * 128 + kc
    q = tl.load(QK + row * 4096 + channel)
    k = tl.load(QK + row * 4096 + 2048 + channel)
    v = tl.load(V + (row * 48 + head) * 128 + vc).to(tl.float32)
    decay = tl.load(Gates + row * 96 + head)
    beta = tl.load(Gates + row * 96 + 48 + head)
    return q, k, v, decay, beta


@tr.jit
def _gdn_candidate(QKV, BA, row, head, kc, vc, log_a, bias):
    channel = head // 3 * 128 + kc
    qkv = QKV + row * 10240
    q = tl.load(qkv + channel).to(tl.float32)
    k = tl.load(qkv + 2048 + channel).to(tl.float32)
    v = tl.load(qkv + 4096 + head * 128 + vc).to(tl.float32)
    q = q / tl.sqrt(tl.sum(q * q) + 1.0e-6)
    k = k / tl.sqrt(tl.sum(k * k) + 1.0e-6)
    a = tl.load(BA + row * 96 + 48 + head).to(tl.float32) + bias
    b = tl.load(BA + row * 96 + head).to(tl.float32)
    g = -log_a * tl.where(a <= 20.0, tl.log(1 + tl.exp(a)), a)
    beta = tl.sigmoid(b).to(tl.bfloat16).to(tl.float32)
    return q * (128**-0.5), k, v, tl.exp(g), beta


@tr.jit
def verify_state(
    QKV,
    BA,
    A,
    Bias,
    State,
    Slots,
    Output,
    Intermediate,
    H_SEQ: tl.constexpr,
    I_SEQ: tl.constexpr,
    I_ROW: tl.constexpr,
    WIDTH: tl.constexpr,
    BV: tl.constexpr,
    PDL: tl.constexpr,
):
    tile, work = tl.program_id(0), tl.program_id(1)
    seq, head = work // 48, work % 48
    kc = tl.arange(0, 128)
    vc = tile * BV + tl.arange(0, BV)
    index = head * 16384 + vc[:, None] * 128 + kc[None, :]
    log_a = tl.exp(tl.load(A + head).to(tl.float32))
    bias = tl.load(Bias + head).to(tl.float32)
    slot = tl.load(Slots + seq).to(tl.int64)
    if PDL:
        wait_inputs()
    state = tl.load(State + slot * H_SEQ + index)
    dest = Intermediate + seq.to(tl.int64) * I_SEQ + index
    for step in tl.static_range(WIDTH):
        row = seq * WIDTH + step
        q, k, v, decay, beta = _gdn_candidate(QKV, BA, row, head, kc, vc, log_a, bias)
        state *= decay
        v = (v - tl.sum(state * k[None, :], 1)) * beta
        state += v[:, None] * k[None, :]
        output = tl.sum(state * q[None, :], 1)
        tl.store(dest + step * I_ROW, state)
        tl.store(Output + (row * 48 + head) * 128 + vc, output)
    if PDL:
        launch_dependents()


@tr.jit
def verify_replay(
    QKV,
    BA,
    A,
    Bias,
    State,
    StateSlots,
    ReplaySlots,
    Accepted,
    Banks,
    Deltas,
    Keys,
    Decays,
    Output,
    STATE_STRIDE: tl.constexpr,
    ACCEPTED_STRIDE: tl.constexpr,
    BANK_STRIDE: tl.constexpr,
    DELTA_SLOT: tl.constexpr,
    KEY_SLOT: tl.constexpr,
    DECAY_SLOT: tl.constexpr,
    WIDTH: tl.constexpr,
    STEP_WIDTH: tl.constexpr,
    HISTORY: tl.constexpr,
    BV: tl.constexpr,
    PDL: tl.constexpr,
    PreparedQK=None,
    PreparedGates=None,
):
    # When every step starts a new bank, the state includes the first
    # guaranteed verifier row and the history starts at its first draft.
    COMMIT_FIRST: tl.constexpr = HISTORY == STEP_WIDTH
    tile, work = tl.program_id(0), tl.program_id(1)
    seq, head = work // 48, work % 48
    kc = tl.arange(0, 128)
    vc = tile * BV + tl.arange(0, BV)
    group = head // 3
    log_a = tl.exp(tl.load(A + head).to(tl.float32))
    bias = tl.load(Bias + head).to(tl.float32)
    if PDL:
        wait_inputs()
    state_slot = tl.load(StateSlots + seq).to(tl.int64)
    replay_slot = tl.load(ReplaySlots + seq).to(tl.int64)
    count = tl.load(Accepted + replay_slot * ACCEPTED_STRIDE)
    bank = tl.load(Banks + replay_slot * BANK_STRIDE)

    dp = Deltas + replay_slot * DELTA_SLOT
    kp = Keys + replay_slot * KEY_SLOT
    gp = Decays + replay_slot * DECAY_SLOT + head * HISTORY
    state_ptr = State + state_slot * STATE_STRIDE
    state_ptr += head * 16384 + vc[:, None] * 128 + kc[None, :]
    state = tl.load(state_ptr, cache_modifier=".cg" if PreparedQK is not None else "")
    for i in range(1 if COMMIT_FIRST else 0, count):
        off = bank * HISTORY + i
        delta = tl.load(dp + (off * 48 + head) * 128 + vc)
        key = tl.load(kp + (off * 16 + group) * 128 + kc)
        decay = tl.load(gp + bank * (48 * HISTORY) + i)
        state = tl.fma(delta[:, None], key[None, :], state * decay)

    if count + STEP_WIDTH > HISTORY:
        if not COMMIT_FIRST:
            tl.store(state_ptr, state)
        count = 0
        bank = 1 - bank
    for i in tl.static_range(WIDTH):
        row = seq * WIDTH + i
        if PreparedQK is None:
            q, key, val, decay, beta = _gdn_candidate(QKV, BA, row, head, kc, vc, log_a, bias)
        else:
            q, key, val, decay, beta = _prepared_candidate(
                PreparedQK, QKV, PreparedGates, row, head, kc, vc
            )
        state *= decay
        delta = (val - tl.sum(state * key[None, :], 1)) * beta
        state = tl.fma(delta[:, None], key[None, :], state)
        result = tl.sum(state * q[None, :], 1)
        tl.store(Output + (row * 48 + head) * 128 + vc, result)
        if COMMIT_FIRST and i == 0:
            tl.store(state_ptr, state, cache_modifier=".cs")
        else:
            off = bank * HISTORY + count + i
            tl.store(dp + (off * 48 + head) * 128 + vc, delta)
            if tile == 0:
                tl.store(gp + bank * (48 * HISTORY) + count + i, decay)
                if head % 3 == 0:
                    tl.store(kp + (off * 16 + group) * 128 + kc, key)
    if PDL:
        launch_dependents()

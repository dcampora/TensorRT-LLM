# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tiled batch verification adapted from aegis-engines verify_gdn.py."""

import triton as tr
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout

from .gated_delta import _gdn_gates


@tr.jit
def prepare(QKV, BA, ALog, Bias, QK, Gates):
    row, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, 128)
    offset = row * 8192 + head * 128 + d
    q = tl.load(QKV + offset).to(tl.float32)
    key = tl.load(QKV + offset + 2048).to(tl.float32)
    q = (q / tl.sqrt(tl.sum(q * q, 0) + 1.0e-6)) * 0.08838834764831845
    key = key / tl.sqrt(tl.sum(key * key, 0) + 1.0e-6)
    out = (row * 16 + head) * 256 + d
    tl.store(QK + out, q)
    tl.store(QK + out + 128, key)
    for group in tl.static_range(2):
        h = head * 2 + group
        g, beta = _gdn_gates(BA, ALog, Bias, row, h)
        index = (row * 32 + h) * 3
        tl.store(Gates + index, g)
        tl.store(Gates + index + 1, tl.exp(g))
        tl.store(Gates + index + 2, beta)


@gluon.jit
def verify(
    QKV,
    QK,
    Gates,
    Slots,
    ReplaySlots,
    State,
    Delta,
    Key,
    Decay,
    Accepted,
    Buffer,
    Y,
    SS: gl.constexpr,
    US: gl.constexpr,
    KS: gl.constexpr,
    GS: gl.constexpr,
    AS: gl.constexpr,
    BS: gl.constexpr,
    HISTORY: gl.constexpr,
    MAX_WIDTH: gl.constexpr,
    WIDTH: gl.constexpr,
    BV: gl.constexpr,
):
    # A full warp owns a value channel, preserving the contraction order
    # of the smaller-batch verifier.
    layout: gl.constexpr = BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    tile, head, request = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    state_slot = gl.load(Slots + request).to(gl.int64)
    replay_slot = gl.load(ReplaySlots + request).to(gl.int64)
    accepted = gl.load(Accepted + replay_slot * AS)
    buffer = gl.load(Buffer + replay_slot * BS)
    rollover = accepted + MAX_WIDTH > HISTORY
    write_buffer = gl.where(rollover, 1 - buffer, buffer)
    write_offset = gl.where(rollover, 0, accepted)

    k = gl.arange(0, 128, layout=SliceLayout(0, layout))
    v = tile * BV + gl.arange(0, BV, layout=SliceLayout(1, layout))
    state_ptr = State + state_slot * SS + (head * 128 + v[:, None]) * 128 + k[None, :]
    s = gl.load(state_ptr, cache_modifier=".cg")
    history_k = Key + replay_slot * KS[0] + (head // 2) * KS[3] + k * KS[4]
    history_u = Delta + replay_slot * US[0] + head * US[3] + v * US[4]
    history_g = Decay + replay_slot * GS[0] + head * GS[2]

    previous = gl.full((), 0.0, gl.float32)
    for token in range(accepted):
        key = gl.load(history_k + buffer * KS[1] + token * KS[2]).to(gl.float32)
        u = gl.load(history_u + buffer * US[1] + token * US[2]).to(gl.float32)
        recorded = gl.load(history_g + buffer * GS[1] + token * GS[3])
        fade = gl.exp(recorded - previous)
        previous = recorded
        s = gl.fma(u[:, None], key[None, :], s * fade)

    if rollover:
        # The manager advances the slot's replay cursor after all layers finish.
        gl.store(state_ptr, s, cache_modifier=".cs")
        previous = gl.full((), 0.0, gl.float32)

    for token in gl.static_range(WIDTH):
        row = request * WIDTH + token
        qk = QK + (row * 16 + head // 2) * 256 + k
        q = gl.load(qk)
        key = gl.load(qk + 128)
        value = gl.load(QKV + row * 8192 + 4096 + head * 128 + v).to(gl.float32)
        gates = Gates + (row * 32 + head) * 3
        decay, beta = gl.load(gates + 1), gl.load(gates + 2)
        previous += gl.load(gates)
        s *= decay
        delta = (value - gl.sum(s * key[None, :], 1)) * beta
        s = gl.fma(delta[:, None], key[None, :], s)
        gl.store(Y + (row * 32 + head) * 128 + v, gl.sum(s * q[None, :], 1))

        index = write_offset + token
        gl.store(history_u + write_buffer * US[1] + index * US[2], delta)
        if tile == 0:
            gl.store(history_g + write_buffer * GS[1] + index * GS[3], previous)
            if head % 2 == 0:
                gl.store(history_k + write_buffer * KS[1] + index * KS[2], key)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tiled Q/K normalization and rotary embedding adapted from aegis-engines prefill_rope.py."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


@gluon.jit(do_not_specialize=["M", "QB", "KB"])
def qkv_rope(
    X,
    QW,
    KW,
    Rope,
    Pos,
    Rows,
    Q,
    KV,
    Gate,
    Table,
    TS: gl.constexpr,
    RS: gl.constexpr,
    PAGE: gl.constexpr,
    PS: gl.constexpr,
    M,
    R: gl.constexpr,
    W: gl.constexpr,
    QB,
    KB,
):
    layout: gl.constexpr = BlockedLayout([1, 2], [1, 32], [W, 1], [1, 0])
    block = gl.program_id(0)
    h = gl.arange(0, R, layout=SliceLayout(1, layout))
    col = gl.arange(0, 64, layout=SliceLayout(0, layout))
    if block < QB + KB:
        key = block >= QB
        group = gl.where(key, block - QB, block)
        i = group * R + h
        heads = gl.where(key, 2, 16)
        token = i // heads
        head = i % heads
        active = token < M
        position = gl.load(Pos + token, active, 0)
        amp = gl.where(key, 8192 + head * 256, head * 512)
        x = X + (token * 9216 + amp)[:, None] + col[None, :]
        w = gl.where(key, KW, QW)
        a = gl.load(x, active[:, None], 0.0).to(gl.float32)
        b = gl.load(x + 64, active[:, None], 0.0).to(gl.float32)
        c = gl.load(x + 128, active[:, None], 0.0).to(gl.float32)
        d = gl.load(x + 192, active[:, None], 0.0).to(gl.float32)
        first = gl.sum(a * a, 1) + gl.sum(c * c, 1)
        second = gl.sum(b * b, 1) + gl.sum(d * d, 1)
        inv = gl.rsqrt((first + second) / 256.0 + 1.0e-6)[:, None]
        if key:
            row = gl.load(Rows + token, active, 0)
            page = gl.load(Table + row * TS + position // PAGE, active, 0).to(gl.int64)
            out = KV + page * PS + (head * PAGE + position % PAGE) * 256
        else:
            out = Q + (token * 16 + head) * 256
        for chunk in gl.static_range(4):
            if chunk == 0:
                values = a
            elif chunk == 1:
                values = b
            elif chunk == 2:
                values = c
            else:
                values = d
            norm = gl.load(w + col + chunk * 64).to(gl.float32)[None, :]
            y = (values * inv * (1.0 + norm)).to(gl.bfloat16).to(gl.float32)
            if chunk == 0:
                pair = col ^ 32
                px = gl.load(
                    X + (token * 9216 + amp)[:, None] + pair[None, :], active[:, None], 0.0
                ).to(gl.float32)
                pw = gl.load(w + pair).to(gl.float32)[None, :]
                py = (px * inv * (1.0 + pw)).to(gl.bfloat16).to(gl.float32)
                rc = gl.load(Rope + position[:, None] * 64 + col[None, :] % 32)
                rs = gl.load(Rope + position[:, None] * 64 + 32 + col[None, :] % 32)
                signed = gl.where(col[None, :] < 32, -rs, rs)
                y = (y * rc + py * signed).to(gl.bfloat16).to(gl.float32)
            gl.store(out[:, None] + col[None, :] + chunk * 64, y, active[:, None])
            if not key:
                raw_gate = gl.load(x + 256 + chunk * 64, active[:, None], 0.0)
                gate_out = (token * 16 + head)[:, None] * 256 + col[None, :]
                gl.store(Gate + gate_out + chunk * 64, raw_gate, active[:, None])
    else:
        i = (block - QB - KB) * R + h
        token = i // 2
        head = i % 2
        active = token < M
        position = gl.load(Pos + token, active, 0)
        row = gl.load(Rows + token, active, 0)
        page = gl.load(Table + row * TS + RS + position // PAGE, active, 0).to(gl.int64)
        v = gl.arange(0, 256, layout=SliceLayout(0, layout))
        value = gl.load(
            X + (token * 9216 + 8704 + head * 256)[:, None] + v[None, :], active[:, None], 0.0
        )
        offset = page * PS + (head * PAGE + position % PAGE) * 256
        gl.store(KV + offset[:, None] + v[None, :], value, active[:, None])

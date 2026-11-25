# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Narrow BF16 gate projection from aegis-engines bf16_ba.py (a6c09c6502)."""

import triton as tr
import triton.language as tl


@tr.jit
def gate_project(X, W, Y):
    n = tl.program_id(0) * 16 + tl.arange(0, 16)
    m = tl.program_id(1)
    row = tl.arange(0, 16)
    k = tl.arange(0, 256)
    first = tl.full((16, 16), 0.0, tl.float32)
    second = tl.full((16, 16), 0.0, tl.float32)
    for step in range(4):
        j = step * 256 + k
        x = tl.load(X + m * 2048 + j[None, :] + row[:, None] * 2048, row[:, None] == 0, 0.0)
        w = tl.load(W + n[:, None] * 2048 + j[None, :])
        first = tl.dot(w, tl.trans(x), first)
        x = tl.load(X + m * 2048 + 1024 + j[None, :] + row[:, None] * 2048, row[:, None] == 0, 0.0)
        w = tl.load(W + n[:, None] * 2048 + 1024 + j[None, :])
        second = tl.dot(w, tl.trans(x), second)
    tl.store(Y + m * 64 + n[:, None] + row[None, :] * 64, first + second, row[None, :] == 0)

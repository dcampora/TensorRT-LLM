# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Triton gates over packed attention and expert rows."""

import triton
import triton.language as tl


@triton.jit
def _sigmoid_gate_kernel(
    X,
    Gate,
    Addend,
    Out,
    width: tl.constexpr,
    x_stride: tl.constexpr,
    gate_stride: tl.constexpr,
    gate_per_row: tl.constexpr,
    add_stride: tl.constexpr,
    has_addend: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * block + tl.arange(0, block)
    x = tl.load(X + row * x_stride + col, col < width, other=0).to(tl.float32)
    if gate_per_row:
        gate = tl.load(Gate + row * gate_stride).to(tl.float32)
    else:
        gate = tl.load(Gate + row * gate_stride + col, col < width, other=0).to(tl.float32)
    y = x / (1.0 + tl.exp(-gate))
    if has_addend:
        y += tl.load(Addend + row * add_stride + col, col < width, other=0).to(tl.float32)
    tl.store(Out + row * width + col, y.to(Out.dtype.element_ty), col < width)


def sigmoid_gate(x, gate, addend, out):
    """Gate a packed matrix and optionally accumulate a second matrix."""
    rows, width = x.shape
    block = 256
    _sigmoid_gate_kernel[(rows, triton.cdiv(width, block))](
        x,
        gate,
        addend,
        out,
        width,
        x.stride(0),
        gate.stride(0),
        gate.shape[1] == 1,
        0 if addend is None else addend.stride(0),
        addend is not None,
        block,
        num_warps=4,
    )

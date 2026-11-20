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


@triton.jit
def _gated_rms_norm_kernel(
    X,
    Z,
    Weight,
    Out,
    heads: tl.constexpr,
    width: tl.constexpr,
    z_stride: tl.constexpr,
    eps: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    x = tl.load(X + row * width + col, col < width, other=0).to(tl.float32)
    z_offset = row // heads * z_stride + row % heads * width
    z = tl.load(Z + z_offset + col, col < width, other=0).to(tl.float32)
    weight = tl.load(Weight + col, col < width, other=0).to(tl.float32)
    x *= tl.rsqrt(tl.sum(x * x, 0) / width + eps)
    out = x * weight * (z / (1.0 + tl.exp(-z)))
    tl.store(Out + row * width + col, out.to(Out.dtype.element_ty), col < width)


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


def gated_rms_norm(x, z, weight, eps, out):
    """Normalize each GDN value head and apply its SiLU gate."""
    rows, heads, width = x.shape
    _gated_rms_norm_kernel[(rows * heads,)](
        x,
        z,
        weight,
        out,
        heads,
        width,
        z.stride(0),
        eps,
        triton.next_power_of_2(width),
        num_warps=1,
    )

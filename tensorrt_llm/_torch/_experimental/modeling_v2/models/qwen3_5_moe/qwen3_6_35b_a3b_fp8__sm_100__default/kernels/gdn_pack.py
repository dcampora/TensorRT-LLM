# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copy the GDN convolution and gate outputs to the scan's packed layout."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_gdn_input(
    Qkv,
    A,
    B,
    Pack,
    Gates,
    rows: tl.constexpr,
    q_stride: tl.constexpr,
    a_stride: tl.constexpr,
    b_stride: tl.constexpr,
    key_width: tl.constexpr,
    value_width: tl.constexpr,
    value_heads: tl.constexpr,
    block: tl.constexpr,
):
    row, slice_id = tl.program_id(0), tl.program_id(1)
    offset = tl.arange(0, block)
    feature = slice_id * block + offset
    key_end: tl.constexpr = 2 * key_width
    width: tl.constexpr = key_end + value_width
    qkv = tl.load(Qkv + row * q_stride + feature, feature < width, 0)
    output = tl.where(
        feature < key_width,
        row * key_width + feature,
        tl.where(
            feature < key_end,
            (rows - 1 + row) * key_width + feature,
            key_end * (rows - 1) + row * value_width + feature,
        ),
    )
    tl.store(Pack + output, qkv, feature < width)
    if slice_id == 0:
        a = tl.load(A + row * a_stride + offset, offset < value_heads, 0)
        b = tl.load(B + row * b_stride + offset, offset < value_heads, 0)
        tl.store(Gates + row * value_heads + offset, a, offset < value_heads)
        tl.store(Gates + (rows + row) * value_heads + offset, b, offset < value_heads)


def pack_gdn_input(qkv, a, b, key_heads, value_heads, head_dim):
    rows = qkv.shape[0]
    key_width = key_heads * head_dim
    value_width = value_heads * head_dim
    shape = (2 * key_width + value_width) * rows
    packed = torch.empty((shape,), dtype=qkv.dtype, device=qkv.device)
    gates = torch.empty((2, rows, value_heads), dtype=qkv.dtype, device=qkv.device)
    block = 512
    _pack_gdn_input[(rows, triton.cdiv(qkv.shape[1], block))](
        qkv,
        a,
        b,
        packed,
        gates,
        rows,
        qkv.stride(0),
        a.stride(0),
        b.stride(0),
        key_width,
        value_width,
        value_heads,
        block,
        num_warps=4,
    )
    q, k, v = packed.split((rows * key_width, rows * key_width, rows * value_width))
    return (
        q.view(rows, key_heads, head_dim),
        k.view(rows, key_heads, head_dim),
        v.view(rows, value_heads, head_dim),
        gates[0],
        gates[1],
    )

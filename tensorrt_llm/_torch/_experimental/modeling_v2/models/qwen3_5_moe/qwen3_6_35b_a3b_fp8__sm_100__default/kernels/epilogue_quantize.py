# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""E4M3 projection inputs produced directly by the gated epilogues."""

import torch
import triton
import triton.language as tl

from .quantize import _to_fp8


@triton.jit(do_not_specialize=["rows"])
def _gated_quantize(
    X,
    Gate,
    Weight,
    PickRows,
    Out,
    Scales,
    rows,
    width: tl.constexpr,
    x_stride: tl.constexpr,
    gate_stride: tl.constexpr,
    eps: tl.constexpr,
    kind: tl.constexpr,
    pick_rows: tl.constexpr,
    groups_per_cta: tl.constexpr,
):
    blocks: tl.constexpr = width // 128
    group = tl.program_id(0) * groups_per_cta + tl.arange(0, groups_per_cta)
    row = group // blocks
    block = group % blocks
    dim = tl.arange(0, 128)
    col = block[:, None] * 128 + dim[None, :]
    mask = (row < rows)[:, None]
    source = tl.load(PickRows + row, row < rows, 0) if pick_rows else row
    x = tl.load(X + source[:, None] * x_stride + col, mask, 0).to(tl.float32)
    if kind == "swiglu":
        up = tl.load(X + source[:, None] * x_stride + width + col, mask, 0).to(tl.float32)
        gated = x / (1.0 + tl.exp(-x)) * up
    else:
        gate = tl.load(Gate + source[:, None] * gate_stride + col, mask, 0).to(tl.float32)
        if kind == "gdn":
            weight = tl.load(Weight + dim).to(tl.float32)
            rms = tl.rsqrt(tl.sum(x * x, 1) / 128 + eps)
            x = x * rms[:, None] * weight[None, :]
            gated = x * (gate / (1.0 + tl.exp(-gate)))
        else:
            gated = x / (1.0 + tl.exp(-gate))
    values, scale = _to_fp8(gated.to(tl.bfloat16).to(tl.float32))
    tl.store(Out + row[:, None] * width + col, values, mask)
    tl.store(Scales + block * rows + row, scale, row < rows)


def gated_quantize(x, gate, weight, eps, kind, pick_rows):
    rows, width = x.shape
    if pick_rows is not None:
        rows = pick_rows.numel()
    if kind == "swiglu":
        width //= 2
    values = torch.empty((rows, width), dtype=torch.float8_e4m3fn, device=x.device)
    blocks = width // 128
    scales = torch.empty((blocks, rows), dtype=torch.float32, device=x.device)
    groups = 1 if rows * blocks < 128 else 4
    _gated_quantize[(triton.cdiv(rows * blocks, groups),)](
        x,
        gate,
        weight,
        pick_rows,
        values,
        scales,
        rows,
        width,
        x.stride(0),
        0 if gate is None else gate.stride(0),
        eps,
        kind,
        pick_rows is not None,
        groups,
        num_warps=groups,
    )
    return values, scales

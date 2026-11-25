# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU NVFP4 quantization adapted from aegis-engines cb6779a."""

import torch
import triton as tr
import triton.language as tl

from .pointwise import launch_dependents, wait_inputs


@tr.jit
def _reciprocal(x):
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;", "=f,f", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@tr.jit
def _swiglu(G, Q, S, Global, WIDTH: tl.constexpr, BLOCK: tl.constexpr, PDL: tl.constexpr):
    row, tile = tl.program_id(0), tl.program_id(1)
    scale = tl.load(Global)
    inv = _reciprocal(scale)
    scales = tile * BLOCK + tl.arange(0, BLOCK)
    offset = scales[:, None] * 16 + tl.arange(0, 16)[None, :]
    if PDL:
        wait_inputs()
    a = tl.load(G + row * (2 * WIDTH) + offset, offset < WIDTH, 0.0).to(tl.float32)
    b = tl.load(G + row * (2 * WIDTH) + WIDTH + offset, offset < WIDTH, 0.0).to(tl.float32)
    value = ((a / (1.0 + tl.exp(-a))) * b).to(tl.bfloat16).to(tl.float32)
    largest = tl.max(tl.abs(value), 1)
    local = tl.minimum(scale * (largest * _reciprocal(tl.full((), 6.0, tl.float32))), 448.0).to(
        tl.float8e4nv
    )
    f32 = local.to(tl.float32)
    factor = tl.where(f32 != 0.0, _reciprocal(f32 * inv), 0.0)
    lo, hi = tl.split(tl.reshape(value * factor[:, None], (BLOCK, 8, 2)))
    bits = tl.inline_asm_elementwise(
        """
        {
          .reg .b8 b0, b1, b2, b3;
          cvt.rn.satfinite.e2m1x2.f32 b0, $5, $1;
          cvt.rn.satfinite.e2m1x2.f32 b1, $6, $2;
          cvt.rn.satfinite.e2m1x2.f32 b2, $7, $3;
          cvt.rn.satfinite.e2m1x2.f32 b3, $8, $4;
          mov.b32 $0, {b0, b1, b2, b3};
        }
        """,
        "=r,f,f,f,f,f,f,f,f",
        [lo, hi],
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )
    tl.store(
        Q + row * (WIDTH // 2) + scales[:, None] * 8 + tl.arange(0, 8)[None, :],
        bits,
        scales[:, None] < WIDTH // 16,
    )
    # R128C4 scales, written directly in tensor-core load order.
    swizzled = (
        ((row // 128 * (WIDTH // 64) + scales // 4) * 32 + row % 32) * 4 + row // 32 % 4
    ) * 4 + scales % 4
    tl.store(S + swizzled, local, scales < WIDTH // 16)

    if PDL:
        launch_dependents()


def swiglu_pack(activation, global_scale, pdl=False):
    rows, fused_width = activation.shape
    width = fused_width // 2
    block = 32 if rows <= 32 else 64
    packed = torch.empty((rows, width // 2), device=activation.device, dtype=torch.uint8)
    scales = torch.empty(
        (tr.cdiv(rows, 128) * 128, width // 16),
        device=activation.device,
        dtype=torch.float8_e4m3fn,
    )
    _swiglu[(rows, tr.cdiv(width // 16, block))](
        activation,
        packed,
        scales,
        global_scale,
        width,
        block,
        pdl,
        num_warps=4,
        enable_fp_fusion=False,
        launch_pdl=pdl,
    )
    return packed, scales

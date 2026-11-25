# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from aegis-engines dense_vectors.py (a6c09c6502)."""

import torch
import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=["M"])
def _vectors(
    X, W, S, Alpha, Y, M, N: tl.constexpr, K: tl.constexpr, BN: tl.constexpr, ROWS: tl.constexpr
):
    tile, group = tl.program_id(0), tl.program_id(1)
    n = tile * BN + tl.arange(0, BN)
    j = tl.arange(0, K // 2)
    b = tl.load(W + n[:, None] * (K // 2) + j[None, :]).to(tl.uint32)
    s = tl.load(S + n[:, None] * (K // 16) + j[None, :] // 8).to(tl.float32)
    lo, hi = tl.inline_asm_elementwise(
        "{ .reg .b8 c; .reg .b32 f; cvt.u8.u32 c, $2; cvt.rn.f16x2.e2m1x2 f, c; mov.b32 {$0,$1}, f; }",
        "=h,=h,r",
        [b],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )
    lo = lo.to(tl.float32)
    hi = hi.to(tl.float32)
    a = tl.load(Alpha).to(tl.float32)
    for r in tl.static_range(ROWS):
        row = group * ROWS + r
        x0 = tl.load(X + row * K + 2 * j, row < M, 0.0).to(tl.float32)
        x1 = tl.load(X + row * K + 2 * j + 1, row < M, 0.0).to(tl.float32)
        y = tl.sum((lo * x0[None, :] + hi * x1[None, :]) * s, axis=1)
        y = y.to(tl.bfloat16).to(tl.float32) * a
        tl.store(Y + row * N + n, y, row < M)


def shared_down(x, w):
    n, k = w.n, w.k
    batch = x.size(0)
    rows = 8 if batch < 64 else 4
    out = torch.empty((batch, n), device=x.device, dtype=x.dtype)
    _vectors[(n // 16, tr.cdiv(batch, rows))](
        x,
        w.w,
        w.scales,
        w.alpha,
        out,
        batch,
        n,
        k,
        16,
        rows,
        num_warps=4,
    )
    return out

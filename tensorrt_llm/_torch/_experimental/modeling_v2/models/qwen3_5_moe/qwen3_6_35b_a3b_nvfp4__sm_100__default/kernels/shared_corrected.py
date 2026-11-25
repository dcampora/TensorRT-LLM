# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from aegis-engines shared_corrected.py (a6c09c6502)."""

import os

import cutlass
import cutlass.cute as cute
import torch
import triton as tr
import triton.language as tl
from flashinfer.cute_dsl.utils import current_cuda_stream, get_max_active_clusters, make_ptr
from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16_kernel import Sm100W4A16GroupedGemmKernel

from .cute_moe import PackedVocabulary
from .quantized import _silu


@tr.jit
def _near_midpoint(x, BAND: tl.constexpr, FLOOR: tl.constexpr):
    bits = x.to(tl.uint32, bitcast=True)
    distance = tl.abs((bits & 65535).to(tl.int32) - 32768)
    # A small absolute guard accounts for strongly cancelled dot products.
    exponent = ((bits >> 23) & 255).to(tl.int32)
    multiplier = tl.maximum(0, FLOOR - exponent)
    threshold = tl.minimum(32768, BAND << tl.minimum(multiplier, 15))
    return distance <= threshold


@tr.jit
def _row_dot(X, W, S, row, n, K: tl.constexpr):
    j = tl.arange(0, K // 2)
    x0 = tl.load(X + row * K + 2 * j).to(tl.float32)
    x1 = tl.load(X + row * K + 2 * j + 1).to(tl.float32)
    w = tl.load(W + n[:, None] * (K // 2) + j[None, :])
    s = tl.load(S + n[:, None] * (K // 16) + j[None, :] // 8).to(tl.float32)
    lo, hi = tl.inline_asm_elementwise(
        "{ .reg .b8 c; .reg .b32 f; cvt.u8.u32 c, $2; cvt.rn.f16x2.e2m1x2 f, c; mov.b32 {$0,$1}, f; }",
        "=h,=h,r",
        [w.to(tl.uint32)],
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )
    return tl.sum((lo.to(tl.float32) * x0[None, :] + hi.to(tl.float32) * x1[None, :]) * s, 1)


@tr.jit
def _round(x, a):
    return (x.to(tl.bfloat16).to(tl.float32) * a).to(tl.bfloat16).to(tl.float32)


@tr.jit
def _correct(
    Raw,
    X,
    W,
    S,
    Alpha,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    GATE: tl.constexpr,
    BN: tl.constexpr,
    BLOCK: tl.constexpr,
    BAND: tl.constexpr,
    FLOOR: tl.constexpr,
):
    row, block = tl.program_id(0), tl.program_id(1)
    D: tl.constexpr = N // 2 if GATE else N
    offset = block * BLOCK
    j = tl.arange(0, BLOCK)
    n = offset + j
    a = tl.load(Alpha).to(tl.float32)
    first = tl.load(Raw + row * N + n)
    suspect = _near_midpoint(first, BAND, FLOOR)
    y = _round(first, a)
    if GATE:
        second = tl.load(Raw + row * N + n + D)
        suspect |= _near_midpoint(second, BAND, FLOOR)
        y = _silu(y) * _round(second, a)
    tl.store(Y + row * D + n, y)

    if BAND:
        # Re-evaluate ambiguous lanes in the same tiles as the ordinary GEMV.
        todo = tl.min(tl.where(suspect, j, BLOCK), axis=0) // BN * BN
        while todo < BLOCK:
            rel = tl.arange(0, BN)
            col = offset + todo + rel
            corrected = _round(_row_dot(X, W, S, row, col, K), a)
            if GATE:
                u = _round(_row_dot(X, W, S, row, col + D, K), a)
                corrected = _silu(corrected) * u
            tl.store(Y + row * D + col, corrected)
            suspect &= j >= todo + BN
            todo = tl.min(tl.where(suspect, j, BLOCK), axis=0) // BN * BN


@tr.jit
def _scan_rounding(
    Raw,
    Alpha,
    Y,
    List,
    Count,
    N: tl.constexpr,
    GATE: tl.constexpr,
    BN: tl.constexpr,
    BAND: tl.constexpr,
    FLOOR: tl.constexpr,
):
    row = tl.program_id(0)
    D: tl.constexpr = N // 2 if GATE else N
    out = tl.arange(0, D)
    alpha = tl.load(Alpha).to(tl.float32)
    acc = tl.load(Raw + row * N + out)
    suspect = _near_midpoint(acc, BAND, FLOOR)
    y = _round(acc, alpha)
    if GATE:
        up = tl.load(Raw + row * N + D + out)
        suspect |= _near_midpoint(up, BAND, FLOOR)
        y = _silu(y) * _round(up, alpha)
    tl.store(Y + row * D + out, y)
    groups = tl.max(tl.reshape(suspect.to(tl.int32), (D // BN, BN)), 1)
    offset = tl.cumsum(groups, 0) - 1
    tl.store(List + row * (D // BN) + offset, tl.arange(0, D // BN), groups != 0)
    tl.store(Count + row, tl.sum(groups, 0))


@tr.jit
def _redo_rounding(
    X,
    W,
    S,
    Alpha,
    Y,
    List,
    Count,
    N: tl.constexpr,
    K: tl.constexpr,
    GATE: tl.constexpr,
    BN: tl.constexpr,
    WORKERS: tl.constexpr,
):
    row, worker = tl.program_id(0), tl.program_id(1)
    D: tl.constexpr = N // 2 if GATE else N
    total = tl.load(Count + row)
    alpha = tl.load(Alpha).to(tl.float32)
    step = tl.arange(0, BN)
    for i in range(worker, total, WORKERS):
        group = tl.load(List + row * (D // BN) + i)
        cols = group * BN + step
        y = _round(_row_dot(X, W, S, row, cols, K), alpha)
        if GATE:
            u = _round(_row_dot(X, W, S, row, cols + D, K), alpha)
            y = _silu(y) * u
        tl.store(Y + row * D + cols, y)


class RoundedShared:
    def __init__(self, weight, band, floor):
        self.weights = weight
        self.packed = PackedVocabulary(weight)
        self.band = band
        self.floor = floor

    def raw(self, x):
        w = self.weights
        xrows = x.size(0)
        tile = 8 if xrows <= 8 else 16 if xrows <= 16 else 32 if xrows <= 32 else 64
        if xrows not in self.packed.launch_metadata:
            rows = tr.cdiv(xrows, tile)
            opts = dict(device=x.device, dtype=torch.int32)
            e = torch.zeros((rows,), **opts)
            end = (torch.arange(1, rows + 1, **opts) * tile).clamp_(max=xrows)
            count = torch.tensor([rows], **opts)
            self.packed.launch_metadata[xrows] = e, end, count
        e, end, count = self.packed.launch_metadata[xrows]
        raw = torch.empty((xrows, w.n), device=x.device, dtype=torch.float32)

        def ptr(t, dt, align=16):
            return make_ptr(dt, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)

        args = (
            ptr(w.w, cutlass.Float4E2M1FN, 32),
            ptr(self.packed.scales, cutlass.Float8E4M3FN),
            ptr(x, cutlass.BFloat16, 32),
            ptr(e, cutlass.Int32),
            ptr(end, cutlass.Int32),
            ptr(count, cutlass.Int32),
            ptr(self.packed.ones, cutlass.Float32),
            ptr(raw, cutlass.Float32, 32),
            None,
            None,
            w.n,
            xrows,
            w.k,
            0,
        )
        stream = current_cuda_stream()
        clusters = 2
        aligned = w.n % 256 == 0
        fragment = 128 if w.k == 256 else 32
        key = tile, aligned, fragment
        if key not in compiled:
            gemm = Sm100W4A16GroupedGemmKernel(
                acc_dtype=cutlass.Float32,
                use_2cta_instrs=False,
                mma_tiler_mnk=(128, tile, 256),
                cluster_shape_mn=(clusters, 1),
                group_count=1,
                activation_type=None,
                swiglu_alpha=1.0,
                swiglu_beta=0.0,
                swiglu_limit=float("inf"),
                situ_beta=None,
                situ_linear_beta=None,
                use_fused_finalize=False,
                enable_pdl=False,
                use_clc_scheduler=False,
                raster_along_m=True,
                transform_fragment_size=fragment,
                m_cluster_aligned=aligned,
            )
            compiled[key] = cute.compile(
                gemm.wrapper,
                *args,
                0,
                max_active_clusters=get_max_active_clusters(clusters),
                stream=stream,
            )
        compiled[key](*args, stream=stream)
        return raw

    def __call__(self, x, gate):
        w = self.weights
        raw = self.raw(x)
        d = w.n // 2 if gate else w.n
        y = torch.empty((x.size(0), d), device=x.device, dtype=x.dtype)
        workers = int(os.getenv("ENGINE_SHARED_WORKERS", "16" if gate and x.size(0) < 128 else "0"))
        bn = 4 if gate else 8
        if workers:
            rows = x.size(0)
            candidates = torch.empty((rows, d // bn), device=x.device, dtype=torch.int32)
            count = torch.empty((rows,), device=x.device, dtype=torch.int32)
            _scan_rounding[(rows,)](
                raw,
                w.alpha,
                y,
                candidates,
                count,
                w.n,
                gate,
                bn,
                self.band,
                self.floor,
                num_warps=4,
            )
            _redo_rounding[(rows, workers)](
                x,
                w.w,
                w.scales,
                w.alpha,
                y,
                candidates,
                count,
                w.n,
                w.k,
                gate,
                bn,
                workers,
                num_warps=4,
            )
        else:
            block = 128
            _correct[(x.size(0), d // block)](
                raw,
                x,
                w.w,
                w.scales,
                w.alpha,
                y,
                w.n,
                w.k,
                gate,
                bn,
                block,
                self.band,
                self.floor,
                num_warps=4,
            )
        return y


compiled = {}

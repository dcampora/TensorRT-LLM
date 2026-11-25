# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from aegis-engines cb6779a, fp4_tma.py.
"""Native NVFP4 decode with independent, exact K partitions."""

import torch
import triton as tr
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    TensorMemoryScalesLayout,
    allocate_tensor_memory,
    mbarrier,
    tcgen05_commit,
    tcgen05_copy,
    tcgen05_mma_scaled,
    tma,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


@gluon.jit
def _unpack_scales(x, K: gl.constexpr):
    return x.reshape((1, K // 64, 32, 4, 4)).permute((0, 3, 2, 1, 4)).reshape((128, K // 16))


@gluon.jit
def _load(W, X, WS, XS, Weight, Tokens, WeightScale, TokenScale, Ready, k, col, row, stage):
    barrier = Ready.index(stage)
    mbarrier.expect(
        barrier,
        W.block_type.nbytes + X.block_type.nbytes + WS.block_type.nbytes + XS.block_type.nbytes,
    )
    tma.async_load(W, [col, k // 2], barrier, Weight.index(stage))
    tma.async_load(X, [row, k // 2], barrier, Tokens.index(stage))
    tma.async_load(WS, [0, col // 128, k // 64, 0, 0], barrier, WeightScale.index(stage))
    tma.async_load(XS, [0, row // 128, k // 64, 0, 0], barrier, TokenScale.index(stage))


@gluon.jit
def _multiply(
    W,
    X,
    WS,
    XS,
    Partial,
    ROWS: gl.constexpr,
    COLS: gl.constexpr,
    PART_K: gl.constexpr,
    STAGES: gl.constexpr,
):
    col = gl.program_id(0) * 128
    row = gl.program_id(1) * 128
    part = gl.program_id(2)
    k_start = part * PART_K
    BK: gl.constexpr = 2 * W.block_type.shape[1]
    steps: gl.constexpr = PART_K // BK
    accumulator = allocate_tensor_memory(
        gl.float32,
        (128, 128),
        TensorMemoryLayout([128, 128], col_stride=1),
    )
    weight_tmem = allocate_tensor_memory(
        gl.float8e4nv,
        (128, BK // 16),
        TensorMemoryScalesLayout(),
    )
    token_tmem = allocate_tensor_memory(
        gl.float8e4nv,
        (128, BK // 16),
        TensorMemoryScalesLayout(),
    )
    weight = gl.allocate_shared_memory(gl.uint8, [STAGES, 128, BK // 2], W.layout)
    tokens = gl.allocate_shared_memory(gl.uint8, [STAGES, 128, BK // 2], X.layout)
    sfa = gl.allocate_shared_memory(gl.uint8, [STAGES, 1, 1, BK // 64, 2, 256], WS.layout)
    sfb = gl.allocate_shared_memory(gl.uint8, [STAGES, 1, 1, BK // 64, 2, 256], XS.layout)
    ready = gl.allocate_shared_memory(gl.int64, [STAGES, 1], mbarrier.MBarrierLayout())
    complete = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(complete, count=1)
    for i in gl.static_range(STAGES):
        mbarrier.init(ready.index(i), count=1)
        _load(W, X, WS, XS, weight, tokens, sfa, sfb, ready, k_start + i * BK, col, row, i)
    for i in range(steps):
        stage = i % STAGES
        mbarrier.wait(ready.index(stage), i // STAGES % 2)
        tcgen05_copy(_unpack_scales(sfa.index(stage), BK), weight_tmem)
        tcgen05_copy(_unpack_scales(sfb.index(stage), BK), token_tmem)
        tcgen05_mma_scaled(
            weight.index(stage),
            tokens.index(stage).permute((1, 0)),
            accumulator,
            weight_tmem,
            token_tmem,
            "e2m1",
            "e2m1",
            use_acc=i != 0,
        )
        tcgen05_commit(complete)
        mbarrier.wait(complete, i % 2)
        if i + STAGES < steps:
            _load(
                W,
                X,
                WS,
                XS,
                weight,
                tokens,
                sfa,
                sfb,
                ready,
                k_start + (i + STAGES) * BK,
                col,
                row,
                stage,
            )
    for i in gl.static_range(STAGES):
        mbarrier.invalidate(ready.index(i))
    mbarrier.invalidate(complete)

    layout: gl.constexpr = gl.BlockedLayout([4, 1], [8, 4], [4, 1], [0, 1])
    features = col + gl.arange(0, 128, layout=gl.SliceLayout(1, layout))
    positions = row + gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    address = part * ROWS * COLS + positions[None, :] * COLS + features[:, None]
    result = gl.convert_layout(accumulator.load(), layout)
    gl.store(Partial + address, result)


@tr.jit
def _sum(Partial, Output, Alpha, ELEMENTS: tl.constexpr, BLOCK: tl.constexpr):
    ix = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = ix < ELEMENTS
    lo = tl.load(Partial + ix, valid, 0.0)
    hi = tl.load(Partial + ELEMENTS + ix, valid, 0.0)
    tl.store(Output + ix, (lo + hi) * tl.load(Alpha), valid)


def _packed_tensor(value, k_tile):
    block = [128, k_tile // 2]
    return TensorDescriptor.from_tensor(
        value,
        block,
        gl.NVMMASharedLayout.get_default_for(block, gl.uint8),
    )


def _scales(value, rows, columns, k_tile):
    shape = (1, rows // 128, columns // 64, 2, 256)
    block = [1, 1, k_tile // 64, 2, 256]
    layout = gl.NVMMASharedLayout(swizzle_byte_width=0, element_bitwidth=8, rank=5)
    return TensorDescriptor.from_tensor(value.view(torch.uint8).reshape(shape), block, layout)


def split_down_tma(x, weight, scales, weight_scales, alpha):
    rows, cols, k = x.shape[0], weight.shape[1], 2 * x.shape[1]
    k_tile, stages = (512, 3) if rows == 128 else (256, 2)
    weight_map = _packed_tensor(weight.T, k_tile)
    tokens_map = _packed_tensor(x, k_tile)
    weight_scale_map = _scales(weight_scales.T, cols, k, k_tile)
    tokens_scale_map = _scales(scales, rows, k, k_tile)
    partial = torch.empty((2, rows, cols), dtype=torch.float32, device=x.device)
    out = torch.empty((rows, cols), dtype=torch.bfloat16, device=x.device)
    _multiply[(cols // 128, rows // 128, 2)](
        weight_map,
        tokens_map,
        weight_scale_map,
        tokens_scale_map,
        partial,
        rows,
        cols,
        k // 2,
        stages,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _sum[(tr.cdiv(out.numel(), 1024),)](partial, out, alpha, out.numel(), 1024, num_warps=4)
    return out

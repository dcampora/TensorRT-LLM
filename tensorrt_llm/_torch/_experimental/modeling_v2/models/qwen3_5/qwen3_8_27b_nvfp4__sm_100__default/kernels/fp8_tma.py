# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from aegis-engines cb6779a, fp8_tma.py.
"""Blackwell TMA staging for the reference's two FP8 QKV K partitions."""

import torch
import triton as tr
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
    tma,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor


@gluon.jit
def _load(Weight, Activations, W, A, Ready, column, row, inner, slot):
    barrier = Ready.index(slot)
    mbarrier.expect(barrier, Weight.block_type.nbytes + Activations.block_type.nbytes)
    tma.async_load(Weight, [column, inner], barrier, W.index(slot))
    tma.async_load(Activations, [row, inner], barrier, A.index(slot))


@gluon.jit
def _parts(
    Weight, Activations, Parts, COLS: gl.constexpr, ROWS: gl.constexpr, STAGES: gl.constexpr
):
    col_tile, row_tile, partition = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    NC: gl.constexpr = Weight.block_type.shape[0]
    NR: gl.constexpr = Activations.block_type.shape[0]
    BK: gl.constexpr = Weight.block_type.shape[1]
    STEPS: gl.constexpr = 2560 // BK
    column, row = col_tile * NC, row_tile * NR
    first = partition * 2560

    accum = allocate_tensor_memory(gl.float32, (NC, NR), TensorMemoryLayout([NC, NR], col_stride=1))
    ws = gl.allocate_shared_memory(Weight.dtype, (STAGES, NC, BK), Weight.layout)
    a_s = gl.allocate_shared_memory(Activations.dtype, (STAGES, NR, BK), Activations.layout)
    ready = gl.allocate_shared_memory(gl.int64, (STAGES, 1), mbarrier.MBarrierLayout())
    complete = gl.allocate_shared_memory(gl.int64, (1,), mbarrier.MBarrierLayout())
    for init in gl.static_range(STAGES):
        mbarrier.init(ready.index(init), count=1)
    mbarrier.init(complete, count=1)
    for preload in gl.static_range(STAGES):
        _load(Weight, Activations, ws, a_s, ready, column, row, first + preload * BK, preload)
    for i in range(STEPS):
        slot = i % STAGES
        mbarrier.wait(ready.index(slot), i // STAGES % 2)
        tcgen05_mma(ws.index(slot), a_s.index(slot).permute((1, 0)), accum, use_acc=i != 0)
        tcgen05_commit(complete)
        mbarrier.wait(complete, i % 2)
        if i + STAGES < STEPS:
            _load(Weight, Activations, ws, a_s, ready, column, row, first + (i + STAGES) * BK, slot)
    for cleanup in gl.static_range(STAGES):
        mbarrier.invalidate(ready.index(cleanup))
    mbarrier.invalidate(complete)

    L: gl.constexpr = gl.BlockedLayout([4, 1], [8, 4], [4, 1], [0, 1])
    c = column + gl.arange(0, NC, layout=gl.SliceLayout(1, L))
    r = row + gl.arange(0, NR, layout=gl.SliceLayout(0, L))
    total = gl.full((NC, NR), 0.0, gl.float32, L) + gl.convert_layout(accum.load(), L)
    dest = Parts + partition * (ROWS * COLS) + r[None, :] * COLS + c[:, None]
    gl.store(dest, total, r[None, :] < ROWS)


def qkv_parts(activation, weight):
    rows = activation.shape[0]
    cols = weight.shape[-1]
    nc, nr, bk, stages = (64, 16, 256, 2) if rows <= 8 else (128, 64, 128, 3)
    weight = weight.squeeze(0).T
    w = TensorDescriptor.from_tensor(
        weight,
        [nc, bk],
        gl.NVMMASharedLayout.get_default_for([nc, bk], gl.float8e4nv),
    )
    x = TensorDescriptor.from_tensor(
        activation,
        [nr, bk],
        gl.NVMMASharedLayout.get_default_for([nr, bk], gl.float8e4nv),
    )
    parts = torch.empty((2, rows, cols), dtype=torch.float32, device=activation.device)
    _parts[(cols // nc, tr.cdiv(rows, nr), 2)](
        w,
        x,
        parts,
        cols,
        rows,
        stages,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return parts

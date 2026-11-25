# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TMA-staged BF16 attention adapted from aegis-engines cb6779a, prefill_attention.py."""

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

SCORE_LAYOUT = gl.DistributedLinearLayout(
    [[0, 1], [0, 2], [0, 4], [0, 8]],
    [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16]],
    [[16, 0], [32, 0], [0, 32]],
    [],
    [64, 64],
)
OUTPUT_LAYOUT = gl.DistributedLinearLayout(
    [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]],
    [[1, 0], [2, 0], [4, 0], [8, 0], [0, 64]],
    [[16, 0], [32, 0], [0, 128]],
    [],
    [64, 256],
)


@gluon.jit
def _load_kv(K, V, keys, values, Ready, actual, hk, stage):
    bar = Ready.index(stage)
    mbarrier.expect(bar, K.block_type.nbytes + V.block_type.nbytes)
    tma.async_load(K, [actual, hk * 256], bar, keys.index(stage))
    tma.async_load(V, [actual, hk * 256], bar, values.index(stage))


@gluon.jit
def _update(q, k, v, score, p, accum, bar, rows, cols, limit, upper, mass):
    tcgen05_mma(q, k.permute((1, 0)), score, use_acc=False)
    tcgen05_commit(bar)
    mbarrier.wait(bar, 0)
    scores = score.load(SCORE_LAYOUT) * 0.0625
    scores = gl.where(
        (rows[:, None] >= cols[None, :]) & (cols[None, :] < limit), scores, -float("inf")
    )
    high = gl.maximum(gl.max(scores, 1), upper)
    probability = gl.exp(scores - high[:, None])
    decay = gl.exp(upper - high)
    mass = mass * decay + gl.sum(probability, 1)
    p.store(gl.convert_layout(probability.to(gl.bfloat16), p.get_reg_layout()))
    value = accum.load(OUTPUT_LAYOUT)
    decay_out = gl.convert_layout(decay, gl.SliceLayout(1, OUTPUT_LAYOUT))
    accum.store(value * decay_out[:, None])
    tcgen05_mma(p, v, accum, use_acc=True)
    tcgen05_commit(bar)
    mbarrier.wait(bar, 1)
    return high, mass


@gluon.jit
def _write(Y, accumulator, sums, base, tile, length, head):
    L: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [8, 1], [1, 0])
    d = gl.arange(0, 256, layout=gl.SliceLayout(0, L))
    r = tile * 64 + gl.arange(0, 64, layout=gl.SliceLayout(1, L))
    layout: gl.constexpr = OUTPUT_LAYOUT
    total = gl.convert_layout(sums, gl.SliceLayout(1, layout))
    out = gl.convert_layout((accumulator.load(layout) / total[:, None]).to(gl.bfloat16), L)
    gl.store(Y + ((base + r[:, None]) * 24 + head) * 256 + d[None, :], out, r[:, None] < length)


@gluon.jit
def run(Q, K, V, Y, Starts, Blocks, STAGES: gl.constexpr):
    work, head = gl.program_id(0), gl.program_id(1)
    hk = head // 6
    seq = gl.load(Blocks + work * 2)
    tile = gl.load(Blocks + work * 2 + 1)
    base = gl.load(Starts + seq)
    length = gl.load(Starts + seq + 1) - base
    limit = gl.minimum((tile + 1) * 64, length)
    q = gl.allocate_shared_memory(gl.bfloat16, (64, 256), Q.layout)
    keys = gl.allocate_shared_memory(gl.bfloat16, (STAGES, 64, 256), K.layout)
    values = gl.allocate_shared_memory(gl.bfloat16, (STAGES, 64, 256), V.layout)
    qready = gl.allocate_shared_memory(gl.int64, (1,), mbarrier.MBarrierLayout())
    kvready = gl.allocate_shared_memory(gl.int64, (STAGES, 1), mbarrier.MBarrierLayout())
    mma = gl.allocate_shared_memory(gl.int64, (1,), mbarrier.MBarrierLayout())
    mbarrier.init(qready, count=1)
    mbarrier.init(mma, count=1)
    mbarrier.expect(qready, Q.block_type.nbytes)
    tma.async_load(Q, [base + tile * 64, head * 256], qready, q)
    for preload in gl.static_range(STAGES):
        mbarrier.init(kvready.index(preload), count=1)
        if preload * 64 < limit:
            _load_kv(K, V, keys, values, kvready, base + preload * 64, hk, preload)

    score = allocate_tensor_memory(gl.float32, (64, 64), TensorMemoryLayout([64, 64], col_stride=1))
    prob = allocate_tensor_memory(gl.bfloat16, (64, 64), TensorMemoryLayout([64, 64], col_stride=1))
    o0 = allocate_tensor_memory(gl.float32, (64, 256), TensorMemoryLayout([64, 256], col_stride=1))
    o0.store(gl.full((64, 256), 0.0, gl.float32, OUTPUT_LAYOUT))
    RS: gl.constexpr = gl.SliceLayout(1, SCORE_LAYOUT)
    rows = tile * 64 + gl.arange(0, 64, layout=RS)
    cols = gl.arange(0, 64, layout=gl.SliceLayout(0, SCORE_LAYOUT))
    m0 = gl.full((64,), -float("inf"), gl.float32, RS)
    l0 = gl.full((64,), 0.0, gl.float32, RS)
    mbarrier.wait(qready, 0)
    mbarrier.invalidate(qready)
    for i in range(gl.cdiv(limit, 64)):
        slot = i % STAGES
        mbarrier.wait(kvready.index(slot), (i // STAGES) % 2)
        k = keys.index(slot)
        v = values.index(slot)
        t = i * 64 + cols
        m0, l0 = _update(q, k, v, score, prob, o0, mma, rows, t, limit, m0, l0)
        if (i + STAGES) * 64 < limit:
            _load_kv(K, V, keys, values, kvready, base + (i + STAGES) * 64, hk, slot)
    for final_stage in gl.static_range(STAGES):
        mbarrier.invalidate(kvready.index(final_stage))
    mbarrier.invalidate(mma)
    _write(Y, o0, l0, base, tile, length, head)


def attention(q, k, v, output, starts, blocks):
    tile = [64, 256]
    layout = gl.NVMMASharedLayout.get_default_for(tile, gl.bfloat16)
    descriptors = tuple(
        TensorDescriptor.from_tensor(tensor.flatten(1), tile, layout) for tensor in (q, k, v)
    )
    run[(blocks.shape[0], 24)](
        *descriptors,
        output,
        starts,
        blocks,
        2,
        num_warps=8,
    )

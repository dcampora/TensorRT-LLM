# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chunked gated delta rule adapted from aegis-engines cb6779a, prefill.py."""

import torch
import triton as tr
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

from .pointwise import launch_dependents, wait_inputs


@tr.jit
def _bounds(Starts, Blocks, work):
    seq = tl.load(Blocks + work * 2)
    block = tl.load(Blocks + work * 2 + 1)
    first = tl.load(Starts + seq)
    end = tl.load(Starts + seq + 1)
    return first + block * 64, end


@tr.jit
def _normalize(QKV, Q, K, N: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        wait_inputs()
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    d = tl.arange(0, 128)
    pos = (
        (rows[:, None] // 16) * 10240
        + (rows[:, None] % 16) * 128
        + tl.program_id(1) * 2048
        + d[None, :]
    )
    x = tl.load(QKV + pos, rows[:, None] < N, 0).to(tl.float32)
    y = x / tl.sqrt(tl.sum(x * x, 1) + 1.0e-6)[:, None]
    packed = rows[:, None] * 128 + d[None, :]
    if tl.program_id(1) == 0:
        tl.store(Q + packed, y, rows[:, None] < N)
    else:
        tl.store(K + packed, y, rows[:, None] < N)

    if PDL:
        launch_dependents()


@tr.jit
def _prefix_decay(G, C, Starts, Blocks, PDL: tl.constexpr = False):
    if PDL:
        wait_inputs()
    h = tl.program_id(1)
    first, end = _bounds(Starts, Blocks, tl.program_id(0))
    rows = first + tl.arange(0, 64)
    x = tl.load(G + rows * 48 + h, rows < end, 0.0)
    tl.store(C + rows * 48 + h, tl.cumsum(x, axis=0), rows < end)

    if PDL:
        launch_dependents()


@tr.jit
def _decay(x):
    return tl.exp(tl.where(x <= 0.0, x, -float("inf")))


@tr.jit
def _gram_block(
    K, C, Beta, start, end, h, ROW_TILE: tl.constexpr, COL_TILE: tl.constexpr, BK: tl.constexpr
):
    i = tl.arange(0, 16)
    ri = start + ROW_TILE * 16 + i
    rj = start + COL_TILE * 16 + i
    inner = tl.arange(0, BK)
    key = K + (h // 3) * 128
    result = tl.full((16, 16), 0.0, tl.float32)
    for p in range(128 // BK):
        d = p * BK + inner
        x = tl.load(key + ri[:, None] * 2048 + d[None, :], ri[:, None] < end, 0.0)
        y = tl.load(key + rj[None, :] * 2048 + d[:, None], rj[None, :] < end, 0.0)
        result += tl.dot(x, y)
    a = tl.load(C + ri * 48 + h, ri < end, 0.0)
    b = tl.load(C + rj * 48 + h, rj < end, 0.0)
    beta = tl.load(Beta + ri * 48 + h, ri < end, 0.0).to(tl.float32)
    result *= _decay(a[:, None] - b[None, :])
    if ROW_TILE == COL_TILE:
        result = tl.where(i[:, None] > i[None, :], result, 0.0)
    return result * beta[:, None]


@tr.jit
def _inverse_diagonal(lower, N):
    d = tl.arange(0, 16)
    inv = -lower
    for i in range(2, tl.minimum(16, N)):
        row = tl.sum(tl.where((d == i)[:, None], -lower, 0.0), 0)
        row = tl.where(d < i, row, 0.0)
        row += tl.sum(row[:, None] * inv, 0)
        inv = tl.where((d == i)[:, None], row[None, :], inv)
    return inv + (d[:, None] == d[None, :])


@tr.jit
def _store_inverse(A, matrix, start, end, h, ROW_TILE: tl.constexpr, COL_TILE: tl.constexpr):
    d = tl.arange(0, 16)
    r = start + ROW_TILE * 16 + d
    c = COL_TILE * 16 + d
    tl.store(A + (r[:, None] * 48 + h) * 64 + c[None, :], matrix, r[:, None] < end)


@tr.jit
def _merge(left, right):
    return tl.dot(left, right, input_precision="tf32")


@tr.jit
def _solve(K, C, Beta, A, Starts, Blocks, BK: tl.constexpr):
    h = tl.program_id(1)
    first, end = _bounds(Starts, Blocks, tl.program_id(0))
    count = end - first
    a00 = _gram_block(K, C, Beta, first, end, h, 0, 0, BK)
    i00 = _inverse_diagonal(a00, count)
    _store_inverse(A, i00, first, end, h, 0, 0)
    if count > 16:
        a10 = _gram_block(K, C, Beta, first, end, h, 1, 0, BK)
        a11 = _gram_block(K, C, Beta, first, end, h, 1, 1, BK)
        i11 = _inverse_diagonal(a11, count - 16)
        i10 = -_merge(_merge(i11, a10), i00)
        _store_inverse(A, i10, first, end, h, 1, 0)
        _store_inverse(A, i11, first, end, h, 1, 1)
        if count > 32:
            a20 = _gram_block(K, C, Beta, first, end, h, 2, 0, BK)
            a21 = _gram_block(K, C, Beta, first, end, h, 2, 1, BK)
            a22 = _gram_block(K, C, Beta, first, end, h, 2, 2, BK)
            i22 = _inverse_diagonal(a22, count - 32)
            i21 = -_merge(_merge(i22, a21), i11)
            i20 = -_merge(i22, _merge(a20, i00) + _merge(a21, i10))
            _store_inverse(A, i20, first, end, h, 2, 0)
            _store_inverse(A, i21, first, end, h, 2, 1)
            _store_inverse(A, i22, first, end, h, 2, 2)
            if count > 48:
                a30 = _gram_block(K, C, Beta, first, end, h, 3, 0, BK)
                a31 = _gram_block(K, C, Beta, first, end, h, 3, 1, BK)
                a32 = _gram_block(K, C, Beta, first, end, h, 3, 2, BK)
                a33 = _gram_block(K, C, Beta, first, end, h, 3, 3, BK)
                i33 = _inverse_diagonal(a33, count - 48)
                i32 = -_merge(_merge(i33, a32), i22)
                i31 = -_merge(i33, _merge(a31, i11) + _merge(a32, i21))
                i30 = -_merge(i33, _merge(a30, i00) + _merge(a31, i10) + _merge(a32, i20))
                _store_inverse(A, i30, first, end, h, 3, 0)
                _store_inverse(A, i31, first, end, h, 3, 1)
                _store_inverse(A, i32, first, end, h, 3, 2)
                _store_inverse(A, i33, first, end, h, 3, 3)


@tr.jit
def _gram_blocks(Key, C, Beta, Tiles, Starts, Blocks, BK: tl.constexpr, SHARE: tl.constexpr = 1):
    chunk, head, pair = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    first, end = _bounds(Starts, Blocks, chunk)
    i = tl.where(pair >= 6, 3, tl.where(pair >= 3, 2, tl.where(pair >= 1, 1, 0)))
    j = pair - i * (i + 1) // 2
    if first + i * 16 >= end:
        return
    idx = tl.arange(0, 16)
    rows = first + i * 16 + idx
    cols = first + j * 16 + idx
    inner = tl.arange(0, BK)
    base = Key + (head // 3 if SHARE == 1 else head) * 128
    result = tl.full((16, 16), 0.0, tl.float32)
    for part in range(128 // BK):
        d = part * BK + inner
        left = tl.load(base + rows[:, None] * 2048 + d[None, :], rows[:, None] < end, 0.0)
        right = tl.load(base + cols[None, :] * 2048 + d[:, None], cols[None, :] < end, 0.0)
        result += tl.dot(left, right)
    for h in tl.static_range(SHARE):
        pos = head * SHARE + h
        grow = tl.load(C + rows * 48 + pos, rows < end, 0.0)
        gcol = tl.load(C + cols * 48 + pos, cols < end, 0.0)
        beta = tl.load(Beta + rows * 48 + pos, rows < end, 0.0).to(tl.float32)
        output = result * _decay(grow[:, None] - gcol[None, :])
        if i == j:
            lower = tl.where(idx[:, None] > idx[None, :], output, 0.0)
            output = _inverse_diagonal(lower * beta[:, None], end - first - i * 16)
        else:
            output *= beta[:, None]
        p = ((chunk * 48 + pos) * 10 + pair) * 256
        tl.store(Tiles + p + idx[:, None] * 16 + idx[None, :], output)


@gluon.jit
def _matrix(Partial, work, pair: gl.constexpr, L: gl.constexpr):
    r = gl.arange(0, 16, layout=gl.SliceLayout(1, L))
    c = gl.arange(0, 16, layout=gl.SliceLayout(0, L))
    return gl.load(Partial + (work * 10 + pair) * 256 + r[:, None] * 16 + c[None, :])


@gluon.jit
def _product(a, b, c, M: gl.constexpr, K: gl.constexpr):
    # The original solve uses packed pairs for input products and the
    # single-column operand layout for each final triangular merge.
    aa = gl.convert_layout(a, gl.DotOperandLayout(0, M, K))
    bb = gl.convert_layout(b, gl.DotOperandLayout(1, M, K))
    return mma_v2(aa, bb, c, input_precision="tf32")


@gluon.jit
def _write(Inv, matrix, first, end, head, i: gl.constexpr, j: gl.constexpr, L: gl.constexpr):
    v = gl.convert_layout(matrix, L)
    t = first + i * 16 + gl.arange(0, 16, layout=gl.SliceLayout(1, L))
    c = j * 16 + gl.arange(0, 16, layout=gl.SliceLayout(0, L))
    gl.store(Inv + (t[:, None] * 48 + head) * 64 + c[None, :], v, t[:, None] < end)


@gluon.jit
def _assemble_exact(Partial, Inv, Starts, Chunks, WARPS: gl.constexpr):
    block, head = gl.program_id(0), gl.program_id(1)
    seq = gl.load(Chunks + block * 2)
    offset = gl.load(Chunks + block * 2 + 1)
    first = gl.load(Starts + seq) + offset * 64
    end = gl.load(Starts + seq + 1)
    work = block * 48 + head
    span = end - first
    memory: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [WARPS, 1], [1, 0])
    mma: gl.constexpr = gl.NVMMADistributedLayout([2, 0], [WARPS, 1], [16, 8])
    z = gl.full((16, 16), 0.0, gl.float32, mma)
    i00 = _matrix(Partial, work, 0, memory)
    _write(Inv, i00, first, end, head, 0, 0, memory)
    if span > 16:
        a10 = _matrix(Partial, work, 1, memory)
        i11 = _matrix(Partial, work, 2, memory)
        p10 = _product(i11, a10, z, mma, 2)
        i10 = -_product(p10, i00, z, mma, 1)
        _write(Inv, i10, first, end, head, 1, 0, memory)
        _write(Inv, i11, first, end, head, 1, 1, memory)
        if span > 32:
            a20 = _matrix(Partial, work, 3, memory)
            a21 = _matrix(Partial, work, 4, memory)
            i22 = _matrix(Partial, work, 5, memory)
            p21 = _product(i22, a21, z, mma, 2)
            i21 = -_product(p21, i11, z, mma, 1)
            p20 = _product(a20, i00, z, mma, 2)
            p20 = _product(a21, i10, p20, mma, 2)
            i20 = -_product(i22, p20, z, mma, 1)
            _write(Inv, i20, first, end, head, 2, 0, memory)
            _write(Inv, i21, first, end, head, 2, 1, memory)
            _write(Inv, i22, first, end, head, 2, 2, memory)
            if span > 48:
                a30 = _matrix(Partial, work, 6, memory)
                a31 = _matrix(Partial, work, 7, memory)
                a32 = _matrix(Partial, work, 8, memory)
                i33 = _matrix(Partial, work, 9, memory)
                p32 = _product(i33, a32, z, mma, 2)
                i32 = -_product(p32, i22, z, mma, 1)
                p31 = _product(a31, i11, z, mma, 2)
                p31 = _product(a32, i21, p31, mma, 2)
                i31 = -_product(i33, p31, z, mma, 1)
                p30 = _product(a30, i00, z, mma, 2)
                p30 = _product(a31, i10, p30, mma, 2)
                p30 = _product(a32, i20, p30, mma, 2)
                i30 = -_product(i33, p30, z, mma, 1)
                _write(Inv, i30, first, end, head, 3, 0, memory)
                _write(Inv, i31, first, end, head, 3, 1, memory)
                _write(Inv, i32, first, end, head, 3, 2, memory)
                _write(Inv, i33, first, end, head, 3, 3, memory)


@tr.jit
def _project(
    K, QKV, C, Beta, A, W, U, Starts, Blocks, MASK_INVERSE: tl.constexpr, PDL: tl.constexpr = False
):
    if PDL:
        wait_inputs()
    h = tl.program_id(1)
    first, end = _bounds(Starts, Blocks, tl.program_id(0))
    t = tl.arange(0, 64)
    r = first + t
    mask = r < end
    valid = mask[:, None]
    if MASK_INVERSE:
        valid = valid & (t[:, None] // 16 >= t[None, :] // 16)
    inv = tl.load(A + (r[:, None] * 48 + h) * 64 + t[None, :], valid, 0.0)
    beta = tl.load(Beta + r * 48 + h, mask, 0.0)
    decay = tl.exp(tl.load(C + r * 48 + h, mask, 0.0))
    for col in range(2):
        d = col * 64 + t
        v = tl.load(QKV + r[:, None] * 10240 + 4096 + h * 128 + d[None, :], mask[:, None], 0.0)
        weighted_v = (v * beta[:, None]).to(tl.bfloat16)
        u = tl.dot(inv, weighted_v)
        tl.store(U + (r[:, None] * 48 + h) * 128 + d[None, :], u, mask[:, None])
    for col in range(2):
        d = col * 64 + t
        k = tl.load(K + r[:, None] * 2048 + h // 3 * 128 + d[None, :], mask[:, None], 0.0)
        weighted_k = (k * beta[:, None] * decay[:, None]).to(tl.bfloat16)
        w = tl.dot(inv, weighted_k)
        tl.store(W + (r[:, None] * 48 + h) * 128 + d[None, :], w, mask[:, None])

    if PDL:
        launch_dependents()


@tr.jit
def _advance(
    K,
    C,
    W,
    U,
    ChunkH,
    VNew,
    State,
    Starts,
    ChunkStarts,
    Slots,
    Initial,
    STATE_STRIDE: tl.constexpr,
    BV: tl.constexpr,
    PDL: tl.constexpr = False,
):
    if PDL:
        wait_inputs()
    tile, h, seq = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    first = tl.load(Starts + seq)
    end = tl.load(Starts + seq + 1)
    block = tl.load(ChunkStarts + seq)
    v = tile * BV + tl.arange(0, BV)
    d = tl.arange(0, 64)
    row = v[:, None] * 128 + d[None, :]
    h0 = tl.full((BV, 64), 0.0, tl.float32)
    h1 = tl.full((BV, 64), 0.0, tl.float32)
    slot = tl.load(Slots + seq).to(tl.int64)
    state = State + slot * STATE_STRIDE + h * 16384
    if tl.load(Initial + seq):
        h0 = tl.load(state + row).to(tl.float32)
        h1 = tl.load(state + row + 64).to(tl.float32)
    for chunk in range(tl.cdiv(end - first, 64)):
        t = first + chunk * 64 + d
        last = tl.minimum(first + chunk * 64 + 63, end - 1)
        saved = ChunkH + ((block + chunk).to(tl.int64) * 48 + h) * 16384
        tl.store(saved + row, h0)
        tl.store(saved + row + 64, h1)
        wp = W + (t[:, None] * 48 + h) * 128 + d[None, :]
        w0 = tl.load(wp, t[:, None] < end, 0.0)
        w1 = tl.load(wp + 64, t[:, None] < end, 0.0)
        u = tl.load(U + (t[:, None] * 48 + h) * 128 + v[None, :], t[:, None] < end, 0.0)
        correction = tl.dot(w0, tl.trans(h0).to(tl.bfloat16))
        correction += tl.dot(w1, tl.trans(h1).to(tl.bfloat16))
        residual = u.to(tl.float32) - correction
        tl.store(VNew + (t[:, None] * 48 + h) * 128 + v[None, :], residual, t[:, None] < end)
        g_last = tl.load(C + last * 48 + h)
        g = tl.load(C + t * 48 + h, t < end, 0.0)
        update = (residual * _decay(g_last - g)[:, None]).to(tl.bfloat16)
        k = K + t[None, :] * 2048 + (h // 3) * 128 + d[:, None]
        k0 = tl.load(k, t[None, :] < end, 0.0)
        k1 = tl.load(k + 64, t[None, :] < end, 0.0)
        h0 *= tl.exp(g_last)
        h1 *= tl.exp(g_last)
        h0 += tl.trans(tl.dot(k0, update))
        h1 += tl.trans(tl.dot(k1, update))
    tl.store(state + row, h0)
    tl.store(state + row + 64, h1)

    if PDL:
        launch_dependents()


@tr.jit
def _output(
    Q,
    K,
    C,
    ChunkH,
    VNew,
    Y,
    Starts,
    Blocks,
    BT: tl.constexpr,
    BV: tl.constexpr = 64,
    PDL: tl.constexpr = False,
):
    if PDL:
        wait_inputs()
    chunk, h, value = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq = tl.load(Blocks + chunk * 2)
    tile = tl.load(Blocks + chunk * 2 + 1)
    first = tl.load(Starts + seq)
    end = tl.load(Starts + seq + 1)
    t = tl.arange(0, BT)
    rows = first + tile * BT + t
    d = tl.arange(0, 128)
    v = value * BV + tl.arange(0, BV)
    offset = rows[:, None] * 2048 + (h // 3) * 128 + d[None, :]
    q = tl.load(Q + offset, rows[:, None] < end, 0.0)
    k = tl.load(K + offset, rows[:, None] < end, 0.0)
    hp = (chunk.to(tl.int64) * 48 + h) * 16384 + v[:, None] * 128 + d[None, :]
    state = tl.load(ChunkH + hp)
    decay = tl.load(C + rows * 48 + h, rows < end, 0.0)
    between = tl.dot(q, tl.trans(k)) * _decay(decay[:, None] - decay[None, :])
    between = tl.where(t[:, None] >= t[None, :], between, 0.0)
    partial = tl.dot(q, tl.trans(state)) * tl.exp(decay)[:, None]
    offset = (rows[:, None] * 48 + h) * 128 + v[None, :]
    update = tl.load(VNew + offset, rows[:, None] < end, 0.0)
    scale = 128**-0.5
    y = partial * scale + tl.dot(between.to(tl.bfloat16), update) * scale
    tl.store(Y + offset, y, rows[:, None] < end)

    if PDL:
        launch_dependents()


def gdn(qkv, g, beta, meta, state, *, share_gram=True, pdl=False):
    n = qkv.shape[0]
    q = qkv.new_empty((n, 2048))
    k = torch.empty_like(q)
    _normalize[(tr.cdiv(n * 16, 16), 2)](
        qkv, q, k, n * 16, pdl, num_warps=8, num_stages=3, launch_pdl=pdl
    )
    chunks = meta.chunks.shape[0]
    c = torch.empty_like(g)
    _prefix_decay[(chunks, 48)](
        g,
        c,
        meta.starts,
        meta.chunks,
        pdl,
        num_warps=2,
        launch_pdl=pdl,
    )
    inverse = qkv.new_empty((n, 48, 64))
    if n >= 128:
        partial = torch.empty((chunks, 48, 10, 256), dtype=torch.float32, device=qkv.device)
        heads = 3 if share_gram and n >= 3072 else 1
        _gram_blocks[(chunks, 48 // heads, 10)](
            k,
            c,
            beta,
            partial,
            meta.starts,
            meta.chunks,
            64,
            heads,
            num_warps=1,
        )
        _assemble_exact[(chunks, 48)](
            partial,
            inverse,
            meta.starts,
            meta.chunks,
            1,
            num_warps=1,
        )
    else:
        _solve[(chunks, 48)](k, c, beta, inverse, meta.starts, meta.chunks, 64, num_warps=4)
    w = qkv.new_empty((n, 48, 128))
    u = torch.empty_like(w)
    _project[(chunks, 48)](
        k,
        qkv,
        c,
        beta,
        inverse,
        w,
        u,
        meta.starts,
        meta.chunks,
        True,
        pdl,
        num_warps=4,
        num_stages=3,
        launch_pdl=pdl,
    )
    h = qkv.new_empty((chunks, 48, 128, 128))
    v = torch.empty_like(u)
    wide = meta.longest >= 3072
    advance_tile = 16 if wide else 32
    _advance[(128 // advance_tile, 48, meta.batch)](
        k,
        c,
        w,
        u,
        h,
        v,
        state,
        meta.starts,
        meta.chunk_starts,
        meta.sequence_slots,
        meta.has_initial_state,
        state.stride(0),
        advance_tile,
        pdl,
        num_warps=4,
        num_stages=3 if wide else 2,
        launch_pdl=pdl,
    )
    out = torch.empty_like(u)
    output_tile = 128 if wide else 64
    _output[(chunks, 48, 128 // output_tile)](
        q,
        k,
        c,
        h,
        v,
        out,
        meta.starts,
        meta.chunks,
        meta.gdn_output_tile,
        output_tile,
        pdl,
        num_warps=8 if wide else 4,
        num_stages=2,
        launch_pdl=pdl,
    )
    return out.view(n, 6144)

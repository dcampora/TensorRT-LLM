# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from aegis-engines cute_moe.py (a6c09c6502)."""

import os

import torch
import triton as tr
import triton.language as tl
from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16 import _run_grouped_gemm
from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16_prepare import moe_permute_w4a16
from flashinfer.fused_moe.cute_dsl.moe_utils import get_max_num_permuted_tokens
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait, libdevice

from .quantized import _silu, _store_normalized

CUTE_PDL = int(os.getenv("ENGINE_CUTE_PDL", "2"))
ROUTE_HIST = int(os.getenv("ENGINE_ROUTE_HIST", "8"))
ROUTE_HIST_LIMIT = int(os.getenv("ENGINE_ROUTE_HIST_LIMIT", "1024"))
PERM_INPUT = os.getenv("ENGINE_PERM_INPUT", "1") == "1"
ROUTE_PART = int(os.getenv("ENGINE_PREFILL_ROUTE_PART", "256"))


@tr.jit
def _swizzle_scales(X, Y, SIZE, GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tile_n = i // (GROUPS * 128)
    tile_k = i // 512 % (GROUPS // 4)
    within = i % 512
    n = tile_n * 128 + (within // 4 % 4) * 32 + within // 16
    k = tile_k * 4 + within % 4
    x = tl.load(X + n * GROUPS + k, i < SIZE, 0)
    tl.store(Y + i, x, i < SIZE)


def swizzle_scales(w):
    e, n, kg = w.scales.shape
    source = w.scales.view(torch.uint8)
    out = torch.empty_like(source)
    _swizzle_scales[(tr.cdiv(out.numel(), 4096),)](source, out, out.numel(), kg, 4096, num_warps=4)
    return convert_sf_to_mma_layout(out, m=n, k=kg * 16, num_groups=e, sf_vec_size=16)


@tr.jit
def _tile_limits(Counts, Prefix, Expert, Limit, TILE: tl.constexpr, Q: tl.constexpr):
    e = tl.program_id(0)
    count = tl.load(Counts + e)
    start = tl.load(Prefix + e)
    t = tl.arange(0, Q)
    ti = start + t
    tl.store(Expert + ti, e, t * TILE < count)
    end = start * TILE + tl.minimum((t + 1) * TILE, count)
    tl.store(Limit + ti, end, t * TILE < count)


@tr.jit
def _map_routed(
    Ids, SortedSlots, Starts, Prefix, Permute, Unpermute, M, TILE: tl.constexpr, BLOCK: tl.constexpr
):
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = t < M
    e = tl.load(Ids + t, valid, 0)
    slot = tl.load(SortedSlots + t, valid, 0).to(tl.int32)
    base = tl.load(Prefix + e) * TILE
    pos = base + t - tl.load(Starts + e)
    tl.store(Permute + pos, slot, valid)
    tl.store(Unpermute + slot, pos, valid)


@tr.jit
def _maximum(left, right):
    return tl.maximum(left, right)


@tr.jit
def _single_cta_routes(
    Ids,
    Expert,
    Limit,
    Permute,
    Inv,
    Total,
    S: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    e = tl.load(Ids + i, i < S, 256).to(tl.uint32)
    keys = tl.sort((e << 16) | i.to(tl.uint32), descending=False)
    sorted_e = (keys >> 16).to(tl.int32)
    slot = (keys & 65535).to(tl.int32)
    previous_expert = tl.gather(sorted_e, tl.maximum(i - 1, 0), axis=0)
    next_expert = tl.gather(sorted_e, tl.minimum(i + 1, BLOCK - 1), axis=0)
    first_token = (i == 0) | (previous_expert != sorted_e)
    first = tl.associative_scan(tl.where(first_token, i, 0), 0, _maximum)
    offset = i - first
    first_in_tile = (offset % TILE == 0) & (i < S)
    tile = tl.cumsum(first_in_tile.to(tl.int32), 0) - 1
    last_in_tile = (offset % TILE == TILE - 1) | (i + 1 == S) | (sorted_e != next_expert)
    where = tile * TILE + offset % TILE
    tl.store(Expert + tile, sorted_e, first_in_tile)
    tl.store(Limit + tile, where + 1, last_in_tile & (i < S))
    tl.store(Permute + where, slot, i < S)
    tl.store(Inv + slot, where, i < S)
    tl.store(Total, tl.sum(first_in_tile.to(tl.int32), 0))


@tr.jit
def _route_histogram(Ids, Counts, Slots, SortedExperts, S, BLOCK: tl.constexpr):
    part = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    i = part * BLOCK + j
    e = tl.load(Ids + i, i < S, 256).to(tl.uint32)
    counts = tl.histogram(e, 256)
    tl.store(Counts + part * 256 + tl.arange(0, 256), counts)
    keys = tl.sort((e << 16) | j.to(tl.uint32), descending=False)
    tl.store(Slots + i, (keys & 65535).to(tl.int32) + part * BLOCK, i < S)
    tl.store(SortedExperts + i, (keys >> 16).to(tl.int32), i < S)


@tr.jit
def _partitions_prefix(Counts, Offsets, Totals, PARTS, BP: tl.constexpr, BE: tl.constexpr):
    p = tl.arange(0, BP)
    expert = tl.program_id(0) * BE + tl.arange(0, BE)
    off = p[:, None] * 256 + expert[None, :]
    counts = tl.load(Counts + off, p[:, None] < PARTS, 0)
    prefix = tl.cumsum(counts, axis=0) - counts
    tl.store(Offsets + off, prefix, p[:, None] < PARTS)
    tl.store(Totals + expert, tl.sum(counts, axis=0))


@tr.jit
def _padded_prefix(Totals, Prefix, Total, TILE: tl.constexpr):
    expert = tl.arange(0, 256)
    n = tl.load(Totals + expert)
    tiles = tl.cdiv(n, TILE)
    ends = tl.cumsum(tiles, axis=0)
    tl.store(Prefix + expert, ends - tiles)
    tl.store(Total, tl.sum(tiles, axis=0))


@tr.jit
def _place_partitions(
    Ids,
    Slots,
    Offsets,
    Totals,
    Prefix,
    Expert,
    Limit,
    Permute,
    Inv,
    S,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    i = p * BLOCK + j
    e = tl.load(Ids + i, i < S, 255)
    slot = tl.load(Slots + i, i < S, 0)
    prev = tl.gather(e, tl.maximum(j - 1, 0), axis=0)
    first = (j == 0) | (e != prev)
    first = tl.associative_scan(tl.where(first, j, 0), 0, _maximum)
    prior = tl.load(Offsets + p * 256 + e)
    offset = j - first + prior
    base = tl.load(Prefix + e)
    tile = base + offset // TILE
    where = base * TILE + offset
    last = tl.load(Totals + e)
    tl.store(Expert + tile, e, (offset % TILE == 0) & (i < S))
    tl.store(
        Limit + tile, where + 1, ((offset % TILE == TILE - 1) | (offset + 1 == last)) & (i < S)
    )
    tl.store(Permute + where, slot, i < S)
    tl.store(Inv + slot, where, i < S)


@tr.jit
def _histogram_metadata(
    Ids, Counts, Prefix, Total, S: tl.constexpr, BLOCK: tl.constexpr, TILE: tl.constexpr
):
    i = tl.arange(0, BLOCK)
    e = tl.load(Ids + i, i < S, 256)
    count = tl.histogram(e, 256)
    tiles = tl.cdiv(count, TILE)
    j = tl.arange(0, 256)
    tl.store(Counts + j, count)
    tl.store(Prefix + j, tl.cumsum(tiles, axis=0) - tiles)
    tl.store(Total, tl.sum(tiles, axis=0))


@tr.jit
def _place_by_expert(
    Ids,
    Counts,
    Prefix,
    Expert,
    Limit,
    Perm,
    Inv,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
    TILE: tl.constexpr,
):
    e = tl.program_id(0)
    n = tl.load(Counts + e)
    if n != 0:
        i = tl.arange(0, BLOCK)
        chosen = tl.load(Ids + i, i < S, 256) == e
        index = tl.cumsum(chosen.to(tl.int32), axis=0) - 1
        base = tl.load(Prefix + e)
        row = base * TILE + index
        tile = base + index // TILE
        tl.store(Expert + tile, e, chosen & (index % TILE == 0))
        last = (index % TILE == TILE - 1) | (index == n - 1)
        tl.store(Limit + tile, row + 1, chosen & last)
        tl.store(Perm + row, i, chosen)
        tl.store(Inv + i, row, chosen)


def group_routes(ids, tile, rows, bins):
    batch = ids.size(0)
    opts = dict(device=ids.device, dtype=torch.int32)
    expert = torch.empty((rows // tile,), **opts)
    limit = torch.empty_like(expert)
    perm = torch.empty((rows,), **opts)
    inv = torch.empty(ids.shape, **opts)
    if ROUTE_HIST and 256 <= ids.numel() <= ROUTE_HIST_LIMIT:
        total = torch.empty((1,), **opts)
        counts = torch.empty((256,), **opts)
        prefix = torch.empty_like(counts)
        n = ids.numel()
        block = tr.next_power_of_2(n)
        _histogram_metadata[(1,)](ids, counts, prefix, total, n, block, tile, num_warps=4)
        _place_by_expert[(256,)](
            ids, counts, prefix, expert, limit, perm, inv, n, block, tile, num_warps=ROUTE_HIST
        )
        return expert, limit, inv, perm, total
    if ids.numel() <= 1024:
        total = torch.empty((1,), **opts)
        _single_cta_routes[(1,)](
            ids,
            expert,
            limit,
            perm,
            inv,
            total,
            ids.numel(),
            tile,
            tr.next_power_of_2(ids.numel()),
            num_warps=8,
        )
        return expert, limit, inv, perm, total
    if ROUTE_PART and ids.numel() <= 65536:
        size = ids.numel()
        parts = tr.cdiv(size, ROUTE_PART)
        counts = torch.empty((parts, 256), **opts)
        offsets = torch.empty_like(counts)
        totals = torch.empty((256,), **opts)
        prefix = torch.empty_like(totals)
        total = torch.empty((1,), **opts)
        slots = torch.empty((size,), **opts)
        sorted_ids = torch.empty_like(slots)
        _route_histogram[(parts,)](
            ids,
            counts,
            slots,
            sorted_ids,
            size,
            ROUTE_PART,
            num_warps=4 if ROUTE_PART <= 512 else 8,
        )
        _partitions_prefix[(16,)](
            counts, offsets, totals, parts, tr.next_power_of_2(parts), 16, num_warps=4
        )
        _padded_prefix[(1,)](totals, prefix, total, tile, num_warps=4)
        _place_partitions[(parts,)](
            sorted_ids,
            slots,
            offsets,
            totals,
            prefix,
            expert,
            limit,
            perm,
            inv,
            size,
            tile,
            ROUTE_PART,
            num_warps=4 if ROUTE_PART <= 512 else 8,
        )
        return expert, limit, inv, perm, total
    sorted_ids, slots = torch.sort(ids.flatten(), stable=True)
    starts = torch.searchsorted(sorted_ids, bins, out_int32=True)
    counts = starts[1:] - starts[:-1]
    lengths = (counts + tile - 1) // tile
    prefix = torch.cumsum(torch.cat((bins[:1], lengths)), dim=0, dtype=torch.int32)
    total = prefix[-1:]
    _tile_limits[(256,)](
        counts, prefix, expert, limit, tile, tr.next_power_of_2(tr.cdiv(batch, tile)), num_warps=4
    )
    _map_routed[(tr.cdiv(ids.numel(), 256),)](
        sorted_ids, slots, starts, prefix, perm, inv, ids.numel(), tile, 256
    )
    return expert, limit, inv, perm, total


@tr.jit
def _permute_input(X, Inv, Out, B: tl.constexpr, PDL: tl.constexpr):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    row, i = tl.program_id(0), tl.program_id(1)
    d = i * B + tl.arange(0, B)
    value = tl.load(X + row * 2048 + d)
    for slot in tl.static_range(8):
        target = tl.load(Inv + row * 8 + slot)
        tl.store(Out + target * 2048 + d, value)


@tr.jit
def _scaled_swiglu(
    Raw,
    Alpha,
    Expert,
    End,
    Active,
    Y,
    TILE: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    R: tl.constexpr,
    PDL: tl.constexpr = False,
):
    if PDL:
        gdc_wait()
        gdc_launch_dependents()
    tile = tl.program_id(0)
    if tile < tl.load(Active):
        sr = tl.load(Expert + tile)
        end = tl.load(End + tile)
        a = tl.load(Alpha + sr).to(tl.float32)
        r = tile * TILE + tl.program_id(1) * R + tl.arange(0, R)
        i = tl.arange(0, INTERMEDIATE)
        g = tl.load(Raw + r[:, None] * 2 * INTERMEDIATE + i[None, :], r[:, None] < end, 0.0).to(
            tl.float32
        )
        u = tl.load(
            Raw + r[:, None] * 2 * INTERMEDIATE + INTERMEDIATE + i[None, :], r[:, None] < end, 0.0
        ).to(tl.float32)
        g = (g * a).to(tl.bfloat16).to(tl.float32)
        u = (u * a).to(tl.bfloat16).to(tl.float32)
        tl.store(Y + r[:, None] * INTERMEDIATE + i[None, :], _silu(g) * u, r[:, None] < end)


@tr.jit
def _sum_experts(
    Raw, Alpha, Ids, P, Unperm, row, d, BLOCK: tl.constexpr, WEIGHTED: tl.constexpr = False
):
    acc = tl.full((BLOCK,), 0.0, tl.float32)
    for slot in tl.static_range(8):
        src = tl.load(Unperm + row * 8 + slot)
        weight = tl.load(P + row * 8 + slot).to(tl.bfloat16).to(tl.float32)
        if not WEIGHTED:
            expert = tl.load(Ids + row * 8 + slot)
            a = tl.load(Alpha + expert).to(tl.float32)
            weight = (a * weight).to(tl.bfloat16).to(tl.float32)
        x = tl.load(Raw + src * 2048 + d).to(tl.float32)
        acc += (x * weight).to(tl.bfloat16).to(tl.float32)
    return acc.to(tl.bfloat16).to(tl.float32)


@tr.jit
def _finish(Raw, Alpha, Ids, P, Unperm, Y, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    d = block * BLOCK + tl.arange(0, BLOCK)
    acc = _sum_experts(Raw, Alpha, Ids, P, Unperm, row, d, BLOCK)
    tl.store(Y + row * 2048 + d, acc)


@tr.jit
def mix_experts_norm(
    Raw,
    Alpha,
    Ids,
    P,
    Unperm,
    Shared,
    Router,
    Residual,
    Norm,
    Y,
    STRIDE: tl.constexpr,
    SAVE: tl.constexpr,
    WEIGHTED: tl.constexpr = False,
    Q=None,
    Scale=None,
    QUANT: tl.constexpr = False,
):
    row = tl.program_id(0)
    d = tl.arange(0, 2048)
    offset = row * 2048 + d
    routed = _sum_experts(Raw, Alpha, Ids, P, Unperm, row, d, 2048, WEIGHTED)
    if WEIGHTED:
        g = tl.load(Router + row).to(tl.float32)
    else:
        g = tl.load(Router + row * STRIDE + 256).to(tl.float32)
        g = (1.0 / (1.0 + libdevice.exp(-g))).to(tl.bfloat16).to(tl.float32)
    shared = (tl.load(Shared + offset).to(tl.float32) * g).to(tl.bfloat16).to(tl.float32)
    mixed = (routed + shared).to(tl.bfloat16).to(tl.float32)
    x = mixed + tl.load(Residual + offset).to(tl.float32)
    if SAVE:
        tl.store(Residual + offset, x)
    w = tl.load(Norm + d).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / 2048.0 + 1e-6)
    _store_normalized(Y, Q, Scale, offset, x, inv, w, QUANT)


class PackedExperts:
    def __init__(self, gate_up, down):
        self.w1 = gate_up
        self.w2 = down
        self.s1 = swizzle_scales(self.w1)
        self.gate = (self.w1.w, self.s1, self.w1.alpha.float()) if self.w1.interleaved else None
        self.s2 = swizzle_scales(self.w2)
        self.ones = torch.ones((256,), dtype=torch.float32, device=gate_up.w.device)
        self.bins = torch.arange(257, dtype=torch.int32, device=gate_up.w.device)

    def __call__(self, x, ids, probs, route_tile=8, tactic_m=128, cluster=1, raw=False):
        m = x.size(0)
        pdl = CUTE_PDL != 0 and m <= 512
        staged_pdl = CUTE_PDL >= 2 and pdl
        max_rows = get_max_num_permuted_tokens(m, 8, 256, route_tile)
        e, end, slot, perm, active = group_routes(ids, route_tile, max_rows, self.bins)
        bufs = dict(dtype=x.dtype, device=x.device)
        pack = torch.empty((max_rows, 2048), **bufs)
        v = torch.empty((max_rows, 512), **bufs)
        y = torch.empty_like(x)

        if PERM_INPUT and m <= 8192:
            size = 128 if m <= 32 else 512
            _permute_input[(m, 2048 // size)](x, slot, pack, size, pdl, num_warps=4, launch_pdl=pdl)
        else:
            moe_permute_w4a16(
                input=x,
                permuted_output=pack,
                tile_idx_to_mn_limit=end,
                permuted_idx_to_expanded_idx=perm,
                num_non_exiting_tiles=active,
                max_num_permuted_tokens=max_rows,
                top_k=8,
                tile_size=route_tile,
                enable_pdl=pdl,
            )
        tactic = ((tactic_m, route_tile, 256), (cluster, 1), True)

        common = dict(
            tile_idx_to_expert_idx=e,
            tile_idx_to_mn_limit=end,
            num_non_exiting_tiles=active,
            alpha=self.ones,
            num_local_experts=256,
            activation_type=None,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
            swiglu_limit=float("inf"),
            situ_beta=None,
            situ_linear_beta=None,
            use_fused_finalize=False,
            permuted_idx_to_expanded_idx=None,
            token_final_scales=None,
            enable_pdl=pdl,
            tactic=tactic,
        )
        if self.gate is not None:
            from .fused_gate import launch

            launch(self.gate, pack, v, common)
        else:
            gu = torch.empty((max_rows, 1024), **bufs)
            _run_grouped_gemm(
                weight=self.w1.w, weight_sf=self.s1, activations=pack, output=gu, **common
            )
            _scaled_swiglu[(max_rows // route_tile, route_tile // 4)](
                gu,
                self.w1.alpha,
                e,
                end,
                active,
                v,
                route_tile,
                512,
                4,
                staged_pdl,
                num_warps=4,
                launch_pdl=staged_pdl,
            )
        _run_grouped_gemm(weight=self.w2.w, weight_sf=self.s2, activations=v, output=pack, **common)
        if raw:
            return pack, slot
        _finish[(m, 4)](pack, self.w2.alpha, ids, probs, slot, y, 512, num_warps=4)
        return y


@tr.jit
def _vocab_tile_max(Raw, Alpha, Peaks, N: tl.constexpr, BLOCK: tl.constexpr):
    row, block = tl.program_id(0), tl.program_id(1)
    i = block * BLOCK + tl.arange(0, BLOCK)
    b = tl.load(Raw + row * N + i, i < N, 0.0).to(tl.float32)
    a = tl.load(Alpha).to(tl.float32)
    logits = (b * a).to(tl.bfloat16).to(tl.float32)
    values = tl.where(i < N, logits, -float("inf"))
    peak = tl.max(values, 0)
    pick = tl.min(tl.where(values == peak, i, N), 0)
    peak = tl.where(peak == 0.0, 0.0, peak)
    bits = peak.to(tl.uint32, bitcast=True)
    order = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000)
    key = (order.to(tl.uint64) << 32) | (0xFFFFFFFF - pick.to(tl.uint32))
    tl.store(Peaks + row * tl.cdiv(N, BLOCK) + block, key)


@tr.jit
def _vocab_finish(Peaks, Ids, TILES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    key = tl.max(tl.load(Peaks + row * TILES + i, i < TILES, 0), 0)
    token = 0xFFFFFFFF - (key & 0xFFFFFFFF).to(tl.uint32)
    tl.store(Ids + row, token.to(tl.int64))


class PackedVocabulary:
    def __init__(self, linear):
        self.weight = linear.w.unsqueeze(0)
        self.alpha = linear.alpha
        n, k = linear.n, linear.k
        kg = k // 16
        source = linear.scales.view(torch.uint8)
        swizzled = torch.empty_like(source)
        _swizzle_scales[(tr.cdiv(swizzled.numel(), 4096),)](
            source,
            swizzled,
            swizzled.numel(),
            kg,
            4096,
            num_warps=4,
        )
        self.scales = convert_sf_to_mma_layout(swizzled, m=n, k=k, num_groups=1, sf_vec_size=16)
        self.ones = torch.ones((1,), device=source.device, dtype=torch.float32)
        self.launch_metadata = {}

    def _raw(self, x, vocab=None):
        m = x.size(0)
        n = self.weight.size(1) if vocab is None else vocab
        w = self.weight[:, :n]
        if n >= 65536 and m >= 96:
            from flashinfer.gemm import mm_bf16_fp4

            return mm_bf16_fp4(x, w[0], self.scales, backend="cute-dsl", enable_pdl=False)
        tile = 8 if m <= 8 else 16 if m <= 16 else 32 if m <= 32 else 64
        tiles = tr.cdiv(m, tile)
        if m not in self.launch_metadata:
            opts = dict(device=x.device, dtype=torch.int32)
            e = torch.zeros((tiles,), **opts)
            end = (torch.arange(1, tiles + 1, **opts) * tile).clamp_(max=m)
            count = torch.tensor([tiles], **opts)
            self.launch_metadata[m] = (e, end, count)
        e, end, count = self.launch_metadata[m]
        raw = torch.empty((m, n), dtype=x.dtype, device=x.device)
        _run_grouped_gemm(
            weight=w,
            weight_sf=self.scales,
            activations=x,
            output=raw,
            tile_idx_to_expert_idx=e,
            tile_idx_to_mn_limit=end,
            num_non_exiting_tiles=count,
            alpha=self.ones,
            num_local_experts=1,
            activation_type=None,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
            swiglu_limit=float("inf"),
            situ_beta=None,
            situ_linear_beta=None,
            use_fused_finalize=False,
            permuted_idx_to_expanded_idx=None,
            token_final_scales=None,
            enable_pdl=False,
            tactic=((128, tile, 256), (2, 1), True),
        )
        return raw

    def logits(self, x):
        return self._raw(x) * self.alpha

    def top1(self, x, vocab=None):
        raw = self._raw(x, vocab)
        m, n = raw.shape
        block = 2048
        tiles = tr.cdiv(n, block)
        peaks = torch.empty((m, tiles), device=x.device, dtype=torch.uint64)
        out = torch.empty((m,), device=x.device, dtype=torch.long)
        _vocab_tile_max[(m, tiles)](raw, self.alpha, peaks, n, block, num_warps=4)
        _vocab_finish[(m,)](peaks, out, tiles, tr.next_power_of_2(tiles), num_warps=4)
        return out

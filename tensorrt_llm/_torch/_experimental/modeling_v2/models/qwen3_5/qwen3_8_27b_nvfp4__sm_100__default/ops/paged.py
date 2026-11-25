# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 paged-cache bindings for the target's causal attention kernels."""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton as tr

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels import paged_attention, prefill_attention
from .attention import HEAD_SIZE, KV_HEADS, QUERY_HEADS

_DECODE_SPLITS = 8


class Cache(NamedTuple):
    layer: int
    group: int
    index_divisor: int
    key: torch.Tensor
    value: torch.Tensor


def bind_layer(manager, index: int, pool_mapping: torch.Tensor) -> Cache:
    """Bind a view whose row spacing is the physical BF16 cache page spacing."""
    layer = manager.layer_offsets[index]
    pages = manager.get_buffers(index, kv_layout="HND")
    return Cache(
        layer,
        int(pool_mapping[layer, 0]),
        pages.stride(0) // pages.stride(1),
        pages[:, 0],
        pages[:, 1],
    )


class AppendPagedKv(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(why="Context crosses a page boundary", spec={"lengths": (39, 5), "width": 1}),
        Cell(why="Four verify rows cross a page boundary", spec={"lengths": (4, 4), "width": 4}),
    )

    def __call__(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cache: Cache,
        table: torch.Tensor,
        requests: torch.Tensor | None,
        rows_per_seq: int = 1,
        request_start: int = 0,
    ) -> None:
        paged_attention.append_paged[(key.shape[0], KV_HEADS)](
            key,
            value,
            cache.key,
            cache.value,
            positions,
            table,
            requests,
            cache.key.shape[2],
            cache.key.stride(0),
            cache.index_divisor,
            table.stride(0),
            rows_per_seq,
            request_start,
            num_warps=4,
        )

    def reference(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cache: Cache,
        table: torch.Tensor,
        requests: torch.Tensor | None,
        rows_per_seq: int = 1,
        request_start: int = 0,
    ) -> None:
        page_size = cache.key.shape[2]
        if requests is None:
            requests = (
                torch.arange(positions.numel(), device=positions.device) // rows_per_seq
                + request_start
            )
        pages = (
            table[requests.long(), positions.long() // page_size] // cache.index_divisor
        ).long()
        index = positions.long() % page_size
        cache.key[pages, :, index, :] = key.view(-1, KV_HEADS, HEAD_SIZE)
        cache.value[pages, :, index, :] = value.view(-1, KV_HEADS, HEAD_SIZE)


def _attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    q = query.float().view(-1, QUERY_HEADS, HEAD_SIZE).transpose(0, 1)
    k = key.float().view(-1, KV_HEADS, HEAD_SIZE).repeat_interleave(6, dim=1).transpose(0, 1)
    v = value.float().view(-1, KV_HEADS, HEAD_SIZE).repeat_interleave(6, dim=1).transpose(0, 1)
    scores = (q @ k.transpose(-1, -2)) * HEAD_SIZE**-0.5
    future = torch.arange(key.shape[0], device=query.device)[None, :] > positions[:, None]
    probabilities = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
    return (probabilities @ v).transpose(0, 1).reshape(query.shape).to(query.dtype)


class CausalContextAttention(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(why="Fresh packed contexts", spec={"lengths": (15, 89)}),
        Cell(why="Full context", spec={"lengths": (1024,)}),
        Cell(why="TMA stages packed contexts at uneven offsets", spec={"lengths": (512, 513, 17)}),
    )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        starts: torch.Tensor,
        chunks: torch.Tensor,
    ) -> torch.Tensor:
        y = torch.empty_like(q)
        if q.shape[0] >= 512:
            prefill_attention.attention(q, k, v, y, starts, chunks)
            return y
        paged_attention.causal_prefill[(chunks.shape[0], QUERY_HEADS)](
            q,
            k,
            v,
            y,
            starts,
            chunks,
            QUERY_HEADS,
            6,
            64,
            64,
            False,
            num_warps=8,
            num_stages=2,
        )
        return y

    def reference(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        starts: torch.Tensor,
        chunks: torch.Tensor,
    ) -> torch.Tensor:
        spans = starts.tolist()
        return torch.cat(
            [
                _attention_reference(
                    q[lo:hi], k[lo:hi], v[lo:hi], torch.arange(hi - lo, device=q.device)
                )
                for lo, hi in zip(spans[:-1], spans[1:], strict=True)
            ]
        )


class CausalPagedContext(OpWrapper):
    """Evaluate a context chunk against its carried and newly appended KV."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(why="Two contexts start inside different pages", spec={"prefixes": (31, 113)}),
        Cell(why="Fresh and continued context", spec={"prefixes": (0, 95)}),
    )

    def __call__(
        self,
        q: torch.Tensor,
        starts: torch.Tensor,
        chunks: torch.Tensor,
        positions: torch.Tensor,
        cache: Cache,
        table: torch.Tensor,
    ) -> torch.Tensor:
        y = torch.empty_like(q)
        paged_attention.causal_prefill[(chunks.shape[0], QUERY_HEADS)](
            q,
            cache.key,
            cache.value,
            y,
            starts,
            chunks,
            QUERY_HEADS,
            6,
            64,
            64,
            False,
            Positions=positions,
            Table=table,
            TABLE_STRIDE=table.stride(0),
            PAGE_SIZE=cache.key.shape[2],
            PAGE_STRIDE=cache.key.stride(0),
            PAGE_DIVISOR=cache.index_divisor,
            num_warps=8,
            num_stages=2,
        )
        return y

    def reference(
        self,
        q: torch.Tensor,
        starts: torch.Tensor,
        chunks: torch.Tensor,
        positions: torch.Tensor,
        cache: Cache,
        table: torch.Tensor,
    ) -> torch.Tensor:
        spans = starts.tolist()
        size = cache.key.shape[2]
        outputs = []
        for request, (lo, hi) in enumerate(zip(spans[:-1], spans[1:], strict=True)):
            n = int(positions[hi - 1]) + 1
            ids = torch.arange(n, device=q.device)
            pages = (table[request, ids // size] // cache.index_divisor).long()
            k = cache.key[pages, :, ids % size, :].reshape(n, -1)
            v = cache.value[pages, :, ids % size, :].reshape(n, -1)
            outputs.append(_attention_reference(q[lo:hi], k, v, positions[lo:hi]))
        return torch.cat(outputs)


class PagedCausalDecode(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            why="Solo row with context spread over unequal splits",
            spec={"lengths": (261,), "width": 1},
        ),
        Cell(why="Mixed verify lanes", spec={"lengths": (31, 4096, 144), "width": 4}),
        Cell(
            why="Pairs straddle split boundaries and finish with an odd lane",
            spec={"lengths": (31, 257, 260, 513, 777, 1024, 129, 33), "width": 3},
        ),
    )

    def __call__(
        self,
        q: torch.Tensor,
        gate: torch.Tensor,
        positions: torch.Tensor,
        cache: Cache,
        table: torch.Tensor,
        scale: torch.Tensor | None = None,
        rows_per_seq: int = 1,
        request_start: int = 0,
    ) -> torch.Tensor:
        rows = q.shape[0]
        # Use one CTA per KV head and split. The merge rounds the attention
        # result through BF16 before applying the gate and static FP8 scale.
        partial = q.new_empty((rows, QUERY_HEADS, _DECODE_SPLITS, HEAD_SIZE), dtype=torch.float32)
        lse = q.new_empty((rows, QUERY_HEADS, _DECODE_SPLITS), dtype=torch.float32)
        nseq = rows // rows_per_seq
        if rows_per_seq > 1 and nseq >= 8:
            pairs = tr.cdiv(rows_per_seq, 2)
            paged_attention.paged_decode_pair[(nseq * pairs, KV_HEADS, _DECODE_SPLITS)](
                q,
                cache.key,
                cache.value,
                partial,
                lse,
                positions,
                table,
                table.stride(0),
                cache.key.shape[2],
                cache.key.stride(0),
                cache.index_divisor,
                QUERY_HEADS,
                6,
                _DECODE_SPLITS,
                rows_per_seq,
                request_start,
                num_warps=4,
                num_stages=3,
            )
        else:
            paged_attention.paged_decode_part[(rows, KV_HEADS, _DECODE_SPLITS)](
                q,
                cache.key,
                cache.value,
                partial,
                lse,
                positions,
                table,
                table.stride(0),
                cache.key.shape[2],
                cache.key.stride(0),
                cache.index_divisor,
                QUERY_HEADS,
                6,
                _DECODE_SPLITS,
                _DECODE_SPLITS,
                32,
                16,
                rows_per_seq,
                request_start,
                num_warps=4,
                num_stages=2,
            )
        out = torch.empty_like(q, dtype=torch.float8_e4m3fn if scale is not None else q.dtype)
        pdl = rows <= 8
        paged_attention.decode_merge[(rows, QUERY_HEADS)](
            partial,
            lse,
            positions,
            gate,
            out,
            QUERY_HEADS,
            _DECODE_SPLITS,
            _DECODE_SPLITS,
            scale,
            scale is not None,
            pdl,
            num_warps=4,
            launch_pdl=pdl,
        )
        return out

    def reference(
        self,
        q: torch.Tensor,
        gate: torch.Tensor,
        positions: torch.Tensor,
        cache: Cache,
        table: torch.Tensor,
        scale: torch.Tensor | None = None,
        rows_per_seq: int = 1,
        request_start: int = 0,
    ) -> torch.Tensor:
        page_size = cache.key.shape[2]
        output = []
        for lo in range(0, q.shape[0], rows_per_seq):
            hi = lo + rows_per_seq
            sequence = request_start + lo // rows_per_seq
            length = int(positions[hi - 1]) + 1
            pages = (table[sequence, : tr.cdiv(length, page_size)] // cache.index_divisor).long()
            key = cache.key[pages].transpose(1, 2).reshape(-1, KV_HEADS * HEAD_SIZE)[:length]
            value = cache.value[pages].transpose(1, 2).reshape(-1, KV_HEADS * HEAD_SIZE)[:length]
            y = _attention_reference(q[lo:hi], key, value, positions[lo:hi])
            output.append(y)
        out = (torch.cat(output).float() * gate.float().sigmoid()).to(q.dtype)
        if scale is not None:
            out = (out.float() * scale.reciprocal()).clamp(-448, 448).to(torch.float8_e4m3fn)
        return out


append_kv = AppendPagedKv()
context_attention = CausalContextAttention()
decode_attention = PagedCausalDecode()
paged_context_attention = CausalPagedContext()

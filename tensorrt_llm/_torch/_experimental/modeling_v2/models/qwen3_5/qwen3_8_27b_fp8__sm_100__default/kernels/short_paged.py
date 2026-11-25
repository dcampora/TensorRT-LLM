# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Short paged GQA using the aegis-engines split-softmax reduction."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _partial(
    Q,
    PAGES,
    TABLE,
    POSITIONS,
    PART,
    LSE,
    Q_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    INDEX_DIVISOR: tl.constexpr,
    CONTEXTS: tl.constexpr,
    TOKENS: tl.constexpr,
    SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split = tl.program_id(2)
    position = tl.load(POSITIONS + row)
    end = position + 1
    start = split * 32
    if start < end and end <= SPLITS * 32:
        seq = CONTEXTS + row // TOKENS
        head = tl.arange(0, 16)
        d = tl.arange(0, 256)
        tokens = start + tl.arange(0, 32)
        valid = tokens < end
        block = tl.load(
            TABLE + seq * TABLE_STRIDE + tokens // PAGE_SIZE,
            valid,
            -1,
        )
        slot = (block // INDEX_DIVISOR).to(tl.int64)
        offset = slot * PAGE_STRIDE + (kv_head * PAGE_SIZE + tokens % PAGE_SIZE) * 256
        query = tl.load(
            Q + row * Q_STRIDE + (kv_head * 6 + head[:, None]) * 256 + d[None, :],
            head[:, None] < 6,
            0,
        )
        key = tl.load(
            PAGES + offset[None, :] + d[:, None],
            (valid & (block >= 0))[None, :],
            0,
        )
        values = tl.load(
            PAGES + KV_STRIDE + offset[:, None] + d[None, :],
            (valid & (block >= 0))[:, None],
            0,
        )
        scores = tl.where(valid[None, :], tl.dot(query, key) * 0.0625, -float("inf"))
        maximum = tl.max(scores, 1)
        prob = tl.exp(scores - maximum[:, None])
        total = tl.sum(prob, 1)
        value = tl.dot(prob.to(tl.bfloat16), values) / total[:, None]
        index = (row * SPLITS + split) * 24 + kv_head * 6 + head
        tl.store(PART + index[:, None] * 256 + d[None, :], value, head[:, None] < 6)
        tl.store(LSE + index, maximum + tl.log(total), head < 6)


@tr.jit
def _merge(
    PART,
    LSE,
    POSITIONS,
    OUTPUT,
    SPLITS: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.program_id(2) * TILE + tl.arange(0, TILE)
    end = tl.load(POSITIONS + row) + 1
    if end > 0 and end <= SPLITS * 32:
        splits = tl.arange(0, SPLITS)
        valid = splits * 32 < end
        index = (row * SPLITS + splits) * 24 + head
        lse = tl.load(LSE + index, valid, -float("inf"))
        weight = tl.exp(lse - tl.max(lse, 0))
        value = tl.load(PART + index[:, None] * 256 + d[None, :], valid[:, None], 0)
        result = tl.sum(value * weight[:, None], 0) / tl.sum(weight, 0)
        tl.store(OUTPUT + (row * 24 + head) * 256 + d, result)


def short_paged(
    q: torch.Tensor,
    pages: torch.Tensor,
    table: torch.Tensor,
    positions: torch.Tensor,
    result: torch.Tensor,
    index_divisor: int,
    contexts: int,
    tokens: int,
) -> torch.Tensor:
    """Replace attention with the short-history reduction for positions below 256."""
    rows, heads, dim = q.shape
    num_splits = 8
    parts = torch.empty((rows, num_splits, heads, dim), device=q.device, dtype=torch.float32)
    lse = torch.empty((rows, num_splits, heads), device=q.device, dtype=torch.float32)
    _partial[(rows, 4, num_splits)](
        q,
        pages,
        table,
        positions,
        parts,
        lse,
        q.stride(0),
        pages.stride(0),
        pages.stride(1),
        table.stride(0),
        pages.shape[-2],
        index_divisor,
        contexts,
        tokens,
        num_splits,
        num_warps=4,
    )
    _merge[(rows, heads, 4)](parts, lse, positions, result, num_splits, 64, num_warps=4)
    return result

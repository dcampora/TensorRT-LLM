# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Append pre-rotated BF16 keys and values to the runtime page table."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _append(
    QKV,
    TABLE,
    OFFSETS,
    SEQ,
    PAGES,
    PAGE_STRIDE: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TABLE_SIZE: tl.constexpr,
    DIVISOR: tl.constexpr,
):
    row, head, kv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if SEQ is None:
        sequence = 0
        position = row
    else:
        sequence = tl.load(SEQ + row)
        position = row - tl.load(OFFSETS + sequence)
    page = position // PAGE_SIZE
    block = tl.load(TABLE + sequence * TABLE_STRIDE + page, (page >= 0) & (page < TABLE_SIZE), -1)
    offset = (block // DIVISOR).to(tl.int64) * PAGE_STRIDE
    target = offset + kv * KV_STRIDE + (head * PAGE_SIZE + position % PAGE_SIZE) * 256
    lane = tl.arange(0, 256)
    data = tl.load(QKV + row * 8192 + 6144 + kv * 1024 + head * 256 + lane)
    tl.store(PAGES + target + lane, data, block >= 0)


def append_kv(
    qkv: torch.Tensor,
    pages: torch.Tensor,
    table: torch.Tensor,
    divisor: int,
    offsets: torch.Tensor,
    seq_ids: torch.Tensor | None,
) -> None:
    """Write only the K and V tokens from each new packed QKV row."""
    _append[(qkv.shape[0], 4, 2)](
        qkv,
        table,
        offsets,
        seq_ids,
        pages,
        pages.stride(0),
        pages.stride(1),
        pages.shape[-2],
        table.stride(0),
        table.shape[-1],
        divisor,
        num_warps=4,
    )

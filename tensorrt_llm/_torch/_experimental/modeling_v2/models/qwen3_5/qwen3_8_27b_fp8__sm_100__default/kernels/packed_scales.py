# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Encode the existing 128-channel FP8 scales for Blackwell tensor-core kernels."""

import torch
import triton as tr
import triton.language as tl

from tensorrt_llm.quantization.utils.fp8_utils import transform_sf_into_required_layout


@tr.jit(do_not_specialize=["HEIGHT", "SOURCES"])
def _pack_mx(
    S, P, HEIGHT, SOURCES, K_TILES: tl.constexpr, TOTAL: tl.constexpr, CHUNK: tl.constexpr
):
    index = tl.program_id(0) * CHUNK + tl.arange(0, CHUNK)
    tile = index // 128
    k_group = tile % K_TILES
    atom_row = index % 128
    row = tile // K_TILES * 128 + atom_row // 4 + (atom_row % 4) * 32
    scale = tl.load(S + row + k_group * SOURCES, (row < HEIGHT) & (index < TOTAL), 1.0)
    exponent = scale.to(tl.uint32, bitcast=True) >> 23
    tl.store(P + index, exponent * 0x01010101, index < TOTAL)


def pack_activation_mx(scale: torch.Tensor, rows: int) -> torch.Tensor:
    """Repeat each group exponent for the four MXFP8 subgroups it covers."""
    groups, source_rows = scale.shape
    count = tr.cdiv(rows, 128) * 128 * groups
    out = torch.empty((count,), dtype=torch.int32, device=scale.device)
    _pack_mx[(tr.cdiv(count, 256),)](scale, out, rows, source_rows, groups, count, 256)
    return out.view(torch.uint8)


@tr.jit(do_not_specialize=["HEIGHT", "SOURCES"])
def _pack_deepgemm(S, P, HEIGHT, SOURCES, K_TILES: tl.constexpr, CHUNK: tl.constexpr):
    row = tl.program_id(0) * CHUNK + tl.arange(0, CHUNK)
    k_start = tl.program_id(1) * 4
    word = tl.full((CHUNK,), 0, tl.uint32)
    for i in tl.static_range(4):
        scale = tl.load(
            S + row + (k_start + i) * SOURCES, (row < HEIGHT) & (k_start + i < K_TILES), 0.0
        )
        exponent = scale.to(tl.uint32, bitcast=True) >> 23
        word |= exponent << (i * 8)
    tl.store(P + row + tl.program_id(1) * SOURCES, word, row < SOURCES)


def pack_activation_deepgemm(scale: torch.Tensor, rows: int) -> torch.Tensor:
    """Pack four consecutive group exponents into each MN-major word."""
    groups, source_rows = scale.shape
    output = torch.empty((tr.cdiv(groups, 4), source_rows), dtype=torch.int32, device=scale.device)
    _pack_deepgemm[(tr.cdiv(source_rows, 32), output.shape[0])](
        scale, output, rows, source_rows, groups, 32
    )
    return output.T[:rows]


def pack_weight(scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """Expand weight scale exponents across the rows within each 128-row group."""
    return transform_sf_into_required_layout(
        scale.float().contiguous(), n, k, (1, 128, 128), is_sfa=False
    )

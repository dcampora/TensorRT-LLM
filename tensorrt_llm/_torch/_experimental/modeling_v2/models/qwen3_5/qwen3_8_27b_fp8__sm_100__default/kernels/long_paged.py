# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-request GQA from aegis-engines kernels.py and attention_packed.py (114cf57c911e)."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _paged_parts(
    Q,
    PAGES,
    TABLE,
    POS,
    PART,
    LSE,
    NS: tl.constexpr,
    SPAN: tl.constexpr,
    PAGE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    KV_STRIDE: tl.constexpr,
    DIVISOR: tl.constexpr,
    Q_STRIDE: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    end = tl.load(POS + row) + 1
    start = split * SPAN
    if start < end:
        head_row = tl.arange(0, 16)
        h = head * 6 + head_row
        d = tl.arange(0, 256)
        tile = tl.arange(0, 32)
        use = head_row < 6
        q = tl.load(Q + row * Q_STRIDE + h[:, None] * 256 + d[None, :], use[:, None], 0)

        out = tl.full((16, 256), 0, tl.float32)
        peak = tl.where(use, -float("inf"), 0.0)
        denominator = tl.full((16,), 0, tl.float32)
        for page_part in range(tl.cdiv(tl.minimum(end - start, SPAN), 32)):
            begin = start + page_part * 32
            t = begin + tile
            block = tl.load(TABLE + begin // PAGE)
            offset = (block // DIVISOR).to(tl.int64) * PAGE_STRIDE
            within_page = ((head * PAGE + begin % PAGE) + tile) * 256
            key = tl.load(
                PAGES + offset + within_page[None, :] + d[:, None],
                (t < end)[None, :] & (block >= 0),
                0.0,
            )
            value = tl.load(
                PAGES + offset + KV_STRIDE + within_page[:, None] + d[None, :],
                (t < end)[:, None] & (block >= 0),
                0.0,
            )
            score = tl.dot(q, key) * 0.0625
            score = tl.where((t[None, :] < end) & use[:, None], score, -float("inf"))
            new_peak = tl.maximum(peak, tl.max(score, 1))
            probability = tl.exp(score - new_peak[:, None])
            ratio = tl.exp(peak - new_peak)
            out = out * ratio[:, None] + tl.dot(probability.to(tl.bfloat16), value)
            denominator = denominator * ratio + tl.sum(probability, 1)
            peak = new_peak
        off = (row * NS + split) * 24 + h
        tl.store(PART + off[:, None] * 256 + d[None, :], out / denominator[:, None], use[:, None])
        tl.store(LSE + off, peak + tl.log(denominator), use)


@tr.jit
def _paged_merge(
    P,
    LSE,
    X,
    POS,
    Y,
    NS: tl.constexpr,
    SPLITS: tl.constexpr,
    SPAN: tl.constexpr,
    SCALE_OUT,
    PADDED: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.program_id(2) * 128 + tl.arange(0, 128)
    end = tl.load(POS + row, row < ROWS, -1) + 1
    if end > 0:
        split = tl.arange(0, SPLITS)
        valid = (split < NS) & (split * SPAN < end)
        ix = (row * NS + split) * 24 + head
        z = tl.load(LSE + ix, valid, -float("inf"))
        alpha = tl.exp(z - tl.max(z, 0))
        sub = tl.load(P + ix[:, None] * 256 + d[None, :], valid[:, None], 0)
        value = tl.sum(alpha[:, None] * sub, 0) / tl.sum(alpha, 0)
        value = value.to(tl.bfloat16).to(tl.float32)
        gate = tl.load(X + row * 14336 + head * 512 + 256 + d).to(tl.float32)
        y = (value * tl.sigmoid(gate)).to(tl.bfloat16)
    else:
        y = tl.full((128,), 0.0, tl.bfloat16)

    y = y.to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(y), 0), 1.0e-10)
    bits = (maximum * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    exp = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    scale = (exp << 23).to(tl.float32, bitcast=True)
    inv = ((254 - exp) << 23).to(tl.float32, bitcast=True)
    y = tl.clamp(y * inv, -448.0, 448.0)
    tl.store(SCALE_OUT + (head * 2 + tl.program_id(2)) * PADDED + row, scale)

    tl.store(Y + (row * 24 + head) * 256 + d, y)


def decode_paged(
    q: torch.Tensor,
    pages: torch.Tensor,
    table: torch.Tensor,
    positions: torch.Tensor,
    projection: torch.Tensor,
    divisor: int,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = q.shape[0]
    span = 128
    n_splits = tr.cdiv(max_seq_len, span)
    part = torch.empty((rows, n_splits, 24, 256), dtype=torch.float32, device=q.device)
    lse = torch.empty((rows, n_splits, 24), dtype=torch.float32, device=q.device)
    _paged_parts[(rows, 4, n_splits)](
        q,
        pages,
        table,
        positions,
        part,
        lse,
        n_splits,
        span,
        pages.shape[-2],
        pages.stride(0),
        pages.stride(1),
        divisor,
        q.stride(0),
        num_warps=4,
    )
    padded = tr.cdiv(rows, 4) * 4
    activations = torch.empty((padded, 6144), dtype=torch.float8_e4m3fn, device=q.device)
    scales = torch.empty((48, padded), dtype=torch.float32, device=q.device)
    _paged_merge[(padded, 24, 2)](
        part,
        lse,
        projection,
        positions,
        activations,
        n_splits,
        tr.next_power_of_2(n_splits),
        span,
        scales,
        padded,
        rows,
        num_warps=4,
    )
    return activations, scales

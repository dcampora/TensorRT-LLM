# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Draft feature normalization adapted from aegis-engines cb6779a, draft_front.py."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _norm_pair(
    Tokens,
    Emb,
    Hidden,
    WEmb,
    WHidden,
    Output,
    WIDTH: tl.constexpr,
    HS: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row, kind = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, BLOCK)
    if kind == 0:
        ptr = Emb + tl.maximum(tl.load(Tokens + row), 0) * WIDTH
        weights = WEmb
    else:
        ptr = Hidden + row * HS
        weights = WHidden
    valid = col < WIDTH
    x = tl.load(ptr + col, valid, 0.0).to(tl.float32)
    w = tl.load(weights + col, valid, 0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, 0) / WIDTH + EPS)
    normalized = x * inv * (1.0 + w)
    tl.store(Output + (row * 2 + kind) * WIDTH + col, normalized, valid)


def normalized_features(tokens, embedding, hidden, embedding_norm, hidden_norm, eps):
    rows, width = hidden.shape
    out = torch.empty((rows, width * 2), device=hidden.device, dtype=torch.bfloat16)
    _norm_pair[(rows, 2)](
        tokens,
        embedding,
        hidden,
        embedding_norm,
        hidden_norm,
        out,
        width,
        hidden.stride(0),
        eps,
        tr.next_power_of_2(width),
        num_warps=4,
    )
    return out

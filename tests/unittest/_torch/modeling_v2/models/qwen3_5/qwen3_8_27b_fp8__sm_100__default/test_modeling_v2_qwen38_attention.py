# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy for short causal GQA and the Q/K and gating epilogues."""

from itertools import accumulate

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_fp8__sm_100__default.ops.attention import (  # noqa: E501
    attention_gate,
    prepare_qkv,
    short_causal_attention,
    short_paged_attention,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp  # noqa: E402

pytest_plugins = ("_validating",)


def _rand(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("cell", prepare_qkv.CELLS, ids=lambda cell: cell.why)
def test_qkv(cell, certifying):
    """Q/K RoPE and RMSNorm leave V bitwise unchanged."""
    assert torch.cuda.is_available()
    torch.manual_seed(17)
    rows = cell.spec["rows"]
    phase = torch.randn(rows + 10, 32, device="cuda")
    rope = torch.cat((phase.cos(), phase.sin()), dim=-1)
    positions = torch.arange(10, rows + 10, dtype=torch.int32, device="cuda")
    args = (_rand(rows, 14336), _rand(256), _rand(256), rope, positions, cell.spec["tiled"])
    certifying(prepare_qkv)
    q, k, v = prepare_qkv(*args)
    rq, rk, rv = prepare_qkv.reference(*args)
    assert_within_ulp(q, rq, 1.0, 0.5)
    assert_within_ulp(k, rk, 1.0, 0.5)
    torch.testing.assert_close(v, rv, atol=0, rtol=0)


@pytest.mark.parametrize("cell", attention_gate.CELLS, ids=lambda cell: cell.why)
def test_gate(cell, certifying):
    """Apply the interleaved gate of each query head to its own output."""
    assert torch.cuda.is_available()
    torch.manual_seed(17)
    rows = cell.spec["rows"]
    args = (_rand(rows, 6144), _rand(rows, 14336))
    certifying(attention_gate)
    assert_within_ulp(attention_gate(*args), attention_gate.reference(*args), 1.0, 0.5)


@pytest.mark.parametrize("cell", short_causal_attention.CELLS, ids=lambda cell: cell.why)
def test_short_attention(cell, certifying):
    """A packed request attends only to its own causal prefix."""
    assert torch.cuda.is_available()
    torch.manual_seed(17)
    lengths = cell.spec["lengths"]
    rows = sum(lengths)
    offsets = torch.tensor(list(accumulate(lengths, initial=0)), device="cuda", dtype=torch.int32)
    pairs = torch.tensor(
        [(seq, tile) for seq, length in enumerate(lengths) for tile in range((length + 63) // 64)],
        device="cuda",
        dtype=torch.int32,
    )
    args = (
        _rand(rows, 24, 256),
        _rand(rows, 4, 256),
        _rand(rows, 4, 256),
        _rand(rows, 14336),
        offsets,
        pairs,
    )
    certifying(short_causal_attention)
    assert_within_ulp(
        short_causal_attention(*args), short_causal_attention.reference(*args), 1.0, 0.5
    )


@pytest.mark.parametrize("cell", short_paged_attention.CELLS, ids=lambda cell: cell.why)
def test_short_paged_attention(cell, certifying):
    """Read layer-specific addresses and preserve unmodified long rows."""
    assert torch.cuda.is_available()
    torch.manual_seed(32)
    rows, contexts, tokens = (cell.spec[name] for name in ("rows", "contexts", "tokens"))
    q = _rand(rows, 24, 256)
    pages = _rand(40, 2, 4, 32, 256)[::2]
    table = torch.stack(
        [
            torch.randperm(20, device="cuda", dtype=torch.int32) * 2
            for _ in range(contexts + rows // tokens)
        ]
    )
    pos = [31, 42, 81] if rows == 3 else [66, 67, 400, 401]
    positions = torch.tensor(pos, dtype=torch.int32, device="cuda")
    output = _rand(rows, 6144)
    expect = output.clone()
    certifying(short_paged_attention)
    short_paged_attention(q, pages, table, positions, output, 2, contexts, tokens)
    short_paged_attention.reference(q, pages, table, positions, expect, 2, contexts, tokens)
    assert_within_ulp(output, expect, 2.0, 1.0)
    if rows == 4:
        torch.testing.assert_close(output[2:], expect[2:], atol=0, rtol=0)

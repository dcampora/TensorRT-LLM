# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 GDN tests with strided runtime-style state pools."""

from itertools import accumulate

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_fp8__sm_100__default.ops.recurrent import (  # noqa: E501
    causal_convolution,
    delta_gates,
    gated_head_norm,
    recurrent_delta,
    short_context_delta,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp  # noqa: E402

pytest_plugins = ("_validating",)


def _rand(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=dtype)


def _int(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("cell", causal_convolution.CELLS, ids=lambda cell: cell.why)
def test_convolution(cell, certifying):
    """Tail writes update exactly the selected runtime slots."""
    assert torch.cuda.is_available()
    torch.manual_seed(81)
    lengths = cell.spec["lengths"]
    slots = _int([5, 2, 7][: len(lengths)])
    projection = _rand(sum(lengths), 16384)
    weight = _rand(10240, 1, 4)
    state_base = _rand(16, 10240, 3)
    expected_base = state_base.clone()
    args = ()
    if not cell.spec["decode"]:
        args = (
            _int(list(accumulate(lengths, initial=0))),
            _int([seq for seq, length in enumerate(lengths) for _ in range(length)]),
            torch.tensor([False, True, True], device="cuda"),
        )
    certifying(causal_convolution)
    y = causal_convolution(projection, weight, state_base[::2], slots, *args)
    expected = causal_convolution.reference(projection, weight, expected_base[::2], slots, *args)
    assert_within_ulp(y, expected, 1.0, 0.5)
    torch.testing.assert_close(state_base, expected_base, atol=0, rtol=0)


@pytest.mark.parametrize("cell", delta_gates.CELLS, ids=lambda cell: cell.why)
def test_delta_gates(cell, certifying):
    """Log decay uses FP32 math and beta rounds only once."""
    assert torch.cuda.is_available()
    torch.manual_seed(10)
    args = (_rand(cell.spec["rows"], 96), _rand(48, dtype=torch.float32), _rand(48))
    certifying(delta_gates)
    decay, beta = delta_gates(*args)
    expected_decay, expected_beta = delta_gates.reference(*args)
    torch.testing.assert_close(decay, expected_decay, atol=1e-6, rtol=3e-6)
    torch.testing.assert_close(beta, expected_beta, atol=0, rtol=0)


@pytest.mark.parametrize("cell", recurrent_delta.CELLS, ids=lambda cell: cell.why)
def test_recurrent_delta(cell, certifying):
    """Decode writes the state of each real sequence and skips dummy slots."""
    assert torch.cuda.is_available()
    torch.manual_seed(16)
    rows = cell.spec["rows"]
    args = (_rand(rows, 10240), _rand(rows, 96), _rand(48, dtype=torch.float32), _rand(48))
    state = _rand(14, 48, 128, 128, dtype=torch.float32) * 0.01
    reference_state = state.clone()
    slots = _int([6, 2, -1, 3][:rows])
    certifying(recurrent_delta)
    y = recurrent_delta(*args, state[::2], slots)
    expected = recurrent_delta.reference(*args, reference_state[::2], slots)
    assert_within_ulp(y, expected, 1.0, 0.5)
    torch.testing.assert_close(state, reference_state, atol=1e-6, rtol=2e-5)


@pytest.mark.parametrize("cell", short_context_delta.CELLS, ids=lambda cell: cell.why)
def test_short_context_delta(cell, certifying):
    """The BF16 triangular solve approximates the sequential FP32 recurrence."""
    assert torch.cuda.is_available()
    torch.manual_seed(16)
    lengths = cell.spec["lengths"]
    counts = [(n + 63) // 64 for n in lengths]
    rows = sum(lengths)
    qkv = _rand(rows, 10240)
    decay = -_rand(rows, 48, dtype=torch.float32).abs()
    beta = _rand(rows, 48).float().sigmoid().bfloat16().float()
    offsets = _int(list(accumulate(lengths, initial=0)))
    chunks = torch.tensor(
        [(seq, i) for seq, count in enumerate(counts) for i in range(count)],
        device="cuda",
        dtype=torch.int32,
    )
    starts = _int(list(accumulate(counts, initial=0)))
    state = torch.zeros(14, 48, 128, 128, device="cuda")
    reference_state = state.clone()
    slots = _int([6, 2][: len(lengths)])
    args = (qkv, decay, beta, offsets, chunks, starts, max(lengths))
    certifying(short_context_delta)
    y = short_context_delta(*args, state[::2], slots)
    expected = short_context_delta.reference(*args, reference_state[::2], slots)
    for actual, reference in ((y, expected), (state, reference_state)):
        error = (actual.float() - reference.float()).norm() / reference.float().norm()
        assert error < 0.009
    assert state[1::2].count_nonzero() == 0


@pytest.mark.parametrize("cell", gated_head_norm.CELLS, ids=lambda cell: cell.why)
def test_gated_head_norm(cell, certifying):
    """The GDN output normalization uses the checkpoint's SiLU gate."""
    assert torch.cuda.is_available()
    torch.manual_seed(123)
    rows = cell.spec["rows"]
    args = (_rand(rows, 48, 128), _rand(rows, 16384), _rand(128))
    certifying(gated_head_norm)
    assert_within_ulp(gated_head_norm(*args), gated_head_norm.reference(*args), 1.0, 0.5)

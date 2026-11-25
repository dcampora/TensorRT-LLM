# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDN speculative verify and replay over mixed cache slots."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_fp8__sm_100__default.ops.speculative import (  # noqa: E501
    ReplayHistory,
    reset_replay,
    verify_convolution,
    verify_delta,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp  # noqa: E402

pytest_plugins = ("_validating",)


def _rand(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=dtype)


def _int(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, device="cuda", dtype=torch.int32)


@pytest.mark.parametrize("cell", verify_convolution.CELLS, ids=lambda cell: cell.why)
def test_verify_convolution(cell, certifying):
    """Verify writes each tentative window and leaves accepted history intact."""
    assert torch.cuda.is_available()
    torch.manual_seed(59)
    seqs, width = (cell.spec[k] for k in ("seqs", "tokens"))
    x = _rand(seqs * width, 16384)
    weight = _rand(10240, 1, 4)
    conv = _rand(2 * seqs + 4, 10240, 3)[::2]
    windows = _rand(2 * seqs + 3, width + 1, 10240, 3)
    expected = windows.clone()
    frozen = conv.clone()
    slots = _int(list(range(seqs, 0, -1)))
    destinations = _int(list(range(1, seqs + 1)))
    certifying(verify_convolution)
    y = verify_convolution(x, weight, conv, windows, slots, destinations, width)
    ref = verify_convolution.reference(x, weight, conv, expected, slots, destinations, width)
    assert_within_ulp(y, ref, 1.0, 0.5)
    torch.testing.assert_close(windows, expected, atol=0, rtol=0)
    torch.testing.assert_close(conv, frozen, atol=0, rtol=0)


@pytest.mark.parametrize("cell", verify_delta.CELLS, ids=lambda cell: cell.why)
def test_verify_delta(cell, certifying):
    """The accepted first row commits and the tentative suffix stays in its own bank."""
    assert torch.cuda.is_available()
    torch.manual_seed(53)
    steps, past = (cell.spec[k] for k in ("tokens", "old"))
    x, ba = _rand(2 * steps, 10240), _rand(2 * steps, 96)
    a_log, dt = _rand(48, dtype=torch.float32), _rand(48)
    slots, replay_slots = _int([2, 1]), _int([4, 2])
    state = _rand(8, 48, 128, 128, dtype=torch.float32)[::2].mul_(0.02)
    expected_state = state.clone()
    meta = torch.zeros(5, 4, device="cuda", dtype=torch.int32)
    meta[:, 0] = past
    meta[2, 1] = 1
    history = ReplayHistory(
        meta[:, 0],
        meta[:, 1],
        (_rand(10, 2, 8, 48, 128, dtype=torch.float32)[::2].mul_(0.05))
        .view(torch.bfloat16)
        .view(5, 2, 16, 48, 128),
        (_rand(10, 2, 8, 16, 128, dtype=torch.float32)[::2].mul_(0.05))
        .view(torch.bfloat16)
        .view(5, 2, 16, 16, 128),
        _rand(10, 2, 48, 16, dtype=torch.float32).sigmoid_()[::2],
    )
    expected = ReplayHistory(*(t.clone() for t in history))
    certifying(verify_delta)
    y = verify_delta(x, ba, a_log, dt, state, slots, replay_slots, history, steps)
    ref = verify_delta.reference(
        x, ba, a_log, dt, expected_state, slots, replay_slots, expected, steps
    )
    assert_within_ulp(y, ref, 1.0, 0.5)
    torch.testing.assert_close(state, expected_state, atol=3e-6, rtol=2e-5)
    torch.testing.assert_close(
        history.value.view(torch.float32), expected.value.view(torch.float32), atol=3e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        history.key.view(torch.float32), expected.key.view(torch.float32), atol=3e-6, rtol=2e-5
    )
    torch.testing.assert_close(history.decay, expected.decay, atol=3e-6, rtol=2e-5)


@pytest.mark.parametrize("cell", reset_replay.CELLS, ids=lambda cell: cell.why)
def test_reset_replay(cell, certifying):
    """Zero metadata of the selected slots without changing neighbor records."""
    assert torch.cuda.is_available()
    slots = _int(list(cell.spec["slots"]))
    actual = torch.arange(28, device="cuda", dtype=torch.int32).reshape(7, 4)
    expected = actual.clone()
    certifying(reset_replay)
    reset_replay(slots, actual[:, 0], actual[:, 1])
    reset_replay.reference(slots, expected[:, 0], expected[:, 1])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

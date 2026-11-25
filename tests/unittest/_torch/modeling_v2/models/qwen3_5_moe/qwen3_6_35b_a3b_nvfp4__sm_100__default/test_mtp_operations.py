# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: F811
"""MTP proposal and recurrent verification kernel tests."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_nvfp4__sm_100__default.ops import (  # noqa: E501
    draft,
    verify,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402, F401

assert torch.cuda.is_available(), "Qwen3.6 MTP operation cells require CUDA"
DEV = "cuda"
BF16 = torch.bfloat16


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(397)


def rand(shape, dtype=BF16, scale=1.0):
    return (torch.randn(shape, device=DEV) * scale).to(dtype)


@pytest.mark.parametrize("cell", verify.convolution_verify.CELLS)
def test_convolution_verify(cell, certifying):
    op = verify.convolution_verify
    certifying(op)
    batch, width, nslots = (cell.spec[k] for k in ("batch", "width", "slots"))
    x = rand((batch * width, 12288))
    weight = rand((8192, 4))
    slots = torch.randperm(nslots, device=DEV, dtype=torch.int32)[:batch]
    history = rand((nslots, 2, 3, 8192))[:, 1].transpose(-1, -2)
    windows = torch.zeros((batch + 2, 4, 8192, 3), dtype=BF16, device=DEV)
    reference = windows.clone()
    actual = op(x, weight, slots, history, windows, width)
    expected = op.reference(x, weight, slots, history, reference, width)
    assert_within_ulp(actual, expected, 1, 1)
    torch.testing.assert_close(windows, reference, atol=0, rtol=0)


@pytest.mark.parametrize("cell", verify.delta_verify.CELLS)
def test_delta_verify(cell, certifying):
    op = verify.delta_verify
    certifying(op)
    batch, width = cell.spec["batch"], cell.spec["width"]
    fused = cell.spec.get("fused", False)
    qkv = rand((batch * width, 12288 if fused else 8192))
    ba = rand((batch * width, 64))
    a_log, dt = rand((32,), torch.float32), rand((32,))
    nslots = 2 * batch + 7
    slots = torch.arange(batch, device=DEV, dtype=torch.int32) * 2
    replay_slots = nslots - slots - 1
    pool = rand((nslots, 2, 32, 128, 128), torch.float32, 0.05)
    state, ref_state = pool[:, 1], pool.clone()[:, 1]
    delta = rand((nslots, 2, 16, 32, 128), scale=0.06)
    key = rand((nslots, 2, 16, 16, 128), scale=0.08)
    decay = -rand((nslots, 2, 32, 16), torch.float32, 0.1).abs().cumsum(-1)
    accepted = torch.zeros((nslots, 4), device=DEV, dtype=torch.int32)[:, 1]
    accepted[replay_slots.long()] = torch.tensor(
        cell.spec["previous"], device=DEV, dtype=torch.int32
    )
    buffer = torch.ones((nslots, 4), device=DEV, dtype=torch.int32)[:, 2]
    expected_buffers = delta.clone(), key.clone(), decay.clone()
    args = qkv, ba, a_log, dt, slots, replay_slots
    conv = ref_conv = None
    if fused:
        weight = rand((8192, 4))
        history = rand((nslots, 2, 3, 8192))[:, 1].transpose(-1, -2)
        windows = torch.zeros((batch + 2, 4, 8192, 3), dtype=BF16, device=DEV)
        conv = weight, history, windows
        ref_conv = weight, history, windows.clone()
    actual = op(*args, state, delta, key, decay, accepted, buffer, width, 4, conv)
    expected = op.reference(
        *args, ref_state, *expected_buffers, accepted, buffer, width, 4, ref_conv
    )
    # The tiled FP32 reduction moves one of 524,288 BF16 values by
    # 1.47 row-normalized ULPs relative to the Torch recurrence.
    assert_within_ulp(actual, expected, 2 if batch >= 32 else 1, 1)
    if conv is not None:
        torch.testing.assert_close(conv[-1], ref_conv[-1], atol=0, rtol=0)
    torch.testing.assert_close(state, ref_state, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(decay, expected_buffers[2], atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(delta, expected_buffers[0], atol=1e-4, rtol=0.008)
    torch.testing.assert_close(key, expected_buffers[1], atol=1e-4, rtol=0.008)


@pytest.mark.parametrize("cell", draft.draft_input_norm.CELLS)
def test_draft_input_norm(cell, certifying):
    op = draft.draft_input_norm
    certifying(op)
    rows = cell.spec["rows"]
    ids = torch.randint(31, (rows,), dtype=torch.int32, device=DEV)
    args = ids, rand((31, 2048)), rand((rows, 2048)), rand((2048,)), rand((2048,))
    assert_within_ulp(op(*args), op.reference(*args), 1, 1)


@pytest.mark.parametrize("cell", draft.draft_experts.CELLS)
def test_draft_experts(cell, certifying):
    op = draft.draft_experts
    certifying(op)
    rows = cell.spec["rows"]
    ids = (torch.rand((rows, 16), device=DEV).argsort(dim=-1)[:, :8]).int().contiguous()
    p = torch.rand((rows, 8), device=DEV).softmax(-1)
    bins = torch.arange(257, dtype=torch.int32, device=DEV)
    args = (
        rand((rows, 2048)),
        rand((16, 1024, 2048), scale=0.025),
        rand((16, 2048, 512), scale=0.025),
        ids,
        p,
        bins,
    )
    assert_within_ulp(op(*args), op.reference(*args), 3, 1)

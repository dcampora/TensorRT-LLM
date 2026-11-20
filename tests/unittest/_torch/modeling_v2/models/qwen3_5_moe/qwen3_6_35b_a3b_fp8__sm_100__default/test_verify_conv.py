# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.verify_conv import (  # noqa: E501
    verify_conv,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "Qwen3.6 FP8 verification tests require CUDA"


@pytest.mark.parametrize("cell", verify_conv.CELLS, ids=lambda cell: cell.why)
def test_verify_conv(cell, certifying):  # noqa: F811
    certifying(verify_conv)
    requests, tokens = cell.spec["requests"], cell.spec["tokens"]
    channels = 8192
    rng = torch.Generator(device="cuda").manual_seed(73)
    options = dict(device="cuda", dtype=torch.bfloat16, generator=rng)
    # Match the projection's QKV slice and the runtime's channel-major pool view.
    qkv = torch.randn(requests * tokens, channels + 4096, **options)[:, :channels]
    weight = torch.randn(channels, 4, **options)
    state = torch.randn(requests + 5, 3, channels, **options).transpose(1, 2)
    slots = torch.arange(requests, device="cuda", dtype=torch.int32).flip(0) + 2
    if requests > 1:
        slots[0] = -1
    actual_states = torch.zeros(requests, tokens, channels, 3, device="cuda", dtype=qkv.dtype)
    reference_states = torch.zeros_like(actual_states)
    result = verify_conv(qkv, weight, state, slots, actual_states, tokens)
    reference = verify_conv.reference(qkv, weight, state, slots, reference_states, tokens)
    assert_within_ulp(result, reference, element_ulp=2, rms_ulp=0.4)
    torch.testing.assert_close(actual_states, reference_states, atol=0, rtol=0)

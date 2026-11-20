# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.fp8_low_m import (  # noqa: E501
    block_scaled_low_m,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "Qwen3.6 FP8 low-M tests require CUDA"


@pytest.mark.parametrize("cell", block_scaled_low_m.CELLS, ids=lambda cell: cell.why)
def test_block_scaled_low_m(cell, certifying):  # noqa: F811
    certifying(block_scaled_low_m)
    m, n, k = (cell.spec[name] for name in ("m", "n", "k"))
    rng = torch.Generator(device="cuda").manual_seed(63)
    opts = dict(device="cuda", generator=rng)
    data = torch.randn(m, k, dtype=torch.bfloat16, **opts)
    x, x_scale = torch.ops.trtllm.fp8_quantize_1x128(data)
    weight = torch.randn(n, k, dtype=torch.bfloat16, **opts).to(torch.float8_e4m3fn)
    weight_scale = torch.rand(n // 128, k // 128, **opts) * 0.1 + 0.01
    args = x, x_scale, weight, weight_scale
    actual = block_scaled_low_m(*args)
    expected = block_scaled_low_m.reference(*args)
    assert_within_ulp(actual, expected, element_ulp=2, rms_ulp=0.1)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.epilogue_quantize import (  # noqa: E501
    gated_quantize,
)

__extra_import_path__ = ["../../.."]
from _validating import certifying  # noqa: E402,F401


@pytest.mark.parametrize("cell", gated_quantize.CELLS, ids=lambda cell: cell.why)
def test_gated_quantize(cell, certifying):  # noqa: F811
    certifying(gated_quantize)
    kind, rows = cell.spec["kind"], cell.spec["rows"]
    rng = torch.Generator(device="cuda").manual_seed(177)
    options = dict(device="cuda", dtype=torch.bfloat16, generator=rng)
    width = 1024 if kind == "swiglu" else 4096
    x = torch.randn(rows, width, **options)
    gate = None if kind == "swiglu" else torch.randn(rows, 2 * width, **options)[:, width:]
    weight = torch.randn(128, **options) if kind == "gdn" else None
    pick_rows = torch.arange(rows - 1, 0, -3, device="cuda") if rows == 96 else None
    args = x, kind, gate, weight, 1e-6, pick_rows
    actual, scales = gated_quantize(*args)
    expected, expected_scales = gated_quantize.reference(*args)
    assert (actual.float() == expected.float()).float().mean() >= 0.9998
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0.008)

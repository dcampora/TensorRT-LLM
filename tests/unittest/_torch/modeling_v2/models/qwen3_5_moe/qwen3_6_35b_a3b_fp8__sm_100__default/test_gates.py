# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.gates import (  # noqa: E501
    gated_rms_norm,
    sigmoid_gate,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "Qwen3.6 FP8 gate tests require CUDA"


@pytest.mark.parametrize("cell", sigmoid_gate.CELLS, ids=lambda cell: cell.why)
def test_sigmoid_gate(cell, certifying):  # noqa: F811
    certifying(sigmoid_gate)
    rows, width, shared = (cell.spec[key] for key in ("rows", "width", "shared"))
    rng = torch.Generator(device="cuda").manual_seed(42)
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16, generator=rng)
    gate_width = 1 if shared else width
    gate = torch.randn(rows, 2 * width + 1024, device="cuda", dtype=torch.bfloat16, generator=rng)[
        :, -gate_width:
    ]
    addend = torch.randn_like(x) if shared else None
    actual = sigmoid_gate(x, gate, addend)
    reference = sigmoid_gate.reference(x, gate, addend)
    assert_within_ulp(actual, reference, element_ulp=1, rms_ulp=0.1)


@pytest.mark.parametrize("cell", gated_rms_norm.CELLS, ids=lambda cell: cell.why)
def test_gated_rms_norm(cell, certifying):  # noqa: F811
    certifying(gated_rms_norm)
    rows, heads, width = (cell.spec[key] for key in ("rows", "heads", "width"))
    rng = torch.Generator(device="cuda").manual_seed(42)
    options = dict(device="cuda", dtype=torch.bfloat16, generator=rng)
    x = torch.randn(rows, heads, width, **options)
    z = torch.randn(rows, 3 * heads, width, **options)[:, -heads:]
    weight = torch.randn(width, **options)
    actual = gated_rms_norm(x, z, weight, 1e-6)
    reference = gated_rms_norm.reference(x, z, weight, 1e-6)
    assert_within_ulp(actual, reference, element_ulp=1, rms_ulp=0.1)

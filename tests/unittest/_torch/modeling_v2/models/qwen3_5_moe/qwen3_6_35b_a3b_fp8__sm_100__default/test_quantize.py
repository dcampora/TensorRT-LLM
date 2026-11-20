# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.quantize import (  # noqa: E501
    add_rmsnorm_quant,
    block_quantize,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402,F401

assert torch.cuda.is_available(), "Qwen3.6 FP8 quantization tests require CUDA"


@pytest.mark.parametrize("cell", block_quantize.CELLS, ids=lambda cell: cell.why)
def test_block_quantize(cell, certifying):  # noqa: F811
    certifying(block_quantize)
    rows, width = cell.spec["rows"], cell.spec["width"]
    rng = torch.Generator(device="cuda").manual_seed(71)
    x = torch.randn(rows, width, device="cuda", generator=rng, dtype=torch.bfloat16)
    x[0, :128] = 0
    actual, scales = block_quantize(x)
    reference, ref_scales = block_quantize.reference(x)
    torch.testing.assert_close(actual.float(), reference.float(), atol=0, rtol=0)
    torch.testing.assert_close(scales, ref_scales, atol=0, rtol=2e-7)
    assert scales.shape == (width // 128, rows)
    assert scales.is_contiguous()


@pytest.mark.parametrize("cell", add_rmsnorm_quant.CELLS, ids=lambda cell: cell.why)
def test_add_rmsnorm_quant(cell, certifying):  # noqa: F811
    certifying(add_rmsnorm_quant)
    rows = cell.spec["rows"]
    rng = torch.Generator(device="cuda").manual_seed(17)
    options = dict(device="cuda", dtype=torch.bfloat16, generator=rng)
    hidden = torch.randn(rows, 2048, **options)
    residual = torch.randn(rows, 2048, **options)
    weight = torch.randn(2048, **options) * 0.2
    ref_hidden, ref_residual = hidden.clone(), residual.clone()
    actual, scales = add_rmsnorm_quant(hidden, residual, weight, 1e-6)
    reference, ref_scales = add_rmsnorm_quant.reference(ref_hidden, ref_residual, weight, 1e-6)
    torch.testing.assert_close(residual, ref_residual, atol=0, rtol=0)
    assert_within_ulp(hidden, ref_hidden, element_ulp=1, rms_ulp=0.05)
    assert (actual.float() == reference.float()).float().mean() >= 0.9999
    torch.testing.assert_close(scales, ref_scales, atol=0, rtol=0.008)

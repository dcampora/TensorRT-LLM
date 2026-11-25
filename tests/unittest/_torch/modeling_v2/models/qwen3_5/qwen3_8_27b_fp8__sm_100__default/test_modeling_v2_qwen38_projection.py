# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM100 Qwen3.8 block-scaled quantization and GEMM tests."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_fp8__sm_100__default.ops.gates import (  # noqa: E501
    fixed_gate_projection,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_fp8__sm_100__default.ops.projection import (  # noqa: E501
    block_matmul,
    block_quantize,
    block_requantize,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp  # noqa: E402

pytest_plugins = ("_validating",)


@pytest.mark.parametrize("cell", block_quantize.CELLS, ids=lambda cell: cell.why)
def test_quantize(cell, certifying):
    """Activation scales and E4M3 values agree including graph padding rows."""
    assert torch.cuda.is_available()
    spec = cell.spec
    generator = torch.Generator(device="cuda").manual_seed(123)
    x = torch.randn(spec["rows"], spec["width"], device="cuda", generator=generator).bfloat16()
    certifying(block_quantize)
    a, sa = block_quantize(x)
    expected_a, expected_sa = block_quantize.reference(x)
    torch.testing.assert_close(sa, expected_sa, atol=0, rtol=0)
    torch.testing.assert_close(a.float(), expected_a.float(), atol=0, rtol=0)


@pytest.mark.parametrize("cell", block_matmul.CELLS, ids=lambda cell: cell.why)
def test_matmul(cell, certifying):
    """The tensor-core sum of E4M3 block products rounds to BF16."""
    assert torch.cuda.is_available()
    rows, n, k = (cell.spec[name] for name in ("rows", "n", "k"))
    generator = torch.Generator(device="cuda").manual_seed(19)
    x = torch.randn(rows, k, device="cuda", generator=generator).bfloat16()
    w = (torch.randn(n, k, device="cuda", generator=generator) * 48).to(torch.float8_e4m3fn)
    sw = torch.exp2(
        torch.randint(-6, -1, (n // 128, k // 128), device="cuda", generator=generator).float()
    )
    a, sa = block_quantize(x)
    scale_mn = sw.T.contiguous()
    workspace = None
    if cell.spec.get("tiled"):
        workspace = torch.empty(16 << 20, device="cuda", dtype=torch.uint8)
        if n == 5120:
            scale_mn = scale_mn.repeat_interleave(2, dim=1)
    args = (a, sa, w, sw, scale_mn, rows, rows == 1, workspace)
    certifying(block_matmul)
    assert_within_ulp(block_matmul(*args), block_matmul.reference(*args), 1.0, 0.5)


@pytest.mark.parametrize("cell", block_requantize.CELLS, ids=lambda cell: cell.why)
def test_requantize(cell, certifying):
    """Requantization reproduces the FP8 representation of the BF16 rounding point."""
    assert torch.cuda.is_available()
    n, k = cell.spec["n"], cell.spec["k"]
    generator = torch.Generator(device="cuda").manual_seed(7)
    w = (torch.randn(n, k, device="cuda", generator=generator) * 48).to(torch.float8_e4m3fn)
    scales = torch.rand(n // 128, k // 128, device="cuda", generator=generator).bfloat16() / 10
    certifying(block_requantize)
    converted, converted_scales = block_requantize(w, scales)
    expected, expected_scales = block_requantize.reference(w, scales)
    torch.testing.assert_close(converted_scales, expected_scales, atol=0, rtol=0)
    torch.testing.assert_close(converted.float(), expected.float(), atol=0, rtol=0)


@pytest.mark.parametrize("cell", fixed_gate_projection.CELLS, ids=lambda cell: cell.why)
def test_gate_projection(cell, certifying):
    """The pinned BF16 kernel uses the same reduction on every tested width."""
    assert torch.cuda.is_available()
    torch.manual_seed(7)
    rows = cell.spec["rows"]
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(96, 5120, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(16 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    certifying(fixed_gate_projection)
    output = fixed_gate_projection(x, w, workspace)
    expect = fixed_gate_projection.reference(x, w, workspace)
    assert_within_ulp(output, expect, 1.0, 0.5)
    torch.testing.assert_close(
        output[:1], fixed_gate_projection(x[:1], w, workspace), atol=0, rtol=0
    )

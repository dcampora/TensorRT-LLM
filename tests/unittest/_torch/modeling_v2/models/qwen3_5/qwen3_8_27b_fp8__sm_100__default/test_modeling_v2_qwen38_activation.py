# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8 fused BF16 producers and FP8 layouts."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_fp8__sm_100__default.ops.activation import (  # noqa: E501
    attention_gate_quantize,
    draft_input_norm,
    gdn_norm_quantize,
    residual_norm_quantize,
    swiglu_quantize,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp  # noqa: E402

pytest_plugins = ("_validating",)


def _rand(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16)


def _assert_quantized(got: tuple[torch.Tensor, ...], want: tuple[torch.Tensor, ...]) -> None:
    a, scales = got
    expected, expected_scales = want
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
    # Downstream GEMM sees at most one FP8 rounding interval from the independent FP32 oracle.
    assert_within_ulp(a, expected, 1.0, 0.05, ulp=2.0**-3)


@pytest.mark.parametrize("cell", swiglu_quantize.CELLS, ids=lambda cell: cell.why)
def test_swiglu_quantize(cell, certifying):
    """Preserve the BF16 rounding before quantizing the SwiGLU activation."""
    torch.manual_seed(73)
    x = _rand(cell.spec["rows"], 34816)
    certifying(swiglu_quantize)
    packed = cell.spec.get("packed", False)
    _assert_quantized(swiglu_quantize(x, packed), swiglu_quantize.reference(x, packed))


@pytest.mark.parametrize("cell", gdn_norm_quantize.CELLS, ids=lambda cell: cell.why)
def test_gdn_norm_quantize(cell, certifying):
    """One FP8 group per head, including padded and grouped-head launches."""
    torch.manual_seed(77)
    rows = cell.spec["rows"]
    args = (_rand(rows, 48, 128), _rand(rows, 16384), _rand(128))
    certifying(gdn_norm_quantize)
    packed = cell.spec.get("packed", False)
    _assert_quantized(gdn_norm_quantize(*args, packed), gdn_norm_quantize.reference(*args, packed))


@pytest.mark.parametrize("cell", attention_gate_quantize.CELLS, ids=lambda cell: cell.why)
def test_attention_gate_quantize(cell, certifying):
    """The FP8 projection input reads each query's own interleaved output gate."""
    torch.manual_seed(79)
    rows = cell.spec["rows"]
    args = (_rand(rows, 6144), _rand(rows, 14336))
    certifying(attention_gate_quantize)
    packed = cell.spec.get("packed", False)
    _assert_quantized(
        attention_gate_quantize(*args, packed), attention_gate_quantize.reference(*args, packed)
    )


@pytest.mark.parametrize("cell", residual_norm_quantize.CELLS, ids=lambda cell: cell.why)
def test_residual_norm_quantize(cell, certifying):
    """Check both the quantized activation and the mutated BF16 residual."""
    torch.manual_seed(83)
    rows, normalized = (cell.spec[k] for k in ("rows", "normalized"))
    x, residual, weight = _rand(rows, 5120), _rand(rows, 5120), _rand(5120) * 0.1
    x_ref, residual_ref = x.clone(), residual.clone()
    certifying(residual_norm_quantize)
    programmatic = rows <= 8
    packed = cell.spec.get("packed", False)
    _assert_quantized(
        residual_norm_quantize(x, residual, weight, normalized, programmatic, packed),
        residual_norm_quantize.reference(
            x_ref, residual_ref, weight, normalized, programmatic, packed
        ),
    )
    torch.testing.assert_close(residual, residual_ref, atol=0, rtol=0)
    assert_within_ulp(x, x_ref, 1.0, 0.25)


@pytest.mark.parametrize("cell", draft_input_norm.CELLS, ids=lambda cell: cell.why)
def test_draft_input_norm(cell, certifying):
    """Gather embeddings and write both BF16 norms in the FC input order."""
    torch.manual_seed(89)
    rows, wide = (cell.spec[k] for k in ("rows", "wide"))
    ids = torch.randint(512, (rows,), device="cuda", dtype=torch.int64 if wide else torch.int32)
    args = (ids, _rand(512, 5120), _rand(rows, 5120), _rand(5120) * 0.1, _rand(5120) * 0.1)
    certifying(draft_input_norm)
    assert_within_ulp(draft_input_norm(*args), draft_input_norm.reference(*args), 1.0, 0.25)

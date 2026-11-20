# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_fp8__sm_100__default.ops.gdn_pack import (  # noqa: E501
    pack_gdn_input,
)

__extra_import_path__ = ["../../.."]
from _validating import certifying  # noqa: E402,F401


@pytest.mark.parametrize("cell", pack_gdn_input.CELLS, ids=lambda cell: cell.why)
def test_pack_gdn_input(cell, certifying):  # noqa: F811
    certifying(pack_gdn_input)
    rows = cell.spec["rows"]
    rng = torch.Generator(device="cuda").manual_seed(221)
    options = dict(device="cuda", dtype=torch.bfloat16, generator=rng)
    qkv = torch.randn(rows, 12288, **options)[:, :8192]
    gates = torch.randn(rows, 64, **options)
    a, b = gates.chunk(2, dim=1)
    actual = pack_gdn_input(qkv, a, b, 16, 32, 128)
    expected = pack_gdn_input.reference(qkv, a, b, 16, 32, 128)
    for data, ref in zip(actual, expected, strict=True):
        assert data.is_contiguous()
        torch.testing.assert_close(data, ref, atol=0, rtol=0)

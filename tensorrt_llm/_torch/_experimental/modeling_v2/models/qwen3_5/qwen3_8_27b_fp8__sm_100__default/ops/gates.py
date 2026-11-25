# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch-invariant GDN gate projection."""

import torch

from .....catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from ..kernels.gate_projection import gate_projection


class FixedGateProjection(OpWrapper):
    """Use the same BF16 × BF16, five-split FP32 reduction across decode widths."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = tuple(
        Cell(f"{rows} request-major decode rows", dict(rows=rows)) for rows in (1, 32, 128, 256)
    )

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, workspace: torch.Tensor
    ) -> torch.Tensor:
        return gate_projection(x, weight, workspace)

    def reference(
        self, x: torch.Tensor, weight: torch.Tensor, workspace: torch.Tensor
    ) -> torch.Tensor:
        with true_fp32_matmul():
            return (x.float() @ weight.float().T).bfloat16()


fixed_gate_projection = FixedGateProjection()

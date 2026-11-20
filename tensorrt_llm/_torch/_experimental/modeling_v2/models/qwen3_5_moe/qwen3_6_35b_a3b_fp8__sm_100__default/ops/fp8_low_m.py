# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-sized projections with the original FP8 weight block scales."""

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels.fp8_low_m import low_m_gemm

_TILES = {
    (2048, 4096): (64, 4),
    (1024, 2048): (64, 2),
    (2048, 512): (32, 4),
}


class BlockScaledLowM(OpWrapper):
    """Multiply small batches of FP8 rows by a block-scaled projection."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(why="one shared expert input row", spec=dict(m=1, n=1024, k=2048)),
        Cell(why="one request verifying four rows", spec=dict(m=4, n=2048, k=4096)),
        Cell(why="shared expert down projection for eight rows", spec=dict(m=8, n=2048, k=512)),
    )

    def supports(self, x, weight):
        return 1 <= x.shape[0] <= 8 and tuple(weight.shape) in _TILES

    def __call__(self, x, x_scale, weight, weight_scale):
        cols, groups = _TILES[tuple(weight.shape)]
        return low_m_gemm(x, x_scale, weight, weight_scale, cols, groups)

    def reference(self, x, x_scale, weight, weight_scale):
        x = x.double() * x_scale.T.repeat_interleave(128, dim=1)
        weight = weight.double() * weight_scale.repeat_interleave(128, dim=0).repeat_interleave(
            128, dim=1
        )
        return (x @ weight.T).bfloat16()


block_scaled_low_m = BlockScaledLowM()

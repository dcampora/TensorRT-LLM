# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wide FFN projection adapted from aegis-engines fp8_wide.py (114cf57c911e)."""

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait
from triton.tools.tensor_descriptor import TensorDescriptor


@tr.jit
def _wide_projection(
    A, WEIGHT, A_SCALE, W_SCALE, OUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr
):
    gdc_launch_dependents()
    gdc_wait()
    tile = tl.program_id(0)
    columns = tile * 128 + tl.arange(0, 128)
    rows = tl.arange(0, 16)
    channels = tl.arange(0, 128)
    accumulator = tl.full((128, 16), 0.0, tl.float32)
    for group in tl.range(K // 128, num_stages=5):
        weight = WEIGHT.load([tile * 128, group * 128])
        activation = tl.load(
            A + rows[:, None] * K + group * 128 + channels[None, :],
            rows[:, None] < M,
            0.0,
        )
        a_scale = tl.load(A_SCALE + group * M + rows, rows < M, 0.0)
        w_scale = tl.load(W_SCALE + tile * (K // 128) + group).to(tl.float32)
        accumulator = tl.fma(tl.dot(weight, activation.T), w_scale * a_scale[None, :], accumulator)
    tl.store(OUT + rows[None, :] * N + columns[:, None], accumulator, rows[None, :] < M)


def project(
    activations: torch.Tensor,
    activation_scales: torch.Tensor,
    weight: TensorDescriptor,
    weight_scales: torch.Tensor,
) -> torch.Tensor:
    """Load row-major E4M3 tiles through TMA and write one BF16 projection."""
    rows, width = activations.shape
    columns = weight.shape[0]
    out = activations.new_empty((rows, columns), dtype=torch.bfloat16)
    _wide_projection[(columns // 128,)](
        activations,
        weight,
        activation_scales,
        weight_scales,
        out,
        rows,
        columns,
        width,
        num_warps=4,
        launch_pdl=True,
    )
    return out

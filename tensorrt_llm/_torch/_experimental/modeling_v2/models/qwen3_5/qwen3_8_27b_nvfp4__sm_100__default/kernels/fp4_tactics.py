# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from aegis-engines cb6779a, fp4_tactics.py.
"""Decode down projection with the same two K partitions at every width."""

from functools import lru_cache

import cutlass
import torch
from flashinfer.gemm.gemm_base import _cute_dsl_gemm_fp4_runner
from flashinfer.gemm.gemm_mm_fp4_cute_dsl import _compile_block_scaled_gemm
from flashinfer.gemm.kernels.dense_blockscaled_gemm_sm100_splitk import (
    Sm100BlockScaledSplitKGemmKernel,
)

PREFILL_TACTICS = {
    (128, 5120, 34816): ((256, 128), (2, 1), True, False, "sm100", None),
    (128, 17408, 5120): ((256, 64), (2, 1), True, False, "sm100", None),
    (1024, 5120, 34816): ((256, 192), (2, 1), False, False, "sm100", None),
    (4096, 17408, 5120): ((256, 192), (2, 1), False, False, "sm100", None),
}


def prefill_gemm(x, weight, scales, weight_scales, alpha, tactic):
    dtype = torch.bfloat16
    out = torch.empty((x.shape[0], weight.shape[1]), dtype=dtype, device=x.device)
    runner = _cute_dsl_gemm_fp4_runner(10, 0, True, dtype, True)
    inputs = [x, weight, scales, weight_scales, alpha, dtype, out, 16, True, None]
    return runner.forward(inputs, tactic)


class WideSplitK(Sm100BlockScaledSplitKGemmKernel):
    SUPPORTED_MMA_TILER_MN = (
        *Sm100BlockScaledSplitKGemmKernel.SUPPORTED_MMA_TILER_MN,
        (128, 64),
        (128, 128),
    )


@lru_cache(maxsize=None)
def compile_down(tile, scale_m, scale_n, scale_k):
    shape = (128, tile)
    key = (16, shape, (1, 1), True, False, "sm100sk", 2, True, torch.bfloat16, None, None)
    gemm, _ = _compile_block_scaled_gemm(
        {},
        key,
        lambda: WideSplitK(16, shape, 2),
        ab_cutlass_dtype=cutlass.Uint8,
        sf_dtype=cutlass.Float8E4M3FN,
        c_cutlass_dtype=cutlass.BFloat16,
        ab_assumed_align=32,
        cluster_shape_mn=(1, 1),
        swap_ab=True,
        sf_m=scale_m,
        sf_n=scale_n,
        sf_k=scale_k,
        batch_size=1,
        cluster_shape_k=2,
        cache_module_name="qwen_fp4_down",
    )
    return gemm


def split_down(x, weight, scales, weight_scales, alpha):
    m, n = x.shape[0], weight.shape[1]
    scale_m, scale_n = (n + 127) // 128, (m + 127) // 128
    scale_k = (2 * x.shape[1] // 16 + 3) // 4
    tile = 64 if m <= 64 else 128
    gemm = compile_down(tile, scale_m, scale_n, scale_k)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    gemm(
        weight.T,
        x,
        out.as_strided(out.shape, (1, m)),
        scale_m,
        scale_n,
        scale_k,
        weight_scales.data_ptr(),
        scales.data_ptr(),
        alpha,
    )
    return out

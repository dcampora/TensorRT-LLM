# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Block-scaled FP8 GEMM using the existing Blackwell kernel interfaces."""

import torch

from tensorrt_llm import deep_gemm

from .....catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from ..kernels.packed_scales import pack_activation_deepgemm, pack_activation_mx


class AcceleratedBlockMatmul(OpWrapper):
    """Project the same FP8 codes with exponents packed for Tensor Core scaling."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            "CuTe SM100 output projection for a B1 verifier",
            dict(rows=5, n=5120, k=6144, small=True),
        ),
        Cell("CuTe SM100 FFN down projection", dict(rows=32, n=5120, k=17408, small=True)),
        Cell(
            "DeepGEMM packed QKVZ at moderate batch", dict(rows=128, n=16384, k=5120, small=False)
        ),
        Cell(
            "DeepGEMM FFN down projection for B128 paired verification",
            dict(rows=256, n=5120, k=17408, small=False),
        ),
        Cell("DeepGEMM short GB200 context tile", dict(rows=144, n=14336, k=5120, small=False)),
        Cell(
            "DeepGEMM consumes packed SwiGLU exponents with a partial row group",
            dict(rows=129, n=5120, k=17408, small=False, packed=True),
        ),
    )

    def __call__(
        self,
        activations: torch.Tensor,
        activation_scales: torch.Tensor,
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        packed_weight_scales: torch.Tensor,
        rows: int,
        small: bool,
    ) -> torch.Tensor:
        a = activations[:rows]
        if small:
            scales = pack_activation_mx(activation_scales, rows)
            return torch.ops.trtllm.fp8_block_scale_low_m_gemm_prequantized(
                a, scales, weight, packed_weight_scales
            )
        scales = (
            activation_scales.T[:rows]
            if activation_scales.dtype == torch.int32
            else pack_activation_deepgemm(activation_scales, rows)
        )
        out = torch.empty((rows, weight.shape[0]), dtype=torch.bfloat16, device=weight.device)
        deep_gemm.fp8_gemm_nt(
            (a, scales), (weight, packed_weight_scales), out, disable_ue8m0_cast=True
        )
        return out

    def reference(
        self,
        activations: torch.Tensor,
        activation_scales: torch.Tensor,
        weight: torch.Tensor,
        weight_scales: torch.Tensor,
        packed_weight_scales: torch.Tensor,
        rows: int,
        small: bool,
    ) -> torch.Tensor:
        if activation_scales.dtype == torch.int32:
            groups = activations.shape[-1] // 128
            packed = activation_scales.view(torch.uint8).view(groups // 4, -1, 4)
            exponents = packed.transpose(1, 2).reshape(groups, -1).to(torch.int32)
            activation_scales = (exponents << 23).view(torch.float32)
        lhs = activations[:rows].float() * activation_scales.T[:rows].repeat_interleave(128, dim=-1)
        rhs = weight.float() * weight_scales.float().repeat_interleave(
            128, dim=0
        ).repeat_interleave(128, dim=1)
        with true_fp32_matmul():
            return (lhs @ rhs.T).bfloat16()


accelerated_block_matmul = AcceleratedBlockMatmul()

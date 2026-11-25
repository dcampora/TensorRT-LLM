# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic block-scaled FP8 projections."""

import torch
from flashinfer.gemm import gemm_fp8_nt_groupwise
from triton.tools.tensor_descriptor import TensorDescriptor

from .....catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from ..kernels import fp8, wide_projection
from ..kernels.tiled_gemm import tiled_gemm


class BlockRequantize(OpWrapper):
    """Rebase checkpoint E4M3 weights to power-of-two 128 x 128 scales."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("Group layout after output-projection fusion", dict(n=256, k=5120)),
        Cell("Several adjacent output groups", dict(n=1024, k=256)),
    )

    def __call__(
        self,
        weight: torch.Tensor,
        scales: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return fp8.requantize_pow2(weight, scales)

    def reference(
        self,
        weight: torch.Tensor,
        scales: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, k = weight.shape
        blocks = weight.float().reshape(n // 128, 128, k // 128, 128)
        values = (blocks * scales.float()[:, None, :, None]).bfloat16().float()
        peak = values.abs().amax(dim=(1, 3)).clamp_min(1e-4)
        block_scales = torch.exp2(torch.ceil(torch.log2(peak / 448))).to(scales.dtype)
        quantized = (values / block_scales[:, None, :, None]).clamp(-448, 448)
        return quantized.reshape(n, k).to(weight.dtype), block_scales


class BlockQuantize(OpWrapper):
    """Quantize each 128-element activation group with an exponent-only scale."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = tuple(
        Cell(f"Pad {rows} activation rows at the hidden width", dict(rows=rows, width=5120))
        for rows in (1, 3, 32, 128)
    )

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return fp8.quantize_aligned(x, True)

    def reference(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return reference_block_quantize(x)


def reference_block_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantization reference shared by the fused BF16 producers."""
    rows, width = x.shape
    groups = torch.nn.functional.pad(x.float(), (0, 0, 0, (-rows) % 4))
    groups = groups.reshape(-1, width // 128, 128)
    scales = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1.0e-10) / 448)))
    quantized = (groups / scales.unsqueeze(-1)).clamp(-448, 448)
    return quantized.reshape(-1, width).to(torch.float8_e4m3fn), scales.T.contiguous()


class BlockMatmul(OpWrapper):
    """Multiply quantized activation rows and weight rows, with FP32 accumulation."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("Split-K single-row output projection", dict(rows=1, n=5120, k=6144)),
        Cell("Padded verify projection", dict(rows=5, n=8192, k=5120)),
        Cell("Batched FFN down projection", dict(rows=32, n=5120, k=17408)),
        Cell("Narrow verifier tile for the fused QKVZ", dict(rows=7, n=16384, k=5120, tiled=True)),
        Cell("Narrow output tiles at batch 64", dict(rows=64, n=5120, k=6144, tiled=True)),
    )

    def __call__(
        self,
        a: torch.Tensor,
        sa: torch.Tensor,
        w: torch.Tensor,
        sw: torch.Tensor,
        sw_mn: torch.Tensor,
        rows: int,
        split_k: bool,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if split_k:
            return fp8.ordered_fp8_product(a, sa, w, sw, a.shape[0])
        if workspace is not None:
            return tiled_gemm(a, w, sa, sw_mn, workspace)[:rows]
        return gemm_fp8_nt_groupwise(
            a, w, sa, sw_mn, out_dtype=torch.bfloat16, backend="cutlass", scale_major_mode="MN"
        )[:rows]

    def reference(
        self,
        a: torch.Tensor,
        sa: torch.Tensor,
        w: torch.Tensor,
        sw: torch.Tensor,
        sw_mn: torch.Tensor,
        rows: int,
        split_k: bool,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _reference_matmul(a, sa, w, sw, rows)


def _reference_matmul(a, sa, w, sw, rows):
    lhs = a[:rows].float() * sa.T[:rows].repeat_interleave(128, dim=-1)
    rhs = w.float() * sw.float().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    with true_fp32_matmul():
        return (lhs @ rhs.T).bfloat16()


class WideBlockMatmul(OpWrapper):
    """Project a narrow verifier into the wide SwiGLU layer using row-major FP8 weights."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("Single row with three FP8 padding rows", dict(rows=1)),
        Cell("Five verifier rows and partially occupied tensor-core tiles", dict(rows=5)),
        Cell("Eight live verifier rows", dict(rows=8)),
    )

    def __call__(
        self,
        a: torch.Tensor,
        sa: torch.Tensor,
        weight: TensorDescriptor,
        sw: torch.Tensor,
        rows: int,
    ) -> torch.Tensor:
        return wide_projection.project(a, sa, weight, sw)[:rows]

    def reference(
        self,
        a: torch.Tensor,
        sa: torch.Tensor,
        weight: TensorDescriptor,
        sw: torch.Tensor,
        rows: int,
    ) -> torch.Tensor:
        return _reference_matmul(a, sa, weight.base, sw, rows)


block_quantize = BlockQuantize()
block_matmul = BlockMatmul()
wide_block_matmul = WideBlockMatmul()

block_requantize = BlockRequantize()

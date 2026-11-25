# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static ModelOpt projections following aegis-engines cb6779a, engine.py."""

from __future__ import annotations

from functools import lru_cache

import flashinfer as fi
import torch
import triton as tr
from torch import nn

from .....catalog._op import Arch, Cell, OpWrapper
from .....catalog.gemm.nvfp4_gemm import Nvfp4Gemm
from ..kernels.fp4_tactics import PREFILL_TACTICS, prefill_gemm, split_down
from ..kernels.fp4_tma import split_down_tma
from ..kernels.fp8_tactics import PREFILL_FP8_TILES, WIDE_TILES, prefill_algorithm, wider_algorithm
from ..kernels.nvfp4_pack import swiglu_pack
from ..kernels.pointwise import fp8_convert, fp8_rescale

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn


class StaticFp8Quantize(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(why="Small speculative step", spec={"rows": 4, "width": 5120}),
        Cell(why="Batched attention", spec={"rows": 128, "width": 6144}),
    )

    def __call__(self, x: torch.Tensor, scale: torch.Tensor, *, pdl: bool = False) -> torch.Tensor:
        y = torch.empty_like(x, dtype=FP8, memory_format=torch.contiguous_format)
        fp8_convert[(tr.cdiv(x.numel(), 1024),)](
            x, scale, y, x.numel(), 1024, pdl, num_warps=4, launch_pdl=pdl
        )
        return y

    def reference(self, x: torch.Tensor, scale: torch.Tensor, *, pdl: bool = False) -> torch.Tensor:
        return (x.float() * scale.reciprocal()).clamp(-448, 448).to(FP8)

    def is_valid(self, x: torch.Tensor, scale: torch.Tensor, *, pdl: bool = False) -> None:
        assert x.is_contiguous() and x.dtype == BF16 and scale.numel() == 1


class Fp8Requantize(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (Cell(why="Checkpoint K shard", spec={"rows": 1024, "width": 5120}),)

    def __call__(
        self, weight: torch.Tensor, src_scale: torch.Tensor, dst_scale: torch.Tensor
    ) -> torch.Tensor:
        result = torch.empty_like(weight)
        n = weight.numel()
        fp8_rescale[(tr.cdiv(n, 2048),)](
            weight.view(torch.uint8), src_scale, dst_scale, result.view(torch.uint8), n, 2048
        )
        return result

    def reference(
        self, weight: torch.Tensor, src_scale: torch.Tensor, dst_scale: torch.Tensor
    ) -> torch.Tensor:
        value = (weight.float() * src_scale.half().float()).half().float()
        return (value * dst_scale.reciprocal()).clamp(-448, 448).to(FP8)

    def is_valid(
        self, weight: torch.Tensor, src_scale: torch.Tensor, dst_scale: torch.Tensor
    ) -> None:
        assert (
            weight.dtype == FP8
            and weight.is_contiguous()
            and src_scale.numel() == dst_scale.numel() == 1
        )


class SwigluFp4Quantize(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(why="Single generation row", spec={"rows": 1, "width": 17408}),
        Cell(why="Batched generation", spec={"rows": 128, "width": 17408}),
    )

    def __call__(
        self, x: torch.Tensor, inv_scale: torch.Tensor, *, pdl: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return swiglu_pack(x, inv_scale, pdl)

    def reference(
        self, x: torch.Tensor, inv_scale: torch.Tensor, *, pdl: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate, up = x.float().chunk(2, dim=-1)
        return fi.fp4_quantize(
            (torch.nn.functional.silu(gate) * up).to(BF16), inv_scale, backend="cute-dsl"
        )

    def is_valid(self, x: torch.Tensor, inv_scale: torch.Tensor, *, pdl: bool = False) -> None:
        assert x.dtype == BF16 and x.is_contiguous() and x.shape[-1] % 512 == 0


fp8_quantize = StaticFp8Quantize()
fp8_requantize = Fp8Requantize()
swiglu_fp4 = SwigluFp4Quantize()


@lru_cache(maxsize=None)
def fp8_gemm_resources(device: torch.device) -> tuple:
    from flashinfer.jit.gemm import gen_gemm_module

    return (
        gen_gemm_module().build_and_load(),
        torch.empty(80 * 1024 * 1024, device=device, dtype=torch.uint8),
        {},
        {},
        {},
    )


class Fp8Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", None)
        self.register_buffer("input_scale", None)
        self.register_buffer("weight_scale", None)

    def forward(self, x: torch.Tensor, *, decode: bool = False) -> torch.Tensor:
        pdl = decode and x.shape[0] <= 8
        return self.gemm(fp8_quantize(x, self.input_scale, pdl=pdl), decode=decode)

    def gemm(self, x: torch.Tensor, *, decode: bool = False) -> torch.Tensor:
        weight = self.weight
        rows, k = x.shape
        n = weight.shape[-1]
        shape = (rows, k, n)
        x = x.unsqueeze(0)
        if not decode and shape not in PREFILL_FP8_TILES:
            return fi.bmm_fp8(
                x, weight, self.input_scale, self.weight_scale, BF16, backend="cublas"
            ).squeeze(0)

        module, workspace, narrow, wide, prefill = fp8_gemm_resources(x.device)
        output = torch.empty((1, rows, n), device=x.device, dtype=BF16)
        if not decode:
            if shape not in prefill:
                prefill[shape] = prefill_algorithm(
                    module,
                    x,
                    weight,
                    output,
                    self.input_scale,
                    self.weight_scale,
                    workspace,
                    *PREFILL_FP8_TILES[shape],
                )
            algo = prefill[shape]
            if algo is None:
                return fi.bmm_fp8(
                    x, weight, self.input_scale, self.weight_scale, BF16, backend="cublas"
                ).squeeze(0)
        else:
            key = (k, n)
            if key not in narrow:
                algo = torch.empty(64, dtype=torch.uint8, device="cpu")
                anchor = torch.empty((1, 1, k), dtype=FP8, device=x.device)
                one_row = torch.empty((1, 1, n), dtype=BF16, device=x.device)
                count = module.bmm_fp8_get_algos(
                    anchor, weight, one_row, self.input_scale, self.weight_scale, workspace, algo
                )
                if not count:
                    raise RuntimeError(f"No cuBLASLt FP8 decode algorithm for {key}")
                narrow[key] = algo
            algo = narrow[key]
            if shape in WIDE_TILES and shape not in wide:
                candidate = wider_algorithm(
                    module, x, weight, output, self.input_scale, self.weight_scale, workspace, algo
                )
                if candidate is not algo:
                    try:
                        self._run(module, x, output, workspace, candidate)
                    except RuntimeError:
                        wide[shape] = algo
                    else:
                        wide[shape] = candidate
                        return output.squeeze(0)
                else:
                    wide[shape] = algo
            algo = wide.get(shape, algo)
        self._run(module, x, output, workspace, algo)
        return output.squeeze(0)

    def _run(
        self,
        module,
        x: torch.Tensor,
        output: torch.Tensor,
        workspace: torch.Tensor,
        algo: torch.Tensor,
    ) -> None:
        module.bmm_fp8_run_with_algo(
            x, self.weight, output, self.input_scale, self.weight_scale, workspace, algo, 0
        )


class PackedFp4Gemm(OpWrapper):
    """Run block-scaled GEMM with the reference engine's wide down tactics."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            why="Two-way split-K down projection",
            spec={"m": 64, "k": 17408, "n": 5120, "down": True},
        ),
        Cell(why="TMA down projection", spec={"m": 128, "k": 17408, "n": 5120, "down": True}),
        Cell(
            why="Tuned prefill gate and up", spec={"m": 1024, "k": 5120, "n": 34816, "down": False}
        ),
    )

    def __call__(
        self,
        packed: torch.Tensor,
        weight: torch.Tensor,
        scales: torch.Tensor,
        weight_scales: torch.Tensor,
        alpha: torch.Tensor,
        *,
        down_decode: bool = False,
    ) -> torch.Tensor:
        scales = scales.view(FP8)
        m = packed.shape[0]
        if down_decode and m > 32:
            if m in (128, 256):
                return split_down_tma(packed, weight, scales, weight_scales, alpha)
            return split_down(packed, weight, scales, weight_scales, alpha)
        tactic = PREFILL_TACTICS.get((m, packed.shape[1] * 2, weight.shape[1]))
        if tactic is not None:
            return prefill_gemm(packed, weight, scales, weight_scales, alpha, tactic)
        return fi.mm_fp4(
            packed,
            weight,
            scales,
            weight_scales,
            alpha=alpha,
            out_dtype=BF16,
            backend="cute-dsl",
        )

    def reference(
        self,
        packed: torch.Tensor,
        weight: torch.Tensor,
        scales: torch.Tensor,
        weight_scales: torch.Tensor,
        alpha: torch.Tensor,
        *,
        down_decode: bool = False,
    ) -> torch.Tensor:
        return Nvfp4Gemm().reference(packed, weight.T, scales, weight_scales.T, alpha, BF16)


packed_fp4_gemm = PackedFp4Gemm()


class Fp4Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", None)
        self.register_buffer("scales", None)
        self.register_buffer("inv_scale", None)
        self.register_buffer("alpha", None)

    @property
    def num_embeddings(self) -> int:
        return self.weight.shape[1]

    @property
    def vocab_size_padded(self) -> int:
        return self.num_embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed, scales = fi.fp4_quantize(x, self.inv_scale, backend="cute-dsl")
        return packed_fp4_gemm(packed, self.weight, scales, self.scales, self.alpha)

    def swiglu(self, x: torch.Tensor, rows: torch.Tensor, *, decode: bool) -> torch.Tensor:
        m = x.shape[0]
        if decode and m <= 512:
            packed, scales = swiglu_fp4(x, self.inv_scale, pdl=m <= 8)
        else:
            packed, scales = fi.silu_and_mul_scaled_nvfp4_experts_quantize(
                x.unsqueeze(0), rows, self.inv_scale.reshape(1)
            )
            packed = packed.squeeze(-1).view(torch.uint8)
            scales = (
                scales.view(torch.uint8)
                .permute(2, 4, 0, 1, 3, 5)
                .reshape(scales.shape[2] * 128, -1)
            )

        return packed_fp4_gemm(
            packed, self.weight, scales, self.scales, self.alpha, down_decode=decode
        )

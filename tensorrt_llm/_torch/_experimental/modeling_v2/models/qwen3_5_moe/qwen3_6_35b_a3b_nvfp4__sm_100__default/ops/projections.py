# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelOpt FP8 and W4A16 projections."""

import torch
import triton
from flashinfer.gemm import bmm_fp8

from .....catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from ..kernels import quantized as k
from ..kernels.bf16_gate import gate_project
from ..weights import BF16, FP8, NUM_EXPERTS


class Fp8Cast(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("hidden projections with non-aligned token count", {"rows": 3, "width": 2048}),
        Cell("gated attention output", {"rows": 32, "width": 4096}),
    )

    def __call__(self, x, scale):
        q = torch.empty_like(x, dtype=FP8)
        k.fp8_cast[(triton.cdiv(x.numel(), 1024),)](x, scale, q, x.numel(), False, 1024)
        return q

    def reference(self, x, scale):
        return (x.float() / scale).clamp(-448, 448).to(FP8)


fp8_cast = Fp8Cast()


class DeltaGateProjection(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("one token", {"rows": 1}),
        Cell("verification with three draft tokens", {"rows": 4}),
    )

    def __call__(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        out = torch.empty((x.shape[0], 64), dtype=BF16, device=x.device)
        gate_project[(4, x.shape[0])](x, weight, out, num_warps=4, num_stages=2)
        return out

    def reference(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        with true_fp32_matmul():
            return (x.float() @ weight.float().T).to(BF16)


delta_gate_projection = DeltaGateProjection()


class ChannelFp8Linear(OpWrapper):
    """Fused FP8 projection with independent scales per output channel."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("linear-attention projection on a decode step", {"rows": 4, "width": 12288}),
        Cell("full-attention projection on a prefill", {"rows": 128, "width": 9216}),
    )

    def __call__(self, x, weight, activation_scales, weight_scales):
        return torch._scaled_mm(
            x, weight.T, activation_scales, weight_scales, out_dtype=BF16, use_fast_accum=True
        )

    def reference(self, x, weight, activation_scales, weight_scales):
        with true_fp32_matmul():
            return ((x.float() @ weight.float().T) * activation_scales * weight_scales).to(BF16)


channel_fp8_linear = ChannelFp8Linear()


def fp8_linear(x, weights):
    if x.dtype != FP8:
        x = fp8_cast(x, weights.input_scale)
    if weights.weight_scale.ndim:
        return channel_fp8_linear(
            x, weights.weight, weights.scales_for_rows(x.shape[0]), weights.weight_scale
        )
    return bmm_fp8(
        x[None],
        weights.weight.T[None],
        weights.input_scale.view(1),
        weights.weight_scale.view(1),
        BF16,
        backend="cublas",
    )[0]


def unpack_nvfp4(weight, scales):
    codes = torch.stack((weight & 15, weight >> 4), -1).flatten(-2)
    magnitude = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=weight.device)
    values = magnitude[(codes & 7).long()] * torch.where(codes < 8, 1, -1)
    return values * scales.float().repeat_interleave(16, dim=-1)


class PackedGemv(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("shared up and gate", {"rows": 1, "mode": 1, "n": 1024, "width": 2048}),
        Cell("shared down", {"rows": 32, "mode": 0, "n": 2048, "width": 512}),
    )

    def __call__(self, x, weight, scales, alpha, mode=0, ids=None, probs=None, interleaved=False):
        n, packed_k = weight.shape[-2:]
        width = packed_k * 2
        routed = ids is not None
        slots = ids.numel() if routed else x.shape[0]
        out_features = n // 2 if mode == 1 else n
        out = torch.empty((slots, out_features), device=x.device, dtype=BF16)
        tile = 4 if width == 2048 else 8
        k.w4_gemv[(out_features // tile, slots)](
            x,
            weight,
            scales,
            alpha,
            out,
            ids,
            probs,
            n,
            width,
            tile,
            routed,
            mode,
            interleaved,
            num_warps=4,
        )
        return out

    def reference(self, x, weight, scales, alpha, mode=0, ids=None, probs=None, interleaved=False):
        w = unpack_nvfp4(weight, scales)
        if interleaved:
            shape = w.shape
            up, gate = w.reshape(*shape[:-2], 8, 2, 64, shape[-1]).unbind(-3)
            w = torch.stack((gate, up), -3).reshape(shape)
        with true_fp32_matmul():
            if ids is None:
                y = x.float() @ w.T
            else:
                rows = x.repeat_interleave(ids.shape[1], 0) if mode == 1 else x
                y = torch.bmm(w[ids.flatten().long()], rows.float().unsqueeze(-1)).squeeze(-1)
                alpha = alpha[ids.flatten().long(), None]
        y = y.to(BF16).float() * alpha.float()
        if mode == 1:
            gate, up = y.to(BF16).float().chunk(2, -1)
            y = torch.nn.functional.silu(gate) * up
        if mode == 2:
            y = y.to(BF16).float() * probs.flatten()[:, None]
        return y.to(BF16)


packed_gemv = PackedGemv()


def gemv(x, weights, mode=0, ids=None, probs=None):
    return packed_gemv(
        x, weights.w, weights.scales, weights.alpha, mode, ids, probs, weights.interleaved
    )


def packed_matmul(x, weights, mode=0, routing=None, tile=16):
    if routing is None:
        sorted_slots = starts = tile_starts = probs = None
        rows = x.shape[0]
        tiles = triton.cdiv(rows, tile)
    else:
        sorted_slots, starts, tile_starts, probs = routing
        rows = sorted_slots.numel()
        tiles = NUM_EXPERTS + triton.cdiv(rows, tile)
    out = torch.empty((rows, weights.n), device=x.device, dtype=BF16)
    k.w4_matmul[(tiles, weights.n // 64)](
        x,
        weights.w,
        weights.scales,
        weights.alpha,
        out,
        sorted_slots,
        starts,
        tile_starts,
        probs,
        rows,
        weights.n,
        weights.k,
        mode,
        routing is not None,
        tile,
        64,
        64,
        weights.interleaved,
        num_warps=4,
        num_stages=3,
    )
    return out

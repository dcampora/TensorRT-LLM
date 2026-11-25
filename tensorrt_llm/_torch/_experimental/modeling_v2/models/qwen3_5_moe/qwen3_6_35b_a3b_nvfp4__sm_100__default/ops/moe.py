# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelOpt MoE routing and expert epilogues from aegis-engines (a6c09c6502)."""

from typing import NamedTuple

import torch
import triton
from torch import nn

from .....catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from ..kernels import quantized as k
from ..kernels.cute_moe import PackedExperts
from ..kernels.cute_moe import mix_experts_norm as _mix_experts_norm
from ..kernels.dense_vectors import shared_down
from ..kernels.shared_corrected import RoundedShared
from ..weights import BF16, EXPERT_WIDTH, HIDDEN, NUM_EXPERTS, TOP_K, PackedWeights, parameter
from .projections import gemv, packed_matmul


class ResidualRmsNorm(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("first layer, record residual", {"rows": 1, "add": False, "save": True}),
        Cell("after attention", {"rows": 128, "add": True, "save": True}),
        Cell("final norm", {"rows": 32, "add": True, "save": False}),
        Cell(
            "emit the next attention operand", {"rows": 32, "add": True, "save": True, "fp8": True}
        ),
    )

    def __call__(self, x, residual, weight, add, save, fp8=None, scale=None):
        out = torch.empty_like(x)
        k.norm[(x.shape[0],)](
            x,
            residual,
            weight,
            out,
            save,
            add,
            x.shape[-1],
            fp8,
            scale,
            fp8 is not None,
            num_warps=4,
        )
        return out

    def reference(self, x, residual, weight, add, save, fp8=None, scale=None):
        value = x.float() + residual.float() if add else x.float()
        if save:
            residual.copy_(value)
        inverse = (value.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        out = (value * inverse * (1 + weight.float())).to(BF16)
        if fp8 is not None:
            fp8.copy_((out.float() / scale).clamp(-448, 448))
        return out


class RouteTopK(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("single request, including exact router ties", {"rows": 1, "weighted": False}),
        Cell("batch with scale folded into the routing weights", {"rows": 128, "weighted": True}),
        Cell("split-K router and weighted top-k", {"rows": 20, "weighted": True, "split": True}),
    )

    def __call__(self, router, scale, weight=None):
        rows = router.shape[0]
        weighted = scale is not None
        probabilities = torch.empty(
            (rows, TOP_K), dtype=BF16 if weighted else torch.float32, device=router.device
        )
        ids = torch.empty((rows, TOP_K), dtype=torch.int32, device=router.device)
        shared_gate = torch.empty(rows, dtype=BF16, device=router.device) if weighted else None
        if weight is not None:
            partial = torch.empty((rows, 16, 272), dtype=BF16, device=router.device)
            cols = 16 if rows <= 16 else 32
            k.router_partials[(triton.cdiv(NUM_EXPERTS + 1, cols), 16, triton.cdiv(rows, 16))](
                router, weight, partial, rows, cols, num_warps=4
            )
            router = partial
        k.router_topk[(rows,)](
            router,
            probabilities,
            ids,
            router.stride(0),
            scale,
            shared_gate,
            weighted,
            weight is not None,
            num_warps=4,
        )
        return ids, probabilities, shared_gate

    def reference(self, router, scale, weight=None):
        if weight is not None:
            with true_fp32_matmul():
                router = sum(
                    (
                        router[:, start : start + 128].float()
                        @ weight[: NUM_EXPERTS + 1, start : start + 128].float().T
                    )
                    .to(BF16)
                    .float()
                    for start in range(0, HIDDEN, 128)
                ).to(BF16)
        scores = router[:, :NUM_EXPERTS].float()
        ids = scores.argsort(descending=True, stable=True)[:, :TOP_K]
        scores = scores.gather(1, ids).softmax(dim=-1)
        shared_gate = None
        if scale is not None:
            scores = (scores.to(BF16).float() * scale[ids].float()).to(BF16)
            shared_gate = router[:, NUM_EXPERTS].float().sigmoid().to(BF16)
        return ids.int(), scores, shared_gate


residual_rms_norm = ResidualRmsNorm()
route_topk = RouteTopK()


def swiglu(x):
    out = torch.empty((x.shape[0], x.shape[1] // 2), device=x.device, dtype=x.dtype)
    k.swiglu[(triton.cdiv(out.numel(), 512),)](x, out, out.numel(), out.size(1), 512)
    return out


class ExpertEpilogue(NamedTuple):
    residual: torch.Tensor
    norm: torch.Tensor
    save: bool
    fp8: torch.Tensor | None
    scale: torch.Tensor | None


class MixExpertsNorm(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("weighted experts and FP8 attention input", {"rows": 4, "weighted": True}),
        Cell("unweighted experts and final norm", {"rows": 7, "weighted": False}),
    )

    def __call__(self, raw, alpha, ids, probs, slots, shared, gate, weighted, epilogue):
        residual, weight, save, fp8, scale = epilogue
        out = torch.empty_like(shared)
        _mix_experts_norm[(shared.shape[0],)](
            raw,
            alpha,
            ids,
            probs,
            slots,
            shared,
            gate,
            residual,
            weight,
            out,
            0 if weighted else gate.stride(0),
            save,
            weighted,
            fp8,
            scale,
            fp8 is not None,
            num_warps=4,
        )
        return out

    def reference(self, raw, alpha, ids, probs, slots, shared, gate, weighted, epilogue):
        if not weighted:
            probs = (alpha[ids.long()].float() * probs.to(BF16).float()).to(BF16)
            gate = gate[:, NUM_EXPERTS].float().sigmoid().to(BF16)
        routed = (raw[slots.long()].float() * probs.float()[:, :, None]).to(BF16).float()
        routed = routed.sum(dim=1).to(BF16).float()
        shared = (shared.float() * gate.float()[:, None]).to(BF16).float()
        mixed = (routed + shared).to(BF16)
        residual, weight, save, fp8, scale = epilogue
        return residual_rms_norm.reference(mixed, residual, weight, True, save, fp8, scale)


mix_experts_norm = MixExpertsNorm()


class Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up = PackedWeights(
            HIDDEN, 2 * EXPERT_WIDTH, experts=NUM_EXPERTS, interleaved=True
        )
        self.down = PackedWeights(EXPERT_WIDTH, HIDDEN, experts=NUM_EXPERTS)
        self.shared_gate_up = PackedWeights(HIDDEN, 2 * EXPERT_WIDTH)
        self.shared_down = PackedWeights(EXPERT_WIDTH, HIDDEN)
        self.router = parameter((NUM_EXPERTS + 16, HIDDEN))

    def prepare(self):
        self.grouped = PackedExperts(self.gate_up, self.down)
        self.correct_gate = RoundedShared(self.shared_gate_up, band=256, floor=139)
        self.correct_down = RoundedShared(self.shared_down, band=128, floor=130)

    def _shared(self, x, packed_prefill, small_verify=False):
        if packed_prefill:
            raw = self.correct_gate.packed._raw(x)
            mid = torch.empty((x.shape[0], EXPERT_WIDTH), dtype=x.dtype, device=x.device)
            k.shared_swiglu[(x.shape[0],)](raw, self.shared_gate_up.alpha, mid, num_warps=4)
            return self.correct_down.packed._raw(mid)
        if x.shape[0] >= 64:
            return self.correct_down(self.correct_gate(x, True), False)
        mid = gemv(x, self.shared_gate_up, 1)
        if x.shape[0] >= 16 and not small_verify:
            return shared_down(mid, self.shared_down)
        return gemv(mid, self.shared_down)

    def forward(self, x, bins, side, contexts, batch, epilogue, width):
        rows = x.shape[0]
        cutoff = (128 if batch == 1 else 1024) if contexts else 1
        grouped = rows >= cutoff
        use_gemv = rows <= 128
        packed_prefill = contexts and grouped
        verifying = not contexts and width > 1
        small_verify = verifying and batch <= 8
        share_in_parallel = side is not None and (grouped or use_gemv)

        if share_in_parallel:
            main = torch.cuda.current_stream()
            side.wait_stream(main)
            with torch.cuda.stream(side):
                shared = self._shared(x, packed_prefill, small_verify)

        router = (
            self.router if grouped and (contexts or batch > 1) else self.router[: NUM_EXPERTS + 1]
        )
        use_split = verifying and epilogue is not None and batch == 1 and width == 4
        if use_split:
            logits = x
        elif small_verify and batch > 1:
            weight = self.router[: NUM_EXPERTS + 1].T
            logits = torch.cat([torch.mm(request, weight) for request in x.split(width)])
        else:
            logits = torch.mm(x, router.T)
        weighted = grouped and epilogue is not None
        ids, probs, shared_gate = route_topk(
            logits, self.down.alpha if weighted else None, self.router if use_split else None
        )
        if grouped:
            tokens = batch if verifying else rows
            tile = 8 if tokens < 64 else 16 if tokens <= 128 else 64
            if verifying and rows >= 96:
                tile = 64 if rows >= 256 else 32
            wide = contexts and rows >= 1024
            if wide:
                tile = 128
            routed = self.grouped(
                x,
                ids,
                probs,
                route_tile=tile,
                tactic_m=256 if wide else 128,
                cluster=2,
                raw=epilogue is not None,
            )
            if share_in_parallel:
                main.wait_stream(side)
            else:
                shared = self._shared(x, packed_prefill, small_verify)
            if epilogue is None:
                out = torch.empty_like(x)
                alpha = self.shared_down.alpha if packed_prefill else None
                k.add_shared[(rows, 4)](
                    routed,
                    shared,
                    logits,
                    out,
                    HIDDEN,
                    HIDDEN // 4,
                    logits.stride(0),
                    alpha,
                    packed_prefill,
                )
                return out
            raw, slots = routed
            return mix_experts_norm(
                raw, self.down.alpha, ids, probs, slots, shared, shared_gate, weighted, epilogue
            )

        if use_gemv:
            mid = gemv(x, self.gate_up, 1, ids, probs)
            experts = gemv(mid, self.down, 2, ids, probs)
            if share_in_parallel:
                main.wait_stream(side)
            else:
                shared = self._shared(x, False)
        else:
            tile = 32 if rows > 2048 else 16
            sorted_ids, slots = torch.sort(ids.flatten(), stable=True)
            starts = torch.searchsorted(sorted_ids, bins, out_int32=True)
            tiles = (starts[1:] - starts[:-1] + tile - 1) // tile
            prefix = torch.cumsum(torch.cat((bins[:1], tiles)), dim=0, dtype=torch.int32)
            routing = (slots, starts, prefix, probs)
            mid = swiglu(packed_matmul(x, self.gate_up, 1, routing, tile))
            experts = packed_matmul(mid, self.down, 2, routing, tile)
            shared = packed_matmul(swiglu(packed_matmul(x, self.shared_gate_up)), self.shared_down)
        out = torch.empty_like(x)
        k.combine[(rows, 4)](experts, shared, logits, out, HIDDEN, HIDDEN // 4)
        return out

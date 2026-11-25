# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelOpt MoE routing and expert epilogues from aegis-engines (a6c09c6502)."""

import torch
import triton
from torch import nn

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels import quantized as k
from ..kernels.cute_moe import PackedExperts, mix_experts_norm
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
    )

    def __call__(self, x, residual, weight, add, save):
        out = torch.empty_like(x)
        k.norm[(x.shape[0],)](x, residual, weight, out, save, add, x.shape[-1], num_warps=4)
        return out

    def reference(self, x, residual, weight, add, save):
        value = x.float() + residual.float() if add else x.float()
        if save:
            residual.copy_(value)
        inverse = (value.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        return (value * inverse * (1 + weight.float())).to(BF16)


class RouteTopK(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("single request, including exact router ties", {"rows": 1, "weighted": False}),
        Cell("batch with scale folded into the routing weights", {"rows": 128, "weighted": True}),
    )

    def __call__(self, router, scale):
        rows = router.shape[0]
        weighted = scale is not None
        probabilities = torch.empty(
            (rows, TOP_K), dtype=BF16 if weighted else torch.float32, device=router.device
        )
        ids = torch.empty((rows, TOP_K), dtype=torch.int32, device=router.device)
        shared_gate = torch.empty(rows, dtype=BF16, device=router.device) if weighted else None
        k.router_topk[(rows,)](
            router, probabilities, ids, router.stride(0), scale, shared_gate, weighted, num_warps=4
        )
        return ids, probabilities, shared_gate

    def reference(self, router, scale):
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

    def _shared(self, x, packed_prefill):
        if packed_prefill:
            raw = self.correct_gate.packed._raw(x)
            mid = torch.empty((x.shape[0], EXPERT_WIDTH), dtype=x.dtype, device=x.device)
            k.shared_swiglu[(x.shape[0],)](raw, self.shared_gate_up.alpha, mid, num_warps=4)
            return self.correct_down.packed._raw(mid)
        if x.shape[0] >= 64:
            return self.correct_down(self.correct_gate(x, True), False)
        mid = gemv(x, self.shared_gate_up, 1)
        if x.shape[0] >= 16:
            return shared_down(mid, self.shared_down)
        return gemv(mid, self.shared_down)

    def forward(self, x, bins, side, contexts, batch, epilogue):
        rows = x.shape[0]
        cutoff = (128 if batch == 1 else 1024) if contexts else 1
        grouped = rows >= cutoff
        use_gemv = rows <= 128
        packed_prefill = contexts and grouped
        share_in_parallel = grouped or use_gemv

        if share_in_parallel:
            main = torch.cuda.current_stream()
            side.wait_stream(main)
            with torch.cuda.stream(side):
                shared = self._shared(x, packed_prefill)

        router = (
            self.router if grouped and (contexts or batch > 1) else self.router[: NUM_EXPERTS + 1]
        )
        logits = torch.mm(x, router.T)
        weighted = grouped and epilogue is not None
        ids, probs, shared_gate = route_topk(logits, self.down.alpha if weighted else None)
        out = torch.empty_like(x)
        if grouped:
            tile = 8 if rows < 64 else 16 if rows <= 128 else 64
            wide = contexts and rows >= 4096
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
            main.wait_stream(side)
            if epilogue is None:
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
            else:
                raw, slots = routed
                residual, weight, save = epilogue
                mix_experts_norm[(rows,)](
                    raw,
                    self.down.alpha,
                    ids,
                    probs,
                    slots,
                    shared,
                    shared_gate,
                    residual,
                    weight,
                    out,
                    logits.stride(0),
                    save,
                    weighted,
                    num_warps=4,
                )
            return out

        if use_gemv:
            mid = gemv(x, self.gate_up, 1, ids, probs)
            experts = gemv(mid, self.down, 2, ids, probs)
            main.wait_stream(side)
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
        k.combine[(rows, 4)](experts, shared, logits, out, HIDDEN, HIDDEN // 4)
        return out

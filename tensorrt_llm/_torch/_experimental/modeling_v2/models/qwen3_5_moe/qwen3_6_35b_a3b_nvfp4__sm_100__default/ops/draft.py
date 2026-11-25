# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 proposal operations for the checkpoint's MTP layer."""

import torch
from flashinfer.fused_moe.cute_dsl.moe_utils import get_max_num_permuted_tokens

from .....catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from ..kernels import draft as k
from ..kernels import quantized as quant
from ..kernels.cute_moe import group_routes
from ..weights import BF16, EXPERT_WIDTH, HIDDEN, NUM_EXPERTS, TOP_K
from .moe import route_topk, swiglu


class DraftInputNorm(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("one selected proposal row", {"rows": 1}),
        Cell("packed target verification rows", {"rows": 128}),
    )

    def __call__(self, ids, embedding, hidden, embedding_norm, hidden_norm):
        rows = hidden.shape[0]
        out = torch.empty((rows, 2 * HIDDEN), dtype=BF16, device=hidden.device)
        k.draft_prepare[(rows, 2)](
            ids,
            embedding,
            hidden,
            embedding_norm,
            hidden_norm,
            out,
            HIDDEN,
            num_warps=4,
        )
        return out

    def reference(self, ids, embedding, hidden, embedding_norm, hidden_norm):
        def normalize(x, weight):
            x = x.float()
            return x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * (1 + weight.float())

        return torch.cat(
            (normalize(embedding[ids.long()], embedding_norm), normalize(hidden, hidden_norm)),
            dim=-1,
        ).to(BF16)


class DraftExperts(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("single request using BF16 expert GEMV", {"rows": 1}),
        Cell("non-uniform routed batch using grouped tensor cores", {"rows": 32}),
    )

    def __call__(self, x, gate_up, down, ids, probabilities, bins):
        rows = x.shape[0]
        slots = rows * TOP_K
        out = torch.empty((slots, HIDDEN), dtype=BF16, device=x.device)
        if rows < 16:
            middle = torch.empty((slots, EXPERT_WIDTH), dtype=BF16, device=x.device)
            k.bf16_experts[(EXPERT_WIDTH // 4, slots)](
                x,
                gate_up,
                ids,
                probabilities,
                middle,
                True,
                4,
                2 * EXPERT_WIDTH,
                HIDDEN,
                num_warps=4,
            )
            k.bf16_experts[(HIDDEN // 8, slots)](
                middle,
                down,
                ids,
                probabilities,
                out,
                False,
                8,
                HIDDEN,
                EXPERT_WIDTH,
                num_warps=4,
            )
        else:
            tile = 8 if rows < 64 else 16
            capacity = get_max_num_permuted_tokens(rows, TOP_K, NUM_EXPERTS, tile)
            experts, end, _, sorted_slots, active = group_routes(ids, tile, capacity, bins)
            middle = torch.empty((capacity, EXPERT_WIDTH), dtype=BF16, device=x.device)
            k.draft_experts[(capacity // tile, EXPERT_WIDTH // 64)](
                x,
                gate_up,
                experts,
                end,
                sorted_slots,
                active,
                probabilities,
                middle,
                True,
                tile,
                64,
                64,
                2 * EXPERT_WIDTH,
                HIDDEN,
                num_warps=4,
                num_stages=3,
            )
            k.draft_experts[(capacity // tile, HIDDEN // 64)](
                middle,
                down,
                experts,
                end,
                sorted_slots,
                active,
                probabilities,
                out,
                False,
                tile,
                64,
                64,
                HIDDEN,
                EXPERT_WIDTH,
                num_warps=4,
                num_stages=3,
            )
        return out

    def reference(self, x, gate_up, down, ids, probabilities, bins):
        slots = ids.flatten().long()
        x = x.repeat_interleave(TOP_K, dim=0)
        out = torch.empty((slots.numel(), HIDDEN), dtype=BF16, device=x.device)
        with true_fp32_matmul():
            for slot in slots.unique().tolist():
                selected = slots == slot
                gate, up = (
                    (x[selected].float() @ gate_up[slot].float().T).to(BF16).float().chunk(2, -1)
                )
                middle = (torch.nn.functional.silu(gate) * up).to(BF16).float()
                expert = (middle @ down[slot].float().T).to(BF16).float()
                p = probabilities.flatten()[selected, None].to(BF16).float()
                out[selected] = (expert * p).to(BF16)
        return out


draft_input_norm = DraftInputNorm()
draft_experts = DraftExperts()


def draft_moe(x, router, gate_up, down, shared_gate, shared_down, bins, side):
    current = torch.cuda.current_stream()
    side.wait_stream(current)
    with torch.cuda.stream(side):
        shared = torch.mm(swiglu(torch.mm(x, shared_gate.T)), shared_down.T)
    logits = torch.mm(x, router.T)
    ids, probs, _ = route_topk(logits, None)
    expert = draft_experts(x, gate_up, down, ids, probs, bins)
    current.wait_stream(side)
    out = torch.empty_like(x)
    quant.combine[(x.shape[0], 4)](expert, shared, logits, out, HIDDEN, HIDDEN // 4)
    return out

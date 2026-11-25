# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet operations over the runtime's convolution and SSM slots."""

import torch
import torch.nn.functional as F

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels import delta, short_gdn


class CausalConvolution(OpWrapper):
    """Depthwise causal SiLU convolution with BF16 slot history."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("Decode one physical slot out of a larger pool", dict(lengths=(1,), decode=True)),
        Cell("Decode several nonadjacent slots", dict(lengths=(1, 1, 1), decode=True)),
        Cell("Fresh and continued context chunks", dict(lengths=(3, 7, 1), decode=False)),
    )

    def __call__(
        self,
        projection: torch.Tensor,
        weight: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
        offsets: torch.Tensor | None = None,
        row_to_seq: torch.Tensor | None = None,
        has_initial: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return delta.convolution(projection, weight, state, slots, offsets, row_to_seq, has_initial)

    def reference(
        self,
        projection: torch.Tensor,
        weight: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
        offsets: torch.Tensor | None = None,
        row_to_seq: torch.Tensor | None = None,
        has_initial: torch.Tensor | None = None,
    ) -> torch.Tensor:
        width = weight.shape[0]
        ends = list(range(slots.numel() + 1)) if offsets is None else offsets.tolist()
        parts = []
        for seq, slot in enumerate(slots.tolist()):
            carry = state.new_zeros((width, 3))
            if slot >= 0 and (has_initial is None or bool(has_initial[seq])):
                carry = state[slot]
            start, end = ends[seq : seq + 2]
            x = torch.cat((carry, projection[start:end, :width].T), dim=1)
            y = (x.float().unfold(1, 4, 1) * weight.float().view(width, 1, 4)).sum(dim=-1)
            parts.append(F.silu(y.T).to(projection.dtype))
            if slot >= 0:
                state[slot].copy_(x[:, -3:])
        return torch.cat(parts)


class DeltaGates(OpWrapper):
    """Compute log decay and BF16-rounded beta from contiguous [beta, decay] logits."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("Single-request context", dict(rows=17)),
        Cell("Packed long contexts", dict(rows=1024)),
    )

    def __call__(
        self,
        ba: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return delta.gdn_gates(ba, a_log, dt_bias, beta_bf16=True)

    def reference(
        self,
        ba: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, a = ba.float().chunk(2, dim=-1)
        return -a_log.float().exp() * F.softplus(
            a + dt_bias.float()
        ), b.sigmoid().bfloat16().float()


def _reference_delta(
    qkv: torch.Tensor,
    decay: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q, k, v = qkv.float().split((2048, 2048, 6144), dim=-1)
    q = q.unflatten(-1, (16, 128))
    k = k.unflatten(-1, (16, 128))
    v = v.unflatten(-1, (48, 128))
    q = (q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt()).repeat_interleave(3, dim=-2)
    k = (k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()).repeat_interleave(3, dim=-2)
    state = state * decay.exp()[..., None, None]
    update = (v - (state * k[..., None, :]).sum(-1)) * beta[..., None]
    state = torch.addcmul(state, update[..., None], k[..., None, :])
    return (state * (q * (128**-0.5))[..., None, :]).sum(-1), state


class RecurrentDelta(OpWrapper):
    """Update the FP32 recurrent state for one token in each active slot."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("Decode on a single nonzero slot", dict(rows=1)),
        Cell("Decode with nonconsecutive slots and a padding row", dict(rows=4)),
    )

    def __call__(
        self,
        x: torch.Tensor,
        ba: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        return delta.recurrence(x, ba, a_log, dt_bias, state, slots)

    def reference(
        self,
        x: torch.Tensor,
        ba: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        b, a = ba.float().chunk(2, dim=-1)
        decay = -a_log.float().exp() * F.softplus(a + dt_bias.float())
        beta = b.sigmoid().bfloat16().float()
        old_state = state[slots.clamp_min(0).long()].float()
        old_state = old_state.masked_fill(slots[:, None, None, None] < 0, 0)
        output, next_state = _reference_delta(x, decay, beta, old_state)
        valid = slots >= 0
        state[slots[valid].long()] = next_state[valid].to(state.dtype)
        return output.to(x.dtype)


class ShortContextDelta(OpWrapper):
    """Triangular GDN solve over fresh contexts of at most 128 rows per request."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("Fresh contexts sharing a 64-token solve tile", dict(lengths=(9, 35))),
        Cell("Context spanning multiple solve tiles", dict(lengths=(98,))),
    )

    def __call__(
        self,
        qkv: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
        offsets: torch.Tensor,
        chunks: torch.Tensor,
        starts: torch.Tensor,
        max_length: int,
        state: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        return short_gdn.chunked_delta(
            qkv, decay, beta, offsets, chunks, starts, max_length, state, slots
        )

    def reference(
        self,
        qkv: torch.Tensor,
        decay: torch.Tensor,
        beta: torch.Tensor,
        offsets: torch.Tensor,
        chunks: torch.Tensor,
        starts: torch.Tensor,
        max_length: int,
        state: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """Sequential FP32 state update. The solve tile schedule is unused."""
        indices = offsets.tolist()
        parts = []
        for seq, slot in enumerate(slots.tolist()):
            carry = state.new_zeros((48, 128, 128), dtype=torch.float32)
            for row in range(indices[seq], indices[seq + 1]):
                output, carry = _reference_delta(qkv[row], decay[row], beta[row], carry)
                parts.append(output)
            if slot >= 0:
                state[slot].copy_(carry)
        return torch.stack(parts).to(qkv.dtype)


class GatedHeadNorm(OpWrapper):
    """Apply headwise RMSNorm and the Gated DeltaNet SiLU output gate."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("Single generation row", dict(rows=1)),
        Cell("Packed context rows", dict(rows=137)),
    )

    def __call__(
        self, x: torch.Tensor, projection: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        return delta.gdn_norm(x, projection, weight)

    def reference(
        self, x: torch.Tensor, projection: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        x = x.float()
        z = projection[:, 10240:].float().view_as(x)
        x = x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * weight.float()
        return (x * F.silu(z)).flatten(1).to(projection.dtype)


causal_convolution = CausalConvolution()
delta_gates = DeltaGates()
recurrent_delta = RecurrentDelta()
short_context_delta = ShortContextDelta()
gated_head_norm = GatedHeadNorm()

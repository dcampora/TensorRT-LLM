# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet operations for linear speculative verification."""

from typing import NamedTuple

import torch
import torch.nn.functional as F

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels import prepared_verify, verify


class ReplayHistory(NamedTuple):
    """The manager's double-buffered history for one layer.

    Each BF16 key/value bank stores half as many FP32 keys/deltas. The decay
    bank already holds FP32 words. The first row anchors the durable state;
    later rows are replayed from this log after the sampler accepts them.
    """

    count: torch.Tensor
    bank: torch.Tensor
    value: torch.Tensor
    key: torch.Tensor
    decay: torch.Tensor


class VerifyConvolution(OpWrapper):
    """Convolve each chain without publishing an unaccepted history window."""

    ARCHS = frozenset({Arch.SM_100})

    CELLS = (
        Cell("Single-slot four-token verification", dict(seqs=1, tokens=4)),
        Cell("Ragged slots with a two-token verifier", dict(seqs=8, tokens=2)),
    )

    def __call__(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        conv: torch.Tensor,
        windows: torch.Tensor,
        slots: torch.Tensor,
        temporary_slots: torch.Tensor,
        tokens: int,
    ) -> torch.Tensor:
        return verify.verify_convolution(x, w, conv, windows, slots, temporary_slots, tokens)

    def reference(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        conv: torch.Tensor,
        windows: torch.Tensor,
        slots: torch.Tensor,
        temporary_slots: torch.Tensor,
        tokens: int,
    ) -> torch.Tensor:
        width = w.shape[0]
        parts = []
        for row, (slot, temporary) in enumerate(zip(slots.tolist(), temporary_slots.tolist())):
            carry = conv[slot] if slot >= 0 else conv.new_zeros((width, 3))
            source = torch.cat((carry, x[row * tokens : (row + 1) * tokens, :width].T), dim=1)
            inputs = source.float().unfold(1, 4, 1)
            y = F.silu((inputs * w.float().view(width, 1, 4)).sum(-1).T)
            parts.append(y.to(x.dtype))
            if temporary >= 0 and slot >= 0:
                windows[temporary, :tokens].copy_(source[:, 1:].unfold(1, 3, 1).transpose(0, 1))
        return torch.cat(parts)


class VerifyDelta(OpWrapper):
    """Verify a draft chain from a checkpoint at the last accepted first row."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("Write the first verify chain into an empty bank", dict(tokens=5, old=0)),
        Cell("Fold accepted draft rows and use the alternate bank", dict(tokens=2, old=4)),
        Cell("Fold the longest possible accepted chain", dict(tokens=8, old=8)),
        Cell("Fold accepted history and commit one direct token", dict(tokens=1, old=6)),
    )

    def __call__(
        self,
        qkv: torch.Tensor,
        ba: torch.Tensor,
        a_log: torch.Tensor,
        dt: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
        replay_slots: torch.Tensor,
        history: ReplayHistory,
        tokens: int,
    ) -> torch.Tensor:
        return verify.replay_delta(
            qkv,
            ba,
            a_log,
            dt,
            state,
            slots,
            replay_slots,
            history.count,
            history.bank,
            history.value,
            history.key,
            history.decay,
            tokens,
        )

    def reference(
        self,
        qkv: torch.Tensor,
        ba: torch.Tensor,
        a_log: torch.Tensor,
        dt: torch.Tensor,
        state: torch.Tensor,
        slots: torch.Tensor,
        replay_slots: torch.Tensor,
        history: ReplayHistory,
        tokens: int,
    ) -> torch.Tensor:
        out = []
        count, bank = history.count.tolist(), history.bank.tolist()
        replay_shape = (history.value.shape[0], 2, history.value.shape[2] // 2)
        value = history.value.view(torch.float32).view(*replay_shape, 48, 128)
        key = history.key.view(torch.float32).view(*replay_shape, 16, 128)
        for seq, (slot, log_slot) in enumerate(zip(slots.tolist(), replay_slots.tolist())):
            n, active = count[log_slot], bank[log_slot]
            carry = state[slot].float()
            for token in range(n - 1):
                direction = key[log_slot, active, token].repeat_interleave(3, dim=0)
                update = value[log_slot, active, token]
                decay = history.decay[log_slot, active, :, token]
                carry = torch.addcmul(
                    carry * decay[:, None, None], update[:, :, None], direction[:, None, :]
                )
            active = 1 - active if n else active
            for token in range(tokens):
                row = seq * tokens + token
                q, k, v = qkv[row].float().split((2048, 2048, 6144))
                query = q.reshape(16, 128)
                direction = k.reshape(16, 128)
                query = query / (query.square().sum(-1, keepdim=True) + 1e-6).sqrt()
                direction = direction / (direction.square().sum(-1, keepdim=True) + 1e-6).sqrt()
                query = query.repeat_interleave(3, dim=0) * (128**-0.5)
                keys = direction.repeat_interleave(3, dim=0)
                b, a = ba[row].float().chunk(2)
                decay = (-a_log.float().exp() * F.softplus(a + dt.float())).exp()
                beta = b.sigmoid().bfloat16().float()
                carry = carry * decay[:, None, None]
                update = (v.reshape(48, 128) - (carry * keys[:, None, :]).sum(-1)) * beta[:, None]
                carry = torch.addcmul(carry, update[:, :, None], keys[:, None, :])
                out.append((carry * query[:, None, :]).sum(-1))
                if token == 0:
                    state[slot].copy_(carry)
                else:
                    key[log_slot, active, token - 1].copy_(direction)
                    value[log_slot, active, token - 1].copy_(update)
                    history.decay[log_slot, active, :, token - 1].copy_(decay)
        return torch.stack(out).to(qkv.dtype)


class ResetReplay(OpWrapper):
    """Clear pending history and the active bank after a direct update."""

    ARCHS = frozenset({Arch.SM_100})

    CELLS = (Cell("Reset nonconsecutive physical slots, skipping padding", dict(slots=(4, 1, -1))),)

    def __call__(self, slots: torch.Tensor, count: torch.Tensor, bank: torch.Tensor) -> None:
        verify.reset_replay(slots, count, bank)

    def reference(self, slots: torch.Tensor, count: torch.Tensor, bank: torch.Tensor) -> None:
        unique = slots[slots >= 0].long()
        count.index_fill_(0, unique, 0)
        bank.index_fill_(0, unique, 0)


verify_convolution = VerifyConvolution()
verify_delta = VerifyDelta()
reset_replay = ResetReplay()


class PreparedVerifyDelta(OpWrapper):
    """Convolve a draft chain while sharing normalized heads across state tiles."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("Four-row verifier across both replay banks", dict(seqs=16, tokens=4, accepted=4)),
        Cell("Two-row verifier across strided physical slots", dict(seqs=64, tokens=2, accepted=2)),
    )

    def __call__(
        self,
        projection,
        weight,
        conv,
        windows,
        temporary_slots,
        ba,
        a_log,
        dt,
        state,
        slots,
        replay_slots,
        history,
        tokens,
    ) -> torch.Tensor:
        query_key, value, gates = prepared_verify.prepare_convolution(
            projection, weight, conv, windows, slots, temporary_slots, ba, a_log, dt, tokens
        )
        return prepared_verify.replay_prepared(
            query_key,
            value,
            gates,
            state,
            slots,
            replay_slots,
            history.count,
            history.bank,
            history.value,
            history.key,
            history.decay,
            tokens,
        )

    def reference(
        self,
        projection,
        weight,
        conv,
        windows,
        temporary_slots,
        ba,
        a_log,
        dt,
        state,
        slots,
        replay_slots,
        history,
        tokens,
    ) -> torch.Tensor:
        convolved = verify_convolution.reference(
            projection, weight, conv, windows, slots, temporary_slots, tokens
        )
        return verify_delta.reference(
            convolved, ba, a_log, dt, state, slots, replay_slots, history, tokens
        )


prepared_verify_delta = PreparedVerifyDelta()

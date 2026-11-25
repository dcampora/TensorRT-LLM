# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated DeltaNet operations for linear speculative verification."""

from typing import NamedTuple

import torch
import torch.nn.functional as F

from .....catalog._op import Cell, OpWrapper
from ..kernels import verify
from .recurrent import _reference_delta


class ReplayHistory(NamedTuple):
    """The manager's double-buffered history for one layer.

    Values and keys use their original BF16 convolution outputs. The two
    BF16 gate logits are stored as one int32 word per value head in the
    manager's 32-bit gate history, so replay reconstructs the exact FP32
    recurrent update without an extra state pool.
    """

    count: torch.Tensor
    bank: torch.Tensor
    value: torch.Tensor
    key: torch.Tensor
    gates: torch.Tensor
    max_tokens: int


class VerifyConvolution(OpWrapper):
    """Convolve each chain without publishing an unaccepted history window."""

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
    """Reconstruct the accepted FP32 state, then run each tentative token."""

    CELLS = (
        Cell("Append a width-five chain to a cold checkpoint", dict(tokens=5, old=0, direct=False)),
        Cell(
            "Replay a previous acceptance and append two tokens",
            dict(tokens=2, old=5, direct=False),
        ),
        Cell(
            "Commit a full history and use the alternate bank", dict(tokens=5, old=13, direct=False)
        ),
        Cell(
            "Commit a single generation token without speculation",
            dict(tokens=1, old=6, direct=True),
        ),
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
        direct: bool = False,
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
            history.gates,
            tokens,
            history.max_tokens,
            direct,
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
        direct: bool = False,
    ) -> torch.Tensor:
        out = []
        gates = history.gates.view(torch.int32)
        capacity = history.value.shape[2]
        for seq, (slot, replay_slot) in enumerate(zip(slots.tolist(), replay_slots.tolist())):
            assert slot >= 0 and replay_slot >= 0
            carry = state[slot].float()
            bank = int(history.bank[replay_slot])
            count = int(history.count[replay_slot])
            for token in range(count):
                packed = gates[replay_slot, bank, :, token].contiguous()
                a = (packed >> 16).short().view(torch.bfloat16).float()
                b = packed.short().view(torch.bfloat16).float()
                v = history.value[replay_slot, bank, token].flatten()
                k = history.key[replay_slot, bank, token].flatten()
                qkv_old = torch.cat((torch.zeros_like(k), k, v))
                decay = -a_log.float().exp() * F.softplus(a + dt.float())
                beta = b.sigmoid().bfloat16().float()
                _, carry = _reference_delta(qkv_old, decay, beta, carry)
            if not direct and count + history.max_tokens > capacity:
                state[slot].copy_(carry)
                bank, count = 1 - bank, 0
            for token in range(tokens):
                row = seq * tokens + token
                b, a = ba[row].chunk(2)
                decay = -a_log.float().exp() * F.softplus(a.float() + dt.float())
                beta = b.float().sigmoid().bfloat16().float()
                y, carry = _reference_delta(qkv[row], decay, beta, carry)
                out.append(y)
                if not direct:
                    history.key[replay_slot, bank, count + token].copy_(
                        qkv[row, 2048:4096].reshape(16, 128)
                    )
                    history.value[replay_slot, bank, count + token].copy_(
                        qkv[row, 4096:].reshape(48, 128)
                    )
                    gates[replay_slot, bank, :, count + token].copy_(
                        (a.view(torch.int16).int() << 16) | (b.view(torch.int16).int() & 0xFFFF)
                    )
            if direct:
                state[slot].copy_(carry)
        return torch.stack(out).to(qkv.dtype)


class ResetReplay(OpWrapper):
    """Clear pending history and the active bank after a direct update."""

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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference and implementation for speculative GDN causal convolution."""

import torch
from torch.nn import functional as F

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels.verify_conv import verify_conv as _verify_conv


class VerifyConv(OpWrapper):
    """Convolve packed candidate rows and stage every candidate's state."""

    ARCHS = frozenset({Arch.SM_100})

    CELLS = (
        Cell(why="batch one with one draft", spec=dict(requests=1, tokens=2)),
        Cell(why="mixed valid slots and padded requests", spec=dict(requests=8, tokens=4)),
    )

    def __call__(self, qkv, weight, state, slots, intermediate, tokens):
        return _verify_conv(qkv, weight, state, slots, intermediate, tokens)

    def reference(self, qkv, weight, state, slots, intermediate, tokens):
        batch = slots.numel()
        channels = weight.shape[0]
        history = state.index_select(0, slots.clamp(min=0))
        history = torch.where(slots[:, None, None] >= 0, history, 0)
        samples = qkv.view(batch, tokens, channels).transpose(1, 2)
        values = torch.cat((history, samples), dim=-1).unfold(-1, 4, 1)
        result = F.silu((values.float() * weight[None, :, None, :].float()).sum(-1))
        intermediate[:, :tokens].copy_(
            torch.where(
                slots[:, None, None, None] >= 0,
                values[..., 1:].transpose(1, 2),
                intermediate[:, :tokens],
            )
        )
        return result.transpose(1, 2).reshape(batch * tokens, channels).to(qkv.dtype)


verify_conv = VerifyConv()

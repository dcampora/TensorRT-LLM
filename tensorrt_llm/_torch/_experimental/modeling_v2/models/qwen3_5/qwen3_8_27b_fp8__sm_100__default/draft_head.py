# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Greedy draft vocabulary adapted from aegis-engines speculative.py (114cf57c911e)."""

import torch
import torch.nn.functional as F


class DraftVocabulary:
    """Sample proposals from a vocabulary prefix plus checkpoint control tokens."""

    def __init__(self, weight: torch.Tensor, control_tokens: list[int]) -> None:
        vocab = 65536
        extras = sorted({token for token in control_tokens if vocab <= token < weight.shape[0]})
        prefix = vocab - len(extras)
        if extras and prefix > 0:
            extra_ids = torch.tensor(extras, device=weight.device)
            self.weight = torch.cat((weight[:prefix], weight.index_select(0, extra_ids)))
            self.token_ids = torch.cat((torch.arange(prefix, device=weight.device), extra_ids))
        else:
            self.weight = weight[:vocab]
            self.token_ids = None

    @staticmethod
    def supports(rows: int) -> bool:
        """The projection handles one row per request, including padded batches."""
        return rows > 0

    def sample(self, hidden: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Return proposal IDs in the full vocabulary's numbering."""
        indices = F.linear(hidden, self.weight).argmax(dim=-1)
        if self.token_ids is not None:
            indices = self.token_ids.index_select(0, indices)
        if out is not None:
            out.copy_(indices)
            return out
        return indices

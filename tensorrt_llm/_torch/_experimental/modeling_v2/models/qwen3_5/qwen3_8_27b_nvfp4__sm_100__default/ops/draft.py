# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare the concatenated BF16 MTP feature projection."""

import flashinfer as fi
import torch

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels.draft_front import normalized_features


class DraftFeatures(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(why="First proposal with a batch-1 verification", spec={"rows": 5}),
        Cell(why="Batched draft step", spec={"rows": 128}),
    )

    def __call__(
        self,
        ids: torch.Tensor,
        embedding: torch.Tensor,
        hidden: torch.Tensor,
        embedding_norm: torch.Tensor,
        hidden_norm: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return normalized_features(ids, embedding, hidden, embedding_norm, hidden_norm, eps)

    def reference(
        self,
        ids: torch.Tensor,
        embedding: torch.Tensor,
        hidden: torch.Tensor,
        embedding_norm: torch.Tensor,
        hidden_norm: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        tokens = embedding.index_select(0, ids.clamp_min(0))
        return torch.cat(
            (
                fi.gemma_rmsnorm(tokens, embedding_norm, eps),
                fi.gemma_rmsnorm(hidden, hidden_norm, eps),
            ),
            dim=-1,
        )


draft_features = DraftFeatures()

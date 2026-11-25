# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint's BF16 draft layer, adapted from aegis-engines cb6779a, engine.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

import flashinfer as fi
import torch
import torch.nn.functional as F
import triton as tr
from torch import nn

from .ops.attention import apply_gate, prepare_qkv
from .ops.draft import draft_features
from .ops.paged import append_kv, bind_layer, decode_attention, paged_context_attention

if TYPE_CHECKING:
    from .modeling import Qwen38Decoder
    from .ops.projections import Fp4Linear


def _draft_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if x.shape[0] <= 8:
        return fi.mm_bf16(x, weight.T, backend="cute-dsl")
    return F.linear(x, weight)


class MtpLayer(nn.Module):
    supports_pick_rows = True

    def __init__(self, index: int, eps: float) -> None:
        super().__init__()
        self.index = index
        self.eps = eps
        for name in (
            "embedding_norm",
            "hidden_norm",
            "fc",
            "input_norm",
            "post_norm",
            "qkv",
            "q_norm",
            "k_norm",
            "out",
            "gate_up",
            "down",
            "final_norm",
        ):
            self.register_buffer(name, None)
        self.register_buffer("rope", None, persistent=False)
        self._manager = None
        self._cache = None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        embed_tokens: torch.Tensor,
        attn_metadata,
        all_rank_num_tokens=None,
        spec_metadata=None,
        pick_rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del all_rank_num_tokens, spec_metadata
        md = attn_metadata
        rows = input_ids.shape[0]
        positions = position_ids.reshape(-1, rows)[0]
        features = draft_features(
            input_ids, embed_tokens, hidden_states, self.embedding_norm, self.hidden_norm, self.eps
        )
        residual = _draft_linear(features, self.fc)
        x = fi.gemma_rmsnorm(residual, self.input_norm, self.eps)
        projection = _draft_linear(x, self.qkv)
        q, k, v, gate = prepare_qkv(
            projection, self.q_norm, self.k_norm, self.rope, positions, self.eps
        )

        manager = md.kv_cache_manager
        if manager is not self._manager:
            # A shared-cache draft view addresses the MTP layer at local index zero.
            index = self.index if self.index in manager.layer_offsets else 0
            self._cache = bind_layer(manager, index, md.host_kv_cache_pool_mapping)
            self._manager = manager
        cache = self._cache
        table = md.kv_cache_block_offsets[cache.group, :, 0]
        contexts = md.num_contexts
        ctx_rows = md.num_ctx_tokens if contexts else 0
        outputs = []
        if contexts:
            lengths = md.seq_lens[:contexts].tolist()
            chunks = torch.tensor(
                [
                    (i, tile)
                    for i, length in enumerate(lengths)
                    for tile in range(tr.cdiv(length, 64))
                ],
                device=q.device,
                dtype=torch.int32,
            )
            mamba = md.mamba_metadata
            req = mamba.seq_idx.reshape(-1)[:ctx_rows]
            starts = mamba.cu_seqlens[: contexts + 1]
            append_kv(k[:ctx_rows], v[:ctx_rows], positions[:ctx_rows], cache, table, req)
            output = paged_context_attention(
                q[:ctx_rows], starts, chunks, positions[:ctx_rows], cache, table
            )
            outputs.append(apply_gate(output, gate[:ctx_rows]))
        if ctx_rows < rows:
            width = (rows - ctx_rows) // (md.num_seqs - contexts)
            qd, gd, kd, vd = (x[ctx_rows:] for x in (q, gate, k, v))
            pd = positions[ctx_rows:]
            append_kv(kd, vd, pd, cache, table, None, width, contexts)
            outputs.append(
                decode_attention(
                    qd, gd, pd, cache, table, rows_per_seq=width, request_start=contexts
                )
            )
        x = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
        if pick_rows is not None:
            x = x.index_select(0, pick_rows)
            residual = residual.index_select(0, pick_rows)
        x = _draft_linear(x, self.out)
        fi.gemma_fused_add_rmsnorm(x, residual, self.post_norm, self.eps)
        x = F.linear(fi.silu_and_mul(F.linear(x, self.gate_up)), self.down)
        fi.gemma_fused_add_rmsnorm(x, residual, self.final_norm, self.eps)
        return x

    def shared_head(
        self,
        hidden_states: torch.Tensor,
        lm_head: Fp4Linear,
        attn_metadata,
        return_context_logits: bool,
    ) -> torch.Tensor:
        return lm_head(hidden_states)


class MtpDraft:
    """The split MTP worker's view of its layer and the shared embedding and head."""

    def __init__(self, model: Qwen38Decoder, layer: MtpLayer, lm_head: Fp4Linear) -> None:
        self.core = model
        self.mtp_layers = (layer,)
        self.lm_head = lm_head

    @property
    def embed_tokens(self) -> torch.Tensor:
        return self.core.embedding

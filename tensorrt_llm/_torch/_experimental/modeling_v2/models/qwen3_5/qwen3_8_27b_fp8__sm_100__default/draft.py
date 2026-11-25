# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The checkpoint's shared one-layer MTP-Eagle draft decoder."""

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from .modeling import DecoderLayer, residual_norm
from .ops.activation import attention_gate_quantize, draft_input_norm
from .ops.attention import prepare_qkv
from .ops.paged_attention import attention_arguments, paged_attention

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.model_config import ModelConfig


class DraftLayer(DecoderLayer):
    """Predict with one full-attention layer and its own paged draft KV."""

    def __init__(self, model_config: "ModelConfig", index: int, aux_stream_dict: dict) -> None:
        super().__init__(index, is_linear=False)
        for name in ("fc", "embedding_norm", "hidden_norm", "output_norm", "rope"):
            self.register_buffer(name, None)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        embed_tokens: nn.Module,
        attn_metadata: "TrtllmAttentionMetadata",
        **kwargs: object,
    ) -> torch.Tensor:
        """Extend the draft cache using the runtime's shifted token stream."""
        inputs = draft_input_norm(
            input_ids.contiguous(),
            embed_tokens.weight,
            hidden_states.contiguous(),
            self.embedding_norm,
            self.hidden_norm,
        )
        hidden = F.linear(inputs, self.fc)
        hidden, residual = residual_norm(hidden, None, self.pre_norm)
        context = attn_metadata.num_contexts
        generation = attn_metadata.num_seqs - context
        gen_tokens = hidden.shape[0] - attn_metadata.num_ctx_tokens
        predicted = gen_tokens // generation if generation else 1
        decode = not context
        projection = self.input_proj(hidden, decode)
        q, k, v = prepare_qkv(
            projection,
            self.q_norm,
            self.k_norm,
            self.rope,
            position_ids.reshape(-1, position_ids.shape[-1])[0],
            tiled=not decode,
        )
        layer = attn_metadata.kv_cache_manager.layer_offsets[self.index]
        result = paged_attention(q, k, v, layer, attention_arguments(attn_metadata, predicted))
        hidden = self.output_proj.quantized(
            *attention_gate_quantize(result, projection), projection.shape[0], decode
        )
        hidden, residual = self.feed_forward(hidden, residual, decode, fused=True)
        return residual_norm(hidden, residual, self.output_norm)[0]

    @staticmethod
    def shared_head(
        hidden: torch.Tensor,
        lm_head: nn.Module,
        attn_metadata: "TrtllmAttentionMetadata",
        gather: bool,
    ) -> torch.Tensor:
        """Read the tied verifier vocabulary weights for MTP-Eagle logits."""
        return lm_head(hidden)

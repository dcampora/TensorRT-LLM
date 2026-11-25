# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-only Qwen3.8-27B on SM100.

The layer order and calibrated projections follow aegis-engines cb6779a.
TensorRT-LLM supplies packed iteration metadata and owns all request state.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import accumulate
from typing import TYPE_CHECKING, NamedTuple

import flashinfer as fi
import torch
import torch.nn.functional as F
import triton as tr
from torch import nn

from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM

from ...._router_index import disable_v2_by_default
from . import weights
from .ops.attention import apply_gate, prepare_qkv
from .ops.gdn import Context, gated_rmsnorm, gdn_context, gdn_decode
from .ops.paged import (
    Cache,
    append_kv,
    bind_layer,
    context_attention,
    decode_attention,
    paged_context_attention,
)
from .ops.projections import Fp4Linear, Fp8Linear

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tensorrt_llm._torch.model_config import ModelConfig

_PREFILL_MIN_ROWS = 128


def _pad(x: torch.Tensor, rows: int) -> torch.Tensor:
    return F.pad(x, (0, 0, 0, rows - x.shape[0])) if rows > x.shape[0] else x


def _device_ints(values: list, device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)


class Step(NamedTuple):
    context: Context | None
    context_rows: int
    context_indices: torch.Tensor | None
    fresh_context: bool
    context_spans: tuple[tuple[int, int], ...]
    state_slots: torch.Tensor
    block_offsets: torch.Tensor
    projection_rows: int
    mlp_rows: torch.Tensor
    positions: torch.Tensor


class Layer(nn.Module):
    def __init__(self, index: int, kind: str, eps: float) -> None:
        super().__init__()
        self.index = index
        self.eps = eps
        self.is_linear = kind == "linear_attention"
        self.qkv = Fp8Linear()
        self.out = Fp8Linear()
        self.gate_up = Fp4Linear()
        self.down = Fp4Linear()
        self.register_buffer("input_norm", None)
        self.register_buffer("post_norm", None)
        if self.is_linear:
            for name in ("ba", "conv_weight", "log_decay", "bias", "gate_norm"):
                self.register_buffer(name, None)
        else:
            self.register_buffer("q_norm", None)
            self.register_buffer("k_norm", None)

    def linear_attention(self, x: torch.Tensor, step: Step, cache) -> torch.Tensor:
        rows = x.shape[0]
        context_rows = step.context_rows
        projection = self.qkv(_pad(x, step.projection_rows), decode=not context_rows)[:rows]
        outputs = []
        if context_rows:
            if len(step.context_spans) == 1 or step.projection_rows == _PREFILL_MIN_ROWS:
                ba = F.linear(_pad(x[:context_rows], step.projection_rows), self.ba)[:context_rows]
            else:
                ba = torch.cat(
                    [
                        F.linear(_pad(x[first:end], _PREFILL_MIN_ROWS), self.ba)[: end - first]
                        for first, end in step.context_spans
                    ]
                )
            prefills = gdn_context(
                projection[:context_rows],
                ba,
                self.conv_weight,
                self.log_decay,
                self.bias,
                cache.conv,
                cache.temporal,
                step.context,
            )
            outputs.append(
                gated_rmsnorm(prefills, projection[:context_rows, 10240:], self.gate_norm, self.eps)
            )
        if context_rows < rows:
            decode_rows = rows - context_rows
            ba = F.linear(_pad(x[context_rows:], _PREFILL_MIN_ROWS), self.ba)[:decode_rows]
            decoded = gdn_decode(
                projection[context_rows:],
                ba,
                self.conv_weight,
                self.log_decay,
                self.bias,
                cache.conv,
                cache.temporal,
                step.state_slots[len(step.context_spans) :],
            )
            if not context_rows:
                packed = gated_rmsnorm(
                    decoded,
                    projection[:, 10240:],
                    self.gate_norm,
                    self.eps,
                    self.out.input_scale,
                    decode=True,
                )
                return self.out.gemm(packed, decode=True)
            outputs.append(
                gated_rmsnorm(
                    decoded,
                    projection[context_rows:, 10240:],
                    self.gate_norm,
                    self.eps,
                    decode=True,
                )
            )
        result = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
        return self.out(_pad(result, step.projection_rows))[:rows]

    def full_attention(
        self, x: torch.Tensor, step: Step, rope: torch.Tensor, cache: Cache
    ) -> torch.Tensor:
        rows = x.shape[0]
        decode = not step.context_rows
        projection = self.qkv(_pad(x, step.projection_rows), decode=decode)[:rows]
        q, k, v, gate = prepare_qkv(
            projection, self.q_norm, self.k_norm, rope, step.positions, self.eps
        )
        table = step.block_offsets[cache.group, :, 0]
        if decode:
            append_kv(k, v, step.positions, cache, table, None)
            result = decode_attention(q, gate, step.positions, cache, table, self.out.input_scale)
            return self.out.gemm(result, decode=True)

        ctx = step.context_rows
        positions = step.positions[:ctx]
        append_kv(k[:ctx], v[:ctx], positions, cache, table, step.context_indices)
        if step.fresh_context:
            output = context_attention(
                q[:ctx], k[:ctx], v[:ctx], step.context.starts, step.context.chunks
            )
        else:
            output = paged_context_attention(
                q[:ctx], step.context.starts, step.context.chunks, positions, cache, table
            )
        output = apply_gate(output, gate[:ctx])
        if ctx < rows:
            positions = step.positions[ctx:]
            first_request = len(step.context_spans)
            append_kv(k[ctx:], v[ctx:], positions, cache, table, None, request_start=first_request)
            generated = decode_attention(
                q[ctx:], gate[ctx:], positions, cache, table, request_start=first_request
            )
            output = torch.cat((output, generated))
        return self.out(_pad(output, step.projection_rows))[:rows]

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        step: Step,
        rope: torch.Tensor,
        layer_cache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = x
            x = fi.gemma_rmsnorm(x, self.input_norm, self.eps)
        else:
            fi.gemma_fused_add_rmsnorm(x, residual, self.input_norm, self.eps)

        if self.is_linear:
            x = self.linear_attention(x, step, layer_cache)
        else:
            x = self.full_attention(x, step, rope, layer_cache)
        x = x.contiguous()
        fi.gemma_fused_add_rmsnorm(x, residual, self.post_norm, self.eps)
        intermediate = self.gate_up(_pad(x, step.projection_rows))
        x = self.down.swiglu(intermediate, step.mlp_rows, decode=not step.context_rows)
        return x[: residual.shape[0]].contiguous(), residual


class Qwen38Decoder(DecoderModel):
    supports_prefill_cuda_graph = False

    def __init__(self, model_config: ModelConfig) -> None:
        super().__init__(model_config)
        config = model_config.pretrained_config.get_text_config()
        self.eps = config.rms_norm_eps
        self.layers = nn.ModuleList(
            Layer(i, kind, self.eps) for i, kind in enumerate(config.layer_types)
        )
        self.register_buffer("embedding", None)
        self.register_buffer("norm_weight", None)
        self.register_buffer("rope", None, persistent=False)
        self.register_buffer("row_counts", None, persistent=False)
        self._manager = None
        self._caches = ()

    def forward(
        self,
        attn_metadata,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        md = attn_metadata
        manager = md.kv_cache_manager
        if manager is not self._manager:
            self._caches = tuple(
                manager.mamba_layer_cache(layer.index)
                if layer.is_linear
                else bind_layer(manager, layer.index, md.host_kv_cache_pool_mapping)
                for layer in self.layers
            )
            self._manager = manager

        x = self.embedding.index_select(0, input_ids)
        positions = position_ids.reshape(-1, x.shape[0])[0]
        contexts = md.num_contexts
        ctx_rows = md.num_ctx_tokens
        projection_rows = max(x.shape[0], _PREFILL_MIN_ROWS) if contexts else x.shape[0]
        mamba = md.mamba_metadata
        state_slots = mamba.state_indices[: md.num_seqs]
        context = None
        spans = ()
        if contexts:
            lengths = md.seq_lens[:contexts].tolist()
            starts = list(accumulate(lengths, initial=0))
            spans = tuple(zip(starts[:-1], starts[1:], strict=True))
            counts = [tr.cdiv(length, 64) for length in lengths]
            chunks = [(seq, chunk) for seq, count in enumerate(counts) for chunk in range(count)]
            context = Context(
                mamba.cu_seqlens[: contexts + 1],
                _device_ints(chunks, x.device),
                _device_ints(list(accumulate(counts, initial=0)), x.device),
                state_slots[:contexts],
                mamba.has_initial_states[:contexts],
                contexts,
                max(lengths),
                min(64, max(16, tr.next_power_of_2(ctx_rows))),
            )
        step = Step(
            context,
            ctx_rows,
            mamba.seq_idx.reshape(-1)[:ctx_rows] if contexts else None,
            not mamba.use_initial_states,
            spans,
            state_slots,
            md.kv_cache_block_offsets,
            projection_rows,
            self.row_counts[projection_rows : projection_rows + 1],
            positions,
        )
        residual = None
        for layer, cache in zip(self.layers, self._caches, strict=True):
            x, residual = layer(x, residual, step, self.rope, cache)
        fi.gemma_fused_add_rmsnorm(x, residual, self.norm_weight, self.eps)
        return x


@disable_v2_by_default
class ModelingV2Qwen38Nvfp4Sm100Default(DecoderModelForCausalLM):
    """Qwen3.8-27B ModelOpt mixed FP8/NVFP4 language model."""

    def __init__(self, model_config: ModelConfig) -> None:
        nn.Module.__init__(self)
        # The loader routes the composite HF architecture. The target
        # exposes the text config to the hybrid cache manager.
        text = model_config.pretrained_config.get_text_config()
        self.model_config = replace(model_config, pretrained_config=text)
        self.model_config.extra_attrs = model_config.extra_attrs
        self.model = Qwen38Decoder(self.model_config)
        self.lm_head = Fp4Linear()
        self.pp_size = 1
        self.pp_rank = 0
        self.has_custom_lm_head = False
        self.prologue = []
        self.epilogue = [self.lm_head]

    def __post_init__(self) -> None:
        pass

    def load_weights(self, weights_dict: Mapping[str, torch.Tensor]) -> None:
        weights.load(self, weights_dict)

    def post_load_weights(self) -> None:
        model = self.model
        text = self.config.get_text_config()
        device = model.embedding.device
        rope = text.rope_parameters
        freq = 1.0 / (
            rope["rope_theta"] ** (torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64)
        )
        phase = torch.outer(
            torch.arange(text.max_position_embeddings, device=device, dtype=torch.float32), freq
        )
        model.rope = torch.cat((phase.cos(), phase.sin()), dim=-1)
        model.row_counts = torch.arange(
            max(self.model_config.max_num_tokens, _PREFILL_MIN_ROWS) + 1,
            dtype=torch.int32,
            device=device,
        )

    def forward(
        self,
        attn_metadata,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            raise ValueError("Qwen3.8-27B text target requires token IDs")
        hidden = self.model(attn_metadata, input_ids, position_ids)
        if not return_context_logits:
            end = torch.cumsum(attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
            hidden = hidden.index_select(0, end)
        return self.lm_head(hidden).float()

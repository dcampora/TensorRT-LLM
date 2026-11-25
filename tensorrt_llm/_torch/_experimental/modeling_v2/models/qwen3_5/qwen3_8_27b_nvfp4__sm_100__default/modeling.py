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
from tensorrt_llm._torch.speculative.interface import should_use_separate_draft_kv_cache
from tensorrt_llm._torch.speculative.utils import get_spec_worker
from tensorrt_llm._torch.utils import is_gdn_replay_enabled

from ...._router_index import disable_v2_by_default
from . import weights
from .mtp import MtpDraft, MtpLayer
from .ops.attention import apply_gate, prepare_qkv
from .ops.gdn import Context, Replay, gated_rmsnorm, gdn_context, gdn_decode, gdn_verify
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
    decode_width: int
    verify: bool
    replay_slots: torch.Tensor | None


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
            x_decode = x[context_rows:]
            if decode_rows <= _PREFILL_MIN_ROWS:
                ba = F.linear(_pad(x_decode, _PREFILL_MIN_ROWS), self.ba)[:decode_rows]
            else:
                ba = torch.cat(
                    [
                        F.linear(_pad(tile, _PREFILL_MIN_ROWS), self.ba)[: tile.shape[0]]
                        for tile in x_decode.split(_PREFILL_MIN_ROWS)
                    ]
                )
            args = (
                projection[context_rows:],
                ba,
                self.conv_weight,
                self.log_decay,
                self.bias,
                cache.conv,
                cache.temporal,
                step.state_slots[len(step.context_spans) :],
            )
            if step.verify:
                replay = None
                if cache.old_x is not None:
                    replay = Replay(
                        step.replay_slots,
                        cache.prev_num_accepted_tokens,
                        cache.cache_buf_idx,
                        cache.old_x,
                        cache.old_B,
                        cache.old_dt,
                    )
                decoded = gdn_verify(
                    *args,
                    cache.intermediate_conv_window,
                    cache.intermediate_ssm,
                    step.decode_width,
                    replay,
                )
            else:
                decoded = gdn_decode(*args)
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
            append_kv(k, v, step.positions, cache, table, None, step.decode_width)
            result = decode_attention(
                q, gate, step.positions, cache, table, self.out.input_scale, step.decode_width
            )
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
            append_kv(
                k[ctx:], v[ctx:], positions, cache, table, None, step.decode_width, first_request
            )
            generated = decode_attention(
                q[ctx:],
                gate[ctx:],
                positions,
                cache,
                table,
                rows_per_seq=step.decode_width,
                request_start=first_request,
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
        self.num_trunk_layers = len(self.layers)
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
        verify: bool = False,
    ) -> torch.Tensor:
        md = attn_metadata
        manager = md.kv_cache_manager
        if manager is not self._manager:
            self._caches = tuple(
                manager.mamba_layer_cache(layer.index)
                if layer.is_linear
                else bind_layer(manager, layer.index, md.host_kv_cache_pool_mapping)
                for layer in self.layers[: self.num_trunk_layers]
            )
            self._manager = manager

        x = self.embedding.index_select(0, input_ids)
        positions = position_ids.reshape(-1, x.shape[0])[0]
        contexts = md.num_contexts
        ctx_rows = md.num_ctx_tokens
        num_decodes = md.num_seqs - contexts
        decode_width = (x.shape[0] - ctx_rows) // num_decodes if num_decodes else 1
        projection_rows = max(x.shape[0], _PREFILL_MIN_ROWS) if contexts else x.shape[0]
        mamba = md.mamba_metadata
        state_slots = mamba.state_indices[: md.num_seqs]
        replay_slots = mamba.replay_state_indices
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
            decode_width,
            verify,
            replay_slots[contexts : md.num_seqs] if replay_slots is not None else None,
        )
        residual = None
        for i, cache in enumerate(self._caches):
            x, residual = self.layers[i](x, residual, step, self.rope, cache)
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
        self.spec_config = model_config.spec_config
        self.draft_config = None
        self.draft_model = None
        self.spec_worker = None
        if self.spec_config is not None:
            text.num_nextn_predict_layers = 1
            if is_gdn_replay_enabled():
                text._gdn_replay_history_dtype = torch.float32
            draft_layer = MtpLayer(text.num_hidden_layers, text.rms_norm_eps)
            self.model.layers.append(draft_layer)
            self.draft_model = MtpDraft(self.model, draft_layer, self.lm_head)
            self.use_separate_draft_kv_cache = should_use_separate_draft_kv_cache(self.spec_config)
            self.spec_worker = get_spec_worker(
                self.spec_config,
                self.model_config,
                self.model_config.mapping,
                use_separate_draft_kv_cache=self.use_separate_draft_kv_cache,
            )
            self.spec_worker.set_draft_model(self.draft_model)

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
        if self.draft_model is not None:
            self.draft_model.mtp_layers[0].rope = model.rope
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
        spec_metadata=None,
        resource_manager=None,
        **kwargs,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if inputs_embeds is not None:
            raise ValueError("Qwen3.8-27B text target requires token IDs")
        speculative = self.spec_worker is not None
        hidden = self.model(attn_metadata, input_ids, position_ids, speculative)
        if speculative:
            logits = self.lm_head(hidden.index_select(0, spec_metadata.gather_ids)).float()
            return self.spec_worker(
                input_ids=input_ids,
                position_ids=position_ids,
                hidden_states=hidden,
                logits=logits,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                draft_model=self.draft_model,
                resource_manager=resource_manager,
            )
        if not return_context_logits:
            end = torch.cumsum(attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
            hidden = hidden.index_select(0, end)
        return self.lm_head(hidden).float()

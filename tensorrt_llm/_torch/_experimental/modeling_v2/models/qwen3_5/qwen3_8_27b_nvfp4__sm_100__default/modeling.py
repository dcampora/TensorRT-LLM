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
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import is_in_breakable_cuda_graph
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks
from tensorrt_llm._torch.speculative.interface import should_use_separate_draft_kv_cache
from tensorrt_llm._torch.speculative.utils import get_spec_worker
from tensorrt_llm._torch.utils import is_gdn_replay_enabled

from ...._router_index import disable_v2_by_default
from . import weights
from .mtp import MtpDraft, MtpLayer
from .ops.attention import apply_gate, fp8_qkv_partitions, prepare_qkv
from .ops.gdn import Context, Replay, gated_rmsnorm, gdn_context, gdn_decode, gdn_verify
from .ops.paged import (
    Cache,
    append_kv,
    bind_layer,
    context_attention,
    decode_attention,
    paged_context_attention,
)
from .ops.projections import Fp4Linear, Fp8Linear, fp8_quantize
from .prefill_graph import PrefillGraph

if TYPE_CHECKING:
    from collections.abc import Mapping

    from tensorrt_llm._torch.model_config import ModelConfig

_PREFILL_MIN_ROWS = 128


def _pad(x: torch.Tensor, rows: int) -> torch.Tensor:
    return F.pad(x, (0, 0, 0, rows - x.shape[0])) if rows > x.shape[0] else x


def _device_ints(values: list, device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, pin_memory=True).to(device, non_blocking=True)


def _context_groups(lengths: list[int]):
    """Pack short contexts at their calibrated projection row count."""
    first, rows = 0, 0
    short = lengths[0] <= _PREFILL_MIN_ROWS
    for seq, length in enumerate(lengths):
        next_short = length <= _PREFILL_MIN_ROWS
        if next_short != short or (short and rows + length > _PREFILL_MIN_ROWS):
            yield first, seq
            first, rows, short = seq, 0, next_short
        rows += length
    yield first, len(lengths)


class Step(NamedTuple):
    context: Context | None
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
    dummy: torch.Tensor | None


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

    def linear_attention(
        self, x: torch.Tensor, projection: torch.Tensor, step: Step, cache
    ) -> torch.Tensor:
        rows = x.shape[0]
        z = projection[:, 10240:]
        if step.context is not None:
            if len(step.context_spans) == 1 or step.projection_rows == _PREFILL_MIN_ROWS:
                ba = F.linear(_pad(x, step.projection_rows), self.ba)[:rows]
            else:
                ba = torch.cat(
                    [
                        F.linear(_pad(x[first:end], _PREFILL_MIN_ROWS), self.ba)[: end - first]
                        for first, end in step.context_spans
                    ]
                )
            prefills = gdn_context(
                projection,
                ba,
                self.conv_weight,
                self.log_decay,
                self.bias,
                cache.conv,
                cache.temporal,
                step.context,
            )
            return gated_rmsnorm(prefills, z, self.gate_norm, self.eps)

        if rows <= _PREFILL_MIN_ROWS:
            ba = F.linear(_pad(x, _PREFILL_MIN_ROWS), self.ba)[:rows]
        else:
            ba = torch.cat(
                [
                    F.linear(_pad(tile, _PREFILL_MIN_ROWS), self.ba)[: tile.shape[0]]
                    for tile in x.split(_PREFILL_MIN_ROWS)
                ]
            )
        args = (
            projection,
            ba,
            self.conv_weight,
            self.log_decay,
            self.bias,
            cache.conv,
            cache.temporal,
            step.state_slots,
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
                dummy=step.dummy,
            )
        else:
            decoded = gdn_decode(*args)
        return gated_rmsnorm(
            decoded, z, self.gate_norm, self.eps, self.out.input_scale, decode=True
        )

    def full_attention(
        self, projection: torch.Tensor, step: Step, rope: torch.Tensor, cache: Cache
    ) -> torch.Tensor:
        table = step.block_offsets[cache.group, :, 0]
        if step.context is None:
            scales = (self.qkv.input_scale, self.qkv.weight_scale) if projection.ndim == 3 else None
            q, _, _, gate = prepare_qkv(
                projection,
                self.q_norm,
                self.k_norm,
                rope,
                step.positions,
                self.eps,
                scales=scales,
                cache=cache,
                table=table,
                rows_per_seq=step.decode_width,
            )
            return decode_attention(
                q, gate, step.positions, cache, table, self.out.input_scale, step.decode_width
            )
        q, k, v, gate = prepare_qkv(
            projection, self.q_norm, self.k_norm, rope, step.positions, self.eps
        )
        append_kv(k, v, step.positions, cache, table, step.context_indices)
        starts, chunks = step.context.starts, step.context.chunks
        if step.fresh_context:
            output = context_attention(q, k, v, starts, chunks)
        else:
            output = paged_context_attention(q, starts, chunks, step.positions, cache, table)
        return apply_gate(output, gate)

    def attention_projections(self, x: torch.Tensor, step: Step) -> torch.Tensor:
        rows = x.shape[0]
        decode = step.context is None
        if decode and not self.is_linear and (rows <= 8 or rows in (128, 256)):
            packed = fp8_quantize(x, self.qkv.input_scale, pdl=rows <= 8)
            return fp8_qkv_partitions(packed, self.qkv.weight)
        return self.qkv(_pad(x, step.projection_rows), decode=decode)[:rows]

    def attention_core(
        self, x: torch.Tensor, projection: torch.Tensor, step: Step, rope: torch.Tensor, cache
    ) -> torch.Tensor:
        if self.is_linear:
            return self.linear_attention(x, projection, step, cache)
        return self.full_attention(projection, step, rope, cache)

    def project_attention(self, x: torch.Tensor, step: Step) -> torch.Tensor:
        if step.context is None:
            return self.out.gemm(x, decode=True)
        return self.out(_pad(x, step.projection_rows))[: x.shape[0]]

    def input_rms_norm(
        self, x: torch.Tensor, residual: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return fi.gemma_rmsnorm(x, self.input_norm, self.eps), x
        fi.gemma_fused_add_rmsnorm(x, residual, self.input_norm, self.eps)
        return x, residual

    def post_rms_norm(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        fi.gemma_fused_add_rmsnorm(x, residual, self.post_norm, self.eps)
        return x

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        step: Step,
        rope: torch.Tensor,
        layer_cache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_rms_norm(x, residual)
        x = self.attention_core(x, self.attention_projections(x, step), step, rope, layer_cache)
        x = self.post_rms_norm(self.project_attention(x, step), residual)
        intermediate = self.gate_up(_pad(x, step.projection_rows))
        x = self.down.swiglu(intermediate, step.mlp_rows, decode=step.context is None)
        return x[: residual.shape[0]].contiguous(), residual

    def forward_grouped(
        self,
        hidden: list[torch.Tensor],
        residuals: list[torch.Tensor | None],
        steps: list[Step],
        rope: torch.Tensor,
        cache,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        # Schedule the same weight matrix back to back across groups.
        pairs = [self.input_rms_norm(x, r) for x, r in zip(hidden, residuals, strict=True)]
        normalized, residuals = zip(*pairs, strict=True)
        del pairs
        projections = [
            self.attention_projections(x, step) for x, step in zip(normalized, steps, strict=True)
        ]
        attn = [
            self.attention_core(x, p, step, rope, cache)
            for x, p, step in zip(normalized, projections, steps, strict=True)
        ]
        del projections, normalized
        projected = [self.project_attention(x, step) for x, step in zip(attn, steps, strict=True)]
        del attn
        intermediate = [
            self.gate_up(_pad(self.post_rms_norm(x, r), step.projection_rows))
            for x, r, step in zip(projected, residuals, steps, strict=True)
        ]
        del projected
        output = [
            self.down.swiglu(x, step.mlp_rows, decode=step.context is None)[
                : r.shape[0]
            ].contiguous()
            for x, r, step in zip(intermediate, residuals, steps, strict=True)
        ]
        return output, list(residuals)


class Qwen38Decoder(DecoderModel):
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
        self.prefill_graph = PrefillGraph(self)

    def forward(
        self,
        attn_metadata,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        verify: bool = False,
    ) -> torch.Tensor:
        if attn_metadata.num_contexts and is_in_breakable_cuda_graph():
            return self.prefill_graph.run(input_ids)
        return self.compute(attn_metadata, input_ids, position_ids, verify)

    def bind_caches(self, metadata) -> None:
        manager = metadata.kv_cache_manager
        if manager is not self._manager:
            self._caches = tuple(
                manager.mamba_layer_cache(layer.index)
                if layer.is_linear
                else bind_layer(manager, layer.index, metadata.host_kv_cache_pool_mapping)
                for layer in self.layers[: self.num_trunk_layers]
            )
            self._manager = manager

    def compute(self, md, input_ids, position_ids, verify: bool) -> torch.Tensor:
        self.bind_caches(md)
        x = self.embedding.index_select(0, input_ids)
        positions = position_ids.reshape(-1, x.shape[0])[0]
        contexts = md.num_contexts
        ctx_rows = md.num_ctx_tokens
        num_decodes = md.num_seqs - contexts
        mamba = md.mamba_metadata
        hidden = []
        steps = []
        # In a mixed iteration each phase retains its own projection tactics
        # and rounding. The worker receives rows in the runtime's packed order.
        if contexts:
            lengths = md.seq_lens[:contexts].tolist()
            starts = list(accumulate(lengths, initial=0))
            for first, last in _context_groups(lengths):
                begin, end = starts[first], starts[last]
                step = self._context_step(
                    md, positions[begin:end], lengths[first:last], first, last, begin, end, x.device
                )
                hidden.append(x[begin:end])
                steps.append(step)
        if num_decodes:
            decode_rows = x.shape[0] - ctx_rows
            replay_slots = mamba.replay_state_indices
            dummy = md.kv_cache_manager._dummy_request_mask if verify else None
            step = Step(
                None,
                None,
                False,
                (),
                mamba.state_indices[contexts : md.num_seqs],
                md.kv_cache_block_offsets[:, contexts : md.num_seqs],
                decode_rows,
                self.row_counts[decode_rows : decode_rows + 1],
                positions[ctx_rows:],
                decode_rows // num_decodes,
                verify,
                replay_slots[contexts : md.num_seqs] if replay_slots is not None else None,
                dummy[contexts : md.num_seqs] if dummy is not None else None,
            )
            hidden.append(x[ctx_rows:])
            steps.append(step)
        if len(steps) == 1:
            return self._compute_layers(hidden[0], steps[0])
        residuals = [None] * len(steps)
        for i, cache in enumerate(self._caches):
            hidden, residuals = self.layers[i].forward_grouped(
                hidden, residuals, steps, self.rope, cache
            )
        for x, r in zip(hidden, residuals, strict=True):
            fi.gemma_fused_add_rmsnorm(x, r, self.norm_weight, self.eps)
        return torch.cat(hidden)

    def _context_step(
        self,
        md,
        positions,
        lengths: list[int],
        first: int,
        last: int,
        begin: int,
        end: int,
        device: torch.device,
    ) -> Step:
        mamba = md.mamba_metadata
        spans = list(accumulate(lengths, initial=0))
        counts = [tr.cdiv(length, 64) for length in lengths]
        chunks = [(seq, chunk) for seq, count in enumerate(counts) for chunk in range(count)]
        slots = mamba.state_indices[first:last]
        starts = mamba.cu_seqlens[first : last + 1]
        indices = mamba.seq_idx.reshape(-1)[begin:end]
        if first:
            starts = starts - begin
            indices = indices - first
        context = Context(
            starts,
            _device_ints(chunks, device),
            _device_ints(list(accumulate(counts, initial=0)), device),
            slots,
            mamba.has_initial_states[first:last],
            last - first,
            max(lengths),
            min(64, max(16, tr.next_power_of_2(end - begin))),
        )
        projection_rows = max(end - begin, _PREFILL_MIN_ROWS)
        return Step(
            context,
            indices,
            not mamba.use_initial_states,
            tuple(zip(spans[:-1], spans[1:], strict=True)),
            slots,
            md.kv_cache_block_offsets[:, first:last],
            projection_rows,
            self.row_counts[projection_rows : projection_rows + 1],
            positions,
            1,
            False,
            None,
            None,
        )

    def _compute_layers(self, x: torch.Tensor, step: Step) -> torch.Tensor:
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
        self.spec_decode_fast_paths = self.spec_config is not None
        self.runtime_hooks = (
            RuntimeHooks(loop_on_execution_stream=True) if self.spec_decode_fast_paths else None
        )
        if self.spec_config is not None:
            text.num_nextn_predict_layers = 1
            if is_gdn_replay_enabled():
                text._gdn_replay_history_dtype = torch.float32
                text._gdn_replay_step_history = True
            draft_layer = MtpLayer(text.num_hidden_layers, text.rms_norm_eps)
            self.model.layers.append(draft_layer)
            self.draft_model = MtpDraft(self.model, draft_layer, self.lm_head)
            self.use_separate_draft_kv_cache = should_use_separate_draft_kv_cache(self.spec_config)
            self.spec_worker = get_spec_worker(
                self.spec_config,
                self.model_config,
                self.model_config.mapping,
                use_separate_draft_kv_cache=self.use_separate_draft_kv_cache,
                spec_decode_fast_paths=self.spec_decode_fast_paths,
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
        breakable_prefill = attn_metadata.num_contexts and is_in_breakable_cuda_graph()
        if breakable_prefill:
            self.model.prefill_graph.prepare(attn_metadata, position_ids, speculative)
        hidden = self.model(attn_metadata, input_ids, position_ids, speculative)
        if breakable_prefill:
            live = attn_metadata.num_tokens
            hidden = hidden[:live]
            input_ids = input_ids[:live]
            position_ids = position_ids[..., :live]
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

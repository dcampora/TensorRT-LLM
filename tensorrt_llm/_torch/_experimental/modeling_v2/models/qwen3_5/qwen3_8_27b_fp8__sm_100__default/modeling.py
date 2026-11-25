# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text decoder for the official Qwen3.8-27B block-scaled FP8 checkpoint."""

import json
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from flashinfer import gemma_fused_add_rmsnorm, gemma_rmsnorm, silu_and_mul
from flashinfer.gdn_prefill import chunk_gated_delta_rule
from torch import nn

from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.models.modeling_utils import DecoderModel
from tensorrt_llm._torch.pyexecutor.runtime_hooks import RuntimeHooks

from .... import disable_v2_by_default
from . import weights
from .draft_head import DraftVocabulary
from .ops.activation import (
    attention_gate_quantize,
    gdn_norm_quantize,
    residual_norm_quantize,
    swiglu_quantize,
)
from .ops.attention import (
    append_short_context_kv,
    attention_gate,
    prepare_qkv,
    short_causal_attention,
    short_paged_attention,
)
from .ops.gates import fixed_gate_projection
from .ops.matmul_accelerated import accelerated_block_matmul
from .ops.paged_attention import attention_arguments, paged_attention, paged_kv_view
from .ops.projection import block_matmul, block_quantize
from .ops.recurrent import (
    causal_convolution,
    delta_gates,
    gated_head_norm,
    prefill_convolution_qkv,
    recurrent_delta,
    short_context_delta,
)
from .ops.speculative import (
    ReplayHistory,
    prepared_verify_delta,
    reset_replay,
    verify_convolution,
    verify_delta,
)

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest


SHORT_HISTORY = 256
MAX_VERIFY_WIDTH = 8


def has_short_generation(md: "TrtllmAttentionMetadata") -> bool:
    """Use the captured decode family or a conservative eager history bound."""
    captured_mode = getattr(md, "decode_has_short_sequence", None)
    if captured_mode is not None:
        return captured_mode
    lengths = md.kv_lens_runtime[md.num_contexts : md.num_seqs]
    return bool((lengths <= SHORT_HISTORY + MAX_VERIFY_WIDTH).any())


class BlockProjection(nn.Module):
    """One projection or multiple with a shared input and a fused output."""

    def __init__(self) -> None:
        super().__init__()
        for name in (
            "weight",
            "scales",
            "scales_mn",
            "narrow_scales",
            "packed_scales",
            "workspace",
        ):
            self.register_buffer(name, None)

    def forward(self, x: torch.Tensor, decode: bool, accelerate: bool = False) -> torch.Tensor:
        return self.quantized(*block_quantize(x), x.shape[0], decode, accelerate)

    def quantized(
        self,
        a: torch.Tensor,
        scales: torch.Tensor,
        rows: int,
        decode: bool,
        accelerate: bool = False,
    ) -> torch.Tensor:
        """Dispatch the measured SM100 tile for this packed activation shape."""
        n, k = self.weight.shape
        wide_tile = decode and rows <= 8 and n == 34816
        if accelerate and not wide_tile:
            return accelerated_block_matmul(
                a,
                scales,
                self.weight,
                self.scales,
                self.packed_scales,
                rows,
                small=decode and rows <= 32 and n == 5120,
            )
        split_k = decode and rows == 1 and n == 5120
        tiled = (
            decode
            and not split_k
            and (
                (n == 5120 and (64 <= rows <= 128 or (k == 6144 and rows <= 8)))
                or (n != 5120 and (rows <= 8 or (n >= 16384 and rows == 32)))
            )
        )
        return block_matmul(
            a,
            scales,
            self.weight,
            self.scales,
            self.narrow_scales if tiled and n == 5120 else self.scales_mn,
            rows,
            split_k,
            self.workspace if tiled else None,
        )


@dataclass(frozen=True)
class ShortContexts:
    """Fresh short requests within a packed context batch."""

    rows: torch.Tensor | None
    slots: torch.Tensor
    offsets: torch.Tensor
    chunks: torch.Tensor
    starts: torch.Tensor
    max_length: int

    def select(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.rows is None else x.index_select(0, self.rows)


@dataclass
class Step:
    """Views of one packed runtime step and its request-owned cache slots."""

    metadata: "TrtllmAttentionMetadata"
    positions: torch.Tensor
    attention_args: dict
    context_rows: int
    context_slots: torch.Tensor
    generation_slots: torch.Tensor
    gate_workspace: torch.Tensor
    predicted: int
    speculative: bool
    short_generation: bool
    context_offsets: torch.Tensor | None = None
    host_offsets: list[int] | None = None
    context_lengths: list[int] | None = None
    initial_slots: torch.Tensor | None = None
    short_contexts: ShortContexts | None = None

    @classmethod
    def prepare(
        cls,
        md: "TrtllmAttentionMetadata",
        positions: torch.Tensor,
        gate_workspace: torch.Tensor,
        speculative: bool,
    ) -> "Step":
        """Use only device metadata for the graph-captured decode path."""
        mamba = md.mamba_metadata
        contexts = md.num_contexts
        generation = md.num_seqs - contexts
        predicted = (md.num_tokens - md.num_ctx_tokens) // generation if generation else 1
        step = cls(
            md,
            positions,
            attention_arguments(md, predicted=predicted),
            md.num_ctx_tokens,
            mamba.state_indices[:contexts],
            mamba.state_indices[contexts : md.num_seqs],
            gate_workspace,
            predicted,
            speculative,
            generation > 0 and has_short_generation(md),
        )
        if contexts:
            lengths = md.seq_lens[:contexts].tolist()
            step.context_lengths = lengths
            step.host_offsets = list(accumulate(lengths, initial=0))
            step.context_offsets = mamba.cu_seqlens[: contexts + 1]
            step.initial_slots = torch.where(
                mamba.has_initial_states[:contexts], step.context_slots, -1
            )
            fresh = [
                seq
                for seq, (length, initial) in enumerate(
                    zip(lengths, mamba.has_initial_states_cpu[:contexts].tolist(), strict=True)
                )
                if 0 < length <= 128 and not initial
            ]
            if fresh:
                step.short_contexts = step._prepare_short_contexts(fresh, lengths)
        return step

    def _prepare_short_contexts(self, sequences: list[int], lengths: list[int]) -> ShortContexts:
        def device_tensor(values: list, dtype: torch.dtype = torch.int32) -> torch.Tensor:
            return torch.tensor(values, dtype=dtype, device="cpu").to(
                self.positions.device, non_blocking=True
            )

        sizes = [lengths[seq] for seq in sequences]
        counts = [(length + 63) // 64 for length in sizes]
        pairs = [(seq, tile) for seq, count in enumerate(counts) for tile in range(count)]
        rows = None
        offsets = self.context_offsets
        assert offsets is not None
        slots = self.context_slots
        if len(sequences) != len(lengths):
            assert self.host_offsets is not None
            rows = device_tensor(
                [
                    row
                    for seq in sequences
                    for row in range(self.host_offsets[seq], self.host_offsets[seq + 1])
                ],
                torch.int64,
            )
            offsets = device_tensor(list(accumulate(sizes, initial=0)))
            slots = slots.index_select(0, device_tensor(sequences, torch.int64))
        return ShortContexts(
            rows,
            slots,
            offsets,
            device_tensor(pairs),
            device_tensor(list(accumulate(counts, initial=0))),
            max(sizes),
        )


def residual_norm(
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gemma RMSNorm and the BF16 input stream saved for the next residual add."""
    if residual is None:
        return gemma_rmsnorm(x, weight, 1e-6), x
    gemma_fused_add_rmsnorm(x, residual, weight, 1e-6)
    return x, residual


class DecoderLayer(nn.Module):
    """Gated DeltaNet or full attention followed by a dense SwiGLU block."""

    def __init__(self, index: int, is_linear: bool) -> None:
        super().__init__()
        self.index = index
        self.is_linear = is_linear
        self.input_proj = BlockProjection()
        self.output_proj = BlockProjection()
        self.gate_up = BlockProjection()
        self.down = BlockProjection()
        names = ("pre_norm", "post_norm")
        if is_linear:
            names += ("ba", "conv", "a_log", "dt_bias", "gdn_norm")
        else:
            names += ("q_norm", "k_norm")
        for name in names:
            self.register_buffer(name, None)

    def _linear_attention(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        step: Step,
    ) -> torch.Tensor:
        manager = step.metadata.kv_cache_manager
        mamba = step.metadata.mamba_metadata
        state = manager.get_ssm_states(self.index)
        conv_state = manager.get_conv_states(self.index)
        split = step.context_rows
        parts = []
        if split:
            short = step.short_contexts
            if short is not None:
                assert step.host_offsets is not None
                ba = torch.cat(
                    [
                        F.linear(x[start:end], self.ba)
                        for start, end in zip(step.host_offsets, step.host_offsets[1:])
                    ]
                )
            else:
                ba = F.linear(x[:split], self.ba)

            convolved = None
            if short is not None:
                convolved = causal_convolution(
                    p[:split],
                    self.conv,
                    conv_state,
                    step.context_slots,
                    step.context_offsets,
                    mamba.seq_idx,
                    mamba.has_initial_states,
                )
            decay, beta = delta_gates(ba, self.a_log, self.dt_bias)
            assert step.context_lengths is not None and step.context_offsets is not None
            max_length = max(step.context_lengths)
            if short is None or short.rows is not None:
                if convolved is None:
                    q, k, v = prefill_convolution_qkv(
                        p[:split],
                        self.conv,
                        conv_state,
                        step.context_slots,
                        step.context_offsets,
                        mamba.seq_idx,
                        mamba.has_initial_states,
                    )
                else:
                    q, k, v = (
                        t.contiguous().view(split, -1, 128)
                        for t in convolved.split((2048, 2048, 6144), dim=1)
                    )
                context, _ = chunk_gated_delta_rule(
                    q,
                    k,
                    v,
                    decay,
                    beta,
                    initial_state=state if mamba.use_initial_states else None,
                    output_final_state=True,
                    output_state=state,
                    state_indices=step.initial_slots,
                    output_state_indices=step.context_slots,
                    cu_seqlens=step.context_offsets,
                    cu_seqlens_cpu=step.host_offsets,
                    use_qk_l2norm_in_kernel=True,
                    use_cp=False,
                    max_seqlen=max_length,
                    backend="flashinfer",
                    gate_domain="log",
                )
            if short is not None:
                assert convolved is not None
                short_out = short_context_delta(
                    short.select(convolved),
                    short.select(decay),
                    short.select(beta),
                    short.offsets,
                    short.chunks,
                    short.starts,
                    short.max_length,
                    state,
                    short.slots,
                )
                if short.rows is None:
                    context = short_out
                else:
                    context.index_copy_(0, short.rows, short_out)
            parts.append(context)
        if split < x.shape[0]:
            gen = x[split:]
            ba = (
                fixed_gate_projection(gen, self.ba, step.gate_workspace)
                if gen.shape[0] <= 512
                else F.linear(gen, self.ba)
            )
            if step.speculative:
                parts.append(self._verify_delta(p[split:], ba, step))
            else:
                convolved = causal_convolution(
                    p[split:], self.conv, conv_state, step.generation_slots
                )
                parts.append(
                    recurrent_delta(
                        convolved, ba, self.a_log, self.dt_bias, state, step.generation_slots
                    )
                )
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def _verify_delta(self, projection: torch.Tensor, ba: torch.Tensor, step: Step) -> torch.Tensor:
        manager = step.metadata.kv_cache_manager
        mamba = step.metadata.mamba_metadata
        context = step.metadata.num_contexts
        replay = manager.get_replay_state_update_metadata()
        cache = manager.mamba_layer_cache(self.index)
        slots = step.generation_slots
        replay_slots = mamba.replay_state_indices[context : step.metadata.num_seqs]
        history = ReplayHistory(
            replay.prev_num_accepted_tokens,
            replay.cache_buf_idx,
            cache.old_x,
            cache.old_B,
            cache.old_dt,
        )
        if step.predicted == 1:
            qkv = causal_convolution(projection, self.conv, cache.conv, slots)
        else:
            temporary_slots = manager.intermediate_state_indices[: slots.numel()]
            if slots.numel() >= 16:
                return prepared_verify_delta(
                    projection,
                    self.conv,
                    cache.conv,
                    cache.intermediate_conv_window,
                    temporary_slots,
                    ba,
                    self.a_log,
                    self.dt_bias,
                    cache.temporal,
                    slots,
                    replay_slots,
                    history,
                    step.predicted,
                )
            qkv = verify_convolution(
                projection,
                self.conv,
                cache.conv,
                cache.intermediate_conv_window,
                slots,
                temporary_slots,
                step.predicted,
            )
        return verify_delta(
            qkv,
            ba,
            self.a_log,
            self.dt_bias,
            cache.temporal,
            slots,
            replay_slots,
            history,
            step.predicted,
        )

    def _full_attention(self, p: torch.Tensor, rope: torch.Tensor, step: Step) -> torch.Tensor:
        qkv = prepare_qkv(
            p,
            self.q_norm,
            self.k_norm,
            rope,
            step.positions,
            tiled=bool(step.context_rows),
        )
        md = step.metadata
        short = step.short_contexts
        split = step.context_rows
        if short is not None and short.rows is None and split == p.shape[0]:
            pages, table, divisor = paged_kv_view(md, self.index)
            seq_ids = md.mamba_metadata.seq_idx if md.num_contexts > 1 else None
            append_short_context_kv(qkv, pages, table, divisor, short.offsets, seq_ids)
            return self._short_attention(qkv, p, short)

        layer = md.kv_cache_manager.layer_offsets[self.index]
        out = paged_attention(qkv, layer, step.attention_args)
        if step.short_generation:
            pages, table, divisor = paged_kv_view(md, self.index)
            query = qkv[split:, :6144].view(-1, 24, 256)
            short_paged_attention(
                query,
                pages,
                table,
                step.positions[split:],
                out[split:],
                divisor,
                md.num_contexts,
                step.predicted,
            )
        if short is None:
            return out
        context = self._short_attention(qkv[:split], p[:split], short)
        if short.rows is not None:
            return attention_gate(out, p).index_copy_(0, short.rows, context)
        return torch.cat((context, attention_gate(out[split:], p[split:])))

    @staticmethod
    def _short_attention(
        qkv: torch.Tensor, projection: torch.Tensor, short: ShortContexts
    ) -> torch.Tensor:
        q, k, v = qkv.split((6144, 1024, 1024), dim=-1)
        return short_causal_attention(
            short.select(q.view(-1, 24, 256)),
            short.select(k.view(-1, 4, 256)),
            short.select(v.view(-1, 4, 256)),
            short.select(projection),
            short.offsets,
            short.chunks,
        )

    def feed_forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        decode: bool,
        fused: bool,
        accelerate: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the post-attention residual and SwiGLU."""
        if fused:
            rows = hidden.shape[0]
            normalized = residual_norm_quantize(
                hidden, residual, self.post_norm, False, decode and rows <= 8
            )
            gate_up = self.gate_up.quantized(*normalized, rows, decode, accelerate)
            return self.down.quantized(
                *swiglu_quantize(gate_up), rows, decode, accelerate
            ), residual
        hidden, residual = residual_norm(hidden, residual, self.post_norm)
        return self.down(silu_and_mul(self.gate_up(hidden, decode)), decode), residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        rope: torch.Tensor,
        step: Step,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decode = step.context_rows == 0
        fused = step.short_contexts is None
        rows = x.shape[0]
        accelerate = not step.short_generation and (decode or (fused and rows > 128))
        if fused and residual is not None:
            normalized = residual_norm_quantize(
                x, residual, self.pre_norm, self.is_linear, decode and rows <= 8
            )
            projection = self.input_proj.quantized(*normalized, rows, decode, accelerate)
        else:
            x, residual = residual_norm(x, residual, self.pre_norm)
            projection = self.input_proj(x, decode, accelerate)
        if self.is_linear:
            hidden = self._linear_attention(x, projection, step)
            if fused:
                activation = gdn_norm_quantize(hidden, projection, self.gdn_norm)
            else:
                hidden = gated_head_norm(hidden, projection, self.gdn_norm)
        else:
            hidden = self._full_attention(projection, rope, step)
            if fused:
                activation = attention_gate_quantize(hidden, projection)
        hidden = (
            self.output_proj.quantized(*activation, rows, decode, accelerate)
            if fused
            else self.output_proj(hidden, decode)
        )
        return self.feed_forward(hidden, residual, decode, fused, accelerate)


class TokenEmbedding(nn.Module):
    """The BF16 token embedding shared by the trunk and the speculative draft."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight", None)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.weight)


class Qwen38TextModel(DecoderModel):
    """Qwen3.8's text embedding, hybrid decoder, and final normalization."""

    supports_prefill_cuda_graph = False

    def __init__(self, model_config: "ModelConfig") -> None:
        super().__init__(model_config)
        self.config = model_config.pretrained_config
        self.embed_tokens = TokenEmbedding()
        self.aux_stream_dict = {}  # The draft is TP1 and has no auxiliary collectives.
        if model_config.spec_config is not None:
            from .draft import DraftLayer

            self.mtp_layer_cls = DraftLayer
        self.layers = nn.ModuleList(
            [
                DecoderLayer(index, layer_type == "linear_attention")
                for index, layer_type in enumerate(self.config.layer_types)
            ]
        )
        for name in ("final_norm", "rope", "gate_workspace", "gemm_workspace"):
            self.register_buffer(name, None)

    def forward(
        self,
        attn_metadata: "TrtllmAttentionMetadata",
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Generate hidden rows in the same packed order supplied by the executor."""
        assert position_ids is not None
        positions = position_ids.reshape(-1, position_ids.shape[-1])[0]
        speculative = self.model_config.spec_config is not None
        step = Step.prepare(attn_metadata, positions, self.gate_workspace, speculative)
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual, self.rope, step)
        if speculative and step.predicted == 1 and step.generation_slots.numel():
            md = attn_metadata.mamba_metadata
            replay = attn_metadata.kv_cache_manager.get_replay_state_update_metadata()
            reset_replay(
                md.replay_state_indices[attn_metadata.num_contexts : attn_metadata.num_seqs],
                replay.prev_num_accepted_tokens,
                replay.cache_buf_idx,
            )
        return residual_norm(x, residual, self.final_norm)[0]


@disable_v2_by_default
class ModelingV2Qwen38_27bFp8Sm100Default(SpecDecOneEngineForCausalLM):
    """Dense text-only Qwen3.8-27B FP8 on one SM100 GPU."""

    def __init__(self, config: "ModelConfig") -> None:
        checkpoint_dir = config.pretrained_config.name_or_path
        text = config.pretrained_config.text_config
        config.pretrained_config = text
        if config.spec_config is not None:
            text.num_nextn_predict_layers = 1
        super().__init__(
            Qwen38TextModel(config),
            config,
            hidden_size=text.hidden_size,
            vocab_size=text.vocab_size,
        )
        self._draft_control_tokens = [text.bos_token_id, text.eos_token_id]
        if self.spec_worker is not None and checkpoint_dir:
            tokenizer_config = Path(checkpoint_dir, "tokenizer_config.json")
            if tokenizer_config.is_file():
                added_tokens = json.loads(tokenizer_config.read_text()).get(
                    "added_tokens_decoder", {}
                )
                self._draft_control_tokens.extend(map(int, added_tokens))

    @classmethod
    def supports_spec_decode_fast_paths(cls, model_config: "ModelConfig") -> bool:
        return True

    @property
    def runtime_hooks(self) -> RuntimeHooks | None:
        if self.spec_config is not None:
            return RuntimeHooks(
                loop_on_execution_stream=True,
                gdn_fused_verify=True,
                gdn_first_row_checkpoint=True,
                decode_short_sequence_threshold=SHORT_HISTORY,
            )
        return RuntimeHooks(decode_short_sequence_threshold=SHORT_HISTORY)

    def load_weights(self, checkpoint: dict[str, torch.Tensor]) -> None:
        """Populate the target using the original HF keys."""
        weights.load_model(self, checkpoint)

    def cache_derived_state(self) -> None:
        super().cache_derived_state()
        if self.spec_worker is not None:
            self.spec_worker.draft_head = DraftVocabulary(
                self.lm_head.weight, self._draft_control_tokens
            )

    def validate_request(self, request: "LlmRequest") -> None:
        """Reject image and video inputs for this text-only target."""
        if set(request.py_multimodal_data or ()) - {"mrope_config"}:
            raise ValueError("The Qwen3.8-27B FP8 Modeling V2 target supports text inputs only")

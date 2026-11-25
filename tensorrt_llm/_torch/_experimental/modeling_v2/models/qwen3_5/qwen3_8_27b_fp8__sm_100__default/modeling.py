# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text decoder for the official Qwen3.8-27B block-scaled FP8 checkpoint."""

from dataclasses import dataclass
from itertools import accumulate
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
from .ops.attention import (
    attention_gate,
    prepare_qkv,
    short_causal_attention,
    short_paged_attention,
)
from .ops.gates import fixed_gate_projection
from .ops.paged_attention import attention_arguments, paged_attention
from .ops.projection import block_matmul, block_quantize
from .ops.recurrent import (
    causal_convolution,
    delta_gates,
    gated_head_norm,
    recurrent_delta,
    short_context_delta,
)
from .ops.speculative import ReplayHistory, reset_replay, verify_convolution, verify_delta

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest


class BlockProjection(nn.Module):
    """One projection or multiple with a shared input and a fused output."""

    def __init__(self) -> None:
        super().__init__()
        for name in ("weight", "scales", "scales_mn"):
            self.register_buffer(name, None)

    def forward(self, x: torch.Tensor, decode: bool) -> torch.Tensor:
        rows = x.shape[0]
        a, scales = block_quantize(x)
        return block_matmul(
            a,
            scales,
            self.weight,
            self.scales,
            self.scales_mn,
            rows,
            split_k=decode and rows == 1 and self.weight.shape[0] == 5120,
        )


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
    context_offsets: torch.Tensor | None = None
    host_offsets: list[int] | None = None
    context_lengths: list[int] | None = None
    initial_slots: torch.Tensor | None = None
    short_chunks: torch.Tensor | None = None
    short_starts: torch.Tensor | None = None

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
        )
        if contexts:
            lengths = md.seq_lens[:contexts].tolist()
            step.context_lengths = lengths
            step.host_offsets = list(accumulate(lengths, initial=0))
            step.context_offsets = mamba.cu_seqlens[: contexts + 1]
            step.initial_slots = torch.where(
                mamba.has_initial_states[:contexts], step.context_slots, -1
            )
            if max(lengths) <= 128 and not mamba.use_initial_states:
                counts = [(length + 63) // 64 for length in lengths]
                pairs = [(seq, tile) for seq, count in enumerate(counts) for tile in range(count)]
                step.short_chunks = torch.tensor(pairs, dtype=torch.int32, device="cpu").to(
                    positions.device, non_blocking=True
                )
                step.short_starts = torch.tensor(
                    list(accumulate(counts, initial=0)), dtype=torch.int32, device="cpu"
                ).to(positions.device, non_blocking=True)
        return step


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
            if step.short_chunks is not None:
                assert step.host_offsets is not None
                ba = torch.cat(
                    [
                        F.linear(x[start:end], self.ba)
                        for start, end in zip(step.host_offsets, step.host_offsets[1:])
                    ]
                )
            else:
                ba = F.linear(x[:split], self.ba)

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
            assert step.context_lengths is not None
            max_length = max(step.context_lengths)
            if step.short_chunks is not None:
                context = short_context_delta(
                    convolved,
                    decay,
                    beta,
                    step.context_offsets,
                    step.short_chunks,
                    step.short_starts,
                    max_length,
                    state,
                    step.context_slots,
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
        out = parts[0] if len(parts) == 1 else torch.cat(parts)
        return gated_head_norm(out, p, self.gdn_norm)

    def _verify_delta(self, projection: torch.Tensor, ba: torch.Tensor, step: Step) -> torch.Tensor:
        manager = step.metadata.kv_cache_manager
        mamba = step.metadata.mamba_metadata
        context = step.metadata.num_contexts
        replay = manager.get_replay_state_update_metadata()
        cache = manager.mamba_layer_cache(self.index)
        slots = step.generation_slots
        replay_slots = mamba.replay_state_indices[context : step.metadata.num_seqs]
        direct = step.predicted == 1
        if direct:
            qkv = causal_convolution(projection, self.conv, cache.conv, slots)
        else:
            qkv = verify_convolution(
                projection,
                self.conv,
                cache.conv,
                cache.intermediate_conv_window,
                slots,
                manager.intermediate_state_indices[: slots.numel()],
                step.predicted,
            )
        history = ReplayHistory(
            replay.prev_num_accepted_tokens,
            replay.cache_buf_idx,
            cache.old_x,
            cache.old_B,
            cache.old_dt,
            replay.replay_step_width,
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
            direct,
        )

    def _full_attention(self, p: torch.Tensor, rope: torch.Tensor, step: Step) -> torch.Tensor:
        q, k, v = prepare_qkv(
            p,
            self.q_norm,
            self.k_norm,
            rope,
            step.positions,
            tiled=bool(step.context_rows),
        )
        layer = step.metadata.kv_cache_manager.layer_offsets[self.index]
        out = paged_attention(q, k, v, layer, step.attention_args)
        manager = step.metadata.kv_cache_manager
        if step.context_rows < p.shape[0]:
            pages = manager.get_buffers(self.index, kv_layout="HND")
            offset_scale = manager.get_layer_page_index_scale(self.index)
            # The V2 view is indexed by K/V pairs when every slot starts
            # on a pair boundary. Otherwise its leading stride spans a slot.
            divisor = 2 if offset_scale % 2 == 0 else offset_scale
            pool = int(step.metadata.host_kv_cache_pool_mapping[layer, 0])
            table = step.metadata.kv_cache_block_offsets[pool, :, 0, :]
            split = step.context_rows
            short_paged_attention(
                q[split:],
                pages,
                table,
                step.positions[split:],
                out[split:],
                divisor,
                step.metadata.num_contexts,
                step.predicted,
            )
        if step.short_chunks is not None:
            split = step.context_rows
            context = short_causal_attention(
                q[:split],
                k[:split],
                v[:split],
                p[:split],
                step.context_offsets,
                step.short_chunks,
            )
            if split == p.shape[0]:
                return context
            return torch.cat((context, attention_gate(out[split:], p[split:])))
        return attention_gate(out, p)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        rope: torch.Tensor,
        step: Step,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decode = step.context_rows == 0
        x, residual = residual_norm(x, residual, self.pre_norm)
        projection = self.input_proj(x, decode)
        if self.is_linear:
            hidden = self._linear_attention(x, projection, step)
        else:
            hidden = self._full_attention(projection, rope, step)
        hidden = self.output_proj(hidden, decode)
        hidden, residual = residual_norm(hidden, residual, self.post_norm)
        hidden = self.down(silu_and_mul(self.gate_up(hidden, decode)), decode)
        return hidden, residual


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
        for name in ("final_norm", "rope", "gate_workspace"):
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

    @classmethod
    def supports_spec_decode_fast_paths(cls, model_config: "ModelConfig") -> bool:
        return True

    @property
    def runtime_hooks(self) -> RuntimeHooks | None:
        if self.spec_config is not None:
            return RuntimeHooks(loop_on_execution_stream=True, gdn_fused_verify=True)
        return None

    def load_weights(self, checkpoint: dict[str, torch.Tensor]) -> None:
        """Populate the target using the original HF keys."""
        weights.load_model(self, checkpoint)

    def validate_request(self, request: "LlmRequest") -> None:
        """Reject image and video inputs for this text-only target."""
        if set(request.py_multimodal_data or ()) - {"mrope_config"}:
            raise ValueError("The Qwen3.8-27B FP8 Modeling V2 target supports text inputs only")

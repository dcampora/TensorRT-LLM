# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text decoder for the ModelOpt Qwen3.6-35B-A3B checkpoint on SM100."""

from dataclasses import dataclass

import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_utils import DecoderModel, DecoderModelForCausalLM
from tensorrt_llm._torch.speculative.interface import should_use_separate_draft_kv_cache

from .... import disable_v2_by_default
from ..routing import LAYER_TYPES
from .kernels.cute_moe import PackedVocabulary
from .mtp import DraftLayer, mtp_step
from .ops.attention import gated_attention, prepare_qkv
from .ops.moe import Experts, residual_rms_norm
from .ops.projections import fp8_linear
from .ops.recurrent import convolution_decode, delta_decode, delta_norm, delta_prefill
from .ops.verify import convolution_verify, delta_verify
from .weights import (
    BF16,
    HIDDEN,
    NUM_EXPERTS,
    VOCAB,
    Fp8Weights,
    PackedWeights,
    load_decoder_weights,
    load_draft_weights,
    parameter,
)


@dataclass(frozen=True)
class Step:
    positions: torch.Tensor
    request_rows: torch.Tensor
    slots: torch.Tensor
    context_tokens: int
    contexts: int
    batch: int
    cu: torch.Tensor | None
    has_initial: torch.Tensor | None
    max_context: int
    capacity: int
    width: int
    replay_slots: torch.Tensor | None
    replay_width: int

    @classmethod
    def from_metadata(cls, md, position_ids, request_indices, spec_metadata):
        mamba = md.mamba_metadata
        contexts = md.num_contexts
        ctx = md.num_ctx_tokens
        batch = md.num_seqs
        positions = position_ids.reshape(-1, position_ids.shape[-1])[0, : md.num_tokens]
        rows = request_indices[contexts:batch]
        width = 1
        replay_slots = None
        replay_width = 0
        if spec_metadata is not None:
            width = 1 + spec_metadata.runtime_draft_len
            if md.num_generations and width > 1:
                rows = rows.repeat_interleave(width)
            replay_slots = mamba.replay_state_indices[contexts:batch]
            replay_width = md.kv_cache_manager.get_replay_state_update_metadata().replay_step_width
        cu = initial = None
        longest = 0
        if contexts:
            context_rows = mamba.seq_idx.reshape(-1)[:ctx].int()
            rows = torch.cat((context_rows, rows)) if md.num_generations else context_rows
            cu = mamba.query_start_loc_long[: contexts + 1]
            longest = int(max(md.seq_lens[:contexts]))
            if mamba.use_initial_states:
                initial = mamba.has_initial_states[:contexts]
        return cls(
            positions,
            rows,
            mamba.state_indices[:batch],
            ctx,
            contexts,
            batch,
            cu,
            initial,
            longest,
            md.kv_cache_manager.max_seq_len,
            width,
            replay_slots,
            replay_width,
        )


class FullAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = Fp8Weights(HIDDEN, 9216)
        self.output = Fp8Weights(4096, HIDDEN)
        self.qnorm = parameter((256,))
        self.knorm = parameter((256,))

    def forward(self, x, step, kv, table, rope):
        qkv = fp8_linear(x, self.qkv)
        q, gate = prepare_qkv(
            qkv, self.qnorm, self.knorm, rope, step.positions, step.request_rows, kv, table
        )
        y = gated_attention(q, gate, kv, table, step, self.output.input_scale)
        return fp8_linear(y, self.output)


class LinearAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkvz = Fp8Weights(HIDDEN, 12288)
        self.output = Fp8Weights(4096, HIDDEN)
        self.ba = parameter((64, HIDDEN))
        self.conv = parameter((8192, 4))
        self.norm = parameter((128,))
        self.a_log = parameter((32,), torch.float32)
        self.dt = parameter((32,))

    def forward(self, x, step, cache, side):
        history, state = cache.conv, cache.temporal
        if step.contexts:
            ba = torch.mm(x, self.ba.T)
            qkvz = fp8_linear(x, self.qkvz)
        else:
            main = torch.cuda.current_stream()
            side.wait_stream(main)
            with torch.cuda.stream(side):
                ba = torch.mm(x, self.ba.T)
            qkvz = fp8_linear(x, self.qkvz)
            main.wait_stream(side)

        ctx = step.context_tokens
        if ctx:
            y = delta_prefill(
                qkvz[:ctx], ba[:ctx], self.conv, self.a_log, self.dt, history, state, step
            )
        if qkvz.shape[0] > ctx:
            slots = step.slots[step.contexts :]
            if step.replay_slots is None:
                qkv = convolution_decode(qkvz[ctx:], self.conv, slots, history)
                decode = delta_decode(qkv, ba[ctx:], self.a_log, self.dt, slots, state)
            else:
                qkv = convolution_verify(
                    qkvz[ctx:],
                    self.conv,
                    slots,
                    history,
                    cache.intermediate_conv_window,
                    step.width,
                )
                decode = delta_verify(
                    qkv,
                    ba[ctx:],
                    self.a_log,
                    self.dt,
                    slots,
                    step.replay_slots,
                    state,
                    cache.old_x,
                    cache.old_B,
                    cache.old_dt,
                    cache.prev_num_accepted_tokens,
                    cache.cache_buf_idx,
                    step.width,
                    step.replay_width,
                )
            y = torch.cat((y, decode)) if ctx else decode
        scale = self.output.input_scale if ctx or step.batch == 1 else None
        return fp8_linear(delta_norm(y, qkvz, self.norm, scale), self.output)


class Layer(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.full_attention = kind == "full_attention"
        self.attention = FullAttention() if self.full_attention else LinearAttention()
        self.before = parameter((HIDDEN,))
        self.after = parameter((HIDDEN,))
        self.moe = Experts()


class PackedHead(PackedWeights):
    vocab_size_padded = VOCAB
    num_embeddings = VOCAB

    def __init__(self):
        super().__init__(HIDDEN, VOCAB)

    def prepare(self):
        super().prepare()
        self.projector = PackedVocabulary(self)

    def forward(self, x):
        return self.projector.logits(x)


class QwenDecoder(DecoderModel):
    supports_prefill_cuda_graph = False

    def __init__(self, model_config):
        super().__init__(model_config)
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN, dtype=BF16, _freeze=True)
        self.layers = nn.ModuleList(Layer(kind) for kind in LAYER_TYPES)
        self.final_norm = parameter((HIDDEN,))
        self._cache_manager = None
        self._layer_caches = ()
        self.mtp = DraftLayer() if model_config.spec_config is not None else None
        self.mtp_cache = None

    def prepare(self, config):
        self.side = torch.cuda.Stream()
        kw = {"device": self.final_norm.device, "dtype": torch.int32}
        self.bins = torch.arange(NUM_EXPERTS + 1, **kw)
        self.requests = torch.arange(4096, **kw)
        rope = config.rope_parameters
        freqs = 1 / (
            rope["rope_theta"]
            ** (torch.arange(0, 64, 2, dtype=torch.float32, device=self.final_norm.device) / 64)
        )
        positions = torch.arange(
            config.max_position_embeddings, device=freqs.device, dtype=freqs.dtype
        )
        radians = positions[:, None] * freqs[None]
        self.rope = torch.cat((radians.cos(), radians.sin()), dim=-1)
        for layer in self.layers:
            layer.moe.prepare()

    def _bind_cache_views(self, md):
        cm = md.kv_cache_manager
        mapping = md.host_kv_cache_pool_mapping.tolist()
        caches = []
        for i, layer in enumerate(self.layers):
            if layer.full_attention:
                kv = cm.get_buffers(i, kv_layout="HND")
                pool = mapping[cm.layer_offsets[i]][0]
                caches.append((kv, pool))
            else:
                caches.append(cm.mamba_layer_cache(i))
        self._cache_manager = cm
        self._layer_caches = tuple(caches)
        if self.mtp is not None:
            i = len(self.layers)
            draft_manager = md.draft_kv_cache_manager
            separate = draft_manager is not None
            if separate:
                draft = draft_manager
                draft_mapping = draft.kv_cache_pool_mapping.tolist()
            else:
                draft = cm
                draft_mapping = mapping
            self.mtp_cache = (
                draft.get_buffers(i, kv_layout="HND"),
                draft_mapping[draft.layer_offsets[i]][0],
                separate,
            )

    def forward(
        self, attn_metadata, input_ids, position_ids, inputs_embeds=None, spec_metadata=None
    ):
        if self._cache_manager is not attn_metadata.kv_cache_manager:
            self._bind_cache_views(attn_metadata)
        step = Step.from_metadata(attn_metadata, position_ids, self.requests, spec_metadata)
        if inputs_embeds is None:
            x = self.embed_tokens(input_ids[: attn_metadata.num_tokens])
        else:
            x = inputs_embeds[: attn_metadata.num_tokens]
        residual = torch.empty_like(x)
        x = residual_rms_norm(x, residual, self.layers[0].before, False, True)
        for i, (layer, cache) in enumerate(zip(self.layers, self._layer_caches)):
            if layer.full_attention:
                kv, pool = cache
                table = attn_metadata.kv_cache_block_offsets[pool]
                x = layer.attention(x, step, kv, table, self.rope)
            else:
                x = layer.attention(x, step, cache, self.side)
            x = residual_rms_norm(x, residual, layer.after, True, True)
            save = i + 1 < len(self.layers)
            weight = self.layers[i + 1].before if save else self.final_norm
            epilogue = None if step.contexts else (residual, weight, save)
            x = layer.moe(x, self.bins, self.side, step.contexts, step.batch, epilogue)
            if epilogue is None:
                x = residual_rms_norm(x, residual, weight, True, save)
        return x, step


@disable_v2_by_default
class ModelingV2Qwen36Nvfp4Sm100Default(DecoderModelForCausalLM):
    """Qwen3.6 language model with packed tensor-core experts and vocabulary."""

    supports_prefill_cuda_graph = False

    def __init__(self, model_config):
        # Use a target-local vocabulary projection in place of the runtime base's
        # default LMHead, which would build a second implementation of the head.
        nn.Module.__init__(self)
        text = model_config.pretrained_config.text_config
        text.architectures = ["Qwen3_5MoeForCausalLM"]
        model_config.pretrained_config = text
        self.model_config = model_config
        self.spec_config = model_config.spec_config
        self.gdn_fused_verify = self.spec_config is not None
        self.spec_worker = None
        self.draft_model = None
        self.draft_config = None
        self.use_separate_draft_kv_cache = should_use_separate_draft_kv_cache(self.spec_config)
        self.model = QwenDecoder(model_config)
        self.lm_head = PackedHead()
        self.pp_rank = model_config.mapping.pp_rank
        self.pp_size = model_config.mapping.pp_size
        self.has_custom_lm_head = False
        self.prologue = []
        self.epilogue = [self.lm_head]

    @classmethod
    def get_model_defaults(cls, llm_args):
        return {
            "kv_cache_config": {"enable_block_reuse": False, "mamba_ssm_cache_dtype": "float32"}
        }

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None):
        return "V2"

    def load_weights(self, weights):
        load_decoder_weights(self.model, weights)
        self.lm_head.load(weights, ("",), ("lm_head",))
        if self.model.mtp is not None:
            load_draft_weights(self.model.mtp, weights)

    def post_load_weights(self):
        for module in self.modules():
            if isinstance(module, PackedWeights) and module is not self.lm_head:
                module.prepare()
        self.lm_head.prepare()
        self.model.prepare(self.config)

    def forward(
        self,
        attn_metadata,
        input_ids=None,
        position_ids=None,
        inputs_embeds=None,
        return_context_logits=False,
        spec_metadata=None,
        **kwargs,
    ):
        hidden, step = self.model(
            attn_metadata, input_ids, position_ids, inputs_embeds, spec_metadata
        )
        if spec_metadata is not None:
            return mtp_step(
                input_ids,
                hidden,
                step,
                attn_metadata,
                spec_metadata,
                self.model.mtp,
                self.lm_head,
                self.model,
            )
        if not return_context_logits:
            last = torch.cumsum(attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
            hidden = hidden[last]
        return self.lm_head(hidden).float()

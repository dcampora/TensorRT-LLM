# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-35B-A3B text decoder with block-scaled FP8 on one SM100 GPU."""

from typing import Literal

import torch
from flashinfer.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2 import disable_v2_by_default
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.models.modeling_utils import DecoderModel

from ....catalog.attention.fused_qk_norm_rope import fused_qk_norm_rope
from ....catalog.gemm.cublas_mm import cublas_mm
from . import weights
from .ops import fp8
from .ops.attention import attention_step, paged_attention
from .ops.epilogue_quantize import gated_quantize
from .ops.gates import sigmoid_gate
from .ops.gdn import RecurrentStep, gdn_scan
from .ops.quantize import add_rmsnorm_quant


def _parameter(shape, dtype=torch.bfloat16):
    return nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)


class _Fp8Linear(nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.weight = _parameter((n, k), torch.float8_e4m3fn)
        self.scale = _parameter((n // fp8.BLOCK, k // fp8.BLOCK), torch.float32)

    def quantized(self, x, scale):
        return fp8.linear_quantized(x, scale, self.weight, self.scale)


class _GatedDeltaNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key_heads = config.linear_num_key_heads
        self.value_heads = config.linear_num_value_heads
        self.head_dim = config.linear_key_head_dim
        self.value_width = self.value_heads * self.head_dim
        self.conv_width = 2 * self.key_heads * self.head_dim + self.value_width
        self.eps = config.rms_norm_eps
        self.qkvz = _Fp8Linear(self.conv_width + self.value_width, config.hidden_size)
        self.ab = _parameter((2 * self.value_heads, config.hidden_size))
        self.conv = _parameter((self.conv_width, config.linear_conv_kernel_dim))
        self.a_log = _parameter((self.value_heads,), torch.float32)
        self.dt_bias = _parameter((self.value_heads,), torch.float32)
        self.norm = _parameter((self.head_dim,))
        self.output = _Fp8Linear(config.hidden_size, self.value_width)

    def forward(self, x, x_fp8, x_scale, cache, step):
        qkv, z = self.qkvz.quantized(x_fp8, x_scale).split(
            (self.conv_width, self.value_width), dim=-1
        )
        a, b = cublas_mm(x, self.ab.T).split(self.value_heads, dim=-1)
        y = gdn_scan(
            qkv,
            a,
            b,
            self.conv,
            self.a_log,
            self.dt_bias,
            cache,
            step,
            self.key_heads,
            self.value_heads,
            self.head_dim,
        )
        return self.output.quantized(
            *gated_quantize(y.view(-1, self.value_width), "gdn", z, self.norm, self.eps)
        )


class _FullAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_width = self.heads * self.head_dim
        self.kv_width = self.kv_heads * self.head_dim
        rope = config.rope_parameters
        self.rotary_dim = int(self.head_dim * rope["partial_rotary_factor"])
        self.rope_theta = rope["rope_theta"]
        self.eps = config.rms_norm_eps
        self.qkvg = _Fp8Linear(2 * self.q_width + 2 * self.kv_width, config.hidden_size)
        self.q_norm = _parameter((self.head_dim,))
        self.k_norm = _parameter((self.head_dim,))
        self.output = _Fp8Linear(config.hidden_size, self.q_width)

    def forward(self, x, x_fp8, x_scale, positions, local_layer_idx, step, pick_rows=None):
        # The loader lays out Q, K and V contiguously ahead of the Q gate.
        projection = self.qkvg.quantized(x_fp8, x_scale)
        qkv_width = self.q_width + 2 * self.kv_width
        qkv, gate = projection.split((qkv_width, self.q_width), dim=-1)
        qkv = qkv.contiguous()
        fused_qk_norm_rope(
            qkv,
            self.heads,
            self.kv_heads,
            self.kv_heads,
            self.head_dim,
            self.rotary_dim,
            self.eps,
            self.q_norm,
            self.k_norm,
            self.rope_theta,
            True,
            positions,
            use_gemma=True,
        )
        out = torch.empty((x.shape[0], self.q_width), dtype=x.dtype, device=x.device)
        paged_attention(qkv, out, local_layer_idx, self.heads, self.kv_heads, self.head_dim, step)
        return self.output.quantized(*gated_quantize(out, "attention", gate, pick_rows=pick_rows))


class _SharedExpertStream:
    """Coordinate shared-expert work with the routed-expert stream."""

    def __init__(self):
        self.stream = torch.cuda.Stream()
        self.ready = torch.cuda.Event()
        self.done = torch.cuda.Event()


class _Moe(nn.Module):
    def __init__(self, config, shared_stream):
        super().__init__()
        self.shared_stream = shared_stream
        experts = config.num_experts
        hidden = config.hidden_size
        intermediate = config.moe_intermediate_size
        shared_intermediate = config.shared_expert_intermediate_size
        self.top_k = config.num_experts_per_tok
        self.router = _parameter((experts, hidden))
        self.up_gate = _parameter((experts, 2 * intermediate, hidden), torch.float8_e4m3fn)
        self.up_gate_scale = _parameter(
            (experts, 2 * intermediate // fp8.BLOCK, hidden // fp8.BLOCK), torch.float32
        )
        self.down = _parameter((experts, hidden, intermediate), torch.float8_e4m3fn)
        self.down_scale = _parameter(
            (experts, hidden // fp8.BLOCK, intermediate // fp8.BLOCK), torch.float32
        )
        self.shared_gate_up = _Fp8Linear(2 * shared_intermediate, hidden)
        self.shared_down = _Fp8Linear(hidden, shared_intermediate)
        self.shared_router = _parameter((1, hidden))

    def _routed(self, x, x_fp8, x_scale):
        return fp8.routed_experts(
            cublas_mm(x, self.router.T),
            x_fp8,
            x_scale,
            self.up_gate,
            self.up_gate_scale,
            self.down,
            self.down_scale,
            self.top_k,
        )

    def _shared(self, x, x_fp8, x_scale):
        shared = fp8.linear_quantized(
            x_fp8, x_scale, self.shared_gate_up.weight, self.shared_gate_up.scale
        )
        shared = self.shared_down.quantized(*gated_quantize(shared, "swiglu"))
        return shared, cublas_mm(x, self.shared_router.T)

    def forward(self, x, x_fp8, x_scale):
        # Large packed batches already saturate the expert kernels.
        if x.shape[0] > 128:
            routed = self._routed(x, x_fp8, x_scale)
            shared, gate = self._shared(x, x_fp8, x_scale)
        else:
            aux = self.shared_stream
            main = torch.cuda.current_stream()
            aux.ready.record(main)
            with torch.cuda.stream(aux.stream):
                aux.ready.wait()
                shared, gate = self._shared(x, x_fp8, x_scale)
                aux.done.record()
            routed = self._routed(x, x_fp8, x_scale)
            aux.done.wait()
            shared.record_stream(main)
            gate.record_stream(main)
        return sigmoid_gate(shared, gate, routed)


class _DecoderLayer(nn.Module):
    def __init__(self, config, layer_type, shared_stream):
        super().__init__()
        self.recurrent = layer_type == "linear_attention"
        self.mixer = _GatedDeltaNet(config) if self.recurrent else _FullAttention(config)
        self.mlp = _Moe(config, shared_stream)
        self.input_norm = _parameter((config.hidden_size,))
        self.post_attention_norm = _parameter((config.hidden_size,))


class _DraftHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = _parameter((hidden_size,))

    def forward(self, hidden, lm_head, metadata, return_context_logits=False):
        if not return_context_logits:
            if metadata is None:
                hidden = hidden[-1:]
            else:
                last = metadata.seq_lens_cuda.cumsum(dim=0, dtype=torch.long) - 1
                hidden = hidden.index_select(0, last)
        return lm_head(hidden, is_spec_decoding_head=True)


class _DraftLayer(_DecoderLayer):
    """Predict the next token from the target state and the preceding token."""

    supports_pick_rows = True

    def __init__(self, model_config, layer_index, aux_streams):
        config = model_config.pretrained_config
        super().__init__(config, "full_attention", _SharedExpertStream())
        self.layer_index = layer_index
        self.eps = config.rms_norm_eps
        hidden_size = config.hidden_size
        self.embedding_norm = _parameter((hidden_size,))
        self.hidden_norm = _parameter((hidden_size,))
        self.fc = _parameter((hidden_size, 2 * hidden_size))
        self.shared_head = _DraftHead(hidden_size)

    def forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        embed_tokens,
        attn_metadata,
        pick_rows=None,
        **kwargs,
    ):
        embeddings = gemma_rmsnorm(embed_tokens(input_ids), self.embedding_norm, self.eps)
        hidden = gemma_rmsnorm(hidden_states, self.hidden_norm, self.eps)
        residual = cublas_mm(torch.cat((embeddings, hidden), dim=-1), self.fc.T)
        hidden = gemma_rmsnorm(residual, self.input_norm, self.eps)
        x_fp8, x_scale = fp8.quantize(hidden)
        positions = position_ids.reshape(-1, hidden.shape[0])[0].to(torch.int32)
        cache_manager = attn_metadata.kv_cache_manager
        hidden = self.mixer(
            hidden,
            x_fp8,
            x_scale,
            positions,
            cache_manager.layer_offsets[self.layer_index],
            attention_step(attn_metadata),
            pick_rows=pick_rows,
        )
        if pick_rows is not None:
            residual = residual.index_select(0, pick_rows)
        x_fp8, x_scale = add_rmsnorm_quant(hidden, residual, self.post_attention_norm, self.eps)
        hidden = self.mlp(hidden, x_fp8, x_scale)
        gemma_fused_add_rmsnorm(hidden, residual, self.shared_head.norm, self.eps)
        return hidden


class _Decoder(DecoderModel):
    mtp_layer_cls = _DraftLayer

    def __init__(self, model_config):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=torch.bfloat16
        )
        shared_stream = _SharedExpertStream()
        self.layers = nn.ModuleList(
            _DecoderLayer(config, kind, shared_stream) for kind in config.layer_types
        )
        self.num_layers = len(self.layers)
        self.graph_batch_sizes = tuple(model_config.cuda_graph_batch_sizes or ())
        self.aux_stream_dict = {}
        self.norm = _parameter((config.hidden_size,))
        self.eps = config.rms_norm_eps

    def forward(
        self, attn_metadata, input_ids=None, position_ids=None, inputs_embeds=None, **kwargs
    ):
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        residual = hidden
        hidden = gemma_rmsnorm(hidden, self.layers[0].input_norm, self.eps)
        x_fp8, x_scale = fp8.quantize(hidden)
        attention = attention_step(attn_metadata)
        recurrent = RecurrentStep.from_metadata(
            attn_metadata, kwargs.get("spec_metadata"), self.graph_batch_sizes
        )
        cache_manager = attn_metadata.kv_cache_manager
        positions = position_ids.reshape(-1, hidden.shape[0])[0].to(torch.int32)
        for index, layer in enumerate(self.layers[: self.num_layers]):
            if layer.recurrent:
                cache = cache_manager.mamba_layer_cache(index)
                hidden = layer.mixer(hidden, x_fp8, x_scale, cache, recurrent)
            else:
                hidden = layer.mixer(
                    hidden, x_fp8, x_scale, positions, cache_manager.layer_offsets[index], attention
                )
            x_fp8, x_scale = add_rmsnorm_quant(
                hidden, residual, layer.post_attention_norm, self.eps
            )
            hidden = layer.mlp(hidden, x_fp8, x_scale)
            if index + 1 < self.num_layers:
                next_norm = self.layers[index + 1].input_norm
                x_fp8, x_scale = add_rmsnorm_quant(hidden, residual, next_norm, self.eps)
            else:
                gemma_fused_add_rmsnorm(hidden, residual, self.norm, self.eps)
        return hidden


@disable_v2_by_default
class ModelingV2Qwen3635bA3bFp8Sm100Default(SpecDecOneEngineForCausalLM):
    """Serve the checkpoint's text tower through the standard causal LM shell."""

    def __init__(self, model_config):
        text_config = model_config.pretrained_config.text_config
        model_config.pretrained_config = text_config
        if model_config.spec_config is not None:
            text_config.num_nextn_predict_layers = 1
        super().__init__(
            _Decoder(model_config),
            model_config=model_config,
            hidden_size=text_config.hidden_size,
            vocab_size=text_config.vocab_size,
        )
        if self.draft_model is not None:
            self.model.layers.extend(self.draft_model.mtp_layers)

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None) -> Literal["V2"]:
        """Use the hybrid runtime's KV and recurrent state manager."""
        return "V2"

    def validate_request(self, request):
        """Reject media requests at admission to this text decoder."""
        data = request.py_multimodal_data
        if data and any(data.get(key) is not None for key in ("image", "video", "audio")):
            raise ValueError("Qwen3.6 FP8 Modeling V2 accepts text inputs only")

    def load_weights(self, checkpoint, *args, **kwargs):
        """Load FP8 tensors and their associated checkpoint block scales."""
        weights.load(self, checkpoint)

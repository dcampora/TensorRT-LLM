# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select the text-only Qwen3.6 ModelOpt target."""

from tensorrt_llm._torch.utils import is_gdn_replay_enabled
from tensorrt_llm.llmapi.llm_args import LoadFormat, MTPDecodingConfig
from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

_SHAPE = {
    "hidden_size": 2048,
    "num_hidden_layers": 40,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 512,
    "vocab_size": 248320,
}
LAYER_TYPES = ("linear_attention", "linear_attention", "linear_attention", "full_attention") * 10


def _quantized_layers():
    layers = {"lm_head": QuantAlgo.W4A16_NVFP4}
    for i, kind in enumerate(LAYER_TYPES):
        prefix = f"model.language_model.layers.{i}"
        if kind == "linear_attention":
            projections = ("in_proj_qkv", "in_proj_z", "out_proj")
            attention = "linear_attn"
        else:
            projections = ("q_proj", "k_proj", "v_proj", "o_proj")
            attention = "self_attn"
        for projection in projections:
            layers[f"{prefix}.{attention}.{projection}"] = QuantAlgo.FP8
        layers[f"{prefix}.mlp.experts"] = QuantAlgo.W4A16_NVFP4
        for projection in ("gate_proj", "up_proj", "down_proj"):
            layers[f"{prefix}.mlp.shared_expert.{projection}"] = QuantAlgo.W4A16_NVFP4
    return layers


_EXPECTED_QUANTIZATION = _quantized_layers()


@routes("Qwen3_5MoeForConditionalGeneration")
def route(model_config, llm_args, compute_capability):
    mapping = model_config.mapping
    if compute_capability != (10, 0) or mapping.world_size != 1:
        raise NoTarget("Qwen3.6-35B-A3B NVFP4 requires SM100 and a single rank")
    if not llm_args.disable_mm_encoder:
        raise NoTarget("the Qwen3.6-35B-A3B NVFP4 target requires disable_mm_encoder=True")
    text = getattr(model_config.pretrained_config, "text_config", None)
    if any(getattr(text, field, None) != value for field, value in _SHAPE.items()):
        raise NoTarget("checkpoint does not have the Qwen3.6-35B-A3B text geometry")
    if (
        tuple(text.layer_types) != LAYER_TYPES
        or not text.attn_output_gate
        or text.tie_word_embeddings
    ):
        raise NoTarget("checkpoint must have gated attention and the Qwen3.6 hybrid layer schedule")
    rope = text.rope_parameters or {}
    if (
        text.attention_bias
        or text.rms_norm_eps != 1e-6
        or rope.get("rope_type", rope.get("type")) not in ("default", "mrope")
        or rope.get("partial_rotary_factor") != 0.25
    ):
        raise NoTarget(
            "the target requires the checkpoint's bias-free 64-dimensional rotary layout"
        )
    quant = model_config.quant_config
    per_layer = model_config.quant_config_dict or {}
    if (
        quant.quant_algo != QuantAlgo.MIXED_PRECISION
        or quant.kv_cache_quant_algo != QuantAlgo.FP8
        or per_layer.keys() != _EXPECTED_QUANTIZATION.keys()
        or any(
            name not in per_layer
            or per_layer[name].quant_algo != algo
            or (algo == QuantAlgo.W4A16_NVFP4 and per_layer[name].group_size != 16)
            for name, algo in _EXPECTED_QUANTIZATION.items()
        )
    ):
        raise NoTarget("checkpoint must have the Qwen3.6 ModelOpt FP8 and W4A16 NVFP4 layout")
    if engine_kv_cache_quant_algo(model_config, llm_args) != QuantAlgo.FP8:
        raise NoTarget("the target requires FP8 KV cache")
    if llm_args.attn_backend != "TRTLLM":
        raise NoTarget("the target consumes TensorRT-LLM attention metadata")
    kv = llm_args.kv_cache_config
    if kv.use_kv_cache_manager_v2 is False or kv.mamba_ssm_cache_dtype != "float32":
        raise NoTarget("the target requires KV cache manager V2 with FP32 recurrent state")
    if kv.enable_block_reuse or kv.max_attention_window is not None:
        raise NoTarget("the target requires full KV history with prefix block reuse disabled")
    if llm_args.cache_transceiver_config is not None:
        raise NoTarget("the target is for colocated prefill and decoding")
    if llm_args.max_beam_width not in (None, 1):
        raise NoTarget("the target supports one beam per request")
    if (
        llm_args.load_format != LoadFormat.AUTO
        or llm_args.enable_lora
        or llm_args.lora_config is not None
        or llm_args.enable_locality_domains
        or llm_args.lm_head_output_dtype != "auto"
    ):
        raise NoTarget("the target requires the Hugging Face checkpoint and disables LoRA")
    spec = llm_args.speculative_config
    if spec is not None:
        if (
            not isinstance(spec, MTPDecodingConfig)
            or not spec.spec_dec_mode.is_mtp_eagle_one_model()
            or spec.use_dynamic_tree
            or spec.max_draft_len not in (1, 2, 3)
            or spec.max_total_draft_tokens not in (None, spec.max_draft_len)
            or spec.use_rejection_sampling
            or spec.enable_penalty
            or spec.use_relaxed_acceptance_for_thinking
            or spec.sa_config is not None
            or spec.speculative_model is not None
            or spec.draft_lm_head_dtype != "auto"
            or spec.draft_moe_dtype != "auto"
            or llm_args.guided_decoding_backend is not None
            or not is_gdn_replay_enabled()
        ):
            raise NoTarget("the target supports linear one-model MTP-Eagle with GDN replay")

    from .qwen3_6_35b_a3b_nvfp4__sm_100__default.modeling import ModelingV2Qwen36Nvfp4Sm100Default

    return ModelingV2Qwen36Nvfp4Sm100Default

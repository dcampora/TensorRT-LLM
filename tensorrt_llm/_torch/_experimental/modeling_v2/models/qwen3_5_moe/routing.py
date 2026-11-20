# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deployment selection for the Qwen3.5 MoE architecture."""

import torch

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

# Qwen3.6-35B-A3B's official checkpoint shares the Qwen3.5 architecture name.
_TEXT_GEOMETRY = {
    "hidden_size": 2048,
    "vocab_size": 248320,
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
}
_LAYER_TYPES = ["linear_attention"] * 3 + ["full_attention"]


@routes("Qwen3_5MoeForConditionalGeneration")
def route(model_config, llm_args, compute_capability):
    """Select the text-only FP8 GB200 target for its supported deployment."""
    if compute_capability != (10, 0) or model_config.mapping.world_size != 1:
        raise NoTarget("Qwen3.6 FP8 requires SM100 and TP 1 on one GPU")
    if not llm_args.disable_mm_encoder or llm_args.mm_encoder_only:
        raise NoTarget("Qwen3.6 FP8 serves text only with disable_mm_encoder=True")
    composite = model_config.pretrained_config
    config = composite.text_config
    for name, expected in _TEXT_GEOMETRY.items():
        if getattr(config, name, None) != expected:
            raise NoTarget(f"Qwen3.6 FP8 requires text_config.{name}={expected}")
    if (
        config.layer_types != _LAYER_TYPES * 10
        or config.hidden_act != "silu"
        or not config.attn_output_gate
        or config.attention_bias
        or config.tie_word_embeddings
        or config.dtype != torch.bfloat16
    ):
        raise NoTarget("Qwen3.6 FP8 requires the checkpoint's BF16 gated hybrid decoder")
    rope = config.rope_parameters
    if rope.get("type") != "mrope" or rope.get("partial_rotary_factor") != 0.25:
        raise NoTarget("Qwen3.6 FP8 requires partial rotary embeddings with factor 0.25")
    quantization = getattr(composite, "quantization_config", None) or {}
    if (
        quantization.get("quant_method") != "fp8"
        or quantization.get("fmt") != "e4m3"
        or quantization.get("weight_block_size") != [128, 128]
        or quantization.get("activation_scheme") != "dynamic"
    ):
        raise NoTarget("Qwen3.6 FP8 requires dynamic E4M3 quantization with 128x128 weight blocks")
    if engine_kv_cache_quant_algo(model_config, llm_args) is not None:
        raise NoTarget("Qwen3.6 FP8 requires an unquantized BF16 attention KV cache")
    if llm_args.kv_cache_config.mamba_ssm_cache_dtype not in ("auto", "bfloat16", "float32"):
        raise NoTarget("Qwen3.6 FP8 requires BF16 or FP32 recurrent state")
    if llm_args.kv_cache_config.use_kv_cache_manager_v2 is False:
        raise NoTarget("Qwen3.6 FP8 requires the V2 hybrid cache manager")
    if llm_args.kv_cache_config.max_attention_window is not None:
        raise NoTarget("Qwen3.6 FP8 requires the full causal attention window")
    if llm_args.cache_transceiver_config is not None:
        raise NoTarget("Qwen3.6 FP8 requires colocated prefill and decode")
    if llm_args.kv_cache_config.enable_block_reuse:
        raise NoTarget("Qwen3.6 FP8 requires KV block reuse disabled")
    if llm_args.attn_backend != "TRTLLM" or model_config.sparse_attention_config is not None:
        raise NoTarget("Qwen3.6 FP8 requires TRTLLM dense attention metadata")
    if model_config.lora_config is not None or llm_args.max_beam_width != 1:
        raise NoTarget("Qwen3.6 FP8 supports no LoRA adapters and one beam")
    if model_config.spec_config is not None:
        raise NoTarget("Qwen3.6 FP8 currently supports decoding without speculation")

    from .qwen3_6_35b_a3b_fp8__sm_100__default.modeling import ModelingV2Qwen3635bA3bFp8Sm100Default

    return ModelingV2Qwen3635bA3bFp8Sm100Default

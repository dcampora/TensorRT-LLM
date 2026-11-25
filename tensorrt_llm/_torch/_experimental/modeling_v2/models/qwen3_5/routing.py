# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route the official Qwen3.8-27B FP8 text tower on GB200."""

from typing import TYPE_CHECKING

from ..._router_index import NoTarget, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs


@routes("Qwen3_5ForConditionalGeneration")
def route(
    model_config: "ModelConfig",
    llm_args: "TorchLlmArgs",
    compute_capability: tuple[int, int],
) -> type:
    """Select the SM100 TP1 text target for the supported checkpoint geometry."""
    if compute_capability != (10, 0):
        raise NoTarget("Qwen3.8-27B FP8 requires SM100")
    mapping = model_config.mapping
    if mapping.world_size != 1 or mapping.enable_attention_dp:
        raise NoTarget("Qwen3.8-27B FP8 requires a single rank with attention DP disabled")

    config = model_config.pretrained_config
    text = getattr(config, "text_config", None)
    expected = {
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "vocab_size": 248320,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attn_output_gate": True,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "partial_rotary_factor": 0.25,
    }
    mismatches = [name for name, value in expected.items() if getattr(text, name, None) != value]
    if mismatches:
        raise NoTarget(f"Qwen3.8-27B text geometry differs in {', '.join(mismatches)}")
    if list(text.layer_types) != (["linear_attention"] * 3 + ["full_attention"]) * 16:
        raise NoTarget("Qwen3.8-27B FP8 requires full attention every fourth layer")

    quantization = getattr(config, "quantization_config", {})
    if any(
        quantization.get(key) != value
        for key, value in {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        }.items()
    ):
        raise NoTarget("Qwen3.8-27B requires dynamic E4M3 with 128 x 128 weight blocks")

    kv = llm_args.kv_cache_config
    if kv.dtype != "auto" or kv.max_attention_window is not None:
        raise NoTarget("Qwen3.8-27B FP8 uses a BF16 KV cache with full history")
    if kv.enable_block_reuse:
        raise NoTarget("Qwen3.8-27B FP8 requires enable_block_reuse=False")
    if llm_args.attn_backend != "TRTLLM" or llm_args.max_beam_width != 1:
        raise NoTarget("Qwen3.8-27B FP8 requires TRTLLM attention metadata and beam width 1")
    if model_config.spec_config is not None:
        raise NoTarget("Qwen3.8-27B FP8 currently supports non-speculative decoding")
    if llm_args.enable_lora or llm_args.mm_encoder_only:
        raise NoTarget("Qwen3.8-27B FP8 serves the text decoder without adapters")
    if llm_args.cache_transceiver_config is not None or llm_args.kv_connector_config is not None:
        raise NoTarget("Qwen3.8-27B FP8 requires a local hybrid KV cache")

    from .qwen3_8_27b_fp8__sm_100__default.modeling import ModelingV2Qwen38_27bFp8Sm100Default

    return ModelingV2Qwen38_27bFp8Sm100Default

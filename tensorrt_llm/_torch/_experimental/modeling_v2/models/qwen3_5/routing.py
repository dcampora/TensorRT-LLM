# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route the text decoder of the Qwen3.8-27B ModelOpt checkpoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, routes

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

_TEXT_FIELDS = {
    "num_hidden_layers": 64,
    "hidden_size": 5120,
    "intermediate_size": 17408,
    "vocab_size": 248320,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "hidden_act": "silu",
    "attn_output_gate": True,
    "output_gate_type": "swish",
    "attention_bias": False,
    "tie_word_embeddings": False,
}

_LAYER_TYPES = ("linear_attention",) * 3 + ("full_attention",)


def _check_quantization(model_config: ModelConfig) -> None:
    quant = model_config.quant_config
    if quant is None or quant.quant_algo != QuantAlgo.MIXED_PRECISION:
        raise NoTarget("Qwen3.8-27B requires the ModelOpt mixed NVFP4/FP8 checkpoint")

    if quant.kv_cache_quant_algo is not None:
        raise NoTarget("Qwen3.8-27B requires BF16 KV entries")

    expected = {"lm_head": QuantAlgo.NVFP4}
    for i, kind in enumerate(_LAYER_TYPES * 16):
        prefix = f"model.language_model.layers.{i}."
        for proj in ("gate_proj", "up_proj", "down_proj"):
            expected[prefix + "mlp." + proj] = QuantAlgo.NVFP4
        if kind == "linear_attention":
            projections = ("in_proj_qkv", "in_proj_z", "out_proj")
            module = "linear_attn"
        else:
            projections = ("q_proj", "k_proj", "v_proj", "o_proj")
            module = "self_attn"
        for proj in projections:
            expected[prefix + module + "." + proj] = QuantAlgo.FP8

    layers = model_config.quant_config_dict or {}
    for name, algo in expected.items():
        layer = layers.get(name)
        if (
            layer is None
            or layer.quant_algo != algo
            or (algo == QuantAlgo.NVFP4 and layer.group_size != 16)
        ):
            raise NoTarget(f"{name} requires {algo.value} with the checkpoint's scale layout")
    if set(layers) != set(expected):
        raise NoTarget("Qwen3.8-27B has an unexpected quantized projection")


@routes("Qwen3_5ForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: tuple[int, int]
) -> type:
    """Select the text-only TP1 SM100 target for its checkpoint and runtime features."""
    mapping = model_config.mapping
    if compute_capability != (10, 0) or mapping.world_size != 1 or mapping.enable_attention_dp:
        raise NoTarget("Qwen3.8-27B NVFP4 requires one SM100 GPU at TP1")
    text = model_config.pretrained_config.get_text_config()
    differences = {
        name: getattr(text, name, None)
        for name, expected in _TEXT_FIELDS.items()
        if getattr(text, name, None) != expected
    }
    rope = getattr(text, "rope_parameters", None) or {}
    if (
        differences
        or tuple(text.layer_types or ()) != _LAYER_TYPES * 16
        or text.dtype != torch.bfloat16
        or rope.get("rope_type", "default") != "default"
        or rope.get("rope_theta", getattr(text, "rope_theta", None)) != 10_000_000
        or rope.get("partial_rotary_factor", getattr(text, "partial_rotary_factor", None)) != 0.25
    ):
        raise NoTarget(f"Qwen3.8-27B text configuration mismatch {differences}")
    _check_quantization(model_config)

    if (
        model_config.attn_backend != "TRTLLM"
        or llm_args.force_dynamic_quantization
        or llm_args.torch_compile_config is not None
    ):
        raise NoTarget("Qwen3.8-27B requires TRTLLM metadata and static checkpoint quantization")
    cache = llm_args.kv_cache_config
    if not llm_args.disable_mm_encoder or llm_args.mm_encoder_only:
        raise NoTarget("Qwen3.8-27B NVFP4 serves text with disable_mm_encoder=True")
    if (
        cache.use_kv_cache_manager_v2 is False
        or cache.dtype not in ("auto", "bfloat16")
        or cache.mamba_ssm_cache_dtype not in ("auto", "float32")
        or cache.enable_block_reuse
        or cache.enable_kv_pool_rebalance
        or cache.max_attention_window is not None
    ):
        raise NoTarget(
            "Qwen3.8-27B requires a full BF16 V2 KV cache with FP32 GDN, "
            "block reuse disabled, and fixed pool ratios"
        )
    if llm_args.enable_chunked_prefill:
        raise NoTarget("Qwen3.8-27B requires whole-request prefill")
    if llm_args.max_beam_width != 1 or model_config.lora_config is not None:
        raise NoTarget("Qwen3.8-27B NVFP4 requires beam width 1 and no LoRA adapters")
    if llm_args.cache_transceiver_config is not None:
        raise NoTarget("Qwen3.8-27B NVFP4 requires one engine for context and generation")
    spec = model_config.spec_config
    if spec is not None:
        if (
            not spec.spec_dec_mode.is_mtp_eagle_one_model()
            or spec.uses_external_draft_model
            or spec.use_dynamic_tree
            or spec.use_relaxed_acceptance_for_thinking
            or spec.sa_config is not None
            or spec.max_concurrency is not None
            or spec.acceptance_rate_window_size
            or not 1 <= spec.max_draft_len <= 4
        ):
            raise NoTarget("Qwen3.8-27B supports linear one-model MTP with up to four drafts")
        schedule = spec.draft_len_schedule
        if schedule is not None and (
            llm_args.max_batch_size is None
            or max(schedule, default=0) < llm_args.max_batch_size
            or any(not 1 <= depth <= spec.max_draft_len for depth in schedule.values())
        ):
            raise NoTarget("Qwen3.8-27B requires at least one draft at every configured batch size")
    if llm_args.prefill_cuda_graph_backend.value not in ("disabled", "breakable"):
        raise NoTarget("Qwen3.8-27B supports breakable prefill CUDA graphs")

    from .qwen3_8_27b_nvfp4__sm_100__default.modeling import ModelingV2Qwen38Nvfp4Sm100Default

    return ModelingV2Qwen38Nvfp4Sm100Default

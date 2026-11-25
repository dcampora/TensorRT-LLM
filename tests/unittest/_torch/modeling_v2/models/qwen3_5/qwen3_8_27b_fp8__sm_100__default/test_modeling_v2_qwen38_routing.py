# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the Qwen3.8 route with the checkpoint's metadata on CPU."""

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MTPDecodingConfig, TorchLlmArgs
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (10, 0))
    monkeypatch.delenv("TRTLLM_USE_GDN_REPLAY", raising=False)
    monkeypatch.delenv("TRTLLM_USE_GDN_FLASHINFER_REPLAY", raising=False)
    text = PretrainedConfig(
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=64,
        num_attention_heads=24,
        num_key_value_heads=4,
        head_dim=256,
        linear_num_key_heads=16,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        vocab_size=248320,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attn_output_gate=True,
        attention_bias=False,
        tie_word_embeddings=False,
        partial_rotary_factor=0.25,
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 16,
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
    )
    return ModelConfig(
        pretrained_config=PretrainedConfig(
            architectures=["Qwen3_5ForConditionalGeneration"],
            text_config=text,
            quantization_config={
                "activation_scheme": "dynamic",
                "fmt": "e4m3",
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            },
        ),
        mapping=Mapping(),
        spec_config=MTPDecodingConfig(max_draft_len=4),
    )


def _arguments(disable_mm_encoder=True, **options):
    return TorchLlmArgs(
        model="/tmp/qwen38-checkpoint",
        kv_cache_config=KvCacheConfig(enable_block_reuse=False, mamba_ssm_cache_dtype="float32"),
        disable_mm_encoder=disable_mm_encoder,
        **options,
    )


@pytest.mark.parametrize("backend", ["disabled", "breakable"])
def test_explicit_route_leaves_the_default_rollout_unchanged(config, backend):
    provided = dict(prefill_cuda_graph_backend=backend)
    assert modeling_v2_resolve(config, _arguments(**provided)) is None
    assert modeling_v2_resolve(config, _arguments(modeling_v2="off", **provided)) is None
    target = modeling_v2_resolve(config, _arguments(modeling_v2="require", **provided))
    assert target is not None and target.__name__ == "ModelingV2Qwen38_27bFp8Sm100Default"


def test_piecewise_prefill_is_declined(config):
    with pytest.raises(ValueError, match="supports breakable prefill CUDA graphs"):
        modeling_v2_resolve(
            config, _arguments(modeling_v2="require", prefill_cuda_graph_backend="piecewise")
        )


def test_media_encoder_is_declined(config):
    with pytest.raises(ValueError, match="media encoder disabled"):
        modeling_v2_resolve(config, _arguments(modeling_v2="require", disable_mm_encoder=False))


def test_incompatible_gpu_and_weight_blocks_are_declined(config, monkeypatch):
    args = _arguments(modeling_v2="require")
    with monkeypatch.context() as other_gpu:
        other_gpu.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0))
        with pytest.raises(ValueError, match="requires SM100"):
            modeling_v2_resolve(config, args)
    config.pretrained_config.quantization_config["weight_block_size"] = [64, 128]
    with pytest.raises(ValueError, match="128 x 128 weight blocks"):
        modeling_v2_resolve(config, args)

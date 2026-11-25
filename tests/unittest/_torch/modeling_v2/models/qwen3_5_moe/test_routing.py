# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the registered Qwen routing policy without weights."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget, registered_routes
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe import routing
from tensorrt_llm.llmapi.llm_args import LoadFormat, MTPDecodingConfig, PrefillCudaGraphBackend
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def deployment():
    text = SimpleNamespace(
        **routing._SHAPE,
        layer_types=routing.LAYER_TYPES,
        attention_bias=False,
        attn_output_gate=True,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.25},
    )
    quant = SimpleNamespace(quant_algo=QuantAlgo.MIXED_PRECISION, kv_cache_quant_algo=QuantAlgo.FP8)
    config = SimpleNamespace(
        pretrained_config=SimpleNamespace(text_config=text),
        quant_config=quant,
        quant_config_dict={
            name: SimpleNamespace(quant_algo=algo, group_size=16)
            for name, algo in routing._EXPECTED_QUANTIZATION.items()
        },
        mapping=SimpleNamespace(world_size=1),
    )
    args = SimpleNamespace(
        disable_mm_encoder=True,
        attn_backend="TRTLLM",
        prefill_cuda_graph_backend=PrefillCudaGraphBackend.DISABLED,
        kv_cache_config=SimpleNamespace(
            dtype="auto",
            use_kv_cache_manager_v2=True,
            mamba_ssm_cache_dtype="float32",
            enable_block_reuse=False,
            enable_kv_pool_rebalance=False,
            max_attention_window=None,
        ),
        max_beam_width=1,
        max_batch_size=128,
        cache_transceiver_config=None,
        load_format=LoadFormat.AUTO,
        enable_lora=False,
        lora_config=None,
        enable_locality_domains=False,
        lm_head_output_dtype="auto",
        speculative_config=None,
        guided_decoding_backend=None,
    )
    return config, args


def test_registered_sm100_target(deployment):
    route = registered_routes()["Qwen3_5MoeForConditionalGeneration"]
    target = route(*deployment, (10, 0))
    assert target.__name__ == "ModelingV2Qwen36Nvfp4Sm100Default"


def test_unsupported_device_or_parallel(deployment):
    config, args = deployment
    with pytest.raises(NoTarget, match="SM100 and a single rank"):
        routing.route(config, args, (10, 3))
    config.mapping.world_size = 2
    with pytest.raises(NoTarget, match="SM100 and a single rank"):
        routing.route(config, args, (10, 0))


def test_refuses_other_quantization(deployment):
    config, args = deployment
    config.quant_config_dict["lm_head"].quant_algo = QuantAlgo.NVFP4
    with pytest.raises(NoTarget, match="ModelOpt FP8 and W4A16 NVFP4"):
        routing.route(config, args, (10, 0))


def test_refuses_other_kv_dtype(deployment):
    config, args = deployment
    args.kv_cache_config.dtype = "bfloat16"
    with pytest.raises(NoTarget, match="FP8 KV cache"):
        routing.route(config, args, (10, 0))


def test_refuses_piecewise_graph(deployment):
    config, args = deployment
    args.prefill_cuda_graph_backend = PrefillCudaGraphBackend.PIECEWISE
    with pytest.raises(NoTarget, match="breakable prefill CUDA graphs"):
        routing.route(config, args, (10, 0))


def test_draft_schedule_keeps_kv_history(deployment):
    config, args = deployment
    args.speculative_config = MTPDecodingConfig(max_draft_len=3, draft_len_schedule={8: 3, 128: 1})
    assert routing.route(config, args, (10, 0))
    args.speculative_config = MTPDecodingConfig(max_draft_len=3, draft_len_schedule={8: 3, 128: 0})
    with pytest.raises(NoTarget, match="retain draft KV history"):
        routing.route(config, args, (10, 0))

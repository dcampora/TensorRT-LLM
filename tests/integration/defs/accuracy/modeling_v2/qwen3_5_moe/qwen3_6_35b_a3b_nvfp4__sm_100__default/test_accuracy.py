# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GSM8K gate for the Qwen3.6 SM100 text target."""

from pathlib import Path

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

from .....conftest import llm_models_root
from ....accuracy_core import GSM8K


@pytest.mark.skipif(get_sm_version() != 100, reason="Qwen3.6 target runs on SM100")
def test_gsm8k(mocker):
    model = Path(llm_models_root()) / "Qwen3.6-35B-A3B-NVFP4"
    if not model.is_dir():
        pytest.skip(f"Model directory {model} does not exist")
    mocker.patch.object(GSM8K, "MAX_OUTPUT_LEN", 512)
    with LLM(
        str(model),
        modeling_v2="require",
        disable_mm_encoder=True,
        max_batch_size=128,
        max_num_tokens=8192,
        max_seq_len=4096,
        enable_chunked_prefill=True,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False, free_gpu_memory_fraction=0.8, mamba_ssm_cache_dtype="float32"
        ),
        cuda_graph_config=CudaGraphConfig(enable_padding=True, max_batch_size=128),
    ) as llm:
        GSM8K("Qwen/Qwen3.6-35B-A3B").evaluate(
            llm,
            extra_evaluator_kwargs={
                "apply_chat_template": True,
                "fewshot_as_multiturn": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

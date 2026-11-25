# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-backed Qwen3.8 text accuracy and graph smoke test."""

from pathlib import Path

import pytest
import torch

from tensorrt_llm import LLM, SamplingParams

from .....conftest import llm_models_root

_PROMPT = [727, 73111, 1393, 1590, 198, 262, 4071, 5423, 279, 307, 7324, 76938, 1324, 71483]


def _checkpoint() -> str:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("The Qwen3.8 NVFP4 target requires SM100")
    checkpoint = Path(llm_models_root()) / "qwen3.8-27b-nvfp4"
    if not checkpoint.is_dir():
        pytest.skip(f"Pretrained checkpoint is unavailable at {checkpoint}")
    return str(checkpoint)


def _options() -> dict:
    return dict(
        modeling_v2="require",
        disable_mm_encoder=True,
        enable_autotuner=False,
        max_batch_size=2,
        max_num_tokens=512,
        max_seq_len=512,
        cuda_graph_config={"batch_sizes": [1, 2]},
        kv_cache_config={
            "free_gpu_memory_fraction": 0.3,
            "enable_block_reuse": False,
            "mamba_ssm_cache_dtype": "float32",
        },
    )


def test_greedy_and_context_logits() -> None:
    prompt = _PROMPT
    with LLM(_checkpoint(), **_options()) as llm:
        scored = llm.generate(
            prompt, SamplingParams(max_tokens=1, temperature=0, return_context_logits=True)
        )
        prompt_logprobs = scored.context_logits[-1].float().log_softmax(dim=-1)
        assert int(prompt_logprobs.argmax()) == 198
        assert float(prompt_logprobs[198]) == pytest.approx(-0.01951, abs=0.04)

        params = SamplingParams(max_tokens=4, temperature=0, ignore_eos=True)
        result = llm.generate([prompt, prompt], params)
        assert result[0].outputs[0].token_ids == result[1].outputs[0].token_ids
        assert result[0].outputs[0].token_ids[0] == 198


def test_mixed_prefill_and_speculative_decode() -> None:
    prompts = (_PROMPT, [*_PROMPT, 271, 272, 198])
    params = SamplingParams(max_tokens=32, temperature=0, ignore_eos=True, detokenize=False)
    with LLM(
        _checkpoint(),
        **_options(),
        speculative_config={"decoding_type": "MTP", "max_draft_len": 4},
    ) as llm:
        solo = [llm.generate(prompt, params).outputs[0].token_ids for prompt in prompts]
        ongoing = llm.generate_async(prompts[0], params, streaming=True)
        assert 0 < len(next(ongoing).outputs[0].token_ids) < params.max_tokens
        new_context = llm.generate_async(prompts[1], params)
        assert [
            ongoing.result().outputs[0].token_ids,
            new_context.result().outputs[0].token_ids,
        ] == solo

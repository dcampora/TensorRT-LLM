# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Development oracle for the SM100 Qwen3.8-27B FP8 token protocol."""

import json
import math
import os
import subprocess
from pathlib import Path

import pytest


def _kl(row: list[list[int | float]], logprobs: list[float]) -> float:
    """KL with each reference token and one bucket for the remaining vocabulary."""
    p_remaining = q_remaining = 1.0
    value = 0.0
    for (_, p_log), q_log in zip(row, logprobs, strict=True):
        p, q = math.exp(p_log), math.exp(q_log)
        p_remaining -= p
        q_remaining -= q
        value += p * (p_log - q_log)
    if p_remaining > 0:
        value += p_remaining * math.log(p_remaining / max(q_remaining, 1e-30))
    return value


@pytest.fixture(scope="module")
def model_path() -> Path:
    root = os.environ.get("LLM_MODELS_ROOT")
    if not root:
        pytest.skip("Set LLM_MODELS_ROOT to the directory containing the FP8 checkpoint")
    model = Path(root) / os.environ.get("QWEN38_FP8_MODEL_DIR", "Qwen3.8-27B-FP8")
    if not (model / "config.json").is_file():
        pytest.skip(f"Qwen3.8-27B FP8 checkpoint unavailable at {model}")
    return model


@pytest.fixture(scope="module")
def dev_oracle() -> dict:
    oracle_path = os.environ.get("QWEN38_DEV_ORACLE")
    if not oracle_path:
        pytest.skip("Set QWEN38_DEV_ORACLE to the development oracle JSON")
    return json.loads(Path(oracle_path).read_text())


def test_qwen38_fp8_dev_oracle(model_path: Path, dev_oracle: dict) -> None:
    """Check normalized scores and every first greedy divergence."""
    model, oracle = model_path, dev_oracle
    root = Path(__file__).resolve().parents[7]
    eos = json.loads((model / "generation_config.json").read_text())["eos_token_id"]
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    vocab_size = json.loads((model / "config.json").read_text())["text_config"]["vocab_size"]
    fragile = {(item["prompt"], item["token"]) for item in oracle["fragile"]}
    total_kl = 0.0
    positions = non_ties = correct_top1 = 0
    failures = []

    with subprocess.Popen(
        [str(root / "engine.sh"), str(model)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        cwd=root,
    ) as engine:
        assert engine.stdin is not None and engine.stdout is not None
        assert json.loads(engine.stdout.readline()) == {"ready": True}

        def exchange(op: str, **args: object) -> dict:
            print(json.dumps({"op": op, **args}), file=engine.stdin, flush=True)
            return json.loads(engine.stdout.readline())

        def call(op: str, **args: object) -> dict:
            response = exchange(op, **args)
            assert "error" not in response, response
            return response

        try:
            assert "error" in exchange(
                "generate", ids=[vocab_size], max_new_tokens=1, ignore_eos=True
            )
            assert call("generate", ids=[1], max_new_tokens=0, ignore_eos=True) == {"ids": []}
            assert call("score", ids=[1], top_k=0, logprob_ids=[]) == {
                "top1": [],
                "top_k": [],
                "logprobs": [],
            }
            for index, item in enumerate(oracle["items"]):
                top_k = item["top_k"]
                prompt = item["prompt_ids"]
                expected = item["gen_ids"]
                probes = [[token for token, _ in row] for row in top_k]
                score = call("score", ids=item["seq_ids"], top_k=2, logprob_ids=probes)
                for row, top1, logprobs in zip(
                    top_k, score["top1"], score["logprobs"], strict=True
                ):
                    total_kl += _kl(row, logprobs)
                    positions += 1
                    if row[0][1] - row[1][1] >= 0.5:
                        non_ties += 1
                        correct_top1 += top1 == row[0][0]

                generated = call(
                    "generate", ids=prompt, max_new_tokens=len(expected), ignore_eos=True
                )["ids"]
                end = next((i + 1 for i, token in enumerate(generated) if token in eos_ids), None)
                if end is not None:
                    stopped = call(
                        "generate", ids=prompt, max_new_tokens=len(expected), ignore_eos=False
                    )["ids"]
                    assert stopped == generated[:end]
                for offset, (actual, wanted) in enumerate(zip(generated, expected, strict=True)):
                    if actual == wanted:
                        continue
                    pos = len(prompt) - 1 + offset
                    row = top_k[pos]
                    if row[0][1] - row[1][1] >= 0.5:
                        actual_scores = dict(score["top_k"][pos])
                        actual_scores.update(zip(probes[pos], score["logprobs"][pos]))
                        is_top1 = actual == score["top1"][pos] or (
                            actual_scores.get(actual) == score["top_k"][pos][0][1]
                        )
                        if (index, offset) not in fragile or not is_top1:
                            failures.append((index, offset, actual, wanted))
                    break
        finally:
            assert call("shutdown") == {"ok": True}
            assert engine.wait(timeout=60) == 0

    assert total_kl / positions <= 0.01
    assert correct_top1 / non_ties >= 0.99
    assert not failures


def test_qwen38_fp8_mixed_prefill(model_path: Path) -> None:
    """Keep short prompt arithmetic when a long request shares the step."""
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    model = model_path
    if oracle_path := os.environ.get("QWEN38_DEV_ORACLE"):
        oracle = json.loads(Path(oracle_path).read_text())["items"]
        short = oracle[28]["prompt_ids"]
        long_a = (oracle[0]["seq_ids"] * 32)[:1024]
        long_b = (oracle[1]["seq_ids"] * 32)[:1024]
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model)
        short = tokenizer.encode("Explain how a paged KV cache reuses memory across requests.")
        long_a = tokenizer.encode("Compare cache latency and memory bandwidth. " * 256)[:1024]
        long_b = tokenizer.encode("Describe one advantage of a streaming scheduler. " * 256)[:1024]
    llm = LLM(
        model,
        modeling_v2="require",
        disable_mm_encoder=True,
        skip_tokenizer_init=True,
        max_batch_size=4,
        max_num_tokens=4096,
        max_seq_len=1536,
        enable_chunked_prefill=True,
        cuda_graph_config={"batch_sizes": [1, 4]},
        speculative_config=MTPDecodingConfig(max_draft_len=3),
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=6144,
            host_cache_size=0,
            mamba_ssm_cache_dtype="float32",
        ),
    )
    params = SamplingParams(
        max_tokens=64, temperature=0, ignore_eos=True, end_id=248046, detokenize=False
    )
    try:
        solo = llm.generate(short, params, use_tqdm=False).outputs[0].token_ids
        mixed = llm.generate([long_a, short, long_b, short], params, use_tqdm=False)
        assert mixed[1].outputs[0].token_ids == solo
        assert mixed[3].outputs[0].token_ids == solo
    finally:
        llm.shutdown()


def test_qwen38_fp8_exact_prefill_graph(model_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay exact buckets and retain the eager result on either side of each bucket."""
    from transformers import AutoTokenizer

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph_runner import BreakableCUDAGraphRunner
    from tensorrt_llm.llmapi import KvCacheConfig
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    buckets = [128, 1024, 4096]
    lengths = [127, 128, 129, 1024, 4096, 1025, 128]
    replays: list[int] = []
    execute = BreakableCUDAGraphRunner.execute

    def count_replay(self, num_tokens, outer_forward):
        if not self.is_capturing_break_graphs:
            replays.append(num_tokens)
        return execute(self, num_tokens, outer_forward)

    monkeypatch.setattr(BreakableCUDAGraphRunner, "execute", count_replay)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    text = tokenizer.encode("Compare the memory usage and throughput of a paged KV cache. " * 512)
    params = SamplingParams(max_tokens=24, temperature=0, ignore_eos=True, detokenize=False)

    def run(prefill_backend: str) -> list[list[int]]:
        with LLM(
            model_path,
            modeling_v2="require",
            disable_mm_encoder=True,
            skip_tokenizer_init=True,
            max_batch_size=1,
            max_num_tokens=8192,
            max_seq_len=8192,
            cuda_graph_config={"batch_sizes": [1]},
            prefill_cuda_graph_backend=prefill_backend,
            prefill_capture_num_tokens=buckets if prefill_backend == "breakable" else None,
            speculative_config=MTPDecodingConfig(max_draft_len=4),
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=False,
                max_tokens=8192,
                host_cache_size=0,
                mamba_ssm_cache_dtype="float32",
            ),
        ) as llm:
            replays.clear()
            return [
                llm.generate(text[:length], params, use_tqdm=False).outputs[0].token_ids
                for length in lengths
            ]

    eager = run("disabled")
    graphed = run("breakable")
    assert graphed == eager
    assert replays == [length for length in lengths if length in buckets]

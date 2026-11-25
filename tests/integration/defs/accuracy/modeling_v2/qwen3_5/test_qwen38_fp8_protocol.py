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


def test_qwen38_fp8_dev_oracle() -> None:
    """Check normalized scores and every first greedy divergence."""
    oracle_path = os.environ.get("QWEN38_DEV_ORACLE")
    if not oracle_path:
        pytest.skip("Set QWEN38_DEV_ORACLE to the development oracle JSON")
    model = Path(os.environ["LLM_MODELS_ROOT"]) / os.environ.get(
        "QWEN38_FP8_MODEL_DIR", "Qwen3.8-27B-FP8"
    )
    root = Path(__file__).resolve().parents[6]
    oracle = json.loads(Path(oracle_path).read_text())
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

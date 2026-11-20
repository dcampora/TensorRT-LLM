# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the Qwen3.6 FP8 development oracle through the JSONL engine."""

import json
import math
import os
import subprocess
from pathlib import Path
from statistics import fmean

import pytest

from tensorrt_llm._utils import get_sm_version


def _request(engine, payload):
    engine.stdin.write(json.dumps(payload) + "\n")
    engine.stdin.flush()
    return _response(engine)


def _response(engine):
    line = engine.stdout.readline()
    assert line, f"engine exited before responding (status {engine.poll()})"
    response = json.loads(line)
    assert "error" not in response, response
    return response


def _check_oracle(engine, oracle):
    fragile = {(entry["prompt"], entry["token"]) for entry in oracle["fragile"]}
    kl, disagreements, eligible = [], 0, 0
    divergences = []
    for index, item in enumerate(oracle["items"]):
        reference = item["top_k"]
        score = _request(
            engine,
            {
                "op": "score",
                "ids": item["seq_ids"],
                "top_k": 2,
                "logprob_ids": [[token for token, _ in row] for row in reference],
            },
        )
        for row, top1, top_k, logprobs in zip(
            reference, score["top1"], score["top_k"], score["logprobs"], strict=True
        ):
            assert len(top_k) == 2 and top_k[0][0] == top1
            kl.append(
                math.fsum(
                    math.exp(ref) * (ref - actual)
                    for (_, ref), actual in zip(row, logprobs, strict=True)
                )
            )
            if row[0][1] - row[1][1] >= 0.5:
                eligible += 1
                disagreements += top1 != row[0][0]

        expected = item["gen_ids"]
        generated = _request(
            engine,
            {
                "op": "generate",
                "ids": item["prompt_ids"],
                "max_new_tokens": len(expected),
                "ignore_eos": True,
            },
        )["ids"]
        assert len(generated) == len(expected)
        for step, (wanted, actual) in enumerate(zip(expected, generated, strict=True)):
            if wanted == actual:
                continue
            position = len(item["prompt_ids"]) + step - 1
            row = reference[position]
            margin = row[0][1] - row[1][1]
            matches_own_top1 = actual == score["top1"][position]
            if margin >= 0.5 and ((index, step) not in fragile or not matches_own_top1):
                divergences.append((index, step, wanted, actual, margin))
            break

    mean_kl = fmean(kl)
    agreement = 1 - disagreements / eligible
    print(f"mean KL {mean_kl:.6f}, non-tie agreement {agreement:.6f}, {divergences=}")
    assert mean_kl <= 0.01
    assert agreement >= 0.99
    assert not divergences


@pytest.mark.skipif(get_sm_version() != 100, reason="Qwen3.6 FP8 Modeling V2 requires SM100")
def test_development_oracle():
    """Check full-vocabulary scores and greedy generations on the development set."""
    default_model = Path(os.environ.get("LLM_MODELS_ROOT", "/")) / "Qwen" / "Qwen3.6-35B-A3B-FP8"
    checkpoint = Path(os.environ.get("QWEN3_6_FP8_CHECKPOINT", default_model))
    oracle = Path(os.environ.get("QWEN3_6_FP8_ORACLE", "/task/oracle_dev.json"))
    if not (checkpoint / "config.json").is_file() or not oracle.is_file():
        pytest.skip("Qwen3.6 FP8 checkpoint and development oracle are required")
    repo = next(
        parent for parent in Path(__file__).resolve().parents if (parent / "engine.sh").is_file()
    )
    with subprocess.Popen(
        [str(repo / "engine.sh"), str(checkpoint)],
        cwd=repo,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as engine:
        try:
            assert _response(engine) == {"ready": True}
            _check_oracle(engine, json.loads(oracle.read_text()))
            assert _request(engine, {"op": "shutdown"}) == {"ok": True}
            assert engine.wait(timeout=60) == 0
        finally:
            if engine.poll() is None:
                engine.terminate()

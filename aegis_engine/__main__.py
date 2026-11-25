# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""JSON-lines interface to the TensorRT-LLM text engine."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from tensorrt_llm import LLM

_MAX_SEQ_LEN = 8192


class Engine:
    def __init__(
        self, llm: LLM, eos_ids: list[int], vocab_size: int, score_llm: LLM | None = None
    ) -> None:
        self.llm = llm
        self.score_llm = score_llm if score_llm is not None else llm
        self.eos_ids = eos_ids
        self.vocab_size = vocab_size

    def generate(self, ids: list[int], length: int, ignore_eos: bool) -> list[int]:
        from tensorrt_llm import SamplingParams

        if length == 0:
            return []
        params = SamplingParams(
            max_tokens=length,
            temperature=0,
            end_id=self.eos_ids[0],
            stop_token_ids=None if ignore_eos else self.eos_ids[1:],
            include_stop_str_in_output=True,
            ignore_eos=ignore_eos,
            detokenize=False,
        )
        llm = self.score_llm if length == 1 else self.llm
        tokens = llm.generate_async(ids, params).result().outputs[0].token_ids
        if ignore_eos and len(tokens) != length:
            raise RuntimeError(f"expected {length} generated tokens, received {len(tokens)}")
        return tokens

    def score(self, ids: list[int], request: dict) -> dict:
        import torch

        from tensorrt_llm import SamplingParams

        count = max(0, len(ids) - 1)
        k = request.get("top_k")
        if "top_k" in request and (type(k) is not int or not 0 <= k <= self.vocab_size):
            raise ValueError("top_k must be an integer between zero and the vocabulary size")
        requested = request.get("logprob_ids")
        if "logprob_ids" in request and (
            not isinstance(requested, list)
            or len(requested) != count
            or any(not self.valid_ids(row) for row in requested)
        ):
            raise ValueError("logprob_ids must contain one list of valid token IDs per position")

        reply = {"top1": []}
        if k is not None:
            reply["top_k"] = []
        if requested is not None:
            reply["logprobs"] = []
        if not count:
            return reply

        params = SamplingParams(
            max_tokens=1,
            temperature=0,
            end_id=self.eos_ids[0],
            ignore_eos=True,
            detokenize=False,
            return_context_logits=True,
        )
        logits = self.score_llm.generate_async(ids, params).result().context_logits
        if logits is None or logits.shape != (len(ids), self.vocab_size):
            raise RuntimeError("TensorRT-LLM did not return one vocabulary vector per position")

        for start in range(0, count, 128):
            block = logits[start : min(start + 128, count)].to(device="cuda", dtype=torch.float32)
            reply["top1"].extend(block.argmax(dim=-1).tolist())
            if k is None and requested is None:
                continue
            log_probs = block.log_softmax(dim=-1)
            if k is not None:
                values, tokens = log_probs.topk(k, dim=-1)
                reply["top_k"].extend(
                    list(zip(token_row, value_row, strict=True))
                    for token_row, value_row in zip(tokens.tolist(), values.tolist(), strict=True)
                )
            if requested is not None:
                rows, columns = [], []
                for row, token_ids in enumerate(requested[start : start + len(block)]):
                    rows.extend([row] * len(token_ids))
                    columns.extend(token_ids)
                selected = iter(log_probs[rows, columns].tolist()) if columns else iter(())
                reply["logprobs"].extend(
                    [next(selected) for _ in token_ids]
                    for token_ids in requested[start : start + len(block)]
                )
        return reply

    def valid_ids(self, ids: object) -> bool:
        return isinstance(ids, list) and all(
            type(token) is int and 0 <= token < self.vocab_size for token in ids
        )

    def dispatch(self, request: dict) -> dict:
        ids = request.get("ids")
        if not ids or not self.valid_ids(ids):
            raise ValueError("ids must be a nonempty list of valid token IDs")
        if request.get("op") == "score":
            if len(ids) >= _MAX_SEQ_LEN:
                raise ValueError(f"score supports sequences shorter than {_MAX_SEQ_LEN} tokens")
            return self.score(ids, request)
        if request.get("op") == "generate":
            length = request.get("max_new_tokens")
            ignore_eos = request.get("ignore_eos")
            if type(length) is not int or length < 0 or type(ignore_eos) is not bool:
                raise ValueError(
                    "generate requires a nonnegative max_new_tokens and boolean ignore_eos"
                )
            if len(ids) + length > _MAX_SEQ_LEN:
                raise ValueError(f"prompt plus output cannot exceed {_MAX_SEQ_LEN} tokens")
            return {"ids": self.generate(ids, length, ignore_eos)}
        raise ValueError("unknown operation")


def run(model: Path, responses: TextIO, mode: str) -> None:
    from tensorrt_llm import LLM

    config = json.loads((model / "config.json").read_text())["text_config"]
    generation = json.loads((model / "generation_config.json").read_text())
    eos_ids = generation.get("eos_token_id", config["eos_token_id"])
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]

    cache_config = dict(
        free_gpu_memory_fraction=0.3,
        enable_block_reuse=False,
        mamba_ssm_cache_dtype="float32",
    )
    options = dict(
        modeling_v2=mode,
        disable_mm_encoder=True,
        skip_tokenizer_init=True,
        generation_config="trtllm",
        enable_autotuner=False,
        max_batch_size=1,
        max_num_tokens=_MAX_SEQ_LEN,
        max_seq_len=_MAX_SEQ_LEN,
        cuda_graph_config={"batch_sizes": [1]},
        kv_cache_config=cache_config,
    )
    if mode == "require":
        options.update(
            prefill_cuda_graph_backend="breakable",
            prefill_capture_num_tokens=[128, 1024, 4096],
        )
    with ExitStack() as stack:
        score_llm = None
        generate_options = options
        if mode == "require":
            generate_options = {
                **options,
                "speculative_config": {"decoding_type": "MTP", "max_draft_len": 4},
            }
        llm = stack.enter_context(LLM(str(model), **generate_options))
        if mode == "require":
            # Ordinary sampling returns context logits and serves requests
            # for a single new token without building drafts.
            score_options = {
                **options,
                "cuda_graph_config": None,
                "kv_cache_config": {**cache_config, "free_gpu_memory_fraction": 0.06},
            }
            score_llm = stack.enter_context(LLM(str(model), **score_options))
        engine = Engine(llm, eos_ids, config["vocab_size"], score_llm)
        for rows in (1, 128, 1024, 4096):
            ids = [config["bos_token_id"]] * rows
            engine.generate(ids, 8, True)
            if score_llm is not None:
                engine.generate(ids, 1, True)
        engine.score([config["bos_token_id"]] * 2, {})
        print(json.dumps({"ready": True}), file=responses, flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("expected a JSON object")
                if request.get("op") == "shutdown":
                    print(json.dumps({"ok": True}), file=responses, flush=True)
                    return
                response = engine.dispatch(request)
            except (RuntimeError, ValueError) as error:
                response = {"error": str(error)}
            print(json.dumps(response), file=responses, flush=True)


def main(mode: str = "require") -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    # Redirect stdout at the descriptor level before loading native libraries.
    sys.stdout.flush()
    with os.fdopen(os.dup(sys.stdout.fileno()), "w") as responses:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        run(args.model, responses, mode)


if __name__ == "__main__":
    main()

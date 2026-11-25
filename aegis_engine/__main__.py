# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TensorRT-LLM driver for the token ID protocol."""

import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from tensorrt_llm import LLM

MAX_SEQ_LEN = 8192


def handle(llm: "LLM", request: dict, eos_ids: list[int], vocab_size: int) -> dict:
    import torch

    from tensorrt_llm import SamplingParams

    operation = request["op"]
    if operation not in ("generate", "score"):
        raise ValueError(f"Unknown operation {operation!r}")
    ids = request["ids"]
    if not isinstance(ids, list) or any(
        type(token) is not int or not 0 <= token < vocab_size for token in ids
    ):
        raise ValueError("ids must be a list of in-vocabulary token IDs")

    if operation == "generate":
        count = request["max_new_tokens"]
        if type(count) is not int or count < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if not ids:
            raise ValueError("generate requires at least one prompt token")
        ignore_eos = request["ignore_eos"]
        if type(ignore_eos) is not bool:
            raise ValueError("ignore_eos must be a boolean")
        if count == 0:
            return {"ids": []}
        if len(ids) + count > MAX_SEQ_LEN:
            raise ValueError(f"Prompt and output exceed the {MAX_SEQ_LEN}-token sequence limit")
        params = SamplingParams(
            temperature=0,
            max_tokens=count,
            end_id=eos_ids[0],
            stop_token_ids=eos_ids[1:],
            include_stop_str_in_output=True,
            ignore_eos=ignore_eos,
            detokenize=False,
        )
        result = llm.generate(ids, params, use_tqdm=False)
        tokens = result.outputs[0].token_ids
        if ignore_eos and len(tokens) != count:
            raise RuntimeError(f"Requested {count} new tokens and received {len(tokens)}")
        return {"ids": tokens}

    top_k = request.get("top_k")
    probes = request.get("logprob_ids")
    count = max(0, len(ids) - 1)
    if top_k is not None and (type(top_k) is not int or not 0 <= top_k <= vocab_size):
        raise ValueError("top_k must be an integer between zero and the vocabulary size")
    if probes is not None and (
        not isinstance(probes, list)
        or len(probes) != count
        or any(
            not isinstance(tokens, list)
            or any(type(token) is not int or not 0 <= token < vocab_size for token in tokens)
            for tokens in probes
        )
    ):
        raise ValueError("logprob_ids must have one list of token IDs per scored position")
    response = {"top1": []}
    if top_k is not None:
        response["top_k"] = []
    if probes is not None:
        response["logprobs"] = []
    if not count:
        return response
    if len(ids) >= MAX_SEQ_LEN:
        # The final supplied ID is the label of the last scored prefix.
        ids = ids[:-1]
    if len(ids) + 1 > MAX_SEQ_LEN:
        raise ValueError(f"Scoring exceeds the {MAX_SEQ_LEN}-token sequence limit")

    params = SamplingParams(
        temperature=0,
        max_tokens=1,
        end_id=eos_ids[0],
        ignore_eos=True,
        return_context_logits=True,
        detokenize=False,
    )
    result = llm.generate(ids, params, use_tqdm=False)
    # Limit FP32 temporaries for long prompts while normalizing each row
    # over the complete vocabulary.
    for begin in range(0, count, 64):
        end = min(begin + 64, count)
        logits = result.context_logits[begin:end].float()
        response["top1"].extend(logits.argmax(dim=-1).tolist())
        if top_k is None and probes is None:
            continue
        logprobs = torch.log_softmax(logits, dim=-1)
        if top_k is not None:
            values, indices = logprobs.topk(top_k, dim=-1)
            response["top_k"].extend(
                list(zip(i, v)) for i, v in zip(indices.tolist(), values.tolist())
            )
        if probes is not None:
            response["logprobs"].extend(
                row[token_ids].tolist()
                for row, token_ids in zip(logprobs, probes[begin:end], strict=True)
            )
    return response


def _build_llm(model: Path, mode: Literal["require", "off"], draft_len: int) -> "LLM":
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    return LLM(
        model,
        modeling_v2=mode,
        skip_tokenizer_init=True,
        disable_mm_encoder=True,
        max_batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        max_num_tokens=MAX_SEQ_LEN,
        cuda_graph_config={"batch_sizes": [1]},
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            max_tokens=MAX_SEQ_LEN,
            host_cache_size=0,
            mamba_ssm_cache_dtype="auto" if mode == "off" and draft_len == 0 else "float32",
        ),
        speculative_config=MTPDecodingConfig(max_draft_len=draft_len) if draft_len else None,
    )


def main(mode: Literal["require", "off"] = "require", draft_len: int = 0) -> None:
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1) as protocol:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        model = Path(sys.argv[1]).resolve()
        vocab_size = json.loads((model / "config.json").read_text())["text_config"]["vocab_size"]
        eos_ids = json.loads((model / "generation_config.json").read_text())["eos_token_id"]
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]

        with ExitStack() as resources:
            generator = _build_llm(model, mode, draft_len)
            resources.callback(generator.shutdown)
            scorer = generator
            if draft_len:
                # The one-model speculative sampler does not expose context logits.
                scorer = _build_llm(model, mode, 0)
                resources.callback(scorer.shutdown)
            print('{"ready": true}', file=protocol)
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    if request["op"] == "shutdown":
                        print('{"ok": true}', file=protocol)
                        break
                    llm = scorer if request["op"] == "score" else generator
                    response = handle(llm, request, eos_ids, vocab_size)
                except (
                    AttributeError,
                    IndexError,
                    KeyError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as error:
                    response = {"error": str(error)}
                print(json.dumps(response), file=protocol)


if __name__ == "__main__":
    main(draft_len=3)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TensorRT-LLM driver for the token ID protocol."""

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from tensorrt_llm import LLM


def handle(llm: "LLM", request: dict, eos_ids: list[int]) -> dict:
    import torch

    from tensorrt_llm import SamplingParams

    operation = request["op"]
    if operation not in ("generate", "score"):
        raise ValueError(f"Unknown operation {operation!r}")
    ids = request["ids"]
    if not isinstance(ids, list) or any(type(token) is not int for token in ids):
        raise ValueError("ids must be a list of token IDs")

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
        return {"ids": result.outputs[0].token_ids}

    top_k = request.get("top_k")
    probes = request.get("logprob_ids")
    count = max(0, len(ids) - 1)
    if top_k is not None and (type(top_k) is not int or top_k < 0):
        raise ValueError("top_k must be a nonnegative integer")
    if probes is not None and (
        not isinstance(probes, list)
        or len(probes) != count
        or any(
            not isinstance(tokens, list)
            or any(type(token) is not int or token < 0 for token in tokens)
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

    params = SamplingParams(
        temperature=0,
        max_tokens=1,
        end_id=eos_ids[0],
        ignore_eos=True,
        return_context_logits=True,
        detokenize=False,
    )
    result = llm.generate(ids, params, use_tqdm=False)
    logits = result.context_logits[:count].float()
    logprobs = torch.log_softmax(logits, dim=-1)
    response["top1"] = logits.argmax(dim=-1).tolist()
    if top_k is not None:
        if top_k > logits.shape[-1]:
            raise ValueError("top_k exceeds the model vocabulary size")
        values, indices = logprobs.topk(top_k, dim=-1)
        response["top_k"] = [list(zip(i, v)) for i, v in zip(indices.tolist(), values.tolist())]
    if probes is not None:
        response["logprobs"] = [
            row[token_ids].tolist() for row, token_ids in zip(logprobs, probes, strict=True)
        ]
    return response


def main(mode: Literal["require", "off"] = "require") -> None:
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1) as protocol:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        from tensorrt_llm import LLM
        from tensorrt_llm.llmapi import KvCacheConfig

        model = Path(sys.argv[1]).resolve()
        eos_ids = json.loads((model / "generation_config.json").read_text())["eos_token_id"]
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        llm = LLM(
            model,
            modeling_v2=mode,
            skip_tokenizer_init=True,
            disable_mm_encoder=True,
            max_batch_size=1,
            cuda_graph_config={"batch_sizes": [1]},
            max_seq_len=8192,
            max_num_tokens=8192,
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=False,
                max_tokens=8192,
                host_cache_size=0,
                mamba_ssm_cache_dtype="auto" if mode == "off" else "float32",
            ),
        )
        try:
            print('{"ready": true}', file=protocol)
            for line in sys.stdin:
                try:
                    request = json.loads(line)
                    if request["op"] == "shutdown":
                        print('{"ok": true}', file=protocol)
                        break
                    response = handle(llm, request, eos_ids)
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
        finally:
            llm.shutdown()


if __name__ == "__main__":
    main()

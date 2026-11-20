# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import sys
import traceback
from contextlib import ExitStack


def _score(llm, request, sampling_params, vocab_size):
    import torch

    ids = request["ids"]
    positions = max(0, len(ids) - 1)
    top_k = request.get("top_k", 0)
    if type(top_k) is not int or not 0 <= top_k <= vocab_size:
        raise ValueError("top_k must be between zero and the vocabulary size")
    requested = request.get("logprob_ids")
    if "logprob_ids" in request and (
        not isinstance(requested, list)
        or len(requested) != positions
        or any(not isinstance(row, list) for row in requested)
    ):
        raise ValueError("logprob_ids must have one list for each scored position")

    response = {"top1": []}
    if "top_k" in request:
        response["top_k"] = []
    if requested is not None:
        response["logprobs"] = []
    if positions == 0:
        return response

    # The final sequence token is the target of the last scored row.
    result = llm.generate(ids[:-1], sampling_params, use_tqdm=False)
    for offset, chunk in enumerate(result.context_logits.split(64)):
        logits = chunk.float()
        values, indices = logits.topk(max(top_k, 1), dim=-1, sorted=True)
        response["top1"].extend(indices[:, 0].tolist())
        if "top_k" not in request and requested is None:
            continue

        log_z = torch.logsumexp(logits, dim=-1)
        if "top_k" in request:
            response["top_k"].extend(
                [
                    list(zip(tokens, scores))
                    for tokens, scores in zip(
                        indices[:, :top_k].tolist(), (values[:, :top_k] - log_z[:, None]).tolist()
                    )
                ]
            )
        if requested is not None:
            row_tokens = requested[offset * 64 : offset * 64 + len(logits)]
            rows = [row for row, tokens in enumerate(row_tokens) for _ in tokens]
            tokens = [token for row in row_tokens for token in row]
            _validate_ids(tokens, vocab_size)
            scores = (logits[rows, tokens] - log_z[rows]).tolist()
            start = 0
            for row in row_tokens:
                end = start + len(row)
                response["logprobs"].append(scores[start:end])
                start = end
    return response


def _validate_ids(ids, vocab_size):
    if not isinstance(ids, list) or any(
        type(token) is not int or not 0 <= token < vocab_size for token in ids
    ):
        raise ValueError("token IDs must be integers in the model vocabulary")


def _run(model, mode):
    # Redirect even native library and worker output away from the protocol.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MTPDecodingConfig

    torch.set_num_threads(4)
    with ExitStack() as stack:

        def load_engine(speculative_config=None):
            return stack.enter_context(
                LLM(
                    model,
                    modeling_v2=mode,
                    disable_mm_encoder=True,
                    skip_tokenizer_init=True,
                    lm_head_output_dtype="float32" if mode == "require" else "auto",
                    max_batch_size=1,
                    max_seq_len=8192,
                    max_num_tokens=8192,
                    kv_cache_config=KvCacheConfig(
                        enable_block_reuse=False,
                        free_gpu_memory_fraction=0.10,
                    ),
                    cuda_graph_config=CudaGraphConfig(batch_sizes=[1]),
                    speculative_config=speculative_config,
                )
            )

        scorer = load_engine()
        # The speculative sampler cannot return teacher-forced context logits.
        generator = load_engine(MTPDecodingConfig(max_draft_len=3)) if mode == "require" else scorer
        _dispatch(protocol, model, scorer, generator, SamplingParams)


def _dispatch(protocol, model, scorer, generator, sampling_params_cls):
    from transformers import GenerationConfig

    config = scorer._hf_model_config
    vocab_size = config.get_text_config().vocab_size
    eos_ids = GenerationConfig.from_pretrained(model).eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    eos, *other_eos = eos_ids
    score_params = sampling_params_cls(
        temperature=0,
        max_tokens=1,
        ignore_eos=True,
        end_id=eos,
        pad_id=eos,
        detokenize=False,
        return_context_logits=True,
    )
    warmup = sampling_params_cls(
        temperature=0, max_tokens=4, ignore_eos=True, end_id=eos, pad_id=eos, detokenize=False
    )
    scorer.generate([eos] * 128, score_params, use_tqdm=False)
    generator.generate([eos] * 128, warmup, use_tqdm=False)
    print(json.dumps({"ready": True}), file=protocol)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            operation = request["op"]
            if operation == "shutdown":
                print(json.dumps({"ok": True}), file=protocol)
                break
            if operation not in ("score", "generate"):
                raise ValueError(f"unknown operation {operation!r}")
            ids = request["ids"]
            _validate_ids(ids, vocab_size)
            if operation == "score":
                response = _score(scorer, request, score_params, vocab_size)
            else:
                count = request["max_new_tokens"]
                ignore_eos = request["ignore_eos"]
                if type(count) is not int or count < 0 or type(ignore_eos) is not bool:
                    raise ValueError(
                        "expected a nonnegative max_new_tokens and a boolean ignore_eos"
                    )
                if not ids:
                    raise ValueError("generation requires at least one prompt token")
                if count:
                    params = sampling_params_cls(
                        temperature=0,
                        max_tokens=count,
                        ignore_eos=ignore_eos,
                        end_id=eos,
                        pad_id=eos,
                        stop_token_ids=None if ignore_eos else other_eos,
                        include_stop_str_in_output=True,
                        detokenize=False,
                    )
                    result = generator.generate(ids, params, use_tqdm=False)
                    response = {"ids": result.outputs[0].token_ids}
                else:
                    response = {"ids": []}
            payload = json.dumps(response, allow_nan=False)
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            traceback.print_exc()
            payload = json.dumps({"error": str(error)})
        print(payload, file=protocol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--mode", choices=("require", "off"), default="require")
    arguments = parser.parse_args()
    _run(arguments.model, arguments.mode)

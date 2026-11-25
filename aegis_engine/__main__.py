# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive the TensorRT-LLM API over the Aegis token-ID protocol."""

import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path


def token_ids(values, vocab_size, *, allow_empty=False):
    if not isinstance(values, list) or (not values and not allow_empty):
        raise ValueError("expected a list of token IDs")
    if any(type(value) is not int or not 0 <= value < vocab_size for value in values):
        raise ValueError("token ID outside the model vocabulary")
    return values


def handle(generator, scorer, request, vocab_size, eos):
    import torch

    from tensorrt_llm import SamplingParams

    op = request["op"]
    if op not in ("generate", "score"):
        raise ValueError(f"unknown operation {op!r}")
    ids = token_ids(request["ids"], vocab_size)
    common = dict(temperature=0, detokenize=False, add_special_tokens=False)
    if op == "generate":
        count = request["max_new_tokens"]
        ignore_eos = request.get("ignore_eos", False)
        if type(count) is not int or count < 0 or type(ignore_eos) is not bool:
            raise ValueError("invalid max_new_tokens or ignore_eos")
        if not count:
            return {"ids": []}
        params = SamplingParams(
            max_tokens=count,
            ignore_eos=ignore_eos,
            end_id=eos[0],
            stop_token_ids=None if ignore_eos else eos[1:],
            include_stop_str_in_output=True,
            **common,
        )
        output = generator.generate(ids, params, use_tqdm=False)
        return {"ids": list(output.outputs[0].token_ids)}

    rows = len(ids) - 1
    top_k = request.get("top_k")
    if "top_k" in request and (type(top_k) is not int or not 0 <= top_k <= vocab_size):
        raise ValueError("invalid top_k")
    wanted = request.get("logprob_ids")
    if "logprob_ids" in request:
        if not isinstance(wanted, list) or len(wanted) != rows:
            raise ValueError("logprob_ids must have one list per scored position")
        for row in wanted:
            token_ids(row, vocab_size, allow_empty=True)

    result = {"top1": []}
    if top_k is not None:
        result["top_k"] = []
    if wanted is not None:
        result["logprobs"] = []
    if not rows:
        return result
    output = scorer.generate(
        ids, SamplingParams(max_tokens=1, return_context_logits=True, **common), use_tqdm=False
    )
    logits = output.context_logits[:rows, :vocab_size].float()
    result["top1"] = logits.argmax(-1).tolist()
    if top_k is not None or wanted is not None:
        logprobs = torch.log_softmax(logits, dim=-1)
    if top_k is not None:
        values, indices = logprobs.topk(top_k, dim=-1)
        result["top_k"] = [list(zip(i, v)) for i, v in zip(indices.tolist(), values.tolist())]
    if wanted is not None:
        result["logprobs"] = [row[indices].tolist() for row, indices in zip(logprobs, wanted)]
    return result


def main():
    sys.stdout.flush()
    reply = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MTPDecodingConfig

    model = Path(sys.argv[1])
    config = json.loads((model / "config.json").read_text())
    vocab_size = config["text_config"]["vocab_size"]
    eos = json.loads((model / "generation_config.json").read_text())["eos_token_id"]
    if isinstance(eos, int):
        eos = [eos]
    mode = os.environ.get("AEGIS_MODELING_V2", "require")
    drafts = int(os.environ.get("AEGIS_MTP_DRAFTS", "3" if mode == "require" else "0"))
    options = dict(
        modeling_v2=mode,
        disable_mm_encoder=True,
        skip_tokenizer_init=True,
        max_batch_size=1,
        max_seq_len=8192,
        max_num_tokens=8192,
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1]),
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False,
            free_gpu_memory_fraction=0.25,
            mamba_ssm_cache_dtype="float32",
            max_tokens=16384 if drafts else None,
        ),
    )
    prefill_graph_options = {}
    if mode == "require" and os.environ.get("AEGIS_PREFILL_GRAPHS", "1") == "1":
        prefill_graph_options.update(
            prefill_cuda_graph_backend="breakable",
            prefill_capture_num_tokens=[128, 256, 512, 1024, 2048, 4096],
        )

    def send(value):
        print(json.dumps(value, separators=(",", ":")), file=reply, flush=True)

    with ExitStack() as stack:
        scorer_options = {**options, **({} if drafts else prefill_graph_options)}
        scorer = stack.enter_context(LLM(str(model), **scorer_options))
        generator = scorer
        if drafts:
            # The speculative sampler needs a separate ordinary decoder for
            # teacher-forced context logits.
            generator = stack.enter_context(
                LLM(
                    str(model),
                    speculative_config=MTPDecodingConfig(max_draft_len=drafts),
                    **options,
                    **prefill_graph_options,
                )
            )
        send({"ready": True})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request["op"] == "shutdown":
                    send({"ok": True})
                    break
                send(handle(generator, scorer, request, vocab_size, eos))
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                send({"error": str(exc)})


if __name__ == "__main__":
    main()

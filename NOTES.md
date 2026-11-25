# 2026-11-25 08:16:05 UTC
<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## Modeling V1 baseline

TensorRT-LLM 1.4.0rc4 from this checkout, GB200, TP1, text-only, block reuse off.
The stock Qwen3.5 implementation loads the Hugging Face FP8 checkpoint as supplied.
Its default recurrent cache is BF16. No package or C++ rebuild was needed.

All 32 development items were scored and generated through the LLM API with
`modeling_v2="off"`. Mean KL was 0.004480 and non-tie top-1 agreement was
99.771%. Eight continuations were exact and 21 first diverged at a near-tie.
Three first divergences were outside the permitted cases (items 15, 25, 28).
The stock baseline therefore misses the greedy gate on this oracle.

Batch 1, OSL 1024. TTFT is the elapsed time for OSL 1. TPOT is the external
elapsed-time difference between OSL 32 and OSL 1024, divided by 992.
The chat prompts quote English passages from this checkout's documentation.

| ISL | OSL | TTFT (ms) | TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 68.92 | 7.712 | 129.66 |
| 1024 | 1024 | 68.11 | 8.159 | 122.56 |
| 4096 | 1024 | 145.65 | 7.765 | 128.78 |

The initial OpenAI sweep used `/task/bench_serving.py` against the stock
server. The requests used the benchmark's default count (at least 16, or
four per concurrency slot). V2 bring-up overlapped parts of this sweep,
so a matched, isolated comparison is needed for final speedups.

| ISL | OSL | Concurrency | Output tokens/s | Median TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 125.92 | 70.00 |
| 1024 | 1024 | 2 | 246.89 | 147.92 |
| 1024 | 1024 | 4 | 471.91 | 166.71 |
| 1024 | 1024 | 8 | 852.73 | 319.27 |
| 1024 | 1024 | 16 | 1599.30 | 475.12 |
| 1024 | 1024 | 32 | 2763.88 | 705.88 |
| 1024 | 1024 | 64 | 3597.54 | 945.20 |
| 1024 | 1024 | 128 | 4972.34 | 955.82 |
| 4096 | 1024 | 1 | 83.05 | 226.16 |
| 4096 | 1024 | 8 | 796.49 | 904.99 |
| 4096 | 1024 | 32 | 1917.37 | 1107.32 |

## Modeling V2 implementation

The SM100 TP1 target loads the language tower from the original HF tensor names.
It fuses QKV, QKVZ and SwiGLU gate/up at load time. Dynamic activation quantization
uses groups of 128 channels. It converts the weight scales to powers of two
using the reference engine's Blackwell conversion. The weight conversion rounds
the intermediate to BF16 within registers and stores only E4M3 weights and
block scales. The vocabulary projection and embeddings stay BF16 as shipped.

The TensorRT-LLM scheduler owns packed batches, cache pages and recurrent slots.
The target feeds normalized, partially rotated Q/K to native paged attention.
Full-attention K/V are BF16. The supplied launchers use FP32 recurrent state
and BF16 convolution history. Fresh short contexts use the triangular Gated
DeltaNet solve and FP32 attention probabilities. FlashInfer handles longer
contexts and continuation chunks with explicit initial and final state slots.

The target-local Triton/Gluon QKV, gated attention, recurrent update, activation
quantizer, FP8 scale conversion and small-M split-K use code from the
`aegis-engines` reference (`114cf57c911e`, `kernels.py`). The short GDN solve
comes from that engine's `gdn.py`. Convolution and state writeback use the
TensorRT-LLM cache manager's physical slots and measured strides. The attention
call uses the shared Modeling V2 catalog's `thop_attention` interface. The
Qwen3.5 route is opt-in, so the default rollout still selects its existing
Modeling V1 implementation.

The first non-speculative V2 build passed all 32 development items. Mean KL was
0.003288, non-tie top-1 agreement was 99.909%, and every greedy continuation
was exact (12) or first diverged at a near-tie (20). The SM100 operation tests
exercise nonconsecutive cache slots, chunk continuation and scale/GEMM layouts.
They and the existing V2 routing tests recorded 473 passes and 76 skips.

The first cold load compiled FlashInfer's SM100 GEMM extension in about five
minutes. Native TensorRT-LLM dense attention requires a single QKV buffer on
this release. Its warmup also passes synthetic mRoPE metadata for text, which
the target accepts while rejecting media payloads.

The serving launcher reached the OpenAI endpoint on port 8092 and returned a
successful completion using the required Modeling V2 route.

The first concurrent serving check exposed sensitivity to batch shape on
development item 28. It passed at concurrency 1 and first diverged at
token 4, at the 0.5-nat cutoff, at concurrency 32 and 128. The reference
engine also found that short contexts and the BF16 GDN gate projection were
sensitive to batching. This remained open in the initial bring-up.

## Standalone reference rerun

I ran a copy of the reference branch's `engine.sh` against the same
checkpoint and development oracle. The cache and native build went under
`/tmp` and the home directory. The run used the installed CUTLASS headers
from FlashInfer and quoted passages from the TensorRT-LLM documentation.

The reference passed all B1, B32 and B128 greedy checks. B1 mean KL was
0.003288 and non-tie top-1 agreement was 99.909%. B1/B32 each had 11 exact,
20 near-tie and one fragile continuation. B128 had 44 exact, 80 near-tie and
four fragile continuations. The reference evaluation reports a memory-gate
failure. The local run also allocated over 120 GiB during startup.

Each row reports the median of two external measurements. TTFT uses OSL 1.
Decode measures output tokens 33 through 1024 and sums TPS across the static
batch. The B1/ISL 1024 measurements overlapped a V2 diagnostic and are
included for completeness.

| Batch | ISL | OSL | TTFT (ms) | TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1024 | 15.71 | 3.670 | 272.47 |
| 1 | 1024 | 1024 | 55.24 | 4.904 | 203.94 |
| 1 | 4096 | 1024 | 181.62 | 4.342 | 230.31 |
| 32 | 1024 | 1024 | 1356.26 | 7.677 | 4168.22 |
| 128 | 1024 | 1024 | 5545.52 | 17.196 | 7443.61 |

## Batch stability

Isolating short prefill GEMMs left the item 28 divergence in place.
Pinning the BF16 GDN gate GEMM alone also left it in place. I retained
the gate kernel for the consistent split-K schedule and added the
reference engine's split-softmax decode reduction for histories up to
256 tokens. The adapted kernel reads TensorRT-LLM's strided BF16 pages.
Native attention appends KV and handles longer histories. The short-history
reduction restored the expected output on the item 28 reproduction at both
B1 and B32.

All 32 batch-1 oracle items still passed with mean KL 0.003288 and 99.909%
non-tie top-1 agreement. The result included 10 exact, 21 near-tie and one
accepted fragile continuation. All 33 target operation tests passed on GB200.

Batch-1 external times for the non-speculative V2 build follow. TTFT uses
OSL 1 and TPOT covers tokens 33 through 1024. This path uses the runtime's
decode graphs and overlap scheduler.

| ISL | OSL | TTFT (ms) | TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 65.24 | 9.867 | 101.34 |
| 1024 | 1024 | 68.49 | 9.873 | 101.29 |
| 4096 | 1024 | 196.38 | 9.929 | 100.71 |

The CUDA toolkit build for the fixed GDN gate projection comes
from reference `ba_lt.cpp` and `ba_lt.py`. Its shared library caches under
`~/.cache/tensorrt_llm/qwen38_fp8_sm100`. The C++ implementation would
move to the target's `csrc/` in a source-built contribution.

The OpenAI token-ID check reproduced the batch-1 continuations at
concurrency 1, 32 and 128. Across the three runs there were zero
unaccepted first divergences against the development oracle and zero
token-ID differences from the batch-1 V2 results.

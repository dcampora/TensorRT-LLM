Wed Nov 25 08:16:00 UTC 2026

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.8-27B NVFP4 on GB200

Text-only inference uses TP1, a BF16 paged KV cache and FP32 Gated DeltaNet state.
The checkpoint calibrates the MLP and vocabulary projections for group-16 NVFP4,
and the full- and linear-attention projections for static FP8. The embedding,
GDN B/A gates, convolution, and MTP weights are BF16 in the checkpoint.

## Baseline

The stock Modeling V1 loader accepted the checkpoint with the vision encoder
disabled. The first attempt spent over six minutes in startup. A stack sample
showed the engine still compiling the low-M CuTe FP8 tactics in the autotuner.
The measured baseline uses autotuning disabled for an initial same-configuration
comparison.

The Modeling V1 LLM API ran all 32 development prompts. It returned mean oracle
top-20 KL 0.019660 and 98.75% non-tie
top-1 accuracy over 2161 non-tie positions. There were
7 unaccepted greedy divergences.

### Modeling V1 batch 1, ISL 128, 1024, 4096 and OSL 1024

Each row is the median of three runs on distinct thinking-enabled prompts with
block reuse disabled. The same prompt is used for the 32- and 1024-output pair.

| ISL | OSL | TTFT (ms) | TPOT after 32 (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 70.44 | 9.963 | 100.37 |
| 1024 | 1024 | 73.97 | 9.969 | 100.31 |
| 4096 | 1024 | 122.84 | 9.990 | 100.10 |

### Modeling V1 serving, ISL 1024 and OSL 1024

The supplied closed-loop serving benchmark used one wave of requests per
concurrency after a 16-output warmup wave. It used technical passages from
the TensorRT-LLM coding and developer guides.

| Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 96.38 | 65.00 | 10.322 |
| 32 | 32 | 2033.31 | 627.67 | 15.121 |
| 128 | 128 | 5277.61 | 1935.27 | 22.265 |

## Modeling V2 bring-up

Routing matches the composite architecture, per-layer quantization map, and
SM100 TP1 deployment. After routing, the target gives the executor the text
sub-config so it allocates the existing Mamba hybrid cache. The target takes
BF16 K/V pages, BF16 convolution history, and FP32 recurrent states from that
manager. Fresh prefills use packed causal GQA and cu-seqlens. Generation uses
the engine's CUDA graph and request slots. Context continuations read the
carried GDN state and use the same causal GQA over the manager's paged K/V.
When an iteration mixes context and generation, each portion uses its own
attention geometry and the runtime's request order.

The target's weight layout, FP8 scale consolidation, padded short prefill,
and SwiGLU packing come from aegis-engines `cb6779a` in `engine.py` and
`nvfp4_pack.py`. The convolution, recurrent update, rotary embedding, and
split-KV attention are adapted from its `kernels.py`. The triangular solve
and recurrent chunk handoff are from `prefill.py`. The target ports its
`fp4_tactics.py`, `fp8_tactics.py`, and `fp4_tma.py`. TMA copies the
FP8 scale bits through byte-typed shared memory on Triton 3.8 and interprets
them as FP8 at the tensor-core instruction. The launcher takes the scale
format and CUDA target from the checkpoint deployment.

The first model pass used the catalog's attention operation throughout and
scored conditional top-20 KL 0.021411, non-tie top-1 98.61%, and 11
unaccepted greedy divergences. Porting the reference attention arithmetic to
TensorRT-LLM's page tables reduced conditional KL to 0.011746 and improved
non-tie top-1 to 99.17%. All 32 greedy outputs passed the divergence rule.
Widening convolution products to FP32 increased conditional KL to 0.022975
and produced six unaccepted greedy divergences.

Keeping the BF16-rounded prefill beta in an FP32 buffer reduced conditional
KL to 0.000804 and raised non-tie top-1 to 99.95%. In that run the one
unaccepted greedy continuation selected token 20403 at a position where the
oracle's top-20 distribution also favored 20403, but the oracle's separately
generated sequence selected 8964. The position was marked fragile and the
target's teacher-forced score favored 8964, so the generated token did not
qualify for the exception. Teacher-forced and incremental inference can
differ because the chunk solve rounds intermediates at different points
from the single-token recurrence.

The initial validated target uses the two-term softplus expansion below
-8.5 to avoid cancellation in the small exponential tail. Its raw top-20
KL was 0.008055, its conditional top-20 KL was 0.008032, and non-tie top-1
agreement was 99.54% over 2161 positions. The full JSON-lines test accepted
all 32 development continuations. One was identical and the remaining 31
were accepted at 26 near-ties and 5 fragile positions with the scored
argmax.

The test suite passed 28 SM100 operation cells and the checkpoint-backed
context-logits and two-request CUDA graph test.

### Modeling V2 batch 1, ISL 128, 1024, 4096 and OSL 1024

Measurements for the prefill-beta diagnostic above. Each row is the median
of three paired runs with 32- and 1024-output controls.

| ISL | OSL | TTFT (ms) | TPOT after 32 (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 65.03 | 6.070 | 164.73 |
| 1024 | 1024 | 70.14 | 6.192 | 161.51 |
| 4096 | 1024 | 126.12 | 6.622 | 151.02 |

The native NVFP4 GEMM tactics, indexed convolutions, GDN state updates, and
heads that stage paged K/V are candidates for the family's `csrc/` when a
full native build is available.

### Validated Modeling V2 serving, ISL 1024 and OSL 1024

`serve.sh` returned the configured model name and token-ID completions.
The OpenAI completions endpoint passed the development oracle at
concurrency 1, 32, and 128. Relative to the batch-1 engine's complete token
lists, 32 of 32, 26 of 32, and 103 of 128 responses respectively were
identical. The oracle checks apply its near-tie and fragile-position rule
to each response.

| Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 160.34 | 65.92 | 6.178 |
| 32 | 32 | 2941.47 | 726.76 | 10.167 |
| 128 | 128 | 5359.29 | 2033.36 | 21.826 |

### Split MTP, batch 1, ISL 128, 1024, 4096 and OSL 1024

MTP loads the checkpoint's BF16 draft layer and shares the trunk's packed
NVFP4 vocabulary projection. The draft runs through TensorRT-LLM's
MTP-Eagle worker, using its own paged attention layer. The indexed
convolution and temporal verify kernels adapt the standalone's
`spec_kernels.py`. They replay the accepted GDN updates on the GPU and
leave candidate deltas in two banks so the scheduler can accept a prefix
without copying a full temporal state for every verified token.

The shared GDN replay allocator accepts a history dtype from the target.
The Qwen3.8 target selects FP32 history to preserve the state update
precision and the existing allocator default is BF16. Disabling replay
uses the runtime's ordinary full-state checkpoints. Both paths produced
identical batch-1 development generations. The contract wrapper uses
ordinary sampling for teacher-forced context logits and four drafts for
generation. Across the 32 development prompts, it measured raw top-20
KL 0.008055, conditional KL 0.008032, and non-tie top-1 agreement of
99.54%. All 32 greedy generations passed and matched the preceding
batch-1 engine token for token. Three operation cells cover the verify
and replay transitions, including bank flips and independent state slots.

Each row is the external median of three paired 32- and 1024-output
measurements with the same natural-English chat prompts as the baseline.

| ISL | OSL | TTFT (ms) | TPOT after 32 (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 64.83 | 3.097 | 322.87 |
| 1024 | 1024 | 64.94 | 3.421 | 292.32 |
| 4096 | 1024 | 133.05 | 3.662 | 273.05 |

### Smaller verifier tiles, batch 1, ISL 128, 1024, 4096 and OSL 1024

Four value coordinates per warp improve parallelism in the narrow GDN
verifier. The deployment also uses the runtime's MTP fast paths and
overlap loop on its execution stream. The full engine check retained
raw KL 0.008055, conditional KL 0.008032, 99.54% non-tie top-1, and
32 accepted generations. The engine outputs matched the preceding
commit token for token.

| ISL | OSL | TTFT (ms) | TPOT after 32 (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 78.08 | 2.472 | 404.51 |
| 1024 | 1024 | 79.36 | 2.748 | 363.86 |
| 4096 | 1024 | 134.41 | 2.984 | 335.09 |

### Dynamic MTP trial, ISL 1024 and OSL 1024

The paired paged-attention kernel adapts `batch_spec_kernels.py` from
aegis-engines. Adjacent verify rows share a K/V tile when their split
widths match. A split-boundary test verifies bitwise agreement with
independent queries, including an odd final query and indexed page tables.

The trial used four drafts through batch 4, three through batch 32,
and one through batch 128. It passed the greedy oracle at concurrency
1, 32, 64, and 128. Intermediate-concurrency tests found divergences
outside the rule, so the wrapper still uses ordinary decoding.

| Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 346.75 | 77.23 | 2.811 |
| 32 | 32 | 3571.17 | 891.94 | 7.553 |
| 128 | 128 | 5334.44 | 2332.52 | 20.960 |

### Dynamic MTP trial, ISL 4096 and OSL 1024

| Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 295.03 | 144.34 | 3.252 |
| 8 | 8 | 1287.49 | 994.78 | 4.950 |
| 32 | 32 | 2293.27 | 2615.56 | 10.664 |

### Mixed-step serving correctness

Context and generation rows now retain their respective projection
tactics inside a mixed scheduler iteration. The target returns both
parts in the runtime's packed order. Short contexts share the calibrated
128-row prefill shape from the standalone's `prefill_groups`, with
separate groups when their combined token count would exceed it. Giving
GDN's output kernel per-sequence tiles alone had no effect on the
remaining differences.

The split phases and short-context grouping accepted all 384 development
responses at concurrency 1, 2, 4, 8, 16, 32, 64, and 128. Of those, 382
matched the batch-1 engine exactly. The other two diverged at the same
position, where its two leading tokens had equal log-probabilities.
The serving wrapper enables the verified dynamic MTP schedule and uses
the TP1 executor in one process with one HTTP frontend.
The engine retained the preceding commit's 32 generations and score
metrics. All 32 operation cells and both checkpoint-backed tests passed,
including overlapping a new prefill with an active speculative decode.

### Closed-loop serving at a3ce5a1c, ISL 1024 and 4096, OSL 1024

The supplied serving benchmark used its default 16-request minimum
and four requests per concurrency slot. The HTTP server stayed up
after a client connection disconnected between cases. The remaining
cases ran with fresh client sessions and their standard warmup waves.

| ISL | Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1 | 16 | 349.73 | 69.85 | 2.801 |
| 1024 | 2 | 16 | 543.73 | 121.48 | 3.443 |
| 1024 | 4 | 16 | 876.97 | 137.13 | 4.359 |
| 1024 | 8 | 32 | 1518.61 | 144.19 | 5.032 |
| 1024 | 16 | 64 | 2281.22 | 148.75 | 6.600 |
| 1024 | 32 | 128 | 3166.02 | 220.57 | 9.670 |
| 1024 | 64 | 256 | 3745.97 | 375.56 | 15.746 |
| 1024 | 128 | 512 | 4616.39 | 439.76 | 26.978 |
| 4096 | 1 | 16 | 310.75 | 141.56 | 3.139 |
| 4096 | 8 | 32 | 1305.71 | 165.80 | 5.645 |
| 4096 | 32 | 128 | 2330.04 | 369.19 | 12.751 |

### Captured batch-1 prefill, ISL 128, 1024, 4096 and OSL 1024

The runtime's breakable prefill runner keeps a graph for each bucket.
The target captures the fresh single-sequence trunk with persistent
position, state-slot, and page-table views. The runtime can run the same
target eagerly for other context shapes. At TP1, runtime warmup registered
the break without capturing its inner graph, so the target records that
graph on the first matching replay. The contract wrapper warms the
configured shapes before readiness.

The graph trial without inner capture took 88.69 and 80.70 ms to first
token at ISL 128 and 1024. With the inner graphs, the oracle scores and
all 32 continuations matched the preceding version.

| ISL | OSL | TTFT (ms) | TPOT after 32 (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 20.45 | 2.476 | 403.90 |
| 1024 | 1024 | 42.75 | 2.749 | 363.78 |
| 4096 | 1024 | 134.04 | 2.985 | 335.06 |

### Batched GDN verification

At ISL 1024 and OSL 64, Nsight Systems attributed 40% of the batch-32
decode graph's kernel time and 52% at batch 128 to GDN verification.
On batch 128, the original request-major view put 144 MiB between
adjacent state slots. Separate native pools gave a 3 MiB stride. Both
layouts averaged 350 microseconds per GDN verify, so the runtime's
existing layout stays in use.

The batched verifier ports convolution with query/key normalization
and gate preparation from aegis-engines' `batch_spec_kernels.py`.
It computes those inputs once per head and updates eight value
coordinates per warp. With four pending history rows in a rotating raw
kernel probe, batch-32 verification fell from 137 to 90 microseconds and
batch 128 fell from 302 to 213 microseconds. The replay cells compare
BF16 outputs and convolution windows bit for bit against full FP32
checkpointing across history bank switches.

### Prepared GDN serving, ISL 1024 and OSL 1024

The one-wave measurements use the same prompts as the dynamic MTP trial.
All 384 serving checks still passed, with the same two tied divergences
from the contract engine.

| Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 32 | 3733.80 | 1122.83 | 6.923 |
| 128 | 128 | 5638.72 | 2852.12 | 19.263 |

## Standalone comparison

The original `/ref-engine/engine.sh` copy crashed while instantiating its
adaptive conditional CUDA graph with this image's CUDA driver. The rerun
with `QWEN_ADAPTIVE_DRAFT=0`, `QWEN_FP4_TMA=0` and prompt caching disabled
processed all 32 development items. Its conditional top-20 KL was 0.011746,
raw top-20 KL was 0.011454, and non-tie top-1 was 99.17%. All 32 greedy
generations passed. The conditional KL uses both distributions
renormalized over the reference's top 20 token IDs.

### Standalone engine, ISL 128, 1024, 4096 and OSL 1024

These measurements use batch 1, thinking-enabled passages and 32-output
controls. Each row is the median of two paired measurements.

| ISL | OSL | TTFT (ms) | TPOT after 32 (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 10.02 | 2.413 | 414.36 |
| 1024 | 1024 | 30.07 | 2.644 | 378.23 |
| 4096 | 1024 | 109.53 | 2.885 | 346.56 |

### Standalone warmed batches, ISL 1024 and OSL 1024

The copied reference ran one full warmup for each batch size, followed
by three one-output requests and three paired 32- and 1024-output
requests. The timings use the same compatibility settings and disable
prompt caching. Each batch uses distinct thinking-enabled chat prompts.

| Batch | TTFT (ms) | TPOT after 32 (ms) | Aggregate decode tokens/s |
| ---: | ---: | ---: | ---: |
| 32 | 782.72 | 6.090 | 5254.26 |
| 128 | 3172.90 | 14.334 | 8929.58 |

## MTP draft front and single-output requests

The draft front ports the embedding gather and the two RMSNorms from
aegis-engines' `draft_front.py` into a single target operator. The
single-request draft projections use FlashInfer's CuTe BF16 GEMM.
One-output contract requests use the ordinary executor's prefill graph,
since the request ends with the first verified token. All 36 operation
cells passed. The contract's 32 oracle outcomes and the 384 serving
comparisons were unchanged.

### Contract engine, ISL 128, 1024, 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | TPOT after 32 (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 14.33 | 2.454 | 407.42 |
| 1024 | 1024 | 36.70 | 2.701 | 370.28 |
| 4096 | 1024 | 125.28 | 2.944 | 339.62 |

### One-wave serving, ISL 1024 and OSL 1024

| Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 32 | 3749.64 | 1098.92 | 6.970 |
| 128 | 128 | 5688.52 | 2783.28 | 19.114 |

## First-row GDN checkpoint

A verifier always consumes its first input token. The recurrence now
commits the FP32 state after that row and records only subsequent draft
corrections, following aegis-engines' `batch_spec_kernels.py`. The target
selects a one-step history in the shared GDN allocator and its capacity
estimator. The runtime's existing bank promotion, state slots, and
convolution promotion apply at each accepted prefix. New operation cells
cover widths five and two when the allocator reserves five rows. All
38 cells passed. The full contract scores and generations matched the
previous commit, and all 384 serving responses passed.

A separate contract probe verified malformed-input recovery, zero-output
generation, ragged score requests, and inclusive EOS stopping. Captured
prefill and eager prefill emitted identical 64-token continuations on ten
distinct prompts at ISLs 127, 128, 257, 1024, and 4096.

### Contract engine, ISL 128, 1024, 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | TPOT after 32 (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 15.04 | 2.414 | 414.22 |
| 1024 | 1024 | 36.42 | 2.655 | 376.61 |
| 4096 | 1024 | 126.00 | 2.914 | 343.19 |

### One-wave serving, ISL 1024 and OSL 1024

| Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 32 | 3887.93 | 1123.86 | 6.601 |
| 128 | 128 | 5789.07 | 3040.77 | 18.510 |

### Modeling V1 dynamic-MTP comparison, ISL 1024 and 4096, OSL 1024

Changing the serving YAML's selector to `"off"` reaches Modeling V1. Its
GDN replay implementation aborts CUDA graph warmup when two verification
rows reach a replay buffer reserved for five rows. Running the same
Modeling V1 configuration with `TRTLLM_USE_GDN_REPLAY=0` enables the
runtime's full-state checkpoint path and starts serving. The benchmark
below used the same one-wave prompts as the Modeling V2 measurements.

| ISL | Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1 | 1 | 188.36 | 79.57 | 5.236 |
| 1024 | 32 | 32 | 3553.43 | 764.28 | 6.824 |
| 1024 | 128 | 128 | 4521.97 | 2483.82 | 24.707 |
| 4096 | 1 | 1 | 202.34 | 134.06 | 4.816 |

## Projection locality for mixed iterations

Context and decode groups keep their calibrated matrix shapes. The target
runs a layer over all the groups before advancing to the next layer and
schedules each weight's projections together, including the output and
MLP matrices. This lets the GPU reuse weights in L2 across groups.
The groups share the runtime-owned cache views and keep independent
state and page slots.

Both checkpoint-backed integration tests and all 38 operation cells
passed. The 32 contract cases and all 384 serving comparisons matched
the first-row checkpoint results. Batch-1 decode measured 414.83,
374.63, and 344.99 tokens/s at ISLs 128, 1024, and 4096 with OSL 1024,
close to the preceding measurements.

### One-wave serving, ISL 1024 and OSL 1024

| Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 32 | 3852.03 | 1047.99 | 6.708 |
| 128 | 128 | 5970.47 | 2626.63 | 18.269 |

### Sustained serving, ISL 1024 and 4096, OSL 1024

The serving benchmark used its standard minimum of 16 requests and four
requests per concurrency slot.

| ISL | Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 32 | 128 | 3591.93 | 153.56 | 8.424 |
| 1024 | 128 | 512 | 5370.95 | 408.16 | 23.244 |
| 4096 | 32 | 128 | 2583.52 | 355.97 | 11.392 |

## FP8 QKV with a paged cache epilogue

The full-attention decode projection ports aegis-engines'
`fp8_tma.py`. The Gluon kernel accumulates two FP32 halves from the
original FP8 storage layout. The rotary kernel adds and scales the
halves, rounds once to BF16, and writes K and V directly to the
runtime's page table. Other row counts still use cuBLASLt and share
the direct paged write. Six rotating checkpoint layers matched
cuBLASLt's BF16 projection bit for bit. At five, 128, and 256 rows
the projection fell from 17.29, 28.13, and 41.88 microseconds to
14.82, 21.10, and 36.79 microseconds respectively. The epilogue and
TMA pipeline are candidates for a native `csrc/` build.

All 42 operator cells passed, including direct writes into shuffled
pages and across page boundaries. The contract's scores and greedy
continuations were identical to the preceding version. The server
started and all 384 serving comparisons passed.

### Contract engine, ISL 128, 1024, 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | TPOT after 32 (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 15.34 | 2.390 | 418.34 |
| 1024 | 1024 | 37.29 | 2.629 | 380.37 |
| 4096 | 1024 | 126.35 | 2.873 | 348.08 |

### One-wave serving, ISL 1024 and OSL 1024

| Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 32 | 3895.50 | 929.55 | 6.788 |
| 128 | 128 | 5869.31 | 2639.91 | 18.524 |

## Projection and recurrent controls

The checkpoint gives every fused FP8 projection's shards the same
activation scale and separate weight scales. The target uses the
checkpoint weight scales to repack the shards to a common FP8 scale,
as in aegis-engines' ModelOpt loader. Running the FP8 shards separately
with their original scales produced raw KL 0.019954, non-tie top-1
98.43%, and six unaccepted development generations. The fused loader's
raw KL is 0.008055.

The two-part TMA tactic for GDN QKVZ took 16.66, 33.68, and 48.99
microseconds at 5, 128, and 256 rows across six checkpoint layers.
The cuBLASLt projections took 14.76, 17.78, and 21.34 microseconds.
GDN QKVZ continues to use cuBLASLt. A factored recurrent update built
with TF32x3 also changed rounding and took 390.86 microseconds per
batch-128 layer, versus about 230 for the exact FP32 verifier.

## Long-prompt serving parity

A longer test used eight distinct prompts at each of ISL 128, 1024,
and 4096, with OSL 128. Chunked prefill produced six distinct
concurrency-32/128 divergences with a batch-1 score gap of at least
0.5 nats. Grouping contexts by equal length had the same result.
Prefilling each long request separately had the same result with
chunking enabled and matched batch 1 when each request's prefill
ran in one iteration.

The serving configuration now disables context chunking. It still
packs context and generation requests into the same scheduler
iteration. The prompt and per-iteration token budgets are both 8192,
so every supported prompt can fit in an iteration. With only this
configuration change, all 208 long-prompt responses through
`serve.sh` at concurrency 1, 8, 32, and 128 matched the batch-1
engine token for token. The 32 contract cases and 384 development
serving checks passed again.

The shared replay allocator has defaults for its two target options.
The CPU tests for the hybrid cache managers passed 257 tests with
one skipped.

### One-wave serving trial, ISL 1024 and OSL 1024

| Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 32 | 4002.88 | 918.52 | 6.675 |
| 128 | 128 | 6087.69 | 2453.34 | 18.078 |

### Sustained whole-prefill serving, ISL 1024 and 4096, OSL 1024

| ISL | Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 32 | 128 | 3576.05 | 145.79 | 8.447 |
| 1024 | 128 | 512 | 5409.93 | 430.49 | 22.934 |
| 4096 | 32 | 128 | 2590.62 | 351.55 | 11.367 |

### Matched Modeling V1 serving, ISL 1024 and 4096, OSL 1024

This run uses the same whole-prefill YAML with `modeling_v2: "off"`.
The V1 dynamic draft schedule requires `TRTLLM_USE_GDN_REPLAY=0`
to use full-state checkpointing. Each cell uses the standard
warmup, 16 requests minimum, and four requests per concurrency
slot. The input hashes match the corresponding V2 cells.

| ISL | Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1 | 16 | 215.44 | 96.14 | 4.911 |
| 1024 | 2 | 16 | 374.71 | 128.47 | 5.170 |
| 1024 | 4 | 16 | 617.53 | 140.78 | 6.125 |
| 1024 | 8 | 32 | 1252.58 | 162.10 | 6.036 |
| 1024 | 16 | 64 | 2340.17 | 161.24 | 6.183 |
| 1024 | 32 | 128 | 3416.93 | 264.62 | 8.781 |
| 1024 | 64 | 256 | 3625.85 | 358.07 | 16.499 |
| 1024 | 128 | 512 | 4612.22 | 448.97 | 25.854 |
| 4096 | 1 | 16 | 202.11 | 138.24 | 4.968 |
| 4096 | 8 | 32 | 1227.94 | 163.27 | 5.897 |
| 4096 | 32 | 128 | 2558.43 | 305.98 | 10.278 |

## Warp-local exact batched recurrence

The wide verifier keeps four value coordinates and each full 128-element
key reduction in one warp. The runtime's existing dummy-request mask
lets both recurrent preparation and verification skip the padding slots.
The FP32 state update and the two-bank replay format stay identical. On
rotating native-stride cache views, the verifier took 88.94 microseconds
at batch 32 with four verifier rows and 202.37 microseconds at batch 128
with two verifier rows. The preceding kernel took 95.90 and 232.47
microseconds. The outputs and committed states were bitwise equal.

All 44 operation cells passed, including graph padding and replaying a
wider accepted prefix in a narrower subsequent iteration. All 32 contract
cases, 384 development serving responses, and 208 long serving responses
passed. The 208 long responses matched batch 1 token for token. The only
development serving differences from batch 1 occurred at the same zero-gap
tie in one concurrency-64 replica and one concurrency-128 replica. An ordinary temperature 0.8 HTTP
request also completed.

### Contract engine, ISL 128, 1024, 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | TPOT after 32 (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 14.08 | 2.389 | 418.56 |
| 1024 | 1024 | 35.66 | 2.632 | 380.00 |
| 4096 | 1024 | 126.27 | 2.867 | 348.80 |

### One-wave serving, ISL 1024 and OSL 1024

| Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 32 | 4110.60 | 927.00 | 6.452 |
| 128 | 128 | 6261.63 | 2681.45 | 17.304 |

## Overlapped gate projections and paged attention

The GDN B/A projection runs on a CUDA stream alongside the FP8 QKVZ
projection, following the two-stream execution in aegis-engines `engine.py`.
Both streams read the same normalized BF16 input and join before updating
recurrent state. Six rotating checkpoint layers preserved every result.
The combined projection time at 5, 128, and 256 rows fell from 27.46,
26.62, and 38.54 microseconds to 23.06, 21.56, and 31.70 microseconds.

For a 32-token decode tile aligned to a page, a single page-table lookup
serves the entire tile. The pair kernel stages three loads. With 32-token
pages and sequence lengths up to 2048, this changed the two-query
kernel from 113.87 to 103.46 microseconds at batch 32 and from 216.47 to
198.17 microseconds at batch 128. The paged attention outputs were bitwise
identical. Grouping several GDN heads per CTA preserved the recurrence but
yielded less than a 1% gain at batch 128, so it stayed experimental.

The 44 operation cells and all 32 contract cases passed with the
same score metrics. All 384 development serving checks passed and all
208 long-prompt serving responses matched batch 1 token for token.

### Contract engine, ISL 128, 1024, 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | TPOT after 32 (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 15.63 | 2.311 | 432.80 |
| 1024 | 1024 | 36.76 | 2.547 | 392.56 |
| 4096 | 1024 | 126.06 | 2.788 | 358.62 |

### One-wave serving, ISL 1024 and OSL 1024

| Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 32 | 4197.19 | 934.87 | 6.265 |
| 128 | 128 | 6423.66 | 2496.88 | 16.977 |

## Additional decode controls

Periodic exact GDN checkpoints stored fewer state rows, with a longer
replay at the following step. All tested buffers and outputs matched the
existing deferred-replay reference bitwise. At batch 128, the best
periodic kernel took 205.90 microseconds per layer, compared with 202.29
for committing the first verifier row. At batch 32 the best periodic
kernel took 90.20 microseconds, compared with 89.11. The first-row
commit remains the default.

A four-query paged attention kernel reused the same K/V loads across
both query pairs. It matched the pair kernel bit for bit and improved
an isolated 2048-token-capacity batch-32 case from 98.59 to 96.74
microseconds. The gain was smaller after adopting pipelined page loads,
so the target continues to use the pair kernel.

The draft MLP kept the original BF16 GEMMs. PyTorch's BF16 linear
projection took 60.08 microseconds for gate and up and 32.95 for down
at one row. Alternate CuTe, cuBLASLt, and cuDNN tactics gave no
consistent gain over the one-, five-, 32-, and 128-row controls.

### Batch-1 draft-depth sweep, ISL 128, 1024, 4096 and OSL 1024

Both diagnostics passed all 32 oracle cases. The engine uses four drafts
for all three lengths. Timings below use the same paired contract requests.

| ISL | Four drafts (decode tokens/s) | Five drafts (decode tokens/s) | Six drafts (decode tokens/s) |
| ---: | ---: | ---: | ---: |
| 128 | 432.80 | 430.23 | 439.43 |
| 1024 | 392.56 | 389.72 | 386.27 |
| 4096 | 358.62 | 348.61 | 344.71 |

## Selected first-draft rows

The split MTP worker supplies one selected output row per request to its
first draft pass. The draft layer writes every candidate K/V row before it
computes attention and the feed-forward path on the selected rows. In
pure decode, it reuses the target's fused rotary and paged-write operation.
The greedy target sampler reads the BF16 head output directly. Advanced
sampling and guided decoding still receive FP32 target logits.

The 44 operation cells, 32 contract cases, and 384 development serving
checks passed. All 208 long-prompt serving responses matched batch 1.
Removing the full-row draft epilogue reduced serving TTFT and slightly
increased throughput. Batch-1 decode decreased by about 1% in this trial.

### Contract engine, ISL 128, 1024, 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | TPOT after 32 (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 13.41 | 2.334 | 428.37 |
| 1024 | 1024 | 35.91 | 2.572 | 388.81 |
| 4096 | 1024 | 125.64 | 2.813 | 355.54 |

### One-wave serving, ISL 1024 and OSL 1024

| Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 32 | 4234.40 | 883.80 | 6.242 |
| 128 | 128 | 6541.41 | 2372.53 | 16.741 |

### Standalone engine on the one-wave serving prompts, ISL 1024 and OSL 1024

The standalone mirror received the exact token IDs from the one-wave
OpenAI benchmark's chat prompts. The full-request rate includes prefill
and is measured at the JSON-lines boundary. Decode throughput uses the
paired 32- and 1024-output requests. The prompt hashes match both OpenAI
cells in the table above. The standalone container used the same driver
compatibility settings as the first comparison.

| Static batch | Full-request output tokens/s | Decode tokens/s | TTFT (ms) | TPOT after 32 (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 4700.79 | 5294.74 | 792.58 | 6.044 |
| 128 | 7373.04 | 8986.41 | 3185.91 | 14.244 |

## Irregular decode projections and TMA prefill

The target rounds irregular decode projection batches to a supported
cuBLASLt row count. Recurrence and paged attention still see the live
rows and the runtime's original request ordering. The pinned tiles use
the original K partitions and reduction precision. The QKV rotary
operation can read FP32 partitions with a padded row stride.

Four rotating checkpoint layers matched on 18 decode row counts for
all three FP8 projection geometries. At 254 live rows, padding to 256
reduced GDN QKVZ from 158.08 to 28.17 microseconds, full-attention QKV
from 230.70 to 46.90, and the attention output projection from 104.92
to 28.24, including the padding. The target also applies its tuned
NVFP4 down projection at the corresponding padded shapes.

Fresh packed prefill uses the TMA attention from aegis-engines
`prefill_attention.py` at 512 or more rows. The existing Triton
attention still handles carried paged contexts. The raw TMA probe
matched at every tested shape, including uneven packed contexts.
It reduced 1024- and 4096-token attention from 107.83 and 1120.43
microseconds to 88.72 and 915.04 per layer. The software pipeline
and paged epilogue are candidates for a native `csrc/` build.

The fixed-depth MTP engine keeps the runtime's fused first-pass
metadata epilogue. The scheduled serving batches keep the selected
first-draft rows. All 46 operation cells, 32 contract cases, and 384
serving oracle checks passed. The 208 long-prompt serving responses
again matched batch 1 token for token.

### Contract engine, ISL 128, 1024, 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | TPOT after 32 (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 15.10 | 2.311 | 432.62 |
| 1024 | 1024 | 36.60 | 2.544 | 393.14 |
| 4096 | 1024 | 121.77 | 2.782 | 359.45 |

### One-wave serving, ISL 1024 and OSL 1024

| Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 32 | 4267.57 | 832.39 | 6.259 |
| 128 | 128 | 6618.51 | 2262.56 | 16.550 |

### Sustained serving with tuned projections, ISL 1024 and 4096, OSL 1024

| ISL | Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 32 | 128 | 3455.01 | 176.05 | 8.964 |
| 1024 | 128 | 512 | 5509.14 | 438.27 | 22.660 |
| 4096 | 32 | 128 | 2848.99 | 298.58 | 10.212 |

The native FlashInfer SM100 paged-attention trial started and served on
BF16 pages. It produced four unaccepted development outputs at
concurrency 32 and sixteen at concurrency 128. The target keeps its
validated paged attention. Two-warp pair attention also changed
rounding and took longer than four warps. The standalone's integer
vocabulary reduction was exact, including ties and NaNs. At 256 rows
it took 70.67 microseconds and the runtime's BF16 argmax took 49.59.

### Serving draft-depth controls, ISL 1024 and OSL 1024

These one-wave runs keep the same requests and graph sizes and vary the
number of drafts at the indicated concurrency.

| Concurrency | Drafts | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 2 | 4030.79 | 820.77 | 6.825 |
| 32 | 3 | 4267.57 | 832.39 | 6.259 |
| 32 | 4 | 3514.13 | 852.45 | 7.559 |
| 128 | 1 | 6618.51 | 2262.56 | 16.550 |
| 128 | 2 | 5665.89 | 2661.62 | 18.835 |

Aligning the context chunks to 64-token cache pages still diverged from
batch 1. For example, a 4096-token input at concurrency 32 diverged at
output position 2, where the batch-1 top-two gap was 0.875 nats.
The whole-prefill configuration retains exact long-prompt parity.

## Benchmark inputs

The serving runs use the supplied `/task/bench_serving.py` and the
following passage. The passage SHA-256 is
`cfcd655d2cbe567c352941de8bba832b610ad39a806c036a39b230e419d56731`.

```bash
cat CODING_GUIDELINES.md \
    docs/source/developer-guide/overview.md \
    docs/source/developer-guide/ci-overview.md \
    docs/source/developer-guide/perf-benchmarking.md > /tmp/qwen-passage
./serve.sh /model --port 8000
```

Run the benchmark against the ready server from a separate shell.

```bash
python3 /task/bench_serving.py --url http://127.0.0.1:8000 \
    --model qwen3.8-27b-nvfp4 --passage /tmp/qwen-passage
```

The one-wave tables use `--minimum 1 --multiplier 1`. Sustained tables
use the benchmark defaults. The contract timings use the same passage
with exact-length thinking-enabled chats and the 32-output control.

## First-token prefill and full-output parity

The ordinary single-context graph now computes only the final query tile
and final feed-forward row in the last layer. It appends the request's
complete K/V state. MTP and teacher-forced scoring still produce all
required hidden rows. The final-row path follows aegis-engines
`engine.py`. The target caches up to 32 immutable chunk layouts keyed by
context lengths and device.

Short single-prompt prefills overlap the two GDN input projections.
Rotating checkpoint layers gave bitwise-identical outputs. At 128 rows
the two projections fell from 26.58 to 21.53 microseconds. At 256 they
fell from 33.79 to 28.49. The pinned FP4 tactics at 128 rows also
matched bitwise. They reduced the gate/up GEMM from 21.81 to 19.70
microseconds and down from 18.78 to 17.43.

The route requires whole-request prefill after the alignment control.
The normal serving configuration continues to mix whole-context and
decode work in one scheduler step.

All 49 operation cells, 32 contract oracle cases, and 384 development
serving checks passed. The one-token and MTP paths picked the same
first token on all 56 checked prompts. At OSL 128, all 208 serving
responses matched batch 1 token for token. At OSL 1024, all 160
responses at concurrency 32 and 128 matched their batch-1 runs
across 24 natural-language prompts at ISLs 128, 1024, and 4096.

### Contract engine, ISL 128, 1024, 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | TPOT after 32 (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 13.42 | 2.309 | 433.07 |
| 1024 | 1024 | 35.63 | 2.547 | 392.66 |
| 4096 | 1024 | 120.61 | 2.784 | 359.19 |

### One-wave serving, ISL 1024 and OSL 1024

| Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 32 | 4263.59 | 852.15 | 6.233 |
| 128 | 128 | 6614.25 | 2305.95 | 16.600 |

## Fresh-context convolution and wide recurrent verification

Fresh packed contexts specialize the existing indexed depthwise
convolution. The convolution omits carried-history loads when the
runtime marks the batch fresh and uses a separate time and channel tile
for long contexts. The carried-context path retains its prior arithmetic.
The raw kernel outputs matched bit for bit at lengths 128, 1024 and 4096,
for eight 1024-token prompts together, and for packed lengths 37, 1024
and 71. At the three individual lengths, kernel times were 5.55, 20.57
and 65.58 microseconds, down from 8.92, 52.25 and 187.99.

In the wide two-row verifier, a warp processes four consecutive
state tiles with the same FP32 reduction order and the same read and
write banks. The raw verifier matched both the outputs and committed
states bit for bit. At batches 64, 96 and 128, the per-layer timings
were 98.87, 144.59 and 189.88 microseconds, down from 102.71,
152.22 and 202.05.

### Contract engine, ISL 128, 1024, 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | TPOT after 32 (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 13.25 | 2.307 | 433.45 |
| 1024 | 1024 | 33.47 | 2.545 | 392.99 |
| 4096 | 1024 | 114.06 | 2.783 | 359.31 |

The operation and checkpoint-integration suite passed all 52 cases.
The contract engine again passed all 32 development prompts with
mean KL 0.008055 and 99.537% non-tie agreement over 2,161 positions.

### Serving configuration controls, ISL 1024 and OSL 1024

The control uses the preceding first-token prefill build. The sustained
control at concurrency 32 delivered 3,630.59 output tokens/s over
128 requests. The three alternatives below use one wave with 32 or
128 requests. The distributed graph buckets and higher token budget
changed wave throughput by less than 1%. Streaming every four steps
raised median TTFT at concurrency 32 from 852 to 1,011 milliseconds.
The launcher retains its original graph buckets, token budget and
stream interval.

| Configuration | Concurrency 32 output tokens/s | Concurrency 128 output tokens/s |
| --- | ---: | ---: |
| Launcher defaults | 4263.59 | 6614.25 |
| Finer CUDA graph buckets | 4276.41 | 6659.56 |
| 12,288-token step budget | 4280.76 | 6630.43 |
| Quiet host and four-step streaming | 4178.17 | 6632.79 |

### One-wave serving with fresh-context convolution, ISL 1024 and OSL 1024

| Concurrency | Requests | Aggregate output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 32 | 4313.19 | 797.92 | 6.206 |
| 128 | 128 | 6655.56 | 2857.06 | 15.994 |

The launched server passed all 384 development comparisons under
the oracle rule. The two differences from batch 1 were at the same
zero-gap token. All 208 long-prompt OSL 128 responses and all 160
long-prompt OSL 1024 responses matched batch 1 exactly. All 56
ordinary one-token requests matched the corresponding first MTP
output. A sampled completion at temperature 0.8 also returned.

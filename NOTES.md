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

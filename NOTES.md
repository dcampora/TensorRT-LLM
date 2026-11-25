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

## Standalone comparison

The original `/ref-engine/engine.sh` copy crashed while instantiating its
adaptive conditional CUDA graph with this image's CUDA driver. The rerun
with `QWEN_ADAPTIVE_DRAFT=0`, `QWEN_FP4_TMA=0` and prompt caching disabled
processed all 32 development items. Its conditional top-20 KL was 0.011746,
non-tie top-1 was 99.17%, and no greedy generation diverged outside the
accepted positions. The conditional KL uses both distributions
renormalized over the reference's top 20 token IDs.

### Standalone engine, ISL 128, 1024, 4096 and OSL 1024

These measurements use batch 1, thinking-enabled passages and 32-output
controls. Each row is the median of two paired measurements.

| ISL | OSL | TTFT (ms) | TPOT after 32 (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 10.02 | 2.413 | 414.36 |
| 1024 | 1024 | 30.07 | 2.644 | 378.23 |
| 4096 | 1024 | 109.53 | 2.885 | 346.56 |

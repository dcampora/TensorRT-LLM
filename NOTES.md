Wed Nov 25 08:21:08 UTC 2026

# Qwen3.6-35B-A3B NVFP4 on GB200

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

## Initial port

The target runs the text tower on SM100 at TP 1. Its launcher sets
`modeling_v2="require"` and the serving launcher uses the same route. The
router checks the checkpoint geometry, hybrid schedule, per-layer ModelOpt
formats, FP8 KV cache and FP32 recurrent state. It keeps the target opt-in.

The decoder uses the TensorRT-LLM context-first token layout. For full
attention it takes the manager's HND pool views and reads K and V page offsets
from the runtime's GPU tables. For Gated DeltaNet it reads state-slot indices
and updates the manager's FP32 state and BF16 convolution history. Context
continuations gather the initial convolution frames before overwriting their
slots. FlashInfer's indexed chunked recurrence updates the same state pool.
Both launchers use the engine's CUDA graphs for generation.

Ported from aegis-engines `a6c09c6502`:

- `engine.py` supplied the text decoder's layer order, projection packing,
  residual rounding, expert dispatch and vocabulary readout.
- `kernels.py` supplied the FP8 casting, ModelOpt NVFP4 arithmetic, residual
  norm, router, convolution, recurrence and attention kernels. The new
  attention and recurrent kernels address runtime tables and slot strides.
- `cute_moe.py`, `fused_gate.py`, `shared_corrected.py` and
  `dense_vectors.py` supplied grouped W4A16, its SwiGLU epilogue, the guarded
  shared-expert path and shared down projection.

Weights stay in their checkpoint formats. FP8 projections fuse shards at the
maximum checkpoint scale using the reference engine's register-level rounding.
The W4A16 experts, shared experts and vocabulary keep E2M1 weights, E4M3 group
scales and FP32 global scales. Their epilogues reproduce the reference engine's
BF16 rounding of the global scale. K and V are E4M3. Native CuTe grouped
projection and SwiGLU integration would belong under the target's `csrc/` in
a full build. The port uses the installed FlashInfer CuTe kernels and caches
its adjusted epilogue in the user's FlashInfer workspace.

The first development pass scored 2,524 token positions, including 2,193
with a reference margin of at least 0.5 nats. Mean KL was 0.005527 and non-tie
top-1 agreement was 99.9088%. Greedy output was identical on 11 prompts, first
diverged at a near-tie on 19, and first diverged at an accepted fragile position
on 2.

The V1 smoke through the same JSON-lines wrapper completed generation and
context-logit scoring. Its first startup spent several minutes compiling
low-M CuTe GEMM candidates during autotuning. Startup compilation is outside
the steady-state measurements.

`serve.sh /model --port 8091` loaded the routed Modeling V2 class, captured
the configured generation CUDA graphs, and returned HTTP 200 from both
`/v1/models` and `/v1/completions`. The operation and routing tests ran on
GB200, including strided recurrent slots, split-KV decode and cached prefill.

## Initial external measurements

Batch 1 with OSL 1024 through `engine.sh`. TTFT is a separate one-token
request. TPOT is the difference between 32- and 1024-token requests
divided by 992. Prompts used the checkpoint chat template and technical
prose from the TensorRT-LLM developer guide.

| ISL | OSL | TTFT (ms) | TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 56.35 | 3.463 | 288.8 |
| 1024 | 1024 | 63.02 | 3.728 | 268.2 |
| 4096 | 1024 | 118.59 | 3.925 | 254.7 |

V1 with FP32 recurrent state completed the same 32-item oracle. Mean KL was
0.019041 and non-tie top-1 agreement was 98.9056%. Four greedy first divergences
missed the allowed tie or fragile checks. The V1 run uses the shipped kernels
and checkpoint loader with `modeling_v2="off"`.

The Modeling V1 batch-1 runs used the same JSON-lines launcher.

| ISL | OSL | TTFT (ms) | TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 65.42 | 5.071 | 197.2 |
| 1024 | 1024 | 61.60 | 5.070 | 197.2 |
| 4096 | 1024 | 69.31 | 5.063 | 197.5 |

Modeling V1 through `serve.sh` with `AEGIS_MODELING_V2=off` and the same
cache, scheduler and CUDA graph configuration. The first sweep completed
through concurrency 64. `bench_serving.py` used prose from the TensorRT-LLM
developer guide because its default source paths were absent.

| ISL | OSL | Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 179.6 | 86.19 | 5.092 |
| 1024 | 1024 | 2 | 380.9 | 159.42 | 5.078 |
| 1024 | 1024 | 4 | 711.0 | 136.68 | 5.495 |
| 1024 | 1024 | 8 | 1285.5 | 153.11 | 6.079 |
| 1024 | 1024 | 16 | 1229.7 | 283.73 | 12.807 |
| 1024 | 1024 | 32 | 3022.8 | 270.22 | 10.388 |
| 1024 | 1024 | 64 | 5111.8 | 271.81 | 12.342 |
| 1024 | 1024 | 128 | 8769.5 | 267.03 | 14.308 |
| 4096 | 1024 | 1 | 193.5 | 89.74 | 5.083 |
| 4096 | 1024 | 8 | 1220.7 | 337.52 | 6.203 |
| 4096 | 1024 | 32 | 3075.4 | 340.40 | 10.045 |

The retries completed the 1024-token concurrency-128 and 4096-token
concurrency-32 measurements. The 4096-token sweep first disconnected
during warmup at concurrency 32. A fresh benchmark client ran that level
in isolation and completed 64 measured requests.

## Standalone comparison

I ran `/ref-engine/engine.sh` from a writable archive of the read-only
checkout with the same chat prompt builder and external timer as the
batch-1 test. The standalone engine passed all 32 items with KL 0.005527
and non-tie top-1 agreement of 99.9088%. Its 32- and 128-sequence batches
also passed all greedy checks against the development oracle.

| Batch | ISL | OSL | TTFT (ms) | TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1024 | 6.97 | 1.327 | 753.8 |
| 1 | 1024 | 1024 | 14.62 | 1.245 | 803.1 |
| 1 | 4096 | 1024 | 36.74 | 1.308 | 764.4 |
| 32 | 1024 | 1024 | — | 4.813 | 6649.1 |
| 128 | 1024 | 1024 | — | 8.993 | 14233.7 |

## Fused MTP verification

`speculation.py`, `spec_kernels.py` and `batch_spec_kernels.py` from
aegis-engines supplied the one-layer draft path, greedy verification and
accepted-row draft dispatch. The target processes all candidate rows in the
main decoder and projects them through the full NVFP4 vocabulary. It runs
the draft head at the accepted row and searches 131,072 draft vocabulary
entries on batches below 96 and 65,536 at larger batch sizes.
Attention shares each request's paged reads across its candidate rows. The
draft layer writes its own runtime-managed KV pool.

Gated DeltaNet verifies candidates against the manager's FP32 anchor and
retains BF16 keys and updates plus FP32 gates in its replay slots. Each
layer advances its anchor before its replay slot rolls over. The one shared
runtime change carries this capability from the routed model to the hybrid
cache manager, which then leaves anchor advancement to the model at every
batch size. The runtime still owns all slot allocation, acceptance updates,
KV lengths, request scheduling and graph replay.

The batch-1 launcher uses three drafts for multi-token generation. It
starts a second ordinary decoder for teacher forcing and one-token
responses because the speculative sampler cannot return context logits.
All 32 development items passed with mean KL 0.005527 and non-tie top-1
agreement 99.9088%. Greedy output matched exactly on 14 prompts, first
diverged at a tie on 16 and passed the fragile check on 2.

Batch 1 through `engine.sh`, measured with the same chat prompts and timer
as the initial run. The one-token TTFT requests use the ordinary decoder.

| ISL | OSL | TTFT (ms) | TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 76.58 | 1.611 | 620.9 |
| 1024 | 1024 | 76.42 | 1.433 | 697.8 |
| 4096 | 1024 | 65.39 | 1.480 | 675.7 |

The native server started with three drafts and matched the batch-1 launcher
exactly for 32 single-client requests. In bursts of 32 and 128 concurrent
requests, there were 1 and 0 respective failures against the launcher's
teacher-forced margins. The former was a first divergence at a margin of
exactly 0.5 nats. A one-draft server produced the same borderline divergence
at concurrency 1 and passed the bursts at both higher concurrencies.

## Exact FP8 scales and fused decode kernels

Every shard in a fused attention input projection shares the checkpoint
activation scale. Their weight scales can differ by up to 2.24×. The fused
matrix now contains the original E4M3 bytes and uses each shard's FP32
weight scale on its output channels. PyTorch's channel-scaled GEMM applies
those scales before rounding to BF16. Attention output projections retain
their original scalar scales.

The narrow BF16 DeltaNet gate projection comes from `bf16_ba.py`. Single
requests fuse verification convolution with recurrence, following
`spec_kernels.py` and writing the runtime's existing convolution windows.
Fresh context convolution and tiled head normalization come from
`prefill_stencil.py`. Residual RMSNorm and the fused MoE epilogue also write
the next layer's FP8 operand. Short decode batches use more splits of the
same paged KV cache to distribute attention work across the GPU.

Two faster candidates retained the standalone engine's FP8 requantization.
They reached 754–759 and 748 decode tokens/s at ISL 1024 and OSL 1024. Each
had one greedy failure on the development oracle. The version with the
original checkpoint scales passed all 32 items with KL 0.005972 and
non-tie top-1 agreement of 99.7264%. It matched 14 prompts exactly,
diverged first at a near-tie on 17 and passed one fragile check.

These batch-1 runs used the original checkpoint scales through `engine.sh`.

| ISL | OSL | TTFT (ms) | TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 76.33 | 1.458 | 685.7 |
| 1024 | 1024 | 75.70 | 1.625 | 615.4 |
| 4096 | 1024 | 65.55 | 1.564 | 639.6 |

The server with three drafts matched the launcher on all 32 single-client
requests. All 32 requests in the concurrency-32 burst and all 128 requests
in the concurrency-128 burst passed the launcher's 0.5-nat tolerance.

The first closed-loop run used two request rounds at each load, with
at least four requests for concurrency 1.

| ISL | OSL | Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 623.3 | 77.27 | 1.530 |
| 1024 | 1024 | 32 | 4422.7 | 899.02 | 6.090 |
| 1024 | 1024 | 128 | 7805.6 | 484.38 | 14.504 |

A batch-dependent draft schedule used three drafts through eight active
requests, two through 64 and one through 128. It passed the single-client
and concurrency-128 comparisons, with one concurrency-32 divergence at a
0.875-nat margin. The serving default remained three drafts for this revision.

| ISL | OSL | Concurrency | Scheduled-draft output tokens/s | Median TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 617.9 | 75.99 |
| 1024 | 1024 | 32 | 5676.1 | 246.56 |
| 1024 | 1024 | 128 | 8922.4 | 531.42 |

## Prefill CUDA graphs

The runtime captures the whole decoder body for a fresh single prompt
whose length matches a configured bucket. The target stages embeddings,
positions, cache slots and page tables at stable addresses before capture
and replay. Mixed steps, continuations, padded lengths and teacher-forced
score requests run eagerly. The runtime change is an opt-in policy that
checks the capture and replay shapes. The launchers capture lengths 128,
256, 512, 1024, 2048 and 4096 for generation.

Initial captures borrowed transient metadata during warmup and faulted on
replay. Staging in the runtime's breakable graph context fixed those
captures. Compute Sanitizer reported zero memory errors on the single
engine with all six buckets. The full launcher matched eager execution on
70 requests interleaving scoring and generation, with exact and unbucketed
prompt lengths. All 32 development items passed at the same KL and top-1
agreement as the exact-scale revision.

Batch 1 through `engine.sh`, with the median of two measurements in each
cell. One-token requests use the speculative generation engine so they take
the same prefill graph as longer generation requests.

| ISL | OSL | TTFT (ms) | TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 11.20 | 1.462 | 684.1 |
| 1024 | 1024 | 19.60 | 1.628 | 614.4 |
| 4096 | 1024 | 48.20 | 1.563 | 639.7 |

`serve.sh` loaded the prefill and generation graphs and passed the
batch-1 comparisons at concurrencies 1, 32 and 128.

The closed-loop sweep used the same inputs and request counts as the
previous three-draft sweep.

| ISL | OSL | Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 643.0 | 24.66 | 1.531 |
| 1024 | 1024 | 32 | 5488.8 | 261.46 | 5.319 |
| 1024 | 1024 | 128 | 8919.2 | 319.14 | 13.241 |

## Replay layout and attention experiments

A decode profile at batch 128, ISL 1024 and OSL 256, used one MTP draft.
The recurrent verifier accounted for 30.8% of total GPU kernel time.
Its median launch was about 320 microseconds. The two guarded shared
projection epilogues and the paged attention kernel were the next
target-owned bottlenecks.

An FP32 replay experiment kept the normalized keys, update vectors and
per-token decay factors at the precision used in the recurrence. It
passed all 32 development items through `engine.sh`, with mean KL
0.005972 and confident top-1 agreement 99.726%. The fixed and scheduled
serving runs both had one concurrency-32 divergence at a 0.5-nat margin
from the batch-1 launcher. The default continued to use the verified
replay implementation while the batch-dependent attention arithmetic
was evaluated.

| ISL | OSL | Concurrency | FP32 replay, three drafts (tokens/s) | FP32 replay, scheduled drafts (tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 662.0 | 662.9 |
| 1024 | 1024 | 32 | 5441.0 | 5477.6 |
| 1024 | 1024 | 128 | 9144.6 | 9372.9 |

## Tiled rotary embedding and short FP8 projections

The rotary kernel from aegis-engines `prefill_rope.py` now processes tiled
heads on long steps. It consumes TensorRT-LLM's per-request K and V page
offsets, including strided layers. An isolated 4,096-row comparison on
the original and tiled kernels took 92.18 and 45.38 microseconds, with
identical FP8 query, key and value bytes and identical BF16 gates.

For short fused projections, the Tensor Core GEMM returns FP32
accumulators and a Triton epilogue applies the input and per-channel
checkpoint scales before rounding to BF16. The two weight shards of each
fused gate/up pair are checked for equal ModelOpt global scales during
loading. The complete 1,319-question GSM8K integration test passed.

Two external runs per batch-1 workload produced these medians.

| ISL | OSL | TTFT (ms) | TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 11.33 | 1.437 | 695.9 |
| 1024 | 1024 | 19.64 | 1.600 | 624.9 |
| 4096 | 1024 | 48.29 | 1.536 | 650.8 |

The launcher passed all 32 development items, with mean KL 0.005972
and confident top-1 agreement 99.726%.
The native server started and passed the batch-1 comparisons at
concurrencies 1, 32 and 128. The launcher's protocol checks covered error
recovery, requested log-probability order, and EOS inclusion.

The first tiled rotary implementation specialized on the number of
input rows. Continuous batching compiled 49 rotary variants after
server startup during the first sweep. The compile stalls affected
throughput and late-arriving prompts.

| ISL | OSL | Concurrency | Output tokens/s | Median TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 655.7 | 25.41 |
| 1024 | 1024 | 32 | 4301.2 | 1035.47 |
| 1024 | 1024 | 128 | 6648.9 | 864.76 |

## Tiled batched DeltaNet verification

The Gluon verifier adapts the tiled state update from aegis-engines
`verify_gdn.py`. It computes Q/K normalization and the decay gates
once per token and head. State tiles then process consecutive value
channels and use the runtime's original recurrent and replay slot
strides. The FP32 anchors, BF16 replay keys and deltas, and cumulative
FP32 decay logs use the existing cache layout.

The isolated verifier, including its metadata preparation, took 55.3
microseconds at batch 32 and width three versus 95.4 before tiling.
At batch 128 and width two it took 196.5 microseconds versus 269.5 on
strided pool views. A tensor-core experiment that projected the replay
state into the query and key subspace took 304.1 microseconds at that
same wide shape. The shared-expert rounding check spreads the gate/up
recomputations across 32 workers. Tests on three checkpoint layers
produced identical BF16 values with the previous and parallel checks.

The continuation convolution uses the tiled prefill stencil with a
separate input view of the previously committed frames. The rotary
launch takes the row and block counts at runtime so mixed batches
can reuse the compiled kernel.

The first four-lane reduction improved serving throughput at concurrency
32 to 6,379 output tokens/s (ISL 1024, OSL 1024). It had one distinct
concurrency-128 divergence at a 0.5-nat margin from the batch-1 launcher.
The full-warp contraction preserves the original reduction order. On
the strided cache fixture, verification fell from 96.0 to 76.4
microseconds at batch 32 and width three, and from 269.1 to 226.4
microseconds at batch 128 and width two. Both outputs were bitwise
identical to the original verifier. The wide operation tests cover
active history, anchor rollover, and fresh slots.

The full-warp version passed all 32 development items with mean KL
0.005972 and confident top-1 agreement 99.726%. The native HTTP
comparisons accepted all 32, 32, and 128 requests at concurrency 1, 32,
and 128. Of these, 32, 12, and 72 matched the batch-1 continuation
exactly in that run. The other requests diverged at eligible ties.

| ISL | OSL | Batch-1 TTFT (ms) | Batch-1 TPOT (ms) | Batch-1 decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 11.18 | 1.434 | 697.4 |
| 1024 | 1024 | 20.12 | 1.594 | 627.5 |
| 4096 | 1024 | 47.95 | 1.532 | 652.7 |

| ISL | OSL | Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 655.0 | 26.06 | 1.501 |
| 1024 | 1024 | 32 | 6072.7 | 199.19 | 4.861 |
| 1024 | 1024 | 128 | 9694.2 | 301.15 | 12.022 |

## Native FP8 context attention and four-lane verification

Four value channels share each warp in the wide DeltaNet verifier. The
contraction explicitly sums the original groups in the original order.
The outputs remained bitwise identical in the strided cache fixture.
Verification fell to 57.6 microseconds at batch 32 and width three,
205.3 microseconds at batch 128 and width two, and 258.7 microseconds
at batch 128 and width four.

Full-attention layers use FlashInfer's native FP8 context kernel with
32-token pages for context steps of at least 128 tokens. It consumes the
runtime's separate K and V role indices, including cached prefixes. The
context and full-KV offsets are prepared once per decoder step and
shared across the ten layers. The kernel uses a reusable 128 MiB
workspace. The local Triton attention path handles smaller contexts and
the other page geometry. At ISL 4096, one FP8 prefill attention call
fell from 398.5 to 166.4 microseconds. At batch 32 with 256 new tokens
and 1024 total tokens per request, the same comparison fell from 505.0
to 128.4 microseconds. The isolated outputs agreed within 0.015%
relative L2 across the measured shapes.

All 52 target unit and routing tests and all 32 development oracle
items passed. Mean KL was 0.005972 and confident top-1 agreement was
99.726%. The native server started and accepted all short development
prompt comparisons to the batch-1 launcher at concurrency 1, 32 and
128.

| ISL | OSL | Batch-1 median TTFT (ms) | Batch-1 median TPOT (ms) | Decode tokens/s from median TPOT |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 11.29 | 1.496 | 668.3 |
| 1024 | 1024 | 19.43 | 1.632 | 612.6 |
| 4096 | 1024 | 44.20 | 1.525 | 655.6 |

The two ISL 1024 decode trials had TPOT of 1.378 and 1.887 ms.

| ISL | OSL | Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 666.1 | 25.97 | 1.471 |
| 1024 | 1024 | 32 | 6359.9 | 221.70 | 4.663 |
| 1024 | 1024 | 128 | 10491.3 | 281.77 | 11.107 |

## Draft scheduling trials

The native server tested three drafts at every batch size, a schedule
with three drafts through batch 32 and one above it, and a schedule
with three through 8, two through 32, and one above 32. Both schedules
passed all short HTTP oracle comparisons. They reduced concurrency-128
p99 TTFT from 2.38 seconds to 1.47 and 1.52 seconds, respectively.
The fixed three-draft setting had the highest aggregate throughput in
this sweep.

| ISL | OSL | Concurrency | Fixed three drafts (output tokens/s) | Scheduled three or one (output tokens/s) | Scheduled three, two or one (output tokens/s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 666.1 | 669.7 | 668.7 |
| 1024 | 1024 | 32 | 6359.9 | 6352.5 | 6040.3 |
| 1024 | 1024 | 128 | 10491.3 | 10457.7 | 10206.4 |

An Nsight Systems trace of the earlier fixed-depth verifier showed a
median batch-1 graph span of 4.13 ms, with 63 microseconds between
graph replays. At batch 128 and width two the median graph span was
20.76 ms, with a 59-microsecond gap. This directed subsequent work
toward in-graph kernel costs.

A short BF16 router GEMV with one FP32 reduction reached 730 output
tokens/s at batch 1, ISL 1024, OSL 1024. It changed two development
continuations at reference margins of 0.75 and 0.5 nats. A trace of the
existing four-row router identified 16 split-K partials stored in BF16
before the final reduction. The FP32-only version was excluded.

## Four-row BF16 router trial and single-token launch

The narrow router trial rounded each of the sixteen 128-product partial
sums to BF16 before reducing in FP32. On the checkpoint router probe it
matched the cuBLAS output for all four rows and their 257 channels.
The isolated launch took 2.55 microseconds versus 9.50 for the cuBLAS
multiply, before the latter's separate reduction. The trial used it
on four-row, single-request verification steps. The separate change to
generation-only input staging passes rows in the order supplied by the
runtime and avoids redundant gathers.

The launcher sends one-output requests to the already loaded ordinary
decoder. Both decoders capture the same eligible prefill buckets.
This skips MTP proposals when the request has already reached its
token limit. The first token agreed exactly with a 35-output request
on 35 chats covering ISL 128, 173, 256, 640, 1024, 2048, and 4096.
Score requests still return the ordinary decoder's full-context logits.
The launcher protocol checks passed, including EOS inclusion.

The oracle passed all 32 items with mean KL 0.005972 and confident
top-1 agreement 99.726%. The test suite passed all 54 target unit and
routing tests. The table reports two-run medians for each chat seed.

| ISL | OSL | Chat seed | Batch-1 TTFT (ms) | Batch-1 TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 100 | 10.33 | 1.383 | 723.0 |
| 1024 | 1024 | 100 | 18.69 | 1.244 | 803.6 |
| 4096 | 1024 | 100 | 43.15 | 1.375 | 727.3 |
| 128 | 1024 | 133 | 9.98 | 1.387 | 720.8 |
| 1024 | 1024 | 133 | 18.12 | 1.417 | 705.9 |
| 4096 | 1024 | 133 | 42.78 | 1.437 | 696.0 |

The rounded-router trial changed one distinct HTTP continuation at
concurrency 32 and 128. The batch-1 teacher-forced gap at the first
difference rounded to 0.5 nats, so this router variant was removed.
The table above records the trial's batch-1 measurements.

## Native decode attention trials

The native FlashInfer decode kernel uses the same FP8 HND paged cache
as context attention. Its independent K and V tables refer to the
runtime's physical role pages. The FP8-query and BF16-query trials
passed the batch-1 comparison at concurrency 32. At concurrency 128
the FP8 trial had one distinct strict-margin disagreement, and the
BF16-query trial had two. These trials used native decode starting
at batch 16.

| ISL | OSL | Concurrency | Native FP8 queries (output tokens/s) | Native BF16 queries on FP8 KV (output tokens/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 669.6 | 669.9 |
| 1024 | 1024 | 32 | 6693.8 | 6672.7 |
| 1024 | 1024 | 128 | 10826.6 | 10949.0 |

The scoped FP8 path uses native decode for 16 to 32 generation rows
per step, with a context-length array shared across the ten layers.
It passed the full 32-item engine oracle and the native HTTP checks
at concurrency 1, 32 and 128. A broader sweep also passed at 4 and 96.
It found strict-margin differences from the batch-1 teacher scores
at concurrency 2, 8, 16, 24, 48 and 64. Concurrency 2 and 8 use
the local Triton attention path throughout.

| ISL | OSL | Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 670.9 | 24.89 | 1.463 |
| 1024 | 1024 | 32 | 6402.6 | 208.46 | 4.601 |
| 1024 | 1024 | 128 | 10618.6 | 288.41 | 10.950 |

| ISL | OSL | Batch-1 median TTFT (ms) | Batch-1 median TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1024 | 10.92 | 1.495 | 668.9 |
| 1024 | 1024 | 18.41 | 1.379 | 725.0 |
| 4096 | 1024 | 44.35 | 1.521 | 657.5 |

## Additional expert-kernel measurements

FlashInfer Cake's SM100 W4A16 kernels consumed the checkpoint's packed
E2M1 values and block scales directly. On the layer-13 fixture, the
four-row projection took 24.2 microseconds versus 23.3 for the existing
CuTe route. At 512 rows the fastest Cake cell took 153.6 microseconds
versus 151.7, and at 4096 rows it took 399.0 versus 349.6. Its unmodified
epilogue also rounded the activation differently. Using FlashInfer's
native route sort with the existing CuTe GEMMs preserved the expert
rows bitwise and changed the 512-row timing from 145.7 to 143.8
microseconds. The existing expert kernels and sorting remained the
default.

## Small mixed scheduler steps

In a two-request trace, BF16 router scores first differed in 100 of
1028 entries when cuBLAS saw eight rows instead of two jobs of four
rows. Independently evaluating each four-row request restored the
same first-layer scores and routing decisions seen at batch 1. The
short DeltaNet and shared-expert paths also use the single-request
tile geometry. Small mixed steps process context and generation
rows in separate model passes, retaining the scheduler's physical
K/V pages and DeltaNet slots. The two-request trace matched 64 output
tokens exactly with CUDA graphs enabled.

All 32 engine oracle cases passed again (mean KL 0.005972, confident
top-1 agreement 99.726%). HTTP passed against both the external oracle
and batch-1 scores at concurrency 1, 2, 4, 32, 96, and 128. At concurrency
2, 29 of the 32 continuations matched batch 1 token for token. The
external comparison still found strict differences at 8, 16, 24,
and 48. The batch-1 comparison also found a 0.75-nat difference at 64.

| ISL | OSL | Concurrency | Output tokens/s | Median TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 667.9 | 25.84 |
| 1024 | 1024 | 32 | 6375.9 | 223.71 |
| 1024 | 1024 | 128 | 10520.2 | 287.70 |

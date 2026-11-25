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

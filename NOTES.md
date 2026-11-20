Fri Nov 20 12:01:15 UTC 2026

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Qwen3.6-35B-A3B-FP8 on GB200

## Modeling V1 baseline

The unmodified v1.4.0rc4 implementation loaded the official checkpoint with
`modeling_v2="off"` and `disable_mm_encoder=True`. Both the paged attention KV
cache and the resolved recurrent-state cache used BF16. The development
oracle passed all 32 items, with mean top-k KL 0.006901 and non-tie top-1
agreement 99.679% over 2,181 positions. The first greedy divergence for 24
prompts was within the near-tie allowance.

Batch-1 timings are medians of two independent quoted-passage prompts at
each length. The external JSONL client timed OSL 1 for TTFT and differenced
OSL 1024 and OSL 32 for decode. CUDA graphs, the overlap scheduler and
autotuning were enabled. Prefix reuse was disabled. The passage corpus was
the checkout's coding guidelines, developer overview, parallel strategies,
KV cache, and quantization documentation. The benchmark's default corpus
paths were absent in the container.

### Modeling V1 batch 1, ISL 128 / 1024 / 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | Decode (tokens/s) |
| ---: | ---: | ---: | ---: |
| 128 | 1024 | 63.30 | 322.74 |
| 1024 | 1024 | 63.15 | 322.19 |
| 4096 | 1024 | 67.24 | 318.90 |

The V1 serving run used chunked prefill, a max batch size of 128, a token
budget of 8192, graph padding, and a BF16 paged KV cache sized from 80% of
free GPU memory. The benchmark used one request per concurrent client (at
least two), with a 16-output-token warmup. At ISL 1024 the first run
completed concurrency 1–64. The benchmark client disconnected during the
concurrency-128 warmup. A fresh client completed that point and the
three ISL 4096 measurements.

### Modeling V1 serving, ISL 1024 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 306.72 | 70.59 | 3.19 |
| 2 | 561.09 | 126.72 | 3.44 |
| 4 | 958.91 | 147.95 | 4.03 |
| 8 | 1614.40 | 153.09 | 4.81 |
| 16 | 2598.64 | 231.60 | 5.93 |
| 32 | 4527.05 | 352.13 | 6.72 |
| 64 | 6922.77 | 514.50 | 8.73 |
| 128 | 9926.17 | 819.13 | 12.05 |

### Modeling V1 serving, ISL 4096 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 305.00 | 73.21 | 3.21 |
| 8 | 1514.50 | 314.53 | 4.97 |
| 32 | 3839.28 | 841.23 | 7.48 |

## Modeling V2 bring-up

The target owns the text decoder, its checkpoint mapping, and the fused
Gated DeltaNet and attention/expert epilogues. It uses TensorRT-LLM's native
block-scale FP8 expert runner, catalog paged GQA and RoPE, and FlashInfer's
recurrent scan and decode kernels. The runtime owns the paged BF16 attention
KV cache, the BF16 convolution and recurrent state, scheduling, and CUDA
graph capture. The new Triton gates are candidates for target-local native
code under `csrc/` in a full build. No shared inference runtime code changed.

The first V2 build used the checkpoint's original FP8 bytes and FP32 block
scales in every projection. It reached mean KL 0.007979 and non-tie top-1
agreement 99.404%. One fragile development position did not match the
teacher-forced top-1 token selected by the score response. Three tokens had
equal teacher-forced scores at that position. This build did not pass the
commit gate.

### Initial Modeling V2 batch 1, ISL 128 / 1024 / 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | Decode (tokens/s) |
| ---: | ---: | ---: | ---: |
| 128 | 1024 | 52.61 | 210.17 |
| 1024 | 1024 | 51.08 | 212.76 |
| 4096 | 1024 | 72.36 | 211.22 |

Changing dense FP8 scales to the packed UE8M0 format enabled the low-M
GEMM and DeepGEMM. The batch-1 version reached 267–268 decode tokens/s but
had a greedy divergence beyond the oracle tolerance. Reconstructing the
Gated DeltaNet input projection in BF16 registers from the original FP8
tiles lowered mean KL to 0.006595 and ran at 242–245 decode tokens/s.
That version also had one greedy miss. Keeping FP32 recurrent state with
the same projections increased the number of greedy divergences in this
test. None of these variants passed the commit gate.

Gated DeltaNet prefill can prepare contiguous Q/K/V and linear decay and
beta in a single Triton launch. The operation passed the short and
multi-tile tests. With the original FP8 weights it reduced TTFT by
6–11 ms on these prompts while leaving decode speed largely unchanged.
This prefill variant missed one greedy continuation outside the allowance.

### Fused Gated DeltaNet prefill trial, ISL 128 / 1024 / 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | Decode (tokens/s) |
| ---: | ---: | ---: | ---: |
| 128 | 1024 | 41.23 | 210.58 |
| 1024 | 1024 | 44.86 | 211.10 |
| 4096 | 1024 | 66.15 | 209.53 |

The first passing target uses the original E4M3 weights and block scales
and emits FP32 LM-head logits. FP32 output avoids rounding distinct logits
to the same value at the vocabulary projection. Mean KL was 0.007827 and
non-tie top-1 agreement was 99.587% over 2,181 positions. No greedy
continuation diverged outside the development oracle allowance.

`serve.sh` loaded the target and served all 32 token-ID completion requests
with exact batch-1 agreement at concurrency 1. Batched runs at concurrency
8, 32 and 128 produced some different tokens. Their first-difference margins
need to be measured on the engine's decode prefixes, including prefixes
that already depart from the external reference's continuation.

### First passing Modeling V2 batch 1, ISL 128 / 1024 / 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | Decode (tokens/s) |
| ---: | ---: | ---: | ---: |
| 128 | 1024 | 50.58 | 211.50 |
| 1024 | 1024 | 50.57 | 211.69 |
| 4096 | 1024 | 71.07 | 210.02 |

The first V2 serving benchmark used the same scheduler, KV budget and graph sizes as the baseline.

### First Modeling V2 serving, ISL 1024 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 208.32 | 58.51 | 4.75 |
| 2 | 375.38 | 107.96 | 5.23 |
| 4 | 674.83 | 111.35 | 5.82 |
| 8 | 1196.53 | 161.16 | 6.53 |
| 16 | 2109.76 | 253.23 | 7.34 |
| 32 | 3610.97 | 373.61 | 8.49 |
| 64 | 5653.24 | 603.12 | 10.71 |
| 128 | 8247.26 | 1048.24 | 14.44 |

### First Modeling V2 serving, ISL 4096 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 208.33 | 78.45 | 4.73 |
| 8 | 1129.14 | 401.09 | 6.69 |
| 32 | 3057.98 | 1101.56 | 9.35 |

## Speculative decode and batch-1 profile

The Nsight Systems capture ran a batch-1 request with ISL 14 and OSL 48
after warmup. The CUDA graph node report attributed 26.2% of GPU kernel
time to the dense FP8 GEMM. The two routed expert GEMMs accounted for
15.7%. Activation quantization and the scale-layout copy together
accounted for 10.1%. These fractions come from
`nsys stats --report cuda_gpu_kern_sum` and include the request's prefill.

The MTP draft owns a single full-attention layer, the shared embedding
and vocabulary projection, and the checkpoint's embedding and hidden
state concatenation projection. The existing speculative worker accepts
tokens against the trunk's logits. During verification a target-local
Triton convolution writes each candidate's convolution history to the
runtime's staging buffer. FlashInfer writes the corresponding recurrent
state to its staging buffer. The hybrid cache manager advances both
histories to the accepted position. The convolution is another candidate
for target-local `csrc/` in a full native build.

One-token MTP passed the 32 greedy development checks in a single-process
LLM API probe against the original scorer. The JSONL runner has separate
TensorRT-LLM executors for generation and scoring because the speculative
sampler cannot return teacher-forced context logits. Both use this target
and warm up before the runner emits its ready response.

### One-token MTP through the JSONL engine, ISL 128 / 1024 / 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | Decode (tokens/s) |
| ---: | ---: | ---: | ---: |
| 128 | 1024 | 49.13 | 334.97 |
| 1024 | 1024 | 50.21 | 332.16 |
| 4096 | 1024 | 72.95 | 329.08 |

The dual-executor JSONL run passed all 32 development cases with mean KL
0.007827 and 99.587% non-tie top-1 agreement. No shared inference runtime
code was changed.

With one-token MTP, the server started and all 32 development completions matched the batch-1 engine exactly at concurrency 1. Batched outputs were checked separately at higher concurrency.

### One-token MTP serving, ISL 1024 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 325.89 | 68.59 | 3.00 |
| 2 | 571.42 | 117.28 | 3.35 |
| 4 | 1037.88 | 124.52 | 3.66 |
| 8 | 1793.82 | 159.17 | 4.20 |
| 16 | 3050.27 | 249.91 | 4.91 |
| 32 | 4803.53 | 400.89 | 6.16 |
| 64 | 7155.62 | 629.15 | 8.19 |
| 128 | 10344.75 | 1097.67 | 11.10 |

### One-token MTP serving, ISL 4096 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 316.21 | 83.84 | 3.08 |
| 8 | 1676.09 | 418.69 | 4.31 |
| 32 | 3929.26 | 1135.55 | 6.82 |

The one-token serving comparison produced 46 unique first-difference
positions across concurrency 8, 32 and 128. Replaying the batch-1
generation with its actual speculative logits showed 43 margins below
0.5 nats and three at or above 0.5. The trace reproduced the batch-1
prefixes at every measured position. It also measured 1.86 accepted
tokens per verification step on the checked development prompts.

Two-token MTP reached 378–434 tokens/s in the direct LLM API probe and had
one greedy miss at the oracle's 0.5-nat cutoff. Three-token MTP passed
through the JSONL engine with mean KL 0.007827, non-tie top-1 agreement
99.587%, and no greedy misses outside the allowance.

### Three-token MTP through the JSONL engine, ISL 128 / 1024 / 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | Decode (tokens/s) |
| ---: | ---: | ---: | ---: |
| 128 | 1024 | 52.22 | 446.36 |
| 1024 | 1024 | 53.59 | 440.80 |
| 4096 | 1024 | 75.39 | 436.57 |

Three-token MTP also served all 32 oracle requests exactly like the batch-1 engine at concurrency 1.

### Three-token MTP serving, ISL 1024 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 427.38 | 62.93 | 2.28 |
| 2 | 735.96 | 122.13 | 2.57 |
| 4 | 1304.14 | 123.75 | 2.85 |
| 8 | 2230.96 | 165.33 | 3.19 |
| 16 | 3584.57 | 247.65 | 4.06 |
| 32 | 5467.29 | 400.88 | 5.22 |
| 64 | 8111.45 | 634.98 | 6.88 |
| 128 | 11461.00 | 1170.86 | 9.46 |

### Three-token MTP serving, ISL 4096 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 413.61 | 81.79 | 2.34 |
| 8 | 2035.44 | 418.80 | 3.39 |
| 32 | 4362.37 | 1138.59 | 5.82 |

The three-token serving comparison found 50 unique first-difference
positions across the checked concurrency levels. All 50 batch-1
prefixes were reproduced in the decode trace. Forty-nine margins were
below 0.5 nats and one was 1.075 nats. The trace measured 2.98 tokens per
verification step and 65.9% draft-token acceptance on the development
requests.

A repeated-row probe returned identical BF16 outputs for the native
dense projections, MoE and residual norm across the tested batch sizes.
FlashInfer verification had small batch-dependent differences in its
BF16 recurrent checkpoints. A Triton variant rounded the Gated DeltaNet
state at each candidate token boundary. Its four-token kernel took
8.76 µs at batch 1 and 286.27 µs at batch 128, compared with 5.85 µs and
180.80 µs for FlashInfer. The Triton variant passed its operation tests
and had one greedy oracle miss, so the target kept FlashInfer verification.

## Matched Modeling V1 serving comparison

This V1 run used the same serving YAML as the three-token V2 revision, with `modeling_v2` set to `off`. Both runs used FP32 output logits and BF16 attention and recurrent caches.

### Matched Modeling V1 with MTP, ISL 1024 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 598.28 | 73.79 | 1.60 |
| 2 | 952.96 | 161.15 | 1.90 |
| 4 | 1655.04 | 165.60 | 2.20 |
| 8 | 2839.39 | 164.77 | 2.53 |
| 16 | 4300.53 | 239.39 | 3.38 |
| 32 | 6566.61 | 354.73 | 4.37 |
| 64 | 9727.25 | 550.47 | 5.79 |
| 128 | 12821.15 | 974.45 | 8.61 |

### Matched Modeling V1 with MTP, ISL 4096 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 565.76 | 84.59 | 1.69 |
| 8 | 2561.77 | 378.78 | 2.72 |
| 32 | 4810.02 | 1525.46 | 4.90 |


## Fused FP8 input preparation and draft-row selection

The new Triton quantizers produce contiguous `[block, token]` FP32 scales
directly. Fused residual addition, Gemma RMSNorm and quantization retain
the BF16 normalized activation used by unquantized projections and the
original FP8 block scales used by quantized projections. Three small
projection shapes use split-K Triton GEMMs on up to eight rows. The draft
layer writes attention KV for every input and runs its output projection
and MoE only for rows the runtime will sample. All 18 GPU operation cases
passed. The quantization, norm and small-row GEMM kernels are candidates
for target-local native code under `csrc/`.

Fused add, norm and quantization reduced the two-row GPU time from
6.48 µs to 2.63 µs and the 1024-row GPU time from 10.71 µs to 5.51 µs.
The width-4096 output projection reduced GPU time from 11.4 µs to 6.1 µs
on four rows. The JSONL engine passed the 32 development prompts at mean KL
0.006933 and 99.587% non-tie top-1 agreement. Protocol checks exercised
malformed requests, single-token scoring, EOS stopping and `ignore_eos`.

### Fused FP8 preparation through the JSONL engine, ISL 128 / 1024 / 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | Decode (tokens/s) |
| ---: | ---: | ---: | ---: |
| 128 | 1024 | 50.96 | 488.17 |
| 1024 | 1024 | 52.76 | 488.93 |
| 4096 | 1024 | 72.88 | 470.19 |

The OpenAI server started with the same revision and matched all 32 engine
completions exactly at concurrency 1.

### Fused FP8 preparation serving, ISL 1024 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 499.05 | 61.73 | 1.95 |
| 2 | 818.57 | 119.21 | 2.31 |
| 4 | 1330.58 | 124.88 | 2.84 |
| 8 | 2298.54 | 164.20 | 3.22 |
| 16 | 3688.60 | 251.03 | 3.94 |
| 32 | 5587.26 | 389.64 | 5.08 |
| 64 | 7149.93 | 621.73 | 7.93 |
| 128 | 7495.97 | 1871.15 | 14.87 |

### Fused FP8 preparation serving, ISL 4096 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 463.63 | 80.67 | 2.08 |
| 8 | 2146.62 | 408.73 | 3.22 |
| 32 | 3053.91 | 1470.96 | 8.73 |

With the same prompt hashes and YAML, this revision increased ISL 1024
throughput at concurrency 1–32 and reduced it at 64 and 128. The largest
drop was at concurrency 128, with a median TTFT of 1.87 s and median TPOT
of 14.87 ms.

The decode-log trace on this revision reproduced the engine prefixes for
all 45 unique first differences across concurrency 8, 32 and 128.
Forty-three margins were below 0.5 nats. The remaining margins were 0.692
and 0.786 nats. Batch-1 MTP accepted 61.5% of proposed drafts on the traced
prompts.

A synthetic four-token Gated DeltaNet test found bitwise equality in the
output and staged BF16 state across batches 1, 2, 8, 32 and 128 when using
FlashInfer's wide-vector kernel. Its small-batch ILP4 kernel used a
different reduction order. At batch 1, a 32-value tile took 5.80 µs
compared with 5.66 µs for the default ILP4 kernel. At batch 128, the
128-value tile remained fastest at 180.82 µs.

Forcing the wide-vector recurrence on small batches passed the operation
tests but the engine diverged at one development generation position
with a reference margin of exactly 0.5 nats. That trial measured 515,
476 and 485 decode tokens/s at ISL 128, 1024 and 4096, respectively,
with OSL 1024. The next candidate restored FlashInfer's default
recurrent dispatch.

## Gated projection epilogues and aligned FP8 GEMMs

Triton now quantizes Gated DeltaNet's normalized gated output, full
attention's sigmoid-gated output and the shared expert's SwiGLU output
directly. Each epilogue rounds to BF16 before quantization. The attention
epilogue also gathers the draft worker's selected rows. Gated DeltaNet
prefill packs Q, K, V and both gates in one launch. The native
recurrence still receives the original values. Both new Triton kernels
are candidates for target-local native code under `csrc/`.

A CuTe DSL GEMM uses the original FP8 scale blocks on aligned larger
activation batches. At 4096 rows the QKVZ projection took 124.58 µs with
CuTe DSL and 254.08 µs with the native block-scaled GEMM. All five target
projection shapes produced bitwise-identical BF16 results in the
measured synthetic comparison. At 512 rows the same projection took
19.85 µs and 37.07 µs, respectively.

All 23 GPU operation cases passed. The JSONL engine passed all 32
development prompts with mean KL 0.006933 and 99.587% non-tie top-1
agreement. Its 32 generations exactly matched the preceding revision.

### Fused epilogues through the JSONL engine, ISL 128 / 1024 / 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | Decode (tokens/s) |
| ---: | ---: | ---: | ---: |
| 128 | 1024 | 47.81 | 509.78 |
| 1024 | 1024 | 48.74 | 510.76 |
| 4096 | 1024 | 60.13 | 490.52 |

The OpenAI server started and all 32 completions matched the engine at
concurrency 1.

### Fused epilogues serving, ISL 1024 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 510.76 | 65.07 | 1.90 |
| 2 | 839.80 | 112.55 | 2.26 |
| 4 | 1356.45 | 131.13 | 2.75 |
| 8 | 2354.18 | 136.91 | 3.17 |
| 16 | 3924.47 | 203.47 | 3.69 |
| 32 | 6052.05 | 312.18 | 4.72 |
| 64 | 7642.15 | 493.94 | 7.55 |
| 128 | 7820.65 | 1805.10 | 14.16 |

### Fused epilogues serving, ISL 4096 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 477.25 | 70.84 | 2.03 |

The client disconnected during the ISL 4096 concurrency-8 warmup.
The ISL 4096 concurrency-8 and concurrency-32 measurements for this
revision remain unmeasured.

Of the 44 unique first differences traced across concurrency 8, 32 and
128, 43 had batch-1 margins below 0.5 nats. One was 0.786 nats.

## Compact recurrent cache

The generator now verifies three-token MTP drafts with FlashInfer's
BF16 recurrent checkpoint and ring of accepted updates. The runtime
hybrid cache manager owns the state, per-request ring cursors and
acceptance updates. The model gathers each step's cursor views once for
all recurrent layers. The scoring engine uses ordinary teacher-forced
prefill. Paged attention KV, convolution state and recurrent checkpoints
all remain BF16.

In a four-token verification microbenchmark at batch 128, replay took
44.47 µs with an empty history, 48.80 µs with eight history entries, and
99.80 µs when it flushed a checkpoint. The intermediate-state kernel
took 180.81 µs. At batch 1 the corresponding replay times were 6.42, 6.62,
and 11.92 µs, compared with 5.66 µs for intermediate-state verification.

The JSONL engine passed all 32 development prompts with mean KL 0.006933
and 99.587% non-tie top-1 agreement. EOS handling and malformed-request
recovery passed. The Modeling V2 CPU checks passed all nine selected
cases.

### Compact recurrent cache through the JSONL engine, ISL 128 / 1024 / 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | Decode (tokens/s) |
| ---: | ---: | ---: | ---: |
| 128 | 1024 | 50.87 | 513.66 |
| 1024 | 1024 | 48.93 | 506.30 |
| 4096 | 1024 | 60.29 | 483.89 |

The OpenAI server started with replay enabled and matched the engine's
32 completions exactly at concurrency 1.

### Compact recurrent cache serving, ISL 1024 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 517.44 | 59.47 | 1.88 |
| 2 | 856.60 | 130.81 | 2.18 |
| 4 | 1421.83 | 124.62 | 2.65 |
| 8 | 2494.14 | 135.84 | 3.02 |
| 16 | 4202.71 | 210.72 | 3.40 |
| 32 | 6798.87 | 308.72 | 4.23 |
| 64 | 9948.23 | 492.78 | 5.54 |
| 128 | 6941.22 | 1759.03 | 16.20 |

### Compact recurrent cache serving, ISL 4096 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 483.63 | 68.18 | 2.00 |
| 8 | 2233.85 | 322.81 | 3.18 |
| 32 | 1793.44 | 4150.25 | 13.66 |

At ISL 1024 and OSL 1024, concurrency 128 reached 12.03 seconds p99
TTFT with replay enabled. Earlier fused Triton revisions also showed
several-second tails at that point. The projection and packing kernels
specialized on each row count, and the strided recurrent replay kernel
specialized on decode batch size during mixed batches.

The replay decode trace found 36 unique first differences across
concurrency 8, 32 and 128. Thirty-five had engine margins below 0.5 nats.
One margin was 0.585 nats. The traced development requests accepted 64.0%
of drafts, or 2.92 output tokens per verification step.

## Matched Modeling V1 serving with compact recurrent replay

V1 repeated the same serving YAML with `modeling_v2: off`. Both
servers used three draft tokens, FP32 logits, and the FlashInfer BF16
replay ring. The client settings and prompt hashes matched.

### Matched Modeling V1 with replay, ISL 1024 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 582.11 | 77.43 | 1.64 |
| 2 | 947.49 | 180.15 | 1.85 |
| 4 | 1577.50 | 179.30 | 2.32 |
| 8 | 2594.00 | 242.77 | 2.76 |
| 16 | 4401.54 | 256.62 | 3.23 |
| 32 | 6958.82 | 401.94 | 3.97 |
| 64 | 10712.72 | 615.43 | 5.13 |
| 128 | 8623.36 | 1426.06 | 13.04 |

### Matched Modeling V1 with replay, ISL 4096 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 547.70 | 79.28 | 1.75 |
| 8 | 2536.79 | 413.99 | 2.66 |
| 32 | 2520.33 | 3516.01 | 9.11 |

At concurrency 128 and ISL 1024, V1 reached 8,623 output tokens/s with
replay and 12,821 with intermediate-state staging in the earlier run.

## Variable batch shapes and shared-expert overlap

Projection quantization, gated epilogues and Gated DeltaNet input packing
take the row count at runtime. Mixed prefill and decode reuses recurrent
replay shapes prepared for CUDA graphs. Padding rows carry no state slot.
The shared expert runs on an auxiliary CUDA stream for up to 128 packed
rows. Larger batches run both expert paths on the main stream. The JSONL
generator uses TensorRT-LLM's in-process worker, and the scorer retains
its separate worker.

Using the first layer's checkpoint weights, the two expert streams were
bitwise identical to the sequential path. At four rows they took 60.92 µs
compared with 76.86 µs sequentially. At 128 rows they took 206.33 µs
compared with 230.45 µs. At 512 rows the sequential path was faster.

All 23 GPU operation cases and the 32-prompt oracle passed. Mean KL was
0.006933 and non-tie top-1 agreement was 99.587%. The JSONL protocol checks
passed.

### Variable batch shapes through the JSONL engine, ISL 128 / 1024 / 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | Decode (tokens/s) |
| ---: | ---: | ---: | ---: |
| 128 | 1024 | 55.29 | 583.19 |
| 1024 | 1024 | 53.90 | 575.33 |
| 4096 | 1024 | 59.71 | 549.24 |

The OpenAI server started and matched all 32 engine generations at
concurrency 1.

### Variable batch shapes serving, ISL 1024 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 590.45 | 58.19 | 1.64 |
| 2 | 970.38 | 122.99 | 1.91 |
| 4 | 1603.97 | 122.44 | 2.33 |
| 8 | 2740.02 | 145.34 | 2.73 |
| 16 | 4588.78 | 211.40 | 3.08 |
| 32 | 7339.69 | 306.44 | 3.91 |
| 64 | 10450.59 | 487.86 | 5.31 |
| 128 | 15494.86 | 935.76 | 7.02 |

### Variable batch shapes serving, ISL 4096 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 550.29 | 69.59 | 1.75 |
| 8 | 2494.16 | 323.01 | 2.79 |
| 32 | 5736.99 | 879.82 | 4.43 |

For ISL 1024 and OSL 1024 at concurrency 128, p99 TTFT fell from
12.03 s to 1.65 s. The throughput rose from 6,941 to 15,495 output
tokens/s. One of 523 Python stack samples taken during the latter run
included the Triton compiler.

### Sustained serving with variable batch shapes, ISL 1024 and OSL 1024

| Concurrency | Requests | Output tokens/s | Median TTFT (ms) | p99 TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 512 | 13870.94 | 215.90 | 1507.08 | 8.78 |

The sustained run used the benchmark client's default four requests per
concurrent client.

An isolated draft-head probe compared the original BF16 head with
MXFP8 and tensor-scaled FP8 projections. At one row the BF16 projection
and argmax took 161.80 µs, the tensor-scaled FP8 path took 104.90 µs, and
the faster of the two MXFP8 paths took 201.79 µs. The tensor-scaled FP8
head changed some proposals at larger row counts. The model's verified
target logits use the original BF16 weights and FP32 output.

The concurrency sweep checked every listed level from 1 through 128.
The replay trace reproduced the engine's token prefixes at all 52 unique
first differences and measured the margins from decode logits. Fifty-one
margins were below 0.5 nats. One position,
observed at concurrency 64 and 128, had a margin of 0.585 nats. The
trace accepted 65.2% of drafts and emitted 2.96 tokens per verification
step across the traced development requests.

FP32 expert routing was also tested through a direct LLM API probe to
reduce sensitivity to BF16 rounding at the expert-selection boundary.
It diverged at two oracle positions whose reference margins were exactly
0.5 nats. The committed target uses BF16 router logits.

## Native small-row projection trial

This trial used the existing TensorRT-LLM FP8 GEMM for small projections.
It passed the same 32-prompt JSONL oracle, with mean KL 0.006933 and
99.587% non-tie top-1 agreement. All 20 operation cases for this
configuration passed.

### Native small-row projection trial through the JSONL engine, ISL 128 / 1024 / 4096 and OSL 1024

| ISL | OSL | TTFT (ms) | Decode (tokens/s) |
| ---: | ---: | ---: | ---: |
| 128 | 1024 | 53.31 | 555.03 |
| 1024 | 1024 | 50.24 | 542.48 |
| 4096 | 1024 | 60.33 | 538.82 |

The server started and matched all 32 JSONL engine generations at concurrency 1.

### Native small-row projection trial serving, ISL 1024 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 548.86 | 56.19 | 1.77 |
| 2 | 906.87 | 117.62 | 2.08 |
| 4 | 1537.40 | 125.85 | 2.40 |
| 8 | 2757.81 | 137.40 | 2.73 |
| 16 | 4601.32 | 203.76 | 3.09 |
| 32 | 7290.28 | 305.00 | 3.89 |
| 64 | 9527.69 | 490.30 | 5.87 |
| 128 | 15537.69 | 940.69 | 6.94 |

### Native small-row projection trial serving, ISL 4096 and OSL 1024

| Concurrency | Output tokens/s | Median TTFT (ms) | Median TPOT (ms) |
| ---: | ---: | ---: | ---: |
| 1 | 508.55 | 72.81 | 1.90 |
| 8 | 1959.45 | 358.78 | 3.66 |
| 32 | 5717.23 | 887.70 | 4.49 |

The native-only trial had three first differences at or above 0.5 nats
across the tested concurrency levels. The selected deployment uses the
small-row Triton projection. It had higher batch-1 decode throughput at
all three input lengths and one serving difference outside the allowance.

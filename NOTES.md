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

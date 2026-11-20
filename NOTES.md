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
concurrency-128 warmup.

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

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

The non-speculative V2 serving sweep used the benchmark's default request
counts with the same prompts as the stock server.

| ISL | OSL | Concurrency | Requests | Output tokens/s | Median TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 16 | 100.79 | 66.70 |
| 1024 | 1024 | 32 | 128 | 1835.41 | 984.97 |
| 1024 | 1024 | 128 | 512 | 4062.14 | 1130.13 |

## Checkpoint MTP

The runtime's linear MTP-Eagle worker runs the checkpoint's single draft
layer with shared embeddings and vocabulary weights. The verifier uses the
full vocabulary. The draft uses its runtime-managed paged BF16 KV cache.
The trunk and draft both use the overlap scheduler and decode CUDA graphs.

The tentative convolution windows and recurrent replay kernels adapt
`verify_conv_rolling.py` and `verify_replay.py` from the reference. The
trunk keeps a durable FP32 state checkpoint, then records BF16 key, value,
and gate inputs in the hybrid manager's two history banks. The kernel
replays the accepted prefix in FP32. The cache manager promotes the
accepted convolution window, updates each request's history length, and
rotates the bank when the kernel commits a checkpoint. The shared runtime
change is the opt-in `RuntimeHooks.gdn_fused_verify` flag, which tells the
manager who owns that checkpoint update.

The JSON-line runner uses a second non-speculative LLM instance for scoring
because the speculative sampler rejects context-logit requests. Both
instances are ready before the engine starts reading requests. The two
workers used 65,162 MiB of GPU memory after warmup.

The three-token draft passed all 32 development items through the JSON-line
runner with KL 0.003288 and non-tie top-1 agreement of 99.909%. The 10 exact,
21 near-tie, and one fragile continuation match the non-speculative result
counts. Both EOS checks returned token 248044 and stopped at the expected
position. The runner also rejected an out-of-vocabulary ID and continued
processing subsequent requests. Seven new operation cases cover tentative
convolution windows, replay rollover, the single-token path, and slot reset.

| Batch | ISL | OSL | TTFT (ms) | TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1024 | 50.21 | 4.638 | 215.61 |
| 1 | 1024 | 1024 | 71.63 | 4.910 | 203.67 |
| 1 | 4096 | 1024 | 209.90 | 5.431 | 184.12 |

Serving uses three draft tokens through batch size 32 and one through
batch size 128. The HTTP token-ID check matched every batch-1 continuation
at concurrency 1, 32, and 128, including the fragile case. The server
used 81,253 MiB of GPU memory after graph capture with 285,000 cache tokens.

The first MTP streaming sweep completed concurrency 1 and 32. The
client disconnected in its concurrency-128 case. The same server had
already completed the concurrency-128 token-ID check.

| ISL | OSL | Concurrency | Requests | Output tokens/s | Median TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 16 | 202.78 | 71.76 |
| 1024 | 1024 | 32 | 128 | 2093.04 | 204.25 |

## Fused activation and projection kernels

The SwiGLU, GDN RMSNorm and dense attention gate now directly write E4M3
activation values and FP32 scales. The residual RMSNorm writes both the
BF16 residual and its quantized projection input. It also writes the
normalized BF16 activations for the GDN gate projection. The MTP input
kernel gathers the token embedding and applies the two RMSNorms before
the BF16 FC. These kernels come from reference `activation_quant.py`
and `norm_quant.cu/.py`. Fresh short contexts keep the existing arithmetic
path. Decode and larger batches use the fused producers.

The target uses the reference `fp8_tiles.cu/.py` CUTLASS kernels for
shape ranges where the GB200 microbenchmark found a faster tile.
For example, a 32-row fused QKVZ projection took 23.61 us with the narrow
tile and 31.28 us with the original GEMM. The 128-row FFN down projection
took 56.42 us and 58.94 us, respectively. I measured the reference's
separate TMA-packed FFN weights as well. Their small-row gain came with
about 10.8 GiB of duplicate E4M3 weights. This build keeps one FP8 copy.
The CUDA norm and CUTLASS implementations would move to the target's
`csrc/` in a source-built contribution.

The draft proposes from 65,536 vocabulary rows, including the checkpoint's
added tokenizer tokens, as in reference `speculative.py`. Every accepted
token still agrees with the verifier's full-vocabulary calculation.

All 59 target operation tests passed on GB200. The default engine again
passed all 32 development items, with KL 0.003288, non-tie top-1 agreement
99.909%, and the same 10 exact, 21 near-tie, and one fragile results.

| Batch | ISL | OSL | TTFT (ms) | TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1024 | 49.99 | 4.227 | 236.56 |
| 1 | 1024 | 1024 | 67.86 | 4.575 | 218.59 |
| 1 | 4096 | 1024 | 207.67 | 5.199 | 192.36 |

The OpenAI token-ID check again matched all batch-1 continuations
at concurrency 1, 32, and 128. This server used 81,991 MiB after graph
capture with 285,000 cache tokens.

Independent benchmark invocations use fresh HTTP connections for each
concurrency. This tuning sweep used one request per concurrency slot,
with a minimum of four. It completed the concurrency-128 streaming case.

| ISL | OSL | Concurrency | Requests | Output tokens/s | Median TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 4 | 216.25 | 68.57 |
| 1024 | 1024 | 32 | 32 | 2143.79 | 1084.45 |
| 1024 | 1024 | 128 | 128 | 3570.34 | 3316.68 |

## First-row recurrent checkpoints

The verifier now commits the always-accepted first row, then records only
the tentative suffix as FP32 keys, deltas and decays. The next step applies
the accepted suffix in the original FP32 recurrence order. This adapts the
reference's first-row design from `verify_replay.py`. It reuses the
manager's two existing history banks and the runtime's accepted-window
promotion. The model opts into the manager's one-step history policy
through `gdn_first_row_checkpoint`.

In the isolated GB200 layer benchmark, the original 128-sequence,
two-row recurrent verifier took 590 us while replaying eight history
entries. The first-row verifier took 220 us while restoring the
preceding accepted draft. At batch 1 and width four, the corresponding
probe took 22.36 us and 10.15 us. These probes exclude convolution, GEMM,
and HTTP. All 59 target operation tests passed again, including physical
slot strides, the first write to an empty bank and both bank directions.

The default engine passed the complete development oracle and both EOS
checks again. The scores and every greedy continuation matched the
previous build.

| Batch | ISL | OSL | TTFT (ms) | TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1024 | 51.38 | 3.978 | 251.38 |
| 1 | 1024 | 1024 | 67.62 | 4.293 | 232.95 |
| 1 | 4096 | 1024 | 202.74 | 4.724 | 211.67 |

The OpenAI endpoint matched the batch-1 continuations exactly at
concurrency 1, 32, and 128 on the short-prompt development suite.

The first-row checkpoint improved the same short streaming sweep at all
three concurrencies, especially once the recurrent pool exceeded L2.

| ISL | OSL | Concurrency | Requests | Output tokens/s | Median TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 4 | 228.53 | 67.61 |
| 1024 | 1024 | 32 | 32 | 3291.88 | 1084.77 |
| 1024 | 1024 | 128 | 128 | 5379.37 | 3319.45 |

## Mixed prompt lengths

Mixing fresh development prompts with 1,024-token prompts exposed the
short-context numerical issue again. Item 28 diverged at the 0.5-nat
cutoff at concurrency 4, 32 and 128 because it took the long-context
prefill path when a long neighbor shared the batch. The target now
identifies fresh short requests separately, selects their rows for the
triangular GDN and FP32 attention calculations, and writes the results
back to their original rows and recurrent slots. A mixed step still
uses the runtime's normal paged KV and packed generation rows.

The mixed HTTP cases at concurrency 4, 32, and 128 now match all
isolated short-prompt continuations. The model also passed all 32
development items through `engine.sh` with the same KL and top-1
agreement, and `serve.sh` passed the homogeneous concurrency checks.

## Blackwell block-scaled projection dispatch

I reran the standalone reference at batch 1, ISL 1024 and OSL 1024 with
the GPU idle between processes. Two runs gave a median 256.04 decode
tokens/s and 50.57 ms TTFT. This supersedes the contended 203.94
tokens/s observation above for that input length.

Long prefill and established histories use DeepGEMM with the existing
128-channel E4M3 codes and packed exponents. The small-row output
projections use TensorRT-LLM's prequantized CuTe DSL operation, with
each 128-channel activation exponent repeated across four hardware
subgroups. The loader packs a second scale view over the same FP8
weights. At four rows the FFN down projection took 22.95 us in the CuTe
path and 46.52 us in the original GEMM. At 256 rows it took 32.97 us
in DeepGEMM and 59.34 us in the original GEMM.

Using the new reductions for all decode lengths changed item 28 at
the 0.5-nat cutoff. The target now keeps the preceding CUTLASS arithmetic
when any active sequence has a short history. It also keeps the stable
short paged-attention reduction in that decode graph. The shared graph
runner carries the opt-in sequence-length classification from confirmed
host token counts, ignoring padding requests. The first trial fell to
1,413 and 3,835 output tokens/s at concurrency 32 and 128 (ISL 1024,
OSL 1024). The CUDA trace showed eager launches because startup used one
dummy of the requested length and filled the rest of the batch with
shorter dummies. During startup the longest dummy now selects the graph
family. The default behavior of other graph families stays the same.

The split passed all 32 development items at KL 0.003288 and 99.909%
non-tie top-1 agreement, with the same isolated continuations as
before. The serving endpoint returned those same continuations at
concurrency 1, 32, and 128. All 64 target operations and 35 CPU CUDA-graph
runner tests passed. The DeepGEMM operation test allows one adjacent
BF16 value for individual outputs and limits the aggregate error
independently.

On three new technical chats per input length, seven of the nine matched
the standalone reference or diverged at a near-tie. One reference run
generated a different token from its own scoring argmax at 0.5 nats on
a 128-token input. On a 1,024-token input the target and reference scorers
agreed, while target generation picked a different token at a reference
gap of 0.625. The preceding CUTLASS-only build showed the same two
divergences.

| Batch | ISL | OSL | TTFT (ms) | TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1024 | 66.79 | 4.015 | 249.08 |
| 1 | 1024 | 1024 | 58.35 | 3.635 | 275.09 |
| 1 | 4096 | 1024 | 146.79 | 3.887 | 257.28 |

With both graph families captured at each batch size, the HTTP endpoint
matched every isolated development continuation in homogeneous batches
at concurrency 1, 32 and 128 and in mixed short/long batches at
concurrency 4, 32 and 128. The short streaming sweep used the same inputs
as the first-row checkpoint sweep.

| ISL | OSL | Concurrency | Requests | Output tokens/s | Median TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 4 | 292.25 | 66.19 |
| 1024 | 1024 | 32 | 32 | 3876.23 | 1021.08 |
| 1024 | 1024 | 128 | 128 | 6558.57 | 2552.01 |

## Packed QKV and fresh short prefill

The Q/K normalization and rotary kernels write one packed BF16 tensor
in the native attention layout. The target's short context and short
decode kernels read the corresponding strided views. A fresh short-only
prefill writes its keys and values directly using the layer's physical
page-table offsets and runs the FP32 causal reduction once. Mixed or
continued contexts still run the native paged operation.

The warp-local triangular solve is adapted from `short_solve.cu` in
aegis-engines. It matched the preceding BF16 solution bit for bit on
single and packed context lengths from 8 through 128. On the two-tile
128-token case the isolated solve fell from 64.27 to 25.47 us per layer.
The runtime-compiled launcher belongs in the family `csrc/` in a full
build. Scoring computes the same full-vocabulary log-softmax in
64-position groups to limit host FP32 scratch for long input sequences.

All 66 target operation tests passed, including strided QKV and physical
KV page writes. The CPU hybrid-manager suite passed 257 tests with one
CUDA-only skip. The complete development oracle and both EOS checks
passed with the same scores and continuations. The OpenAI endpoint also
matched every isolated development continuation at concurrency 1, 32,
and 128.

| Batch | ISL | OSL | TTFT (ms) | TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1024 | 61.56 | 3.709 | 269.58 |
| 1 | 1024 | 1024 | 65.92 | 3.505 | 285.33 |
| 1 | 4096 | 1024 | 141.69 | 3.825 | 261.46 |

The mixed HTTP checks at concurrency 4, 32, and 128 matched every
isolated short continuation. The packed-QKV streaming sweep measured
289.34, 3881.50 and 6593.92 output tokens/s at concurrency 1, 32 and 128
with ISL 1024 and OSL 1024.

## Batched recurrent inputs and temporal convolution

The batched verifier follows aegis-engines' rolling convolution and
prepared GDN recurrence. Each Q/K head is normalized once per input row,
and each value head writes its FP32 beta and decay once. The physical
convolution windows and the manager's double-buffered history keep the
same layouts. The isolated four-row verifier at batch 32 went from 117
to 103 us. The two-row verifier at batch 128 went from 235 to 216 us.
Outputs, FP32 states, and tentative histories matched exactly. Wider
Gluon CTA layouts raised the batch-128 latency. The earlier verifier is
used for small batches, where shared preparation lowered throughput.

Long prefill uses the temporal convolution tile from aegis-engines
`kernels.py`, writing Q, K and V directly into FlashInfer's three input
tensors. A separate pass commits the convolution taps so every tile
reads the prior window when a context chunk continues an existing
request. The isolated layer took 33.81 us for 1,024 rows, 122.95 us
for 4,096 rows, and 239.15 us for eight 1,024-token sequences. The prior
path took 101.26, 395.58, and 781.20 us. Every output and convolution
state entry matched. The B200 accuracy test list follows the target directory.
The mixed-prompt regression can run with the checkpoint alone. The
protocol test also uses the supplied development oracle.

All 70 operation tests and all 32 development oracle items passed.
The score metrics stayed at KL 0.003288 and 99.909% non-tie top-1
agreement.

| Batch | ISL | OSL | TTFT (ms) | TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1024 | 58.87 | 3.697 | 270.52 |
| 1 | 1024 | 1024 | 48.58 | 3.539 | 282.57 |
| 1 | 4096 | 1024 | 137.84 | 3.809 | 262.55 |

The OpenAI endpoint matched isolated development continuations at
concurrency 1, 32, and 128, including short requests in mixed batches.
Among the nine additional 64-output chats, a 1,024-input prompt returned
a different token at concurrency 32 where solo scoring separated the
leading logits by 0.625 nats. The other first differences in that mixed
HTTP sweep were below 0.5 nats.

| ISL | OSL | Concurrency | Requests | Output tokens/s | Median TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 4 | 292.52 | 62.00 |
| 1024 | 1024 | 32 | 32 | 3993.39 | 831.84 |
| 1024 | 1024 | 128 | 128 | 6797.90 | 2538.36 |

Changing only `modeling_v2` to `'off'` in the serving YAML reached
the Modeling V1 CUDA-graph capture and raised `GDN replay does not
support dynamic draft length yet: 2 != 4`. The earlier Modeling V1
results use its non-speculative configuration.

## Batch-1 draft-width pilot

The pilot used the packed-QKV build, the production generate handler,
and one LLM executor. Each run used the same prompt for a given ISL and
measured output tokens 33 through 1024. The suffix automaton trial used
a four-token minimum match. Its lookup overhead and draft choices
lowered throughput at all three lengths.

| Draft tokens | Suffix automaton | OSL | ISL 128 tokens/s | ISL 1024 tokens/s | ISL 4096 tokens/s |
| ---: | --- | ---: | ---: | ---: | ---: |
| 3 | Disabled | 1024 | 269.74 | 284.64 | 262.77 |
| 4 | Disabled | 1024 | 288.96 | 288.78 | 273.47 |
| 5 | Disabled | 1024 | 296.49 | 292.91 | 264.87 |
| 6 | Disabled | 1024 | 310.97 | 289.36 | 248.87 |
| 5 | Minimum match 4 | 1024 | 274.17 | 283.66 | 255.81 |

## Exact-length prefill and serving configuration pilots

The batch-1 generator captures breakable prefill CUDA graphs at 128,
1,024, and 4,096 tokens. Fresh single-request batches of these exact
lengths can replay a graph. The target stages the current positions,
recurrent slot, and physical KV page table before replay. It rebuilds
these plans when the cache manager changes between capacity profiling
and serving. The shared decoder input preparer applies the exact-shape
filter only when a model requests it. Other shapes, mixed steps,
continuation chunks, and scoring run eagerly.

With four MTP drafts and the three prefill graphs, the token protocol
passed all 32 development items and both EOS checks. Mean KL stayed at
0.003288 and non-tie top-1 agreement stayed at 99.909%. Decode
throughput varied across the two chats at each ISL.

| Batch | ISL | OSL | Prompt | TTFT (ms) | TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1024 | 1 | 28.74 | 3.384 | 295.49 |
| 1 | 128 | 1024 | 2 | 29.58 | 3.481 | 287.28 |
| 1 | 1024 | 1024 | 1 | 44.75 | 3.434 | 291.19 |
| 1 | 1024 | 1024 | 2 | 44.84 | 3.988 | 250.76 |
| 1 | 4096 | 1024 | 1 | 130.69 | 4.052 | 246.82 |
| 1 | 4096 | 1024 | 2 | 135.61 | 3.398 | 294.32 |

The HTTP launcher returned the same development continuations at
concurrency 1, 32, and 128. Five target routing tests and seven
shared prefill graph tests passed on CPU. An initial integration
control with a 4,352-token KV cap hit the runtime's page-table capacity
error when scheduling a long request. The exact-bucket regression
uses the launcher's 8,192-token cap.

At ISL 1,024 and OSL 1,024, the serving pilots sent four chats at
concurrency 1 and one wave of chats at each higher concurrency. All
variants used the same passages from the checkout's README. The
two-draft schedule applies above 32 requests.

| Pilot | ISL | OSL | C1 tokens/s | C32 tokens/s | C128 tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prepared GDN control | 1024 | 1024 | 292.52 | 3993.39 | 6797.90 |
| Two drafts above batch 32 | 1024 | 1024 | — | 3973.75 | 6508.59 |
| Stream every three iterations | 1024 | 1024 | 293.07 | 4027.66 | 6820.40 |
| 16,384-token step budget | 1024 | 1024 | — | 3996.21 | 6701.08 |
| Four HTTP frontend processes | 1024 | 1024 | 294.04 | 4002.85 | 6793.25 |
| 6,144-token sequence limit | 1024 | 1024 | 299.67 | 4048.07 | 6849.98 |

The 16,384-token step budget produced 2,760.30 output tokens/s in the
concurrency-32 pilot with ISL 4,096 and OSL 1,024. The smaller budget
served the 1,024-token workload faster at concurrency 128.

## Full prepared-GDN serving sweep

The sweep used the default request counts in `/task/bench_serving.py`,
with the checkout's README as the passage source. Each concurrency ran
against the same server and included the benchmark's warm-up request.
The benchmark measured streamed content and checked the exact input
and output token counts.

| ISL | OSL | Concurrency | Requests | Output tokens/s | Median TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 16 | 278.80 | 65.55 |
| 1024 | 1024 | 2 | 16 | 511.31 | 85.45 |
| 1024 | 1024 | 4 | 16 | 923.30 | 92.65 |
| 1024 | 1024 | 8 | 32 | 1618.18 | 98.13 |
| 1024 | 1024 | 16 | 64 | 2612.22 | 97.59 |
| 1024 | 1024 | 32 | 128 | 4017.40 | 143.91 |
| 1024 | 1024 | 64 | 256 | 5236.03 | 202.63 |
| 1024 | 1024 | 128 | 512 | 6941.87 | 390.08 |
| 4096 | 1024 | 1 | 16 | 273.67 | 138.01 |
| 4096 | 1024 | 8 | 32 | 1466.84 | 154.37 |
| 4096 | 1024 | 32 | 128 | 2937.32 | 294.45 |

The endpoint preserved the isolated development continuations at
concurrency 1, 32, and 128 and the short prompts in mixed batches.
On the additional nine 64-output chats, all first HTTP differences
from isolated generation had gaps of 0.125 nats or less at concurrency
4, 32, and 128. The two isolated differences from the standalone
reference persisted.

The exact-bucket regression at the launcher capacity replayed each
of the three configured lengths and matched eager generation on
all seven requests, including neighboring lengths and a repeated
128-token request.

The recurrent verifier layout sweep tried 4 to 64 values per CTA,
multiple warp counts, three CTA orders, and L1-bypassing state loads.
The two-row batch-128 state update took 192.14 us per layer in the
existing layout, which remained the fastest exact case. At batch 32,
the four-row update took 90.60 us and the best alternative took
90.29 us. Grouping convolution heads into CTAs made the fastest
batch-128 preparation 19.89 us instead of 23.90 us but changed the
FP32 normalization reduction. Layouts that reproduced the original
normalization were slower.

## Row-major TMA and overlapping gate projections

The wide FFN decode projection streams 128-channel E4M3 tiles from the
existing row-major weight. It uses the accumulation order from
aegis-engines `fp8_wide.py` and leaves the loader with one FP8 weight
allocation per projection. The isolated eight-row projection produced
the same BF16 result as the tiled CUTLASS path in 32.86 us versus
34.95 us. A second tile-major weight would have taken 31.28 us at the
cost of a second allocation. The narrower projection favored the
existing tiled kernel.

The BF16 GDN gate projection uses one auxiliary stream while the FP8
QKVZ projection runs. It retains the pinned five-way split-K reduction.
The execution stream joins the gate stream before the recurrence
reads the gates. All 79 target tests passed, including the FP32 GEMM
reference and caller-owned gate output buffer checks. The token
protocol passed all 32 development items and both EOS checks.

| Batch | ISL | OSL | Prompt | TTFT (ms) | TPOT (ms) | Decode tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1024 | 1 | 28.44 | 3.222 | 310.37 |
| 1 | 128 | 1024 | 2 | 29.21 | 3.297 | 303.30 |
| 1 | 1024 | 1024 | 1 | 44.37 | 3.235 | 309.07 |
| 1 | 1024 | 1024 | 2 | 44.37 | 3.138 | 318.68 |
| 1 | 4096 | 1024 | 1 | 132.33 | 3.442 | 290.51 |
| 1 | 4096 | 1024 | 2 | 135.21 | 3.223 | 310.31 |

The HTTP parity checks passed at concurrency 1, 32, and 128 and
on the mixed short and long batches. The nine additional chats
first differed from solo at gaps of at most 0.375 nats.

| ISL | OSL | Concurrency | Requests | Output tokens/s | Median TTFT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1024 | 1 | 4 | 307.32 | 66.39 |
| 1024 | 1024 | 32 | 32 | 4056.53 | 811.44 |
| 1024 | 1024 | 128 | 128 | 6866.35 | 2518.10 |

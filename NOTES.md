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

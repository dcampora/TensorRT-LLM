# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TensorRT-LLM native group-scaled FP8 projections and routed experts."""

import torch

from .fp8_low_m import block_scaled_low_m
from .quantize import block_quantize

BLOCK = 128


def quantize(x):
    """Quantize BF16 activations row-wise in blocks of 128 input features."""
    return block_quantize(x)


def linear(x, weight, scale):
    """Multiply [tokens, K] activations by the FP8 [N, K] weight."""
    return linear_quantized(*quantize(x), weight, scale)


def linear_quantized(x, x_scale, weight, scale):
    """Run the TRTLLM-Gen block-scaled GEMM on a prequantized activation."""
    if block_scaled_low_m.supports(x, weight):
        return block_scaled_low_m(x, x_scale, weight, scale)
    return torch.ops.trtllm.fp8_block_scaling_gemm(x, weight, x_scale, scale)


def routed_experts(logits, x, x_scale, up_gate, up_gate_scale, down, down_scale, top_k):
    """Select experts, normalize the selected logits, and apply SwiGLU experts."""
    expert_count, _, intermediate = down.shape
    return torch.ops.trtllm.fp8_block_scale_moe_runner(
        logits,
        None,
        x,
        x_scale,
        up_gate,
        up_gate_scale,
        down,
        down_scale,
        expert_count,
        top_k,
        None,
        None,
        None,
        intermediate,
        0,
        expert_count,
        None,
        1,  # Top-k then softmax over the selected logits.
    )

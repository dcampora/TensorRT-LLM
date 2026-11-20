# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage speculative causal convolution states for the runtime's acceptance update."""

import torch
import triton
import triton.language as tl


@triton.jit
def _verify_conv(
    X,
    Weight,
    State,
    Slots,
    Intermediate,
    Y,
    channels: tl.constexpr,
    tokens: tl.constexpr,
    x_stride: tl.constexpr,
    state_batch: tl.constexpr,
    state_channel: tl.constexpr,
    state_time: tl.constexpr,
    interm_batch: tl.constexpr,
    interm_step: tl.constexpr,
    interm_channel: tl.constexpr,
    interm_time: tl.constexpr,
    block: tl.constexpr,
):
    batch = tl.program_id(0)
    col = tl.program_id(1) * block + tl.arange(0, block)
    slot = tl.load(Slots + batch)
    mask = (slot >= 0) & (col < channels)
    state = State + slot * state_batch + col * state_channel
    oldest = tl.load(state, mask, 0).to(tl.float32)
    second = tl.load(state + state_time, mask, 0).to(tl.float32)
    newest = tl.load(state + 2 * state_time, mask, 0).to(tl.float32)
    weight = Weight + col * 4
    w0 = tl.load(weight, col < channels, 0).to(tl.float32)
    w1 = tl.load(weight + 1, col < channels, 0).to(tl.float32)
    w2 = tl.load(weight + 2, col < channels, 0).to(tl.float32)
    w3 = tl.load(weight + 3, col < channels, 0).to(tl.float32)
    intermediate = Intermediate + batch * interm_batch + col * interm_channel
    for step in tl.static_range(tokens):
        row = batch * tokens + step
        sample = tl.load(X + row * x_stride + col, col < channels, 0).to(tl.float32)
        y = oldest * w0 + second * w1 + newest * w2 + sample * w3
        y *= tl.sigmoid(y)
        tl.store(Y + row * channels + col, y, col < channels)
        window = intermediate + step * interm_step
        tl.store(window, second, mask)
        tl.store(window + interm_time, newest, mask)
        tl.store(window + 2 * interm_time, sample, mask)
        oldest, second, newest = second, newest, sample


def verify_conv(qkv, weight, state, slots, intermediate, tokens):
    """Evaluate every candidate and write its corresponding three-tap history."""
    rows, channels = qkv.shape
    output = torch.empty((rows, channels), device=qkv.device, dtype=qkv.dtype)
    block = 128
    _verify_conv[(slots.numel(), triton.cdiv(channels, block))](
        qkv,
        weight,
        state,
        slots,
        intermediate,
        output,
        channels,
        tokens,
        qkv.stride(0),
        *state.stride(),
        *intermediate.stride(),
        block,
        num_warps=4,
    )
    return output

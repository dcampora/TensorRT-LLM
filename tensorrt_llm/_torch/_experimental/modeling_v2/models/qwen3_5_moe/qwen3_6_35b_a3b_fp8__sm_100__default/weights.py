# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint mapping for the Qwen3.6 text decoder."""

import numpy as np
import torch


def _copy(dst, checkpoint, key):
    dst.copy_(checkpoint[key], non_blocking=True)


def _project(dst, checkpoint, paths):
    """Pack projections along the output axis while preserving the FP8 block grid."""
    rows = 0
    blocks = 0
    for path in paths:
        weight = checkpoint[path + ".weight"]
        scale = checkpoint[path + ".weight_scale_inv"]
        dst.weight[rows : rows + weight.shape[0]].copy_(weight, non_blocking=True)
        dst.scale[blocks : blocks + scale.shape[0]].copy_(scale, non_blocking=True)
        rows += weight.shape[0]
        blocks += scale.shape[0]


def _local_experts(dst, checkpoint, prefix):
    # Assemble a layer at a time on the host to transfer the packed expert tensors
    # instead of launching a small host-to-device copy for each expert.
    gate_up = np.empty(dst.up_gate.shape, dtype=np.uint8)
    down = np.empty(dst.down.shape, dtype=np.uint8)
    intermediate = dst.down.shape[-1]
    scale_rows = dst.down_scale.shape[-1]
    gate_up_scales = np.empty(dst.up_gate_scale.shape, dtype=np.int16)
    down_scales = np.empty(dst.down_scale.shape, dtype=np.int16)
    for i in range(dst.down.shape[0]):
        expert = f"{prefix}.experts.{i}"
        gate_up[i, :intermediate] = checkpoint[expert + ".up_proj.weight"].view(torch.uint8).numpy()
        gate_up[i, intermediate:] = (
            checkpoint[expert + ".gate_proj.weight"].view(torch.uint8).numpy()
        )
        down[i] = checkpoint[expert + ".down_proj.weight"].view(torch.uint8).numpy()
        gate_up_scales[i, :scale_rows] = (
            checkpoint[expert + ".up_proj.weight_scale_inv"].view(torch.int16).numpy()
        )
        gate_up_scales[i, scale_rows:] = (
            checkpoint[expert + ".gate_proj.weight_scale_inv"].view(torch.int16).numpy()
        )
        down_scales[i] = (
            checkpoint[expert + ".down_proj.weight_scale_inv"].view(torch.int16).numpy()
        )
    dst.up_gate.copy_(torch.from_numpy(gate_up).view(torch.float8_e4m3fn))
    dst.down.copy_(torch.from_numpy(down).view(torch.float8_e4m3fn))
    dst.up_gate_scale.copy_(torch.from_numpy(gate_up_scales).view(torch.bfloat16))
    dst.down_scale.copy_(torch.from_numpy(down_scales).view(torch.bfloat16))


def _load_moe(moe, checkpoint, prefix):
    _copy(moe.router, checkpoint, prefix + ".gate.weight")
    _copy(moe.shared_router, checkpoint, prefix + ".shared_expert_gate.weight")
    _project(
        moe.shared_gate_up,
        checkpoint,
        (prefix + ".shared_expert.gate_proj", prefix + ".shared_expert.up_proj"),
    )
    _project(moe.shared_down, checkpoint, (prefix + ".shared_expert.down_proj",))
    _local_experts(moe, checkpoint, prefix)


def _load_gdn(mixer, checkpoint, prefix):
    _project(mixer.qkvz, checkpoint, (prefix + ".in_proj_qkv", prefix + ".in_proj_z"))
    _project(mixer.output, checkpoint, (prefix + ".out_proj",))
    _copy(mixer.ab[: mixer.value_heads], checkpoint, prefix + ".in_proj_a.weight")
    _copy(mixer.ab[mixer.value_heads :], checkpoint, prefix + ".in_proj_b.weight")
    _copy(mixer.a_log, checkpoint, prefix + ".A_log")
    _copy(mixer.dt_bias, checkpoint, prefix + ".dt_bias")
    _copy(mixer.norm, checkpoint, prefix + ".norm.weight")
    mixer.conv.copy_(checkpoint[prefix + ".conv1d.weight"].squeeze(1), non_blocking=True)


def _load_attention(mixer, checkpoint, prefix):
    # HF stores [Q head, gate head] pairs. 256 rows per slice are aligned
    # to the 128-row weight blocks for both the main and the draft attention.
    weight = checkpoint[prefix + ".q_proj.weight"]
    scale = checkpoint[prefix + ".q_proj.weight_scale_inv"]
    q, gate = weight.view(mixer.heads, 2, mixer.head_dim, -1).unbind(1)
    q_scale, gate_scale = scale.view(mixer.heads, 2, mixer.head_dim // 128, -1).unbind(1)
    q_width = mixer.q_width
    _project(
        mixer.qkvg,
        {
            "q.weight": q.reshape(q_width, -1),
            "q.weight_scale_inv": q_scale.reshape(q_width // 128, -1),
            "k.weight": checkpoint[prefix + ".k_proj.weight"],
            "k.weight_scale_inv": checkpoint[prefix + ".k_proj.weight_scale_inv"],
            "v.weight": checkpoint[prefix + ".v_proj.weight"],
            "v.weight_scale_inv": checkpoint[prefix + ".v_proj.weight_scale_inv"],
            "gate.weight": gate.reshape(q_width, -1),
            "gate.weight_scale_inv": gate_scale.reshape(q_width // 128, -1),
        },
        ("q", "k", "v", "gate"),
    )
    _copy(mixer.q_norm, checkpoint, prefix + ".q_norm.weight")
    _copy(mixer.k_norm, checkpoint, prefix + ".k_norm.weight")
    _project(mixer.output, checkpoint, (prefix + ".o_proj",))


@torch.no_grad()
def load(model, checkpoint):
    """Populate model-owned tensors from the original HF checkpoint keys."""
    prefix = "model.language_model"
    _copy(model.model.embed_tokens.weight, checkpoint, prefix + ".embed_tokens.weight")
    _copy(model.model.norm, checkpoint, prefix + ".norm.weight")
    _copy(model.lm_head.weight, checkpoint, "lm_head.weight")
    for i, layer in enumerate(model.model.layers):
        base = f"{prefix}.layers.{i}"
        _copy(layer.input_norm, checkpoint, base + ".input_layernorm.weight")
        _copy(layer.post_attention_norm, checkpoint, base + ".post_attention_layernorm.weight")
        if layer.recurrent:
            _load_gdn(layer.mixer, checkpoint, base + ".linear_attn")
        else:
            _load_attention(layer.mixer, checkpoint, base + ".self_attn")
        _load_moe(layer.mlp, checkpoint, base + ".mlp")

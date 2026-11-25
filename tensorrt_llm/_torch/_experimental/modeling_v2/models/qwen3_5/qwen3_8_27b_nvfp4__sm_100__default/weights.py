# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load the text tensors and pack the projections from ModelOpt shards.

Fused FP8 scaling and NVFP4 scale swizzling follow aegis-engines cb6779a,
engine.py. Main-network weights retain their FP8 or packed NVFP4 storage.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from .ops.projections import FP8, Fp4Linear, Fp8Linear, fp8_requantize

if TYPE_CHECKING:
    from .modeling import ModelingV2Qwen38Nvfp4Sm100Default


class Checkpoint:
    def __init__(self, weights: Mapping[str, torch.Tensor]) -> None:
        self.weights = weights

    def tensor(self, key: str) -> torch.Tensor:
        return self.weights[key].to(device=torch.cuda.current_device())

    def fp8(self, linear: Fp8Linear, *names: str) -> None:
        parts = [self.tensor(name + ".weight") for name in names]
        in_scales = [self.tensor(name + ".input_scale") for name in names]
        out_scales = [self.tensor(name + ".weight_scale") for name in names]
        linear.input_scale = torch.stack(in_scales).max().reshape(1)
        linear.weight_scale = torch.stack(out_scales).max().reshape(1)
        if len(parts) > 1:
            parts = [
                fp8_requantize(weight, scale, linear.weight_scale)
                for weight, scale in zip(parts, out_scales, strict=True)
            ]
            weight = torch.cat([part.view(torch.uint8) for part in parts]).view(FP8)
        else:
            weight = parts[0]
        linear.weight = weight.T.unsqueeze(0)

    def fp4(self, linear: Fp4Linear, *names: str) -> None:
        in_scale = torch.stack([self.tensor(name + ".input_scale") for name in names]).max()
        w_scale = torch.stack([self.tensor(name + ".weight_scale_2") for name in names]).max()
        linear.inv_scale = in_scale.reciprocal()
        linear.alpha = (in_scale * w_scale).reshape(1)
        parts = [self.tensor(name + ".weight") for name in names]
        scales = [self.tensor(name + ".weight_scale").view(torch.uint8) for name in names]
        linear.weight = (torch.cat(parts) if len(parts) > 1 else parts[0]).T
        scale = torch.cat(scales) if len(scales) > 1 else scales[0]
        rows, cols = scale.shape
        linear.scales = (
            scale.reshape(rows // 128, 4, 32, cols // 4, 4)
            .permute(0, 3, 2, 1, 4)
            .contiguous()
            .reshape(rows, cols)
            .view(FP8)
            .T
        )


def load(model: ModelingV2Qwen38Nvfp4Sm100Default, weights: Mapping[str, torch.Tensor]) -> None:
    """Consume the matching text checkpoint weights, ignoring vision tensors."""
    checkpoint = Checkpoint(weights)
    prefix = "model.language_model."
    core = model.model
    core.embedding = checkpoint.tensor(prefix + "embed_tokens.weight")
    core.norm_weight = checkpoint.tensor(prefix + "norm.weight")
    for i, layer in enumerate(core.layers[: core.num_trunk_layers]):
        base = f"{prefix}layers.{i}."
        layer.input_norm = checkpoint.tensor(base + "input_layernorm.weight")
        layer.post_norm = checkpoint.tensor(base + "post_attention_layernorm.weight")
        checkpoint.fp4(layer.gate_up, base + "mlp.gate_proj", base + "mlp.up_proj")
        checkpoint.fp4(layer.down, base + "mlp.down_proj")
        if layer.is_linear:
            name = base + "linear_attn."
            checkpoint.fp8(layer.qkv, name + "in_proj_qkv", name + "in_proj_z")
            checkpoint.fp8(layer.out, name + "out_proj")
            layer.ba = torch.cat(
                [checkpoint.tensor(name + f"in_proj_{gate}.weight") for gate in ("b", "a")]
            )
            layer.conv_weight = checkpoint.tensor(name + "conv1d.weight").flatten(1)
            layer.log_decay = checkpoint.tensor(name + "A_log")
            layer.bias = checkpoint.tensor(name + "dt_bias")
            layer.gate_norm = checkpoint.tensor(name + "norm.weight")
        else:
            name = base + "self_attn."
            checkpoint.fp8(layer.qkv, name + "q_proj", name + "k_proj", name + "v_proj")
            checkpoint.fp8(layer.out, name + "o_proj")
            layer.q_norm = checkpoint.tensor(name + "q_norm.weight")
            layer.k_norm = checkpoint.tensor(name + "k_norm.weight")
    checkpoint.fp4(model.lm_head, "lm_head")
    if model.draft_model is not None:
        draft = model.draft_model.mtp_layers[0]
        draft.embedding_norm = checkpoint.tensor("mtp.pre_fc_norm_embedding.weight")
        draft.hidden_norm = checkpoint.tensor("mtp.pre_fc_norm_hidden.weight")
        draft.fc = checkpoint.tensor("mtp.fc.weight")
        draft.final_norm = checkpoint.tensor("mtp.norm.weight")
        base = "mtp.layers.0."
        draft.input_norm = checkpoint.tensor(base + "input_layernorm.weight")
        draft.post_norm = checkpoint.tensor(base + "post_attention_layernorm.weight")
        draft.q_norm = checkpoint.tensor(base + "self_attn.q_norm.weight")
        draft.k_norm = checkpoint.tensor(base + "self_attn.k_norm.weight")
        draft.qkv = torch.cat(
            [checkpoint.tensor(base + f"self_attn.{name}_proj.weight") for name in ("q", "k", "v")]
        )
        draft.out = checkpoint.tensor(base + "self_attn.o_proj.weight")
        draft.gate_up = torch.cat(
            [checkpoint.tensor(base + f"mlp.{name}_proj.weight") for name in ("gate", "up")]
        )
        draft.down = checkpoint.tensor(base + "mlp.down_proj.weight")

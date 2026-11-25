# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack the Qwen3.8 checkpoint's text projections for Blackwell."""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from .ops.projection import block_requantize

if TYPE_CHECKING:
    from .modeling import BlockProjection, DecoderLayer, ModelingV2Qwen38_27bFp8Sm100Default


TEXT_PREFIX = "model.language_model."


def load_projection(
    module: "BlockProjection",
    weights: Mapping[str, torch.Tensor],
    names: Sequence[str],
    device: torch.device,
    workspace: torch.Tensor,
) -> None:
    """Concatenate output blocks and convert to exponent-only FP8 scales."""
    tensors = [weights[name + ".weight"] for name in names]
    scales = [weights[name + ".weight_scale_inv"] for name in names]
    for value, scale in zip(tensors, scales, strict=True):
        if (
            value.dtype != torch.float8_e4m3fn
            or value.ndim != 2
            or any(dim % 128 for dim in value.shape)
            or scale.shape != (value.shape[0] // 128, value.shape[1] // 128)
        ):
            raise ValueError(f"Expected 128 x 128 block-scaled E4M3 weights for {names}")
    if len(tensors) == 1:
        value, block_scales = tensors[0], scales[0]
    else:
        value = torch.cat([weight.view(torch.uint8) for weight in tensors]).view(
            torch.float8_e4m3fn
        )
        block_scales = torch.cat(scales)
    module.weight, module.scales = block_requantize(
        value.to(device).contiguous(), block_scales.to(device).contiguous()
    )
    module.scales_mn = module.scales.T.float().contiguous()
    if module.weight.shape[0] == 5120:
        module.narrow_scales = module.scales_mn.repeat_interleave(2, dim=1)
    module.workspace = workspace


@torch.no_grad()
def load_model(
    model: "ModelingV2Qwen38_27bFp8Sm100Default", weights: Mapping[str, torch.Tensor]
) -> None:
    """Load the text tower from the composite HF checkpoint."""
    decoder = model.model
    device = model.lm_head.weight.device
    decoder.embed_tokens.weight = weights[TEXT_PREFIX + "embed_tokens.weight"].to(device)
    decoder.final_norm = weights[TEXT_PREFIX + "norm.weight"].to(device)
    model.lm_head.weight.copy_(weights["lm_head.weight"])
    decoder.gemm_workspace = torch.empty(16 * 1024 * 1024, device=device, dtype=torch.uint8)

    for index, layer in enumerate(decoder.layers):
        load_layer(layer, f"{TEXT_PREFIX}layers.{index}.", weights, device, decoder.gemm_workspace)

    inv_freq = 1.0 / (
        decoder.config.rope_theta
        ** (torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64)
    )
    phase = torch.outer(
        torch.arange(decoder.config.max_position_embeddings, device=device, dtype=torch.float32),
        inv_freq,
    )
    decoder.rope = torch.cat((phase.cos(), phase.sin()), dim=1)
    decoder.gate_workspace = torch.empty(16 * 1024 * 1024, device=device, dtype=torch.uint8)

    if model.draft_model is not None:
        draft = model.draft_model.mtp_layers[0]
        load_layer(draft, "mtp.layers.0.", weights, device, decoder.gemm_workspace)
        draft.fc = weights["mtp.fc.weight"].to(device)
        draft.embedding_norm = weights["mtp.pre_fc_norm_embedding.weight"].to(device)
        draft.hidden_norm = weights["mtp.pre_fc_norm_hidden.weight"].to(device)
        draft.output_norm = weights["mtp.norm.weight"].to(device)
        draft.rope = decoder.rope


def load_layer(
    layer: "DecoderLayer",
    prefix: str,
    weights: Mapping[str, torch.Tensor],
    device: torch.device,
    workspace: torch.Tensor,
) -> None:
    """Load one decoder layer from its checkpoint prefix."""
    layer.pre_norm = weights[prefix + "input_layernorm.weight"].to(device)
    layer.post_norm = weights[prefix + "post_attention_layernorm.weight"].to(device)
    load_projection(
        layer.gate_up,
        weights,
        [prefix + "mlp.gate_proj", prefix + "mlp.up_proj"],
        device,
        workspace,
    )
    load_projection(layer.down, weights, [prefix + "mlp.down_proj"], device, workspace)
    if layer.is_linear:
        linear = prefix + "linear_attn."
        load_projection(
            layer.input_proj,
            weights,
            [linear + "in_proj_qkv", linear + "in_proj_z"],
            device,
            workspace,
        )
        load_projection(layer.output_proj, weights, [linear + "out_proj"], device, workspace)
        layer.ba = torch.cat(
            [weights[linear + f"in_proj_{name}.weight"] for name in ("b", "a")]
        ).to(device)
        layer.conv = weights[linear + "conv1d.weight"].to(device)
        layer.a_log = weights[linear + "A_log"].to(device)
        layer.dt_bias = weights[linear + "dt_bias"].to(device)
        layer.gdn_norm = weights[linear + "norm.weight"].to(device)
    else:
        attention = prefix + "self_attn."
        load_projection(
            layer.input_proj,
            weights,
            [attention + name + "_proj" for name in ("q", "k", "v")],
            device,
            workspace,
        )
        load_projection(layer.output_proj, weights, [attention + "o_proj"], device, workspace)
        layer.q_norm = weights[attention + "q_norm.weight"].to(device)
        layer.k_norm = weights[attention + "k_norm.weight"].to(device)

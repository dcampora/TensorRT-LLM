# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint packing and ModelOpt-scale preparation for Qwen3.6."""

import torch
import triton
from torch import nn

from .kernels import quantized
from .kernels.fused_gate import interleave

BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
HIDDEN = 2048
VOCAB = 248320
NUM_EXPERTS = 256
EXPERT_WIDTH = 512
TOP_K = 8
LANGUAGE = "model.language_model"


def parameter(shape, dtype=BF16):
    return nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)


class Fp8Weights(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = parameter((out_features, in_features), FP8)
        self.input_scale = parameter((), torch.float32)
        self.weight_scale = parameter((), torch.float32)

    def load(self, checkpoint, names):
        input_scales = torch.stack([checkpoint[f"{name}.input_scale"] for name in names])
        weight_scales = torch.stack([checkpoint[f"{name}.weight_scale"] for name in names])
        self.input_scale.copy_(input_scales.max())
        self.weight_scale.copy_(weight_scales.max())
        if len(names) == 1:
            self.weight.copy_(checkpoint[f"{names[0]}.weight"])
            return

        # Calibrate the fused GEMM at the largest scale among its shards.
        # FP16 rounding occurs in registers before writing the fused E4M3 tensor.
        offset = 0
        for name, scale in zip(names, weight_scales.to(self.weight.device)):
            weight = checkpoint[f"{name}.weight"].to(self.weight.device)
            rows = weight.shape[0]
            quantized.requant_fp8[(triton.cdiv(weight.numel(), 4096),)](
                weight,
                scale,
                self.weight_scale,
                self.weight[offset : offset + rows],
                weight.numel(),
                4096,
            )
            offset += rows


class PackedWeights(nn.Module):
    def __init__(self, in_features, out_features, *, experts=None, interleaved=False):
        super().__init__()
        shape = (experts,) if experts is not None else ()
        self.w = parameter((*shape, out_features, in_features // 2), torch.uint8)
        self.scales = parameter((*shape, out_features, in_features // 16), FP8)
        self.global_scales = parameter(shape, torch.float32)
        self.alpha = None
        self.n = out_features
        self.k = in_features
        self.interleaved = interleaved

    def load(self, checkpoint, roots, projections, *, shared_gate=False):
        def pack(root, suffix):
            prefix = f"{root}." if root else ""
            shards = [checkpoint[f"{prefix}{name}.{suffix}"] for name in projections]
            if len(shards) == 1:
                return shards[0]
            if suffix == "weight_scale_2":
                return torch.maximum(*shards) if shared_gate else shards[0]
            if suffix == "weight_scale":
                return torch.cat([shard.view(torch.uint8) for shard in shards]).view(FP8)
            return torch.cat(shards)

        for suffix, dest in (
            ("weight", self.w),
            ("weight_scale", self.scales),
            ("weight_scale_2", self.global_scales),
        ):
            items = [pack(root, suffix) for root in roots]
            if dest.ndim in (1, 3):
                if suffix == "weight_scale":
                    items = [item.view(torch.uint8) for item in items]
                data = torch.stack(items).view(dest.dtype)
            else:
                data = items[0]
            if self.interleaved and suffix != "weight_scale_2":
                data = interleave(data.view(torch.uint8)).view(dest.dtype)
            dest.copy_(data.reshape(dest.shape))

    def prepare(self):
        self.alpha = self.global_scales.to(BF16)


def load_decoder_weights(decoder, checkpoint):
    decoder.embed_tokens.weight.copy_(checkpoint[f"{LANGUAGE}.embed_tokens.weight"])
    decoder.final_norm.copy_(checkpoint[f"{LANGUAGE}.norm.weight"])
    for i, layer in enumerate(decoder.layers):
        root = f"{LANGUAGE}.layers.{i}"
        layer.before.copy_(checkpoint[f"{root}.input_layernorm.weight"])
        layer.after.copy_(checkpoint[f"{root}.post_attention_layernorm.weight"])
        attn = layer.attention
        if layer.full_attention:
            name = f"{root}.self_attn"
            attn.qkv.load(checkpoint, [f"{name}.{proj}_proj" for proj in "qkv"])
            attn.output.load(checkpoint, [f"{name}.o_proj"])
            attn.qnorm.copy_(checkpoint[f"{name}.q_norm.weight"])
            attn.knorm.copy_(checkpoint[f"{name}.k_norm.weight"])
        else:
            name = f"{root}.linear_attn"
            attn.qkvz.load(checkpoint, [f"{name}.in_proj_qkv", f"{name}.in_proj_z"])
            attn.output.load(checkpoint, [f"{name}.out_proj"])
            attn.ba.copy_(torch.cat([checkpoint[f"{name}.in_proj_{p}.weight"] for p in "ba"]))
            attn.conv.copy_(checkpoint[f"{name}.conv1d.weight"].reshape_as(attn.conv))
            attn.norm.copy_(checkpoint[f"{name}.norm.weight"])
            attn.a_log.copy_(checkpoint[f"{name}.A_log"])
            attn.dt.copy_(checkpoint[f"{name}.dt_bias"])

        moe = layer.moe
        name = f"{root}.mlp"
        experts = [f"{name}.experts.{expert}" for expert in range(NUM_EXPERTS)]
        shared = [f"{name}.shared_expert"]
        moe.gate_up.load(checkpoint, experts, ("gate_proj", "up_proj"))
        moe.down.load(checkpoint, experts, ("down_proj",))
        moe.shared_gate_up.load(checkpoint, shared, ("gate_proj", "up_proj"), shared_gate=True)
        moe.shared_down.load(checkpoint, shared, ("down_proj",))
        moe.router[:NUM_EXPERTS].copy_(checkpoint[f"{name}.gate.weight"])
        moe.router[NUM_EXPERTS].copy_(checkpoint[f"{name}.shared_expert_gate.weight"][0])
        moe.router[NUM_EXPERTS + 1 :].zero_()

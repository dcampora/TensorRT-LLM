# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from aegis-engines fused_gate.py (a6c09c6502)."""

import functools
import importlib.util
import os
import sys
from pathlib import Path

import torch
from flashinfer.fused_moe.cute_dsl.blackwell import moe_w4a16, moe_w4a16_kernel
from flashinfer.jit.env import FLASHINFER_WORKSPACE_DIR
from flashinfer.tllm_enums import ActivationType

CACHE = FLASHINFER_WORKSPACE_DIR / "qwen36_modelopt"


def _copy_module(module, name, edit):
    shortname = name.rpartition(".")[-1]
    file = CACHE / f"{shortname}.py"
    CACHE.mkdir(exist_ok=True, parents=True)
    source = edit(Path(module.__file__).read_text())
    if not file.exists() or source != file.read_text():
        pending = file.with_suffix(f".{os.getpid()}.py")
        pending.write_text(source)
        pending.replace(file)
    spec = importlib.util.spec_from_file_location(name, file)
    out = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = out
    spec.loader.exec_module(out)
    return out


def _edit_kernel(source):
    start = source.index("                            up_pair = cute.arch.mul_packed_f32x2(")
    end = source.index("                            if cutlass.const_expr(", start)
    return (
        source[:start]
        + """                            up0 = cutlass.Float32(cutlass.BFloat16(up[i])) * gated_alpha_f32
                            up1 = cutlass.Float32(cutlass.BFloat16(up[i + 1])) * gated_alpha_f32
                            gate0 = cutlass.Float32(cutlass.BFloat16(gate[i])) * gated_alpha_f32
                            gate1 = cutlass.Float32(cutlass.BFloat16(gate[i + 1])) * gated_alpha_f32
                            up_pair = (cutlass.Float32(cutlass.BFloat16(up0)), cutlass.Float32(cutlass.BFloat16(up1)))
                            gate_pair = (
                                cutlass.Float32(cutlass.BFloat16(gate0)),
                                cutlass.Float32(cutlass.BFloat16(gate1)),
                            )
"""
        + source[end:]
    )


@functools.cache
def _packed_gate_runner():
    kernel_module = moe_w4a16_kernel.__package__ + ".modelopt_w4a16_kernel"
    _copy_module(moe_w4a16_kernel, kernel_module, _edit_kernel)
    launch = _copy_module(
        moe_w4a16,
        moe_w4a16.__package__ + ".modelopt_w4a16",
        lambda s: s.replace("from .moe_w4a16_kernel import", "from .modelopt_w4a16_kernel import"),
    )
    return launch._run_grouped_gemm


def interleave(tensor):
    """Pack the up and gate projections as 64-output-row pairs."""
    shape = tensor.shape
    gate, up = tensor.reshape(-1, 2, 8, 64, shape[-1]).unbind(1)
    return torch.stack((up, gate), dim=2).contiguous().view(shape)


def launch(params, packed_input, output, common):
    w, sf, alpha = params
    opts = dict(common, alpha=alpha, activation_type=ActivationType.Swiglu)
    _packed_gate_runner()(weight=w, weight_sf=sf, activations=packed_input, output=output, **opts)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused residual RMSNorm, adapted from aegis-engines norm_quant.py (114cf57c911e)."""

import ctypes
from functools import cache

import torch

from .activation_quant import buffers
from .native import load_library


@cache
def _library() -> ctypes.CDLL:
    return load_library("norm_quant.cu", "nvcc", ("-Xcompiler", "-fPIC", "-arch=sm_100a"))


@cache
def _norm() -> ctypes._CFuncPtr:
    function = _library().norm_quant
    ptr = ctypes.c_void_p
    function.argtypes = [
        ptr,
        ptr,
        ptr,
        ptr,
        ptr,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.c_bool,
        ptr,
    ]
    function.restype = ctypes.c_int
    return function


def residual_norm_quantize(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    store_normalized: bool,
    programmatic: bool = False,
    packed_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add the BF16 residual and produce FP8 activation groups in one kernel."""
    rows, cols = x.shape
    a, scales = buffers(x, rows, cols, packed_scales)
    stream = torch.cuda.current_stream(x.device).cuda_stream
    status = _norm()(
        x.data_ptr(),
        residual.data_ptr(),
        weight.data_ptr(),
        a.data_ptr(),
        scales.data_ptr(),
        rows,
        a.shape[0],
        store_normalized,
        programmatic,
        packed_scales,
        stream,
    )
    if status:
        raise RuntimeError(f"Fused residual norm failed (CUDA status {status})")
    return a, scales


@cache
def _draft_norm() -> ctypes._CFuncPtr:
    function = _library().draft_dual_norm
    ptr = ctypes.c_void_p
    function.argtypes = [ptr, ptr, ptr, ptr, ptr, ptr, ctypes.c_int, ctypes.c_bool, ptr]
    function.restype = ctypes.c_int
    return function


def dual_input_norm(
    ids: torch.Tensor,
    table: torch.Tensor,
    hidden: torch.Tensor,
    token_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
) -> torch.Tensor:
    """Gather the draft token and normalize both BF16 inputs directly into the FC layout."""
    rows, cols = hidden.shape
    output = hidden.new_empty((rows, 2 * cols))
    status = _draft_norm()(
        ids.data_ptr(),
        table.data_ptr(),
        hidden.data_ptr(),
        token_weight.data_ptr(),
        hidden_weight.data_ptr(),
        output.data_ptr(),
        rows,
        ids.dtype == torch.int64,
        torch.cuda.current_stream(hidden.device).cuda_stream,
    )
    if status:
        raise RuntimeError(f"MTP input norms failed (CUDA status {status})")
    return output

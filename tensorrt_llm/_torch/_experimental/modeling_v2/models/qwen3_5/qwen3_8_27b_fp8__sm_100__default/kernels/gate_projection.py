# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned BF16 GDN gate projection from aegis-engines ba_lt.py (114cf57c911e)."""

import ctypes
from functools import cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME

from .native import load_library


@cache
def _operation() -> ctypes._CFuncPtr:
    if CUDA_HOME is None:
        raise RuntimeError("The Qwen3.8 gate kernel requires a CUDA toolkit")
    cuda = Path(CUDA_HOME)
    library = load_library(
        "gate_projection.cpp",
        "g++",
        ("-fPIC", f"-I{cuda / 'include'}", f"-L{cuda / 'lib64'}", "-lcublasLt"),
    )
    function = library.project_gdn_gates
    ptr = ctypes.c_void_p
    function.argtypes = [ptr, ptr, ptr, ctypes.c_int, ptr, ctypes.c_size_t, ptr]
    function.restype = ctypes.c_int
    return function


def gate_projection(x: torch.Tensor, weight: torch.Tensor, workspace: torch.Tensor) -> torch.Tensor:
    """Execute the same split-K BF16 reduction at each decode batch size."""
    rows = x.shape[0]
    out = x.new_empty((rows, 96))
    stream = torch.cuda.current_stream(x.device).cuda_stream
    status = _operation()(
        x.data_ptr(),
        weight.data_ptr(),
        out.data_ptr(),
        rows,
        workspace.data_ptr(),
        workspace.numel(),
        stream,
    )
    if status:
        raise RuntimeError(f"GDN gate projection failed at {rows} rows (cuBLASLt status {status})")
    return out

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM100 CUTLASS tiles adapted from aegis-engines fp8_tiles.py (114cf57c911e)."""

import ctypes
from functools import cache
from pathlib import Path

import flashinfer
import torch

from .native import load_library


@cache
def _operation() -> ctypes._CFuncPtr:
    cutlass = Path(flashinfer.__file__).parent / "data" / "cutlass"
    library = load_library(
        "fp8_tiles.cu",
        "nvcc",
        (
            "-Xcompiler",
            "-fPIC",
            "--expt-relaxed-constexpr",
            "-arch=sm_100a",
            "-diag-suppress=20012",
            f"-I{cutlass / 'include'}",
            f"-I{cutlass / 'tools/util/include'}",
        ),
    )
    function = library.fp8_tiled_matmul
    ptr = ctypes.c_void_p
    function.argtypes = [
        ptr,
        ctypes.c_size_t,
        ptr,
        ptr,
        ptr,
        ptr,
        ptr,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ptr,
        ctypes.c_bool,
        ctypes.c_bool,
    ]
    function.restype = ctypes.c_int
    return function


def tiled_gemm(
    a: torch.Tensor,
    weight: torch.Tensor,
    a_scales: torch.Tensor,
    w_scales: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    """Project aligned activation rows using the small-N or small-M SM100 tile."""
    m, k = a.shape
    n = weight.shape[0]
    result = torch.empty((m, n), device=a.device, dtype=torch.bfloat16)
    stream = torch.cuda.current_stream(a.device).cuda_stream
    status = _operation()(
        workspace.data_ptr(),
        workspace.numel(),
        a.data_ptr(),
        weight.data_ptr(),
        a_scales.data_ptr(),
        w_scales.data_ptr(),
        result.data_ptr(),
        m,
        n,
        k,
        stream,
        m <= 8,
        True,
    )
    if status:
        raise RuntimeError(f"SM100 FP8 projection {m} x {n} x {k} failed (CUTLASS status {status})")
    return result

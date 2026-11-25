# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Short GDN triangular solve adapted from aegis-engines short_solve.py (114cf57c911e)."""

import ctypes
from functools import cache

import torch

from .native import load_library


@cache
def _operation() -> ctypes._CFuncPtr:
    library = load_library(
        "short_solve.cu", "nvcc", ("-Xcompiler", "-fPIC", "-arch=sm_100a", "--fmad=false")
    )
    function = library.short_solve
    ptr = ctypes.c_void_p
    function.argtypes = [ptr, ptr, ptr, ptr, ctypes.c_int, ptr]
    function.restype = ctypes.c_int
    return function


def solve_short(lower: torch.Tensor, offsets: torch.Tensor, chunks: torch.Tensor) -> torch.Tensor:
    """Return the BF16 solution in the existing chunk/head layout."""
    output = torch.empty_like(lower, dtype=torch.bfloat16)
    status = _operation()(
        lower.data_ptr(),
        offsets.data_ptr(),
        chunks.data_ptr(),
        output.data_ptr(),
        chunks.shape[0],
        torch.cuda.current_stream(lower.device).cuda_stream,
    )
    if status:
        raise RuntimeError(f"Short-prefill GDN solve failed with CUDA status {status}")
    return output

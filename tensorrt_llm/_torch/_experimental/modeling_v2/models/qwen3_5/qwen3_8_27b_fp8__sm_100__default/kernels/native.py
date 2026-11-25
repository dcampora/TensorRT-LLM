# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build target-local shared libraries when the wheel lacks their kernels."""

import ctypes
import hashlib
import os
import subprocess
import sys
from functools import cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME


@cache
def load_library(name: str, compiler: str, flags: tuple[str, ...]) -> ctypes.CDLL:
    """Build the exact source for this CUDA toolkit and load it once per process."""
    source = Path(__file__).with_name(name)
    cache_dir = Path.home() / ".cache" / "tensorrt_llm" / "qwen38_fp8_sm100"
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity = repr((compiler, flags, torch.version.cuda, CUDA_HOME)).encode()
    digest = hashlib.sha256(source.read_bytes() + identity).hexdigest()[:16]
    library = cache_dir / f"{source.stem}_{digest}.so"
    if not library.exists():
        temporary = library.with_suffix(f".{os.getpid()}.so")
        try:
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-O3",
                    "-shared",
                    str(source),
                    *flags,
                    "-o",
                    str(temporary),
                ],
                check=True,
                stdout=sys.stderr,
            )
            temporary.replace(library)
        finally:
            temporary.unlink(missing_ok=True)
    return ctypes.CDLL(str(library))

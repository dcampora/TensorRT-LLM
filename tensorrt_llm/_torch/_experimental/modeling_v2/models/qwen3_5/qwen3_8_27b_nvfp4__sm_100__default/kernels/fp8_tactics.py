# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from aegis-engines cb6779a, fp8_tactics.py.
"""Select wider cuBLASLt tiles without changing the decode reduction."""

import ctypes
from functools import cache

import torch

ALGORITHM_BYTES = 64
# cublasLtMatmulAlgoConfigAttributes_t
ALGORITHM_ID = 0
TILE_ID = 1
SPLIT_K = 2
REDUCTION_SCHEME = 3
CUSTOM_OPTION = 5
STAGES_ID = 6

# (batch, input width, output width): (tile ID, custom option).
# Measured on a GB200 using rotating checkpoint layers and real activations.
WIDE_TILES = {
    (32, 5120, 16384): (16, 1),
    (32, 6144, 5120): (16, 1),
    (32, 5120, 14336): (13, 1),
    (64, 5120, 16384): (15, 1),
    (64, 6144, 5120): (18, 1),
    (64, 5120, 14336): (15, 1),
    (96, 5120, 16384): (33, 1),
    (96, 6144, 5120): (26, 1),
    (96, 5120, 14336): (458, 1),
    (128, 5120, 16384): (165, 0),
    (128, 6144, 5120): (165, 0),
    (128, 5120, 14336): (28, 0),
    (256, 5120, 16384): (165, 2),
    (256, 6144, 5120): (184, 2),
    (256, 5120, 14336): (31, 1),
    (384, 5120, 16384): (201, 0),
    (384, 6144, 5120): (197, 0),
    (384, 5120, 14336): (209, 1),
    (512, 5120, 16384): (184, 2),
    (512, 6144, 5120): (184, 2),
    (512, 5120, 14336): (184, 2),
}


# (rows, input width, output width): (tile, custom option).
# Exact among the same full-K cuBLASLt family on captured model activations.
PREFILL_FP8_TILES = {
    (128, 5120, 16384): (165, 0),
    (1024, 5120, 16384): (184, 2),
    (4096, 5120, 14336): (201, 2),
    (4096, 5120, 16384): (513, 1),
}


@cache
def _config_api():
    lib = ctypes.CDLL("libcublasLt.so.13")
    get = lib.cublasLtMatmulAlgoConfigGetAttribute
    get.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    get.restype = ctypes.c_int
    set_value = lib.cublasLtMatmulAlgoConfigSetAttribute
    set_value.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    set_value.restype = ctypes.c_int
    return get, set_value


def _get(algorithm, attr):
    value = ctypes.c_int()
    result_size = ctypes.c_size_t()
    status = _config_api()[0](
        algorithm.data_ptr(),
        attr,
        ctypes.byref(value),
        ctypes.sizeof(value),
        ctypes.byref(result_size),
    )
    return None if status else value.value


def _set(algorithm, attr, n):
    value = ctypes.c_int(n)
    status = _config_api()[1](algorithm.data_ptr(), attr, ctypes.byref(value), ctypes.sizeof(value))
    return status == 0


def wider_algorithm(module, q, weight, out, input_scale, weight_scale, workspace, reference):
    shape = (q.shape[1], q.shape[2], out.shape[-1])
    tuning = WIDE_TILES.get(shape)
    if tuning is None:
        return reference
    tile, option = tuning
    available = torch.empty(16 * ALGORITHM_BYTES, dtype=torch.uint8)
    count = module.bmm_fp8_get_algos(
        q, weight, out, input_scale, weight_scale, workspace, available
    )
    family, stages = _get(reference, ALGORITHM_ID), _get(reference, STAGES_ID)
    splits, reduction = _get(reference, SPLIT_K), _get(reference, REDUCTION_SCHEME)
    if None in (family, stages, splits, reduction):
        return reference
    for i in range(count):
        config = available[i * ALGORITHM_BYTES : (i + 1) * ALGORITHM_BYTES]
        if (
            _get(config, ALGORITHM_ID) != family
            or _get(config, STAGES_ID) != stages
            or _get(config, TILE_ID) != tile
            or _get(config, CUSTOM_OPTION) != option
        ):
            continue
        # A tile can cover more rows while keeping the K partitions and
        # floating-point reduction identical to the single-row algorithm.
        if _set(config, SPLIT_K, splits) and _set(config, REDUCTION_SCHEME, reduction):
            return config.clone()
    return reference


def prefill_algorithm(module, q, weight, out, input_scale, weight_scale, workspace, tile, option):
    bank = torch.empty(16 * ALGORITHM_BYTES, dtype=torch.uint8)
    count = module.bmm_fp8_get_algos(q, weight, out, input_scale, weight_scale, workspace, bank)
    for i in range(count):
        algo = bank[i * ALGORITHM_BYTES : (i + 1) * ALGORITHM_BYTES]
        if (
            _get(algo, ALGORITHM_ID) == 66
            and _get(algo, STAGES_ID) == 36
            and _get(algo, TILE_ID) == tile
            and _get(algo, CUSTOM_OPTION) == option
            and _get(algo, SPLIT_K) == 1
            and _get(algo, REDUCTION_SCHEME) == 0
        ):
            return algo.clone()
    return None

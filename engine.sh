#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# -ne 1 ]]; then
    printf 'Usage: %s MODEL_DIR\n' "$0" >&2
    exit 2
fi
exec python3 -m aegis_engine "$1"

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# -ne 1 && ( $# -ne 3 || $2 != --port ) ]]; then
    printf 'Usage: %s MODEL_DIR [--port PORT]\n' "$0" >&2
    exit 2
fi
script_dir="$(cd "$(dirname "$0")" && pwd)"
export TRTLLM_USE_GDN_REPLAY=0
exec trtllm-serve "$1" --port "${3:-8000}" --config "$script_dir/aegis_engine/serve.yaml" \
    --served_model_name qwen3.6-35b-a3b-fp8

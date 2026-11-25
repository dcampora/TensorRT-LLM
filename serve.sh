#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# -ne 1 && ( $# -ne 3 || ${2:-} != --port ) ]]; then
    echo "Usage: $0 <model directory> [--port N]" >&2
    exit 2
fi
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
exec trtllm-serve "$1" --port "${3:-8000}" --served_model_name qwen3.8-27b-fp8 \
    --config "$root/aegis_engine/serve.yaml"

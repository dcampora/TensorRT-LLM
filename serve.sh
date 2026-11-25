#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# != 1 && ( $# != 3 || ${2:-} != --port ) ]]; then
    echo "usage: $0 MODEL_DIRECTORY [--port PORT]" >&2
    exit 2
fi
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export TLLM_WORKER_USE_SINGLE_PROCESS=1
exec trtllm-serve serve "$1" --port "${3:-8000}" \
    --served_model_name qwen3.8-27b-nvfp4 --config "$root/aegis_engine/serve.yaml"

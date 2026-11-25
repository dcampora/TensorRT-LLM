#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if [[ $# != 1 && ( $# != 3 || $2 != --port ) ]]; then
    echo "usage: $0 <model directory> [--port N]" >&2
    exit 2
fi
model=$(realpath -- "$1")
port=${3:-8000}
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec trtllm-serve "$model" --port "$port" --served_model_name qwen3.6-35b-a3b-nvfp4 \
    --config aegis_engine/serve.yaml \
    --set "modeling_v2='${AEGIS_MODELING_V2:-require}'"

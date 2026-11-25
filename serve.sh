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
mode=${AEGIS_MODELING_V2:-require}
default_drafts=0
if [[ "$mode" == require ]]; then
    default_drafts=3
fi
options=(--config aegis_engine/serve.yaml --set "modeling_v2='$mode'")
if [[ "$mode" == require && "${AEGIS_PREFILL_GRAPHS:-1}" == 1 ]]; then
    options+=(--set "prefill_cuda_graph_backend='breakable'")
    options+=(--set "prefill_capture_num_tokens=[128, 256, 512, 1024, 2048, 4096]")
fi
drafts=${AEGIS_MTP_DRAFTS:-$default_drafts}
if [[ "$drafts" =~ ^[123]$ ]]; then
    options+=(--set "speculative_config={decoding_type: MTP, max_draft_len: $drafts}")
elif [[ "$drafts" != 0 ]]; then
    echo "AEGIS_MTP_DRAFTS must be between 0 and 3" >&2
    exit 2
fi
exec trtllm-serve "$model" --port "$port" --served_model_name qwen3.6-35b-a3b-nvfp4 "${options[@]}"

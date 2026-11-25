#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${root}${PYTHONPATH:+:${PYTHONPATH}}"
export TLLM_WORKER_USE_SINGLE_PROCESS=1
exec python3 -u -m aegis_engine "$@"

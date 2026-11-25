# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-context prefill captures through the runtime's breakable CUDA graphs."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import (
    eager_on_graph,
    get_current_replay_token,
)

if TYPE_CHECKING:
    from .modeling import Qwen38Decoder, Step


class PrefillGraph:
    def __init__(self, core: Qwen38Decoder) -> None:
        self.core = core
        self.metadata = None
        self.positions: torch.Tensor | None = None
        self.verify = False
        self.last_only = False
        self.steps: dict[tuple[int, int], Step] = {}
        self.graphs: dict[tuple[int, int, bool], tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
        self.run = eager_on_graph(self._eager, graph_key=self._key, graph_body=self._capture)

    def prepare(
        self, metadata, positions: torch.Tensor, verify: bool, return_context_logits: bool
    ) -> None:
        self.metadata = metadata
        self.positions = positions[..., : metadata.num_tokens]
        self.verify = verify
        self.last_only = not verify and not return_context_logits

    def _key(self, ids: torch.Tensor) -> tuple[int, int, bool] | None:
        md = self.metadata
        rows = ids.shape[0]
        if (
            md.num_contexts != 1
            or md.num_seqs != 1
            or md.num_ctx_tokens != rows
            or md.mamba_metadata.use_initial_states
        ):
            return None
        self.core.bind_caches(md)
        key = rows, id(md.kv_cache_manager)
        assert self.positions is not None
        positions = self.positions.reshape(-1, rows)[0]
        if key not in self.steps:
            step = self.core._context_step(md, positions, [rows], 0, 1, 0, rows, ids.device)
            ctx = step.context
            assert ctx is not None and step.context_indices is not None
            # Runtime metadata is rebuilt outside the graph. Keep the exact
            # shape's tensors alive and update request-specific values in place.
            slots = step.state_slots.clone()
            self.steps[key] = step._replace(
                context=replace(
                    ctx,
                    starts=ctx.starts.clone(),
                    sequence_slots=slots,
                    has_initial_state=ctx.has_initial_state.clone(),
                ),
                context_indices=step.context_indices.clone(),
                state_slots=slots,
                block_offsets=step.block_offsets.clone(),
                positions=positions.clone(),
            )
        static = self.steps[key]
        static.state_slots.copy_(md.mamba_metadata.state_indices[:1])
        static.block_offsets.copy_(md.kv_cache_block_offsets[:, :1])
        static.positions.copy_(positions)
        return (*key, self.last_only)

    def _eager(self, ids: torch.Tensor) -> torch.Tensor:
        if get_current_replay_token() is not None and (key := self._key(ids)) is not None:
            # Capture the exact bucket when the runtime's break warmup did
            # not encounter a matching batch.
            if key not in self.graphs:
                graph = torch.cuda.CUDAGraph()
                graph.capture_begin(capture_error_mode="thread_local")
                try:
                    output = self._capture(ids)
                finally:
                    graph.capture_end()
                self.graphs[key] = graph, output
            graph, output = self.graphs[key]
            graph.replay()
            return output
        rows = self.metadata.num_tokens
        hidden = self.core.compute(self.metadata, ids[:rows], self.positions, self.verify)
        padding = ids.shape[0] - rows
        return F.pad(hidden, (0, 0, 0, padding)) if padding else hidden

    def _capture(self, ids: torch.Tensor) -> torch.Tensor:
        key = ids.shape[0], id(self.metadata.kv_cache_manager)
        step = self.steps[key]
        if self.last_only:
            return self.core.compute_last_context_row(ids, step)
        return self.core._compute_layers(self.core.embedding.index_select(0, ids), step)

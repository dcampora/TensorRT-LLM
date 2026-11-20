# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contiguous Gated DeltaNet inputs for the chunk prefill kernel."""

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels.gdn_pack import pack_gdn_input as _pack_gdn_input


class PackGdnInput(OpWrapper):
    """Pack the convolution output and both gates with one launch."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(why="single-token context", spec=dict(rows=1)),
        Cell(why="mixed context and decode slices", spec=dict(rows=135)),
        Cell(why="more than one prefill chunk", spec=dict(rows=1025)),
    )

    def __call__(self, qkv, a, b, key_heads, value_heads, head_dim):
        return _pack_gdn_input(qkv, a, b, key_heads, value_heads, head_dim)

    def reference(self, qkv, a, b, key_heads, value_heads, head_dim):
        key_width = key_heads * head_dim
        value_width = value_heads * head_dim
        q, k, v = qkv.split((key_width, key_width, value_width), dim=-1)
        return (
            q.reshape(-1, key_heads, head_dim).contiguous(),
            k.reshape(-1, key_heads, head_dim).contiguous(),
            v.reshape(-1, value_heads, head_dim).contiguous(),
            a.contiguous(),
            b.contiguous(),
        )


pack_gdn_input = PackGdnInput()

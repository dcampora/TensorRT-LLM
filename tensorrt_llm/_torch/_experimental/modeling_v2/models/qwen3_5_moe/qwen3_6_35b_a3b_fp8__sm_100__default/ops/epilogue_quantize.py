# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused projection input preparation following gated attention and experts."""

from torch.nn import functional as F

from .....catalog._op import Arch, Cell, OpWrapper
from ..kernels.epilogue_quantize import gated_quantize as _gated_quantize
from .quantize import _reference_quantize


class GatedQuantize(OpWrapper):
    """Round a gated epilogue to BF16 and quantize it with 128-wide FP8 scales."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(why="recurrent decode with a row-strided SiLU gate", spec=dict(kind="gdn", rows=4)),
        Cell(why="recurrent prefill with many rows", spec=dict(kind="gdn", rows=1024)),
        Cell(why="gated full attention", spec=dict(kind="attention", rows=32)),
        Cell(why="draft attention on nonconsecutive rows", spec=dict(kind="attention", rows=96)),
        Cell(why="shared expert SwiGLU", spec=dict(kind="swiglu", rows=129)),
    )

    def __call__(self, x, kind, gate=None, weight=None, eps=1e-6, pick_rows=None):
        return _gated_quantize(x, gate, weight, eps, kind, pick_rows)

    def reference(self, x, kind, gate=None, weight=None, eps=1e-6, pick_rows=None):
        if pick_rows is not None:
            x = x.index_select(0, pick_rows)
            gate = gate.index_select(0, pick_rows)
        if kind == "gdn":
            heads = x.float().reshape(x.shape[0], -1, 128)
            y = F.rms_norm(heads, (128,), weight.float(), eps).reshape_as(x)
            y = y * F.silu(gate.float())
        elif kind == "attention":
            y = x.float() * gate.float().sigmoid()
        else:
            gate, up = x.float().chunk(2, dim=1)
            y = F.silu(gate) * up
        return _reference_quantize(y.bfloat16())


gated_quantize = GatedQuantize()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen's gated FP8 GQA over TensorRT-LLM paged KV."""

from typing import NamedTuple

import torch
import triton
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.prefill import trtllm_batch_context_with_kv_cache

from .....catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul
from ..kernels import paged_attention as k
from ..kernels import prefill_rope
from ..weights import BF16, FP8


class PrepareQkv(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell("nonconsecutive pages and one partially written page", {"page_size": 32}),
        Cell("multiple requests sharing one pool", {"page_size": 64}),
        Cell("tiled prefill over strided pages", {"page_size": 32, "tiled": True}),
    )

    def __call__(self, qkv, qnorm, knorm, rope, positions, requests, kv, table):
        rows = qkv.shape[0]
        q = torch.empty((rows, 16, 256), dtype=FP8, device=qkv.device)
        gate = torch.empty((rows, 4096), dtype=BF16, device=qkv.device)
        args = (
            qkv,
            qnorm,
            knorm,
            rope,
            positions,
            requests,
            q,
            kv,
            gate,
            table,
            table.stride(0),
            table.stride(1),
            kv.shape[-2],
            kv.stride(1),
        )
        if rows >= 512:
            tile, warps = 4, 2
            query_blocks = triton.cdiv(rows * 16, tile)
            cache_blocks = triton.cdiv(rows * 2, tile)
            prefill_rope.qkv_rope[(query_blocks + 2 * cache_blocks,)](
                *args, rows, tile, warps, query_blocks, cache_blocks, num_warps=warps
            )
        else:
            k.qkv_rope[(rows, 20)](*args, num_warps=4)
        return q, gate

    def reference(self, qkv, qnorm, knorm, rope, positions, requests, kv, table):
        pos = positions.long()
        batch = qkv.shape[0]
        q, gate = qkv[:, :8192].reshape(batch, 16, 512).chunk(2, dim=-1)
        key, value = qkv[:, 8192:].reshape(batch, 4, 256).chunk(2, dim=1)
        c, s = rope[pos, None, :].chunk(2, -1)

        def rotate(x, w):
            x = x.float()
            x = x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt() * (1 + w.float())
            x = x.to(BF16).float()
            lo, hi, tail = x.split((32, 32, 192), dim=-1)
            return torch.cat((lo * c - hi * s, hi * c + lo * s, tail), -1).to(BF16).to(FP8)

        q = rotate(q, qnorm)
        key = rotate(key, knorm)
        page_size = kv.shape[-2]
        # The reference interprets the same physical page offsets as the kernel.
        pages = kv.as_strided(
            ((kv.shape[0] - 1) * kv.stride(0) // kv.stride(1) + 2, 2, page_size, 256),
            (kv.stride(1), *kv.stride()[-3:]),
        )
        keys = table[requests.long(), 0, pos // page_size].long()
        values = table[requests.long(), 1, pos // page_size].long()
        pages[keys, :, pos % page_size] = key
        pages[values, :, pos % page_size] = value.to(FP8)
        return q, gate.contiguous().reshape(batch, 4096)


prepare_qkv = PrepareQkv()


def paged_decode_lengths(step, *, min_batch=16) -> torch.Tensor | None:
    batch = step.batch - step.contexts
    if batch < min_batch or step.width > 4:
        return None
    tails = step.positions[step.context_tokens + step.width - 1 :: step.width]
    return tails.to(torch.int32) + 1


class PagedContextInputs(NamedTuple):
    query_offsets: torch.Tensor
    kv_lengths: torch.Tensor
    kv_offsets: torch.Tensor


def paged_context_inputs(step) -> PagedContextInputs | None:
    if not step.contexts or step.max_context < 128:
        return None
    query_offsets = step.cu.to(torch.int32)
    tails = query_offsets[1:].long() - 1
    kv_lengths = step.positions[tails].to(torch.int32) + 1
    kv_offsets = torch.nn.functional.pad(kv_lengths.cumsum(dim=0, dtype=torch.int32), (1, 0))
    return PagedContextInputs(query_offsets, kv_lengths, kv_offsets)


def _role_pages(kv):
    # The block table indexes K and V role pages independently. The pool
    # view already points at this layer's first K page.
    page_size = kv.shape[-2]
    stride = kv.stride(1)
    return kv.as_strided(
        ((kv.shape[0] - 1) * kv.stride(0) // stride + 2, 2, page_size, 256),
        (stride, *kv.stride()[-3:]),
    )


def _native_context(q, kv, table, step, workspace, metadata):
    pages = _role_pages(kv)
    return trtllm_batch_context_with_kv_cache(
        q[: step.context_tokens],
        (pages, pages),
        workspace,
        table[: step.contexts],
        metadata.kv_lengths,
        step.max_context,
        step.capacity,
        0.0625,
        1.0,
        step.contexts,
        metadata.query_offsets,
        metadata.kv_offsets,
        out_dtype=BF16,
        uses_shared_paged_kv_idx=False,
    ).reshape(step.context_tokens, 4096)


def _native_decode(q, kv, table, step, workspace, lengths):
    pages = _role_pages(kv)
    return trtllm_batch_decode_with_kv_cache(
        q,
        (pages, pages),
        workspace,
        table[step.contexts : step.batch],
        lengths,
        step.capacity,
        bmm1_scale=0.0625,
        out_dtype=BF16,
        backend="trtllm-gen",
        q_len_per_req=step.width,
        uses_shared_paged_kv_idx=False,
    ).reshape(-1, 4096)


class GatedAttention(OpWrapper):
    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(
            "packed continuations with context and generation together",
            {"context": True, "width": 1},
        ),
        Cell(
            "native FP8 context with cached prefix and a decode row",
            {"context": True, "width": 1, "native": True},
        ),
        Cell("split-KV decode across nonconsecutive cache pages", {"context": False, "width": 1}),
        Cell("multi-token verification grouped by request", {"context": False, "width": 3}),
        Cell(
            "native FP8 decode with independent K and V pages",
            {"context": False, "width": 3, "native_decode": True},
        ),
        Cell(
            "local attention for a short prefix and native FMHA for a long prefix",
            {
                "context": False,
                "width": 4,
                "native_decode": True,
                "keep_short_local": True,
                "batch": 10,
            },
        ),
    )

    def __call__(
        self,
        q,
        gate,
        kv,
        table,
        step,
        output_scale,
        workspace=None,
        paged_context=None,
        paged_decode=None,
        keep_short_local=False,
    ):
        ctx = step.context_tokens
        m = q.shape[0]
        quantized = output_scale is not None
        attention = torch.empty((m, 4096), dtype=FP8 if quantized else BF16, device=q.device)
        strides = (table.stride(0), table.stride(1), kv.shape[-2], kv.stride(1))
        if ctx:
            if paged_context is not None and kv.shape[-2] == 32:
                y = _native_context(q, kv, table, step, workspace, paged_context)
            else:
                y = torch.empty((ctx, 4096), dtype=BF16, device=q.device)
                longest = step.max_context
                bq = 128 if longest >= 1024 else 64 if longest > 128 else 32
                k.attention[(triton.cdiv(longest, bq // 8), 2, step.contexts)](
                    q,
                    kv,
                    y,
                    None,
                    step.cu,
                    step.positions,
                    step.request_rows,
                    table,
                    *strides,
                    True,
                    bq,
                    128,
                    0,
                    0,
                    448.0,
                    num_warps=8 if bq == 128 else 4,
                    num_stages=2,
                )
            k.attn_gate[(triton.cdiv(y.numel(), 1024),)](
                y,
                gate,
                attention,
                y.numel(),
                1024,
                output_scale,
                quantized,
            )
        if m > ctx:
            q, gate = q[ctx:], gate[ctx:]
            native = None
            if paged_decode is not None and kv.shape[-2] == 32:
                native = _native_decode(q, kv, table, step, workspace, paged_decode)
                if step.batch - step.contexts <= 32 and not keep_short_local:
                    k.attn_gate[(triton.cdiv(native.numel(), 1024),)](
                        native, gate, attention[ctx:], native.numel(), 1024, output_scale, quantized
                    )
                    return attention
            positions, requests = step.positions[ctx:], step.request_rows[ctx:]
            rows = m - ctx
            width = step.width
            short_tile = rows <= width * 8
            if short_tile:
                split, block, warps = 64, 64, 4
            else:
                split = max(256, triton.next_power_of_2(step.capacity) // 32)
                block, warps = 128, 8
            # Keep the same local reduction for short prefixes after a request
            # moves into a larger decode graph. Long prefixes use native FMHA.
            local_limit = 256 if native is not None else 0
            splits = triton.cdiv(min(step.capacity, local_limit or step.capacity), split)
            part = torch.empty((rows, 16, splits, 256), dtype=torch.float32, device=q.device)
            stats = torch.empty((rows, 16, splits, 2), dtype=torch.float32, device=q.device)
            query_tile = max(16, triton.next_power_of_2(width * 8))
            k.attention[(splits, 2, rows // width)](
                q,
                kv,
                part,
                stats,
                None,
                positions,
                requests,
                table,
                *strides,
                False,
                query_tile,
                block,
                split,
                splits,
                448.0,
                width,
                local_limit,
                num_warps=warps,
                num_stages=2,
            )
            if local_limit:
                k.mixed_decode_gate[(rows * 4,)](
                    part,
                    stats,
                    native,
                    positions,
                    gate,
                    output_scale,
                    attention[ctx:],
                    width,
                    local_limit,
                    quantized,
                    num_warps=4,
                )
                return attention
            k.attn_reduce[(rows * 16,)](
                part,
                stats,
                positions,
                attention[ctx:],
                splits,
                split,
                triton.next_power_of_2(splits),
                gate,
                output_scale,
                quantized,
                True,
            )
        return attention

    def reference(
        self,
        q,
        gate,
        kv,
        table,
        step,
        output_scale,
        workspace=None,
        paged_context=None,
        paged_decode=None,
        keep_short_local=False,
    ):
        rows = []
        page_size = kv.shape[-2]
        pages = _role_pages(kv).float()
        with true_fp32_matmul():
            for query, row, position in zip(
                q.float(), step.request_rows.tolist(), step.positions.tolist()
            ):
                pos = torch.arange(position + 1, device=q.device)
                key_page = table[row, 0, pos // page_size].long()
                value_page = table[row, 1, pos // page_size].long()
                key = (
                    pages[key_page, :, pos % page_size].transpose(0, 1).repeat_interleave(8, dim=0)
                )
                val = (
                    pages[value_page, :, pos % page_size]
                    .transpose(0, 1)
                    .repeat_interleave(8, dim=0)
                )
                probability = torch.bmm(query[:, None], key.transpose(-1, -2)) * (256**-0.5)
                rows.append(torch.bmm(probability.softmax(-1), val).reshape(-1))
        y = torch.stack(rows).to(BF16).float() * gate.float().sigmoid()
        y = y.to(BF16)
        if output_scale is not None:
            y = (y.float() / output_scale).clamp(-448, 448).to(FP8)
        return y


gated_attention = GatedAttention()

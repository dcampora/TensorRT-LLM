# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Causal convolution and FlashInfer Gated DeltaNet scan against runtime state pools."""

from dataclasses import dataclass

import torch
from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose
from flashinfer.gdn_kernels.gdn_decode_bf16_state import gated_delta_rule_mtp
from flashinfer.gdn_prefill import chunk_gated_delta_rule

from .verify_conv import verify_conv


@dataclass(frozen=True)
class RecurrentStep:
    """State and sequence views shared by all recurrent layers of a step."""

    context_tokens: int
    decode_width: int
    verify: bool
    context_slots: torch.Tensor
    decode_slots: torch.Tensor
    initial_slots: torch.Tensor | None
    has_initial_state: torch.Tensor | None
    cu_seqlens: torch.Tensor | None
    cu_seqlens_long: torch.Tensor | None

    @classmethod
    def from_metadata(cls, metadata, spec_metadata=None):
        """Bind the context-then-generation order supplied by the scheduler."""
        mamba = metadata.mamba_metadata
        count = metadata.num_contexts
        batch = metadata.seq_lens.shape[0]
        slots = mamba.state_indices[:batch]
        context_slots = slots[:count]
        if count:
            has_initial_state = mamba.has_initial_states[:count]
            initial_slots = torch.where(has_initial_state, context_slots, -1)
            cu_seqlens = mamba.query_start_loc[: count + 1]
            cu_seqlens_long = mamba.query_start_loc_long[: count + 1]
            decode_slots = mamba.state_indices_decode[: batch - count]
        else:
            has_initial_state = initial_slots = cu_seqlens = cu_seqlens_long = None
            decode_slots = slots
        return cls(
            metadata.num_ctx_tokens,
            spec_metadata.runtime_draft_len + 1 if spec_metadata is not None else 1,
            spec_metadata is not None and metadata.kv_cache_manager.is_speculative(),
            context_slots,
            decode_slots,
            initial_slots,
            has_initial_state,
            cu_seqlens,
            cu_seqlens_long,
        )


def _split_qkv(x, num_key_heads, num_value_heads, head_dim):
    key_width = num_key_heads * head_dim
    value_width = num_value_heads * head_dim
    q, k, v = x.split((key_width, key_width, value_width), dim=-1)
    return (
        q.view(-1, num_key_heads, head_dim),
        k.view(-1, num_key_heads, head_dim),
        v.view(-1, num_value_heads, head_dim),
    )


def gdn_scan(qkv, a, b, conv_weight, a_log, dt_bias, cache, step, key_heads, value_heads, head_dim):
    """Advance the convolution and recurrent state for the current packed rows."""
    out = torch.empty((qkv.shape[0], value_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    context = step.context_tokens
    if context:
        x = qkv[:context]
        convolved = torch.empty_like(x, memory_format=torch.contiguous_format)
        torch.ops.trtllm.causal_conv1d_fwd(
            x.T,
            conv_weight,
            None,
            cache.conv,
            step.cu_seqlens,
            step.context_slots,
            step.has_initial_state,
            True,
            -1,
            convolved.T,
        )
        q, k, v = _split_qkv(convolved, key_heads, value_heads, head_dim)
        chunk_gated_delta_rule(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            a[:context].contiguous(),
            b[:context].contiguous(),
            initial_state=cache.temporal,
            output_final_state=True,
            cu_seqlens=step.cu_seqlens_long,
            use_qk_l2norm_in_kernel=True,
            output=out[:context],
            output_state=cache.temporal,
            use_cp=False,
            state_indices=step.initial_slots,
            output_state_indices=step.context_slots,
            backend="flashinfer",
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
            beta_is_logit=True,
        )
    if context < qkv.shape[0]:
        x = qkv[context:]
        width = step.decode_width
        if step.verify:
            x = verify_conv(
                x, conv_weight, cache.conv, step.decode_slots, cache.intermediate_conv_window, width
            )
        else:
            x = x.contiguous()
            torch.ops.trtllm.causal_conv1d_update(
                x.unsqueeze(-1), cache.conv, conv_weight, None, True, None, step.decode_slots, -1
            )
        batch_size = step.decode_slots.numel()
        q, k, v = _split_qkv(x, key_heads, value_heads, head_dim)
        q = q.view(batch_size, width, key_heads, head_dim)
        k = k.view_as(q)
        output = out[context:].view(batch_size, width, value_heads, head_dim)
        v = v.view_as(output)
        a = a[context:].contiguous().view(batch_size, width, value_heads)
        b = b[context:].contiguous().view(batch_size, width, value_heads)
        if step.verify:
            gated_delta_rule_mtp(
                A_log=a_log,
                a=a,
                dt_bias=dt_bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=q,
                k=k,
                v=v,
                b=b,
                initial_state_source=cache.temporal,
                initial_state_indices=step.decode_slots,
                intermediate_states_buffer=cache.intermediate_ssm[:batch_size],
                disable_state_update=True,
                use_qk_l2norm_in_kernel=True,
                scale=head_dim**-0.5,
                output=output,
            )
        else:
            gated_delta_rule_decode_pretranspose(
                q,
                k,
                v,
                None,
                a_log,
                a,
                dt_bias,
                b,
                output=output,
                initial_state=cache.temporal,
                initial_state_indices=step.decode_slots,
                backend="flashinfer",
            )
    return out

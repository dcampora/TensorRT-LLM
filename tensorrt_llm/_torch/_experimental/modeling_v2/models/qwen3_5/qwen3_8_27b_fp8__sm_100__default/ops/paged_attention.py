# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arguments to the catalog's native dense paged attention operation."""

from typing import TYPE_CHECKING

import torch

from .....catalog.attention.thop_attention import thop_attention

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata


def attention_arguments(md: "TrtllmAttentionMetadata", predicted: int) -> dict:
    """Extract the runtime's batch state once for all full-attention layers."""
    return dict(
        sequence_length=md.kv_lens_cuda_runtime,
        host_past_key_value_lengths=md.kv_lens_runtime,
        host_total_kv_lens=md.host_total_kv_lens,
        context_lengths=md.prompt_lens_cuda_runtime,
        host_context_lengths=md.prompt_lens_cpu_runtime,
        host_request_types=md.host_request_types_runtime,
        kv_cache_block_offsets=md.kv_cache_block_offsets,
        host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
        workspace=md.effective_workspace,
        tokens_per_block=md.tokens_per_block,
        max_num_requests=md.max_num_requests,
        max_context_length=md.max_context_length,
        max_seq_len=md.max_seq_len,
        attention_window_size=md.max_seq_len,
        num_contexts=md.num_contexts,
        num_ctx_tokens=md.num_ctx_tokens,
        trtllm_gen_jit_warmup=md.trtllm_gen_jit_warmup,
        use_paged_context_fmha=md.use_paged_context_fmha,
        beam_width=md.effective_beam_width,
        cache_indirection=md.cache_indirection,
        block_ids_per_seq=md.block_ids_per_seq,
        max_context_q_len_override=md.max_context_q_len_override,
        predicted_tokens_per_seq=predicted,
        max_num_sequences=md.max_num_sequences,
    )


def paged_kv_view(
    md: "TrtllmAttentionMetadata", layer: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return the BF16 physical pages and table for the model layer."""
    manager = md.kv_cache_manager
    pages = manager.get_buffers(layer, kv_layout="HND")
    assert pages is not None
    scale = manager.get_layer_page_index_scale(layer)
    # Even scales index contiguous K/V pairs. Other slots use a strided view.
    divisor = 2 if scale % 2 == 0 else scale
    local_layer = manager.layer_offsets[layer]
    pool = int(md.host_kv_cache_pool_mapping[local_layer, 0])
    table = md.kv_cache_block_offsets[pool, :, 0, :]
    return pages, table, divisor


def paged_attention(qkv: torch.Tensor, layer: int, arguments: dict) -> torch.Tensor:
    """Append packed, pre-rotated keys and values, then compute causal GQA."""
    rows = qkv.shape[0]
    return thop_attention(
        q=qkv,
        k=None,
        v=None,
        output=qkv.new_empty((rows, 6144)),
        local_layer_idx=layer,
        num_heads=24,
        num_kv_heads=4,
        head_size=256,
        output_sf=None,
        is_fused_qkv=True,
        update_kv_cache=True,
        attention_input_type=0,
        is_mla_enable=False,
        mask_type=1,
        quant_mode=0,
        q_scaling=1.0,
        attention_sinks=None,
        position_embedding_type=0,
        rotary_inv_freq=None,
        rotary_cos_sin=None,
        rope_dim=0,
        rope_base=10000.0,
        rope_scale_type=0,
        rope_scale=1.0,
        rope_short_m_scale=1.0,
        rope_long_m_scale=1.0,
        rope_max_positions=1024,
        rope_original_max_positions=1024,
        out_scale=None,
        kv_scale_orig_quant=None,
        kv_scale_quant_orig=None,
        latent_cache=None,
        q_pe=None,
        q_lora_rank=None,
        kv_lora_rank=None,
        qk_nope_head_dim=None,
        qk_rope_head_dim=None,
        v_head_dim=None,
        rope_append=None,
        chunked_prefill_buffer_batch_size=1,
        attention_chunk_size=None,
        softmax_stats_tensor=None,
        sparse_kv_indices=None,
        sparse_kv_offsets=None,
        sparse_attn_indices=None,
        sparse_attn_offsets=None,
        sparse_attn_indices_block_size=0,
        mrope_rotary_cos_sin=None,
        mrope_position_deltas=None,
        helix_position_offsets=None,
        helix_is_inactive_rank=None,
        is_spec_decoding_enabled=False,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        spec_decoding_generation_lengths=None,
        spec_decoding_position_offsets=None,
        spec_decoding_packed_mask=None,
        spec_decoding_bl_tree_mask_offset=None,
        spec_decoding_bl_tree_mask=None,
        spec_bl_tree_first_sparse_mask_offset_kv=None,
        **arguments,
    )

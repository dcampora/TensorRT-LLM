# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind packed gated GQA to the runtime's BF16 paged attention."""

from .....catalog.attention.thop_attention import thop_attention


def attention_step(metadata):
    """Read the attention arguments shared by every full-attention layer."""
    num_generations = metadata.seq_lens.shape[0] - metadata.num_contexts
    num_gen_tokens = metadata.num_tokens - metadata.num_ctx_tokens
    # SM100 uses ordinary causal FMHA with multiple query rows for a linear draft.
    predicted_tokens = num_gen_tokens // num_generations if num_generations else 1
    return dict(
        sequence_length=metadata.kv_lens_cuda_runtime,
        host_past_key_value_lengths=metadata.kv_lens_runtime,
        host_total_kv_lens=metadata.host_total_kv_lens,
        context_lengths=metadata.prompt_lens_cuda_runtime,
        host_context_lengths=metadata.prompt_lens_cpu_runtime,
        host_request_types=metadata.host_request_types_runtime,
        max_context_q_len_override=metadata.max_context_q_len_override,
        kv_cache_block_offsets=metadata.kv_cache_block_offsets,
        host_kv_cache_pool_pointers=metadata.host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping=metadata.host_kv_cache_pool_mapping,
        cache_indirection=metadata.cache_indirection,
        workspace=metadata.effective_workspace,
        block_ids_per_seq=metadata.block_ids_per_seq,
        tokens_per_block=metadata.tokens_per_block,
        max_num_requests=metadata.max_num_requests,
        max_context_length=metadata.max_context_length,
        max_seq_len=metadata.max_seq_len,
        attention_window_size=metadata.max_seq_len,
        use_paged_context_fmha=metadata.use_paged_context_fmha,
        num_contexts=metadata.num_contexts,
        num_ctx_tokens=metadata.num_ctx_tokens,
        trtllm_gen_jit_warmup=metadata.trtllm_gen_jit_warmup,
        max_num_sequences=metadata.max_num_sequences,
        predicted_tokens_per_seq=predicted_tokens,
    )


def paged_attention(qkv, output, local_layer_idx, num_heads, num_kv_heads, head_dim, step):
    """Append packed QKV's K/V and attend to the full causal sequence."""
    return thop_attention(
        q=qkv,
        k=None,
        v=None,
        output=output,
        output_sf=None,
        kv_scale_orig_quant=None,
        kv_scale_quant_orig=None,
        out_scale=None,
        rotary_inv_freq=None,
        rotary_cos_sin=None,
        latent_cache=None,
        q_pe=None,
        attention_sinks=None,
        is_fused_qkv=True,
        update_kv_cache=True,
        local_layer_idx=local_layer_idx,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        beam_width=1,
        mask_type=1,
        quant_mode=0,
        q_scaling=1.0,
        position_embedding_type=0,
        rope_dim=0,
        rope_base=10000.0,
        rope_scale_type=0,
        rope_scale=1.0,
        rope_short_m_scale=1.0,
        rope_long_m_scale=1.0,
        rope_max_positions=1024,
        rope_original_max_positions=1024,
        attention_input_type=0,
        is_mla_enable=False,
        chunked_prefill_buffer_batch_size=1,
        q_lora_rank=None,
        kv_lora_rank=None,
        qk_nope_head_dim=None,
        qk_rope_head_dim=None,
        v_head_dim=None,
        rope_append=None,
        mrope_rotary_cos_sin=None,
        mrope_position_deltas=None,
        helix_position_offsets=None,
        helix_is_inactive_rank=None,
        attention_chunk_size=None,
        softmax_stats_tensor=None,
        is_spec_decoding_enabled=False,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        spec_decoding_generation_lengths=None,
        spec_decoding_position_offsets=None,
        spec_decoding_packed_mask=None,
        spec_decoding_bl_tree_mask_offset=None,
        spec_decoding_bl_tree_mask=None,
        spec_bl_tree_first_sparse_mask_offset_kv=None,
        sparse_kv_indices=None,
        sparse_kv_offsets=None,
        sparse_attn_indices=None,
        sparse_attn_offsets=None,
        sparse_attn_indices_block_size=0,
        **step,
    )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused MTP-Eagle step with the checkpoint's BF16 draft layer.

The proposal layer and shortlist are adapted from aegis-engines
speculation.py. The step returns TensorRT-LLM's speculative sampler contract
and uses the runtime's paged block tables for the draft layer.
"""

from dataclasses import replace

import torch
from torch import nn

from tensorrt_llm._torch.pyexecutor.sampler.ops import custom as fused_sampling
from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as flashinfer_sampling
from tensorrt_llm._torch.speculative.interface import INVALID_PROMPT_LOOKAHEAD_TOKEN
from tensorrt_llm._torch.speculative.spec_tail_kernels import strict_accept

from .ops.attention import gated_attention, prepare_qkv
from .ops.draft import draft_input_norm, draft_moe
from .ops.moe import residual_rms_norm
from .weights import EXPERT_WIDTH, HIDDEN, NUM_EXPERTS, VOCAB, parameter


class DraftLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_norm = parameter((HIDDEN,))
        self.hidden_norm = parameter((HIDDEN,))
        self.fc = parameter((HIDDEN, 2 * HIDDEN))
        self.norm = parameter((HIDDEN,))
        self.before_attention = parameter((HIDDEN,))
        self.before_moe = parameter((HIDDEN,))
        self.qkv = parameter((9216, HIDDEN))
        self.output = parameter((HIDDEN, 4096))
        self.qnorm = parameter((256,))
        self.knorm = parameter((256,))
        self.gate_up = parameter((NUM_EXPERTS, 2 * EXPERT_WIDTH, HIDDEN))
        self.down = parameter((NUM_EXPERTS, HIDDEN, EXPERT_WIDTH))
        self.router = parameter((NUM_EXPERTS + 1, HIDDEN))
        self.shared_gate = parameter((2 * EXPERT_WIDTH, HIDDEN))
        self.shared_down = parameter((HIDDEN, EXPERT_WIDTH))

    def forward(self, ids, hidden, step, rows, embedding, kv, table, rope, bins, side):
        prepared = draft_input_norm(ids, embedding, hidden, self.embedding_norm, self.hidden_norm)
        x = torch.mm(prepared, self.fc.T)
        residual = torch.empty_like(x)
        x = residual_rms_norm(x, residual, self.before_attention, False, True)
        q, gate = prepare_qkv(
            torch.mm(x, self.qkv.T),
            self.qnorm,
            self.knorm,
            rope,
            step.positions,
            step.request_rows,
            kv,
            table,
        )
        if rows is not None:
            # Every row publishes K/V. Only the last accepted row of a request
            # needs an attention output and routed expert evaluation.
            q = q.view(torch.uint8)[rows].view(q.dtype)
            gate, residual = gate[rows], residual[rows]
            step = replace(
                step,
                positions=step.positions[rows],
                request_rows=step.request_rows[rows],
                context_tokens=0,
                contexts=0,
                width=1,
                max_context=0,
                cu=None,
            )
        attention = gated_attention(q, gate, kv, table, step, None)
        x = torch.mm(attention, self.output.T)
        x = residual_rms_norm(x, residual, self.before_moe, True, True)
        x = draft_moe(
            x, self.router, self.gate_up, self.down, self.shared_gate, self.shared_down, bins, side
        )
        return residual_rms_norm(x, residual, self.norm, True, False)


def sample_verified(hidden, head, spec):
    if spec.is_all_greedy_sample:
        return head.projector.top1(hidden)
    logits = head(hidden).float()
    rows = logits.shape[0]
    temperatures = spec.temperatures[:rows]
    top_ks, top_ps, min_ps = spec.top_ks[:rows], spec.top_ps[:rows], spec.min_ps[:rows]
    seed, offset = spec.seeds[:rows], spec.offsets[:rows]
    mode = spec.advanced_sampling_mode
    if mode.is_fused:
        return fused_sampling.fused_sample_from_logits(
            logits,
            temperatures,
            top_ks,
            top_ps,
            min_ps,
            seed=seed,
            offset=offset,
        )
    top_ks, top_ps = flashinfer_sampling.resolve_advanced_sampling_filters(mode, top_ks, top_ps)
    return flashinfer_sampling.sample_from_logits_op(
        logits,
        temperatures,
        top_ks,
        top_ps,
        seed=seed,
        offset=offset,
    )


def mtp_step(input_ids, hidden, step, md, spec, draft_layer, head, decoder):
    """Accept the prior proposals and draft from the accepted hidden rows."""
    # On a generation-only step every row is a verification position,
    # in the same order as the runtime's gather indices.
    selected = hidden[spec.gather_ids] if step.contexts else hidden
    target = sample_verified(selected, head, spec)
    depth = spec.runtime_draft_len
    batch = step.batch
    num_gens = batch - step.contexts
    if depth == 0:
        accepted = target.to(torch.int32).reshape(batch, 1)
        lengths = torch.ones(batch, dtype=torch.int32, device=target.device)
        next_drafts = accepted[:, :0]
        next_tokens = accepted
        if num_gens:
            md.kv_cache_manager.update_mamba_states(md, lengths, md.mamba_metadata.state_indices)
    else:
        proposed = (
            spec.draft_tokens.reshape(num_gens, depth)
            if num_gens
            else target.new_empty((0, depth), dtype=torch.int32)
        )
        accepted, lengths = strict_accept(target, proposed, step.contexts, batch)
        if num_gens:
            md.kv_cache_manager.update_mamba_states(md, lengths, md.mamba_metadata.state_indices)
        next_drafts = draft_chain(
            input_ids,
            hidden,
            step,
            md,
            spec,
            draft_layer,
            head,
            decoder,
            accepted,
            lengths,
        )
        last = accepted[spec.batch_indices_cuda[:batch], lengths - 1]
        next_tokens = torch.cat((last[:, None], next_drafts), dim=-1)
    return {
        "logits": None,
        "new_tokens": accepted,
        "new_tokens_lens": lengths,
        "next_draft_tokens": next_drafts,
        "next_new_tokens": next_tokens,
    }


def draft_chain(input_ids, hidden, step, md, spec, draft_layer, head, decoder, accepted, lengths):
    context_tokens = step.context_tokens
    contexts = step.contexts
    gens = step.batch - contexts
    depth = spec.runtime_draft_len
    gathered = spec.gather_ids[:contexts]
    ids = accepted[contexts:].flatten()
    if contexts:
        prompt = input_ids[:context_tokens].to(torch.int32)
        shifted = torch.empty_like(prompt)
        shifted[:-1] = prompt[1:]
        tail = accepted[:contexts, 0]
        lookahead = spec.context_prompt_lookahead_tokens
        if lookahead is not None:
            lookahead = lookahead[:contexts]
            tail = torch.where(lookahead != INVALID_PROMPT_LOOKAHEAD_TOKEN, lookahead, tail)
        shifted[gathered] = tail
        ids = torch.cat((shifted, ids)) if gens else shifted

    gen_rows = (
        context_tokens + spec.batch_indices_cuda[:gens] * (depth + 1) + lengths[contexts:] - 1
    )
    rows = torch.cat((gathered, gen_rows)) if contexts else gen_rows
    positions = step.positions[rows] + 1
    requests = decoder.requests[: step.batch]
    next_step = replace(
        step,
        positions=positions,
        request_rows=requests,
        contexts=0,
        context_tokens=0,
        max_context=0,
        width=1,
        cu=None,
    )
    kv, pool, separate = decoder.mtp_cache
    table = (md.draft_kv_cache_block_offsets if separate else md.kv_cache_block_offsets)[pool]
    drafts = []
    vocab = min(VOCAB, 65536 if step.batch >= 96 else 131072)
    args = (decoder.embed_tokens.weight, kv, table, decoder.rope, decoder.bins, decoder.side)
    hidden = draft_layer(ids, hidden, step, rows, *args)
    tokens = head.projector.top1(hidden, vocab).int()
    drafts.append(tokens)
    for _ in range(1, depth):
        hidden = draft_layer(tokens, hidden, next_step, None, *args)
        tokens = head.projector.top1(hidden, vocab).int()
        drafts.append(tokens)
        next_step = replace(next_step, positions=next_step.positions + 1)
    return torch.stack(drafts, dim=-1)

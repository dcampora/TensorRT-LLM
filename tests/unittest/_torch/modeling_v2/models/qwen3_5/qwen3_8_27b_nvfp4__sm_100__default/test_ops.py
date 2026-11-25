# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Operator parity over the Qwen3.8 decode and context layouts."""

from itertools import accumulate

import flashinfer as fi
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_100__default.ops.attention import (  # noqa: E501
    apply_gate,
    fp8_qkv_partitions,
    prepare_qkv,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_100__default.ops.draft import (  # noqa: E501
    draft_features,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_100__default.ops.gdn import (
    Context,
    Replay,
    gated_rmsnorm,
    gdn_context,
    gdn_decode,
    gdn_verify,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_100__default.ops.paged import (  # noqa: E501
    Cache,
    append_kv,
    context_attention,
    decode_attention,
    paged_context_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_100__default.ops.projections import (  # noqa: E501
    fp8_quantize,
    fp8_requantize,
    packed_fp4_gemm,
    swiglu_fp4,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="The Qwen3.8 NVFP4 target requires SM100.",
)


def _random(*size: int) -> torch.Tensor:
    return torch.randn(size, device="cuda", dtype=torch.bfloat16)


def _assert_bf16(actual: torch.Tensor, expected: torch.Tensor, ulps: float = 1) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    expected = expected.float().flatten(1)
    delta = (actual.float().flatten(1) - expected).abs()
    scale = expected.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    eps = torch.finfo(torch.bfloat16).eps
    assert (delta <= ulps * eps * scale).all(), (delta / (eps * scale)).amax().item()


@pytest.mark.parametrize("cell", draft_features.CELLS)
def test_mtp_features(cell) -> None:
    torch.manual_seed(24)
    rows = cell.spec["rows"]
    ids = torch.arange(-1, rows - 1, device="cuda", dtype=torch.int32).clamp(max=63)
    embedding, hidden = _random(64, 5120), _random(rows, 5120)
    embedding_norm, hidden_norm = _random(5120), _random(5120)
    args = ids, embedding, hidden, embedding_norm, hidden_norm, 1.0e-6
    _assert_bf16(draft_features(*args), draft_features.reference(*args))


@pytest.mark.parametrize("cell", fp8_quantize.CELLS)
def test_static_fp8(cell) -> None:
    torch.manual_seed(9)
    x = _random(cell.spec["rows"], cell.spec["width"])
    scale = torch.tensor(0.037, device="cuda")
    actual = fp8_quantize(x, scale)
    expected = fp8_quantize.reference(x, scale)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("cell", fp8_requantize.CELLS)
def test_fp8_requantize(cell) -> None:
    torch.manual_seed(10)
    w = (_random(cell.spec["rows"], cell.spec["width"]) * 105).to(torch.float8_e4m3fn)
    src, dst = (torch.tensor(s, device="cuda") for s in (0.00063, 0.0012))
    actual = fp8_requantize(w, src, dst)
    expected = fp8_requantize.reference(w, src, dst)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("cell", swiglu_fp4.CELLS)
def test_swiglu_fp4(cell) -> None:
    torch.manual_seed(11)
    x = _random(cell.spec["rows"], 2 * cell.spec["width"])
    scale = torch.tensor(39.125, device="cuda")
    actual, actual_scale = swiglu_fp4(x, scale)
    expected, expected_scale = swiglu_fp4.reference(x, scale)
    # Scale bytes for padded rows have no corresponding activation values.
    from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.nvfp4_gemm import Nvfp4Gemm

    decode = Nvfp4Gemm._dequantize
    a = decode(actual, actual_scale.view(torch.uint8)).bfloat16()
    b = decode(expected, expected_scale.view(torch.uint8)).bfloat16()
    _assert_bf16(a, b)


@pytest.mark.parametrize("cell", packed_fp4_gemm.CELLS)
def test_fp4_gemm(cell) -> None:
    torch.manual_seed(12)
    m, k, n, down = (cell.spec[key] for key in ("m", "k", "n", "down"))
    alpha = torch.tensor([1.0 / 256], device="cuda")
    activations = _random(m, k)
    weight = _random(n, k)
    inv_scale = torch.tensor(16.0, device="cuda")
    a, a_scale = fi.fp4_quantize(activations, inv_scale, backend="cute-dsl")
    b, b_scale = fi.fp4_quantize(weight, inv_scale, backend="cute-dsl")
    args = a, b.T, a_scale, b_scale.T, alpha
    actual = packed_fp4_gemm(*args, down_decode=down)
    expected = packed_fp4_gemm.reference(*args, down_decode=down)
    _assert_bf16(actual, expected)


@pytest.mark.parametrize("cell", [cell for cell in prepare_qkv.CELLS if "rows" in cell.spec])
def test_qkv_and_rope(cell) -> None:
    torch.manual_seed(13)
    rows = cell.spec["rows"]
    x = _random(rows, 14336)
    pos = torch.arange(rows, device="cuda") + cell.spec["position"]
    angle = torch.outer(
        torch.arange(pos.numel() + cell.spec["position"], device="cuda").float(),
        1 / 10000000 ** (torch.arange(0, 64, 2, device="cuda").float() / 64),
    )
    table = torch.cat((angle.cos(), angle.sin()), dim=-1)
    args = x, _random(256) / 4, _random(256) / 4, table, pos, 1e-6
    for actual, expected in zip(prepare_qkv(*args), prepare_qkv.reference(*args), strict=True):
        _assert_bf16(actual, expected)


@pytest.mark.parametrize("cell", fp8_qkv_partitions.CELLS)
def test_fp8_qkv_partitions(cell) -> None:
    torch.manual_seed(42)
    rows = cell.spec["rows"]
    shape = (rows, 5120)
    packed = torch.randint(-16, 17, shape, device="cuda").to(torch.float8_e4m3fn)
    shape = (14336, 5120)
    weight = torch.randint(-16, 17, shape, device="cuda").to(torch.float8_e4m3fn)
    args = packed, weight.T.unsqueeze(0)
    torch.testing.assert_close(
        fp8_qkv_partitions(*args), fp8_qkv_partitions.reference(*args), rtol=0, atol=0
    )


@pytest.mark.parametrize("cell", apply_gate.CELLS)
def test_attention_gate(cell) -> None:
    torch.manual_seed(14)
    x, gate = (_random(cell.spec["rows"], 6144) for _ in range(2))
    _assert_bf16(apply_gate(x.clone(), gate), apply_gate.reference(x, gate))


@pytest.mark.parametrize("cell", gated_rmsnorm.CELLS)
def test_gated_norm(cell) -> None:
    torch.manual_seed(15)
    rows = cell.spec["rows"]
    x, z = (_random(rows, 6144) for _ in range(2))
    scale = torch.tensor(0.032, device="cuda") if cell.spec["pack"] else None
    args = x, z, _random(128), 1e-6, scale
    actual = gated_rmsnorm(*args, decode=cell.spec["pack"])
    expected = gated_rmsnorm.reference(*args, decode=cell.spec["pack"])
    _assert_bf16(actual.bfloat16(), expected.bfloat16())


def _gdn_inputs(rows: int, batch: int) -> tuple[torch.Tensor, ...]:
    slots = torch.arange(batch - 1, -1, -1, device="cuda", dtype=torch.int32) * 2
    return (
        _random(rows, 16384),
        _random(rows, 96) / 4,
        _random(10240, 4) / 8,
        _random(48) - 2,
        _random(48),
        _random(batch * 2, 3, 10240).transpose(1, 2),
        _random(batch * 2, 48, 128, 128).float() / 512,
        slots,
    )


def _check_recurrent(
    op, x, ba, weight, log_decay, bias, hist, state, slots, metadata, ulps
) -> None:
    ref_hist, ref_state = hist.clone(), state.clone()
    actual = op(x, ba, weight, log_decay, bias, hist, state, metadata)
    expected = op.reference(x, ba, weight, log_decay, bias, ref_hist, ref_state, metadata)
    _assert_bf16(actual, expected, ulps=ulps)
    torch.testing.assert_close(hist, ref_hist, rtol=0, atol=0)
    _assert_bf16(
        state[slots].bfloat16().flatten(0, 1), ref_state[slots].bfloat16().flatten(0, 1), ulps=ulps
    )


@pytest.mark.parametrize("cell", gdn_decode.CELLS)
def test_gdn_decode(cell) -> None:
    torch.manual_seed(16)
    batch = cell.spec["batch"]
    args = _gdn_inputs(batch, batch)
    _check_recurrent(gdn_decode, *args, args[-1], 1)


@pytest.mark.parametrize("cell", [cell for cell in gdn_verify.CELLS if not cell.spec.get("replay")])
def test_gdn_verify(cell) -> None:
    torch.manual_seed(22)
    batch, width = (cell.spec[key] for key in ("batch", "width"))
    args = _gdn_inputs(batch * width, batch)
    history, state = args[-3:-1]
    saved_history, saved_state = history.clone(), state.clone()
    conv = torch.empty((batch, width, *history.shape[1:]), dtype=history.dtype, device="cuda")
    ssm = torch.empty((batch, width, *state.shape[1:]), dtype=state.dtype, device="cuda")
    expected_conv, expected_ssm = torch.empty_like(conv), torch.empty_like(ssm)
    output = gdn_verify(*args, conv, ssm, width)
    expected = gdn_verify.reference(*args, expected_conv, expected_ssm, width)
    _assert_bf16(output, expected)
    _assert_bf16(ssm.flatten(0, 2).bfloat16(), expected_ssm.flatten(0, 2).bfloat16())
    torch.testing.assert_close(conv, expected_conv, atol=0, rtol=0)
    torch.testing.assert_close(history, saved_history, atol=0, rtol=0)
    torch.testing.assert_close(state, saved_state, atol=0, rtol=0)


@pytest.mark.parametrize("cell", [cell for cell in gdn_verify.CELLS if cell.spec.get("replay")])
def test_gdn_replay_acceptance(cell) -> None:
    torch.manual_seed(23)
    batch, width = (cell.spec[key] for key in ("batch", "width"))
    step_width = cell.spec.get("commit_width", width)
    history_size = cell.spec.get("commit_width", width + 2)
    replay_capacity = 2 * batch
    _, _, weight, decay, bias, conv, temporal, slots = _gdn_inputs(batch * width, batch)
    full_conv, full_state = conv.clone(), temporal.clone()
    conv_planes = (batch, step_width, *conv.shape[1:])
    verify_conv = conv.new_empty(conv_planes)
    full_conv_steps = conv.new_empty(conv_planes)
    full_ssm_steps = temporal.new_empty((batch, width, *temporal.shape[1:]))
    replay = Replay(
        slots.flip(0),
        torch.zeros(replay_capacity, dtype=torch.int32, device="cuda"),
        torch.zeros(replay_capacity, dtype=torch.int32, device="cuda"),
        temporal.new_empty((replay_capacity, 2, history_size, 48, 128)),
        temporal.new_empty((replay_capacity, 2, history_size, 16, 128)),
        temporal.new_empty((replay_capacity, 2, 48, history_size)),
    )
    batches = torch.arange(batch, device="cuda")
    state_slots = slots.long()
    dummy = batches % 2 == 1 if cell.spec.get("masked") else None
    active = ~dummy if dummy is not None else slice(None)
    replay_slots = replay.slots.long()[active]
    saved = temporal[state_slots[dummy]].clone() if dummy is not None else None
    widths = cell.spec.get("verify_widths", (width,) * 4)
    for offset, width in zip((1, 2, 0, 1), widths, strict=True):
        x, ba = _random(batch * width, 16384), _random(batch * width, 96)
        inputs = x, ba, weight, decay, bias
        actual = gdn_verify(
            *inputs, conv, temporal, slots, verify_conv, None, width, replay, dummy=dummy
        )
        expected = gdn_verify(
            *inputs, full_conv, full_state, slots, full_conv_steps, full_ssm_steps, width
        )
        actual = actual.view(batch, width, -1)
        expected = expected.view(batch, width, -1)
        torch.testing.assert_close(actual[active], expected[active], atol=0, rtol=0)
        torch.testing.assert_close(
            verify_conv[active, :width], full_conv_steps[active, :width], atol=0, rtol=0
        )
        if dummy is not None:
            torch.testing.assert_close(
                actual[dummy], torch.zeros_like(actual[dummy]), atol=0, rtol=0
            )
            torch.testing.assert_close(temporal[state_slots[dummy]], saved, atol=0, rtol=0)
        accepted = (batches + offset).to(torch.int32) % width + 1
        conv[state_slots[active]] = verify_conv[batches[active], accepted[active] - 1]
        full_conv[state_slots] = full_conv_steps[batches, accepted - 1]
        full_state[state_slots] = full_ssm_steps[batches, accepted - 1]
        prior = replay.accepted[replay_slots]
        checkpoint = prior + step_width > history_size
        accepted = accepted[active]
        replay.accepted[replay_slots] = torch.where(checkpoint, accepted, prior + accepted)
        bank = replay.banks[replay_slots]
        replay.banks[replay_slots] = torch.where(checkpoint, 1 - bank, bank)


@pytest.mark.parametrize("cell", gdn_context.CELLS)
def test_gdn_context(cell) -> None:
    torch.manual_seed(17)
    lengths = cell.spec["lengths"]
    args = _gdn_inputs(sum(lengths), len(lengths))
    counts = [(n + 63) // 64 for n in lengths]
    chunks = [(i, c) for i, n in enumerate(counts) for c in range(n)]
    inputs = (list(accumulate(lengths, initial=0)), chunks, list(accumulate(counts, initial=0)))
    ctx = Context(
        *(torch.tensor(x, device="cuda", dtype=torch.int32) for x in inputs),
        args[-1],
        torch.tensor(cell.spec["carried"], device="cuda"),
        len(lengths),
        max(lengths),
        64,
    )
    # The chunked solve rounds normalized Q/K, triangular factors and its
    # boundary-state dot products to BF16 before the final output rounding.
    _check_recurrent(gdn_context, *args, ctx, 4)


def _paged_cache(lengths: tuple[int, ...], head_init: bool) -> tuple[Cache, torch.Tensor]:
    blocks = [(n + 31) // 32 for n in lengths]
    all_pages = sum(blocks)
    shuffled = torch.randperm(all_pages, device="cuda", dtype=torch.int32)
    table = torch.zeros((len(lengths), max(blocks)), device="cuda", dtype=torch.int32)
    first = 0
    for i, count in enumerate(blocks):
        table[i, :count] = 2 * shuffled[first : first + count]
        first += count
    dims = all_pages, 2, 4, 32, 256
    pool = _random(*dims) if head_init else torch.zeros(dims, device="cuda", dtype=torch.bfloat16)
    return Cache(0, 0, 2, pool[:, 0], pool[:, 1]), table


@pytest.mark.parametrize("cell", [cell for cell in prepare_qkv.CELLS if "width" in cell.spec])
def test_qkv_writes_paged_cache(cell) -> None:
    torch.manual_seed(43)
    width = cell.spec["width"]
    rows = width * 2
    cache, table = _paged_cache((30 + width, 63 + width), False)
    ref = cache._replace(key=cache.key.clone(), value=cache.value.clone())
    standalone = cache._replace(key=cache.key.clone(), value=cache.value.clone())
    table = table.repeat_interleave(2, dim=0)[::2]
    positions = torch.cat([torch.arange(s, s + width, device="cuda") for s in (30, 63)])
    phi = _random(63 + width, 32).float()
    rope = torch.cat((phi.cos(), phi.sin()), dim=-1)
    x = _random(rows, 14336)
    scales = None
    args = _random(256) / 4, _random(256) / 4, rope, positions, 1e-6
    if cell.spec.get("parts", False):
        padded_rows = cell.spec.get("pad_rows", rows)
        x = _random(2, padded_rows, 14336).float()[:, :rows]
        scales = x.new_tensor(0.25), x.new_tensor(0.125)
    q, _, _, gate = prepare_qkv(
        x, *args, scales=scales, cache=cache, table=table, rows_per_seq=width
    )
    combined = x if scales is None else ((x[0] + x[1]) * (scales[0] * scales[1])).bfloat16()
    expected_q, k, v, expected_gate = prepare_qkv(combined, *args)
    append_kv(k, v, positions, standalone, table, None, width)
    ref_outputs = prepare_qkv.reference(
        x, *args, scales=scales, cache=ref, table=table, rows_per_seq=width
    )
    for actual, expected in zip((q, cache.key, cache.value, gate), ref_outputs, strict=True):
        _assert_bf16(actual, expected)
    for actual, expected in (
        (q, expected_q),
        (gate, expected_gate),
        (cache.key, standalone.key),
        (cache.value, standalone.value),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("cell", append_kv.CELLS)
def test_append_paged(cell) -> None:
    torch.manual_seed(18)
    lengths, width = cell.spec["lengths"], cell.spec["width"]
    starts = [0, 0] if width == 1 else [30, 62]
    cache, table = _paged_cache(tuple(n + s for n, s in zip(lengths, starts, strict=True)), False)
    reference = cache._replace(key=cache.key.clone(), value=cache.value.clone())
    rows = sum(lengths)
    k, v = _random(rows, 1024), _random(rows, 1024)
    pos = torch.cat(
        [torch.arange(s, s + n, device="cuda") for s, n in zip(starts, lengths, strict=True)]
    )
    req = (
        torch.arange(len(lengths), device="cuda").repeat_interleave(
            torch.tensor(lengths, device="cuda")
        )
        if width == 1
        else None
    )
    append_kv(k, v, pos, cache, table, req, width)
    append_kv.reference(k, v, pos, reference, table, req, width)
    torch.testing.assert_close(cache.key, reference.key, rtol=0, atol=0)
    torch.testing.assert_close(cache.value, reference.value, rtol=0, atol=0)


@pytest.mark.parametrize("cell", context_attention.CELLS)
def test_context_attention(cell) -> None:
    torch.manual_seed(19)
    lengths = cell.spec["lengths"]
    starts = torch.tensor(list(accumulate(lengths, initial=0)), device="cuda", dtype=torch.int32)
    chunks = torch.tensor(
        [(i, c) for i, size in enumerate(lengths) for c in range((size + 63) // 64)],
        device="cuda",
        dtype=torch.int32,
    )
    args = (
        _random(sum(lengths), 6144) / 2,
        _random(sum(lengths), 1024) / 2,
        _random(sum(lengths), 1024),
    )
    last_only = cell.spec.get("last_only", False)
    # Softmax tiles round the unnormalized probabilities to BF16 for the V dot product.
    _assert_bf16(
        context_attention(*args, starts, chunks, last_only=last_only),
        context_attention.reference(*args, starts, chunks, last_only=last_only),
        ulps=2,
    )
    if last_only:
        torch.testing.assert_close(
            context_attention(*args, starts, chunks, last_only=True),
            context_attention(*args, starts, chunks)[-1:],
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("cell", decode_attention.CELLS)
def test_paged_decode(cell) -> None:
    torch.manual_seed(20)
    lengths, width = cell.spec["lengths"], cell.spec["width"]
    cache, table = _paged_cache(lengths, True)
    pos = torch.cat([torch.arange(n - width, n, device="cuda", dtype=torch.int32) for n in lengths])
    q, gate = _random(pos.numel(), 6144) / 2, _random(pos.numel(), 6144)
    output = decode_attention(q, gate, pos, cache, table, rows_per_seq=width)
    _assert_bf16(
        output,
        decode_attention.reference(q, gate, pos, cache, table, rows_per_seq=width),
        ulps=2,
    )
    if len(lengths) >= 8:
        individual = [
            decode_attention(
                q[begin:end],
                gate[begin:end],
                pos[begin:end],
                cache,
                table,
                rows_per_seq=width,
                request_start=seq,
            )
            for seq, (begin, end) in enumerate(
                zip(range(0, len(pos), width), range(width, len(pos) + 1, width), strict=True)
            )
        ]
        torch.testing.assert_close(output, torch.cat(individual), atol=0, rtol=0)


@pytest.mark.parametrize("cell", paged_context_attention.CELLS)
def test_paged_context(cell) -> None:
    torch.manual_seed(21)
    prefixes = cell.spec["prefixes"]
    lengths = (95, 11)
    cache, table = _paged_cache(tuple(p + n for p, n in zip(prefixes, lengths, strict=True)), True)
    starts = torch.tensor(list(accumulate(lengths, initial=0)), device="cuda", dtype=torch.int32)
    chunks = torch.tensor(
        [(i, c) for i, length in enumerate(lengths) for c in range((length + 63) // 64)],
        device="cuda",
        dtype=torch.int32,
    )
    positions = torch.cat(
        [
            torch.arange(prefix, prefix + length, device="cuda", dtype=torch.int32)
            for prefix, length in zip(prefixes, lengths, strict=True)
        ]
    )
    q = _random(sum(lengths), 6144) / 2
    args = q, starts, chunks, positions, cache, table
    _assert_bf16(paged_context_attention(*args), paged_context_attention.reference(*args), ulps=2)

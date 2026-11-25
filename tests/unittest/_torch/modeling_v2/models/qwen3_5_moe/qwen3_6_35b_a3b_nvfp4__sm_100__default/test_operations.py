# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: F811

"""Test SM100 Qwen target kernels against their independent references."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5_moe.qwen3_6_35b_a3b_nvfp4__sm_100__default.ops import (  # noqa: E501
    attention,
    moe,
    projections,
    recurrent,
)

__extra_import_path__ = ["../../.."]
from _validating import assert_within_ulp, certifying  # noqa: E402, F401

assert torch.cuda.is_available(), "Qwen3.6 operation cells require CUDA"
BF16 = torch.bfloat16
FP8 = torch.float8_e4m3fn
DEV = "cuda"


def fixture_rand(shape, dtype=BF16, factor=1.0):
    return (torch.randn(shape, device=DEV) * factor).to(dtype)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(972)


@pytest.mark.parametrize("cell", projections.fp8_cast.CELLS)
def test_fp8_cast(cell, certifying):
    op = projections.fp8_cast
    certifying(op)
    x = fixture_rand((cell.spec["rows"], cell.spec["width"]), factor=5)
    scale = torch.tensor(0.019, device=DEV)
    torch.testing.assert_close(op(x, scale).float(), op.reference(x, scale).float(), rtol=0, atol=0)


@pytest.mark.parametrize("cell", projections.delta_gate_projection.CELLS)
def test_delta_gate_projection(cell, certifying):
    op = projections.delta_gate_projection
    certifying(op)
    x = fixture_rand((cell.spec["rows"], 2048), factor=0.25)
    weight = fixture_rand((64, 2048), factor=0.25)
    assert_within_ulp(op(x, weight), op.reference(x, weight), 1, 0)


@pytest.mark.parametrize("cell", projections.channel_fp8_linear.CELLS)
def test_channel_fp8_linear(cell, certifying):
    op = projections.channel_fp8_linear
    certifying(op)
    rows, width = cell.spec["rows"], cell.spec["width"]
    x = fixture_rand((rows, 2048), FP8, 0.5)
    weight = fixture_rand((width, 2048), FP8, 0.5)
    activation_scales = torch.full((rows, 1), 0.0037, device=DEV)
    weight_scales = torch.linspace(0.001, 0.002, width, device=DEV)[None, :]
    args = x, weight, activation_scales, weight_scales
    # One BF16 rounding at a few FP32 accumulation boundaries. The
    # 128-row cell measured relative RMS error of about 1.05e-5.
    assert_within_ulp(op(*args), op.reference(*args), 1, 0.01)


@pytest.mark.parametrize("cell", moe.residual_rms_norm.CELLS)
def test_residual_rms_norm(cell, certifying):
    op = moe.residual_rms_norm
    certifying(op)
    shape = (cell.spec["rows"], 2048)
    x, residual = fixture_rand(shape), fixture_rand(shape)
    weight = fixture_rand((2048,), factor=0.15)
    reference_residual = residual.clone()
    quantized = cell.spec.get("fp8", False)
    fp8 = torch.empty_like(x, dtype=FP8) if quantized else None
    reference_fp8 = torch.empty_like(x, dtype=FP8) if quantized else None
    scale = torch.tensor(0.037, dtype=torch.float32, device=DEV) if quantized else None
    args = (weight, cell.spec["add"], cell.spec["save"])
    actual = op(x, residual, *args, fp8, scale)
    reference = op.reference(x, reference_residual, *args, reference_fp8, scale)
    assert_within_ulp(actual, reference, 1, 1)
    torch.testing.assert_close(residual, reference_residual, rtol=0, atol=0)
    if quantized:
        torch.testing.assert_close(fp8.float(), reference_fp8.float(), rtol=0, atol=0)


@pytest.mark.parametrize("cell", moe.route_topk.CELLS)
def test_route_topk(cell, certifying):
    op = moe.route_topk
    certifying(op)
    logits = fixture_rand((cell.spec["rows"], 272))
    logits[0, :12] = 8.0
    scale = fixture_rand((256,), factor=0.1).abs() if cell.spec["weighted"] else None
    actual_ids, actual_weights, actual_gate = op(logits, scale)
    ids, weights, gate = op.reference(logits, scale)
    torch.testing.assert_close(actual_ids, ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, weights)
    torch.testing.assert_close(actual_gate, gate)


@pytest.mark.parametrize("cell", moe.mix_experts_norm.CELLS)
def test_mix_experts_norm(cell, certifying):
    op = moe.mix_experts_norm
    certifying(op)
    rows, weighted = cell.spec["rows"], cell.spec["weighted"]
    shape = (rows, 2048)
    raw = fixture_rand((rows * 8 * 2, 2048), dtype=torch.float32, factor=0.1)
    alpha = fixture_rand((256,), factor=0.1).abs()
    ids = torch.randint(256, (rows, 8), device=DEV, dtype=torch.int32)
    probs = torch.rand((rows, 8), device=DEV).softmax(-1).to(BF16)
    slots = torch.randperm(rows * 8 * 2, device=DEV, dtype=torch.int32)[: rows * 8].reshape(rows, 8)
    shared = fixture_rand(shape)
    gate = fixture_rand((rows, 272))
    if weighted:
        gate = gate[:, 256].float().sigmoid().to(BF16)
    residual = fixture_rand(shape)
    weight = fixture_rand((2048,), factor=0.15)
    ref_residual = residual.clone()
    fp8 = torch.empty(shape, dtype=FP8, device=DEV) if weighted else None
    scale = torch.tensor(0.0037, dtype=torch.float32, device=DEV) if weighted else None
    args = raw, alpha, ids, probs, slots, shared, gate, weighted
    actual = op(*args, moe.ExpertEpilogue(residual, weight, weighted, fp8, scale))
    expected = op.reference(*args, moe.ExpertEpilogue(ref_residual, weight, weighted, None, None))
    # Accumulation and RMSNorm can each move the result across a BF16 midpoint.
    assert_within_ulp(actual, expected, 2, 1)
    assert_within_ulp(residual, ref_residual, 2, 1)
    if weighted:
        torch.testing.assert_close(
            fp8.float(), projections.fp8_cast.reference(actual, scale).float(), atol=0, rtol=0
        )


@pytest.mark.parametrize("cell", projections.packed_gemv.CELLS)
def test_packed_gemv(cell, certifying):
    op = projections.packed_gemv
    certifying(op)
    rows, mode, n, width = (cell.spec[k] for k in ("rows", "mode", "n", "width"))
    x = fixture_rand((rows, width), factor=0.1)
    weight = torch.randint(256, (n, width // 2), dtype=torch.uint8, device=DEV)
    scales = fixture_rand((n, width // 16), dtype=FP8, factor=0.15)
    alpha = torch.tensor(0.16, dtype=BF16, device=DEV)
    args = (x, weight, scales, alpha, mode)
    assert_within_ulp(op(*args), op.reference(*args), 2, 1)


@pytest.mark.parametrize("cell", recurrent.convolution_decode.CELLS)
def test_convolution_decode(cell, certifying):
    op = recurrent.convolution_decode
    certifying(op)
    rows, nslots = (cell.spec[key] for key in ("rows", "slots"))
    x = fixture_rand((rows, 12288))
    weight = fixture_rand((8192, 4))
    slots = torch.randperm(nslots, device=DEV, dtype=torch.int32)[:rows]
    # Cache history uses time-major backing storage with an extra layer per slot.
    pool = fixture_rand((nslots, 2, 3, 8192))
    history, reference = pool[:, 1].transpose(-1, -2), pool.clone()[:, 1].transpose(-1, -2)
    actual = op(x, weight, slots, history)
    expected = op.reference(x, weight, slots, reference)
    assert_within_ulp(actual, expected, 1, 1)
    torch.testing.assert_close(history, reference, rtol=0, atol=0)


@pytest.mark.parametrize("cell", recurrent.delta_decode.CELLS)
def test_delta_decode(cell, certifying):
    op = recurrent.delta_decode
    certifying(op)
    rows, nslots = (cell.spec[key] for key in ("rows", "slots"))
    qkv, ba = fixture_rand((rows, 8192)), fixture_rand((rows, 64))
    a_log, dt = fixture_rand((32,), torch.float32), fixture_rand((32,))
    slots = torch.randperm(nslots, device=DEV, dtype=torch.int32)[:rows]
    pool = fixture_rand((nslots, 2, 32, 128, 128), torch.float32, factor=0.05)
    state, reference = pool[:, 1], pool.clone()[:, 1]
    actual = op(qkv, ba, a_log, dt, slots, state)
    expected = op.reference(qkv, ba, a_log, dt, slots, reference)
    assert_within_ulp(actual, expected, 1, 1)
    torch.testing.assert_close(state, reference, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("cell", recurrent.delta_norm.CELLS)
def test_delta_norm(cell, certifying):
    op = recurrent.delta_norm
    certifying(op)
    rows = cell.spec["rows"]
    x, qkvz = fixture_rand((rows, 32, 128)), fixture_rand((rows, 12288))
    weight = fixture_rand((128,))
    scale = torch.tensor(0.015, device=DEV) if cell.spec["quantized"] else None
    actual, expected = op(x, qkvz, weight, scale), op.reference(x, qkvz, weight, scale)
    if scale is None:
        assert_within_ulp(actual, expected, 1, 1)
    else:
        torch.testing.assert_close(actual.float(), expected.float(), atol=0.001, rtol=0.125)


@pytest.mark.parametrize("cell", attention.prepare_qkv.CELLS)
def test_prepare_qkv(cell, certifying):
    op = attention.prepare_qkv
    certifying(op)
    page = cell.spec["page_size"]
    qkv = fixture_rand((4, 9216), factor=0.2)
    qnorm, knorm = fixture_rand((256,), factor=0.15), fixture_rand((256,), factor=0.15)
    angles = fixture_rand((2 * page, 32), torch.float32)
    rope = torch.cat((angles.cos(), angles.sin()), dim=-1)
    # The second layer shares the slot interleave, so the first layer's K/V
    # page offsets increment by four rather than by the two K/V roles.
    storage = torch.zeros((6, 2, 2, 2, page, 256), dtype=FP8, device=DEV)
    ref = storage.clone()
    kv = storage[:, 0]
    requests = torch.tensor([0, 0, 1, 1], device=DEV, dtype=torch.int32)
    positions = torch.tensor([0, page + 3, 2, page], device=DEV, dtype=torch.int32)
    table = torch.tensor([[[8, 0], [9, 1]], [[16, 12], [17, 13]]], device=DEV, dtype=torch.int32)
    actual_q, actual_gate = op(qkv, qnorm, knorm, rope, positions, requests, kv, table)
    q, gate = op.reference(qkv, qnorm, knorm, rope, positions, requests, ref[:, 0], table)
    torch.testing.assert_close(actual_gate, gate, rtol=0, atol=0)
    torch.testing.assert_close(actual_q.float(), q.float(), atol=1e-4, rtol=0.125)
    torch.testing.assert_close(storage.float(), ref.float(), atol=1e-4, rtol=0.125)


@pytest.mark.parametrize("cell", recurrent.convolution_prefill.CELLS)
def test_convolution_prefill(cell, certifying):
    op = recurrent.convolution_prefill
    certifying(op)
    lengths = cell.spec["lengths"]
    cu = torch.tensor([0, lengths[0], sum(lengths)], device=DEV)
    request_rows = torch.arange(2, device=DEV, dtype=torch.int32).repeat_interleave(
        torch.tensor(lengths, device=DEV)
    )
    x = fixture_rand((sum(lengths), 12288))
    weight = fixture_rand((8192, 4))
    slots = torch.tensor([3, 1], dtype=torch.int32, device=DEV)
    pool = fixture_rand((5, 3, 8192))
    history, ref_history = pool.transpose(-1, -2), pool.clone().transpose(-1, -2)
    initial = None
    if cell.spec["initial"]:
        initial = history[slots.long()].contiguous()
        initial[1].zero_()
    args = (x, weight, request_rows, cu, slots)
    output = op(*args, history, initial)
    reference = op.reference(*args, ref_history, initial)
    for actual, expected in zip(output, reference):
        assert_within_ulp(actual, expected, 1, 1)
    torch.testing.assert_close(history, ref_history, atol=0, rtol=0)


@pytest.mark.parametrize("cell", recurrent.delta_gates.CELLS)
def test_delta_gates(cell, certifying):
    op = recurrent.delta_gates
    certifying(op)
    ba = fixture_rand((cell.spec["rows"], 64))
    a_log, dt = fixture_rand((32,), torch.float32), fixture_rand((32,))
    actual_decay, actual_beta = op(ba, a_log, dt)
    decay, beta = op.reference(ba, a_log, dt)
    torch.testing.assert_close(actual_decay, decay, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(actual_beta, beta, atol=0.004, rtol=0)


@pytest.mark.parametrize("cell", attention.gated_attention.CELLS)
def test_gated_attention(cell, certifying):
    from types import SimpleNamespace

    op = attention.gated_attention
    certifying(op)
    page = 32
    storage = fixture_rand((32, 2, 2, page, 256), FP8, factor=0.4)
    table = (2 * torch.randperm(32, device=DEV, dtype=torch.int32)).reshape(2, -1)
    table = torch.stack((table, table + 1), dim=1)
    ctx = 3 if cell.spec["context"] else 0
    width = cell.spec["width"]
    if ctx:
        positions = [40, 41, 42, 303]
        requests = [0, 0, 0, 1]
    else:
        positions = list(range(166 - width, 166)) + list(range(304 - width, 304))
        requests = [0] * width + [1] * width
    q = fixture_rand((len(positions), 16, 256), FP8)
    gate = fixture_rand((len(positions), 4096))
    step = SimpleNamespace(
        positions=torch.tensor(positions, device=DEV, dtype=torch.int32),
        request_rows=torch.tensor(requests, device=DEV, dtype=torch.int32),
        context_tokens=ctx,
        contexts=int(bool(ctx)),
        max_context=ctx,
        cu=torch.tensor([0, ctx], device=DEV) if ctx else None,
        capacity=512,
        width=width,
    )
    scale = torch.tensor(0.01, device=DEV)
    args = (q, gate, storage, table, step, scale)
    output, reference = op(*args).float(), op.reference(*args).float()
    relative_l2 = (output - reference).norm() / reference.norm()
    # Quantizing the online softmax probabilities and then the gated output
    # adds two E4M3 roundings. A missing sigmoid exceeds this tolerance by
    # over an order of magnitude on the same cell.
    assert relative_l2 < 0.06, relative_l2
    assert (output - reference / gate.float().sigmoid()).norm() / reference.norm() > 0.8

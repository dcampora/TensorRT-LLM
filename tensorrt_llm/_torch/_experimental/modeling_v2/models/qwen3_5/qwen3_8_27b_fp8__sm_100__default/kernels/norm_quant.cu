/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted from aegis-engines norm_quant.cu (114cf57c911e).
 */
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace
{

using Bf16 = __nv_bfloat16;

__device__ __forceinline__ float rsqrt_reference(float value)
{
    float out;
    asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(value));
    return out;
}

template <bool StoreNormalized, bool Programmatic>
__global__ void fused_norm(
    Bf16* x, Bf16* residual, Bf16 const* weight, unsigned char* quantized, float* scales, int rows, int padded_rows)
{
    constexpr int H = 5120;
    constexpr int RowThreads = 64;
    constexpr int Vec = 8;
    constexpr int VecStride = RowThreads * Vec;
    constexpr int VecBlocks = H / VecStride;
    int const row = blockIdx.x;
    int const tid = threadIdx.x;
    int const lane = tid % 32;
    int const col = tid * Vec;
    int const offset = row * H + col;
    __shared__ float subtotals[2];
    if constexpr (Programmatic)
    {
        asm volatile("griddepcontrol.launch_dependents;\n");
    }

    if (row >= rows)
    {
#pragma unroll
        for (int i = 0; i < VecBlocks; ++i)
        {
            auto* out = reinterpret_cast<uint2*>(quantized + offset + i * VecStride);
            *out = make_uint2(0, 0);
            if (col % 128 == 0)
            {
                scales[(col / 128 + i * (VecStride / 128)) * padded_rows + row] = 0x1p-42f;
            }
        }
        return;
    }

    uint4 gammas[VecBlocks];
    if constexpr (Programmatic)
    {
#pragma unroll
        for (int i = 0; i < VecBlocks; ++i)
        {
            gammas[i] = *reinterpret_cast<uint4 const*>(weight + col + i * VecStride);
        }
        asm volatile("griddepcontrol.wait;\n" ::: "memory");
    }

    __shared__ __align__(16) Bf16 in_tile[H];
    __shared__ __align__(16) Bf16 shortcut_tile[H];
#pragma unroll
    for (int i = 0; i < VecBlocks; ++i)
    {
        int const global = offset + i * VecStride;
        int const local = col + i * VecStride;
        const uint32_t sx = __cvta_generic_to_shared(in_tile + local);
        const uint32_t sr = __cvta_generic_to_shared(shortcut_tile + local);
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;\n"
            "cp.async.cg.shared.global [%2], [%3], 16;\n" ::"r"(sx),
            "l"(x + global), "r"(sr), "l"(residual + global)
            : "memory");
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");

    float values[VecBlocks][Vec];
    float partial = 0.0f;
#pragma unroll
    for (int i = 0; i < VecBlocks; ++i)
    {
        int const pos = offset + i * VecStride;
        uint4 x_vec = *reinterpret_cast<uint4 const*>(in_tile + col + i * VecStride);
        uint4 r_vec = *reinterpret_cast<uint4 const*>(shortcut_tile + col + i * VecStride);
        auto const* x_values = reinterpret_cast<Bf16 const*>(&x_vec);
        auto const* r_values = reinterpret_cast<Bf16 const*>(&r_vec);
        auto* updated = reinterpret_cast<Bf16*>(&r_vec);
#pragma unroll
        for (int j = 0; j < Vec; ++j)
        {
            float const h = __bfloat162float(x_values[j]) + __bfloat162float(r_values[j]);
            values[i][j] = h;
            updated[j] = __float2bfloat16_rn(h);
            partial = __fmaf_rn(h, h, partial);
        }
        *reinterpret_cast<uint4*>(residual + pos) = r_vec;
    }

#pragma unroll
    for (int d = 1; d < 32; d *= 2)
    {
        partial += __shfl_xor_sync(0xffffffffu, partial, d);
    }
    if (lane == 0)
    {
        subtotals[tid / 32] = partial;
    }
    __syncthreads();
    float const mean = __fdiv_rn(subtotals[0] + subtotals[1], static_cast<float>(H));
    float const inv_rms = rsqrt_reference(mean + 1.0e-6f);

#pragma unroll
    for (int i = 0; i < VecBlocks; ++i)
    {
        int const pos = offset + i * VecStride;
        int const group = col / 128 + i * (VecStride / 128);
        const uint4 w_vec = Programmatic ? gammas[i] : *reinterpret_cast<uint4 const*>(weight + col + i * VecStride);
        auto const* w_values = reinterpret_cast<Bf16 const*>(&w_vec);
        uint4 y_vec;
        auto* rounded = reinterpret_cast<Bf16*>(&y_vec);
        float maximum = 1.0e-10f;
#pragma unroll
        for (int j = 0; j < Vec; ++j)
        {
            float const y = (values[i][j] * inv_rms) * (1.0f + __bfloat162float(w_values[j]));
            rounded[j] = __float2bfloat16_rn(y);
            values[i][j] = __bfloat162float(rounded[j]);
            maximum = fmaxf(maximum, fabsf(values[i][j]));
        }
        if constexpr (StoreNormalized)
        {
            *reinterpret_cast<uint4*>(x + pos) = y_vec;
        }
#pragma unroll
        for (int d = 8; d >= 1; d /= 2)
        {
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffu, maximum, d, 16));
        }
        const uint32_t bits = __float_as_uint(maximum * (1.0f / 448.0f));
        const uint32_t exponent = (bits >> 23) + ((bits & 0x7fffffu) != 0);
        float const reciprocal = __uint_as_float((254u - exponent) << 23);
        if (col % 128 == 0)
        {
            scales[group * padded_rows + row] = __uint_as_float(exponent << 23);
        }
        float const* a = values[i];
        const __nv_fp8x4_e4m3 low(
            make_float4(a[0] * reciprocal, a[1] * reciprocal, a[2] * reciprocal, a[3] * reciprocal));
        const __nv_fp8x4_e4m3 high(
            make_float4(a[4] * reciprocal, a[5] * reciprocal, a[6] * reciprocal, a[7] * reciprocal));
        *reinterpret_cast<uint2*>(quantized + pos) = make_uint2(low.__x, high.__x);
    }
}

template <bool StoreNormalized>
int launch(Bf16* x, Bf16* residual, Bf16 const* weight, unsigned char* output, float* scale, int rows, int padded,
    bool programmatic, cudaStream_t stream)
{
    if (programmatic)
    {
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t config{dim3(padded), dim3(64), 0, stream, &attribute, 1};
        return static_cast<int>(cudaLaunchKernelEx(
            &config, fused_norm<StoreNormalized, true>, x, residual, weight, output, scale, rows, padded));
    }
    fused_norm<StoreNormalized, false><<<padded, 64, 0, stream>>>(x, residual, weight, output, scale, rows, padded);
    return static_cast<int>(cudaGetLastError());
}

template <typename Id>
__global__ void draft_dual_norm_kernel(Id const* ids, Bf16 const* table, Bf16 const* hidden, Bf16 const* token_weight,
    Bf16 const* hidden_weight, Bf16* output)
{
    constexpr int width = 5120;
    constexpr int stride = 64 * 8;
    constexpr int slices = width / stride;
    int const offset = threadIdx.x * 8;
    int const row = blockIdx.x;
    int const part = blockIdx.y;
    int const lane = threadIdx.x % 32;
    Bf16 const* weight = part ? hidden_weight : token_weight;
    __shared__ float sums[2];
    uint4 weight_vectors[slices];
#pragma unroll
    for (int i = 0; i < slices; ++i)
    {
        weight_vectors[i] = *reinterpret_cast<uint4 const*>(weight + offset + i * stride);
    }
    Bf16 const* source = part ? hidden + row * width : table + static_cast<int64_t>(ids[row]) * width;
    float values[slices][8];
    float local = 0.0f;
#pragma unroll
    for (int i = 0; i < slices; ++i)
    {
        const uint4 payload = *reinterpret_cast<uint4 const*>(source + offset + i * stride);
        auto const* val = reinterpret_cast<Bf16 const*>(&payload);
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            float const x = __bfloat162float(val[j]);
            values[i][j] = x;
            local = __fmaf_rn(x, x, local);
        }
    }
// Match the two warp partial sums and RMSNorm's BF16 output rounding.
#pragma unroll
    for (int bit = 1; bit <= 16; bit *= 2)
    {
        local = __fadd_rn(local, __shfl_xor_sync(~0u, local, bit));
    }
    if (lane == 0)
        sums[threadIdx.x / 32] = local;
    __syncthreads();
    float const mean = __fdiv_rn(sums[0] + sums[1], float(width)) + 1.0e-6f;
    float const inverse = rsqrt_reference(mean);
    Bf16* dest = output + (2 * row + part) * width + offset;
#pragma unroll
    for (int i = 0; i < slices; ++i)
    {
        const uint4 gamma = weight_vectors[i];
        auto const* g = reinterpret_cast<Bf16 const*>(&gamma);
        uint4 normalized;
        auto* val = reinterpret_cast<Bf16*>(&normalized);
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            val[j] = __float2bfloat16_rn((values[i][j] * inverse) * (1.f + __bfloat162float(g[j])));
        }
        *reinterpret_cast<uint4*>(dest + i * stride) = normalized;
    }
}

} // namespace

extern "C" int norm_quant(void* x, void* residual, void const* weight, void* a, float* scale, int m, int padded,
    bool store_normalized, bool programmatic, cudaStream_t stream)
{
    auto* activations = static_cast<Bf16*>(x);
    auto* shortcut = static_cast<Bf16*>(residual);
    auto* w = static_cast<Bf16 const*>(weight);
    auto* out = static_cast<unsigned char*>(a);
    if (store_normalized)
    {
        return launch<true>(activations, shortcut, w, out, scale, m, padded, programmatic, stream);
    }
    return launch<false>(activations, shortcut, w, out, scale, m, padded, programmatic, stream);
}

extern "C" int draft_dual_norm(void const* ids, void const* table, void const* hidden, void const* token_weight,
    void const* hidden_weight, void* output, int rows, bool wide_ids, cudaStream_t stream)
{
    auto const* embedding = static_cast<Bf16 const*>(table);
    auto const* state = static_cast<Bf16 const*>(hidden);
    auto const* token_norm = static_cast<Bf16 const*>(token_weight);
    auto const* hidden_norm = static_cast<Bf16 const*>(hidden_weight);
    auto* result = static_cast<Bf16*>(output);
    if (wide_ids)
    {
        draft_dual_norm_kernel<<<dim3(rows, 2), 64, 0, stream>>>(
            static_cast<int64_t const*>(ids), embedding, state, token_norm, hidden_norm, result);
    }
    else
    {
        draft_dual_norm_kernel<<<dim3(rows, 2), 64, 0, stream>>>(
            static_cast<int32_t const*>(ids), embedding, state, token_norm, hidden_norm, result);
    }
    return static_cast<int>(cudaGetLastError());
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted from aegis-engines short_solve.cu (114cf57c911e).
 */
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace
{

constexpr int chunk = 64;
constexpr int heads = 48;
constexpr int warps = 4;
constexpr unsigned warp_mask = ~0u;

__device__ __forceinline__ float reduce_product(float coefficient, float value)
{
    float product = __fmul_rn(coefficient, value);
    // Keep the contraction and lane order from the original FP32 reduction.
    product = __fmaf_rn(coefficient, value, __shfl_xor_sync(warp_mask, product, 8));
#pragma unroll
    for (int stride = 4; stride >= 1; stride /= 2)
    {
        product = __fadd_rn(product, __shfl_xor_sync(warp_mask, product, stride));
    }
    return product;
}

__device__ __forceinline__ float add_row(float coefficient, float partial)
{
    float const sum = __fadd_rn(partial, __shfl_xor_sync(warp_mask, partial, 16));
    // Preserve the original reduction's upper-half representative.
    return __fadd_rn(coefficient, __shfl_sync(warp_mask, sum, 8));
}

__global__ void solve_short_gdn(
    float const* lower, int32_t const* offsets, int32_t const* chunks, __nv_bfloat16* output)
{
    int const thread = threadIdx.x;
    int const lane = thread % 32;
    int const column = blockIdx.z * warps + thread / 32;
    int const group = blockIdx.x;
    int const head = blockIdx.y;
    int const sequence = chunks[2 * group];
    int const start = offsets[sequence] + chunks[2 * group + 1] * chunk;
    int const length = min(chunk, offsets[sequence + 1] - start);
    int const offset = (group * heads + head) * chunk * chunk;
    float const* input = lower + offset;
    __shared__ __align__(16) float matrix[chunk * chunk];
    bool const has_updates = int(blockIdx.z) * warps + 1 < length;
    for (int i = thread; has_updates && i < length * chunk / 4; i += warps * 32)
    {
        reinterpret_cast<float4*>(matrix)[i] = reinterpret_cast<float4 const*>(input)[i];
    }
    __syncthreads();

    float lower_row = 0.0f;
    float upper_row = 0.0f;
    // The second half of the state is still zero through row 32.
    for (int row = column + 1; row < min(33, length); ++row)
    {
        float const* coefficients = matrix + row * chunk;
        float const low = reduce_product(coefficients[lane], lower_row);
        float const value = add_row(coefficients[column], low);
        lower_row = lane == row ? value : lower_row;
        upper_row = lane + 32 == row ? value : upper_row;
    }
    for (int row = max(33, column + 1); row < length; ++row)
    {
        float const* coefficients = matrix + row * chunk;
        float const low = reduce_product(coefficients[lane], lower_row);
        float const high = reduce_product(coefficients[lane + 32], upper_row);
        float const value = add_row(coefficients[column], __fadd_rn(low, high));
        lower_row = lane == row ? value : lower_row;
        upper_row = lane + 32 == row ? value : upper_row;
    }

    auto* out = output + offset;
    out[lane * chunk + column] = __float2bfloat16_rn(lower_row + float(lane == column));
    out[(lane + 32) * chunk + column] = __float2bfloat16_rn(upper_row + float(lane + 32 == column));
}

} // namespace

extern "C" int short_solve(
    float const* lower, int32_t const* offsets, int32_t const* chunks, void* output, int groups, cudaStream_t stream)
{
    solve_short_gdn<<<dim3(groups, heads, chunk / warps), warps * 32, 0, stream>>>(
        lower, offsets, chunks, static_cast<__nv_bfloat16*>(output));
    return static_cast<int>(cudaGetLastError());
}

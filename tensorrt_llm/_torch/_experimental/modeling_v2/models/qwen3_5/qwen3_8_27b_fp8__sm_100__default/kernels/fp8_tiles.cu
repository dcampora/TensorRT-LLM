/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted from aegis-engines fp8_tiles.cu (114cf57c911e).
 */
#define CUTLASS_ENABLE_GDC_FOR_SM100
#include <cstddef>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/detail/blockwise_scale_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/float8.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <type_traits>

namespace
{

using namespace cute;

template <int TileM, int TileN, int WeightGroup, bool Swap = false, int Pipeline = 0>
int matmul(void* workspace, size_t workspace_bytes, void const* input, void const* weight, float const* input_scale,
    float const* weight_scale, void* output, int m, int n, int k, cudaStream_t stream, bool pdl)
{
    using E4M3 = cutlass::float_e4m3_t;
    using Output = cutlass::bfloat16_t;
    using Row = cutlass::layout::RowMajor;
    using Col = cutlass::layout::ColumnMajor;
    using OutputLayout = std::conditional_t<Swap, Col, Row>;
    using Tile = Shape<Int<TileM>, Int<TileN>, _128>;
    using Cluster = Shape<_1, _1, _1>;
    using Scales = cutlass::detail::Sm100BlockwiseScaleConfig<Swap ? WeightGroup : 1, Swap ? 1 : WeightGroup, 128,
        UMMA::Major::MN, UMMA::Major::MN>;
    using InputScaleLayout = decltype(Scales::deduce_layoutSFA());
    using WeightScaleLayout = decltype(Scales::deduce_layoutSFB());
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto, float, float,
        void, OutputLayout, 8, Output, OutputLayout, 8, cutlass::epilogue::TmaWarpSpecialized1Sm,
        cutlass::epilogue::fusion::LinearCombination<Output, float, void, float>>::CollectiveOp;
    using Stages = std::conditional_t<Pipeline == 0,
        cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
        cutlass::gemm::collective::StageCount<Pipeline>>;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm100,
        cutlass::arch::OpClassTensorOp, E4M3, tuple<Row, InputScaleLayout>, 16, E4M3, tuple<Col, WeightScaleLayout>, 16,
        float, Tile, Cluster, Stages, cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    int const rows = Swap ? n : m;
    int const columns = Swap ? m : n;
    auto problem = make_shape(rows, columns, k, 1);
    auto stride_a = cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, make_shape(rows, k, 1));
    auto stride_b = cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, make_shape(columns, k, 1));
    auto stride_d = cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, make_shape(rows, columns, 1));
    auto hardware = cutlass::KernelHardwareInfo::make_kernel_hardware_info<Kernel>();
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        {static_cast<const E4M3*>(Swap ? weight : input), stride_a, static_cast<const E4M3*>(Swap ? input : weight),
            stride_b, Swap ? weight_scale : input_scale, Scales::tile_atom_to_shape_SFA(problem),
            Swap ? input_scale : weight_scale, Scales::tile_atom_to_shape_SFB(problem)},
        {{}, nullptr, stride_d, static_cast<Output*>(output), stride_d},
        hardware,
    };
    arguments.epilogue.thread.alpha = 1.0f;
    arguments.epilogue.thread.beta = 0.0f;

    Gemm gemm;
    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess)
    {
        return static_cast<int>(status);
    }
    if (gemm.get_workspace_size(arguments) > workspace_bytes)
    {
        return -1;
    }
    status = gemm.initialize(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess)
    {
        return static_cast<int>(status);
    }
    return static_cast<int>(gemm.run(stream, nullptr, pdl));
}

} // namespace

extern "C" int fp8_tiled_matmul(void* workspace, size_t workspace_bytes, void const* input, void const* weight,
    float const* input_scale, float const* weight_scale, void* output, int m, int n, int k, cudaStream_t stream,
    bool pdl, bool wide_pipeline)
{
    if (n == 5120)
    {
        if (m <= 64)
        {
            return matmul<64, 64, 64>(
                workspace, workspace_bytes, input, weight, input_scale, weight_scale, output, m, n, k, stream, pdl);
        }
        return matmul<128, 64, 64>(
            workspace, workspace_bytes, input, weight, input_scale, weight_scale, output, m, n, k, stream, pdl);
    }
    if (m <= 8)
    {
        if (wide_pipeline && n >= 32768)
        {
            return matmul<128, 16, 128, true, 6>(
                workspace, workspace_bytes, input, weight, input_scale, weight_scale, output, m, n, k, stream, pdl);
        }
        return matmul<128, 16, 128, true>(
            workspace, workspace_bytes, input, weight, input_scale, weight_scale, output, m, n, k, stream, pdl);
    }
    return matmul<64, 128, 128>(
        workspace, workspace_bytes, input, weight, input_scale, weight_scale, output, m, n, k, stream, pdl);
}

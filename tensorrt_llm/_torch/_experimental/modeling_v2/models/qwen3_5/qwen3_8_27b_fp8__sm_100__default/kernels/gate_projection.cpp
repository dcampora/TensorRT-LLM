/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted from aegis-engines ba_lt.cpp (114cf57c911e).
 */
#include <array>
#include <cublasLt.h>
#include <cuda_runtime.h>

namespace
{

constexpr int kOutput = 96;
constexpr int kInput = 5120;
constexpr int kMaxRows = 512;
constexpr size_t kWorkspace = 16 * 1024 * 1024;

using Settings = std::array<int, 7>;

cublasStatus_t find_algorithm(cublasLtHandle_t handle, cublasLtMatmulDesc_t operation, cublasLtMatrixLayout_t weight,
    cublasLtMatrixLayout_t input, cublasLtMatrixLayout_t output, size_t workspace, Settings const& expected,
    cublasLtMatmulAlgo_t& algorithm)
{
    cublasLtMatmulPreference_t preference = nullptr;
    auto result = cublasLtMatmulPreferenceCreate(&preference);
    if (result != CUBLAS_STATUS_SUCCESS)
        return result;
    result = cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace));
    std::array<cublasLtMatmulHeuristicResult_t, 32> choices{};
    int count = 0;
    if (result == CUBLAS_STATUS_SUCCESS)
    {
        result = cublasLtMatmulAlgoGetHeuristic(
            handle, operation, weight, input, output, output, preference, choices.size(), choices.data(), &count);
    }
    cublasLtMatmulPreferenceDestroy(preference);
    if (result != CUBLAS_STATUS_SUCCESS)
        return result;

    constexpr std::array attributes{
        CUBLASLT_ALGO_CONFIG_ID,
        CUBLASLT_ALGO_CONFIG_TILE_ID,
        CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
        CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
        CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
        CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
        CUBLASLT_ALGO_CONFIG_STAGES_ID,
    };
    for (int i = 0; i < count; ++i)
    {
        Settings actual{};
        for (size_t j = 0; j < attributes.size(); ++j)
        {
            cublasLtMatmulAlgoConfigGetAttribute(&choices[i].algo, attributes[j], &actual[j], sizeof(int), nullptr);
        }
        if (actual == expected)
        {
            algorithm = choices[i].algo;
            return CUBLAS_STATUS_SUCCESS;
        }
    }
    return CUBLAS_STATUS_NOT_SUPPORTED;
}

struct Layout
{
    cublasLtMatrixLayout_t value = nullptr;

    ~Layout()
    {
        if (value)
            cublasLtMatrixLayoutDestroy(value);
    }

    cublasStatus_t prepare(int rows, int columns)
    {
        if (value)
            return CUBLAS_STATUS_SUCCESS;
        return cublasLtMatrixLayoutCreate(&value, CUDA_R_16BF, rows, columns, rows);
    }
};

struct GateProjection
{
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t operation = nullptr;
    Layout weight;
    std::array<Layout, kMaxRows + 1> inputs;
    std::array<Layout, kMaxRows + 1> outputs;
    cublasLtMatmulAlgo_t algorithm{};
    cublasStatus_t status;

    GateProjection()
        : status(prepare())
    {
    }

    ~GateProjection()
    {
        if (operation)
            cublasLtMatmulDescDestroy(operation);
        if (handle)
            cublasLtDestroy(handle);
    }

    cublasStatus_t prepare()
    {
        auto result = cublasLtCreate(&handle);
        if (result != CUBLAS_STATUS_SUCCESS)
            return result;
        result = cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        if (result != CUBLAS_STATUS_SUCCESS)
            return result;
        const cublasOperation_t transpose = CUBLAS_OP_T;
        result = cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose));
        if (result != CUBLAS_STATUS_SUCCESS)
            return result;
        result = weight.prepare(kInput, kOutput);
        if (result != CUBLAS_STATUS_SUCCESS)
            return result;
        result = inputs[1].prepare(kInput, 1);
        if (result != CUBLAS_STATUS_SUCCESS)
            return result;
        result = outputs[1].prepare(kOutput, 1);
        if (result != CUBLAS_STATUS_SUCCESS)
            return result;

        // Both M=1 and M=32 use this five-way split-K reduction on GB200.
        // Retain its arithmetic when the verifier concatenates several such groups.
        return find_algorithm(handle, operation, weight.value, inputs[1].value, outputs[1].value, kWorkspace,
            {66, 12, 5, 2, 0, 0, 35}, algorithm);
    }

    cublasStatus_t run(
        void const* x, void const* w, void* y, int rows, void* workspace, size_t workspace_bytes, cudaStream_t stream)
    {
        if (status != CUBLAS_STATUS_SUCCESS)
            return status;
        if (rows < 1 || rows > kMaxRows)
            return CUBLAS_STATUS_INVALID_VALUE;
        auto result = inputs[rows].prepare(kInput, rows);
        if (result != CUBLAS_STATUS_SUCCESS)
            return result;
        result = outputs[rows].prepare(kOutput, rows);
        if (result != CUBLAS_STATUS_SUCCESS)
            return result;
        float const alpha = 1.0f, beta = 0.0f;
        return cublasLtMatmul(handle, operation, &alpha, w, weight.value, x, inputs[rows].value, &beta, y,
            outputs[rows].value, y, outputs[rows].value, &algorithm, workspace, workspace_bytes, stream);
    }
};

} // namespace

extern "C" int project_gdn_gates(
    void const* x, void const* w, void* y, int rows, void* workspace, size_t workspace_bytes, cudaStream_t stream)
{
    thread_local GateProjection projection;
    return projection.run(x, w, y, rows, workspace, workspace_bytes, stream);
}

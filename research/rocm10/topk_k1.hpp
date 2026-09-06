#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <limits>

// Research K=1 candidate, not a registered or qualified TOPK provider.
// Caller supplies finite, non-NaN row-major F32 values and rows I32 outputs.
// The buffers must be valid, non-overlapping, and remain alive until the stream
// completes. Finite-input validation belongs to the caller's CPU preflight;
// this asynchronous wrapper does not copy/inspect device inputs on the host.
// Numeric ties (including +0/-0) select the lower original column index.
// No caller scratch is needed. Each row uses one 256-thread block and 2 KiB
// static shared memory. The caller must check later stream completion errors.
namespace strixlab_topk_k1_detail {
constexpr unsigned kThreads = 256;

__host__ __device__ constexpr bool prefer(float candidate_value, int candidate_index,
                                         float current_value, int current_index) {
    return candidate_index >= 0 &&
           (current_index < 0 || candidate_value > current_value ||
            (candidate_value == current_value && candidate_index < current_index));
}

template<unsigned Threads>
__global__ void argmax_rows(const float* src, int* dst, int columns) {
    static_assert(Threads == kThreads);
    __shared__ float values[Threads];
    __shared__ int indices[Threads];
    const unsigned lane = threadIdx.x;
    const std::size_t row_offset = static_cast<std::size_t>(blockIdx.x) *
                                   static_cast<std::size_t>(columns);
    float best_value = 0.0f;
    int best_index = -1; // An invalid lane never defeats a real (even negative) value.
    for (int column = static_cast<int>(lane); column < columns;
         column += static_cast<int>(Threads)) {
        const float value = src[row_offset + static_cast<std::size_t>(column)];
        if (prefer(value, column, best_value, best_index)) {
            best_value = value;
            best_index = column;
        }
    }
    values[lane] = best_value;
    indices[lane] = best_index;
    __syncthreads();
    for (unsigned stride = Threads / 2; stride != 0; stride /= 2) {
        if (lane < stride && prefer(values[lane + stride], indices[lane + stride],
                                     values[lane], indices[lane])) {
            values[lane] = values[lane + stride];
            indices[lane] = indices[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) dst[blockIdx.x] = indices[0];
}
} // namespace strixlab_topk_k1_detail

inline hipError_t strixlab_topk_k1_hip(const float* src, int* dst, int rows,
                                      int columns, hipStream_t stream) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    static_assert(sizeof(int) == 4 && std::numeric_limits<int>::digits == 31);
    if (src == nullptr || dst == nullptr || rows <= 0 || columns <= 0 || columns > 1024) {
        return hipErrorInvalidValue;
    }
    const auto row_count = static_cast<std::size_t>(rows);
    const auto column_count = static_cast<std::size_t>(columns);
    if (row_count > std::numeric_limits<std::size_t>::max() / column_count ||
        row_count * column_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        row_count > std::numeric_limits<std::size_t>::max() / sizeof(int)) {
        return hipErrorInvalidValue;
    }
    hipLaunchKernelGGL((strixlab_topk_k1_detail::argmax_rows<256>),
                       dim3(static_cast<unsigned>(rows)), dim3(256), 0, stream,
                       src, dst, columns);
    return hipGetLastError();
}

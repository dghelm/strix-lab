#pragma once
#include <hip/hip_runtime.h>
#include <cassert>
#include <cstddef>
#include <limits>

// Research-only finite F32 K=1 variants. Same buffer/size/tie contract as
// topk_k1.hpp: non-overlapping row-major input and rows I32 outputs, kept alive
// through stream completion. Numeric ties, including signed zero, use the lower
// original column index. Caller preflights finite input and checks completion.
// No caller scratch. Consumer MUST check gfx1151 and properties.warpSize == 32
// before using this fixture. Do not put device-property queries in timed calls.
namespace strixlab_topk_k1_variants_detail {
__host__ __device__ constexpr bool prefer(float v, int i, float best, int index) {
    return i >= 0 && (index < 0 || v > best || (v == best && i < index));
}
inline bool valid(const float* src, int* dst, int rows, int columns) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    static_assert(sizeof(int) == 4 && std::numeric_limits<int>::digits == 31);
    if (!src || !dst || rows <= 0 || columns <= 0 || columns > 1024) return false;
    const auto r = static_cast<std::size_t>(rows), c = static_cast<std::size_t>(columns);
    return r <= std::numeric_limits<std::size_t>::max() / c &&
           r * c <= std::numeric_limits<std::size_t>::max() / sizeof(float) &&
           r <= std::numeric_limits<std::size_t>::max() / sizeof(int);
}
template<unsigned Threads>
__global__ void shared_rows(const float* src, int* dst, int columns) {
    static_assert(Threads == 32 || Threads == 256);
    __shared__ float values[Threads];
    __shared__ int indices[Threads];
    const unsigned lane = threadIdx.x;
    const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * columns;
    float best = 0.0f;
    int index = -1;
    for (int c = static_cast<int>(lane); c < columns; c += static_cast<int>(Threads)) {
        const float v = src[offset + c];
        if (prefer(v, c, best, index)) { best = v; index = c; }
    }
    values[lane] = best;
    indices[lane] = index;
    __syncthreads();
    for (unsigned stride = Threads / 2; stride; stride /= 2) {
        if (lane < stride && prefer(values[lane + stride], indices[lane + stride],
                                    values[lane], indices[lane])) {
            values[lane] = values[lane + stride];
            indices[lane] = indices[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) dst[blockIdx.x] = indices[0];
}
__device__ inline void wave_reduce(float& best, int& index) {
    // Every lane of the participating wave executes both shuffles, even invalid
    // lanes. Width 32 partitions each physical wave; out-of-range self values
    // are ignored explicitly. A value and its original index travel together.
    const unsigned lane = threadIdx.x % 32;
    for (unsigned delta = 16; delta; delta /= 2) {
        const float v = __shfl_down(best, delta, 32);
        const int i = __shfl_down(index, delta, 32);
        if (lane + delta < 32 && prefer(v, i, best, index)) { best = v; index = i; }
    }
}
__global__ void wave_rows(const float* src, int* dst, int columns) {
    assert(warpSize == 32);
    __shared__ float values[8];
    __shared__ int indices[8];
    const unsigned thread = threadIdx.x, lane = thread % 32, wave = thread / 32;
    const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * columns;
    float best = 0.0f;
    int index = -1;
    for (int c = static_cast<int>(thread); c < columns; c += 256) {
        const float v = src[offset + c];
        if (prefer(v, c, best, index)) { best = v; index = c; }
    }
    wave_reduce(best, index);
    if (lane == 0) { values[wave] = best; indices[wave] = index; }
    __syncthreads(); // All eight wave leaders publish before wave 0 reads.
    if (wave == 0) {
        best = lane < 8 ? values[lane] : 0.0f;
        index = lane < 8 ? indices[lane] : -1;
        wave_reduce(best, index); // All 32 lanes of wave 0 participate.
        if (lane == 0) dst[blockIdx.x] = index;
    }
}
} // namespace strixlab_topk_k1_variants_detail

inline hipError_t strixlab_topk_k1_small_hip(const float* src, int* dst, int rows,
                                           int columns, hipStream_t stream) {
    using namespace strixlab_topk_k1_variants_detail;
    if (!valid(src, dst, rows, columns)) return hipErrorInvalidValue;
    if (columns <= 32) {
        hipLaunchKernelGGL((shared_rows<32>), dim3(static_cast<unsigned>(rows)), dim3(32),
                           0, stream, src, dst, columns);
    } else {
        hipLaunchKernelGGL((shared_rows<256>), dim3(static_cast<unsigned>(rows)), dim3(256),
                           0, stream, src, dst, columns);
    }
    return hipGetLastError();
}
inline hipError_t strixlab_topk_k1_wave_hip(const float* src, int* dst, int rows,
                                          int columns, hipStream_t stream) {
    using namespace strixlab_topk_k1_variants_detail;
    if (!valid(src, dst, rows, columns)) return hipErrorInvalidValue;
    hipLaunchKernelGGL(wave_rows, dim3(static_cast<unsigned>(rows)), dim3(256),
                       0, stream, src, dst, columns);
    return hipGetLastError();
}

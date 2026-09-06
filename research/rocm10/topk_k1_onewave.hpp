#pragma once
#include "topk_k1_variants.hpp"

// Research-only one-wave K=1 candidate. Inherits the finite F32, non-aliasing,
// positive shape (columns <= 1024), I32 original-index, lower-index numeric tie
// and asynchronous lifetime/error contract of topk_k1_variants.hpp. Consumer
// must check gfx1151 and properties.warpSize == 32 before allocation/timing.
// One 32-thread block per row, at most 32 strided reads per lane, no shared
// memory, block barriers, or caller scratch. All lanes participate in shuffles.
namespace strixlab_topk_k1_onewave_detail {
__global__ void rows(const float* src, int* dst, int columns) {
    assert(warpSize == 32);
    using namespace strixlab_topk_k1_variants_detail;
    const unsigned lane = threadIdx.x;
    const std::size_t offset = static_cast<std::size_t>(blockIdx.x) * columns;
    float best = 0.0f;
    int index = -1;
    for (int c = static_cast<int>(lane); c < columns; c += 32) {
        const float v = src[offset + c];
        if (prefer(v, c, best, index)) { best = v; index = c; }
    }
    // Reuse the reviewed uniform width-32 value/index shuffle and sentinel rule.
    wave_reduce(best, index);
    if (lane == 0) dst[blockIdx.x] = index;
}
} // namespace strixlab_topk_k1_onewave_detail

inline hipError_t strixlab_topk_k1_onewave_hip(const float* src, int* dst, int rows,
                                            int columns, hipStream_t stream) {
    if (!strixlab_topk_k1_variants_detail::valid(src, dst, rows, columns))
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(strixlab_topk_k1_onewave_detail::rows,
                       dim3(static_cast<unsigned>(rows)), dim3(32), 0, stream,
                       src, dst, columns);
    return hipGetLastError();
}

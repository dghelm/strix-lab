#pragma once
#include <hip/hip_runtime.h>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

// Bounded research extraction, NOT a registered GGML provider or proof of parity
// with a compiled full backend. Immutable reference: halo-box/strix-llama.cpp
// c7af5c6c29902eb1f7b3bd7952607e2349e1c668. SHA256, source lines:
// ggml/src/ggml-cuda/norm.cu:77-154,342-354
//   26cf54b8d74abcad072dfb7d4863bc47ac98e4cbedb46197626a57880deacd07
// ggml/src/ggml-cuda/scale.cu:5-24 (scale.cuh:3 block size 256)
//   38450f05603bf7ea3b2885402c91d0cc40acbe8b29cd1385e0523ff3b065bbe9
// ggml/src/ggml-cuda/common.cuh:464-471,638-659
//   cfff031867bea904f56bcc86161c6ad99e7601e4c428d536a7fcaf2abce9d009
// src/models/models.h:14-18
//   4f4fffd3bc53e2ff1e3604faadcd6e2b05fc4c24e15ad198372afeaaf190480d
// src/models/qwen35.cpp:419-446
//   ca17b850d1a8cfa3ff77c3f93b0094513cd2ef7ce66baae81af5b3afca442952
// Qwen4B metadata: state_size=128, group_count=16, inner_size=4096,
// time_step_rank=32, epsilon=1e-6. Q/K have 16 heads, NOT 32 value heads.
// Conv QKV: [8192,tokens,sequences], Q offset 0, K offset 2048 floats;
// strides {128,8192,8192*tokens}. Tokens {1,16,512}, sequence=1 are workload
// cases; sequence=2 and separate epsilons are additional contract diagnostics.
//
// Contract: width128, wave32, finite F32 inputs, finite positive graph epsilon
// whose FP32 epsilon/128 remains positive. Finite inputs alone do not guarantee
// finite intermediate squares/sums. Float64-oracle checks cover the bounded
// test vectors (including underflow), not all finite FP32 extremes.
// No host reads of device contents. Caller owns sufficiently sized live device
// buffers until stream completion and checks asynchronous completion errors.
// Strides are float elements; outputs are contiguous [128,heads,tokens,seq].
// Read-only Q/K inputs may overlap. Outputs and scratch must not overlap any
// inputs or each other (bounding-span checks conservatively include padding).
// Baseline: Q RMS -> Q SCALE -> K RMS -> K SCALE, four launches. Scratch has
// two contiguous elements(shape)-float slices. Candidate: one launch, one
// 256-thread block per Q OR K row; grid.z joins the two sequence ranges. It
// removes two SCALE launches and combines the two RMS launches; it does NOT
// combine Q and K reductions or change their 256-thread arithmetic topology.
// PDL/backend scheduling omitted. Baseline arithmetic is frozen separately.
// Compile without fast-math/reassociation. Candidate uses a volatile F32
// intermediate to retain the former RMS output store's rounding. SCALE retains
// `scale*x + bias`, including bias=+0 and signed-zero/FMA behavior. GPU bitwise
// equality to this extraction must be measured; host emulation is insufficient.
#ifdef __FAST_MATH__
#error "gdn_norm requires strict floating-point semantics (no fast-math)"
#endif

namespace strixlab_gdn_norm {
struct Shape { int width = 128, heads = 16, tokens = 1, sequences = 1; };
struct Input {
    const float * data;
    int64_t stride_head, stride_token, stride_sequence;
    float eps; // ORIGINAL graph epsilon, not epsilon/width.
};
namespace detail {
constexpr int threads = 256;
inline bool valid_shape(Shape s) {
    return s.width == 128 && s.heads > 0 && s.heads <= 65535 &&
           s.tokens > 0 && s.tokens <= 65535 && s.sequences > 0 &&
           s.sequences <= 32767;
}
// Limits also make all output/launch products representable in uint64_t.
inline uint64_t count(Shape s) {
    return uint64_t(s.width) * uint64_t(s.heads) * uint64_t(s.tokens) * uint64_t(s.sequences);
}
struct Span { uintptr_t first, last; }; // half-open, checked for wrap
inline bool span(const void * p, uint64_t bytes, Span & out) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    if (!p || a % alignof(float) || !bytes || bytes > std::numeric_limits<uintptr_t>::max() - a) return false;
    out = {a, a + uintptr_t(bytes)};
    return true;
}
inline bool input_span(Input in, Shape s, Span & out) {
    if (!(std::isfinite(in.eps) && in.eps > 0) ||
        !(in.eps / float(s.width) > 0) || in.stride_head < s.width ||
        in.stride_token <= 0 || in.stride_sequence <= 0) return false;
    // Reject arithmetic overflow before computing any span or device offsets.
    uint64_t extent = uint64_t(s.width);
    const uint64_t limit = uint64_t(std::numeric_limits<int64_t>::max()) / sizeof(float);
    const auto extend = [&](int n, int64_t stride) {
        if (uint64_t(stride) < extent || uint64_t(n - 1) > (limit - extent) / uint64_t(stride)) return false;
        extent += uint64_t(n - 1) * uint64_t(stride);
        return true;
    };
    if (!extend(s.heads, in.stride_head) || !extend(s.tokens, in.stride_token) ||
        !extend(s.sequences, in.stride_sequence)) return false;
    return span(in.data, extent * sizeof(float), out);
}
inline bool overlap(Span a, Span b) { return a.first < b.last && b.first < a.last; }
inline bool valid(Input q, Input k, float * qo, float * ko, float * scratch, Shape s, bool baseline) {
    if (!valid_shape(s)) return false;
    Span qi{}, ki{}, oq{}, ok{}, tmp{};
    const uint64_t bytes = count(s) * sizeof(float);
    if (!input_span(q, s, qi) || !input_span(k, s, ki) ||
        !span(qo, bytes, oq) || !span(ko, bytes, ok)) return false;
    if (overlap(oq, ok) || overlap(oq, qi) || overlap(oq, ki) ||
        overlap(ok, qi) || overlap(ok, ki)) return false;
    if (baseline && (!span(scratch, bytes * 2, tmp) || overlap(tmp, qi) ||
        overlap(tmp, ki) || overlap(tmp, oq) || overlap(tmp, ok))) return false;
    return true;
}

__device__ inline float warp_sum(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) x += __shfl_xor(x, offset, 32);
    return x;
}
__device__ inline float block_sum(float val, float * shared_vals) {
    val = warp_sum(val);
    const int warp_id = threadIdx.x / 32, lane_id = threadIdx.x % 32;
    if (lane_id == 0) shared_vals[warp_id] = val;
    __syncthreads();
    val = 0.0f;
    if (lane_id < 8) val = shared_vals[lane_id];
    return warp_sum(val);
}

// Frozen bounded extraction of rms_norm_f32<256,false>. Do not substitute
// corrected-L2 algebra here: the divisions, rsqrt and store are the baseline.
__global__ void baseline_rms(const float * x, float * dst, int ncols,
                            int64_t stride_row, int64_t stride_channel,
                            int64_t stride_sample, float eps) {
    assert(warpSize == 32);
    const int nrows = gridDim.x, nchannels = gridDim.y;
    const int row = blockIdx.x, channel = blockIdx.y, sample = blockIdx.z;
    const int tid = threadIdx.x;
    x += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((int64_t(sample)*nchannels + channel)*nrows + row)*ncols;
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += 256) { const float xi = x[col]; tmp += xi * xi; }
    extern __shared__ float s_sum[];
    tmp = block_sum(tmp, s_sum);
    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);
    for (int col = tid; col < ncols; col += 256) dst[col] = scale * x[col];
}
__global__ void baseline_scale(const float * x, float * dst, float scale, float bias, int64_t nelements) {
    int64_t tid = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    int64_t stride = int64_t(blockDim.x)*gridDim.x;
    for (int64_t i = tid; i < nelements; i += stride) dst[i] = scale * x[i] + bias;
}

__global__ void fused_qk(Input q, Input k, float * qo, float * ko, Shape s,
                         float q_eps, float k_eps, float post_scale, float bias) {
    assert(warpSize == 32);
    const bool is_k = int(blockIdx.z) >= s.sequences;
    const int sample = int(blockIdx.z) - (is_k ? s.sequences : 0);
    const Input in = is_k ? k : q;
    const float * x = in.data + int64_t(sample)*in.stride_sequence +
        int64_t(blockIdx.y)*in.stride_token + int64_t(blockIdx.x)*in.stride_head;
    float * dst = (is_k ? ko : qo) +
        ((int64_t(sample)*s.tokens + blockIdx.y)*s.heads + blockIdx.x)*s.width;
    const int tid = threadIdx.x;
    float tmp = 0.0f;
    for (int col = tid; col < s.width; col += 256) { const float xi = x[col]; tmp += xi * xi; }
    extern __shared__ float s_sum[];
    tmp = block_sum(tmp, s_sum);
    const float mean = tmp / s.width;
    const float scale = rsqrtf(mean + (is_k ? k_eps : q_eps));
    for (int col = tid; col < s.width; col += 256) {
        volatile float normalized = scale * x[col];
        dst[col] = post_scale * normalized + bias;
    }
}
} // namespace detail

inline size_t elements(Shape s) {
    return detail::valid_shape(s) ? size_t(detail::count(s)) : 0;
}
inline size_t scratch_bytes(Shape s) { return 2 * elements(s) * sizeof(float); }
inline hipError_t baseline(Input q, Input k, float * qo, float * ko, float * scratch, Shape s, hipStream_t stream) {
    if (!detail::valid(q,k,qo,ko,scratch,s,true)) return hipErrorInvalidValue;
    const int64_t n = int64_t(elements(s));
    const dim3 grid(s.heads,s.tokens,s.sequences), block(256);
    const dim3 scale_grid(unsigned((n + 255)/256 > 0x7fffffff ? 0x7fffffff : (n + 255)/256));
    const float post_scale = 1.0f/sqrtf(float(s.width));
    // Each error check is immediate; callers additionally check completion.
    hipLaunchKernelGGL(detail::baseline_rms,grid,block,32*sizeof(float),stream,
        q.data,scratch,s.width,q.stride_head,q.stride_token,q.stride_sequence,q.eps/float(s.width));
    hipError_t err = hipGetLastError(); if (err != hipSuccess) return err;
    hipLaunchKernelGGL(detail::baseline_scale,scale_grid,block,0,stream,scratch,qo,post_scale,0.0f,n);
    err = hipGetLastError(); if (err != hipSuccess) return err;
    hipLaunchKernelGGL(detail::baseline_rms,grid,block,32*sizeof(float),stream,
        k.data,scratch+n,s.width,k.stride_head,k.stride_token,k.stride_sequence,k.eps/float(s.width));
    err = hipGetLastError(); if (err != hipSuccess) return err;
    hipLaunchKernelGGL(detail::baseline_scale,scale_grid,block,0,stream,scratch+n,ko,post_scale,0.0f,n);
    return hipGetLastError();
}
inline hipError_t fused(Input q, Input k, float * qo, float * ko, Shape s, hipStream_t stream) {
    if (!detail::valid(q,k,qo,ko,nullptr,s,false)) return hipErrorInvalidValue;
    const dim3 grid(s.heads,s.tokens,2*s.sequences), block(256);
    hipLaunchKernelGGL(detail::fused_qk,grid,block,32*sizeof(float),stream,
        q,k,qo,ko,s,q.eps/float(s.width),k.eps/float(s.width),1.0f/sqrtf(float(s.width)),0.0f);
    return hipGetLastError();
}
} // namespace strixlab_gdn_norm

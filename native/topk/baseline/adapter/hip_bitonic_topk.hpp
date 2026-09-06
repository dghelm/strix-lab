#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>

// Standalone extraction of the pinned HIP bitonic path. The caller owns src,
// dst, and scratch; src is contiguous row-major F32 storage and dst is I32
// storage with rows*k elements. src and dst must not alias. These datatype and
// aliasing requirements are an interface contract, not runtime type evidence.
// src, dst, and scratch must be pairwise non-overlapping and remain valid until
// all work queued on stream completes. scratch_capacity is an element count
// (not a byte count); it must provide rows*columns int elements. The kernel
// writes only the unpadded columns in each row.
hipError_t strixlab_baseline_topk_hip(const float * src,
                                      int *       dst,
                                      int         rows,
                                      int         columns,
                                      int         k,
                                      int *       scratch,
                                      std::size_t scratch_capacity,
                                      hipStream_t stream);

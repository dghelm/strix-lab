#include "hip_bitonic_topk.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

static_assert(sizeof(float) == 4, "F32 contract requires 32-bit float");
static_assert(sizeof(int) == 4, "I32 contract requires 32-bit int");

namespace {

__device__ __forceinline__ void swap_int(int & a, int & b) {
    int tmp = a;
    a = b;
    b = tmp;
}

template<bool Ascending>
__global__ void bitonic_rows(const float * x, int * dst, int rows, int columns, int padded_columns) {
    const int col = static_cast<int>(threadIdx.x);
    const int row = static_cast<int>(blockIdx.x);
    if (col >= padded_columns || row >= rows) {
        return;
    }

    const float * x_row = x + static_cast<std::size_t>(row) * columns;
    extern __shared__ int indices[];
    indices[col] = col;
    __syncthreads();

    for (int k = 2; k <= padded_columns; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = col ^ j;
            if (ixj > col) {
                if ((col & k) == 0) {
                    if (indices[col] >= columns ||
                        (indices[ixj] < columns &&
                         (Ascending ? x_row[indices[col]] > x_row[indices[ixj]]
                                    : x_row[indices[col]] < x_row[indices[ixj]]))) {
                        swap_int(indices[col], indices[ixj]);
                    }
                } else if (indices[ixj] >= columns ||
                           (indices[col] < columns &&
                            (Ascending ? x_row[indices[col]] < x_row[indices[ixj]]
                                       : x_row[indices[col]] > x_row[indices[ixj]]))) {
                    swap_int(indices[col], indices[ixj]);
                }
            }
            __syncthreads();
        }
    }

    if (col < columns) {
        dst[static_cast<std::size_t>(row) * columns + col] = indices[col];
    }
}

int next_power_of_two(int value) {
    int result = 1;
    while (result < value) {
        if (result > std::numeric_limits<int>::max() / 2) {
            return 0;
        }
        result *= 2;
    }
    return result;
}

} // namespace

hipError_t strixlab_baseline_topk_hip(const float * src,
                                      int *       dst,
                                      int         rows,
                                      int         columns,
                                      int         k,
                                      int *       scratch,
                                      std::size_t scratch_capacity,
                                      hipStream_t stream) {
    if (src == nullptr || dst == nullptr || scratch == nullptr || rows <= 0 || columns <= 0 ||
        k <= 0 || k > columns || columns > 1024) {
        return hipErrorInvalidValue;
    }
    const int padded_columns = next_power_of_two(columns);
    if (padded_columns == 0 || padded_columns > 1024) {
        return hipErrorInvalidValue;
    }
    if (static_cast<std::size_t>(rows) > std::numeric_limits<std::size_t>::max() /
                                             static_cast<std::size_t>(columns)) {
        return hipErrorInvalidValue;
    }
    const std::size_t elements = static_cast<std::size_t>(rows) * columns;
    if (scratch_capacity < elements ||
        elements > std::numeric_limits<std::size_t>::max() / sizeof(int)) {
        return hipErrorInvalidValue;
    }

    const dim3 block(static_cast<unsigned>(padded_columns), 1, 1);
    const dim3 grid(static_cast<unsigned>(rows), 1, 1);
    const std::size_t shared_bytes = static_cast<std::size_t>(padded_columns) * sizeof(int);
    hipLaunchKernelGGL((bitonic_rows<false>), grid, block, shared_bytes, stream,
                       src, scratch, rows, columns, padded_columns);
    hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        return error;
    }
    return hipMemcpy2DAsync(dst, static_cast<std::size_t>(k) * sizeof(int),
                            scratch, static_cast<std::size_t>(columns) * sizeof(int),
                            static_cast<std::size_t>(k) * sizeof(int), rows,
                            hipMemcpyDeviceToDevice, stream);
}

#pragma once
// Synthetic HIP for correctness testing only: execute actual kernel bodies with
// one host thread per lane. Blocks run sequentially; shared arrays are static.
#include <array>
#include <cassert>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#define __host__
#define __device__
#define __global__
#define __shared__ static
struct dim3 { unsigned x; explicit dim3(unsigned value = 1) : x(value) {} };
inline thread_local dim3 threadIdx, blockIdx;
inline int warpSize = 32;
using hipStream_t = void*;
using hipError_t = int;
constexpr int hipSuccess = 0, hipErrorInvalidValue = 1;
inline int host_error = 0;
inline unsigned host_launches = 0, host_block_size = 0;
struct Barrier {
    explicit Barrier(unsigned count) : count(count) {}
    const unsigned count;
    unsigned arrived = 0, generation = 0;
    std::mutex mutex;
    std::condition_variable condition;
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        const auto old = generation;
        if (++arrived == count) { arrived = 0; ++generation; condition.notify_all(); }
        else condition.wait(lock, [&] { return generation != old; });
    }
};
struct Wave {
    Barrier barrier{32};
    std::array<std::array<unsigned char, 4>, 32> slots{};
};
inline thread_local Barrier* host_block = nullptr;
inline thread_local Wave* host_wave = nullptr;
inline void __syncthreads() { host_block->wait(); }
template<class T> T __shfl_down(T value, unsigned delta, int width) {
    static_assert(sizeof(T) == 4);
    assert(width == 32);
    const unsigned lane = threadIdx.x % 32;
    std::memcpy(host_wave->slots[lane].data(), &value, 4);
    host_wave->barrier.wait(); // Every physical lane must participate.
    T result;
    const unsigned source = lane + delta < 32 ? lane + delta : lane;
    std::memcpy(&result, host_wave->slots[source].data(), 4);
    host_wave->barrier.wait(); // No next shuffle may overwrite unread values.
    return result;
}
template<class Function, class... Args>
void host_launch(Function fn, dim3 grid, dim3 block, hipStream_t, Args... args) {
    ++host_launches;
    host_block_size = block.x;
    assert(block.x == 32 || block.x == 256);
    for (unsigned row = 0; row < grid.x; ++row) {
        Barrier barrier(block.x);
        std::array<Wave, 8> waves;
        std::vector<std::thread> threads;
        for (unsigned lane = 0; lane < block.x; ++lane) {
            threads.emplace_back([&, lane, row] {
                threadIdx = dim3(lane); blockIdx = dim3(row);
                host_block = &barrier; host_wave = &waves[lane / 32];
                fn(args...);
            });
        }
        for (auto& thread : threads) thread.join();
    }
}
#define hipLaunchKernelGGL(fn, grid, block, shared, stream, ...) \
    host_launch(fn, grid, block, stream, __VA_ARGS__)
inline hipError_t hipGetLastError() { return host_error; }

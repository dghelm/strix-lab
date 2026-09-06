#pragma once
// CPU-only HIP execution model. Actual kernel bodies, all 256 lanes, true
// block/wave barriers and 3D grids. rsqrt uses host sqrt, NOT GPU instruction
// semantics; passing this harness never establishes GPU numerical parity.
#include <array>
#include <cassert>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#define __host__
#define __device__
#define __global__
#define __shared__
struct dim3 { unsigned x,y,z; explicit dim3(unsigned x=1,unsigned y=1,unsigned z=1):x(x),y(y),z(z){} };
inline thread_local dim3 threadIdx,blockIdx,gridDim,blockDim;
inline int warpSize=32;
using hipStream_t=void*;
using hipError_t=int;
constexpr hipError_t hipSuccess=0,hipErrorInvalidValue=1;
struct Barrier {
    explicit Barrier(unsigned count):count(count){}
    unsigned count,arrived=0,generation=0;
    std::mutex mutex;
    std::condition_variable condition;
    void wait() {
        std::unique_lock<std::mutex> lock(mutex);
        const auto old=generation;
        if (++arrived==count) { arrived=0; ++generation; condition.notify_all(); }
        else condition.wait(lock,[&]{return generation!=old;});
    }
};
struct Wave { Barrier barrier{32}; std::array<float,32> slots{}; };
inline thread_local Barrier* host_block=nullptr;
inline thread_local Wave* host_wave=nullptr;
inline void __syncthreads(){host_block->wait();}
inline float __shfl_xor(float v,int offset,int width) {
    assert(width==32);
    const auto lane=threadIdx.x%32;
    host_wave->slots[lane]=v;
    host_wave->barrier.wait();
    const float out=host_wave->slots[lane^offset];
    host_wave->barrier.wait();
    return out;
}
inline float rsqrtf(float x){return 1.0f/std::sqrt(x);}
namespace strixlab_gdn_norm { namespace detail { inline float s_sum[32]; } }
struct Launch {dim3 grid,block;size_t shared;};
inline std::vector<Launch> host_launches;
inline size_t host_fail_at=0;
inline int host_error=0;
inline bool host_record_only=false;
inline hipError_t hipGetLastError(){int e=host_error;host_error=0;return e;}
template<class F,class... Args>
void host_launch(F f,dim3 grid,dim3 block,size_t shared,hipStream_t,Args... args) {
    host_launches.push_back({grid,block,shared});
    if(host_fail_at==host_launches.size()){host_error=73;return;}
    if(host_record_only)return;
    assert(block.x==256 && block.y==1 && block.z==1);
    Barrier barrier(256);
    std::array<Wave,8> waves;
    std::vector<std::thread> pool;
    for(unsigned lane=0;lane<256;++lane)pool.emplace_back([&,lane]{
        threadIdx=dim3(lane,0,0);gridDim=grid;blockDim=block;
        host_block=&barrier;host_wave=&waves[lane/32];
        for(unsigned z=0;z<grid.z;++z)for(unsigned y=0;y<grid.y;++y)for(unsigned x=0;x<grid.x;++x){
            blockIdx=dim3(x,y,z);f(args...);barrier.wait();
        }
    });
    for(auto& t:pool)t.join();
}
#define hipLaunchKernelGGL(fn,grid,block,shared,stream,...) host_launch(fn,grid,block,shared,stream,__VA_ARGS__)

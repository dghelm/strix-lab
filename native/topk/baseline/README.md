# Pinned HIP baseline source record

This directory records the exact source boundary proposed for the bounded
`baseline-hip` extraction. The files under `upstream/` are byte-for-byte copies
from `halo-box/strix-llama.cpp` at commit
`ca94157f70a2776e8da6b6849b50b45a083d0478`; their blob IDs and local SHA-256
values are in `provenance.json`. The repository MIT license is retained.

This is source and conformance evidence, not a production target or manifest.
The repository's CMake has a host-only regression-test target for the network;
it does not define a GPU baseline target or launch path.
The HIP preprocessor disables `GGML_CUDA_USE_CUB`, so the active path is the
single all-row bitonic launch in `top-k.cu` followed by one 2-D device copy.
Capability admission remains `columns <= 1024`; larger TOPK v1 shapes are
explicitly unsupported by this extraction. No guard is removed and no long-row
algorithm is introduced.

The extracted call boundary requires contiguous F32 input, I32 index output,
row-major input (`x + row * columns`), one block per row, padded power-of-two
thread count, and `padded_columns * sizeof(int)` dynamic shared memory. The
host regression test mirrors the unchanged comparison network and retains the
`columns=11, K=10` equal-value membership counterexample. It is a source-level
simulation, never GPU evidence.

If an existing ROCm 7.2.4 `hipcc` is available, a future read-only check may
compile this source boundary with the exact command, compiler version, and
preprocessor defines recorded as control-toolchain evidence. That check must not
launch a kernel or imply ROCm 10/gfx1151 support. No GPU CMake target, provider
registry, scenario manifest, value gathering, NaN preflight, or tie repair is
defined here.

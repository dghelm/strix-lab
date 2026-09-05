# Dependency-isolated HIP adapter provenance

This adapter is derived from the pinned files in `../upstream/` at commit
`ca94157f70a2776e8da6b6849b50b45a083d0478`. The comparison kernel body and its
all-row launch structure are derived from `upstream/argsort.cu`, lines 160–249;
the 2-D device copy shape is derived from `upstream/top-k.cu`, lines 99–102.
The upstream files are unchanged. Host ggml context/pool types are replaced by
an explicit caller-owned scratch pointer and capacity. The public interface
documents contiguous row-major F32 input, I32 output, pairwise non-overlapping
buffers, and buffer validity through completion of queued stream work; it does
not claim runtime datatype proof. `scratch_capacity` is an element count. The
adapter's differences are limited to `size_t` row offsets, explicit bounds,
overflow, pointer, and scratch-capacity checks returning `hipErrorInvalidValue`,
and caller-owned scratch instead of ggml pool/context classes. No value
gathering, NaN handling, tie repair, provider registration, or production
target is included.

Control-toolchain command (compile only; no link, run, or GPU launch):

```sh
/opt/rocm/bin/hipcc --version
/opt/rocm/bin/hipcc --offload-arch=gfx1151 -std=c++17 -I native/topk/baseline/adapter \
  -c native/topk/baseline/adapter/hip_bitonic_topk.cu \
  -o /tmp/strixlab-hip-bitonic-topk.o
```

The persistent compiler/object receipt is `compile-receipt.md` beside this
file; it records the complete compiler version, command/result, source
contents SHA-256 values, object path, and object SHA-256. Compilation was
compile-only; no link, run, or GPU execution was performed.

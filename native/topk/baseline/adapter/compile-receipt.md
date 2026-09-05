# HIP compile receipt

Control compiler (local, no GPU execution):

```text
HIP version: 7.2.53211-9999
AMD clang version 22.0.0git (/srcdest/rocm-llvm f58b06dce1f9c15707c5f808fd002e18c2accf7e)
Target: x86_64-pc-linux-gnu
Thread model: posix
InstalledDir: /opt/rocm/lib/llvm/bin
```

Command:

```text
/opt/rocm/bin/hipcc --offload-arch=gfx1151 -std=c++17 -I native/topk/baseline/adapter -c native/topk/baseline/adapter/hip_bitonic_topk.cu -o /tmp/strixlab-hip-bitonic-topk.o
```

Result: exit status `0`; `/tmp/strixlab-hip-bitonic-topk.o` is an ELF64
x86-64 relocatable object. No link, run, or GPU launch was performed.

SHA-256 values from this receipt's compile:

```text
c1b8d9b4cf69dff5056eb5f2233ab01821766bb4972ecafe2814e2131b63e319  hip_bitonic_topk.hpp
f8eaf6c57ac0e558323518ccf39ce951e610f34203fd3861997162e510e57672  hip_bitonic_topk.cu
9baa652317227601e6d22817b893372fb8e06acd4cbb2819baea598f2ce12e4e  PROVENANCE.md
a903a0d16328c09e34bf5c6d766c5bd37b191c72c0ab6c4e099dea32126b9bf4  /tmp/strixlab-hip-bitonic-topk.o
```

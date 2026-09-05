# TOPK-001 host correctness foundation

This C++17 library implements the matrix, `topk-input-v1` generator and digest,
NaN preflight, stable CPU reference, and value/index validation specified in
[the scenario contract](../../docs/rocm10-topk-gfx1151.md). It has no HIP/ROCm,
provider, timing, protocol executable, or runnable scenario manifest. The CPU
reference is independently buildable and testable without any candidate code.
Its output must never canonicalize a provider's measured output on the CPU.

## Required host test dependencies

Install a C++17 compiler, CMake >=3.20, Make (or configure Ninja manually), and
OpenSSL development headers/library (Crypto). On Debian/Ubuntu these are
`build-essential cmake libssl-dev`; no ROCm installation is needed. OpenSSL EVP
provides SHA-256 rather than adding a cryptographic implementation here.

`uv run pytest tests/unit/test_topk_reference.py` builds in pytest temporary
storage and runs native failure vectors plus independent Python generator,
SHA-256, and full-sort comparisons. These tests run under the existing required
CI pytest commands; missing compiler/CMake/OpenSSL dependencies **fail**, never
skip. Tests cover all matrix metadata and all families on cases 1, 3, and 8,
including non-power-of-two row boundaries and near-full K, without sorting the
largest matrix cases on every ordinary unit run. Native checks use exceptions,
so Release/NDEBUG builds retain the checks.

Direct host verification:

```sh
cmake -S native/topk -B /tmp/strixlab-topk-host -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/strixlab-topk-host --parallel 2
ctest --test-dir /tmp/strixlab-topk-host --output-on-failure
```

The native test executable also accepts `matrix` or `CASE_ORDINAL FAMILY_ID`.
The latter prints seed/digest, all input words, and reference pairs for an
explicit deeper check, including larger cases. Large cases produce large
stdout; they are not part of routine verification. No exhaustive largest-case
validation or GPU conformance is claimed by this slice.

## Contract boundary and negative findings

Input words and returned values are raw binary32 bits. Numeric ties, including
signed zeros, use ascending original column index across the selection boundary;
selected bits retain their original sign. Indices are row-local. Family 5 repeats
its pattern over the row-major array, while increasing/decreasing families restart
at each row. The digest explicitly encodes little-endian dimensions and words,
including the domain's terminating NUL. Generator metadata is supplied alongside
the digest; the prescribed digest itself does not encode family or seed.

`preflight` scans every input word without allocating and returns `nan-input` for
quiet/signaling NaNs of either sign. Future wrappers must call it before device
allocation, capture, or launch; this library cannot enforce future call ordering.
Reference and validation also call preflight before scratch/output allocation.
Malformed shapes/sizes and malformed output return distinct structured statuses.
Input generation throws for unknown case/family and digesting throws for invalid
shape/size or cryptographic failures. NaN probe bits remain digestible.

The generic capsule opaque payload has no correctness-admission authority.
These host statuses are not a TOPK protocol interpreter. Pinned rocPRIM API,
stability, boundary-tie wrapper, graph behavior, and provider availability remain
unverified and deferred to the coordinator's separately authorized work.

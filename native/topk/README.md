# TOPK-001 host correctness foundation

This C++17 library implements the matrix, `topk-input-v1` generator and digest,
NaN preflight, stable CPU reference, and value/index validation specified in
[the scenario contract](../../docs/rocm10-topk-gfx1151.md). The production scenario still has no HIP/ROCm provider, timing implementation,
or runnable manifest. A separate compiled host transport fixture is described below. The CPU
reference is independently buildable and testable without any candidate code.
Its output must never canonicalize a provider's measured output on the CPU.

## Required host test dependencies

Install C and C++17 compilers, CMake >=3.20, Make (or configure Ninja manually), and
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


## Compiled native transport fixture (TOPK-001B)

`topk_capsule_host_test` implements the existing `native-capsule-v1` wire contract
with the **test-only** compiled identity `strixlab-topk-host-test-v1`. Its candidate
is `host-fixture`; its scenario SHA is SHA-256 of the ASCII identity, with no NUL
or newline. The two coordinates are `fixture-direct` and `fixture-replay`, both
under the `host-tie-fixture` case/input ID. This fixture uses a tiny six-element
input and hand-specified output, not the production matrix or generator. Its
benchmark latencies are explicitly synthetic `[0.001, 0.002, 0.003, 0.004, 0.005]`
seconds for each coordinate, and its declared warmup is synthetic too. No device,
HIP stream, event, graph or GPU timing exists in this executable.

The accepted invocation is exactly:

```text
<executable> describe|correctness|benchmark --request /proc/self/fd/N
```

The inherited descriptor must match `capsules.py`: read-only, regular, sealed with
`F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE`, immutable size/identity.
There is no extra memfd name, link-count or descriptor-number requirement. The
reader uses `pread`, since the existing runner can leave the shared file offset
at EOF. Requests are bounded to 1 MiB. The native parser rejects duplicate or
unknown keys, invalid UTF-8, numeric coercions, nonfinite/overflow values, wrong
operation shapes/bindings, and noncanonical bytes. It checks the executable hash
against `/proc/self/exe`. The fixture accepts only its exact ASCII/string/integer
request domain; it does not claim arbitrary cross-language float serialization.
Response bytes are checked against Python's canonical serializer and generic
response models in all three phases. Rejections produce bounded fixed stderr,
never reflected request text.

The trusted native gate checks input before simulated setup/operation callbacks,
validates independent output pairs against the CPU reference, and compares both
fixture modes bit-for-bit. Its result derives the outer correctness flags; the
opaque payload only contains test diagnostics. Every benchmark process runs the
gate again before emitting samples: a prior response SHA is an echoed chain
binding, not prior response content or a readiness receipt. Compile-time fault
fixtures include one whose correctness process passes but whose benchmark-local
readiness fails. No request field, environment setting or runtime flag selects a
fault. Real IDs `baseline-hip`, `rocprim-topk`, and `rocprim-segmented-topk` always
fail correctness as unavailable, with no provider setup or benchmark samples.
Production scenario IDs are rejected outright.

Run the transport/gate checks with:

```sh
uv run pytest tests/unit/test_topk_capsule_transport.py tests/unit/test_topk_reference.py
```

Pytest compiles the actual executable plus fault variants, checks canonical FD
transport directly and through the real `run_capsule_protocol`, and verifies
failure suppresses benchmark invocation and executable drift fails closed. The
fixture manifest exists only in test memory; its schema-required gfx field is
never treated as a host hardware observation. These are component/transport
tests, **not production `run_capsule` admission**, finalized production TOPK
results, exhaustive matrix correctness, or rocPRIM conformance.

## Fixed build interface and vendored dependency

The paired Python build-policy work owns source/build/lease authentication.
The native interface is source subtree `native/topk`, fixed source adapter
`strixlab_native`, executable target `topk_capsule_host_test`, and CMake
`LANGUAGES C CXX`. The adapter supplies reserved cache values
`STRIXLAB_NATIVE_BUILD_COMMIT` (its exact-base plus `-dirty` rule) and
`STRIXLAB_NATIVE_BUILD_NUMBER=0` and checks them during build authentication.
Direct standalone builds default to `unbound-host-test`, making no source-version
attestation. Host builds define no LLAMA/GGML/HIP/gfx metadata; production host
admission is not added to satisfy this fixture.

The official nlohmann JSON `v3.12.0` single header and MIT license are vendored
unchanged at commit `55f93686c01528224f448c19128836e7df245f72`. Exact upstream URLs
and SHA-256 identities are in [provenance.json](third_party/nlohmann/provenance.json).
Tests verify those content hashes. Configure/build/runtime never download a
JSON dependency, and no system JSON package is required. OpenSSL remains the
existing host Crypto dependency. Upstream JSON notices are retained in both
header and license.

Negative verification findings: the local Python build does not expose named
seal constants or `os.memfd_create`, so tests use the existing runner's memfd
helper and Linux ABI constants, as the runner itself does. Production provider,
canonical scenario payload interpreter, real device/graph readiness, and
ROCm/gfx leased execution remain separate gates. This fixture does not widen
those contracts or freeze future timing/conformance schemas.

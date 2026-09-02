# Planned ROCm 10 top-k scenario for gfx1151

> **Status: planned; not runnable.** The generic native-capsule protocol, production
> runner, finalized snapshot authentication, closed comparison contract, and pure offline
> directional comparison are delivered. There is still intentionally no runnable top-k
> capsule manifest: comparison evidence/CLI dispatch and TOPK-001 payload/provider
> semantics remain deferred. Do not
> create experiments until the trusted TOPK capsule and versioned scenario manifest land.

The planned scenario asks which one trusted top-k provider gives the best
correct behavior on gfx1151 across small routing rows and long sparse-attention
rows under both eager execution and HIP graph replay. It is a local,
evidence-producing scenario, not a hosted challenge, score, hidden judge,
submission service, campaign engine, or claim that one provider wins every
shape.

## Current gap

StrixLab can bind a run to a machine profile, pinned source and build records, a
resolved manifest digest, immutable evidence, a correctness-first schedule, portable
bundles, and a conservative model-suite comparison. It also has the generic capsule
manifest/protocol/runner/snapshot boundary and the closed scenario-neutral capsule
comparison contract. This top-k scenario is still inactive because:

- `SuiteManifestV1` requires a model plus `test-backend-ops`, `llama-server`,
  and `llama-bench` cases.
- `run_suite` leases exactly those three llama.cpp executables and emits
  model-bound suite results.
- the comparison judge authenticates and compares those finalized model-suite
  results and their throughput samples.
- derived comparison evidence and CLI dispatch are deferred;
- there is no trusted top-k payload interpreter, provider implementation/registry,
  correctness reference, or runnable capsule manifest.

Forcing top-k through the llama-server adapter would obscure the operation boundary and
invent model inputs that the capsule does not need. The delivered capsule seam remains
narrow and has no adapter plugin ABI or generic workflow engine; the missing work is
comparison publication/dispatch and the scenario-specific trusted TOPK capsule.

## Fixed scenario contract

The first executable version should use the ID `rocm10-topk-gfx1151-v1`. Any
material change to the matrix, correctness semantics, timing boundary, or
comparison policy requires a new scenario ID.

### Source, toolchain, and hardware binding

Every baseline and candidate arm must bind and record all of the following:

- the full StrixLab commit containing the executable scenario and capsule;
- the full source commit and patch digest for the provider implementation;
- the exact capsule build recipe, executable digest, compiler identity, CMake
  and Ninja versions, dynamic-library closure, and environment snapshot;
- ROCm Core SDK `10.0.0`, rocPRIM `4.6.0`, and the resolved installation prefix;
- AMD's official `ROCm/rocm-libraries` tag `therock-10.0`, whose annotated tag
  resolves to commit
  `8d1ae90eff7d022f26019ec55b2ec6a7674b3112`;
- a resolved machine profile whose observed GPU target is exactly `gfx1151`,
  plus the ordinary exclusive lock and validity observations.

The official ROCm 10 release note describes newly added top-k operations and a
stable variant, while the rocPRIM 4.6 API material at the pinned source uses
different public spellings and contains conflicting stability notes. The
relevant primary sources are:

- [ROCm 10.0.0 release notes](https://rocm.docs.amd.com/en/docs-10.0.0/about/release-notes.html)
- [`therock-10.0` source tree at the resolved commit](https://github.com/ROCm/rocm-libraries/tree/8d1ae90eff7d022f26019ec55b2ec6a7674b3112/projects/rocprim)
- [top-k public header at that commit](https://github.com/ROCm/rocm-libraries/blob/8d1ae90eff7d022f26019ec55b2ec6a7674b3112/projects/rocprim/rocprim/include/rocprim/device/device_topk.hpp)
- [segmented top-k public header at that commit](https://github.com/ROCm/rocm-libraries/blob/8d1ae90eff7d022f26019ec55b2ec6a7674b3112/projects/rocprim/rocprim/include/rocprim/device/device_segmented_topk.hpp)
- [rocPRIM top-k API source at that commit](https://github.com/ROCm/rocm-libraries/blob/8d1ae90eff7d022f26019ec55b2ec6a7674b3112/projects/rocprim/docs/device_ops/topk.rst)

This disagreement is a hard gate, not a documentation detail. TOPK-001 must
compile a small conformance fixture against that exact commit, capture the
actual enabled headers and callable signatures, and test ordering, tie, and
graph behavior. Until it passes, the scenario must not claim a concrete C++ API
spelling or stable-ordering capability, and the rocPRIM providers remain
unavailable. Changing the source tag or commit invalidates that verification.

### Candidate surface

The v1 candidate is deliberately smaller than a dispatch-policy challenge. One
experiment selects exactly one trusted provider ID for the complete public
matrix:

- `baseline-hip`: the reviewed existing HIP top-k path extracted into the
  capsule;
- `rocprim-topk`: the pinned rocPRIM non-segmented provider, available only
  after the source conformance gate passes;
- `rocprim-segmented-topk`: the pinned rocPRIM segmented provider, available
  only after the same gate passes.

The provider IDs are scenario vocabulary, not promises about upstream C++
symbol names. The trusted capsule owns all allocation, launch, graph, ordering,
and validation code. A selection candidate contains only one provider ID; it
cannot supply flags, source, binaries, build commands, or per-shape dispatch.

For a multi-row case, `rocprim-topk` executes exactly one row operation per row,
in row-index order, on the same stream. It queries and allocates one workspace
large enough for a single row, reuses that workspace for every row, and measures
the complete ordered row batch as one sample. Its captured graph contains every
row operation and the required output ordering. `rocprim-segmented-topk` executes
the complete row batch as one segmented operation and uses one batch workspace.
`baseline-hip` executes the complete row batch in one GPU provider launch, with
rows mapped inside that launch, and uses one batch workspace. For both batch
providers, the captured graph contains that batch operation plus required output
ordering. These batching rules are part of provider identity and cannot vary by
candidate.

A reviewed source experiment may instead patch the implementation behind
exactly one provider ID. Such a patch is materialized through StrixLab's normal
pinned-source and build path, reviewed in its experiment PR, and identified by
its source/build evidence. It cannot modify the capsule reference, matrix,
correctness gate, timing, evidence projection, or comparison policy. There is
no remote execution or source-submission path.

Per-shape dispatch tables, hipCUB/radix-sort alternatives, provider parameters,
and additional kernels are possible later scenario versions after v1 produces
honest evidence. They are not part of this first public surface.

### Exact reduced public matrix

The reduced matrix is intentionally small enough for early feedback while
covering single-row, segmented, long-row, near-row-length K, and non-power-of-two
cases. `columns` is the length of each row; each case emits exactly `rows * k`
value/index pairs.

| Case | Set | Rows | Columns | K |
|---|---|---:|---:|---:|
| `train-r1-c128-k1` | training | 1 | 128 | 1 |
| `train-r1-c4096-k20` | training | 1 | 4,096 | 20 |
| `train-r2-c16381-k8` | training | 2 | 16,381 | 8 |
| `train-r8-c65536-k20` | training | 8 | 65,536 | 20 |
| `train-r32-c16384-k64` | training | 32 | 16,384 | 64 |
| `train-r128-c4096-k256` | training | 128 | 4,096 | 256 |
| `eval-r1-c262144-k64` | evaluation | 1 | 262,144 | 64 |
| `eval-r2-c257-k256` | evaluation | 2 | 257 | 256 |
| `eval-r8-c16381-k20` | evaluation | 8 | 16,381 | 20 |
| `eval-r32-c65536-k64` | evaluation | 32 | 65,536 | 64 |
| `eval-r128-c262144-k20` | evaluation | 128 | 262,144 | 20 |

Every shape runs correctness in eager and captured-graph replay modes against
each accepted input family:

1. deterministic uniform finite `float32` values;
2. tie-heavy duplicates drawn from a fixed finite value table;
3. strictly increasing rows;
4. strictly decreasing rows;
5. finite values mixed with `+0`, `-0`, `+infinity`, and `-infinity`.

These are all public cases. The `evaluation` subset fixes which cases determine
the verdict, but it is not hidden and must not be described as a generalization
holdout. Contributors can inspect and tune against it.

Input generation is `topk-input-v1`. Number the table cases from 1 through 11
in their displayed order. For each case/family, initialize one SplitMix64 state
to `0x5354524958544f50 ^ (case_ordinal << 8) ^ family_id`. For each row-major
element, with all arithmetic modulo 2^64, compute:

```text
state = state + 0x9e3779b97f4a7c15
z = state
z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
z = (z ^ (z >> 27)) * 0x94d049bb133111eb
output = z ^ (z >> 31)
```

Family IDs and values are fixed as follows:

- `1`, uniform: output bit 63 supplies the sign, the binary32 exponent
  is `126`, and the low 23 output bits supply the mantissa. This constructs
  finite values with magnitudes in `[0.5, 1.0)` without host rounding.
- `2`, duplicates: select `output % 6` from binary32 bit patterns
  `c0400000`, `bf800000`, `80000000`, `00000000`, `3f800000`, `40400000`.
- `3`, increasing: column `j` is the exactly representable binary32 value `j`.
- `4`, decreasing: column `j` is the exactly representable binary32 value
  `columns - j`.
- `5`, special: repeat binary32 patterns `ff800000`, `bf800000`, `80000000`,
  `00000000`, `3f800000`, `7f800000`, `c0000000`, `40000000`.

Families 1 and 2 consume the SplitMix64 output. Families 3 through 5 are the
fixed constructions above and do not advance the generator.

The input digest is SHA-256 over ASCII `strixlab.topk.input.v1` followed by one
NUL byte, unsigned 64-bit little-endian `rows`, `columns`, and `k`, then the
row-major binary32 bit patterns encoded little-endian. The capsule records the
generator ID, initial state, family ID, dimensions, and digest. The NaN probe is
`rows=1, columns=128, k=1`, uses the uniform vector with element zero replaced
by quiet-NaN bits `7fc00001`, and binds generation to case ordinal 1 and family
ID 1 (initial state `0x5354524958544e51`). It runs once per provider and
execution mode. It is not a performance case. Performance sampling uses only
family 1.

After v1 passes correctness and produces useful replications, a new scenario
version may expand toward columns `128` through `1,048,576`, rows `1` through
`1,024`, K values `1, 2, 8, 20, 64, 256`, more non-power-of-two lengths, and
model-derived sparse-attention shapes. Expansion is not silent mutation of v1.

### Reference correctness

Inputs are row-major `float32`. For every row, the operation returns the K
largest values and their zero-based column indices. The required output order is
descending numeric value, then ascending original column index for equal
values. This makes duplicates deterministic and defines stable tie behavior
without depending on a provider's native output order.

The CPU reference applies that ordering directly to `(value, index)` pairs.
Candidate output must contain exactly K distinct in-range indices per row, the
bit-identical input value at each reported index, and the exact reference order.
`+0` and `-0` compare numerically equal and therefore break ties by input index;
the selected value retains its original sign bit. Infinities participate in the
ordinary numeric order. Any NaN in an input row is rejected before allocation,
capture, or provider launch with a structured `nan-input` correctness result.
NaNs are never silently ordered or included in timing.

Selection is lexicographic over every input `(value, index)` pair, not merely a
sort of whichever K items the provider returned. At a tie crossing the K
boundary, the lowest original indices must win. If a provider does not natively
enforce that membership and order, its trusted wrapper must perform explicit
GPU-side boundary-tie resolution and final ordering. All of that work is part
of the measured operation. CPU canonicalization or fallback is forbidden.
Correctness must pass before any performance result or comparison is
interpreted, and graph replay must reproduce the exact eager value/index output.

### Execution modes and timing boundary

Each performance case has two independently reported modes:

- `eager`: direct provider execution on one fixed HIP stream;
- `graph-replay`: capture the complete measured operation once, instantiate it,
  and replay the resulting graph on that stream.

Graph capture is a required capability, not a best-effort metric. Capture or
instantiation failure makes that provider arm correctness-ineligible. Capture
latency is recorded separately and is never mixed into graph-replay latency.

For each case and mode the capsule performs five unreported warmups followed by
30 ordered measurements. Before warmups it separately records the first-call
latency. Device-side timing uses HIP events on the operation stream and includes
the provider kernels plus any required GPU-side value/index ordering. For graph
replay it also includes graph launch and execution. The measured interval ends
only when the completion event has finished.

The measured interval excludes input generation, host-to-device and
device-to-host copies, reference computation, validation, temporary-storage size
queries, allocation/free, graph capture and instantiation, executable startup,
and evidence serialization. Those excluded phases still have explicit
wall-clock observations where applicable. Temporary storage and all input and
output buffers are allocated before the first call and reused unchanged across
warmups and samples.

Peak temporary storage may not exceed 268,435,456 bytes for any case. The cap
includes provider workspace plus every GPU-side boundary-tie, ordering, and
wrapper scratch allocation. Input, output, graph, and evidence storage are
reported separately and are not hidden inside that workspace figure. Exceeding
the cap is a correctness-ineligible resource failure, not a slow sample.

The capsule records allocation/setup time, workspace bytes, first-call latency,
graph capture and instantiation latency, every warm eager or graph-replay
latency sample, rows/s, and elements/s. No profiled run may be compared as a
clean performance arm.

### Evidence and comparison semantics

A successful arm must preserve, through the existing immutable run and bundle
mechanisms:

- the resolved executable scenario bytes and digest;
- machine, source, build, toolchain, executable, provider, candidate, and input
  identities;
- the conformance-gate result for either rocPRIM provider;
- per-case correctness results for all input families and both modes;
- workspace, setup, first-call, capture, instantiation, and ordered sample data;
- bounded raw process output, structured capsule results, run outcome, and
  portable record identities.

Missing cases, a correctness failure, CPU fallback, provider substitution,
graph failure, non-finite timing, incomplete samples, executable drift, or an
invalid machine observation fails the arm closed. Failed arms remain useful
evidence but do not enter performance comparison.

Comparison will be local and offline between two distinct authenticated finalized arms.
This scenario instantiates the generic contract exactly as follows:

```yaml
comparison:
  policy: paired-latency-log-bootstrap-v1
  protected_regression_bps: 500
  permitted_arm_differences:
    - candidate-id
    - source-candidate
    - build-output
```

Each public matrix row is one `case_id`; its eager and graph-replay coordinates use
distinct coordinate IDs and modes while sharing that case ID, case set, input ID, and
input SHA-256. Each coordinate declares five warmups and 30 ordered samples. D2b1 applies
the generic positive `ln(baseline/candidate)` effect, `math.fsum` mean, exponential
latency ratio, `100 * expm1` improvement percentage, 4,096-replicate positional paired
bootstrap, generic length-framed SHA-256 selection, R-7 95 percent interval, baseline
log-MAD noise, inclusive coordinate verdicts, and evaluation-only aggregate exactly as
specified in [Capsule comparison contract](design.md#capsule-comparison-contract-and-offline-comparison).
The 500-basis-point protected-regression threshold is strict: a coordinate is protected
only when both interval endpoints are strictly negative and its candidate median is more
than 5 percent slower. A protected coordinate changes only a provisional aggregate
improvement to mixed; exactly 5 percent and every other provisional aggregate are
unchanged. Workspace remains report-only.

All three permitted differences are required because this scenario supports both a
provider ID selected inside an otherwise identical arm and a reviewed one-provider
source patch with its authenticated build output. `candidate-id` covers only the exact
candidate fields. `source-candidate` covers only the candidate-derived source reproducer
and nested source-evidence fields enumerated by the generic closed table; source ID,
commit, locator, base commit, branch hint, adapter, and submodule policy/evidence remain
equal. `build-output` covers only the derived recipe/build/attempt identities, existing
targets' content identities, inspections, capture tools, CMake/compile digests, canonical
record digest, and selected executable digest enumerated there. Profile, toolchain,
canonical environment, requested targets, selections, tools, and target topology remain
equal. Arm-local record, manifest, evidence, request/response, build-record, and
executable digests are independently authenticated and normalized only as derived
consequences of those exact semantic differences—not under a broad provenance label.

The generic scenario envelope does not interpret TOPK data. The optional bounded opaque
payload remains authenticated in each raw response and must carry the future TOPK-001
provider, conformance, value/index correctness, setup, capture, first-call, throughput,
and detailed workspace projections. It does not affect generic admission, equivalence,
statistics, or verdicts until TOPK-001 supplies its separate strict interpreter. Training
coordinates remain diagnostic; all public evaluation cases and both modes determine the
future aggregate. These are provisional field comparisons, never official scores or
universal provider rankings.

## Staged implementation sequence

### CAPSULE-001: narrow native capsule seam

The fixed adapter, production `strixlab run capsule` orchestration, host fake, finalized
snapshot loader, D2a comparison contracts, and D2b1 pure offline comparison below are
delivered. Comparison evidence/CLI dispatch remains deferred; step 5 does not authorize a
runnable TOPK manifest.

1. Define one versioned subprocess contract for a trusted native executable
   with bounded `describe`, `correctness`, and `benchmark` operations and
   canonical JSON input/output. Do not add dynamic adapter discovery or a plugin
   registry.
2. Add a capsule-specific manifest/result model and `run capsule` path that
   binds one executable from an authenticated build record, acquires the
   existing machine lock, and reuses `RunSession`, records, bundles, secret
   checks, process bounds, and integrity checks.
3. Add a host-only fake capsule fixture that exercises success, correctness
   failure, malformed output, timeout, executable drift, and incomplete sample
   handling without ROCm or a GPU.
4. D2a defines and authenticates the fixed lower-is-better paired contract. D2b1 loads
   two arms, applies the closed normalization table, and returns canonical in-memory
   statistics and verdicts. A later slice may publish derived evidence and dispatch it.
   Keep model-suite comparison behavior unchanged.
5. Document and test the new manifest, evidence layout, CLI, and no-hardware
   verification path. No top-k manifest lands in this stage.

### TOPK-001: trusted top-k capsule and runnable scenario

1. Add the CPU reference, deterministic generators, exact public matrix, NaN
   rejection, value/index checks, and a trivial host reference test.
2. Add the reviewed `baseline-hip` implementation and the symbolic provider
   registry. Implement the pinned rocPRIM compile conformance fixture and keep
   both rocPRIM providers disabled until it establishes their exact callable
   API, ordering wrapper, and graph behavior at the pinned commit.
3. Add GPU-side ordering, eager execution, graph capture/replay, workspace
   accounting, the fixed timing boundary, and structured evidence projection.
4. Add the executable `rocm10-topk-gfx1151-v1` manifest only when CAPSULE-001
   can validate and execute it. Repository tests validate the manifest and
   host-only protocol without installing ROCm or touching a GPU.
5. On a separately authorized gfx1151/ROCm 10 machine, run the conformance gate,
   all correctness cases, and a no-op baseline/baseline comparison. Record
   hardware verification honestly; do not fabricate it in CI.
6. Only after those gates pass, accept one community experiment PR per provider
   selection or reviewed one-provider patch under the existing scenario and
   experiment workflow.

CAMPAIGN-001, automatic candidate enumeration, a central verifier, hidden
vectors, a leaderboard, and web submission are explicitly out of scope. The
normal proposal Issue, scenario PR, candidate experiment PR, and local
replication workflow in [Community experiment workflow](community-workflow.md)
remains authoritative.

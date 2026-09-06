# Reusable GPU primitive research portfolio

Updated 2026-09-05 after the Qwen3.5 baseline and six-trial profiling follow-up.
The integration target is [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp).
The research unit is a reusable GPU operation with a frozen correctness and
measurement contract. Models supply relevance and integration validation; a
primitive benchmark need not wait for model downloads.

## Execution lanes and actual readiness

| Lane | Purpose | Current boundary |
|---|---|---|
| ROCm 10 primitives | Compare and improve reusable providers, beginning with stable top-k on gfx1151. | Generic capsule runner, authenticated snapshots, comparison publication/CLI, and the TOPK host reference exist. Trusted native capsule/provider implementation remains a prerequisite. ROCm 10 is not installed in the expected prefix. No runnable top-k scenario is claimed. |
| ROCm 7.2.4 integration | Establish whether an optimization helps the pinned model consumer. | Qwen3.5-4B Q4_K_M calibration, PP/TG profiling, and evidence export have executed. A finite-patch model campaign runner is available. |

A ROCm 10 primitive result cannot be compared directly with a ROCm 7.2.4 model
baseline. Integration needs a separately pinned ROCm 10 build and no-op
calibration. The installed rocPRIM is 4.2.0 and lacks both top-k headers; pointing
at the control SDK cannot enable the ROCm 10 providers. Preserve that control.
Prepare the authentic isolated prefix and its verifier before admitting a
bounded GPU conformance run. Successful provider correctness and graph replay
then admit timed arms; they cannot be established by host-only tests.

The [ROCm 10 top-k contract](rocm10-topk-gfx1151.md) remains authoritative for
SDK/rocPRIM pins, public matrix, provider surface, correctness, timing, workspace,
and comparison. This portfolio changes priority and work sequencing, not those
contracts. Coordinator-managed bounded research can begin while capsule campaign
automation remains deferred.

## What the completed studies establish

- [Baseline calibration and profiling](../field-reports/2026-09-05-qwen35-baseline-profile.md):
  two same-build suites and separate PP/TG traces completed. The bundle exporter
  false positive was fixed; all three preserved evidence bundles exported and
  verified. Original failed records remain intact.
- [Shape attribution and six-trial diagnostics](../field-reports/2026-09-05-qwen35-mmvq-shape-attribution.md):
  repeated Q4_K fused FFN gate/up uses reconstructed K=2560, N=9216; the Q6_K
  vocabulary projection uses K=2560, N=248320 and typically takes about 2.3 ms,
  with occasional larger delays. These bindings come from source, GGUF metadata,
  and dispatch order, not observed runtime tensor pointers/strides. The original
  10.646 ms FFN-slot outlier did not recur. Large delays moved between kernels.
  Six trials do not establish a causal profiler-mode effect or justify a patch.
- [Profiler investigation](../field-reports/2026-09-05-rocprofv3-json-string-lifetime.md):
  deferred HIP string formatting contains a source-level lifetime hazard.
  CSV-only capture avoids that JSON branch; it does not fix the upstream cause.
- Historical [RDNA4 MMVQ table](../experiments/smoke-qwen35/rdna35-mmvq-rdna4-nwarps/README.md)
  regressed TG by 8.44%; [Q4_K two-warps](../experiments/smoke-qwen35/rdna35-mmvq-q4k-nwarps2/README.md)
  was inconclusive in two replications. That blind warp sweep remains closed.

The control [source](../configs/sources/strix-llama.yaml) stays pinned to
`ca94157f70a2776e8da6b6849b50b45a083d0478`, with the
[ROCm 7.2.4 build](../configs/builds/hip-rocm-gfx1151.yaml) and
[gfx1151 machine](../configs/machines/strix-halo-128g.yaml).
The [4B suite](../configs/suites/smoke-qwen35.yaml) measures separate PP512,
PP2048, and empty-prompt TG128. It does not measure decode after a populated
2048-token context. Benchmark defaults and server correctness settings are
separate inputs. The pinned 2B model still needs its own verified execution;
larger draft registry entries do not establish artifact availability or coverage.

## Ranked problems

Priority combines reusable value, evidence, and contract readiness; it is not a
speedup forecast. Provider ineligibility or a well-supported negative result is
useful progress.

| Priority / ID | Question | First deliverable | Performance-arm admission |
|---|---|---|---|
| 1 / `stable-topk` | Can pinned ROCm 10 selection providers satisfy deterministic membership/order efficiently across small and long rows? | Host reference/input contract, source/API conformance, then trusted provider implementations. | Authentic isolated toolchain, complete capsule, correctness including graph replay, and no-op calibration. |
| 2 / `quantized-matvec` | Which costs in repeatedly used Q4_K/Q6_K shapes can be removed without changing the operation contract? | Model-free fixture using source/GGUF-derived type/layout metadata pending runtime verification, independent reference checks, explicit timing boundary. | Fixture correctness, reproducible baseline, and mechanism-backed candidate. |
| 3 / `quantized-prefill` | Are recurring PP shapes better served by existing valid kernels or tiles? | Attribute existing PP traces to shape/type/dispatch families; choose at most three alternatives. | Quantized reference and neighboring boundary/tail cases frozen before evaluation. |
| 4 / `reduction-and-fusion` | Do repeated reduction/normalization or state-update boundaries permit useful reuse or fusion? | Map a real consumer to its existing implementation and identify intermediate traffic or launch work. | Consumer evidence, reference semantics, layout/aliasing and numerical policy, useful measured opportunity. |
| 5 / `graph-execution` | Is there avoidable recurring setup/submission around these primitives? | Separate stable replay from intermittent delays and identify one demonstrated redundancy. | Replay/invalidation correctness and trace-supported mechanism. |

### Stable top-k: first ROCm 10 loop

Start with the existing public matrix and symbolic providers `baseline-hip`,
`rocprim-topk`, and `rocprim-segmented-topk`. IDs do not establish upstream API
names or ordering guarantees. The exact pinned source must pass compile and
behavioral conformance before either rocPRIM provider is enabled.

Membership at the K boundary is part of correctness: sorting an arbitrary set
of selected items does not repair unstable tie membership. GPU-side tie
resolution and ordering stay inside timing; CPU fallback/canonicalization cannot
rescue a provider. Preserve signed-zero bits, infinities, NaN rejection,
eager/replay equality, deterministic inputs, and the workspace cap.

Compare the three trusted provider selections under the same complete matrix.
Only after those results, propose at most two one-provider implementation patches
if there is a concrete mechanism. No per-shape dispatch or matrix expansion in
v1. A provider that fails conformance stays unavailable. If the baseline wins,
record that result and explain the cost before proposing another candidate.

Standalone selection research does not require an MoE model. A primitive gain
remains a primitive result until consumer shapes, ordering requirements, and
end-to-end behavior are independently established.

### Quantized matvec and prefill

Use synthetic public inputs with model-derived type/layout metadata, not model
weights copied into fixtures. Preserve block formats and strides and exercise
tails and neighboring shapes. Q4_K_M is a model quantization label, not a tensor
type. Test Q4_K and Q6_K separately.

A large recurring vocabulary projection is worth investigating, but its duration
does not prove inefficiency or a bandwidth bottleneck. Require a proposed
mechanism. Numerical changes need an independently frozen primitive contract;
passing its tolerance does not waive the model campaign's exact cross-arm token
gate. Quality-tradeoff research requires a separate evaluator.

Keep MMVQ warp tables unchanged while investigating PP. Each PP candidate uses
one existing valid dispatch/tile alternative for one observed family on gfx1151,
retaining other types/devices. Stop after three candidates or earlier if
attribution is diffuse or the possible gain is too small.

### Reduction, state, and graph follow-ons

Review [target PRs](https://github.com/halo-box/strix-llama.cpp/pulls) and their
actual source before proposing existing optimizations again. Distinguish merged
from open, HIP from Vulkan, and author benchmark claims from local reproduction.
These cards remain discovery questions until a real consumer and cost are known.

HIP graph support is already enabled by default in the pin. A patch must remove
demonstrated work while preserving invalidation on buffer, layout, sequence,
and state changes. Stop if replay is stable without material avoidable gaps.
Moving stalls are a measurement question, not a reason to rewrite whichever
kernel has the largest isolated interval.

State/attention work needs chunked versus unchunked prefill, reset, sequence
isolation, and continuation checks. Add a versioned prefill-plus-decode workload
before sustained-context claims. Check existing snapshot fusion first.

## Operating the loops

Each loop has an owner and a finite next batch. Agents investigate independently;
the coordinator serializes GPU work and owns admission, review, and baseline
promotion. The aim is accepted evidence, including negative results.

Each research card records:

1. **Operation and consumer:** shapes, types, layouts, semantics, and whether
   relevance is observed, source-derived, or hypothetical.
2. **Objective and protected cases:** a fixed metric/policy, minimum useful effect
   chosen before results, and correctness independent of candidates.
3. **Baseline and allowed surface:** exact source/toolchain/build identities and
   one mechanism per patch; inputs, evaluator, and timing code are protected.
4. **Discriminating test:** what result supports or rejects the hypothesis,
   including conformance prerequisites.
5. **Budget and stop:** maximum candidates/runs and closure for ineligibility,
   small opportunity, repeated inconclusive results, or correctness failure.
6. **Result and next decision:** patch, evidence, identities, verdict, and
   explanation; a new iteration needs an evidence-backed question.

Cycle: inspect evidence, propose a mechanism, review candidate and evaluator,
calibrate, screen a finite batch, confirm with fresh evidence, then promote or
close. Interrupted work remains spent and visible. Changing the baseline,
toolchain, workload, matrix, or acceptance policy starts a new identified
experiment rather than repairing an unfavorable result.

The two evaluation paths remain distinct:

- **Model campaign:** finite source patches, two calibration suites, four AB/BA
  screening suites per candidate, four fresh confirmation suites per survivor.
  Three candidates cost at most 26 suites, excluding separate diagnostics.
  Frozen objectives/protected cases and cross-arm token parity apply. Keep the
  pilot's zero protected-regression margin. `objective_met_provisional` does not
  change the raw judge verdict or establish campaign-wide confidence.
- **Primitive capsule:** the trusted scenario owns provider semantics, timing,
  correctness, and comparison policy. Until capsule campaign automation exists,
  the coordinator schedules explicit bounded arms and records order/budget.
  Do not pass provider selections through the model campaign or claim it already
  automates capsules. Apply the top-k scenario's existing comparison policy
  unchanged, rather than substituting the model campaign's margin.

Keep clean timing separate from profiled diagnostics. Overlapping trace durations
are not additive savings. Under an unchanged-rest assumption, component fraction
`f` and speedup `s` bound total speedup by `1 / (1 - f + f/s)`; use a measured
critical-path fraction when work overlaps. No numerical gain forecast is justified.

## Kickoff, 2026-09-05

Three Codex workers have separate worktrees and bounded ownership. The first
iteration's [upstream research cards](primitive-opportunities.md),
[native host reference](../native/topk/README.md), and capsule-comparison
publication path have been reviewed and integrated locally. The following table
records their initial assignments; follow-on work is admitted separately:

| Worker | First iteration | Completion evidence |
|---|---|---|
| `strix-topk` | Host-only TOPK reference and deterministic input foundation; implementation plan approved. | Independent fixed vectors and semantic tests, reviewed commit; no provider/GPU availability claim. |
| `strix-capsule` | Authenticated comparison publication and CLI around the pure comparator; implementation plan approved. | Host-only success/failure/authentication tests; unchanged statistics and model behavior. |
| `strix-primitives` | Upstream primitive mapping and read-only ROCm 10 readiness investigation. | Linked consumer evidence, ranked next cards, isolated toolchain preparation recommendation. |

The coordinator reviews each result and admits the next bounded iteration.
These first iterations establish the executable boundary; they are not measured
provider experiments. The trusted capsule and authentic toolchain admit bounded
GPU conformance work. After correctness and graph checks pass, timed provider
arms and no-op calibration can begin. Preparation need not wait for new models.

Use the [community workflow](community-workflow.md) for scenarios, experiments,
and replication. Primitive improvements and model integration results stay
separately attributable. Neither an agent's proposed winner nor same-machine
confirmation automatically becomes `best-known`.

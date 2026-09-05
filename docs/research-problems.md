# Profile-guided llama.cpp research problems

This portfolio ranks five bounded questions for Strix Halo, with the smallest
post-merge pilot first. These are research hypotheses, not measured gains or
registered experiments. No GPU experiment, model download, or ROCm provisioning
is part of this document's preparation.

## Evidence and executable boundary

Use [strix-llama](../configs/sources/strix-llama.yaml) at
`ca94157f70a2776e8da6b6849b50b45a083d0478`, the installed ROCm 7.2.4 control
[build profile](../configs/builds/hip-rocm-gfx1151.yaml), and
[gfx1151 machine profile](../configs/machines/strix-halo-128g.yaml).
Upstream paths below refer to that pinned source, whose HIP backend shares
implementation under `ggml/src/ggml-cuda/`.

The v1 campaign accepts a finite reviewed patch list against one frozen source,
build profile, machine, model, and native smoke suite. It does not search runtime
arguments, CMake options, model choices, or capsule providers. Candidate patches
must respect its source allowlist and protected evaluator/build/test files.
An allowable patch is not automatically a scientifically adequate experiment.

| Input | Actual coverage and readiness |
|---|---|
| [4B smoke suite](../configs/suites/smoke-qwen35.yaml) | Executable contract today, conditional on authorized hardware, build and verified local GGUF. Qwen3.5-4B Q4_K_M; separate `pp512`, `pp2048`, and empty-prompt `tg128` measurements. Two warmups per case, five windows of three measurements. Greedy gate: one prompt, seed 1234, 64 output tokens, context capacity 4096, 999 GPU layers. |
| [2B model](../configs/models/qwen35-2b-smoke.yaml) | Registered Qwen3.5-2B Q4_K_M with artifact size/hash/revisions; execution remains unverified. Needs a separately frozen suite and local verification before use. |
| [4B model](../configs/models/qwen35-4b-smoke.yaml) | Registered artifact, not proof of local availability. Q4_K_M does not mean every tensor is Q4_K; collect the actual tensor-type/shape inventory. Quantization calibration and tensor-policy provenance remain unknown. |
| [Ornith 35B A3B](../configs/models/ornith15-35b-a3b.yaml), [Qwen3.8 27B](../configs/models/qwen38-27b.yaml), [Qwen3.8 Flash Next](../configs/models/qwen38-flash-next.yaml) | Draft entries without artifact pins or quantization recipes. No runnable model claims, substituted local weights, or invented hashes. |

The benchmark adapter currently emits only model, prompt count, generation
count, repetitions and JSONL output arguments. The greedy server's 4096 context
and GPU-layer settings do not establish benchmark settings. Capture resolved
benchmark defaults and actual offload before attributing a result. `tg128` does
not measure decoding after a 2048-token prefill; context capacity is not occupied
context. There is no current long-context, concurrent-serving, vision, MTP, or
multi-quantization performance claim.

Two existing records constrain the search:

- [RDNA4 MMVQ table](../experiments/smoke-qwen35/rdna35-mmvq-rdna4-nwarps/README.md):
  correctness passed; `tg128` regressed 8.44% (95% interval -9.91% to -6.98%);
  PP cases were inconclusive.
- [Q4_K two-warps](../experiments/smoke-qwen35/rdna35-mmvq-q4k-nwarps2/README.md):
  two same-machine replications were inconclusive, with TG point estimates
  -0.69% and +0.43%. The record explicitly stops this branch before `nwarps=4`.

These are historical calibration/regression fixtures, not invitations to rerun
or expand the warp sweep. They do not establish the bottleneck: correctness and
aggregate timings cannot distinguish occupancy, bandwidth, launch costs, or
dispatch effects.

## Ranked portfolio

Rank combines practical scope, cost and readiness, conditional on the required
profile evidence. Confidence describes the hypothesis before new measurements;
none has high confidence of a speedup. Patch questions lead because they fit v1.

| Rank / ID | Question | Kind | Conditional payoff / confidence | Start status |
|---|---|---|---|---|
| 1 / `decode-graph-reuse` | Can repeated graph setup or launch gaps be removed without stale bindings? | Patch | Modest broad benefit if submission dominates; medium-low confidence | Smoke baseline/profile ready; patch needs trace evidence |
| 2 / `quant-shape-dispatch` | Are specific PP matrix shapes sent to an inefficient existing kernel? | Patch | Potentially useful PP benefit; medium-low confidence | Smoke baseline/profile ready; quant correctness expansion required |
| 3 / `runtime-batching` | Which batch and CPU submission settings avoid wasted host work? | Configuration | Low implementation cost, potentially useful operational benefit; low confidence of gain | Outside v1 candidate surface |
| 4 / `hybrid-state-traffic` | Does hybrid attention/state movement dominate sustained-context inference? | Patch | Potentially substantial at long context; low confidence | Missing workload and state-correctness contract |
| 5 / `routing-selection` | Is selection a material end-to-end bottleneck for a real routing workload? | Patch; later provider research | Unknown payoff; very low confidence | Model/workload/attribution blocked; ROCm 10 lane separately blocked |

### 1. Decode graph reuse

**Hypothesis and locus.** In `ggml/src/ggml-cuda/ggml-cuda.cu` and `common.cuh`,
graph invalidation, rebuilding, or synchronization might recur unnecessarily
when decode shapes remain stable. The pin already defaults `GGML_HIP_GRAPHS`
to ON; "enable graphs" is not a novel patch. Verify the built feature and actual
replay behavior first. Shared code recognizes `GGML_CUDA_DISABLE_GRAPHS`, but
changing that environment is a diagnostic configuration experiment, not a v1
patch candidate.

**Evidence needed.** Separately trace host HIP calls and device execution for
`tg128` and both PP cases. Count captures, instantiations, updates, launches,
synchronizations and allocations; map gaps to host stacks and graph invalidation
reasons. Distinguish first-use setup from steady replay. A short GPU kernel or
smoke speedup alone cannot establish launch overhead causally.

**Bounded search.** At most two reviewed patches, each removing one demonstrated
redundant update or allowing reuse under one explicit shape/layout/address
invariant. Preserve fallback on invariant changes. Do not change graph mode,
math, batching, or global synchronization policy in the same candidate.

**Correctness and stop.** Require cross-arm tokens plus graph replay tests with
changed input contents, buffers, sequence lengths and reset state; exercise
capture invalidation rather than only steady shapes. Stop if the trace already
shows stable replay without material host gaps, if safe invalidation cannot be
specified, or after two candidates fail fresh confirmation. Smoke can measure
PP/TG impact now; broader graph correctness must be established before acceptance.

### 2. Quantized PP shape dispatch, without another warp sweep

**Hypothesis and locus.** Inspect `ggml/src/ggml-cuda/ggml-cuda.cu`, `mmq.cu`,
`mmq.cuh`, and `mmq-config-rdna3-5.cuh`: a measured Q4_K_M tensor shape may use
an inefficient MMQ tile or dispatch boundary. Prior MMVQ decode failures do not
answer this PP question. Keep `mmvq.cu`/`mmvq.cuh` launch tables unchanged.

**Evidence needed.** Attribute `pp512` and `pp2048` time to actual matrix shapes,
tensor types, selected MMQ versus library paths, padding/tail work, launches,
memory traffic and available occupancy counters. Measure kernel fractions and
repeatability before choosing a threshold. Do not assume a bandwidth bottleneck
from the GPU model or the artifact's quantization label.

**Bounded search.** At most three patches using existing valid dispatch/tile
choices for one observed shape family on gfx1151. One boundary or tile choice
per candidate; retain other devices/types and MMVQ behavior. Do not combine
vec-dot rewrites, new quantization, or blanket library-force options.

**Correctness and stop.** Add independently reviewed CPU-reference matmul cases
for the observed quant types, boundary/tail shapes, and neighboring dispatch
shapes before candidate evaluation; freeze tolerances before results. Test all
affected paths, not just the fastest tensor. Protect `tg128` and both PP cases.
Stop if attribution is diffuse, an existing alternative is already selected,
or the three-candidate budget produces no reproducible end-to-end benefit.

### 3. Runtime batching and CPU submission tuning

**Hypothesis.** Logical batch, physical microbatch and CPU thread counts may
trade host overhead against GPU utilization on shared-memory Strix Halo. This
has the lowest implementation cost once typed configuration support exists;
it does not have the highest demonstrated speedup probability.

**Workload/evidence.** Start with 4B Q4_K_M PP512/PP2048 and TG128, recording
host scheduling, GPU idle gaps, page faults, memory use, offload and resolved
defaults. Later add 2B under its own suite. Keep context, cache type, graph mode,
offload and model fixed. Verify available options against the pinned binaries.

**Bounded search.** Propose batch values 256/512, microbatch values 128/256
(not exceeding batch), and CPU threads 4/8: eight settings maximum, screened
one axis at a time. These are proposed values, not supported StrixLab flags.
The current adapter has no authenticated tuning surface. Add typed argv and
comparison semantics first; separate frozen scenarios are not automatically
eligible for direct comparison. Never emulate this search by patching defaults
inside a fixed-scenario campaign.

**Correctness and stop.** Require cross-arm tokens and matched effective
settings between correctness and timing paths. No quantized-cache accuracy
tradeoff is included. Stop after the bounded grid if effects remain below noise
or PP benefit trades away TG. No clocks, power, driver or system tuning.

### 4. Hybrid recurrent-state and attention traffic

**Hypothesis and locus.** The 2B/4B registry describes gated DeltaNet plus full
attention. Inspect `ggml/src/ggml-cuda/gated_delta_net.cu`, its dispatch in
`ggml-cuda.cu`, and `fattn.cu`. The pin already contains fused recurrent-state
snapshot copies: establish whether that path is active before proposing fusion.

**Missing workload.** Propose separately versioned occupied contexts 2048 and
8192, each followed by 128 generated tokens, batch one; retain the short smoke
cases as regressions. Verify model/runtime context support and memory bounds
before adopting 8192. First use pinned 4B Q4_K_M, then separately pinned 2B.
The current single-metric benchmark adapter cannot express combined prefill and
generation; increasing its PP count alone does not fill this gap.

**Evidence/search.** Attribute time and bytes to full attention, recurrent
updates, state snapshots and copies as context grows. Select only the dominant
path. At most two patches: one narrowly guarded copy elimination or one existing
tile/layout choice, never both mechanisms together. Preserve cache precision.

**Correctness and stop.** Require recurrent-state/reference checks, chunked
versus unchunked prefill, reset/sequence isolation, continuation tokens and
long-context cross-arm parity. Freeze fixtures outside candidate patches. Stop
if the existing fusion already removes the traffic, if another path dominates,
or if correctness requires relaxing numerical/semantic rules after seeing a gain.

### 5. Routing selection, with a top-k relevance gate

**Hypothesis.** A future verified model may spend material time in routing
selection, including ordering and synchronization. The current dense 2B/4B
smoke inputs are not a proxy for MoE routing or sparse attention. Draft model
names establish neither architecture support nor measured routing shapes.

**Prerequisites/evidence.** Obtain reviewed provenance and quantization pins,
local verification, a supported source/build and a versioned workload first.
Trace routing/selection call counts, rows, columns, K, ordering semantics,
surrounding work and total token latency. Establish whether time is actually
in selection rather than expert matmul or data movement. A source bump, if
needed, is a new baseline, not a candidate exception.

**Bounded search.** Only after attribution, evaluate at most two narrow
existing-HIP selection patches under the model scenario. Separately,
[planned ROCm 10 top-k](rocm10-topk-gfx1151.md) defines a public microbenchmark
matrix (rows 1–128, columns 128–262144, K 1–256), not measured model shapes.
Its trusted capsule, payload/provider semantics, conformance, and ROCm 10
authenticity/isolation gates remain prerequisites. That scenario does not need
model weights, but cannot establish model relevance by itself. Do not provision
ROCm 10 for this portfolio or treat its providers as v1 campaign candidates.

**Correctness and stop.** Respect each operation's actual semantic contract.
For the planned capsule this includes exact stable value/index membership,
boundary ties, signed zeros, infinities, NaN rejection and eager/replay equality;
keep GPU ordering overhead inside timing. Model acceptance additionally needs
cross-arm routing/token checks and end-to-end improvement. Stop before kernel
research if measured selection fraction cannot support a useful total gain.

## Common evidence, estimates and acceptance

No numerical gain forecast is justified yet. Use measured unprofiled time
fraction `f` to bound a proposed optimization: if its component speedup is
hypothesized to be `s`, total speedup is at most `1 / (1 - f + f/s)` under the
unchanged-rest assumption. For illustration only, `f=0.10`, `s=2` yields about
5.3% throughput improvement; this is not an observed Strix Halo result. For
overlapping host/device work, derive a critical-path fraction rather than summing
overlapping profiler durations. Set a minimum useful effect before candidates;
stop if even eliminating the component cannot reach it.

Keep diagnostic profiled runs separate from clean timing evidence. Record tool
identity, source/build/model digests, workload settings, trace commands and
redacted summaries. Missing profiler/counter support blocks causal attribution;
do not substitute invented counters. Preserve failed and inconclusive runs.

The smoke backend gate covers selected **f32** MUL_MAT, MUL_MAT_ID, TOP_K,
ARGSORT and RMS_NORM cases. It does not cover quantized kernel math or recurrent
state. Greedy repeatability within one arm is also insufficient: correctness-
preserving acceptance requires authenticated baseline/candidate token parity.
The v1 exact-token gate targets launch/layout-preserving candidates; operator
reference tolerances never waive campaign cross-arm token identity. A candidate
that changes numerical behavior and needs quality-tolerance acceptance requires
a separately reviewed, frozen evaluator outside this v1 campaign.
Broader operator fixtures and token prompts must be reviewed and frozen before
the campaign, never modified by candidate patches. A changed suite gets a new
scenario identity. If the merged runner does not enforce a required check,
record that acceptance blocker explicitly.

Require fresh confirmation after screening, under serial exclusive GPU use and
matched source/model/machine/toolchain/settings, with both execution orders.
The native judge requires improvement in **all** cases for overall improvement.
A target gain with inconclusive protected cases remains `mixed` in the raw judge.
The campaign may separately declare `objective_met_provisional` using its frozen
plan: `objective_cases` is a nonempty unique subset of performance case IDs
(default: all); every objective must have an existing `improvement` verdict.
Every remaining case is automatically protected and must satisfy
`percent_ci_low >= -protected_regression_margin_percent`. The explicit margin
is at least zero and less than 100 percent, default zero. For this initial pilot,
keep the zero default; a nonzero tolerance needs a practical justification
before screening. Both screening comparisons and both fresh confirmations must
pass, with cross-arm parity throughout. Preserve the raw overall verdict even
when it is mixed. The campaign label is neither judge improvement nor best-known.
Per-case intervals are not simultaneous or campaign-level confidence, and this
rule is not a formal noninferiority proof.

Do not pool PP/TG, change objectives after screening, or relax margins to rescue
a candidate. Correctness failure, a failed protected-case bound, exhausted budget
or absent mechanism evidence stops acceptance. Maintainership alone assigns `best-known`;
same-machine confirmation is not independent community replication.

## Smallest pilot after merge

1. **Calibrate.** On later explicitly authorized hardware, verify inputs and run
   two distinct same-build smoke suites and one comparison, matching the built-in
   calibration cost. Inspect the no-op verdict/noise and run/bundle integrity.
   Additional order diagnostics require a separately budgeted calibration step.
   Use existing MMVQ summaries as negative interpretation fixtures without scheduling their patches again. Stop for unstable conditions
   or correctness/evidence failure.
2. **Profile one baseline.** Collect separate PP and TG traces with resolved
   settings. Choose problem 1 or 2 only if attribution and the payoff bound
   justify it. No evidence means no candidate campaign.
3. **Freeze gates and candidates.** Add any missing correctness coverage through
   ordinary reviewed harness/scenario work, then freeze it. Review a finite list
   of at most two graph patches or three dispatch patches, each against the same
   original source base; do not accumulate unconfirmed patches. Keep tests and
   evaluators outside the candidate patch surface.
4. **Screen and confirm.** Use the merged campaign's bounded screening and fresh
   confirmation stages, including AB/BA order checks. Confirm against the
   original baseline with new evidence; preserve unsuccessful attempts and stop
   when the finite list is exhausted. Preserve mixed judge verdicts; report a
   separately satisfied frozen campaign objective only as provisional.

Thus one pinned model, one existing smoke workload, and one selected patch
mechanism form the initial portfolio. Problems 3–5 remain useful design work
with explicit input gaps; none is a reason to download models or activate ROCm
10 now. Follow the [community workflow](community-workflow.md) when converting
these questions into versioned scenarios and experiment records.

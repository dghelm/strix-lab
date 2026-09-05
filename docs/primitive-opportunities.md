# Primitive opportunities and ROCm 10 readiness

Evidence checked 2026-09-05. This is a read-only upstream/host investigation
and a ranked next-work proposal. No GPU workload, compiler conformance run,
SDK provisioning, model download, or benchmark reproduction was performed.
The coordinator owns hardware admission and integration. Standalone primitive
research does not require an MoE model; a primitive gain is not a model gain.

**Key integration mismatch:** PR #20's Vulkan selector is repeatable but emits
indices above the threshold in input order, followed by boundary ties. It does
not produce descending-value order, and its radix key distinguishes signed zeros.
It therefore does not directly satisfy StrixLab TOPK v1's numeric-value/index
contract. See card 1 before reusing its algorithm or interpreting its validation.

## Upstream state and implemented work

Authenticated GitHub PR metadata and changed-source patches were inspected.
PR descriptions contain historical measurements and sometimes stale branch
instructions; API state and head identity below take precedence. “Implemented”
means present in the inspected PR source, not verified in a local executable.

| PR | State at inspection | API head | Backend and reusable mechanisms |
|---|---|---|---|
| [18](https://github.com/halo-box/strix-llama.cpp/pull/18) | Merged 2026-09-05 11:39:37 UTC | `4021b991c6245044ecda10f1e9759e28308610e0` | HIP through shared `ggml-cuda`: RDNA3.5 quantized matmul/vector kernels, compact expert maps, fused activation quantization, small-top-k routing, dual normalization, GDN and hyper-connection fusion, D=256 attention. Shared model graph/PLE and checkpoint changes also occur. |
| [20](https://github.com/halo-box/strix-llama.cpp/pull/20) | Merged 2026-09-04 03:01:03 UTC | `3407302658bbc9a2bab159e912c6798cdcbb7b4a` | Vulkan radix selection determinism; shared KV free-cell zeroing and model GDN normalization correction. Its body still describes a stack on #19. |
| [11](https://github.com/halo-box/strix-llama.cpp/pull/11) | Closed, **not merged** | `3dae714dd2bff24db2b30806394dac1bd2e7b303` | HIP and Vulkan predecessor work: 512-expert/top-10 routing, MMQ/MMVQ tuning, reduction/state and Vulkan dispatch/concat. Overlapping code appears in #18; do not add the two PRs' benchmark gains. |
| [17](https://github.com/halo-box/strix-llama.cpp/pull/17) | Open | `5450978e5194f03ab769304388864bdbccc21b37` | Vulkan attention dequant/contiguity, expert GEMM tiles, tiled state concat, sparse indexer/gather/attention and hyper-connections. An alternative implementation source, not a merged HIP provider. |

The StrixLab control remains pinned to `ca94157f70a2776e8da6b6849b50b45a083d0478`
in [its source profile](../configs/sources/strix-llama.yaml). A merged upstream PR
does not update that pin. Extracting or integrating newer work needs a separately
identified source and baseline calibration. PR #18's body reports validation on
“ROCm 7.14” and a different final head; neither establishes local ROCm 10 behavior.
All upstream throughput/test counts remain author reports, not our reproduction.

## Practical ROCm 10 blocker

Read-only host observations:

| Observation | Consequence |
|---|---|
| `/opt/rocm/.info/version` contains `7.2.4` | This is the control SDK identity observation. |
| `/opt/rocm-10` is absent | The checked-in ROCm 10 profile's absolute compiler/prefix paths cannot resolve. |
| `/opt/rocm/include/rocprim/rocprim_version.hpp` declares 4.2.0; neither top-k public header occurs in that include tree | The installed control headers cannot supply the proposed 4.6 top-k APIs. Changing an include path to the control does not unblock ROCm 10. |
| `/etc/os-release`: Omarchy 4.0.2; kernel `7.1.8-arch1-3` | AMD's Ryzen table lists Ubuntu 26.04 / 24.04.4 and associated kernels, not this host. Future local results are unsupported-host field observations. |

The OS comparison was checked against AMD's release-specific
[compatibility matrix](https://rocm.docs.amd.com/en/docs-10.0.0/compatibility/compatibility-matrix.html).
No runtime GPU/driver compatibility was inferred from these filesystem facts.

The [bring-up contract](rocm10-bringup.md) additionally requires an accepted
artifact-authenticity basis and a reviewed, versioned archive/prefix verifier.
Its September 1 search found no vendor digest/signature for the chosen tarball.
This investigation reconfirmed that AMD's [installation page](https://rocm.docs.amd.com/en/docs-10.0.0/install/rocm.html)
offers a custom-directory gfx1151 tarball; a text search there found no SHA-256
entry. It did **not** repeat the exhaustive sidecar search, so it makes no claim
that no authenticity anchor exists anywhere today. Missing compiler files,
missing accepted provenance, and missing verification are distinct blockers.

Proposed isolated route, **not executed**:

1. Resolve the exact release artifact's vendor anchor, or obtain the existing
   contract's explicit risk acceptance for unauthenticated bytes. Retain URL,
   retrieval timestamp and observed digest with their actual meaning.
2. Implement/review the bring-up document's archive admission, extraction and
   complete prefix inventory verifier. Stage in a new operator-owned directory;
   verify the accepted manifest against every staged entry and preserve the
   installed ROCm 7.2.4 baseline inventory.
3. Prefer a separately reviewed user-owned ROCm 10 prefix for this no-global-install
   project. This requires a new/amended profile with absolute compiler/library
   paths and an explicit revision of the existing `/opt/rocm-10` location contract;
   do not silently repoint the checked-in profile. The older `/opt/rocm-10` copy
   route remains a separate proposal requiring authorization outside this scope.
4. Verify staged/installed equality, compiler and headers, link/runtime closure,
   and absence of control-prefix library substitution. Keep global PATH,
   alternatives, drivers, firmware and `/opt/rocm` unchanged. A container can
   isolate user space, but does not turn an Arch host kernel into supported Ubuntu.
5. Coordinator first admits a bounded conformance run on the exclusive queue;
   graph behavior necessarily needs hardware. Only after correctness does it
   admit baseline/baseline calibration and timed provider comparisons.

## Pinned rocPRIM source findings

Context7 was queried first for rocPRIM top-k. It returned sorting documentation,
not sufficient top-k API evidence. Authenticated GitHub git-tree/blob reads then
retrieved the exact public headers from the already documented
`8d1ae90eff7d022f26019ec55b2ec6a7674b3112` pin. The recursive repository tree was
truncated, but contained both requested blob entries; the complete header blobs
were read. No absence conclusion was drawn from that truncated listing.

- [Non-segmented header](https://github.com/ROCm/rocm-libraries/blob/8d1ae90eff7d022f26019ec55b2ec6a7674b3112/projects/rocprim/rocprim/include/rocprim/device/device_topk.hpp):
  source declarations include `topk` and `topk_pairs`, with null-workspace size
  query, stream, and `Descending`, `Ordered`, `Deterministic`, `Stable` template
  switches. In `TopKImpl::algo_impl`, [lines 97–104](https://github.com/ROCm/rocm-libraries/blob/8d1ae90eff7d022f26019ec55b2ec6a7674b3112/projects/rocprim/rocprim/include/rocprim/device/device_topk.hpp#L97-L104)
  require `UseRadix && radix_checker::use_radix`, no custom decomposer,
  `Ordered == false` and `Deterministic == false`. These assertions precede
  `if constexpr(Stable)`, so selecting that branch does not bypass them.
  Separately, [lines 106–158](https://github.com/ROCm/rocm-libraries/blob/8d1ae90eff7d022f26019ec55b2ec6a7674b3112/projects/rocprim/rocprim/include/rocprim/device/device_topk.hpp#L106-L158)
  contain a Stable branch that invokes radix sort using the input buffers and
  copies K results; the other branch calls AIR. This conflicts with the public
  “stable unavailable” comment at lines 317–318, not with the assertions above.
  Input mutation, temporary-storage queries and replay must be exercised by the
  exact instantiation; source inspection is no compile/runtime conclusion.
  Graph support is documented, not locally tested.
- [Segmented header](https://github.com/ROCm/rocm-libraries/blob/8d1ae90eff7d022f26019ec55b2ec6a7674b3112/projects/rocprim/rocprim/include/rocprim/device/device_segmented_topk.hpp):
  declarations include `segmented_topk` / `segmented_topk_pairs`, a segment count
  and begin/end offset iterators. AIR dispatch requires unordered, unstable,
  nondeterministic settings; the alternative “naive” branch is described as
  ordered/stable but not deterministic. Unsupported combinations reach a static
  assertion. These comments do not establish boundary-tie or signed-zero behavior.

These are **source spellings, not compiled provider availability**. Keep the
[TOPK contract](rocm10-topk-gfx1151.md) conformance gate: compile the exact intended
instantiation, record enabled headers, and test eager/replay membership/order.
Do not equate `Stable=true` with StrixLab's descending-value/index tie contract.
A numeric-key plus index payload may still need GPU boundary-tie resolution and
final ordering. Sorting an arbitrary selected K cannot recover omitted low-index
ties. Both repair costs and scratch belong to the measured provider.

## Ranked research cards

Ranking favors a useful bounded next deliverable, not a speedup prediction.
Cards 2–5 need their own frozen fixture/evaluator before performance work; they
do not alter the existing TOPK v1 matrix or provider surface.

### 1. Deterministic row selection and segmented selection

**Contract.** Use TOPK v1 unchanged: row-major f32, exact K original value/index
pairs, descending numeric value then ascending index; signed-zero bits preserved,
infinities included, NaNs rejected. Include GPU membership repair/order, fixed
batching, eager and graph replay, and the 256 MiB workspace cap.

**Consumer evidence.** [#20's shader](https://github.com/halo-box/strix-llama.cpp/blob/3407302658bbc9a2bab159e912c6798cdcbb7b4a/ggml/src/ggml-vulkan/vulkan-shaders/topk_radix_select.comp)
uses histogram thresholding and subgroup ballot scans to stabilize sparse QSA
membership/order. It emits above-threshold indices in input order, then threshold
ties; it does not sort by value. Its float-to-uint mapping distinguishes signed
zeros. [#18's fused MoE path](https://github.com/halo-box/strix-llama.cpp/blob/4021b991c6245044ecda10f1e9759e28308610e0/ggml/src/ggml-cuda/topk-moe.cu)
already reduces output-weight storage for 512 experts with K at most a warp.
Its softmax/bias/weight semantics exceed plain selection.

**Baseline/alternatives.** Reviewed extracted HIP baseline versus the two symbolic
rocPRIM providers after conformance. Full stable radix sort/hipCUB and Vulkan
ballot selection are later research references, not additional v1 providers.
#18's `top-k.cu` hipCUB include does not enable `CUB_TOP_K_AVAILABLE` on HIP.

**Wrapper work and unknowns.** One possible trusted GPU wrapper canonicalizes
signed zeros in selection keys while retaining original value bits and indices,
obtains a valid numeric cutoff from the provider, then rescans the complete row:
retain every value above the cutoff and fill remaining slots with the lowest
original indices equal to it. Only then sort the exact membership by descending
value/ascending index and gather original bits. This is a proposed algorithm,
not implemented or validated here. An alternative is a proven total-order key
encoding, but the pinned nonsegmented path rejects custom decomposers; primitive
key support, packing width and preservation of the v1 provider contract need
review. A Stable branch may need input copies to preserve immutable originals
across replay. All transforms, copies, scans, membership repair, ordering and
scratch must be accounted inside the measured operation/workspace boundary.
Unknowns include enabled signatures, native cutoff/tie/zero behavior, graph safety,
input mutation, total scratch and whether a repaired provider beats the baseline.
Final sorting alone never repairs missing boundary members.

**Prerequisites/test.** Host foundation and trusted native capsule first; isolated
SDK next. First discriminating correctness probe: K=1 duplicate maxima with
opposing signed zeros, plus K=256 in a 257-column row and two different rows to
catch segment leakage. These probes supplement, never replace, the full fixed
matrix, input families and graph gate before timing.

**Stop.** One conformance attempt per intended provider configuration; record
unsupported APIs/ordering as ineligibility and repair only a demonstrated wrapper
defect. Compare admitted providers over the complete v1 matrix; at most two
subsequent one-provider patches with a concrete cost mechanism. No model gate
for this work. Model integration later requires matching consumer semantics and
shapes, routing/attention correctness and end-to-end measurement.

### 2. Quantized dot products, activation quantization and expert compaction

**Contract.** Freeze Q4_K/Q5_K/Q6_K block formats, input quantization and accumulation
policy, strides/tails, expert IDs and destination mapping. Treat dense MMVQ,
dense prefill and routed MMQ as separate coordinates. Test dequantized independent
reference math; freeze tolerances before candidates and preserve the model
campaign's separate exact-token gate.

**Consumer evidence.** [#18 MMQ dispatch](https://github.com/halo-box/strix-llama.cpp/blob/4021b991c6245044ecda10f1e9759e28308610e0/ggml/src/ggml-cuda/mmq.cu)
contains guarded 512-expert Q5_K geometries 2560-by-768 and their transpose,
and Q6_K 768-by-2560, with a 32-token dispatch condition.
[Expert-map construction](https://github.com/halo-box/strix-llama.cpp/blob/4021b991c6245044ecda10f1e9759e28308610e0/ggml/src/ggml-cuda/mmid.cu)
already histograms/scatters 512-expert/top-10 slots. Its atomic within-expert
ordering is explicitly irrelevant because inverse destination maps restore each
row and no cross-row reduction occurs. This is a different determinism contract
from selection feeding attention sums.

**Baseline/alternatives.** Same-source generic kernels versus one existing guarded
tile, or fused versus separate activation quantization. A rocBLAS/dequantize+GEMM
reference is a separate end-to-end provider boundary that must count conversion
and scratch; it is not a drop-in packed GGML quantized provider. #17's cooperative
matrix variants are Vulkan-only alternatives.

**Prerequisites/test.** Confirm one recurring consumer family and dispatch from
source/trace evidence, then synthetic packed inputs. Start with the 768/2560
routed pair at 31/32/33 tokens, empty/heavy expert buckets, both mapping directions
and a legal partial output tile. Include all-zero activations and independent
destination-map checks. Measure compaction and compute separately plus together.

**Stop.** At most three existing valid dispatch alternatives for one family;
close on correctness failure, diffuse attribution or insufficient measured
opportunity. Do not restart the previously rejected blind MMVQ warp sweep.
Standalone dot-product gains require separate consumer frequency and total-token
latency evidence before model claims.

### 3. Norm reductions with explicit epsilon semantics

**Contract.** Specify the formula, reduction precision, output layout and aliasing
before fusion. Old L2 `x / sqrt(max(sum(x*x), eps*eps))` and FLA-style
`x / sqrt(sum(x*x) + eps)` are different operations near zero.

**Consumer evidence.** [#18 dual normalization](https://github.com/halo-box/strix-llama.cpp/blob/4021b991c6245044ecda10f1e9759e28308610e0/ggml/src/ggml-cuda/norm.cu)
already combines two 128-wide L2 norms into one launch, with separate q/k epsilon
values. Its graph matcher requires L2/view/L2, compatible shared-source views,
contiguous rows/outputs, single-use intermediate and safe memory ranges.
[#20's model helper](https://github.com/halo-box/strix-llama.cpp/blob/3407302658bbc9a2bab159e912c6798cdcbb7b4a/src/models/models.h)
instead builds RMS_NORM with eps/n then scales by 1/sqrt(n). Thus old L2 fusion
cannot simply be assumed active in corrected GDN graphs.

**Baseline/alternatives.** Existing unfused corrected graph versus fusion that
preserves its formula; custom warp/block reductions versus a pinned rocPRIM
block reduction are potential internal implementations after API verification.
No provider performance or availability claim is made for the latter.

**Prerequisites/test.** Trace whether the corrected graph leaves a material
launch/intermediate boundary. First fixture: two 128-wide q/k rows, zero and
near-epsilon magnitudes, unequal eps values, strided source views, plus an external
consumer and overlapping output that must prevent unsafe fusion.

**Stop.** At most two formula-preserving candidates; stop if the actual graph is
already fused or no material cost remains. An old-formula speedup is not evidence
for corrected model semantics. Validate model logits/tokens separately.

### 4. State layout transforms and sparse-attention gather

**Contract.** Begin with exact row relocation/transpose, preserved per-query masks
and sequence ownership. State updates additionally need a frozen recurrence,
accumulation policy, snapshot lifetime and chunk/reset behavior.

**Consumer evidence.** [#17 concat shader](https://github.com/halo-box/strix-llama.cpp/blob/5450978e5194f03ab769304388864bdbccc21b37/ggml/src/ggml-vulkan/vulkan-shaders/concat_transpose.comp)
already uses a padded 32-by-33 shared tile for coalesced transposed state concat.
Its [sparse gather](https://github.com/halo-box/strix-llama.cpp/blob/5450978e5194f03ab769304388864bdbccc21b37/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_gather.comp)
copies raw row words, retains per-token masks, and zeros invalid/padded rows;
multi-token blocks remain separate rather than deduplicated. #18 also contains
HIP GDN decode/prefill fusion; generic snapshot fusion already exists in the
StrixLab source pin.

**Baseline/alternatives.** Generic strided concat versus existing tiled transform;
indexed attention versus gather+compact attention, counting gather, mask and
scratch costs. Vulkan implementation is a design reference for HIP porting,
not evidence that a HIP provider exists.

**Prerequisites/test.** Choose only one boundary with measured avoidable traffic.
First layout fixture: 31/32/33 tile edges, two sequences, duplicate/invalid gather
indices and padding. For any attention/state integration, add repeated requests,
reset/reuse, chunked versus unchunked prefill and continuation checks. #20's
free-cell zeroing explicitly does not solve masked cells owned by another live
sequence in a unified cache; retain that case as a separate correctness gate.

**Stop.** At most two implementations; stop if existing fusion removes the copy,
scratch/materialization negates savings, or isolation fails. Pure copy speedup
does not establish attention throughput or sustained-context model relevance.

### 5. Graph closure, input preparation and replay

**Contract.** Same outputs/state after replay with new contents; invalidate or
rebind on changed addresses/layouts/sequence state. Preserve intermediate external
consumers and alias exclusions; measure preparation separately from replay.

**Consumer evidence.** [#18 Qwen4exp graph/input code](https://github.com/halo-box/strix-llama.cpp/blob/4021b991c6245044ecda10f1e9759e28308610e0/src/models/qwen4exp.cpp)
already gathers host-resident PLE tables during input preparation, preserves
buffers needed by combine+norm fusion, and bypasses QSA selection when the whole
KV extent fits the selection width. Its backend has graph-closure/memory-range
guards. The StrixLab pin already has HIP graph support and snapshot fusion;
“enable graphs” is not a new optimization.

**Baseline/alternatives.** Same pinned operation in eager/replay for diagnostics;
then one demonstrated redundant input-preparation/submission boundary versus
its guarded removal. Keep graph-mode changes outside fixed model patch comparisons.

**Prerequisites/test.** Obtain capture/update/launch counts and a recurring host
gap, not an isolated slow kernel. First replay fixture changes input contents,
then buffer identity and layout, inserts an external intermediate consumer and
resets sequence state. PLE work additionally verifies gather output and mapping
before timing preparation+execution together.

**Stop.** One mechanism, at most two patches. Close if replay is already stable
without avoidable work or safe invalidation cannot be specified. Sporadic moving
stalls do not justify graph rewrites; microsecond replay gains need critical-path
and model-workload evidence before a throughput claim.

## Bounded iteration outcome

Completed: four PR states and representative actual source reviewed; pinned
rocPRIM public headers read; host availability and existing activation gates
checked; five next-work cards recorded. Negative findings are actionable:
installed rocPRIM lacks top-k, source flags do not guarantee stable sorted
selection, Vulkan determinism differs from TOPK v1, old L2 fusion does not prove
corrected GDN coverage, and much proposed fusion/tile work already exists upstream.

The coordinator's refreshed portfolio was independently reviewed read-only and
approved for its doc-only kickoff. The coordinator accepted clarifications to
distinguish conformance hardware admission from timed arms and source/GGUF-derived
shape metadata from observed runtime layout, and committed that kickoff locally.
This report does not edit that
portfolio, implement providers, or authorize hardware work.

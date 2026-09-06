# Card 3: corrected q/k normalization, width 128

Read-only research, 2026-09-05. **Source-level opportunity, not an observed runtime
boundary or speedup.** No compilation, model/GPU execution, or SDK installation.
Proposed boundary: two independent strided f32 q/k views to two contiguous
normalized outputs, preserving the corrected RMS_NORM+SCALE formula.

## Evidence identities and actual consumer

- Existing benchmark source remains
  `ca94157f70a2776e8da6b6849b50b45a083d0478` in
  [the source profile](../configs/sources/strix-llama.yaml).
  Its [Qwen3.5 graph, lines 432–433][control] calls `ggml_l2_norm` directly.
  It is **not** the corrected-formula baseline for this proposed experiment.
- Current upstream master was resolved through authenticated GitHub API to
  `c7af5c6c29902eb1f7b3bd7952607e2349e1c668`; source was inspected at that exact
  commit in a separate temporary clone. This does not update the benchmark pin.
  [Qwen3.5 `build_layer_attn_linear`][consumer] builds SSM convolution → SiLU →
  q/k views → `build_gdn_l2_norm` → optional head repeat → `build_recurrent_attn`.
  [The helper][helper] constructs `RMS_NORM(x, eps/n)` then `SCALE(1/sqrtf(n))`.
  [The recurrence dispatcher][recurrence] consumes these outputs in fused GDN
  or autoregressive/chunked paths. Qwen3Next uses the same helper at lines 513–514.
- Shape evidence is the **previously recorded**, receipt-linked Qwen3.5-4B
  metadata, not a new model inspection or runtime layout observation. Under
  `~/.local/share/strixlab/`, metadata file
  `models/metadata/v1/05b6c74baf4c9f33a00135081eff7141ab8372fcceb5e934117fc7c00ed64d14.json`
  has that exact SHA-256 (rechecked). Receipt
  `models/receipts/v1/qwen35-4b-smoke/e07e48ea62bcd1b625776438b27329acb877f898ca490f28f749524f6a3047ff.json`
  links it to artifact SHA-256
  `13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983`, matching
  [the model profile](../configs/models/qwen35-4b-smoke.yaml). Metadata gives
  state size 128, group count 16, inner size 4096, time-step rank 32 and
  RMS epsilon `9.999999974752427e-7`. The artifact itself was not rehashed here.

Combining that metadata with the current graph gives q/k shape `[128,16,T,S]`
in GGML dimension order, value shape `[128,32,T,S]`, and packed token width 8192.
Input byte strides are `[4,512,32768,32768*T]`; q offset is 0, k offset 8192,
v offset 16384. Output strides are `[4,512,8192,8192*T]` per branch. This is a
source-derived layout expectation; allocator addresses, scheduler placement,
dispatch counts and frequency remain unobserved.

## Existing fusion is not this corrected boundary

[Current HIP-shared backend source][matcher] still matches
`L2_NORM → VIEW → L2_NORM` for the dual width-128 launch (lines 3680–3703).
The larger GDN decode matcher also requires two `L2_NORM` nodes (3055–3078).
The corrected helper emits different opcodes; these matchers cannot consume it.
Existing RMS fusion dispatch at 4709–4728 requires `MUL` (optionally ROPE/ADD),
not scalar `SCALE`. The [GGML constructors][constructors] preserve those distinct
opcodes. Thus source predicts two RMS launches plus two scale launches for q/k;
it does not demonstrate four observed launches. Existing convolution/recurrence
fusion is outside this boundary and is not proposed for rediscovery.

## Frozen math, layout and safety contract

For branch b ∈ {q,k}, each independent row has n=128 and epsilon ε_b > 0:

`E_b = sum_i x_b[i]^2;  y_b[i] = x_b[i] / sqrt(E_b + ε_b)`.

Equivalently in real arithmetic:
`(x_b[i] / sqrt(E_b/128 + ε_b/128)) / sqrt(128)`.
Old L2 instead uses `x_b[i]/sqrt(max(E_b, ε_b²))`; never use it as this baseline.
For a one-hot row with value ε=1e-6, old L2 is approximately 1 while corrected
output is approximately 0.001. For E=ε the corrected row's squared norm is 1/2.

The execution contract preserves the actual graph's f32 parameters
`e_b = float(ε_b/128)` and `a = float(1/sqrtf(128))`, f32 square/sum reduction,
`r = rsqrtf(float(E_b/128 + e_b))`, f32 intermediate `z_i = r*x_i`, then
`y_i = a*z_i + 0`. Keep this two-stage epilogue; do not silently substitute
`rsqrt(E+ε)` or combine the multipliers under reassociation. Record compiler
FP/contraction/denormal flags. Different reduction trees are allowed within the
frozen tolerance; bitwise agreement across implementations is not required.

Initial domain: finite f32 `|x_i| <= 16`, finite f32 ε in `[1e-8,1e-4]`, fixed
width 128, positive float-aligned outer strides, contiguous elements (`nb0=4`),
nonoverlapping logical input rows, contiguous outputs. Source q/k views may share
a backing allocation but their logical elements must not overlap. No in-place
or partial input/output alias, overlapping q/k outputs, or unsupported strides:
reject the standalone request or retain the original graph. Check full byte
ranges, bounds and integer overflow. Eliminated RMS intermediates/views must
have no external readers or output flag; final q/k outputs may have consumers.
Retain conservative fusion range checks even if an allocator could reuse storage.

## Smallest standalone problem and frozen probes

Minimum: one f32 backing buffer with `[q128,k128,v128]`, q/k offsets 0/512 bytes,
two separate 512-byte outputs, independent epsilon arguments. v and padding are
canaries. No convolution, model weights, state, repeat, or recurrence is needed.
The model-layout coordinate is `[128,16,T,S]` above with `(T,S)=(1,1),(2,2)`;
the second catches token/sequence addressing. Neither is a model workload.

Freeze these probes before either candidate exists:

| Probe | Inputs / required result |
|---|---|
| Zero | All +0, then mixed signed zeros; finite numerical zero outputs, unchanged inputs/canaries. No output zero-sign requirement. |
| Epsilon semantics | One-hot x=ε, and constant rows with x=f32(sqrt(c*ε/128)), c=`2^-10,1,2^10`; test ε=`1e-8,1e-6,1e-4`. Compute expectations from actual rounded input bytes. |
| Unequal epsilon | Identical q/k data near E=1e-6, ε_q=1e-6 and ε_k=1e-4, then swap; outputs must follow the branch epsilon. Actual consumer currently supplies equal values. |
| Reduction/layout | Alternating signs, one-hot at columns 0/31/32/127, alternating magnitudes 16 and 2^-12; distinct row/head/token/sequence sentinels. Test model strides and padded rows (`nb1=528`, with disjoint padded outer planes). |
| Exclusions | `nb0=8`, width 127/129, ε=0/negative/NaN, nonfinite input, exact/partial input-output alias, overlapping outputs, external RMS reader/output: reject or decline fusion. No writes on standalone rejection. |
| Replay | Later admitted GPU fixture: ten replays at fixed addresses alternating zero, near-epsilon and asymmetric contents; recheck every output and canary. |

Independent reference: host f64 sum of squares in index order and sqrt, using
the supplied f32 bytes and ε, outside provider code. For **every element** require
finite output and `abs(y-reference) <= 2e-7 + 2e-5*abs(reference)`; exact numerical
zero for zero input, byte-exact unchanged inputs/canaries. Also check against an
f64 evaluation using the rounded graph parameters e_b and a, and compare candidate
to baseline with the same bound. All gates apply to baseline too. No tolerance
relaxation after seeing a candidate; a baseline failure stops for contract review.

## Baseline, candidates and prerequisites

**Baseline:** extract the current-source [f32 RMS kernel][norm] (256 threads/row
for n<1024) and [scale kernel][scale], preserving strides, scalar parameters and
two passes per branch. Its custom block reduction uses warp shuffle sums,
shared warp partials and broadcasts the result to participating warps. Measure
the whole q+k operation if later admitted, including scratch/materialization.

**Candidate A:** one launch dispatching q/k rows independently, one logical
32-lane warp per row, four values per lane, the existing HIP shuffle reduction,
and the corrected two-stage epilogue before contiguous stores. Reuse only the
old dual kernel's row/branch mapping, never its `max(E,ε²)` epilogue. The proposed
mechanism removes scalar passes and batches launches; net cost is unmeasured.

**Candidate B:** same operation with 128 threads/row and one squared value/thread,
`rocprim::block_reduce<float,128,rocprim::block_reduce_algorithm::using_warp_reduce>`.
Context7 reduction docs were checked first; exact source at rocPRIM pin
`8d1ae90eff7d022f26019ec55b2ec6a7674b3112` supplies the [public API][rocapi] and
[warp-partial implementation][rocwarp]. Unlike GGML's helper, do not assume every
thread receives the final sum: thread 0 must publish it and all writers synchronize.
Provide `storage_type`, `rocprim::plus<float>`, and a barrier before storage reuse.
The same pin's [raking alternative][rocrake] stores per-thread values then reduces
segments in one warp; it adds a distinct shared-memory/reduction order and is
not a third candidate or presumed improvement. Floating addition is not
associative; the public API explicitly warns of precision/order variation.

Standalone compilation needs an admitted HIP compiler/runtime, gfx1151 target,
identified wave size, and complete pinned rocPRIM headers/generated configuration
for B; header inspection is not compiled availability. No BLAS is intrinsically
required for this standalone reduction. Full backend integration additionally
uses [GGML HIP's CMake dependencies][hipbuild] (HIP, hipBLAS, rocBLAS). Freeze the
compiler, source/header hashes, architecture/FP flags, launch geometry and shared
storage; establish runtime library closure before execution. The existing host
transport target cannot admit this GPU operation. ROCm 10 provisioning remains
under the separate [bring-up contract](rocm10-bringup.md).

**Stop:** first obtain separate approval for a standalone correctness fixture;
then verify dispatch on that fixture before timing. Close if the chosen source
already removes this boundary, safe matching cannot be stated, baseline/candidate
fails frozen probes, the exact API cannot compile, or no material cost remains.
At most these two candidates, no warp/epsilon/tolerance sweep. Any later model
claim needs separately admitted current-source graph/layout evidence and
logit/token validation; comparing against the old-formula control is invalid.

[control]: https://github.com/halo-box/strix-llama.cpp/blob/ca94157f70a2776e8da6b6849b50b45a083d0478/src/models/qwen35.cpp#L432-L433
[consumer]: https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/src/models/qwen35.cpp#L351-L464
[helper]: https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/src/models/models.h#L14-L18
[recurrence]: https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/src/models/delta-net-base.cpp#L425-L445
[matcher]: https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/ggml/src/ggml-cuda/ggml-cuda.cu
[constructors]: https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/ggml/src/ggml.c#L3396-L3420
[norm]: https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/ggml/src/ggml-cuda/norm.cu#L77-L155
[scale]: https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/ggml/src/ggml-cuda/scale.cu
[rocapi]: https://github.com/ROCm/rocm-libraries/blob/8d1ae90eff7d022f26019ec55b2ec6a7674b3112/projects/rocprim/rocprim/include/rocprim/block/block_reduce.hpp
[rocwarp]: https://github.com/ROCm/rocm-libraries/blob/8d1ae90eff7d022f26019ec55b2ec6a7674b3112/projects/rocprim/rocprim/include/rocprim/block/detail/block_reduce_warp_reduce.hpp
[rocrake]: https://github.com/ROCm/rocm-libraries/blob/8d1ae90eff7d022f26019ec55b2ec6a7674b3112/projects/rocprim/rocprim/include/rocprim/block/detail/block_reduce_raking_reduce.hpp
[hipbuild]: https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/ggml/src/ggml-hip/CMakeLists.txt#L42-L64

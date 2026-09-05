# Qwen3.5-4B MMVQ attribution and six-trial diagnostic study

Date: 2026-09-05. The parent coordinator executed six serial TG128 trials;
this research agent prepared the temporary launcher and analyzed evidence offline.
No baseline source/build/model change, candidate, campaign, or synthetic benchmark.

## Findings

Logical shape membership is stable across all four profiled repetitions. The
largest Q4_K fused group is all 32 FFN gate/up pairs. The large Q6_K nonfused
groups are recurrent QKV/dense Q projections and the tied vocabulary output.
The vocabulary output consistently takes about 2.3 ms per late token in the
new traces. The original 10.646 ms FFN layer-15 interval does not recur at that
slot; isolated large intervals occur at other layers/shapes in both trace modes.

Two repetitions per mode, variable clean timings, and occasional large intervals
do not establish a profiler-mode effect or its cause. Observed device-start
order differs from dispatch order in five kernel-only tokens; overlaps stay
within their tokens. Validated dispatch signatures establish membership, not
independent timing accuracy. No safely removable work or dispatch change is
established, and no end-to-end gain is claimed.

## Provenance and method

- Reused baseline diagnostic `baseline-profile-20260905-DdQvJ0`, case `tg128`.
- Source HEAD verified read-only as `ca94157f70a2776e8da6b6849b50b45a083d0478`.
  All five cited source files also pass `git diff --quiet HEAD` (including
  staged changes); source observations therefore match that pinned revision.
- Historical authenticated build: `af360810acf2b021d92a3bad67e752c9cb9b2b0ddac8bff76cdedb3fe08af474`.
- Model identity inherited from verified baseline: GGUF SHA-256
  `13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983`.
  This pass read only the GGUF header, metadata, and tensor directory, ending
  at byte 10,969,043; no weight payload read or printed. No fresh full-model hash.
- Recomputed kernel CSV SHA-256
  `b4783d37bdd568362e5daaf5ce3b905577b3f5761af5de34042429d611d5d650`
  and HIP CSV SHA-256
  `895d9a60cf172c9a5eedff528103fe45a3b71cb286d048bf1ee1f890ef066979`;
  both match preserved derived summary input hashes. Historical launcher status
  records build/model verification before and after collection.
- Reused late96 proxy bounds `[1651093944315452,1651095851874999)` ns,
  corresponding to measured tokens 33–128. Every selected kernel is wholly
  inside the window. Grouped by full kernel variant and grid/workgroup tuple.
- All 96 graph correlations contain 869 kernels. Sorting each by device start
  timestamp gives identical kernel-name/grid-X/grid-Y sequences. Layer assignments
  below are reconstructed from source, tensor inventory, counts, and that order;
  they are not runtime tensor-name annotations.

## Ranked map from the original baseline

Geometry is GGML weight `ne=[K,N]`, with activation `[K,1]` and output `[N,1]`.
All listed launch workgroups are `(32,1,1)` and recorded grid Y/Z are one.
The rocprof grid X is total work-items: dividing by 32 gives launch block X.
Source establishes one output row per block here, allowing N to be recovered;
grid X alone cannot establish K or tensor identity.

| Rank | Binding, logical geometry | Type / fusion | Calls per token | Late96 sum ms | Median µs | Recorded grid X |
|---|---|---|---:|---:|---:|---:|
| 1 | `blk.0–31.ffn_up` plus `ffn_gate`, each `[2560,9216]` | Q4_K / gate + SwiGLU | 32 | 444.821 | 140.144 | 294912 |
| 2 | recurrent `attn_qkv` plus four dense `attn_q`, `[2560,8192]` | Q6_K / none | 28 | 238.515 | 87.846 | 262144 |
| 3 | tied `token_embd.weight` used as output, `[2560,248320]` | Q6_K / none | 1 | 236.986 | 2450.012 | 7946240 |
| 4 | `ffn_down`, `[9216,2560]` | Q6_K / residual addition | 16 | 164.667 | 108.324 | 81920 |
| 5 | `ffn_down`, `[9216,2560]` | Q4_K / residual addition | 16 | 115.308 | 76.224 | 81920 |

Ranks 1 and 5 sum to the previously reported Q4_K fused 560.129 ms.
Rank 2 plus rank 3 plus four Q6_K attention V projections `[2560,1024]`
(4.936 ms) sum to Q6_K nonfused 480.437 ms. Rank 2 separates by reconstructed
layer order into recurrent QKV 204.998 ms (24/token) and dense Q 33.517 ms
(4/token). These are duration sums, not removable critical-path fractions.

Layer sets, zero-based:

- Recurrent QKV: 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30.
- Q6 dense Q and V: 3,15,27,31. Remaining dense Q/V are Q4_K at 7,11,19,23.
- Q6 FFN down: 0,1,2,3,6,9,12,15,18,21,24,27,28,29,30,31.
- Q4 FFN down: 4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26.

GGUF has 33 blocks including one NextN/MTP block. Source excludes MTP block 32
from the ordinary decoder graph; the 32 FFN dispatches per token agree. There
is no separate `output.weight` in the tensor directory; source duplicates the
token embedding for output when absent. The final graph slot 868, N=248320,
matches that binding uniquely.

## Dispatch, strides, and fusion

Pinned `ggml/src/ggml-cuda/mmvq.cu:77` and `:95` map RDNA3_5 to the RDNA2
parameter table. `calc_nwarps` at `:397` falls through to one warp for that
table; `calc_rows_per_block` at `:524` gives one row. `calc_launch_params`
at `:837` uses `(ceil(N/rows_per_block),channels,samples)` blocks and
`(warp_size,nwarps,1)` threads. Recorded `(32,1,1)` agrees. These are not
the RDNA3_0 Q6 two-warp or RDNA4 eight-warp paths. Template arguments explicitly
show `ncols_dst=1`, `small_k=false`, `halve_iters=false`.

At `mmvq.cu:1249`, F32 activation is quantized into temporary Q8_1 rows, with
K padded to 512. All relevant K values (2560,4096,9216) already satisfy that
padding. Source computes weight row stride as `src0.nb[1]/type_size`, activation
column stride as padded K/32 Q8_1 blocks, and output column stride as
`dst.nb[1]/4`. Ordinary contiguous weights imply the following byte row strides:

| Weight | Blocks per row (K/256) | Quantized byte row stride |
|---|---:|---:|
| Q4_K K2560 | 10 | 1440 |
| Q6_K K2560 | 10 | 2100 |
| Q4_K K9216 | 36 | 5184 |
| Q6_K K9216 | 36 | 7560 |

Block sizes follow `ggml/src/ggml-common.h:338` (Q4_K=144 bytes) and `:368`
(Q6_K=210 bytes). These strides are source-derived contiguous-layout expectations;
CSV does not record actual runtime `nb[]` or pointers. There is no evidence of
a transposed/view weight in these bindings, but runtime strides are not directly
observed.

`has_fusion` at `mmvq.cu:859` tests gate, bias, or scale pointers; it does not
mean every fused kernel evaluates two matrices. Gate fusion requires matching
type and strides (`:1291`), and graph fusion checks matching shapes and shared
activation. The FFN graph uses parallel SILU gate/up with no bias
(`src/models/qwen35.cpp:474`); the adjacent matmul/matmul/GLU path binds up to
src0 and gate to `fusion.gate` (`ggml-cuda.cu:3767`).

Conversely, FFN down is followed by residual addition (`qwen35.cpp:193`).
The matmul/add fusion path (`ggml-cuda.cu:3901`) supplies that residual as
`fusion.x_bias`. It is an activation residual, not a learned model bias. Counts,
types, and row geometry resolve the 16 Q4 and 16 Q6 fused down calls. Scale
fusion is restricted to NVFP4 in this MMVQ entry point, excluding that explanation.

## Original baseline intervals

- Vocabulary output: 96/96 intervals exceed 1 ms; median 2450.012 µs,
  p90 2560.500, p99 2695.795, max 2720.241. This is a consistently large
  dispatch within the recorded run.
- Q6 N8192: p99 103.555 µs, max 124.875; no interval exceeds 1 ms.
- Q4 gate/up: 3072 calls; p99 170.962 µs; exactly one exceeds 1 ms.
  The outlier is token 37, dispatch 32835, correlation 57826, graph slot 430,
  reconstructed FFN layer 15: 10646.040 µs. That same slot across all 96
  tokens has median 134.253 µs, p99 146.075 µs. Its exceptional excess is
  about 10.5 ms. It does not explain the whole 444.821 ms group cost.
- The host is inside `hipStreamSynchronize` throughout that outlier. Neighboring
  RMS norm and activation quantization take 3.286/1.363 µs; following quantization
  takes 2.124 µs. This excludes neither GPU scheduling effects nor profiler
  interference, and the waiting API does not identify the cause.
- Quantiles use sorted sample index floor((n-1)*p); medians use the standard
  midpoint convention. The follow-up repetitions below add cross-run observations;
  no scheduler trace or hardware counters were collected. Repetition within a
  profiled graph is not unprofiled repeatability.

## Is avoidable work established?

The existing layout handling is appropriate: contiguous quantized rows, a
single token, one row per block, Q8_1 activation conversion, gate/up fusion,
and residual-add fusion. K already satisfies the 512 padding requirement;
there is no padded K tail to remove in these shapes. FFN fusion already avoids
separate gate/up launches and the residual fusion avoids a separate add launch.
The full 248320-row output is consistent with exact full-vocabulary logits;
truncating it is not an authorized exact-token optimization.

This establishes correct dispatch coverage, not optimality. The RDNA2 table
uses one warp regardless of these different K/N shapes; timing alone does not
show that an alternative is better. No unnecessary matrix, wrong layout,
redundant copy, or safely removable operation is established by this evidence.
Activation conversion is visible but its safe reuse across operations has not
been demonstrated. No candidate mechanism is ready.

The vocabulary projection has the same K as QKV/Q but 30.3125 times as many
output rows. Its median duration is about 27.9 times that group's median.
This is consistent with a much larger logical workload and gives no evidence
that the millisecond scale alone is anomalous. It is not a bandwidth measurement
or a proof that the output dispatch is optimal.

## Executed six-trial protocol

The coordinator completed exactly six separate serial invocations in fixed order:
`clean, full-trace, kernel-only, kernel-only, full-trace, clean`. Every child used
the same authenticated build/model and `-m <leased-fd> -p 0 -n 128 -r 1 -o jsonl`.
The preserved temporary launcher retained build/model pre/post leases, canonical
child environment, the exclusive GPU lock, a private process group, a 300-second
timeout and 15-second termination grace. No hardware trial was rerun.

Kernel-only used `--kernel-trace --stats --output-config --output-format csv json`,
with no HIP/copy/allocation/Perfetto collection flags. Full-trace used the original
HIP/kernel/copy/allocation trace flags, stats/config, CSV/JSON/pftrace output,
and the in-process Perfetto backend. Clean invoked the same bench directly.
The launcher was frozen throughout all six invocations, SHA-256
`0152bce7bca73363de7dd2ccc8095669174ed892acecacd702b48b66b6f6dbc5`.

**Preserved readiness failure and protocol amendment:** after trial 1, the initial
pre-trial-2 readiness check was blocked by utilization samples 30%,16%,9% against
the 10% limit; no trial-2 child existed. `pre-trial2-blocked-doctor.json` was
preserved. One deliberate cooldown recheck passed, then a fixed 10-second cooldown
preceded subsequent readiness probes. This amended pacing within the study and
is retained as a limitation. It was not a repeated hardware trial. Trials 2–6
completed, and the final doctor check passed.

All trial build/model leases and expected source identities passed, with one
JSONL sample each. The coordinator validated all collected CSV/JSON counts.
Offline analysis independently checked benchmark settings and identities against
the original baseline, strict profiler JSON decoding, kernel CSV/JSON timestamp
pairs, and kernel domain counts/duration sums. Kernel-only JSON had no nonempty
HIP/copy/allocation record domains. Each profiled trial contains 112,352 kernels.
Original raw evidence and the blocked readiness record remain unchanged.

| Trial | Mode | JSONL sample ms | Observed tokens/s |
|---|---|---:|---:|
| 1 | clean | 2480.775 | 51.597 |
| 2 | full-trace | 2561.368 | 49.973 |
| 3 | kernel-only | 3022.006 | 42.356 |
| 4 | kernel-only | 2637.340 | 48.534 |
| 5 | full-trace | 2435.134 | 52.564 |
| 6 | clean | 2750.694 | 46.534 |

These are individual diagnostic observations, not an optimization result or
causal estimate of profiler overhead. There are only two processes per mode;
clean timings also vary, and full-trace is not consistently slower. Do not pool
these samples with clean suite scoring evidence.

## Validated membership and observed timing order

Each profiled trial has exactly 129 vocabulary-output anchors: warmup plus 128
measured tokens. Using unique dispatch-ID order, every token 33–128 contains
exactly the same complete 869-kernel signature as the baseline, including full
kernel name, grid, and workgroup dimensions. That gives 83,424 late dispatches
per trace. Source/metadata-based binding and layer inference use this stable
membership. Both full traces independently have 129 D2H/terminal-sync boundaries,
and their selected late dispatch sets exactly match the original proxy method.

The initial analyzer sorted by device-start timestamp and failed at trial 3,
token 60. This failure and original analyzer were preserved before changing the
membership order. A bounded check found the following adjacent start inversions
in dispatch order, all on **queue 2, stream 3**:

| Trial | Tokens affected | Adjacent inversions | Start inversion ns | Pair overlap ns |
|---|---|---:|---:|---:|
| 3 (kernel-only) | 60 | 1 | 317 | 1283 |
| 4 (kernel-only) | 102–105 | 17 | 131–12396 | 1082–2925 |

Neither full trace has these order inversions in late96. No late dispatch in any
profiled trace starts before the preceding token's vocabulary-output end or ends
after its own vocabulary-output end. The overlaps therefore stay within token
bounds. The observations can reflect scheduling/overlap; they do **not** establish
a profiler defect. Raw start/end timestamps and durations were not repaired or
reordered in the evidence. Dispatch ordering is used only for membership and
matching slots; it does not prove execution order, timestamp accuracy, or cause.
No kernel-only proxy wall-time, utilization, or critical-path reconstruction is
reported.

## Shape-specific observations across repetitions

Late96 duration sums in ms, retaining all outliers:

| Binding | Trial 2 full | Trial 3 kernel | Trial 4 kernel | Trial 5 full |
|---|---:|---:|---:|---:|
| Q4_K FFN gate/up | 411.712 | 409.316 | 408.302 | 423.484 |
| Q6_K recurrent QKV | 305.942 | 193.376 | 190.933 | 199.277 |
| Q6_K dense Q | 32.245 | 32.655 | 31.705 | 31.635 |
| Q6_K vocabulary output | 245.808 | 224.564 | 220.669 | 222.677 |
| Q6_K FFN down + residual | 155.541 | 156.851 | 154.379 | 155.441 |
| Q4_K FFN down + residual | 110.103 | 110.097 | 261.959 | 111.436 |

These sums describe traced intervals and are not removable wall-time fractions.
The preserved derived data also contains per-token and per-layer slot statistics.

| Statistic | Trial 2 full | Trial 3 kernel | Trial 4 kernel | Trial 5 full |
|---|---:|---:|---:|---:|
| Vocabulary median µs | 2314.623 | 2333.158 | 2288.012 | 2307.990 |
| Vocabulary p90 µs | 2382.451 | 2391.670 | 2350.190 | 2376.520 |
| Vocabulary maximum µs | 24946.460 | 2465.608 | 2387.861 | 2435.671 |
| FFN gate/up median µs | 133.491 | 133.171 | 132.770 | 132.490 |
| Original FFN layer-15 slot maximum µs | 152.046 | 142.228 | 145.635 | 143.591 |

All 384 late vocabulary intervals exceed 1 ms. The recurring multi-millisecond
vocabulary workload is distinct from the original 10.646 ms layer-15 FFN outlier:
that slot's new medians are 125.416–129.404 µs, with no new interval above 1 ms.

Large isolated intervals are nevertheless present elsewhere: trial 2 recurrent
QKV layer 21 takes 102.274 ms (token 49), trial 4 Q4_K FFN down layer 19 takes
154.885 ms (token 94), and trial 5 FFN gate/up layer 0 takes 14.742 ms (token 119).
Other preserved MMVQ outliers include trial-2 QKV layer 1 at 11.169 ms, a Q4_K
N2560 nonfused interval at 3.450 ms, a vocabulary interval at 24.946 ms, and
trial-5 Q8_0 N2560 at 41.009 ms and QKV layer 22 at 7.219 ms. They must not be
silently discarded or attributed to normal shape cost. Their cause remains
unresolved; occurrence in both collection modes does not establish or exclude
profiler interference, scheduling effects, or another source of delay.

## Evidence preservation and limits

The local diagnostic root `mmvq-shape-profile-20260905-6trial` retains all six
trial directories and readiness records. Its `shape-analysis/` directory contains
the original failed timestamp-order analyzer, failure note, final dispatch-order
analyzer, full derived JSON, overlap check script/results, full-trace membership
checks, metadata-only tensor inventory, and a SHA-256 manifest. These are local
research artifacts, not a portable verified suite bundle. The report is the
privacy-safe publication artifact; no weights or raw evidence are checked in.

The six benchmark runs did not emit greedy token sequences. They retain the
unchanged baseline identity and prior exact cross-run 64-token gate, digest
`8c8b081313ea49a1ddbfe87a2f582897507822bbc5439cca7f86d196ff3bd50a`;
no new token-parity claim is made from llama-bench. Runtime tensor names/strides
remain source-and-metadata inferences rather than direct dispatch annotations.
No scheduler trace, hardware counters, or profiler-free per-kernel times were
collected. Current dispatch handles these shapes, but its optimality is not
established. This completed study supports attribution and identifies sporadic
large intervals; it selects no optimization mechanism or candidate campaign.

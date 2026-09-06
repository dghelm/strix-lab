# ROCm 10 kernel research

The local compile → correctness → timing loop works on the Radeon 8060S (`gfx1151`).
The first optimization replaces a complete bitonic sort and output-index copy with
one block-per-row reduction for K=1. Both implementations use the existing CPU
bit-value/index oracle.

## First measured optimization

First trial, optimized builds, microseconds per provider call across all rows:

| Rows | Columns | Bitonic adapter | K1 reduction | Ratio of medians |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 6.845 | 2.425 | 2.82× |
| 1 | 256 | 10.117 | 2.446 | 4.14× |
| 1 | 1024 | 14.961 | 2.800 | 5.34× |
| 64 | 32 | 6.951 | 2.451 | 2.84× |
| 64 | 256 | 10.874 | 2.509 | 4.33× |
| 64 | 1024 | 28.773 | 2.881 | 9.99× |

These results cover **finite, distinct values within each row, K=1**, using identical
deterministically shuffled inputs for both providers. This is a separate research
experiment. It does not qualify the full TOPK-v1 matrix or measure rocPRIM or the
ROCm 7.2.4 control.

The [portable results](results/2026-09-05-k1.json) retain raw samples, repeated-trial
results, source/binary hashes, and measurement details. The 256-element HIP smoke
also passed, followed by five restricted bitonic TOPK cases with exact oracle
checks. All completed execution phases preserved SDK semantic/physical identity;
metadata coverage remains unknown and vendor authenticity unverified under the
[private execution amendment](../../docs/rocm10-private-execution-plan.md).

## Second iteration: smaller blocks and wave reduction

Two further variants are measured directly against the original 256-thread K1
reduction above, using the same six shapes. The original kernel remains frozen.
The small-block variant uses 32 threads for rows with at most 32 columns and 256
threads otherwise. The wave variant keeps 256 threads but replaces the shared
reduction with wave32 shuffles and one block-wide barrier.

Ratios of baseline median to candidate median (larger is faster):

| Rows | Columns | Small, trial 1 | Small, trial 2 | Wave, trial 1 | Wave, trial 2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 1.080× | 1.080× | 1.015× | 1.016× |
| 1 | 256 | 0.998× | 0.993× | 1.023× | 1.021× |
| 1 | 1024 | 1.000× | 0.997× | 1.024× | 1.022× |
| 64 | 32 | 1.089× | 1.087× | 1.009× | 1.008× |
| 64 | 256 | 0.998× | 1.000× | 1.018× | 1.020× |
| 64 | 1024 | 1.000× | 0.998× | 1.014× | 1.016× |

The clearest next candidate is the small block for 32-column rows: both trials
show roughly 1.08–1.09× speedup over the first K1 optimization. In the repeat,
64 rows of 32 values took 2.453 µs with the original K1 kernel and 2.256 µs with
the small block. Wider-row results include small regressions and are retained.
The wave variant's observed gains are modest (roughly 0.8–2.4%); these two trials
do not establish statistical significance or a general performance guarantee.

[Second-iteration results](results/2026-09-05-k1-variants.json) preserve all 960 raw
samples across two independent processes using the same executable. Each
candidate has its own paired baseline measurements; small and wave timings are
not paired directly with each other. Every case passed the CPU oracle before
and after timing. The fixture checks `gfx1151`, PCI identity and `warpSize == 32`
before allocation. Host tests additionally exercise actual kernel bodies with
emulated block/wave synchronization across ties, signed zeros and partial rows;
the GPU timing experiment remains limited to finite, distinct values.

## Third iteration: one wave per row

A 32-thread shuffle reduction eliminates shared memory and block-wide barriers.
It reads longer rows with a stride of 32. Its baseline is the frozen small-block
variant from the second iteration, not the bitonic adapter or llama's argmax.

| Rows | Columns | Baseline / one-wave, trial 1 | Baseline / one-wave, trial 2 |
| ---: | ---: | ---: | ---: |
| 1 | 32 | 1.036× | 1.034× |
| 1 | 256 | 0.840× | 0.838× |
| 1 | 1024 | 0.513× | 0.515× |
| 64 | 32 | 1.032× | 1.033× |
| 64 | 256 | 0.845× | 0.847× |
| 64 | 1024 | 0.511× | 0.513× |

The short-row improvement is about 3%, while 1024-column rows are roughly 1.95×
slower. Do not replace the wider-row implementation with this candidate.
[Third-iteration results](results/2026-09-05-k1-onewave.json) retain all 480 raw
samples from two runs, including regressions. All six cases in each run passed
pre/post CPU-oracle checks. The same finite-distinct research scope and timing
limitations apply.

### Relationship to Halo llama

Current `halo-box/strix-llama.cpp` already uses warp-shuffle argmax reductions and
32-thread dispatch for short rows. Greedy backend sampling calls that argmax;
our sort-versus-reduction speedup does not measure an improvement over it. HIP
TOP_K remains a separate sorting/copying path without a K=1 shortcut. None of
these research kernels has been ported into that repository.

The [upstream comparison](../../docs/rocm-k1-upstream-overlap.md) pins source and
call-site evidence. Before claiming a useful integration, measure actual K=1
TOP_K/ARGSORT usage and compare with the existing argmax implementation at real
workload sizes. These experiments demonstrate the local research loop, not a
new argmax algorithm or an end-to-end llama speedup.

## Actual workload follow-up

The recorded Qwen3.5-4B smoke server used CPU sampling (`backend_sampling: false`),
with temperature 0 and top-k 40. Its dense model graph has no MoE selection path.
The existing random-token `llama-bench` traces show no selection-kernel names;
that absence describes those benchmark runs, not every possible sampler setup.

Consequently, further K1 micro-optimization is not the current priority for this
workload. [Workload priorities](../../docs/qwen35-workload-priorities.md) join the
actual model metadata, pinned source and late-decode trace. The largest matched
Q4_K fused feed-forward group accounts for about 23.3% of that diagnostic interval;
the tied Q6_K output projection accounts for about 12.4%, with a 2.45 ms median
per dispatch in the original diagnostic trace (about 2.3 ms in the follow-up
repetitions). These are trace observations and source-supported attributions,
not measurements of an optimization or proof of a bandwidth bottleneck.

These shape results were already established in the
[six-trial attribution study](../../field-reports/2026-09-05-qwen35-mmvq-shape-attribution.md),
which also preserves recurring vocabulary-projection cost, sporadic stalls and
analysis limitations. This follow-up adds the actual server sampling evidence
and connects it to the choice of primitive; it does not repeat that study.
The profiled source is the older `ca94157` ROCm 7.2.4 control, separate from the
current Halo source inspection and private ROCm 10 primitive experiments.

## Corrected GDN q/k normalization fusion

The next experiment follows Qwen3.5's actual recurrent-layer operations. The
reference is a bounded extraction of current Halo `c7af5c6`:
`RMS_NORM(eps / 128)` then `SCALE(1 / sqrtf(128))`, separately for q and k.
The candidate combines the four launches into one while retaining the same
256-thread reduction order, separate q/k epsilon, FP32 intermediate rounding,
and scale-plus-zero-bias operation. It does not use the older L2 epsilon formula.

Both implementations use 128-wide rows, 16 q/k heads, one sequence, and the
model's interleaved layout: 8192 floats per token, with k starting 2048 floats
after q. Inputs are deterministic synthetic values; model weights are not loaded.

Captured graph results on `gfx1151`, microseconds per complete q/k provider call:

| Tokens | Reference, trial 1 | Fused, trial 1 | Ratio, trial 1 | Reference, trial 2 | Fused, trial 2 | Ratio, trial 2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.174 | 2.507 | 3.261× | 8.193 | 2.528 | 3.241× |
| 16 | 10.919 | 4.989 | 2.189× | 10.865 | 4.994 | 2.175× |
| 512 | 131.665 | 109.086 | 1.207× | 130.217 | 106.190 | 1.226× |

Each graph contains 100 calls on fixed inputs; capture and instantiation are
excluded. Each shape also has separate direct-dispatch measurements. Both modes
use five warmup batches and 20 alternating measurement pairs. These are two
independent processes using the same executable, with 480 raw timing samples
in total. [Portable results](results/2026-09-05-gdn-norm.json) preserve both modes,
all samples, source/binary identities, and phase receipts.

Every case passed bitwise GPU reference/candidate parity before and after timing,
as well as a float64 corrected-formula check. Additional two-sequence diagnostics
cover signed zero, near-epsilon, tiny, large, and mixed values with unequal q/k
epsilon. Output/scratch guards and input-integrity checks passed. The float64
thresholds are diagnostic tolerances, not a proven bound for all finite inputs;
GPU bitwise parity is a separate requirement. Host tests execute the actual kernel
bodies with emulated barriers/shuffles and check invalid geometry and aliasing.

This is a measured standalone primitive improvement, not a full llama speedup.
The captured batches are not llama token graphs, and the extracted reference has
not been qualified against a compiled current Halo backend. Full integration
still needs graph-matcher/consumer guards and model recurrence, logits, and token
parity checks. Clocks, power, and unrelated GPU clients remain uncontrolled.

To reproduce, use the existing launcher with fresh output directories:

```bash
PYTHONPATH=src .venv/bin/python research/rocm10/run.py compile-gdn-norm \
  --output "$STRIX_TRIAL_DIR/build" --diagnostic
PYTHONPATH=src .venv/bin/python research/rocm10/run.py run-gdn-norm \
  --binary "$STRIX_TRIAL_DIR/build/work/gdn-norm-compare" \
  --binary-sha256 REPLACE_WITH_EXECUTABLE_SHA256 \
  --output "$STRIX_TRIAL_DIR/run"
```

Set `STRIX_TRIAL_DIR` to a new existing parent directory first. Apply the same
compile dependency review and runtime receipt checks described below. This adds
no SDK installation or global configuration changes.

## Measurement method

`topk_k1_compare.cpp` preflights all six inputs before GPU allocation and checks both
providers against the CPU oracle before and after timing. For each shape it runs
five warmup batches per provider, then 20 alternating measurement pairs, with 100
provider calls per batch. HIP event batch duration is divided by 100. The table
uses each provider's median and divides baseline median by candidate median.

The measured boundary includes provider stream work, the baseline's D2D index
copy, and possible host enqueue gaps. It excludes input upload, CPU validation,
allocation, and event creation. This is a provider-call improvement; it is not a
claim about kernel instructions alone. Loader diagnostics are disabled during
measurement. Clocks/power and other GPU clients are not experimentally controlled;
the workspace GPU lock serializes cooperating research jobs.

The K1 implementation uses 256 threads and 2 KiB shared memory per row. Each lane
reads up to four values, followed by a shared-memory reduction. It writes the final
index directly and needs no caller scratch. Its host tests cover ties and signed
zeros, but the measured comparison's supported inputs remain the narrower set
above.

## Iterate

From the repository root, edit `research/rocm10/topk_k1_variants.hpp` to continue
the second experiment while keeping `topk_k1.hpp` as the baseline. Use fresh
output paths for every build and run:

```bash
STRIX_TRIAL_DIR=/tmp/strix-k1-trial-001
mkdir -m 700 "$STRIX_TRIAL_DIR"
PYTHONPATH=src .venv/bin/python research/rocm10/run.py compile-k1-variants \
  --output "$STRIX_TRIAL_DIR/build" --diagnostic
sha256sum "$STRIX_TRIAL_DIR/build/work/topk-k1-variants-compare"
```

Review a changed dependency or compiler path against the recorded closure before
executing it. Use the displayed executable hash:

```bash
PYTHONPATH=src .venv/bin/python research/rocm10/run.py run-k1-variants \
  --binary "$STRIX_TRIAL_DIR/build/work/topk-k1-variants-compare" \
  --binary-sha256 REPLACE_WITH_EXECUTABLE_SHA256 \
  --output "$STRIX_TRIAL_DIR/run"
```

`run/phase.json` records process completion, inputs, commands, limits, before/after
SDK inventories, output hashes and the held GPU lock. `lock-release.json` records
release after receipt finalization. The final JSON line in `stdout.log` contains
`success`, `research_comparison_valid`, exact input identities and raw timings.
Process completion alone is not a correctness or comparison claim. Require the
fixture checks, reviewed loaded paths, stable baselines and successful lease
release before accepting a result. Keep failed and slower trials; repeat promising
results with the same measurement method.

The launcher binds this machine's exact private SDK checkpoint; it performs no
installation or global activation. Compilation exposes no GPU nodes. Runtime
exposes only the selected KFD/render nodes and reviewed read-only sysfs paths.
Every phase rechecks the retained SDK; output directories are exclusive and bounded.

The original `compile-k1` / `run-k1` actions reproduce the first bitonic-versus-K1
experiment. For the third iteration, edit `topk_k1_onewave.hpp`, use
`compile-k1-onewave` / `run-k1-onewave`, and substitute
`topk-k1-onewave-compare` for the executable name above. Keep the variants header
frozen because it supplies this experiment's baseline and shuffle helpers.

The next useful step is comparison with Halo's existing kernels and real call
shapes. Small-K specializations and a separately qualified rocPRIM comparison
remain candidates after that check. Change one factor at a time and preserve
the oracle.

## Checks

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  research/rocm10/test_launcher.py research/rocm10/test_preflight.py \
  research/rocm10/test_k1_variants.py research/rocm10/test_gdn_norm.py -q
.venv/bin/ruff check research/rocm10
.venv/bin/ruff format --check research/rocm10
MYPYPATH=src .venv/bin/mypy research/rocm10/run.py research/rocm10/preflight_exec.py
```

These host tests exercise launcher limits, failure cleanup, binary/header pinning
and real Bubblewrap preflight checks with a dummy SDK. They do not execute ROCm.
The preflight integration test requires the local UID 1000/Bubblewrap environment.

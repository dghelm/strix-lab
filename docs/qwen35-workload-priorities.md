# Qwen3.5-4B workload priorities

The [published MMVQ attribution and six-trial study](../field-reports/2026-09-05-qwen35-mmvq-shape-attribution.md)
already establishes the dense model's shape families, fusion paths, and repeated
vocabulary projection cost. This note adds recorded server-sampling evidence and
connects that study to the [Halo K1 overlap investigation](rocm-k1-upstream-overlap.md).
It introduces no new profiling run, shape-attribution result, or optimization gain.

## The actual smoke request does not use GPU sampling

The [registered Qwen3.5-4B model](../configs/models/qwen35-4b-smoke.yaml) is dense,
with gated delta net and full attention, without MoE or QSA. Its synthetic suite
checks include TOP_K and MUL_MAT_ID; that does not put those operations in the
dense model graph.

The recorded one-slot server response explicitly reports `backend_sampling=false`,
`temperature=0.0`, `top_k=40`, 68 evaluated prompt tokens and 64 predicted tokens.
This is observed response metadata, not an assumption based on defaults:

```text
~/.local/share/strixlab/runs/records/run-20260905T194254Z-smoke-qwen35-680a6498d54f5de7454f18515b7f97dc/adapters/llama-server/greedy-token-parity-short-sequence/requests/0001/response.body.bin
SHA256 d87b6097dec2cfce54c2943d1594c85bab7be4012c0ab07dbfcf00f63cb6a1a5
```

Explicit backend top-k 1 would be a different workload. Initial logits have
vocabulary width 248320, beyond the K1 fixture's maximum width of 1024; preceding
filters and masks also matter. Greedy backend sampling already calls argmax,
while this recorded request sampled on the CPU.

The separate `llama-bench` traces bypass sampling. Scanning recorded kernel names
for `argmax|top.?k|argsort|sort` found zero matches among 2485 pp512, 9169 pp2048,
and 112352 tg128 dispatches. Those CSVs are under
`~/.local/share/strixlab/research/baseline-profile-20260905-DdQvJ0/`, named
`<case>/<case>_kernel_trace.csv`. Absence of matching names is scoped to these
traces; it does not prove absence in every configuration or in fused operations.

## Keep the profiled source distinct from current Halo master

The published study profiles the existing ROCm 7.2.4 control build
`af360810acf2b021d92a3bad67e752c9cb9b2b0ddac8bff76cdedb3fe08af474` at
[`ca94157f70a2776e8da6b6849b50b45a083d0478`](https://github.com/halo-box/strix-llama.cpp/commit/ca94157f70a2776e8da6b6849b50b45a083d0478).
Use the exact clean checkout
`~/.local/share/strixlab/sources/worktrees/prep-strix-llama-082114a213b3d5ff927c7000`;
its `ggml/src/ggml-cuda/mmvq.cu` SHA-256 is
`6c8a629f890f64c8a514555326049df356e946d4091d3e22aa3f5f7403803b3b`.
A neighboring worktree can share HEAD while containing modified tuning bytes.
The overlap investigation's newer master `c7af5c6` and the private ROCm 10 K1
fixtures are separate; neither was the source of these inference timings.

The existing 441-tensor GGUF inventory and old source establish the tied head:
there is no `output.weight`; the loader
[falls back to `token_embd.weight`](https://github.com/halo-box/strix-llama.cpp/blob/ca94157f70a2776e8da6b6849b50b45a083d0478/src/models/qwen35.cpp#L44-L52),
and the [LM head multiplies `model.output`](https://github.com/halo-box/strix-llama.cpp/blob/ca94157f70a2776e8da6b6849b50b45a083d0478/src/models/qwen35.cpp#L219-L226).
Its Q6_K weight shape is `[2560,248320]`. The
[210-byte Q6_K block](https://github.com/halo-box/strix-llama.cpp/blob/ca94157f70a2776e8da6b6849b50b45a083d0478/ggml/src/ggml-common.h#L358-L368)
holds 256 weights, giving `2560 × 248320 / 256 × 210 = 521472000` bytes
(**497.31 MiB**) for this matrix alone. The previous 256 MiB private kernel
harness is not sized for a full-shape head test; the existing inference build
already handles it. This sizing fact is not a bandwidth-bottleneck diagnosis.

## Newer Halo already changes the Q6_K baseline

A source comparison with `c7af5c6` finds a fused activation-quantization path
absent from the profiled implementation. Its
[type gate](https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/ggml/src/ggml-cuda/mmvq.cu#L1498-L1514)
defaults to Q8_0/Q6_K and explicitly excludes Q4_K/Q5_K. The
[eligibility checks](https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/ggml/src/ggml-cuda/mmvq.cu#L2794-L2817)
require RDNA3.5, a single output column, aligned F32 activations/output, compatible
fusion, K divisible by 32, and at most 16384 bytes of shared Q8_1 storage.
K=2560 needs 2880 bytes and K=9216 needs 10368 bytes. The known Q6_K head, QKV/Q,
and down-projection shapes therefore fit the size gate; runtime layout and other
eligibility conditions have not been observed on this newer build.

The newer code tries this path before the separate activation-quantization
allocation/launch. Its
[block organization](https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/ggml/src/ggml-cuda/mmvq.cu#L1866-L1905)
uses one output row per wave and 16, 8, or 4 waves per block, with quantization
repeated per block. That materially changes the old one-row/one-wave-per-block
baseline. It is not evidence of a speedup or exact-output parity.

The dominant Q4_K gate/up path is outside this default fused-quantization path.
The newer grouped-projection path is Q8_0-only and does not cover the Q4_K/Q6_K
groups either. The compared Q4_K/Q6_K dot-product helpers are unchanged.
A useful next comparison is an unchanged current-Halo build against the older
control, followed by verification of actual dispatch and clean inference
correctness/performance. Do not invent a duplicate Q6_K fusion before evaluating
what the target repository already supplies.

## Next unresolved question

The six-trial study finds vocabulary projection medians around 2.3 ms per late
token in all four profiled repetitions. The original 10.646 ms FFN layer-15 stall
does not recur at that slot; isolated large intervals appear elsewhere in both
collection modes. Clean timings also vary. Repeating the original aggregate
analysis would neither explain those stalls nor establish profiler interference.

The next question is whether an exact-output change to the recurring matrix work
can improve clean inference, not whether the known shape attribution reproduces.
Start from the unchanged GGML graph path for the tied Q6_K head, Q4_K fused FFN
gate/up `[2560,9216]`, or Q6_K QKV/Q `[2560,8192]`, preserving activation conversion
and existing fusion. The head's much larger row count already explains why it
represents more work; a long interval alone does not identify avoidable cost.
No redundant matrix, removable padding, or safe full-vocabulary truncation has
been established. A candidate mechanism and an end-to-end comparison remain
outstanding; no new K1 benchmark addresses this workload.

## Remaining diagnostic options

The existing control binaries are `llama-bench`, `llama-server` and
`test-backend-ops`. `GGML_SCHED_DEBUG=2` with verbose logging enables existing
[scheduler output](https://github.com/halo-box/strix-llama.cpp/blob/ca94157f70a2776e8da6b6849b50b45a083d0478/ggml/src/ggml-backend.cpp#L988-L1024),
which gives operations, names and backend assignments, but rounded byte sizes
rather than full dimensions. It is planning evidence and can include reserve or
warmup graphs, not per-kernel execution attribution.

Existing [`common_debug_cb_eval`](https://github.com/halo-box/strix-llama.cpp/blob/ca94157f70a2776e8da6b6849b50b45a083d0478/common/debug.cpp#L143-L181)
prints operations and exact input/output shapes; the existing
[`llama-debug` example](https://github.com/halo-box/strix-llama.cpp/blob/ca94157f70a2776e8da6b6849b50b45a083d0478/examples/debug/debug.cpp#L218-L235)
installs it and accepts `--tensor-filter`. That executable is absent from the
recorded managed build, so using it would require a separate build artifact,
not new callback code. It requests every node and copies device tensors even
when a filter hides output: use it for shape/correctness diagnostics, not timing.
It also does not supply a direct annotation of each fused device dispatch or
resolve the sporadic-stall cause. Those remain limits of the published evidence.

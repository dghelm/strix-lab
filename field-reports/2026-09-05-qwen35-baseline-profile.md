# Field report: Qwen3.5-4B same-build smoke calibration and diagnostic baseline profile

- **Report kind:** self-service hardware smoke suite
- **StrixLab commit:** `ac86624e588cf151d7bcacac57fecdb12a94ee4e`
- **Date:** 2026-09-05
- **Reporter:** strix-research-report
- **Publication:** local artifact only (`/tmp/strix-baseline-profile-report.md`). No repository copy and no pull request from this reporter.

## Environment

- **Machine / OS / GPU summary:** Strix Halo 128G class, Linux, gfx1151 integrated (AMD Radeon 8060S), existing `/opt/rocm` ROCm 7.2.4 toolkit (hipcc reports HIP 7.2). Doctor machine id `strix-halo-128g`; `rocminfo` arch `gfx1151`; telemetry source `rocm-smi` (auto fallback warning). Optional `rocprof-compute` and `rocprof-sys` unavailable (warnings). Doctor status `ready` with exclusive lock `/tmp/strixlab-gpu.lock` acquired. No hostnames, serials, or user paths.

Pinned evaluator (no candidate patch):

- **Suite:** `configs/suites/smoke-qwen35.yaml` (`id: smoke-qwen35`)
- **Resolved scenario SHA-256:** `6957b0b5d0af9f72011c8610d31fa76384771e21aa589e3c9c7905c1878d984b`
- **Source:** `configs/sources/strix-llama.yaml` → `https://github.com/halo-box/strix-llama.cpp.git` @ `ca94157f70a2776e8da6b6849b50b45a083d0478`
- **Build profile:** `configs/builds/hip-rocm-gfx1151.yaml`
- **Build ID:** `build-sha256:af360810acf2b021d92a3bad67e752c9cb9b2b0ddac8bff76cdedb3fe08af474`
- **Build canonical record:** `record-sha256:5d716057d31984acc228db2406f96cedef8cb3ab9710b4a6cd10a75acaf039e4`
- **Model:** `configs/models/qwen35-4b-smoke.yaml` — actual artifact Qwen3.5-4B Q4_K_M (`Qwen_Qwen3.5-4B-Q4_K_M.gguf`, sha256 `13c16f426047e2de38cd075bdade4a7bcbc8c774384876f677740cda65f8a983`)
- **Model receipt SHA-256:** `e156f99a42a47394fd0517f9c938e1f0dca4e5457c7371d2cb48493589fee923`
- **Source snapshot (unpatched):** `snapshot-sha256:e8c80efdc309ec7149cb63add58345364b638541730e710c6cb82657ee8af1f9` / `candidate-sha256:dd1388648c5e53315fdcc3cd57ee3acd8ac45c56a32168e6deeb46c862a697f5` (`patches: []`)

## Commands run

This pass reused an already-prepared source and an already-attested build. It did **not** run `source prepare` or `build prepare`. Prerequisite identities, authenticated with the public `inspect_build` API before reuse:

- prepared source: `prep-strix-llama-082114a213b3d5ff927c7000`
- build: `build-sha256:af360810acf2b021d92a3bad67e752c9cb9b2b0ddac8bff76cdedb3fe08af474`

Work was serial: doctor readiness, exclusive machine lock `/tmp/strixlab-gpu.lock`, then model and build leases. Suites did not overlap. `model verify` leased the existing preparation; it was not overlapped with a build.

```bash
export MODELS=/absolute/path/to/model-root

clean_run() {
  env -i HOME="$HOME" PATH="$PATH" LANG=C.UTF-8 MODELS="$MODELS" "$@"
}

clean_run uv run strixlab doctor \
  --machine configs/machines/strix-halo-128g.yaml

# Existing build authenticated with strixlab.build_cache.inspect_build.
BUILD_ID='build-sha256:af360810acf2b021d92a3bad67e752c9cb9b2b0ddac8bff76cdedb3fe08af474'

clean_run uv run strixlab model verify configs/models/qwen35-4b-smoke.yaml \
  --source prep-strix-llama-082114a213b3d5ff927c7000
MODEL_RECEIPT_SHA256='e156f99a42a47394fd0517f9c938e1f0dca4e5457c7371d2cb48493589fee923'

clean_run uv run strixlab run suite configs/suites/smoke-qwen35.yaml \
  --machine configs/machines/strix-halo-128g.yaml \
  --build "$BUILD_ID" \
  --model-receipt "$MODEL_RECEIPT_SHA256"
BASELINE_RUN='run-20260905T193843Z-smoke-qwen35-b7b7017f43dc3b51d7f40c3cc7b9573b'

clean_run uv run strixlab run suite configs/suites/smoke-qwen35.yaml \
  --machine configs/machines/strix-halo-128g.yaml \
  --build "$BUILD_ID" \
  --model-receipt "$MODEL_RECEIPT_SHA256"
SECOND_RUN='run-20260905T194254Z-smoke-qwen35-680a6498d54f5de7454f18515b7f97dc'

clean_run uv run strixlab compare "$BASELINE_RUN" "$SECOND_RUN"
COMPARISON_RUN='run-20260905T194628Z-compare-cd2578cb76a4cd86f1000f38-a834f9b544447e3cc30df1291a321ee5'
```

`strixlab run inspect` was not invoked. This reporter authenticated the three finalized runs through the public APIs `inspect_run` and `load_finalized_suite_snapshot`, then read `comparison/report.json` via `read_record_member`. First baseline `bundle export` was attempted and failed the sensitive-value guard; `bundle verify` was not reached.

Diagnostic profiler collection (not part of the smoke-suite scoring path). Coordinator owned doctor before and after each case. Launcher acquired the machine lock and verified build and model leases pre/post; it did not run doctor. Model operand was `/proc/self/fd/6`. No new source patch.

```bash
clean_run uv run strixlab doctor \
  --machine configs/machines/strix-halo-128g.yaml
clean_run .venv/bin/python /tmp/strix-profile-launcher.py tg128
clean_run uv run strixlab doctor \
  --machine configs/machines/strix-halo-128g.yaml

clean_run .venv/bin/python /tmp/strix-profile-launcher.py pp512
clean_run uv run strixlab doctor \
  --machine configs/machines/strix-halo-128g.yaml

clean_run uv run strixlab doctor \
  --machine configs/machines/strix-halo-128g.yaml
clean_run .venv/bin/python /tmp/strix-profile-launcher.py pp2048
clean_run uv run strixlab doctor \
  --machine configs/machines/strix-halo-128g.yaml
```

Redacted child argv (same binary and leases for all three cases):

```text
/opt/rocm/bin/rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
  --memory-allocation-trace --stats --output-config \
  --output-format csv json pftrace --perfetto-backend inprocess \
  --output-directory <redacted-diagnostic-root>/<case> --output-file <case> -- \
  <redacted-build-root>/bin/llama-bench -m /proc/self/fd/6 \
  -p <0|512|2048> -n <128|0|0> -r 1 -o jsonl
```

## Outcome

- **Result:** inconclusive

Both correctness suites passed, and the same-build offline comparison is `inconclusive` on every case. That pairing satisfies the **initial configured calibration gate**. **Inconclusive does not establish equivalence or stability.** Baseline noise is broad (`pp512` 18.141253%, `pp2048` 6.833320%, `tg128` 9.603953%). Point deltas are not optimization gains.

Diagnostic profiler collection and Codex analysis are finished (no more GPU runs). `tg128` is a **valid** complete diagnostic. `pp512` and `pp2048` are **partial** (the same two invalid `hipModuleGetFunction` `kname` fields; CSV-only timing). No source patch and no campaign were selected. Profiled `-r 1` samples must not be pooled with the clean suite comparison.

### Hardware smoke-suite results

- **Build ID:** `build-sha256:af360810acf2b021d92a3bad67e752c9cb9b2b0ddac8bff76cdedb3fe08af474`
- **Model receipt SHA-256:** `e156f99a42a47394fd0517f9c938e1f0dca4e5457c7371d2cb48493589fee923`
- **Run ID:** `run-20260905T193843Z-smoke-qwen35-b7b7017f43dc3b51d7f40c3cc7b9573b`
- **Second run ID:** `run-20260905T194254Z-smoke-qwen35-680a6498d54f5de7454f18515b7f97dc`
- **Comparison run ID:** `run-20260905T194628Z-compare-cd2578cb76a4cd86f1000f38-a834f9b544447e3cc30df1291a321ee5`
- **Suite result status:** both arms `passed` (`reason: passed`); backend-ops passed; greedy passed
- **Verified bundle summary:**
  - `outcome`: N/A
  - `run_record_sha256`: N/A
  - `member_count`: N/A

No verified bundle exists. The first baseline `bundle export` failed the sensitive-value guard. Authenticated local records are available through `inspect_run` / `load_finalized_suite_snapshot` and the comparison report members; this is not independent public reproducibility.

Authenticated local identities (`inspect_run`, `load_finalized_suite_snapshot`, `comparison/report.json`):

| Arm | Run ID | Record | Suite-result blob |
|---|---|---|---|
| baseline | `run-20260905T193843Z-smoke-qwen35-b7b7017f43dc3b51d7f40c3cc7b9573b` | `record-sha256:1dd5733c51bd81cd31d58cf5b5339aedfeb29b1549c65ffb16e8da5b5a7fd9a7` | `396256efd818d375f0b7c0551c9de0c4e23ff162bd6065d37483a2ac87b8b6b5` |
| second | `run-20260905T194254Z-smoke-qwen35-680a6498d54f5de7454f18515b7f97dc` | `record-sha256:a6cd039d7cd07327a446d4f7c75d157036ce34c1e42315dd2cf031283a888d55` | `2d47b07322bcf1eb2d4dc7dcbeeeefed34270ef49fa80aa2013385804b341d21` |
| comparison | `run-20260905T194628Z-compare-cd2578cb76a4cd86f1000f38-a834f9b544447e3cc30df1291a321ee5` | `record-sha256:94ff7a1bc7a0cad1d92d05e151185bde47a2f305e22e3df9228b6517afcdb6b0` | N/A (derived comparison members only) |

Both arms used the same build ID and the same build canonical record. Comparison inspect checksums sha256 `6401e3787ca00d72565a50369a6f83e01201fb0d0ffa34972f3672f0198a79d0`. Comparison report blob sha256 `beeca793f9f541a810249fa2d73c1578eb962ece0d85e0c53d56da2d82c73459`. Policy `paired-log-bootstrap-v1`.

### Calibration comparison (authenticated local report)

Offline, matched-sample, conservative comparison of two same-build suite runs (baseline → second). Higher `samples_ts` is better for the judge, but these deltas are **not** optimization gains: there is no candidate patch, and every case is `inconclusive`. Noise is broad. The table does **not** establish equivalence or stability. No universal score. Not independently verifiable without both source-run bundles.

- **Configured gate:** both arms passed correctness, and every performance case is `inconclusive`. Gate satisfied.
- **Explicitly not shown:** run-to-run equivalence or stability. Broad noise means the second arm can move several percent without leaving `inconclusive`. Positive deltas are not gains.
- **Overall verdict:** inconclusive
- **Schedule:** `windowed-interleaved-v1`; 6/6 warmups and 15/15 measurements completed on both arms; 15 paired samples per case

| case | metric | baseline mean | second mean | delta % | percent CI | noise % | pairs | verdict |
|---|---|---|---|---|---|---|---|---|
| pp512 | samples_ts | 1593.725333 | 1655.892667 | 3.327238 | [-7.753166, 15.457907] | 18.141253 | 15 | inconclusive |
| pp2048 | samples_ts | 1665.666000 | 1696.867600 | 1.155846 | [-12.444079, 14.825985] | 6.833320 | 15 | inconclusive |
| tg128 | samples_ts | 52.183733 | 53.647753 | 2.820077 | [0.881961, 4.778377] | 9.603953 | 15 | inconclusive |

Means are baseline → second: `pp512` 1593.725333 → 1655.892667; `pp2048` 1665.666000 → 1696.867600; `tg128` 52.183733 → 53.647753. `tg128` percent CI is above zero but remains `inconclusive` because the log-delta magnitude is at or below baseline noise (`paired-log-bootstrap-v1`). `pp512` noise is 18.141253%; `pp2048` 6.833320%; `tg128` 9.603953%.

### Cross-run greedy tokens

Prompt `short-sequence` (`greedy-token-parity`, seed 1234, 64 output tokens): both arms passed, intra-run `tokens_equal`, and exact cross-run identity.

- token_count: 64 (both responses, both runs)
- tokens_sha256: `8c8b081313ea49a1ddbfe87a2f582897507822bbc5439cca7f86d196ff3bd50a`

Exact cross-run greedy tokens are part of the configured calibration gate. They do not establish performance equivalence, stability, or a gain.

## Failure excerpt or friction notes

- First baseline `bundle export` failed the sensitive-value guard. No verified bundle. Bundle fields remain `N/A`. Authenticated local inspect/comparison records exist; do not treat them as a portable public bundle.
- Doctor remained `ready` with three warnings: optional `rocprof-compute` unavailable, optional `rocprof-sys` unavailable, telemetry fell back to `rocm-smi`.
- Diagnostic rocprofv3 JSON for both PP cases is not a fully valid UTF-8 trace bundle. The same two fields are invalid in each file: `buffer_records.hip_api[39861].args[2].value` and `hip_api[39863].args[2].value` (`kname` of `hipModuleGetFunction`, correlations 39862 and 39864). First invalid bytes at offsets 37260469 (PP512) and 40189203 (PP2048). Raw preserved; not repaired; cause unproven; no rerun. CSV omits the corrupted strings and supports limited timing only.
- Same-build comparison is overall `inconclusive` on all three cases. Together with both suites passing correctness, that satisfies the configured calibration gate. It is not a failed suite, and it is not proof of equivalence or stability. Noise is broad; do not read the positive point deltas as gains.

## Reproduction notes

This pass depended on a pre-existing prepared source and attested build. Another operator would need those same identities, or would have to prepare them first; this report does not record a prepare on 2026-09-05. With the pinned GGUF, this StrixLab commit, the existing `/opt/rocm` 7.2.4 toolkit, and an idle gfx1151 Strix Halo 128G machine, the executed path is cleanenv doctor → `build inspect` of the existing build → `model verify` against `prep-strix-llama-082114a213b3d5ff927c7000` → two serial `run suite` → `compare`. Suites must not overlap each other or other GPU work. The machine lock is `/tmp/strixlab-gpu.lock`.

Independent public reproduction of these exact runs is **not** claimed: bundle export did not succeed, so no verified bundle exists. Authenticated local records are available on the originating machine only.

Profiler traces are diagnostic-only. They are not confirmation runs, not campaign screening, and not an optimization result. Any later confirmation still needs clean equivalent arms without the profiler wrapper.

## Profile analysis

Codex `strix-profile` findings, with interval methods and claims reviewed by the parent coordinator. This reporter did not independently parse the raw traces. No optimization candidate or campaign is justified. No source patch.

**Collection:** `tg128`, `pp512`, `pp2048` serially under the suite GPU lock; child exit 0; build and model leases verified pre/post; doctor `ready` before and after. Adapter argv used `-r 1`. Materialized cache records HIP graphs ON; that does not itself prove replay. JSONL reports batch 2048, microbatch 512, 16 threads, f16 K/V, GPU-layer setting -1, FA auto, ROCm backend. Requested defaults do not independently prove exact offload. The greedy server's context 4096 / GPU layers 999 are not benchmark settings.

**Integrity:**

| Case | Status | Kernel | HIP | Copy | Alloc |
|---|---|---:|---:|---:|---:|
| `tg128` | **valid** — strict UTF-8 JSON parses | 112352 | 58208 | 3007 | 52 |
| `pp512` | **partial** — CSV timing only; malformed raw JSON | 2485 | 51296 | 2880 | 52 |
| `pp2048` | **partial** — CSV timing only; malformed raw JSON | 9169 | 84511 | 2880 | 46 |

Both PP files have invalid UTF-8 in exactly the same two fields: `hip_api[39861].args[2].value` and `hip_api[39863].args[2].value` (`kname` of `hipModuleGetFunction`). CSV omits those strings and parses strictly. Numeric counts and timestamp pairs agree with domain stats and JSON numeric records; HIP records belong to the launched target PID; no loss/overflow warnings in stderr. That supports limited timing inspection. It does not make the JSON valid, prove the string-lifetime/serialization defect is harmless, or prove every unobserved event was captured. Original bytes were not repaired. No rerun.

**Interval method (parent-reviewed):** TG measured proxy starts at warmup terminal-sync return and ends at token 128 terminal-sync return (129 output-copy/terminal-sync motifs = 1 warmup + 128 tokens). Late interval is tokens 33–128, fixed before looking at timings. Each PP case has two output-copy/terminal-sync boundaries (warmup + measured). No exact benchmark timer marker is present; these are proxy intervals. Kernel-active time is the union of clipped device kernel intervals across queues. Device-idle means neither kernel nor device-copy appears in this trace; it does not imply GPU hardware idleness beyond the traced process. Family shares are descriptive work attribution, not removable critical-path fractions. Profiled single-sample rates must not be pooled with clean suite evidence.

| Interval | Proxy ms | Kernel union ms | Kernel-active | Device-idle ms |
|---|---:|---:|---:|---:|
| TG128 measured proxy | 3037.678 | 2424.923 | 79.828% | 610.535 |
| TG tokens 33–128 (late 96) | 1907.560 | 1636.921 | 85.812% | 268.959 |
| PP512 measured proxy (partial) | 306.929 | 255.959 | 83.394% | 50.955 |
| PP2048 measured proxy (partial) | 1229.926 | 1204.730 | 97.951% | 25.180 |

**TG graph behavior:** Measured TG has 127 graph launches, 2 captures, 2 instantiations, and 2 executable updates. Capture/instantiate for token 1 sits inside the measured interval (`hipGraphInstantiate` 213.232 ms). Token 2 launches directly; token 3 recaptures with update then reinstantiate (early setup, not repeated steady rebuild). Late 96 tokens: 96 graph launches, 96 output-copy APIs, 96 terminal syncs, **zero** capture, instantiate, update, or direct-launch APIs. Each late graph launch correlates with 869 kernel dispatches. Late `hipGraphLaunch` API union is 97.546 ms, but only 19.077 ms intersects device-idle (~1.00% of the late window). Of 268.959 ms late device-idle, 239.577 ms intersects `hipStreamSynchronize` after graph submission. Largest gaps sit between kernels of an already submitted graph while the host waits. This fails the evidence prerequisite for a late host graph-rebuild patch. Treating all API duration as removable launch overhead would overstate the evidence.

**Kernel families** (summed family device duration / stated proxy interval):

| Case/interval | Family | Duration ms | Interval share |
|---|---|---:|---:|
| TG late96 | Q4_K MMVQ, grouped variants | 704.827 | 36.949% |
| TG late96 | Q6_K MMVQ, grouped variants | 645.105 | 33.818% |
| TG late96 | Q8_0 MMVQ | 64.571 | 3.385% |
| TG late96 | Q5_K MMVQ | 37.961 | 1.990% |
| PP512 partial | Q4_K MMQ, tile parameter 128 | 104.541 | 34.060% |
| PP512 partial | Library GEMM, Cijk symbols | 44.521 | 14.505% |
| PP512 partial | Gated delta net | 28.101 | 9.156% |
| PP2048 partial | Q4_K MMQ, tile parameter 128 | 512.432 | 41.664% |
| PP2048 partial | Library GEMM, Cijk symbols | 234.605 | 19.075% |
| PP2048 partial | Gated delta net | 113.414 | 9.221% |

Q4_K_M is not “all Q4_K kernels.” Logical tensor shapes and layer identities are not established from template names or grid dimensions. PP512's measured pass includes one capture/instantiate/update/launch (instantiate 35.827 ms); that is not repeated PP replay overhead. PP2048 has no graph/capture APIs anywhere and is 97.951% kernel-active; graph absence does not justify “enable graphs.”

**Strongest next attribution question:** which actual tensor/layer shapes within Q4_K and Q6_K MMVQ account for the late TG device time, and are their large dispatch intervals repeatable? Dominant observed variants include Q4_K fused (`mul_mat_vec_q<(ggml_type)12, 1, true, false, false>`, 560.129 ms) and Q6_K nonfused (`…14, 1, false, false, false>`, 480.437 ms). Attribute these to real shapes and bindings before judging dispatch efficiency. A launch grid is not a logical tensor shape. Time alone does not distinguish bandwidth, occupancy, arithmetic, scheduling, or profiling effects.

Separately, the PP `hipModuleGetFunction.kname` serialization defect is a profiler-evidence issue, not an optimization candidate. Future work should be bounded and reviewed. Do not rerun the rejected historical MMVQ warp sweep, change math, introduce a graph patch, or open a campaign on the present evidence. No safe redundant operation or alternative dispatch mechanism has been demonstrated.

## Privacy confirmation

- [x] This report contains no model weights, credentials, private data, raw
      StrixLab home, or evidence bundle, and identifying values are redacted.

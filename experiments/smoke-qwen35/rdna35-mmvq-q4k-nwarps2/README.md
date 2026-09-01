# Experiment: RDNA 3.5 Q4_K MMVQ uses two warps

- **Experiment ID:** rdna35-mmvq-q4k-nwarps2
- **Scenario:** `configs/suites/smoke-qwen35.yaml`
- **Resolved scenario SHA-256:** `6957b0b5d0af9f72011c8610d31fa76384771e21aa589e3c9c7905c1878d984b`
- **Status:** observed
- **Candidate author:** mmvq-followup-codex (agent-assisted local pilot)

## Question

On gfx1151, Q4_K single-column MMVQ decode currently inherits the RDNA2
fallback and uses `nwarps=1`. The first experiment routed RDNA 3.5 to the full
RDNA4 table, moving Q4_K and other quant types to as many as eight warps. It
preserved correctness but regressed `tg128` by 8.44% (95% interval -9.91% to
-6.98%), while both prompt-processing cases were inconclusive.

The pinned table uses `nwarps=2` for K-quants on Turing and for tuned Q6_K on
RDNA 3.0, so two warps is an existing conservative launch choice rather than a
new parameter sweep. Does changing only Q4_K at `ncols_dst=1` from one to two
warps improve `smoke-qwen35` token generation without the overhead observed at
eight warps or a correctness failure?

This candidate adds an explicit RDNA 3.5 table so every non-Q4_K type and every
multi-column path retains the baseline one-warp behavior. It changes launch
configuration only; vec_dot math, rows per block, the scenario, model, source
commit, build profile, and comparison policy are unchanged.

## Candidate

- **Source repository:** https://github.com/halo-box/strix-llama.cpp.git
- **Base commit:** `ca94157f70a2776e8da6b6849b50b45a083d0478`
- **Candidate commit or patch:** `experiments/smoke-qwen35/rdna35-mmvq-q4k-nwarps2/patches/mmvq-rdna35-q4k-nwarps2.patch`
- **Patch SHA-256:** `e1f9eefbf502fa6e87032602fdf9564fe746d53ae287a9efbdfe2c6bd9e3e7a0`
- **Build profile:** `configs/builds/hip-rocm-gfx1151.yaml`

Apply the patch only through `strixlab source prepare --patch` on the pinned
source lock. Do not edit an existing prepared source tree in place.

## Replications

### Replication 1 — 2026-09-01, authorized local replication

- **Machine summary:** Strix Halo 128G class, Linux, gfx1151 integrated (Radeon 8060S), HIP 7.2 from `/opt/rocm`
- **StrixLab commit:** `d7bb76a751f3204743879c2edb962fbf7c787218`
- **Baseline build:** `build-sha256:af360810acf2b021d92a3bad67e752c9cb9b2b0ddac8bff76cdedb3fe08af474` (canonical `record-sha256:5d716057d31984acc228db2406f96cedef8cb3ab9710b4a6cd10a75acaf039e4`)
- **Candidate build:** `build-sha256:88da0b4690ffc31f101ee1104c5ffc6e17498e87a54e9226586fd5fc7caf3c05` (canonical `record-sha256:df5f8cc7ce33301d81d772abec4c29aec68a17189247bf2264ae9021d183fb58`)
- **Model receipt registry key:** `e07e48ea62bcd1b625776438b27329acb877f898ca490f28f749524f6a3047ff`
- **Baseline run:** `run-20260901T201331Z-smoke-qwen35-6abbcb606a0f8b4398aca6fce87ec069` (`record-sha256:4c699750a656a021d8895dac306979e112047358198abe5d549b7dc8a8973dfb`)
- **Candidate run:** `run-20260901T201539Z-smoke-qwen35-fa1a4e751e50eaa856179b6ae2cb87c7` (`record-sha256:c12ce01c34f976774de381b7d6abecf4bf2402650a5a21a77ec1332b177691ec`)
- **Comparison run:** `run-20260901T201732Z-compare-07cf25b364bbd50d0f5feafb-2f79320a654e43e1565b72e2f3d78ae3` (`record-sha256:94912b3149f7f7525808d783c4cfc1932209ce695ee26e5c39db5d43e7addd5a`)
- **Comparison verdict:** inconclusive
- **Correctness:** passed
- **Notes:** Both arms passed correctness. Each case contains 15 matched measurement pairs. `pp512` was inconclusive at +5.256398% (95% interval -6.592878% to +22.932365%; baseline noise 5.454745%). `pp2048` was inconclusive at +2.020680% (+0.474227% to +3.750155%; noise 2.352852%). `tg128` was inconclusive at -0.689816% (-2.096573% to +0.900332%; noise 0.422824%).

## Finding

Under this exact `smoke-qwen35` scenario, Q4_K `nwarps=2` does not reproduce
the earlier clear `tg128` regression from full RDNA4 routing, but it establishes
no improvement: all three cases and the overall comparison are inconclusive.
This candidate is not suitable for best-known or upstream status without
further independent replication.

## Reproduction

```bash
export MODELS=/absolute/path/to/model-root

clean_run() {
  env -i HOME="$HOME" PATH="$PATH" LANG=C.UTF-8 MODELS="$MODELS" "$@"
}

clean_run uv run strixlab doctor \
  --machine configs/machines/strix-halo-128g.yaml

clean_run uv run strixlab source prepare configs/sources/strix-llama.yaml
BASELINE_PREP='prep-strix-llama-...'

clean_run uv run strixlab build prepare "$BASELINE_PREP" \
  configs/builds/hip-rocm-gfx1151.yaml
BASELINE_BUILD='build-sha256:...'

MODEL_RECEIPT_SHA256="$(
  clean_run uv run strixlab model verify configs/models/qwen35-4b-smoke.yaml \
    --source "$BASELINE_PREP"
)"

clean_run uv run strixlab source prepare configs/sources/strix-llama.yaml \
  --patch experiments/smoke-qwen35/rdna35-mmvq-q4k-nwarps2/patches/mmvq-rdna35-q4k-nwarps2.patch
CANDIDATE_PREP='prep-strix-llama-...'

clean_run uv run strixlab build prepare "$CANDIDATE_PREP" \
  configs/builds/hip-rocm-gfx1151.yaml
CANDIDATE_BUILD='build-sha256:...'

# Before running, ensure the GPU is idle. Run the baseline and candidate suites
# serially; do not overlap them with each other or with any other GPU workload.
clean_run uv run strixlab run suite configs/suites/smoke-qwen35.yaml \
  --machine configs/machines/strix-halo-128g.yaml \
  --build "$BASELINE_BUILD" \
  --model-receipt "$MODEL_RECEIPT_SHA256"
BASELINE_RUN='run-...'

clean_run uv run strixlab run suite configs/suites/smoke-qwen35.yaml \
  --machine configs/machines/strix-halo-128g.yaml \
  --build "$CANDIDATE_BUILD" \
  --model-receipt "$MODEL_RECEIPT_SHA256"
CANDIDATE_RUN='run-...'

clean_run uv run strixlab run inspect "$BASELINE_RUN"
clean_run uv run strixlab run inspect "$CANDIDATE_RUN"
clean_run uv run strixlab compare "$BASELINE_RUN" "$CANDIDATE_RUN"
COMPARISON_RUN='run-...'
clean_run uv run strixlab run inspect "$COMPARISON_RUN"
```

`source prepare --patch` is the only source-mutation mechanism for this
candidate. `model verify` and `build prepare` both lease a prepared source, so
they must not overlap.

## Privacy confirmation

- [x] This record contains no model weights, credentials, private data, raw
      StrixLab home, evidence bundle, hostname, device serial, or identifying
      absolute path.

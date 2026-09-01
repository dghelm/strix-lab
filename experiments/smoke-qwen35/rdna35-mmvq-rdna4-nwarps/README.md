# Experiment: RDNA 3.5 MMVQ uses the RDNA4 nwarps table

- **Experiment ID:** rdna35-mmvq-rdna4-nwarps
- **Scenario:** `configs/suites/smoke-qwen35.yaml`
- **Resolved scenario SHA-256:** `6957b0b5d0af9f72011c8610d31fa76384771e21aa589e3c9c7905c1878d984b`
- **Status:** observed
- **Candidate author:** pilot003-grok (agent-assisted local pilot)

## Question

On gfx1151, quantized decode currently selects the RDNA2 MMVQ parameter table
(`nwarps` mostly 1). RDNA4 uses `nwarps=8` for Q4_K / Q6_K at `ncols_dst=1`.
Does routing RDNA 3.5 to that RDNA4 table change `smoke-qwen35` throughput
without failing the correctness gates?

This is a launch-configuration candidate only. It does not change vec_dot math.
RDNA 3.0 keeps a stricter whitelist (Q4_K stays at `nwarps=1`); this experiment
tests the more aggressive RDNA4 choice on Strix Halo. The result may be an
improvement, a regression, mixed, or inconclusive.

## Candidate

- **Source repository:** https://github.com/halo-box/strix-llama.cpp.git
- **Base commit:** `ca94157f70a2776e8da6b6849b50b45a083d0478`
- **Candidate commit or patch:** `experiments/smoke-qwen35/rdna35-mmvq-rdna4-nwarps/patches/mmvq-rdna35-use-rdna4-table.patch`
- **Patch SHA-256:** `ec38719387331729ece4033b055faa139fa1bcb90214e7fea68ee027c5b26d72`
- **Build profile:** `configs/builds/hip-rocm-gfx1151.yaml`

Apply the patch only through `strixlab source prepare --patch` on the pinned
source lock. Do not edit an existing worktree in place.

## Replications

### Replication 1 — 2026-09-01, pilot003-grok

- **Machine summary:** Strix Halo 128G class, Linux, gfx1151 integrated (Radeon 8060S), HIP 7.2 from `/opt/rocm`
- **StrixLab commit:** `b86c413fc8bae61f81e3cea791ad9bb33e299d17`
- **Baseline build:** `build-sha256:af360810acf2b021d92a3bad67e752c9cb9b2b0ddac8bff76cdedb3fe08af474` (canonical `record-sha256:5d716057d31984acc228db2406f96cedef8cb3ab9710b4a6cd10a75acaf039e4`)
- **Candidate build:** `build-sha256:e193f0a3d149293fdfe723fa24722e199b2c383e60392c50f174f2b8f6bc8d74` (canonical `record-sha256:60d2c2658a972c11e04ecd99b908fc0ada8211003bb55901138062da6ce6a89c`)
- **Baseline run record:** `record-sha256:1fd1fa3746cc2de2c47112bb3bdbb9bfe1d903681a4bccb97360a163ebaf7f15`
- **Candidate run record:** `record-sha256:9aef587c5ad267d1fe07e4de92bbae8896b20f9374a6c7b7e7a0b59e657b98f5`
- **Comparison run record:** `record-sha256:153e4abb37dc88fb04e1f26fec8aa2a58c3cb9f0a6b0d263468b30a6622120c5`
- **Comparison verdict:** mixed
- **Correctness:** passed
- **Notes:** Both arms passed backend-ops and greedy correctness and completed 6 warmups plus 15 measurements. `pp512` was inconclusive (candidate mean -5.71%; 95% interval -12.39% to +0.95%), `pp2048` was inconclusive (-1.56%; -3.93% to +1.00%), and `tg128` regressed (-8.44%; -9.91% to -6.98%). The overall verdict is `mixed` because the case verdicts combine inconclusive and regression.

## Finding

Under this exact scenario, routing RDNA 3.5 MMVQ to the RDNA4 parameter table is
not an improvement. Prompt-processing cases are inconclusive, while the
measured token-generation result is a regression. The candidate is therefore
not suitable for upstreaming on the current evidence. A later
independent replication may refine the uncertainty, but must not pool cases or
hide the `tg128` regression behind the overall `mixed` label.

## Pilot friction

The first candidate attempt exposed a harness incompatibility before token
correctness: patched source trees advertise the pinned commit with a `-dirty`
suffix. [StrixLab PR #9](https://github.com/dghelm/strix-lab/pull/9) taught the
pinned capability grammar to accept only that exact suffix while continuing to
bind candidate identity through authenticated source and build evidence. The
replication above reran both arms from the merged fix commit and supersedes the
blocked attempt.

The CLI also remains silent during multi-minute builds and suite runs. A normal
shell may contain credentials or short values under sensitive-looking names, so
the working reproduction below uses an explicit clean environment for every
StrixLab command.

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

clean_run uv run strixlab model verify configs/models/qwen35-4b-smoke.yaml \
  --source "$BASELINE_PREP"
MODEL_RECEIPT_SHA256='...'

clean_run uv run strixlab run suite configs/suites/smoke-qwen35.yaml \
  --machine configs/machines/strix-halo-128g.yaml \
  --build "$BASELINE_BUILD" \
  --model-receipt "$MODEL_RECEIPT_SHA256"
BASELINE_RUN='run-...'

clean_run uv run strixlab source prepare configs/sources/strix-llama.yaml \
  --patch experiments/smoke-qwen35/rdna35-mmvq-rdna4-nwarps/patches/mmvq-rdna35-use-rdna4-table.patch
CANDIDATE_PREP='prep-strix-llama-...'

clean_run uv run strixlab build prepare "$CANDIDATE_PREP" \
  configs/builds/hip-rocm-gfx1151.yaml
CANDIDATE_BUILD='build-sha256:...'

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

`source prepare --patch` is the CLI mechanism for a reviewed candidate patch; it
is not shown in the README smoke-suite copy sequence. `model verify` and
`build prepare` both need the source lock, so they cannot overlap.

## Privacy confirmation

- [x] This record contains no model weights, credentials, private data, raw
      StrixLab home, evidence bundle, hostname, device serial, or identifying
      absolute path.

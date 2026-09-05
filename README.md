# StrixLab

Evidence-first, reproducible optimization research tooling for AMD Strix Halo.

StrixLab is a local, CLI-first research harness for running reproducible
optimization experiments on AMD Strix Halo hardware. It orchestrates existing
runtimes and isolated native builds, records immutable run evidence, and exports
deterministic, verifiable bundles. Bounded profile-guided campaigns sit above
that evidence scaffold; they never push upstream. StrixLab is **not** an
inference runtime, a benchmark leaderboard, or an automatic upstream patch bot.

Why it exists: optimization claims are only worth as much as the evidence behind
them. StrixLab exists to make experiments reproducible, to keep correctness gates
ahead of performance scoring, and to preserve failed and inconclusive results as
first-class evidence rather than discarding them.

## Status: pre-alpha

This is early, local research tooling. Interfaces and manifests can still change.

### What works today

A clean checkout provides a strict, evidence-oriented CLI:

- **Machine doctor** — read-only machine-readiness observation.
- **Schema / manifest validation** — versioned, packaged JSON Schemas.
- **Isolated source preparation** — StrixLab-owned disposable Git worktrees.
- **Reproducible builds** — pinned, inspectable native build trees.
- **Self-service pilot inputs** — checked-in source/build profiles and local
  model verification produce the IDs required by the smoke suite.
- **Deterministic smoke suite** — correctness-first suite execution that
  finalizes an immutable run (`run suite`).
- **Run inspection** — verify a finalized run's index, record, and checksums
  (`run inspect`).
- **Deterministic bundles** — export and verify portable run-evidence bundles
  (`bundle export`, `bundle verify`).
- **Offline comparison judge** — compare two finalized, equivalent, successful
  runs into one immutable comparison run with a conservative, non-scoring
  verdict (`compare`).

### Bounded profile-guided campaigns

A campaign is a local, finite investigation of reviewed source patches against
one strict frozen evaluator (suite, machine, source, build, model, and
comparison policy). The controller freezes a plan, screens candidates, and
requires a fresh independent confirmation before retain or reject. Calibration
is two distinct baseline/baseline runs. Interrupted attempts stay spent. The
suite-run budget counts calibration, confirmation, failures, and interruptions
and is not a wall-clock deadline. Failed and inconclusive findings stay in the
record. Campaigns do not replace the evidence CLI, do not open upstream pull
requests, and do not require ROCm 10.

See [`docs/autoresearch.md`](docs/autoresearch.md) for the procedure, frozen
evaluator, spent-interrupt and count-budget rules, and the controller command
surface. Executable README command examples wait on the checked-in campaign
CLI. See [`docs/research-problems.md`](docs/research-problems.md)
("Profile-guided llama.cpp research problems") for the ranked hypothesis
portfolio. Hardware campaign execution is later.

### What does not exist yet

- No web service, official judge, leaderboard, score, or "winning solution".
- No model downloader; you must obtain the pinned GGUF yourself and keep it
  outside the repository.
- No automatic upstream patch bot; StrixLab never pushes runtime changes or
  opens upstream pull requests. A retain decision is local evidence, not a PR.
- No fabricated campaign measurements; GPU execution of campaigns is later.

Community experiments (see below) are locally executed, reviewable
investigations—not remotely executed submissions or official scores.

### Planned scenarios

- [ROCm 10 top-k on gfx1151](docs/rocm10-topk-gfx1151.md) is a reviewed scenario
  contract and staged implementation plan. The generic capsule runner now exists, but
  no checked-in capsule manifest or TOPK payload interpretation makes it runnable yet.

## Requirements

- Linux
- [`uv`](https://docs.astral.sh/uv/) 0.9.12 or newer
- Python 3.12 or newer (installed automatically by `uv` when needed)

## Five-minute quickstart (no special hardware)

You do not need a Strix Halo machine to explore the repository and run its
developer test suite:

```bash
uv sync --locked --all-groups
uv run strixlab --help
uv run pytest
```

That path validates the checkout end to end without touching a GPU.

### Optional: Strix Halo machine check

On real Strix Halo hardware you can run the read-only machine doctor:

```bash
uv run strixlab doctor --machine configs/machines/strix-halo-128g.yaml
```

The doctor never changes clocks, power settings, drivers, or packages. It is
non-mutating to machine configuration, but it **does** write a versioned
`doctor.json` report beneath the resolved StrixLab home. See
[`docs/doctor.md`](docs/doctor.md) for the report and exit contracts.

## Self-service Strix Halo smoke suite

The pilot path is now self-service for a Strix Halo owner with ROCm installed at
`/opt/rocm`. It builds the pinned runtime locally and never downloads or copies
model weights. Obtain the pinned `Qwen_Qwen3.5-4B-Q4_K_M.gguf` yourself, verify
that it matches the size and SHA-256 in
[`configs/models/qwen35-4b-smoke.yaml`](configs/models/qwen35-4b-smoke.yaml), and
place it at:

```text
$MODELS/qwen35-4b/Qwen_Qwen3.5-4B-Q4_K_M.gguf
```

Then run the following commands from the repository root. Source and build
preparation print their IDs on the first line, model verification prints only
the receipt digest, and the suite prints its ID after `run:`. Copy only the ID
or digest—not a label—into the corresponding shell variable.

```bash
export MODELS=/absolute/path/to/your/model-root

uv run strixlab doctor --machine configs/machines/strix-halo-128g.yaml

uv run strixlab source prepare configs/sources/strix-llama.yaml
PREPARATION_ID='prep-strix-llama-...'

uv run strixlab build prepare "$PREPARATION_ID" \
  configs/builds/hip-rocm-gfx1151.yaml
BUILD_ID='build-sha256:...'

uv run strixlab model verify configs/models/qwen35-4b-smoke.yaml \
  --source "$PREPARATION_ID"
MODEL_RECEIPT_SHA256='...'

uv run strixlab run suite configs/suites/smoke-qwen35.yaml \
  --machine configs/machines/strix-halo-128g.yaml \
  --build "$BUILD_ID" \
  --model-receipt "$MODEL_RECEIPT_SHA256"
RUN_ID='run-...'

uv run strixlab run inspect "$RUN_ID"
uv run strixlab bundle export "$RUN_ID" "../strixlab-bundle-$RUN_ID"
uv run strixlab bundle verify "../strixlab-bundle-$RUN_ID"
```

Optionally, run the suite a second time to produce an equivalent run and compare
the two offline. The comparison is conservative and non-scoring; its no-op result
is `inconclusive`, never a fabricated win. A comparison bundle is a derived report
that is **not** independently verifiable without both source-run bundles, so export
all three when offline verification is required.

```bash
# CANDIDATE_RUN_ID is a second `run suite` of the same suite/machine/model.
uv run strixlab compare "$RUN_ID" "$CANDIDATE_RUN_ID"
COMPARISON_RUN_ID='run-...'

uv run strixlab run inspect "$COMPARISON_RUN_ID"
uv run strixlab bundle export "$COMPARISON_RUN_ID" "../strixlab-bundle-$COMPARISON_RUN_ID"
uv run strixlab bundle export "$CANDIDATE_RUN_ID" "../strixlab-bundle-$CANDIDATE_RUN_ID"
```

### ROCm 10 side-by-side bring-up

The installed `/opt/rocm` lane remains the ROCm 7.2.4 control. A separate
`configs/builds/hip-rocm10-gfx1151.yaml` profile describes the expected
`/opt/rocm-10` ROCm Core SDK 10.0.0 lane without changing global paths or system
alternatives. This lane is provisional, inactive, and not runnable until both its
artifact-authenticity gate and its separately reviewed prefix-inventory verifier
gate are cleared. The profile is not an installer and its environment is not
proof of runtime isolation; build evidence also cannot authenticate untracked
external ROCm library bytes.

Read [`docs/rocm10-bringup.md`](docs/rocm10-bringup.md) before provisioning or
using that prefix. It records the current blockers, every later human-approved
system mutation, and the required same-source no-op build comparison.

The quoted `...` assignments are deliberate copy points, not literal values.
Run the pilot from a credential-clean shell: StrixLab scans evidence and terminal
output for values held in sensitive-named environment variables and fails closed
rather than risk publishing one. Do not unset variables you do not understand;
start a clean shell or remove only credentials you know are unnecessary.

## Local state and safety

Generated sources, worktrees, builds, runs, caches, and locks live outside the
checkout. StrixLab resolves its state root in this order:

1. an explicit command-level override (for example `--home`);
2. `STRIXLAB_HOME`;
3. the platform data directory (normally `~/.local/share/strixlab`).

Resolving the path never creates it; commands that own generated state create
their own directories. StrixLab never bundles model weights, sidecars, or private
datasets implicitly, and treats imported source and commands as untrusted data.
The full architecture and v1 manifest contracts are recorded in
[`docs/design.md`](docs/design.md).

## Get involved

The most useful early contributions are small and honest:

- **Onboarding friction** — anything unclear or broken while following this
  README is worth a report.
- **Campaign problems** — bounded optimization hypotheses belong in
  [`docs/research-problems.md`](docs/research-problems.md); the controller
  procedure is [`docs/autoresearch.md`](docs/autoresearch.md).
- **Scenarios** — propose a stable benchmark question in an Issue, then add its
  checked-in suite and supporting configs in a scenario PR.
- **Experiments** — propose one reproducible candidate in a PR and collect
  matched baseline/candidate replications from community GPU owners.
- **Hardware field reports** — lightweight smoke-suite observations and setup
  friction that do not yet constitute a candidate experiment.
- **Docs and tests** — narrow clarifications and coverage.
- **Narrow code contributions** — keep them small, evidence-oriented, and
  independently testable.

The core vocabulary is intentionally small: a suite manifest defines a
**scenario**; a **candidate** is the change being tested; a **run** is one local
execution; a **replication** is one contributor's matched baseline and candidate
attempt, plus a comparison when both pass correctness; and an **experiment** is
the catalog entry that can collect many replications. Another person's rerun is
a replication—not universal "validation."

Start with the community workflow, then choose the matching GitHub template:

- Workflow: [`docs/community-workflow.md`](docs/community-workflow.md)
- Experiment catalog and template: [`experiments/`](experiments/)
- Report template: [`field-reports/TEMPLATE.md`](field-reports/TEMPLATE.md)
- Contribution policy: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Coding-agent execution contract: [`llms.txt`](llms.txt)

GitHub coordinates review and catalogs small summaries, identifiers, digests,
commands, and redacted notes. Contributors execute candidate code only on
machines they control. Never commit model weights, raw StrixLab homes, private
data, credentials, or evidence bundles.

## License

MIT

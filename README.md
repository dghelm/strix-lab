# StrixLab

Evidence-first, reproducible optimization research tooling for AMD Strix Halo.

StrixLab is a local, CLI-first research harness for running reproducible
optimization experiments on AMD Strix Halo hardware. It orchestrates existing
runtimes and isolated native builds, records immutable run evidence, and exports
deterministic, verifiable bundles. It is **not** an inference runtime, a
benchmark leaderboard, or an automatic upstream patch bot.

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
- **Deterministic smoke suite** — correctness-first suite execution that
  finalizes an immutable run (`run suite`).
- **Run inspection** — verify a finalized run's index, record, and checksums
  (`run inspect`).
- **Deterministic bundles** — export and verify portable run-evidence bundles
  (`bundle export`, `bundle verify`).

### What does not exist yet

- No web service, official judge, leaderboard, score, or "winning solution".
- No model-registration CLI, and no shipped public source-lock or build-profile
  examples that provision a build ID and model receipt end to end.
- No upstream pull-request bot; StrixLab never pushes changes or opens PRs.

Field reports (see below) are provisional observations, **not** judged
submissions or official scores.

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

## Provisioned-pilot smoke suite

Running the current smoke suite end to end requires an already-prepared build ID
and a machine-local verified model receipt SHA-256. Both currently require a
provisioned, operator-assisted setup — the repository does not yet ship a
self-service path that produces them.

```bash
uv run strixlab run suite configs/suites/smoke-qwen35.yaml \
  --machine configs/machines/strix-halo-128g.yaml \
  --build <BUILD_ID> \
  --model-receipt <MODEL_RECEIPT_SHA256>
uv run strixlab run inspect <RUN_ID>
uv run strixlab bundle export <RUN_ID> <NEW_BUNDLE_DIRECTORY>
uv run strixlab bundle verify <NEW_BUNDLE_DIRECTORY>
```

Replace every `<...>` placeholder with a real value before running: angle
brackets are shell redirection syntax, so the block above is illustrative and is
not directly pasteable.

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
- **Hardware field reports** — provisioned-pilot smoke-suite observations,
  including failures and inconclusive runs.
- **Docs and tests** — narrow clarifications and coverage.
- **Narrow code contributions** — keep them small, evidence-oriented, and
  independently testable.

To share an observation, copy the field-report template and open a pull request:

- Report template: [`field-reports/TEMPLATE.md`](field-reports/TEMPLATE.md)
- Contribution policy and the field-report route: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Coding-agent execution contract: [`llms.txt`](llms.txt)

Field reports carry small summaries, identifiers, digests, exact commands, and
redacted excerpts only. Never commit model weights, raw StrixLab homes, private
data, credentials, or evidence bundles.

## License

MIT

# StrixLab

## Machine readiness

Run the read-only machine doctor before GPU work:

```bash
uv run strixlab doctor --machine configs/machines/strix-halo-128g.yaml
```

It writes a versioned `doctor.json` beneath the resolved StrixLab home, returns 0 only when ready, and never changes clocks, power settings, drivers, or packages. See `docs/doctor.md` for the report and lock contracts.

StrixLab is evidence-first research tooling for reproducible AMD Strix Halo
optimization experiments. It orchestrates existing runtimes and isolated native
capsules; it is not an inference runtime or an automatic upstream patch bot.

The project is currently at **LAB-000**. This foundation provides strict,
versioned manifests, packaged JSON Schemas, deterministic subprocess execution,
and path isolation. It does not yet build GPU code, run models, or benchmark
ROCm workloads.

## Requirements

- Linux
- [`uv`](https://docs.astral.sh/uv/) 0.9.12 or newer
- Python 3.12 or newer (installed automatically by `uv` when needed)

## Development

```bash
uv sync --locked --all-groups
uv run strixlab --help
uv run pytest
```

Useful validation commands:

```bash
uv run strixlab schema show source-lock
uv run strixlab manifest validate source-lock path/to/source.yaml
uv run ruff format --check .
uv run ruff check .
uv run mypy src
```

`manifest validate` checks the raw, unresolved manifest structure. It deliberately
does not expand `${NAME}` placeholders. Trusted execution code must resolve those
values and validate the resolved result again with
`resolve_and_validate_manifest`; raw validation alone is not final validation.

## Local state

Generated sources, worktrees, builds, runs, caches, and locks will live outside
the checkout. StrixLab resolves its state root in this order:

1. an explicit command-level override;
2. `STRIXLAB_HOME`;
3. the platform data directory (normally `~/.local/share/strixlab`).

Resolving the path never creates it. Commands that own generated state are
responsible for explicit directory creation.

## Design boundaries

The adopted architecture and v1 manifest contracts are recorded in
[`docs/design.md`](docs/design.md). In particular:

- source repositories under test remain disposable inputs;
- correctness gates precede performance scoring;
- profiled diagnostics remain separate from clean measurements;
- large models, sidecars, and private datasets are never bundled implicitly;
- StrixLab does not push changes or create upstream pull requests;
- a future challenge service remains separate from the local research core.

## License

MIT

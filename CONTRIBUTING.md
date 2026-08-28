# Contributing

StrixLab is pre-alpha. Keep changes narrow, evidence-oriented, and independently
testable. Do not modify a developer's unrelated `llama.cpp`, ROCmFPX, or model
checkout; future source operations must use StrixLab-owned disposable worktrees.

## Local checks

```bash
uv sync --locked --all-groups
uv lock --check
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv run pytest --cov=strixlab --cov-report=term-missing --cov-fail-under=90
uv build
```

Run `uv run ruff format .` to apply formatting. The repository deliberately has
no second task runner or pre-commit framework; these `uv` commands are the
canonical development interface.

## Contribution boundaries

- Preserve failed, negative, and inconclusive experimental evidence.
- Never include model weights, credentials, or private datasets.
- Treat imported commands and candidate source as untrusted data.
- Do not add GitHub write automation, source pushes, or pull-request creation.
- Introduce dependencies and directories with their owning executable behavior,
  rather than scaffolding unused future structure.

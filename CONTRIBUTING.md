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

## Field reports

Field reports are small, privacy-safe observations — repository-validation or
self-service hardware smoke-suite runs, including failures and inconclusive
results. To
submit one, copy [`field-reports/TEMPLATE.md`](field-reports/TEMPLATE.md) to
`field-reports/YYYY-MM-DD-<short-slug>.md`, fill it in, and open a pull request.

- One report per file, one report per pull request. Field-report PRs are
  strictly report-only; any code or documentation fix belongs in a separate PR.
- Redact usernames, absolute home paths, hostnames, environment values, and
  device serials. Never include weights, credentials, private data, a raw
  StrixLab home, or an evidence bundle.
- Reports are provisional observations, not official scores or solutions.

When opening the PR on GitHub, pick **field-report** from the template chooser.
If the chooser does not appear, append `?template=field-report.md` to the
compare URL (for example
`.../compare/main...your-branch?expand=1&template=field-report.md`). Linking the
[field-report pull-request template](.github/PULL_REQUEST_TEMPLATE/field-report.md)
source file alone does not apply it.

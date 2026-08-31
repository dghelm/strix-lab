<!--
Copy this file to field-reports/YYYY-MM-DD-<short-slug>.md and fill it in.
One report per file, one report per pull request. Reports are provisional
observations, not official scores, submissions, or solutions.

Privacy: keep command structure and repository-relative paths, but redact
usernames, absolute home paths, hostnames, environment values, device serials,
and any other identifying values. Never include weights, credentials, private
data, a raw StrixLab home, or an evidence bundle. Delete these comments.
-->

# Field report: <short title>

- **Report kind:** repository validation | self-service hardware smoke suite
- **StrixLab commit:** <git rev-parse HEAD>
- **Date:** YYYY-MM-DD
- **Reporter:** <name or handle — optional>

## Environment

- **Machine / OS / GPU summary:** <e.g. Strix Halo 128G, Linux, gfx1151 — no
  hostnames or serials>

## Commands run

Keep command structure and repository-relative paths; redact identifying values.

```bash
<exact commands, e.g. uv run pytest>
```

## Outcome

- **Result:** pass | fail | inconclusive

### Hardware smoke-suite results (omit for repository-validation reports)

Use `N/A` for any field not reached.

- **Build ID:** <BUILD_ID | N/A>
- **Model receipt SHA-256:** <MODEL_RECEIPT_SHA256 | N/A>
- **Run ID:** <RUN_ID | N/A>
- **Suite result status:** <status | N/A>
- **Verified bundle summary** (from `strixlab bundle verify`; `N/A` if bundle
  verification was not reached):
  - `outcome`: <outcome | N/A>
  - `run_record_sha256`: <record-sha256:... | N/A>
  - `member_count`: <count | N/A>

## Failure excerpt or friction notes

<Redacted error excerpt, or notes on setup/docs friction.>

## Reproduction notes

<What another person would need to reproduce this, e.g. config paths, ordering.>

## Privacy confirmation

- [ ] This report contains no model weights, credentials, private data, raw
      StrixLab home, or evidence bundle, and identifying values are redacted.

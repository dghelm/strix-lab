# StrixLab foundation design

This document is the tracked authority for decisions adopted into StrixLab. It
intentionally replaces the private implementation handoff as repository-facing
documentation; the handoff remains ignored reference material.

## Architectural invariants

1. StrixLab is a standalone orchestration and evidence repository. Runtime
   sources are pinned inputs materialized in StrixLab-owned worktrees.
2. Python is the control plane. Timed kernels and capsules remain native
   C++/HIP executables behind a process contract.
3. Immutable YAML, JSON, JSONL, logs, patches, and checksums are authoritative.
   Any future index is derived and rebuildable.
4. Correctness gates run before performance ranking. Failed and inconclusive
   candidates remain evidence.
5. Profiled runs are diagnostics; final comparisons use clean equivalent arms.
6. Model weights, sidecars, calibration data, and secrets are referenced by
   identity and hash, never redistributed implicitly.
7. StrixLab never pushes runtime changes or opens upstream pull requests.
8. Imported challenge and candidate artifacts are data. Parsing is pure;
   environment resolution is an explicit trusted operation followed by full
   validation of the resolved result.

## Validation stages

Manifest handling has two deliberate entry points:

1. `parse_manifest_text` or `read_manifest` parses untrusted YAML without
   interpolation. `validate_manifest` may then establish raw structural validity
   while placeholders remain literal strings.
2. Trusted execution code calls `resolve_and_validate_manifest`, which expands
   environment values and validates the resulting object again. Resolution can
   change whether a constrained value is valid, so raw validation is never a
   substitute for this final step.

Mapping keys are never interpolated. Environment replacements may not introduce
unresolved tokens unless explicitly escaped as literals.

## Future challenge boundary

StrixLab may later own local challenge bundles, capsules, practice runs, evidence,
and verification protocols. A separate StrixGolf service would own authentication,
central judging, submissions, leaderboards, and agent-facing web APIs. Local
measurements are provisional; only a pinned canonical judge can issue an official
score. Declarative policies should precede review-gated source submissions, and
the subprocess timeout contract is not a security sandbox for untrusted GPU code.

Challenge support begins only after immutable evidence, comparison judging, and
the capsule process contract exist. The foundation merely keeps manifest and CLI
registries extensible so later challenge kinds do not require restructuring.

## Common lexical rules

- Dash identifier: `^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$`
- Underscore identifier: `^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$`
- Environment/CMake key: `^[A-Za-z_][A-Za-z0-9_]*$`
- Build target: `^[A-Za-z0-9][A-Za-z0-9._+-]*$`

Constrained strings reject empty values, surrounding whitespace, and NUL bytes.
Models use strict types, reject unknown fields, and reject non-finite numbers.

## Source lock v1

| Field | Contract |
|---|---|
| `schema_version` | Integer literal `1` |
| `id` | Dash identifier |
| `kind` | String literal `git` |
| `url` | Required nonempty string |
| `commit` | Exactly 40 lowercase hexadecimal characters |
| `branch_hint` | Optional nonempty string; defaults to `null` |
| `submodules` | Required Boolean |
| `adapter` | Underscore identifier |
| `allowed_dirty_state` | Boolean literal `false` |

Branches are metadata, never reproducibility anchors.

## Machine profile v1

All fields are required; the schema enforces structural and physical validity,
not operational policy.

| Field | Contract |
|---|---|
| `schema_version` | Integer literal `1` |
| `id` | Dash identifier |
| `expect.gpu_arch` | Nonempty string |
| `expect.integrated_gpu` | Boolean |
| `expect.memory_gib_min` | Finite number greater than zero |
| `exclusive_lock.path` | Absolute, NUL-free path |
| `telemetry.amd_smi` | `auto`, `required`, or `disabled` |
| `telemetry.sample_interval_ms` | Positive integer |
| `validity.require_ac_power` | Boolean |
| `validity.max_background_gpu_busy_pct` | Finite number from 0 through 100 |
| `validity.min_available_memory_gib` | Finite nonnegative number |
| `validity.temperature_warn_c` | Finite number |

`BASE-000` will interpret configured thresholds as warnings or invalidation rules.

## Build profile v1

All fields are required and list order is preserved.

| Field | Contract |
|---|---|
| `schema_version` | Integer literal `1` |
| `id`, `source` | Dash identifiers |
| `generator` | String literal `Ninja` |
| `build_type` | `Release`, `RelWithDebInfo`, or `Debug` |
| `environment` | Environment-key to string mapping |
| `cmake` | CMake-key to finite scalar mapping |
| `targets` | Nonempty, ordered-unique build-target list |
| `post_build_capture` | Nonempty, ordered-unique allowlisted capture list |

Allowed capture values are `cmake_cache`, `binary_hashes`, `ldd`,
`elf_dynamic_section`, and `compile_commands`.

## Versioning and schemas

Manifest dispatch is by external kind plus integer `schema_version`. Version 1 is
the only accepted version. Future versions receive parallel models and schemas;
StrixLab never silently migrates an input.

Canonical JSON Schema 2020-12 resources are distributed inside the Python wheel
with IDs of the form `urn:strixlab:schema:{kind}:1`. Checked-in schema bytes are
generated deterministically and verified by tests.

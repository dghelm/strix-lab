# StrixLab foundation design

The implemented BASE-000 machine-readiness contract is specified in [Machine doctor](doctor.md). It deliberately stops before immutable evidence bundles (`EVIDENCE-001`).

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

Post-build evidence is captured unconditionally by the adapter, not selected by a
manifest field. Before the final configure the adapter installs a versioned CMake
File API `codemodel-v2` query so that configure emits the reply (whose major
version is enforced at both the response descriptor and the loaded object), and,
after the build, discovers each requested target's artifacts from that reply. Each
artifact is stream-hashed with no whole-file buffer or fixed size cap — only a
small header is read to detect format — and pre/post descriptor metadata must
match. For every artifact it records the normalized root-relative path, mode, byte
size, SHA-256, and format: an `ar` archive, or an ELF with its parsed header type
(`ET_REL`, `ET_EXEC`, `ET_DYN`, `ET_CORE`). The target's CMake kind must match the
produced format — executables are ELF `ET_EXEC`/`ET_DYN`, shared/module libraries
`ET_DYN`, object libraries `ET_REL`, static libraries `ar` archives.

It captures the raw `CMakeCache.txt`, `compile_commands.json` (or an explicit
absent observation; a dangling symlink at that path is divergent state, not an
absent observation, and fails closed), and, for every ELF, raw and parsed
`readelf -d` evidence (the `NEEDED` shared-library list plus the `SONAME`,
`RPATH`, and `RUNPATH` dynamic entries). Each requested target's File API reply
must also self-identify with the `id` and `name` of its codemodel reference.
Only dynamic ELF (`ET_EXEC`/`ET_DYN`)
is additionally inspected with `ldd`; relocatable objects and archives are not.
Truncated inspection output is never parsed as complete. Any in-root dynamic
dependency joins a recomputed runtime closure and is hashed into the artifact set;
external system libraries remain recorded observations. The `ldd`/`readelf`
capture tools and each mandatory executable's `--help`/`--version` output are
captured as provenance. A symlink, special file, escaping path, malformed or
truncated inspection, target/format mismatch, or missing dependency fails the
build closed.

These records form the immutable artifact set and canonical build record that key
the reproducible build cache; identical inputs reuse a materialized build root
(cache hit) or rebuild one into the same stable path after cleanup (rehydration),
and any integrity divergence fails closed rather than silently rebuilding. Before
publishing the canonical record, the producing attempt durably records
`build/provenance.json` binding the build ID, artifact-set ID, recipe, snapshot,
and candidate, together with a complete digest/size inventory of every required
pre-publication evidence file (artifacts, profile, environment, source evidence,
snapshot manifest, configure caches, compile database, File API replies, tool and
process observations, and source reproducer bytes). That inventory is accumulated
from the exact bytes at the moment each evidence file is written, so it needs no
pre-publication re-read or re-hash of the tree; the immutable record publication
(`publish_record`) remains the independent on-disk copy/hash boundary, so a later
tamper of the evidence still fails through record verification. Build inspection
later authenticates that immutable producer attempt record — read lock-free to
avoid a recipe→build-ID lock inversion — requires the allowlisted inventory to be
complete and byte-authentic against the record's own manifest (rejecting missing,
duplicate, unsafe, or digest-divergent entries), and verifies the referenced source
diff/patch bytes against the canonical reproducer.

Rehydration equivalence is exactly the artifact-set ID plus the requested-target
and CMake-selection identity; a rehydrated root's own non-identity observations
(CMake cache, compile database, inspections, capture tools) may differ from the
original producer's. Each materialization therefore durably records its own
validated artifact evidence, journal-bound by digest, and a later cache hit,
inspection, or cleanup verifies the current root against *that* materialization's
evidence — never the original producer's — while the canonical record and its
producer provenance are never mutated.

A materialized root becomes reusable as a cache hit only once an immutable
post-finalization `build attestation` names the finalized `SUCCESS` attempt that
completed it, binding the canonical digest, build ID, artifact-set ID, execution
class, and the exact content-addresses of the producer and attestor attempt
records. Those record-digest anchors are required to match at reuse, so a
self-consistent replacement of either immutable record (whose content-address then
differs) fails closed; during crash-forward recovery, before any attestation
exists, the producer record digest is additionally anchored to the authoritative
recipe-index entry (read under the already-held recipe lock, never inverting the
recipe→build-ID lock order). A normal build's attestor is the producer itself; a
crash between publication and finalization leaves a PRESENT-but-unattested root,
which the next attempt completes forward and attests as an explicit `recovered`
attestation (a distinct finalized SUCCESS attempt, never a false producer SUCCESS
claim) before the root is treated as reusable. `build inspect` reports that
crash-forward state as explicitly unattested rather than fully verified, and
cleanup refuses to destructively remove it.

Publication itself is journaled so it can always be completed forward. Before the
canonical record is published, the producing attempt writes the canonical bytes to
a per-build publication-staging file and advances the materialization journal to
`PUBLISHING`, binding that file's digest. A crash during publication is recovered
by re-reading those journal-bound staged bytes (or an already-published canonical
that matches them) and completing the record/index publication and the transition
to `PRESENT`; the materialized root is never discarded for a publication that had
begun. A `PUBLISHING` state whose staged bytes and canonical are both unrecoverable
fails closed as an integrity failure when its root is missing or unverifiable, and
otherwise discards only a fully owner-authenticated root back to `VACANT`. The
canonical record, build index, and attestation are each published crash-atomically
— a fully fsynced same-directory temporary is renamed no-replace and the parent
fsynced (and accepting a byte-identical existing target still repeats that parent
fsync) — so a crash can never leave a visible partial immutable file. After a
canonical digest is journal-bound, any recovery that finds the canonical
record/index pair missing, corrupt, or divergent is an integrity failure that
preserves the root and evidence rather than regressing to a fresh state or
destructively cleaning.

Descriptor-anchored, no-follow containment scopes to the untrusted **read and
recovery** surface, where an attacker could swap a per-build directory for a
symlink between operations. Every per-build journal, events, and publication-staging
directory is opened once from its validated parent storage root with
`O_DIRECTORY | O_NOFOLLOW`, and all subsequent enumeration, reads, orphan-event
reconciliation, and the reconciled `current.json` write are performed relative to
those held descriptors, so a directory symlinked in after validation cannot
redirect anything, and a dangling per-build entry is corruption rather than absence.
The write side (a live attempt creating and appending to its own owned journal, and
`publish_record` copying an owned evidence tree) validates each component owner and
type and creates entries no-follow, but is not required to re-anchor every write
through a single held descriptor: it operates on directories the process itself
created and owns under `0o700` roots, not on attacker-swappable inputs. The parsed
`readelf`/`ldd` evidence is
authenticated through the producer provenance, so a cache hit and `build inspect`
deliberately do not re-execute the inspection tools — a hit still skips all final
configure and compiler work, and instead reverifies the owned root, the current
materialization's artifacts, and the attestation immediately before success. The
only inspection value excluded from any comparison is the raw `ldd` process
digest, and only from rehydrate equivalence, because it is attempt-variable (ASLR
randomizes the load addresses in `ldd` output); every other captured field must
match.

## Versioning and schemas

Manifest dispatch is by external kind plus integer `schema_version`. Version 1 is
the only accepted version. Future versions receive parallel models and schemas;
StrixLab never silently migrates an input.

Canonical JSON Schema 2020-12 resources are distributed inside the Python wheel
with IDs of the form `urn:strixlab:schema:{kind}:1`. Checked-in schema bytes are
generated deterministically and verified by tests.

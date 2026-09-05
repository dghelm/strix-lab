# StrixLab foundation design

The implemented BASE-000 machine-readiness contract is specified in
[Machine doctor](doctor.md). It deliberately stops before immutable evidence
bundles (`EVIDENCE-001`).

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
9. Profile-guided campaigns freeze the evaluator before patches. Screening
   and confirmation are separate phases. Calibration is distinct
   baseline/baseline evidence. An interrupted candidate stays spent while
   untouched candidates may continue; interrupted calibration stops. Count
   budgets include calibration, confirmation, failures, and interruptions
   and are not wall-clock deadlines. The raw judge verdict is preserved;
   `objective_met_provisional` is campaign-local. A retain decision is never
   an upstream push.

### Local storage trust boundary

StrixLab trusts the operating-system account that owns its private storage. Its
filesystem defenses cover crashes, malformed or stale state, foreign ownership,
symlinks, special files, path traversal, and cooperating StrixLab processes that
honor the advisory locks. They also detect many unexpected inode changes and refuse
divergent state.

Version 1 does not claim isolation from a malicious process running concurrently as
that same account. Such a process can already rewrite or rename files inside the
account's `0o700` directories between any two POSIX system calls. Defending that
boundary requires stronger isolation—such as a separate UID, mount namespace, or
sandbox—not an indefinitely expanding collection of pathname race checks. The code
therefore keeps security-sensitive reads and destructive traversal descriptor-
anchored where practical, while treating the owning UID as part of the trusted
computing base.

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

## Native capsule v1 boundary

`CAPSULE-001A` defines a library-only protocol and evidence adapter. It has no provider
registry, comparison path, or implicit integration with the suite runtime. A trusted caller must
already hold an active `RunSession` and supply the resolved `CapsuleManifestV1` plus its
digest, a trusted absolute executable path plus expected SHA-256, a complete child
environment and working directory, a caller-owned scratch root, and the applicable
redaction context. The adapter does not lease or build artifacts, acquire a machine lock,
observe hardware, reconstruct an environment, begin or finalize a run, or decide the
caller's overall run outcome.

The frozen capsule manifest binds `id`, opaque dash-identifier `candidate`, `machine`,
and an authoritative build requirement: `source_id`, exact 40-character
`source_commit`, `toolchain_mode` (`host` or `rocm`), `gfx_target`, and executable
`target`. No profile label substitutes for those leased-build coordinates. Its contract
is exactly `native-capsule-v1`, pins `scenario_sha256`, and carries the required frozen
`CapsuleComparisonContractV1`. That comparison contract has the sole policy
`paired-latency-log-bootstrap-v1`, a nullable strict integer
`protected_regression_bps` from 0 through 10,000, and one of exactly two canonical
`permitted_arm_differences` tuples: `("candidate-id",)` or
`("candidate-id", "source-candidate", "build-output")`. Independent `describe`,
`correctness`, and `benchmark` timeouts are finite, positive, and at most 3600 seconds.

The adapter executes exactly three allowlisted child operations, in this order:
`describe`, `correctness`, then `benchmark`. Each argv is fixed as
`<trusted-executable> <operation> --request /proc/self/fd/<N>`; the literal
`--request` marker and four-entry argv are mandatory, with no shell, discovery,
free-form flag, or child-selected path. `<N>` is an inherited descriptor opened read-only
over an immutable, write/grow/shrink/seal-locked memfd containing the exact canonical
JSON request. The child receives only the caller's complete
environment (`inherit_env=False`) and cwd. Stdout and stderr have independent hard byte
bounds and the process always runs through the shared bounded runner. The executable is
stream-hashed for stable descriptor metadata and content, matched to the caller's digest,
and rechecked immediately before and after every child and once more before terminal
publication. Request descriptors are checked against their pre-launch identity, seals,
read-only access mode, and exact bytes; captured stream spools are independently checked
against their exact bytes, sizes, and SHA-256 values.

Every request binds protocol, operation, capsule ID, opaque candidate ID, scenario SHA,
manifest SHA, and executable SHA. `correctness` carries the accepted `describe` response
SHA, while `benchmark` carries the accepted `correctness` response SHA; both later
requests also carry the accepted typed scenario contract and its digest. Every response
must be exact UTF-8 canonical JSON and echo all of those bindings. Noncanonical JSON,
unknown fields, coercible values, wrong echoes, non-finite values, or a response outside
the hard bound is rejected. An optional canonical opaque payload is limited to 256 KiB
and retained only in raw response evidence. It is deliberately absent from the generic
terminal result and never participates in correctness, equivalence, statistics, or
pass/fail; a later scenario-specific implementation such as `TOPK-001` may interpret it.

`describe` returns the complete ordered scenario contract and must repeat the manifest's
comparison contract exactly before the phase is accepted. Each zero-based ordered
coordinate declares an exact coordinate ID, case ID, training/evaluation set, mode,
input ID and SHA-256, warmup count, and sample count. Coordinate IDs and `(case_id,
mode)` pairs are unique, at least one coordinate is evaluation, every sample count is at
least five, and coordinates sharing a case ID must share their case set, input ID, and
input digest. Metric, direction, and statistical policy are scenario-level contract
facts rather than coordinate-local fields. `correctness` must reproduce the accepted
coordinates exactly, without omission, duplication, or reordering, and every coordinate
must pass before `benchmark` can start. `benchmark` must reproduce the same coordinates
exactly and report exactly the declared number of positive finite ordered latency samples
plus nonnegative workspace bytes for each. Later requests and terminal results propagate
the accepted scenario unchanged.

### Capsule comparison contract and offline comparison

`paired-latency-log-bootstrap-v1` is scenario-neutral and lower-is-better. D2a defines
and authenticates this contract. D2b1 adds the pure offline
`compare_finalized_capsule_runs` boundary: it loads baseline then candidate, applies the
closed admission projection, computes the directional statistics and verdicts, and returns
a canonical in-memory report. The pure comparator does not write evidence, allocate a run,
publish a bundle, or interpret TOPK payloads. The separate publication wrapper below
provides CLI dispatch. Reversing the arms intentionally
produces a different directional comparison; equal candidate IDs remain legal.

For a coordinate with `n` positionally paired positive finite samples, the mathematical
effect is `delta_i = ln(baseline_i / candidate_i)`, but implementations must evaluate it
as `math.log(baseline_i) - math.log(candidate_i)` so the intermediate ratio cannot
overflow. Positive delta means that the candidate is faster. The point estimate is
`mean_log_effect = math.fsum(deltas) / n`; the geometric latency ratio is
`math.exp(mean_log_effect)`, and improvement percent is
`100 * math.expm1(mean_log_effect)`. Thus positive percentages are improvements and
negative percentages are regressions. Workspace bytes are authenticated and reported
per arm and as a delta, but never alter admission, intervals, noise, or verdicts.

The interval is exactly 4,096 deterministic positional paired-bootstrap replicates.
Each replicate makes `n` draws with replacement from the delta positions. For replicate
`r` and draw `d`, compute SHA-256 over this exact call to
`strixlab.source_identity.length_frame`:

```python
length_frame(
    "strixlab.capsule.paired-latency-log-bootstrap.v1",
    (
        ("policy_id", b"paired-latency-log-bootstrap-v1"),
        ("baseline_record_sha256", baseline_record_sha256.encode("ascii")),
        ("candidate_record_sha256", candidate_record_sha256.encode("ascii")),
        ("case_id", case_id.encode("utf-8")),
        ("mode", mode.encode("utf-8")),
        ("replicate", struct.pack(">Q", r)),
        ("draw", struct.pack(">Q", d)),
    ),
)
```

The record values are the authenticated ASCII `record-sha256:` identities (that prefix
followed by 64 lowercase hexadecimal characters), `r` is zero through 4,095, and `d` is
zero through `n - 1`. `length_frame`
contributes its fixed `b"strixlab-lf-v1\0"` prefix, UTF-8 domain preceded by its
unsigned 32-bit big-endian byte length, unsigned 32-bit big-endian field count, then for
each field in the shown order its ASCII label preceded by an unsigned 32-bit big-endian
byte length and its value preceded by an unsigned 64-bit big-endian byte length. No
comparison-contract digest, case set, or other input enters the seed. Interpret the first
eight SHA-256 bytes as an unsigned big-endian integer modulo `n` to select the paired
position. Replicate means also use `math.fsum / n`.

For sorted zero-based values `x[0]` through `x[N - 1]`, the R-7 quantile at `p` uses
`q = (N - 1) * p`. It returns `x[0]` when `q <= 0`, `x[N - 1]` when `q >= N - 1`, and
otherwise sets `j = floor(q)` and `g = q - j` and returns
`x[j] + g * (x[j + 1] - x[j])`. The 95 percent interval is this quantile at exactly
`p = 0.025` and `p = 0.975` over the sorted 4,096 replicate means.

Every median sorts its input first. For odd length `N`, it is `x[N // 2]`; for even
length `N`, it is `math.fsum((x[N // 2 - 1], x[N // 2])) / 2`. Baseline noise is the
log-MAD value `1.4826 * median(abs(log(baseline_i) - median(log(baseline))))`, using that
median rule for both the baseline logs and their absolute deviations. The protected
threshold uses the same rule on each arm's raw latency samples.

A coordinate is inclusively `inconclusive` when `lower <= 0 <= upper` or
`abs(mean_log_effect) <= baseline_log_mad`; otherwise it is `improvement` when both
interval endpoints are strictly positive and `regression` when both are strictly
negative. Only evaluation coordinates enter the provisional aggregate: all improvement,
all regression, or all inconclusive preserves that verdict, while every other mixture is
`mixed`. When `protected_regression_bps` is non-null, a coordinate is a protected
regression only when both interval endpoints are strictly negative and
`candidate_median / baseline_median > 1 + bps / 10_000`. A protected regression changes
only a provisional aggregate `improvement` to `mixed`; every other provisional aggregate
is unchanged. Thus 500 basis points protects strictly more than 5 percent regression,
while exactly 5 percent is not protected. A null value disables only that additional
guard.

D2b1 fails closed rather than return a comparison whenever positional structure or
arm lengths diverge, or whenever any input or derived delta, `math.fsum`, mean, digest
index, quantile interpolation, logarithm, exponential, `expm1`, median, ratio, MAD,
threshold, or percentage is non-finite, overflows, or otherwise cannot be represented by
the specified operation.

D2b1 admission uses this closed field-path normalization table. A listed
difference is legal only when its exact token appears in the identical comparison
contract on both arms; every unlisted semantic field remains byte-for-byte equal after
strict authentication.

| Difference token | Exact semantic field paths normalized arm-locally |
|---|---|
| `candidate-id` | `manifest.candidate`; `capsule/result.json.candidate`; `capsule/protocol/result.json.candidate`; and `candidate` in every accepted capsule protocol request and response. |
| `source-candidate` | `capsule/build.json.canonical.source.{candidate_id,content_tree_id,snapshot_id,source_evidence_sha256,snapshot_manifest,diff,patches}` and only `capsule/build.json.canonical.source.source_evidence.{preparation_id,request_digest,patches,root_tree,content_tree_id,candidate_id,diff_file,diff_sha256,diff_size_bytes,status,created_at}` within the nested evidence object. `manifest.build.source_id`, `manifest.build.source_commit`, and nested `source_evidence.{source_id,source_locator,source_locator_sha256,base_commit,branch_hint,adapter,submodules_enabled,submodules}` are not normalized; locator, base source, adapter, and submodule policy/evidence remain equal. |
| `build-output` | `capsule/build.json.{build_id,canonical_record_sha256}`; `capsule/build.json.canonical.{recipe_id,build_id,producer_attempt_id}`; `capsule/build.json.canonical.artifacts.{artifact_set_id,inspections,capture_tools,cmake_cache_sha256,compile_commands_sha256}`; `capsule/build.json.canonical.artifacts.targets[*].target_id`; `capsule/build.json.canonical.artifacts.artifacts[*].{mode,size_bytes,sha256}`; `capsule/result.json.{build_id,canonical_record_sha256,executable_sha256}`; `capsule/protocol/result.json.executable_sha256`; and `executable_sha256` in every accepted capsule protocol request and response. Target and artifact tuple cardinality/order and `targets[*].{schema_version,name,target_type,artifacts}` plus `artifacts[*].{schema_version,path,kind,elf_type,targets,runtime_dependency}` remain equal, preserving the target topology and associating changed content only with existing targets/paths. `profile_sha256`, toolchain mode, canonical environment, requested targets, selections, tools, and the manifest target remain equal. |

Derived and authentication digests are never broad semantic-difference tokens. Each arm
must independently authenticate its record SHA, manifest SHA, portable entry/blob SHA,
input-snapshot SHA, result SHA, protocol-result SHA, request SHA, response SHA, canonical
build-record SHA, and executable SHA. They may differ only as the recomputed consequence
of an allowed exact field above and are removed from semantic equality rather than
whitelisted by a provenance label. The scenario-contract SHA, comparison-contract SHA,
machine-profile SHA, coordinate-structure SHA, ordered `(case_set, case_id, mode)` keys,
and all other unaffected digests must remain equal.

Admission uses explicit frozen projections with exact v1 field guards. Stable enclosing
result input roles and paths, protocol/phase status, process outcome and capture semantics,
stderr identity, correctness, and coordinate structure remain equal. Per-arm snapshot
authentication establishes the accepted request/response echoes and chains; admission
removes only their recomputed digests and stdout byte identities, process duration,
benchmark latency/workspace inputs, and the already non-semantic opaque payload.

Portable evidence is confined to `capsule/protocol/{describe,correctness,benchmark}/`:
canonical `request.json`, a secret-free `process.json`, canonical `stdout.json` when
accepted as canonical JSON or one text fallback otherwise, and optional exact UTF-8
`stderr.txt`. Correctness roles are used for describe/correctness, samples roles for
benchmark, and the summary role for `capsule/protocol/result.json`, which is always
published last. Exact bytes are never duplicated under conflicting media types. Every
prospective artifact is secret-scanned before publication, and the active `RunSession`
rechecks it at the write boundary.

Ordinary spawn, timeout, output-limit, exit, parsing, echo, correctness, coverage, and
sample-completeness failures produce a strict `failed` protocol result with the truthful
completed-prefix evidence. Unsafe child output is withheld and produces a safe failed
result. Executable mismatch/drift, request or spool divergence, a pre-existing protocol
subtree, or a publication-integrity failure raises `CapsuleIntegrityError`; such a path
never publishes a success claim. In every case the caller remains solely responsible for
the enclosing run's terminal outcome.

`CAPSULE-001C` adds the narrow production caller for that adapter and the command
`strixlab run capsule CAPSULE_MANIFEST --machine MACHINE_MANIFEST --build BUILD_ID
[--home PATH]`. It leases only the named existing build; it never leases a source or
creates either source or build state. Before allocating a run it requires the exact
machine ID, source ID and base commit, toolchain mode, requested gfx membership, and
unique executable target recorded by the canonical leased build, then reverifies the
lease. The SHA-256 passed to the adapter is computed from the exact canonical
`manifest.resolved.yaml` serialization used by `begin_run`.

Acquisition order is build lease, `RunSession`, then machine lock; release is the reverse.
After run allocation, the complete canonical `capsule/build.json` and
`capsule/machine.json` payloads are preflighted together against the union of ambient and
canonical-child secrets before either is published, then published in that order. A lock
refusal produces a strict failed enclosing
`capsule/result.json` without executing the protocol. Under an acquired lock the runner
allocates a mode-0700 scratch root, reconstructs the complete canonical child environment
without ambient inheritance, invokes the library adapter, and removes scratch on every
exit. It reverifies the build lease while the machine lock is still held and publishes the
enclosing `capsule/result.json` only after authenticating the adapter's actual portable
`capsule/protocol/result.json` entry against the returned canonical result bytes. The
enclosing result binds that digest, the manifest target, the recorded executable digest,
and the protocol's exact closed reason without embedding a duplicate protocol result; the
run succeeds if and only if the protocol passed. Trusted executable and scratch
host-absolute paths never enter portable evidence.

Failures before allocation raise `CapsuleRunError` and create no run. Any exception after
allocation finalizes failure and raises `CapsuleExecutionError`, which exposes only the run
ID, an optional finalized record, and one fixed safe message. Ordinary protocol failures
remain structured failed results. CLI diagnostics use the ambient redaction context and do
not relay child output, exception causes, or free-form failure text. The generic runner,
finalized successful-capsule snapshot loader, and pure directional comparison library are
available. The snapshot independently
reauthenticates the scenario comparison contract against the manifest and exposes its
canonical digest, permitted-difference tuple, and ordered `(case_set, case_id, mode)`
alignment keys without deciding admission. The planned TOPK scenario remains inactive:
there is no checked-in runnable capsule configuration or TOPK payload interpretation.

### Derived capsule comparison publication

`strixlab compare BASELINE_RUN_ID CANDIDATE_RUN_ID --kind capsule [--home PATH]`
selects the offline capsule comparator. The existing two positional arguments and default
suite comparison remain unchanged; `--kind suite` explicitly selects that default. Mixed
run kinds fail authentication rather than falling back to another comparator.

`compare_capsule_runs` in `capsule_comparison_runs.py` calls the unchanged pure comparator,
renders a deterministic Markdown projection, and preflights exact output bytes for portable
media, member/aggregate capacity, and secrets before allocating a run. The captured and
resolved request binds the ordered source run IDs and record digests, comparison policy and
contract digest, canonical JSON report digest, and Markdown digest. Its experiment ID is
`capsule-compare-` plus the first 24 hex characters of SHA-256 over canonical request JSON.
Immediately before publication, both finalized capsule snapshots are fully reauthenticated
and their record digests must still match the report. This uses the existing owning-UID
storage trust boundary; it does not add hardware or build leases.

The wrapper owns a fresh derived `RunSession` and never modifies either source arm. It
writes exactly `comparison/report.json` (`application/json`) and `comparison/report.md`
(`text/markdown`), both with role `comparison`. The JSON is the exact canonical comparator
output; Markdown only renders its typed fields. Opaque payloads remain non-semantic.
Every valid statistical verdict finalizes `SUCCESS`, including regression, mixed, and
inconclusive. Errors before allocation create no run. Errors after allocation use existing
failure finalization and expose only a fixed-safe error, run ID, and available record path;
any already committed portable entries remain evidence. Integrity failures that prevent
finalization retain the existing recovery behavior rather than claiming a terminal record.

Repeating a command creates a new run with identical request/report bytes for unchanged arms;
it does not deduplicate or overwrite evidence. Existing run finalization and recovery remain
idempotent. Normal bundle export/verification applies. This is a derived report, not a
standalone proof of both arms: export both source-run bundles for independent verification.
Host-only publication tests cover authenticated fake arms, exact bytes, repeat and reversed
arms, bundle verification, rejected inputs, secret preflight, record drift, publication and
finalization failures, and CLI dispatch. No GPU or TOPK semantics are required.

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

Local profile-guided campaigns are a different layer: they reuse the existing
suite and comparison judge against a frozen evaluator and a finite reviewed
patch list. They are not hosted challenges, official scores, or an upstream
patch bot. The procedure, v1 commands, and staged implementation plan are in
[Bounded profile-guided campaigns](autoresearch.md). Ranked, bounded
hypotheses and the smallest honest post-merge pilot are in
[Profile-guided llama.cpp research problems](research-problems.md). That
portfolio is docs-only, not an experiment catalog, and separates v1 patch
campaigns from configuration tuning and blocked workloads.

For the pilot, GitHub is the collaboration and catalog boundary—not an execution
service. Existing suite manifests are called **scenarios**. A **candidate** is a
reproducible change evaluated under one scenario; a local execution is a **run**;
one contributor's matched baseline and candidate attempt—with a comparison when
both pass correctness—is a **replication**; and the cataloged investigation of
one candidate under one scenario is an **experiment**. An experiment may
collect many replications.

Scenario proposals begin as Issues and their immutable rules land through
scenario PRs. Candidate experiments use separate PRs. Comments coordinate
replications, but accepted summaries are copied into checked-in Markdown records
before merge, with later observations added by follow-up PR. This deliberately
avoids a premature submission API or database. A future site may render the
repository catalog read-only; executing community candidate code on hosted GPU
machines remains a separate security and operations problem, not an implied
StrixLab capability.

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

## Run evidence v1

A run is the first trustworthy, reusable evidence boundary layered above the build
subsystem. It allocates a collision-safe identity, captures the raw and resolved
manifests that scope it, records an append-only crash-recoverable state journal, and
finalizes — on **success and on failure alike** — into an immutable, independently
verifiable run record with canonical checksums. A run does not execute a model and
defines no suite, model, or candidate semantics; those are later boundaries.

Storage lives under `runs/` in the data home: `allocation-staging/` (per-attempt
staging trees), `active/<run-id>/` (the live run), `records/<run-id>/` (the immutable
finalized record), `indexes/<run-id>.json` (the terminal projection), and
`locks/<run-id>.lock` (the per-run advisory lock). Bundle export instead stages
beside its own destination. Storage preparation validates `home` first and then
creates each directory descriptor-relative under its already-validated parent's held,
no-follow descriptor, validating owner, type, and the documented `0o700` mode at each
step — so a symlinked or foreign `home` or storage root is caught and never created
through or beneath. Every entry fails closed on detected drift. As stated in the local
storage trust boundary, this does not attempt to defeat the owning UID replacing a
validated directory in the interval between system calls.

A run ID is `run-<UTC-basic>-<experiment-dash-id>-<128-bit-random>` — an RFC-basic UTC
timestamp, the validated experiment dash identifier (bounded to 64 bytes), and 32 lower
hex characters of fresh randomness. Allocation stages a complete run tree (both manifests,
the run descriptor, the first `ALLOCATED` event, and the status projection), fsyncs it,
and renames it no-replace into `active/<run-id>`; a collision retries with a fresh ID up
to a bounded number of attempts. An ID is considered taken if `active/`, `records/`, or
`indexes/` already holds it, so a finalized run's identity is never reissued even though
its `active/` tree has been torn down. The per-run lock is acquired non-blockingly and
held for the whole session lifetime; a contended lock reports the run as busy rather than
failing.

State is `ALLOCATED → ACTIVE → TERMINAL`, with the only legal transitions `None→ALLOCATED`,
`ALLOCATED→ACTIVE`, `ALLOCATED→TERMINAL`, and `ACTIVE→TERMINAL`. Each transition appends a
length-framed event (digest domain `strixlab.run.event.v1`) chained by `previous_sha256` to
its predecessor, then atomically rewrites the `status.json` projection. Event and status
metadata legality is model-enforced everywhere the models are constructed or parsed: a terminal
event or status carries an outcome (its reason stays optional), and a nonterminal one carries
neither an outcome nor a reason. Committed-chain verification, orphan/temp adoption, and bundle
verification all authenticate the chain through one shared validator, so recovery and export
apply an identical linking, transition-legality, and status-projection predicate. Every
adapter-driven
write — local evidence, portable blobs and entries, events, and the status projection — is
**descriptor-anchored**: the live run root is opened once no-follow, each parent component is
opened or created relative to that held descriptor with `O_NOFOLLOW`, and the file is
published by a fsynced writer temp renamed no-replace within the same directory descriptor, so
an intermediate directory swapped for a symlink between operations cannot redirect the write.

Recovery authenticates the `active/<run-id>` root at entry: it opens the root no-follow
(rejecting a symlinked, special, or foreign directory), checks its device and inode against the
descriptor's recorded staged-root identity, and requires both the descriptor and status
projection to name the requested run. Subsequent operations repeat no-follow, ownership, and
content bindings at their own boundaries; detected cross-run or inode divergence fails closed.
Allocation-stage reclamation additionally runs under the stage's own per-run lock: a live
allocator holds that lock for the whole of its staging (including the brief pre-descriptor
empty-directory window), so a contended lock leaves the stage untouched instead of deleting it
out from under the allocator. Bulk recovery — both the allocation-staging and active-run loops —
skips only a *contended* lock; an unavailable or unsafe lock (a missing lock parent, or a
foreign-owned, non-regular, or over-permissive lock file) fails closed rather than being treated
as contention. Recovery then replays and authenticates the event chain and
reconciles at most one writer temporary per journal directory, and it fails closed on every
unexpected, malformed, symlinked, special, or foreign directory entry rather than filtering it
out. An event or portable-entry writer
temp is removed only after it is authenticated (owned, `0600`, non-symlink) and either matches
the byte-identical committed record or parses and links as the *exact* next event/entry — for
an uncommitted next entry, its sequence, closed role/media policy, logical-path uniqueness, and
referenced blob digest/size are all revalidated first. A portable blob writer temp is removed
only after its bytes hash to the digest embedded in its own writer-temp name. An unreferenced
committed blob is removed only after its owned mode and content-address are reverified. All
such removals are descriptor-relative. A session that exits its context without an explicit
outcome is finalized `FAILURE`; `INTERRUPTED` is reserved for recovery of a dead
nonterminal owner. An escaping exception finalizes `FAILURE` without masking the original
error, and an exception whose text fails secret-safety scanning is finalized with a fixed
neutral reason rather than leaking it.

Finalization ordering is fixed and crash-forward at every step: before checksum generation,
reconcile the run-root atomic-writer temps (a `status.json` or `checksums.sha256` temp left by a
crash between its fsync and rename is removed under a strict namespace, and any other unexpected
root writer temp fails closed) so a stray temp can never be hashed into the record; then generate
and fsync `checksums.sha256`, publish the immutable record, publish the terminal index, then tear
down the `active/` tree. A crash before any step is completed forward from the durable prior step
on recovery — and when recovery *accepts* a record or index published by a prior crashed
finalization, it fsyncs that entry's parent directory before teardown, so a crash between the
prior rename and its parent fsync cannot leave the accepted copy non-durable while the only
remaining active copy is deleted. Index rebuild and inspection share **one complete
finalized-record verifier**:
beyond the generic record check (which authenticates the exact file set and each file's digest
against the record manifest), it binds the record's semantics — the descriptor and status must
name this run and be terminal with an outcome, the captured manifest digests must match the
bundled manifest bytes, the full committed event chain must authenticate and project exactly
onto the status, and `checksums.sha256` must declare exactly the record payload set (minus
itself) with digests matching the authenticated files — so a record whose manifest is
self-consistent but whose semantics are corrupt fails closed. Teardown deletes the live tree
descriptor-relative: it empties the authenticated inode through a held, no-follow descriptor
(never re-resolving `active/<run-id>` by name) and removes the now-empty directory only after
re-confirming the name still resolves to that same inode. POSIX has no `rmdir`-by-descriptor;
under a malicious same-UID race the final name removal can target another empty directory, but
recursive deletion never follows that replacement or removes its contents. Inspection re-derives
and binds
the record digest from the published manifest bytes without re-copying the tree, and binds the
index to the fully verified record.

`checksums.sha256` is the canonical `sha256␣␣path` inventory of every regular file in the
record **except itself**, produced by the same `hash_owned_tree` primitive the build records
use (no second hashing path to drift), sorted by the path's UTF-8 bytes with exactly one
trailing newline. Verification re-hashes and rejects any missing, extra, or divergent entry.

Evidence written during a run is either **complete-local** or **portable**. Complete-local
evidence (`write_evidence`) is arbitrary owned run output that stays in the record and is
never exported. Portable evidence (`write_portable`) is content-addressed under a closed v1
policy: a bounded, ordered-unique set of roles (`environment`, `source`, `build`,
`correctness`, `samples`, `profiler-summary`, `comparison`, `summary`) and media types
(`application/json`, `application/x-ndjson`, `application/yaml`, `text/plain`, `text/csv`,
`text/markdown`, `text/x-diff`), each entry naming a deduplicated blob by digest. Logical
paths are validated run-relative and additionally rejected when they fall in the reserved
control namespace (`run.json`, `status.json`, `events/`, `portable/`, the manifests,
checksums, and record metadata): no backslash, absolute, `..`, control character, or
over-255-UTF-8-byte path, and all payload and text bytes are secret-scanned, with both
sensitive-interpolation and unsafe-output failures surfaced as a neutral run error. The v1
numeric limits — at most 1024 portable entries, 16 MiB per member, 64 MiB aggregate payload,
and 4096 total files — are enforced at write time over the deduplicated blob set, and again at
bundle export and verification.

A **bundle** is a deterministic, read-only directory export of exactly the portable surface
of a finalized run — never the complete-local evidence. Export inspects the finalized run,
collects the control files, the event chain, the portable entries, and only the blobs those
entries reference, secret-scans the whole set against the caller's environment, validates the
exact in-memory snapshot with the same binding logic `verify_bundle` applies, and only then
publishes a staged directory no-replace — a concurrent record mutation between inspection and
collection therefore fails closed before any destination exists. Both collection and
verification read each member while its own directory descriptor is held (single-component
`openat` with an `lstat`/`fstat` device+inode identity check), never re-resolving an
intermediate component by pathname, so a same-uid rename between enumeration and read cannot
redirect a read. The stage is a sibling under the destination's own parent (not under the data
home), so a cross-filesystem destination still publishes by an atomic same-directory rename
through a held parent descriptor; the destination parent is validated as an owned, non-symlink
directory and an existing destination is refused. Staging is durable before publication: each
member is written and fsynced, and every intermediate directory created under the stage
(`run/`, `portable/`, `blobs/`, `entries/`, `events/`) has its parent fsynced when the directory
entry is created, so a crash cannot leave a published bundle missing an intermediate directory
whose file was already flushed. The verified stage's descriptor is held across the publishing
rename and the published destination inode is checked against that stage identity. Detected
divergence removes the destination and fails the export; the same-UID concurrency exclusion
remains the local-storage trust boundary above. `verify_bundle` is a standalone read-only check:
it enumerates and reads the whole tree in one descriptor-held traversal — accumulating member
bytes as it goes and aborting before a member that would cross the 64 MiB aggregate bound is read
into memory — requires the member set to match `bundle.json` exactly, re-hashes every member
against its declared digest and size, and rejects any symlinked, executable, or undeclared
member. Beyond bytes, it
binds the bundle to its own control files:
the embedded `record-manifest.json` must be canonical and digest-match the manifest's declared
run-record digest; `run.json` and `status.json` must parse, agree on the run ID, and show a
terminal status whose outcome equals the manifest outcome; the checksum declarations must
cover exactly the record payload set minus `checksums.sha256` itself and agree with the record
digests; the manifest-input and resolved-manifest digests in `run.json` must match
their bundled bytes; and the complete, contiguous event chain must authenticate and
project exactly to `status.json`, including its terminal outcome. Every portable entry
filename must match its contiguous sequence and satisfy the closed role/media/path policy
with a present, digest-named blob (no missing or extra blob), and all references to a
deduplicated blob must agree on its media type. Because publication and verification are
descriptor-anchored and no-follow, a destination parent or declared member replaced by a
symlink fails closed as unavailable rather than being followed.

## llama-bench adapter v1

The `llama-bench` adapter is the first runner adapter: a library boundary that executes a
verified `llama-bench` binary for exactly one typed benchmark case, preserves bounded raw
process evidence in an active `RunSession`, and returns one versioned sample whether the
child succeeds, exits nonzero, times out, cannot spawn, truncates output, or produces
unparseable output. It defines no suite, model registry, comparison statistics, correctness
gate, or run command; later layers call it. It adds no generic runner framework, no
`llama-server`/capsule/profiler support, and no JSON or Markdown execution profile.

Its public surface is one typed case model, one executable/model provenance model, one
capability model, one result/sample model, a pure command builder and pure parsers, and one
orchestration entry point. All v1 models are strict, frozen, `extra="forbid"`,
finite-number-checked Pydantic models reusing the repository `DashId` and canonical JSON
serializer. A case carries exactly one nonzero token count from 0 through 1,048,576 — `pp`
or `tg`, never both — with `repetitions` positive and capped at 32, and a `metric_kind`
whose canonical value (`prompt-processing` or `text-generation`) must agree with that count.
The inputs model binds a build ID, source commit, absolute binary path, verified binary
SHA-256, model ID, absolute model path, and a caller-asserted model SHA-256 explicitly
labelled `asserted`: MODEL-001 will replace it with a verified receipt without changing the
benchmark or parser contract. The source commit is fixed to the ca94157 profile revision;
another revision requires another explicit profile.

Capability discovery is pinned to
`strix-llama.cpp@ca94157f70a2776e8da6b6849b50b45a083d0478` and its checked-in
`tools/llama-bench/README.md`. Before discovery the binary is verified as a non-symlink
regular executable and stream-hashed against its asserted digest with pre/post metadata
stability; that identity is re-required before every child and re-hashed after the final
child. The model is validated as a stable regular file before and after the window but not
hashed here. A pre-child mismatch aborts before launching that child; a final binary-hash or
model-metadata mismatch raises a typed integrity error and leaves no `sample.json`, because
no truthful binding can then be claimed. Discovery runs `<binary> --help` and an advisory
`<binary> --version` attempt through `run_process` — never a shell — with a caller-provided
complete environment (`inherit_env=False`), an explicit working directory, and separate
bounded capability and benchmark timeouts. Both probes are always attempted so their process
evidence is complete, unless an intervening integrity failure aborts before the next child.
No benchmark child runs unless both probes stay within bounds, the required help probe exits
successfully, is valid UTF-8, and matches the supported grammar (`-m/--model`, `-p/--n-prompt`,
`-n/--n-gen`, `-r/--repetitions`, and `-o/--output <csv|json|jsonl|md|sql>`, matched by exact
token spellings), and the advisory version attempt matches ca94157's expected unsupported
outcome — return code 1 plus stable `invalid parameter for argument` and `--version`
markers, since that
revision has no `--version` branch. ca94157-v1
always selects JSONL because that exact grammar advertises it; a future revision lacking JSONL
must introduce its own explicit capability/parser profile. Only an allowlisted argv is
constructed — binary, model, prompt tokens, generated tokens, repetitions, and the discovered
`-o jsonl` flag/value — never arbitrary extra arguments.

Parsers are pure and independently tested. The JSONL parser accepts exactly one nonblank
object line and rejects duplicate keys, non-finite values, trailing data, wrong types, missing
fields, and extra result rows. It normalizes only the v1 execution fields — prompt/generated
token counts, `avg_ts`, `stddev_ts`, and ordered `samples_ts` — and requires the row's
`model_filename` to equal the invoked absolute model path, its token counts to match the
one-metric case, `len(samples_ts) == repetitions`, every rate finite and positive, and the
standard deviation finite and nonnegative. Raw output is preserved separately; a nominal exit
code 0 is not success until parsing and case binding both succeed.

Evidence is written with `RunSession.write_portable`, role `samples`, so the existing 16 MiB
per-member, 64 MiB aggregate, path, media, and secret policies remain the single boundary.
Deterministic logical paths live under `adapters/llama-bench/<case-id>/`:
`capabilities/{help,version}.{stdout,stderr}.txt`, `capabilities/{help,version}.process.json`,
`capabilities/attempt.json`, `invocations/<zero-padded-ordinal>/{stdout,stderr}.txt` and
`process.json`, and `sample.json` written last. No full-output spools are used; each child
runs under a 256 KiB in-memory limit per stream, and the three-child worst case retains at
most six stream prefixes well beneath the aggregate budget. `run_process` always counts and
hashes the complete drained stream but decodes the bounded prefix with replacement; a stream
is exact and publishable only when it is not truncated and re-encoding its returned text
reproduces both the recorded byte count and SHA-256. Only exact streams are published, as
`text/plain`; truncated or non-UTF-8 streams have no text artifact, while `process.json` still
preserves total byte counts, complete-stream digests, truncation, and a neutral failure
category. Empty exact streams are recorded deterministically and never parsed.

Failure precedence is fixed for both probes and the benchmark child:
`capture-failed` > `spawn-failed` > `timed-out` > `output-oversized` > `encoding-failed` >
`nonzero-exit` > `parse-failed` > `success`, classified across both output streams. Every
secondary fact stays visible in the process projection, which carries only neutral categories
and never raw exception prose. A capability failure runs no benchmark child and still writes a
`sample.json` binding the probe evidence. Terminal sample status is one of `success`,
`capability-failed`, `process-failed`, `output-truncated`, or `parse-failed`. The adapter
returns a sample for child, capability, and parse failures; it never finalizes the
caller-owned `RunSession`. Evidence-boundary exceptions (secret refusal, duplicate paths) and
integrity drift propagate instead, leaving the session partially written for the owning caller
to finalize as failure rather than retrying the same case in place.

The supported grammar and JSONL protocol are pinned by golden fixtures under
`tests/fixtures/llama_bench/ca94157/` with a provenance note. Help and version-attempt
streams are captured from a local CPU build of the exact pinned commit. The upstream JSONL
example is byte-for-byte from that revision's README and carries both a `pp` and a `tg` row,
so it is a documentation-protocol fixture only and is not adapter-valid. The single-metric
golden is documentation-derived — that README `pp` row rebound to an absolute model path —
not a local model measurement, which requires a real gfx1151 build and model run. Synthetic
executables exercise orchestration; parser conformance is pinned by the single-metric golden
and upstream documentation fixture, and every fixture digest is locked by tests.

## llama-server adapter v1

The `llama-server` adapter is a ca94157-bound library boundary for one minimal
two-request smoke case. It consumes a caller-verified `llama-server` artifact from
a build with `LLAMA_BUILD_SERVER=ON`, checks exact `--version` and required
`--help` option grammar, binds only IPv4 loopback, waits for the pinned `/health`
loading/ready contract, and sends exactly two sequential non-streaming
`POST /completion` requests. Each request uses greedy seeded decoding, disabled
prompt-cache reuse, raw token return, and a fresh 128-bit sentinel in the prompt;
each response must return its own sentinel and exclude the other request's sentinel.
This proves request/response correlation, not absence of hidden model-state effects,
and the random sentinel intentionally means token IDs need not repeat across runs.

The long-lived child has explicit capability, readiness, per-request, and shutdown
deadlines. Its stdout and stderr are continuously stream-hashed with bounded retained
bytes; HTTP bodies are bounded before strict UTF-8/JSON parsing. Teardown polls before
signalling, sends one SIGTERM only to a live child process group, escalates to SIGKILL
after the shutdown deadline, reaps the child, and stops the nonblocking pipe collector.
Binary and model file identities are rechecked across every child/request boundary and
after shutdown. Integrity or evidence-boundary drift propagates without `sample.json`;
ordinary capability, lifecycle, transport, response, isolation, capture, and shutdown
failures produce terminal structured evidence. The adapter never finalizes the
caller-owned `RunSession` and defines no load generator, suite, model registry, CLI,
streaming client, retry policy, or generic service framework.

Server evidence is deliberately complete-local rather than portable in v1: bounded
malformed or non-UTF-8 response and process bytes must remain available for diagnosis,
while the portable evidence policy admits only its closed text/structured media set.
The adapter inventories each local artifact's digest and size; adding a binary-capable
portable role is a later evidence-protocol decision, not an adapter-local exception.

## test-backend-ops adapter v1

The `test-backend-ops` adapter is the correctness hard-gate runner adapter: a library
boundary that executes a verified `test-backend-ops` binary for exactly one filtered
operation set against one named backend, preserves bounded raw process evidence in an
active `RunSession`, and returns one versioned sample whether the child passes the gate,
fails it, exits nonzero, times out, cannot spawn, truncates output, or produces
unparseable output. Its only policy decision is fail-closed correctness. It defines no
suite manifest, operation-set policy, ranking, `perf`/`support`/`grad` mode,
SQL/console parser, `test-file` ingestion, arbitrary-command adapter, or run command;
later layers call it. It never acquires the GPU lock and never finalizes the
caller-owned `RunSession`.

This adapter consumes a **caller-provided verified binary**: the trusted caller supplies
and attests the executable, and a build profile that produces it must configure
`LLAMA_BUILD_TESTS=ON` (this milestone changes no build orchestration and adds no
profile). Its public surface is one typed case model, one verified-binary provenance
model, one capability model, one CSV row, one gate summary, one terminal sample model,
a pure command builder, pure help/list/CSV parsers and a pure gate function, and one
orchestration entry point. All v1 models are strict, frozen, `extra="forbid"`,
finite-number-checked Pydantic models reusing the repository `DashId`,
`AbsolutePathString`, and canonical JSON serializer. A case carries a one-to-64-character
dash-form id, one to 32 unique uppercase operation names in manifest order, one
non-empty `params_regex` bounded to 512 UTF-8 bytes with no C0/DEL controls (retained as
one argv data item and never translated through Python's regex engine), one
printable-ASCII backend selector of one to 128 characters compared exactly with the
observed CSV `backend_name`, and the fixed `test`/`csv`/`1` mode, output, and worker
values. The inputs model binds the fixed ca94157 source commit, a bounded build id, an
absolute binary path, and a lowercase binary SHA-256.

Executable identity reuses the ADAPTER-001 security primitive, extracted to a private
adapter-local module (`adapters/_executable_identity.py`) that stream-hashes a
non-symlink regular executable no-follow and asserts complete pre/post descriptor
metadata stability. Each adapter passes its own integrity-exception factory, so the
shared primitive preserves adapter-specific exception translation rather than imposing a
common exception type; nothing else (command construction, capability policy, outcome
classification, evidence layout, or sample finalization) is shared. The binary is hashed
against its asserted digest before discovery, re-required before every child, and
re-hashed after the terminal child; drift raises a typed integrity error and leaves no
`sample.json` because no truthful binding can then be claimed. Repeated path-identity
checks narrow but do not eliminate the small check-to-exec race, because the child is
launched by path rather than descriptor.

Capability discovery always attempts two bounded probes — `<binary> --help` then
`<binary> --list-ops` — through `run_process`, never a shell, with a caller-provided
complete environment (`inherit_env=False`), an explicit working directory, and a bounded
capability timeout; each retains 256 KiB per stream and never spools. The ca94157 help
probe is valid only when it exits **1**, its stdout is complete UTF-8, its stderr is
complete and empty, and its normalized nonempty lines match the pinned usage and
option-description grammar at whitespace-delimited token boundaries; only the single
program-path token after `Usage:` is normalized, so suffix, substring, reordered-option,
and weakened-choice lookalikes all fail, and free substring membership is never used. The
list probe is valid only when it exits 0, both streams are complete UTF-8, stderr is
empty, and the pure parser accepts the exact header/list/blank-line/total grammar with
exactly 128 unique operations and `Total: 128 operations`; any drift is a capability
failure rather than silent forward compatibility. Discovery also fails closed if any
requested operation is absent from the parsed list. Across both probes, process defects
are selected in the order capture, spawn, timeout, oversized, invalid UTF-8 (ties select
help before list-ops) and yield the corresponding terminal process status; otherwise an
exit/stream/grammar violation or an absent operation yields `capability-failed` with a
bounded refining reason. No test child runs on a capability failure, which still returns
a terminal sample after the final integrity check.

Only an allowlisted argv is constructed —
`<binary> test -o <comma-joined ops> -b <backend> -p <params_regex> --output csv -j 1` —
driven entirely by the typed case, so injection-shaped `params_regex` or backend values
remain single argv data items. The pure CSV parser uses the standard-library reader in
strict mode over complete UTF-8 stdout, never modifies the process-global field-size
limit, and translates every `csv.Error` (including the default field-limit failure) into
a domain parse error. It preserves upstream row order and rejects a missing, reordered,
duplicated, or extended header; malformed quoting, wrong column counts, blank records,
embedded record newlines, trailing content, or decoded C0/DEL controls in any field (one
ordinary terminal line ending is allowed); zero rows or more than 4096 rows; a non-`test`
mode, an unrequested operation, an empty or mismatched backend, a non-pinned `supported`
value, and duplicate full-identity rows. Structural parsing and the hard gate are
separate pure functions: a case passes only when the child has no process defect and
exits 0, parsing yields at least one row, every requested operation appears, every row is
supported (`"1"`), every `error_message` is empty, and every row binds the requested
backend and test mode. Unsupported, missing, or error-bearing rows are retained and
produce a terminal hard-gate failure, not a parser exception. Exact parseable stdout is
parsed even when the child exits nonzero and its rows are retained, but `child-failed`
takes precedence over parse and gate status. A process defect on one stream never
suppresses publication or parsing of independently exact output on the other.

Evidence is written with `RunSession.write_portable`, role `correctness`, under
`adapter/backend-ops/<case-id>/`: `capabilities/{help,list-ops}.{stdout.txt,stderr.txt}`
and `.process.json`, `capabilities/attempt.json`, `invocations/0001/{stdout.csv,
stderr.txt}` and `process.json`, and `sample.json` written last. A stream is exact and
publishable only when it is not truncated and re-encoding its returned text reproduces
both the recorded byte count and SHA-256; truncated or non-UTF-8 streams have no text
artifact while `process.json` still preserves their complete byte counts, digests,
truncation, and neutral category. Zero-byte exact streams are recorded only in
`process.json`, so no empty text blob collides across the `text/plain` and `text/csv`
media types. The terminal sample inventories every prior artifact with its relative path,
size, media type, role, and SHA-256; `sample.json` is excluded from its own inventory and
is authenticated by the enclosing run finalization/checksum contract. `sample.json` is
written last so it never claims evidence that was not durably accepted first. Terminal
status is one of `capability-failed`, `capture-failed`, `spawn-failed`, `timed-out`,
`oversized-output`, `encoding-failed`, `child-failed`, `parse-failed`, `hard-gate-failed`,
or `passed`; a bounded reason refines only a capability failure. Ordinary capability,
process, parser, and hard-gate failures return structured terminal samples; only
evidence-boundary refusals (secret refusal, duplicate paths) and binary-integrity drift
propagate, leaving the session for the owning caller to finalize.

The capability grammar, operation list, and CSV result parser are pinned by golden
fixtures under `tests/fixtures/backend_ops/ca94157/` with a provenance note. Help,
operation-list, and a filtered `ABS`/`type=f32` CSV run are captured from a local CPU
build of the exact pinned commit; the single `Usage:` program-path token is normalized to
a stable basename, matching what the capability parser observes. The CSV capture is a
CPU reference result and a parser/provenance fixture only — not a gfx1151 or HIP
correctness claim. Synthetic executables exercise orchestration; every fixture digest is
locked by tests, and no GPU, model, network, external checkout, or live ca94157 binary is
required.

## Verified model registry v1

The model registry is the first trust boundary above the runner adapters. It has three
explicit trust states and never fetches weights, keeps model bytes in memory beyond a
bounded hash, or copies model artifacts into StrixLab's home.

1. **unregistered** — a local GGUF was safely inspected but no registered manifest
   claims it (`ModelObservationV1`). This is practice evidence only and can never become
   a verified receipt.
2. **registered** — a `model` manifest pins upstream/local identity, exact size and
   SHA-256, and compatibility predicates. Local presence is not implied; a registered
   manifest may be checked in while its file is absent.
3. **verified** — the local primary artifact and every required sidecar matched the
   registered manifest and the stable-file checks, and the pinned inspector output
   passed (`ModelReceiptV1`).

`publishable` is a separate derived receipt property. Byte verification alone does not
turn an unknown quant recipe, an asserted opaque-sidecar relationship, or unknown
calibration into publishable quant-policy provenance: a receipt is `publishable` only
when compatibility is `verified`, every quant-policy provenance field is known, and a
measured bits-per-weight is present.

### Model manifest v1

The `model` kind reuses the manifest registry, alias grammar, `${...}` environment
resolution, sensitive-name rejection, and raw pre-resolution model exactly as `build`
does, so `${MODELS}` is validated before resolution and the resolved manifest is
validated again. `ModelManifestV1` is strict, finite, extra-forbid, and models two
variants selected by `registry_status`. A **registered** manifest requires base
repository/revision/license, an architecture, and artifact repository/revision/filename/
local-path/size/SHA-256, and forbids a draft reason. A **draft** requires a bounded
`draft_reason` and forbids local identity, receipt predicates, and sidecars; absent
identity stays absent rather than a placeholder. The literal `unknown` is legal only for
bounded quantization provenance fields. Sidecar ids and local paths are unique and never
alias the primary artifact. Metadata predicates use one strict shape: a bounded key, a
GGUF value type from the inspector's closed vocabulary, and exactly one of a
type-strict scalar value (no coercion, bool-as-int, or non-finite float) or a nonempty
nested array-type tuple compared by element type only.

### Source-compatible GGUF inspector

Inspection binds the pinned `gguf-py/gguf/scripts/gguf_dump.py … --json` tool to an
already-authenticated Git candidate rather than inventing another source-tree hash. The
caller holds a `sources.lease_source(...)` lease for the clean, unpatched
`strix-llama.cpp@ca94157f70a2776e8da6b6849b50b45a083d0478` preparation and supplies a
`GgufInspectorBindingV1` recording that lease's preparation/candidate/content-tree
identities and the interpreter's resolved-realpath and SHA-256. The script is resolved
only beneath the leased worktree at its exact relative path and SHA-256, the lease's base
commit is required with no patches or dirty status, and `lease.verify()` runs before and
after the child. The model is opened once no-follow, validated as an owned regular file,
bound `fstat`-to-`lstat`, hashed, and retained through inspection; the child receives
only `/proc/self/fd/<fd>` through a new `run_process(pass_fds=…)` parameter, a fixed
module-level bootstrap validates its argv shape and inserts the verified `gguf-py` root
before executing the verified script under `runpy`, and the environment is a small
constructed set (`LANG`/`LC_ALL`/`TZ` plus private scratch `HOME`/`TMPDIR`) with
`inherit_env=False`. Interpreter, lease, descriptor, and pathname bindings are rechecked
after inspection.

`run_process` gains separate optional hard total-byte ceilings for stdout and stderr:
the complete chunk that crosses one is still counted and hashed, then the process group
is terminated and the outcome is `CAPTURE_FAILED` with `capture_error`
`stdout:hard-limit-exceeded` or `stderr:hard-limit-exceeded`, and any partial spool is
aborted rather than published. For the inspector, stdout has a 64 MiB ceiling and stderr
256 KiB. Inspector stdout is validated strictly as UTF-8 from the captured raw spool
bytes, its exact byte count and SHA-256 are recorded, and its JSON is normalized into a
portable projection that drops the input filename and positional index/offset fields but
keeps endianness, each metadata key/type/scalar-or-array-type, and every tensor's
name/shape/type. Duplicate JSON keys, non-finite numbers, absolute-path strings, and
output beyond the bound are rejected. Spawn, timeout, nonzero-exit, limit, UTF-8, JSON,
and shape defects are typed verification failures, never partial receipts.

### Stable file identity, cache, and receipts

The model-file identity primitive (no-follow regular-file opens/lstats comparing device,
inode, size, and modify/change times across every boundary) is shared, not copied per
adapter. Full-file SHA-256 work is cached under `<home>/cache/model-hashes/v1/`, keyed by
a canonical digest of absolute path plus the complete stable identity; a hit is honored
only after a fresh no-follow stat matches every identity field, and a cold miss acquires
a per-key advisory lock with a bounded wait (`wait_for_exclusive_lock`, default 300 s;
expiry raises `ModelCacheBusyError` without hashing), rechecks, hashes once, and publishes
a crash-safe record. Concurrent cold verifiers therefore serialize to one hash; a
malformed, unsafe, or divergent entry is a typed cache-integrity failure. The cache is an
optimization only: manifest size/SHA mismatch always fails and identity is rechecked after
the inspector.

The complete normalized metadata projection is a content-addressed local registry
artifact under `<home>/models/metadata/v1/<sha256>.json`; the receipt records its digest
and relative path. Local receipt envelopes are canonical JSON under
`<home>/models/receipts/v1/<manifest-id>/<local-receipt-sha256>.json` with private,
crash-safe, no-replace publication; identical content is reusable and divergent content
at one address is an integrity failure. Digest domains are explicit canonical-JSON
objects (`strixlab.model-hash-cache-key.v1`, `strixlab.gguf-metadata.v1`,
`strixlab.model-receipt-evidence.v1`, `strixlab.model-receipt.v1`, and the manifest digest
over `strixlab.model-manifest.v1` plus the resolved, validated manifest dump); no model
carries its own digest, and loads reject a filename/content-digest mismatch. Neither local
registry file is portable evidence by itself and neither contains model bytes.
`ModelLease` is separately a process-local, non-serializable handle owning the descriptor,
its `/proc/self/fd` path, and verification callbacks.

Artifact compatibility requires the primary GGUF architecture metadata to match the
closed v1 family map (`qwen3_5` requires the exact `general.architecture` STRING `qwen35`;
an unmapped family is a typed failure), nonzero tensors, every declared predicate, every
required sidecar, and every GGUF sidecar's own predicates. A hash-only sidecar is
byte-verified but sets compatibility to `asserted` and forces `publishable` false.
Execution requirements are recorded as unverified for later build/model binding and do
not contribute to compatibility or publishability here.

The checked-in smoke manifests (`configs/models/qwen35-2b-smoke.yaml`,
`qwen35-4b-smoke.yaml`) pin immutable public Qwen base and bartowski GGUF revisions,
sizes, and SHA-256s reviewed against `tests/fixtures/models/smoke-provenance.json`; they
reference `${MODELS}` paths and are never downloaded during build, test, or verification.
Three decision models ship as explicit drafts with no fabricated local identity.

### Adapter migration

`LlamaBenchInputsV1` and `LlamaServerInputsV1` retain `model_id`, `model_path`, and
`model_sha256`, change `model_digest_status` from `asserted` to `verified`, and add
`model_receipt_sha256` plus an embedded compact receipt-evidence projection (a
`schema_version`-discriminated union) whose canonical digest equals `model_receipt_sha256`,
so exported sample evidence independently substantiates what `verified` means after the
local registry disappears. New verification issues `ModelReceiptEvidenceV2`, which adds an
authenticated `ModelExecutionProjectionV1` of the manifest's execution requirements
(`verification_status` and the bounded, unique `required_sources`/`required_features`
tuples) so a downstream consumer can fail closed on a non-empty requirement set without
re-reading the model manifest. The legacy `ModelReceiptEvidenceV1` shape (no execution
projection) is retained as a reader, so receipts published before this pre-release change
remain readable and authentic; adapters and `require_receipt_inputs_match` accept either
version. Each runner
receives the corresponding `ModelReceiptV1`, binds it to the inputs, and holds
`lease_verified_model` across the complete child lifetime, passing the lease's
`/proc/self/fd/<fd>` operand and inherited descriptor to every child so each opens the
receipt-bound inode even if the pathname is swapped mid-run. Only the model-path operand
changes; all flags, order, capability grammar, request JSON, parsing, status vocabulary,
and evidence layout are unchanged, and the llama-bench parser compares the tool-reported
filename against the effective descriptor operand. `ModelLease.verify()` gates every
terminal `sample.json`; context-manager exit repeats it defensively. Adapters never rehash
multi-gigabyte model bytes per sample. This is an intentional in-place V1 migration: the
repository is pre-release and has no supported persisted adapter samples.

## Deterministic smoke suite v1

The smoke suite is the first user-facing boundary that composes the three ca94157
adapters into one immutable run. It is a thin orchestration milestone: the adapters keep
owning every child process, capability probe, parser, raw stream, stable-executable
check, model lease, and per-case `sample.json`; the suite adds no comparison statistics,
ranking, candidate pairing, profiler integration, generic workflow engine, adapter
plugin registry, downloader, build creation, or model verification. A later CLI polish
milestone may add name resolution without changing the suite library.

A strict `suite` manifest v1 is registered in the manifest registry and checked in as
`configs/suites/smoke-qwen35.yaml` with a generated `suite.schema.json`. It prefers
typed sections (`build`, `correctness.backend_ops`, `correctness.greedy`, `performance`,
`timeouts`) over a generic step list, so the exact deterministic prompt corpus is
manifest data captured verbatim in `manifest.input.yaml`/`manifest.resolved.yaml` with no
second prompt registry. Field validation bounds prompt count/text, operation count, case
count, repetition/window counts, and timeouts, and requires ordered unique prompt and
case ids. After field validation, aggregate limits reject a run before allocation: at
most 4 prompts and 8 performance cases; at most 128 total adapter invocations across
correctness, warmups, and measurements; at most 512 total benchmark repetitions under
`cases * (warmup_runs + measurement_windows * repetitions_per_window)`; and at most 32
KiB of aggregate prompt text with each prompt also within the server adapter's 16 KiB
UTF-8 limit. The deterministically generated adapter case ids are cross-validated against
the adapters' dash-id and 64-byte length contracts and must be collision-free. The
checked-in manifest is a policy template, not proof that `ROCm0` exists on any machine;
the backend adapter's runtime capability and hard-gate evidence decides that.

The CLI takes explicit paths and addresses — `strixlab run suite <suite.yaml> --machine
<machine.yaml> --build <BUILD_ID> --model-receipt <LOCAL_RECEIPT_SHA256> [--server-port]
[--home]` — never an ambiguous "latest" object, and performs no implicit config-name
search. The supplied machine profile must have the manifest's machine id, and
`load_model_receipt(manifest.model, local_receipt_sha256, home=...)` re-authenticates the
explicit local receipt and binds its exact model id. The adapter-facing portable receipt
digest is `receipt_evidence_digest(receipt.evidence)`, distinct from the local
receipt-envelope address used by `load_model_receipt`. As an intentional pre-release
change, new verification issues `ModelReceiptEvidenceV2`, which carries an authenticated
`ModelExecutionProjectionV1` (the manifest's `verification_status` plus its bounded, unique
`required_sources`/`required_features` tuples), populated in `_build_evidence` so the
receipt/evidence digests bind it; the legacy v1 evidence shape is retained as a reader.
SUITE-001 fails closed before `begin_run` on a legacy v1 receipt (it cannot prove
requirements) and on any v2 receipt whose requirement tuples are non-empty — emptiness is
authenticated, never inferred. The checked-in smoke manifests declare no requirements;
supporting a non-empty set is a future milestone. The suite still binds the exact
receipt/model id and enforces the manifest's explicit source, toolchain, and gfx
requirements.

`build_cache.py` gains a small read-only `BuildLease`/`lease_build`, analogous in shape to
the source lease but keyed on the existing build-ID lock. It acquires that exact lock,
calls the existing locked inspection primitive directly (never public `inspect_build`,
which would try to reacquire the held lock), and yields only when the build is `PRESENT`,
attested, and fully verified; the lock is held for the context lifetime so `cleanup_build`
and materialization cannot race the run. At acquisition it captures the strict resolved
root path and its no-follow directory device/inode; `verify()` reruns the locked
inspection (re-verifying the canonical record, digest, and root artifacts) and additionally
rejects a symlink or directory replacement by requiring the same root device/inode. From
the leased canonical artifact inventory the suite resolves exactly one regular ELF
executable for each of `test-backend-ops`, `llama-server`, and `llama-bench`, beneath the
leased root, using the recorded artifact SHA-256; the adapters still re-hash each binary.
Before allocating a run it requires the source evidence to name the manifest's source id
and exact ca94157 commit, the toolchain mode and gfx-target selection to match, and all
three targets to be present.

Child environments are reconstructed from the leased canonical build environment tuple,
never from ambient `os.environ`. Names must be unique and match the environment-name
grammar; every value is split and rejoined at `os.pathsep` boundaries, an exact
`{BUILD_ROOT}` component (or one beginning `{BUILD_ROOT}` + `os.sep`) is rehydrated with
the leased root, and canonical `HOME`/`TMPDIR` are replaced with separate directories
beneath one fresh mode-0700 temporary root. A missing `HOME`/`TMPDIR`, a residual
`{SOURCE_ROOT}`/`{BUILD_HOME}`/`{BUILD_TMP}` or any other placeholder-shaped component, a
NUL, a duplicate name, or a wrong `LANG`/`LC_ALL`/`TZ` value fails closed; all other
authenticated entries (including `PATH`, ROCm path lists, and `SOURCE_DATE_EPOCH`) are
retained byte-for-byte. The temporary root is scratch, not evidence, and is removed after
the adapters stop, including on failure.

The high-level executor owns `begin_run`, the machine-profile exclusive lock, the build
lease, the protocol, `suite/result.json`, and finalization; the adapters keep their
caller-owned `RunSession` contract and never finalize. The global acquisition order is
build lease, then run session, then machine lock; release is machine lock, run
finalization, then build lease. Manifests are validated and the receipt authenticated
before the build lease; the build is bound under the held lease before a run is allocated.
Immediately after `begin_run`, before the machine lock, the executor writes canonical
portable snapshots `suite/build.json` (role `build`), `suite/model.json` (role
`environment`), and `suite/machine.json` (role `environment`, the validated profile and
its canonical digest, not live telemetry); `environment` is the normative v1 role for an
authenticated runtime input. A lock refusal becomes a structured failed result and a
finalized failure run. `BuildLease.verify()` is called before leaving every structured
path after acquisition, including machine-lock refusal and, on adapter paths, before
releasing the machine lock.

The correctness-first protocol runs one `BackendOpsCaseV1` (passing only when the sample
status is `passed` with a present, passing gate), then one `LlamaServerCaseV1` per ordered
greedy prompt (passing only when the adapter status is `success`, both nonce-isolated
temperature-zero responses exist, each has at least one token, and their exact token-ID
tuples are equal). It stops before performance on the first correctness failure; this is a
hard gate, not a score, and never compares against another build/model arm. A pure planner
then expands the single-arm `windowed-interleaved-v1` schedule — warmup round 1 across all
cases in order, warmup round 2, then measurement windows 1..N across all cases — with
`repetitions=1` per warmup and `repetitions_per_window` per measurement, and a distinct
evidence namespace per case so JUDGE-001 can pair case/window coordinates later without
statistics. It stops on the first warmup or measurement failure; warmup samples are raw
evidence excluded from the measurement projection, and each successful measurement
window/case contributes the adapter's normalized `samples_ts`, average, and standard
deviation plus a compact sample reference, with no pooling, intervals, outlier removal,
ranking, or regression labels.

Compact strict/frozen/finite v1 runtime models live in `suites.py`, including an
authenticated input reference for build/model/machine, a compact adapter sample reference
(adapter, phase, case id, logical path, canonical sample SHA-256, persistence, adapter
status, and category), backend-op and greedy-parity verdicts, planned/completed warmup and
measurement counts, one measurement projection per successful window/case, and a terminal
`passed`/`failed` status with a closed reason (`passed`, `lock-unavailable`,
`backend-ops-failed`, `greedy-parity-failed`, `warmup-failed`, `measurement-failed`, or
`integrity-failed`). A closed mapping folds every declared backend/server/bench status
(and the bench closed `reason`) into a fixed set of suite categories, never copying
free-form adapter reason text; any future unmapped status is `adapter-failed` and fails
closed, so a new adapter status cannot silently pass. The sample-reference digest is the
shared portable blob SHA-256 —
`SHA-256(canonical_json_bytes(sample.model_dump(mode="json")))`. Every adapter publishes
its terminal `sample.json` as a portable entry at the predictable logical path
(`adapter/backend-ops/<case>/sample.json`, `adapters/llama-server/<case>/sample.json`,
`adapters/llama-bench/<case>/sample.json`), and the suite authenticates each sample only
against its content-addressed portable blob — a local-only write fails closed. Because the
`llama-server` tree also carries binary response bytes, that adapter additionally keeps a
byte-identical local copy of `sample.json` beside those siblings (so the run checksums
cover it), publishing the same canonical bytes both ways after its `lease.verify()`. A
returned failure sample counts as completed once its portable blob authenticates; a thrown
adapter integrity exception or a failed authentication does not, and yields
`integrity-failed`. Planned counts always describe the full deterministic schedule;
completed counts describe only returned/authenticated samples.

`suite/result.json` (role `summary`) is written exactly once at the end of every
structured adapter/verdict path, while the machine lock is still held. Ordinary adapter
failure samples, token-parity failures, and authenticated sample-digest failures lead to a
complete result and `RunOutcome.FAILURE`; a fully clean protocol leads to
`RunOutcome.SUCCESS`. A backend case passes only when the adapter status maps to success
and its gate is present and passed, so a tampered success-status sample with a failed gate
is rejected. The single mode-0700 scratch root is removed on every exit including adapter
failure; a deletion failure is not swallowed but escapes so the run finalizes failure
without a successful `suite/result.json`. Failure to publish an input snapshot or the
result itself is an evidence-store integrity exception, as is any unexpected exception: it
is never concealed and never claims a structured result, and `RunSession.__exit__`
performs its fail-safe finalization without a `suite/result.json`. The suite executor
takes narrow,
explicit test seams — the `begin_run` clock/token factories plus a temporary-directory
factory, a machine-lock factory, and the three adapter callables, all defaulting to
production implementations — making lock refusal, filesystem identity changes, cleanup,
and exact call order deterministic in unit tests without a GPU, ROCm, model weights,
network, or real binaries.

## Offline comparison judge v1

The comparison judge (`strixlab compare BASELINE_RUN_ID CANDIDATE_RUN_ID`) is the first
boundary above the smoke suite. It is purely offline: it never reruns an adapter, acquires
the GPU lock, inspects a live binary or model, or mutates either arm. It authenticates two
**distinct, finalized, successful** smoke-suite runs, compares their matched throughput
samples conservatively, and finalizes one immutable comparison run carrying a canonical
JSON report and a Markdown rendering of it. Comparing a run to itself is rejected; a no-op
comparison is two independently finalized runs, which may share the same build. A
statistical `regression`, `mixed`, or `inconclusive` report is still a *successfully
executed* comparison run — the evidence-run outcome and the statistical verdict are
separate concepts, and the required no-op result is `inconclusive`, never a fabricated win.

### Authenticated suite snapshot

`load_finalized_suite_snapshot(run_id, *, home)` is the one reusable, descriptor-anchored
seam the judge uses to obtain an immutable, fully re-authenticated view of one finalized
successful run. It calls `inspect_run` (requiring `RunOutcome.SUCCESS` and retaining the
authenticated `record-sha256` digest), then reads the resolved manifest and every portable
blob through the shared `read_record_member` primitive — owned, no-follow, identity-checked
reads that reject a symlink or same-uid inode swap during the read — and rebinds each byte
to its content address. It parses `manifest.resolved.yaml` strictly as `SuiteManifestV1` and
requires its bytes to equal `canonical_yaml_bytes` of that model's dump; authenticates the
single `suite/result.json` summary entry and requires its bytes to equal `canonical_json_bytes`
of the strictly parsed `SuiteResultV1`; binds the result's suite/machine/model IDs, passed
status, correctness projections, and planned-equals-completed schedule to the manifest;
authenticates the three input snapshots (`suite/build.json`, `suite/model.json`,
`suite/machine.json`) and binds their payload IDs and digests to the result; and recomputes
**every** correctness and measurement projection from the authenticated terminal adapter
samples (`BackendOpsSampleV1`, `LlamaServerSampleV1`, `LlamaBenchSampleV1`), requiring the
stored `BackendOpsVerdictV1`, each `GreedyVerdictV1` prompt, and every measurement
projection to equal the recomputation exactly. `result.samples` must be the exact ordered
schedule of a passed run (backend correctness, ordered greedy checks, ordered warmups, then
ordered measurements) with no extra, missing, duplicated, or orphan reference, and the
measurement coordinates must be exactly one projection per `(case_id, window)` with
`repetitions_per_window` positive finite samples each. A canonical but semantically misbound
record fails closed, and a tampered blob is rejected by the immutable record verifier before
the snapshot loads.

### Equivalence, statistics, and verdict

Two arms are comparable only when their run IDs are distinct, their canonical resolved-manifest
bytes are identical, their suite/machine/model IDs and their machine and model input-payload
digests match (the build input may differ), their ordered measurement coordinates and
repetition counts match exactly, and each case carries at least `MIN_PAIRED_SAMPLES = 5`
paired samples. Build IDs, build-record digests, sample digests, and measurements may differ:
this milestone compares two executions of the *same resolved suite*, never arbitrary
compatible-looking workloads.

Samples are paired by `(case_id, window, repetition_index)` in manifest order and compared
only within the same performance case (higher is better); cases are never pooled and no
cross-case score is invented. For positive finite pairs, the judge computes the paired
log-delta mean with `math.fsum`, the arithmetic means, `speed_ratio = exp(mean_d)`, and
`delta_percent = 100·expm1(mean_d)`. The `paired-log-bootstrap-v1` interval is exactly 4096
replicate means and the R-7 95% percentile interval; each replicate resamples the paired log
deltas with replacement, selecting indices from SHA-256 over a length-framed
`strixlab.judge.bootstrap.v1` domain binding both arms' record digests, the case ID, and the
zero-based replicate and draw indexes, so the interval is deterministic and bound to these two
runs and this case only — it describes matched positions in these two runs, not run-to-run or
population uncertainty. Baseline noise is `1.4826·MAD` of the baseline log values. The
per-case verdict is `inconclusive` when the interval includes zero or `|mean_d|` is within the
baseline noise, `improvement` when the interval is wholly above zero, and `regression`
otherwise; the overall verdict is the exact projection (`inconclusive`/`improvement`/`regression`
when uniform, else `mixed`). Every arithmetic domain/overflow failure, a nonpositive or
non-finite value, and any report-model validation failure caused by the derived statistics is
translated to a typed statistics error before any run is allocated.

### Immutable comparison run and reports

Before allocation the judge completely loads and re-authenticates both snapshots, checks
equivalence, computes the statistics, instantiates the strict frozen report models
(`ComparisonRequestV1`, `ComparisonArmV1`, `CaseComparisonV1`, `ComparisonReportV1`; each
recomputes its verdict and validates every delta/CI/noise percentage against its log form with
a single `rel_tol=abs_tol=1e-12` tolerance), renders the JSON and Markdown, and preflights the
exact two portable outputs against the member, aggregate, entry, path, file-count, and
single-media-per-blob rules. No arm payloads are copied, so capacity is bounded. The
experiment ID is `compare-` plus the first 24 hex characters of SHA-256 over the canonical
request JSON; the canonical request YAML is the captured input and its dump the resolved
input. The run writes exactly `comparison/report.json` (`application/json`, role `comparison`)
and `comparison/report.md` (`text/markdown`, role `comparison`) and finalizes
`RunOutcome.SUCCESS` for every valid report. A load, equivalence, or statistics failure is a
pre-allocation error that creates no run; a publication or integrity failure after allocation
finalizes failure and surfaces the run ID and immutable record.

The Markdown is a pure rendering of the validated report — identities and digests, policy, the
scope warning, the overall verdict, and one stable table row per case — ending in exactly one
newline with no environment-derived free text. Crucially, a comparison report authenticates
its source run IDs, record digests, result digests, and the shared resolved-manifest digest,
but it does **not** copy either arm's evidence tree. A comparison bundle is therefore a
portable *derived* report, not a standalone proof of its arms: to verify the arms offline,
export the two source-run bundles as well. The judge reuses the existing bundle system rather
than building a second bundler.

## Bounded profile-guided campaigns

Campaigns are the next local layer above the smoke suite and offline judge. A
campaign freezes one source, build, machine, model, suite, and judge policy,
then evaluates a finite reviewed patch list. It does not download models,
provision ROCm, interpret TOPK payloads, or push upstream. ROCm 10 is not a
prerequisite.

The procedure is hypothesis → fixed evaluator → patch → screen → fresh
confirmation → retain/reject. Next campaign is a new reviewed plan informed by
`campaign report`; old campaigns are never rewritten. Failed, negative,
inconclusive, and interrupted findings are retained. The existing `compare`
contract is unchanged. Campaigns add cross-arm greedy token parity (v1 is
launch/layout-preserving only) and a frozen objective/protected-regression
gate. Optional `objective_cases` defaults to all performance cases; every
objective must have existing `improvement`; every remaining case is
automatically protected with `percent_ci_low >=
-protected_regression_margin_percent` (default `0`, finite `0 <= m < 100`).
Both screening AB/BA comparisons and both fresh confirmation AB/BA
comparisons must pass those gates plus token parity. The raw judge verdict
stays as-is, including `mixed`. The campaign label is
`objective_met_provisional`, not judge improvement and not `best-known`.
Per-case intervals are not simultaneous or campaign-level confidence, and
the protected-case bound is not a formal noninferiority proof. Calibration
is unchanged.

A whole phase is reserved before suite work. Reserved slots stay spent on
failure, crash, or interruption. An interrupted candidate is spent and
terminal; resume continues untouched candidates and does not replay it.
Interrupted calibration stops the campaign. Some lower-layer failures never
return an authenticatable run ID; the phase records that a failure-evidence
link is unavailable rather than claiming every failure is linked. The
evaluator is frozen at create and rechecked on resume; drift fails closed.
There is no campaign JSON Schema registration; `create` uses a strict
`CampaignPlanV1` parser.

### v1 command surface

`strixlab campaign create PLAN.yaml [--home PATH]` freezes a plan and prints
canonical JSON; it does not run suites. `strixlab campaign resume ID [--home
PATH]` evaluates the finite remainder on hardware and prints JSON.
`strixlab campaign inspect ID [--home PATH]` re-validates evidence links and
prints JSON. `strixlab campaign report ID [--home PATH]` prints Markdown from
that verified state. There are no phase-separating flags.

Required plan fields: `schema_version` `1`, dash-id `id`, relative paths
`suite`, `machine`, `source`, `build`, `model`, `candidates` as `{id,
patches}` (ids unique and not `baseline`), `max_candidates` (`>=` list
length, `<= 100`), and `max_suite_runs` (`>= 0`, `<= 10000`; zero stops
before preparation). Optional `objective_cases` and
`protected_regression_margin_percent` as above. Paths are relative to the
plan file. Each candidate patches the same unmodified frozen source commit.
v1 patches may modify existing unrenamed `ggml/src` source files
(`.c`/`.cc`/`.cpp`/`.h`/`.hpp`/`.cu`/`.cuh`/`.hip`) and must not edit
tests, build files, inspector files, or the evaluator. Duplicate ordered
patch identities are rejected at create. Durable state is
`<home>/campaigns/<id>/state.json` with `frozen.json` and copied patches.

Calibration is two baseline/baseline suite runs plus one `inconclusive`
token-matching comparison. Screening is four AB/BA suite runs per candidate.
Confirmation, if screening is eligible, is four fresh suite runs and reuses
the screened candidate build. `max_suite_runs` is a count budget, not a
wall-clock deadline. Guaranteed complete capacity for `N` candidates is
`2 + 8N` suite runs.

The demonstration plan is
[`configs/campaigns/historical-mmvq-demo.yaml`](../configs/campaigns/historical-mmvq-demo.yaml):
a historical known-negative MMVQ patch, not a suggested optimization.

### Implementation plan

1. **Core controller** — freeze-only `create`, hardware `resume`, JSON
   `inspect`, Markdown `report`.
2. **Problem portfolio** — [research-problems.md](research-problems.md).
   Docs only; no structured catalog.
3. **Reviewer / verification** — existing judge plus campaign-only token
   parity and frozen objective/protected gates; preserve mixed verdicts.
4. **Later hardware experiments** — post-merge pilot in the problem
   portfolio. Not implied by landing the controller.

Command examples and the full procedure live in
[autoresearch.md](autoresearch.md).

## Versioning and schemas

Manifest dispatch is by external kind plus integer `schema_version`. Version 1 is
the only accepted version. Future versions receive parallel models and schemas;
StrixLab never silently migrates an input.

Canonical JSON Schema 2020-12 resources are distributed inside the Python wheel
with IDs of the form `urn:strixlab:schema:{kind}:1`. Checked-in schema bytes are
generated deterministically and verified by tests.

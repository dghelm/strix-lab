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

## Versioning and schemas

Manifest dispatch is by external kind plus integer `schema_version`. Version 1 is
the only accepted version. Future versions receive parallel models and schemas;
StrixLab never silently migrates an input.

Canonical JSON Schema 2020-12 resources are distributed inside the Python wheel
with IDs of the form `urn:strixlab:schema:{kind}:1`. Checked-in schema bytes are
generated deterministically and verified by tests.

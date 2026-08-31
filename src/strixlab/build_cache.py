"""Durable cache, materialization journal, inspection, and cleanup for builds."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from strixlab.build_artifacts import BuildArtifactsV1, verify_artifact_capture
from strixlab.build_identity import ROOT_PLACEHOLDERS, IdentityEntry, ToolObservation
from strixlab.build_paths import build_storage_roots, is_unsafe_directory, prepare_storage_tree
from strixlab.builds import (
    AttemptOutcome,
    VerifiedAttemptRecord,
    recipe_index_record_digest,
    verify_attempt_record,
)
from strixlab.locks import exclusive_lock
from strixlab.secure_fs import (
    directory_open_flags,
    ensure_directory_fsynced,
    exclusive_create_flags,
    fsync_directory,
    readonly_open_flags,
    rename_noreplace,
    try_open_owned_directory,
    write_all,
    write_exclusive,
)
from strixlab.serialization import canonical_json_bytes

_BUILD_RE = re.compile(r"^build-sha256:[0-9a-f]{64}$")
_ATTEMPT_RE = re.compile(r"^attempt-[0-9a-f]{24}-[0-9a-f]{32}$")
_RECORD_RE = re.compile(r"^record-sha256:[0-9a-f]{64}$")
_OWNER_LIMIT = 4 * 1024
_MODEL_LIMIT = 64 * 1024 * 1024


class BuildCacheError(RuntimeError):
    """Cache state is unsafe, inconsistent, corrupt, or busy."""


class CacheClassification(StrEnum):
    MISS = "miss"
    HIT = "hit"
    REHYDRATE = "rehydrate"


class MaterializationState(StrEnum):
    VACANT = "vacant"
    BUILDING = "building"
    PUBLISHING = "publishing"
    PRESENT = "present"
    CLEANING = "cleaning"
    CLEANED = "cleaned"
    REHYDRATING = "rehydrating"
    DISCARDING = "discarding"


class _StoredModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class BuildRootOwnerV1(_StoredModel):
    schema_version: Literal[1] = 1
    attempt_id: str = Field(pattern=_ATTEMPT_RE.pattern)
    build_id: str = Field(pattern=_BUILD_RE.pattern)
    root_device: int = Field(ge=0)
    root_inode: int = Field(gt=0)


class IdentityEntryV1(_StoredModel):
    name: str
    value: str


class ToolObservationV1(_StoredModel):
    role: str
    path: str
    realpath: str
    mode: int = Field(ge=0, le=0o7777)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    version_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    search_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SourceBlobRefV1(_StoredModel):
    schema_version: Literal[1] = 1
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class SourcePatchRefV1(_StoredModel):
    schema_version: Literal[1] = 1
    order: int = Field(ge=1)
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class SourceReproducerV1(_StoredModel):
    schema_version: Literal[1] = 1
    candidate_id: str = Field(pattern=r"^candidate-sha256:[0-9a-f]{64}$")
    content_tree_id: str = Field(pattern=r"^content-tree-sha256:[0-9a-f]{64}$")
    snapshot_id: str = Field(pattern=r"^snapshot-sha256:[0-9a-f]{64}$")
    source_evidence: dict[str, Any]
    source_evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_manifest: dict[str, Any]
    diff: SourceBlobRefV1 | None
    patches: tuple[SourcePatchRefV1, ...]


class BuildIdentityProjectionV1(_StoredModel):
    """The reproducible identity shared by every invocation of one build ID.

    Everything here must match the canonical record before a cache hit or a
    rehydration may reuse the materialized artifacts; the producer attempt ID
    and the artifact evidence itself are intentionally excluded.
    """

    schema_version: Literal[1] = 1
    recipe_id: str = Field(pattern=r"^recipe-sha256:[0-9a-f]{64}$")
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    toolchain_mode: Literal["host", "rocm"]
    environment: tuple[IdentityEntryV1, ...]
    requested_targets: tuple[str, ...]
    selections: tuple[IdentityEntryV1, ...]
    tools: tuple[ToolObservationV1, ...]
    source: SourceReproducerV1


class CanonicalBuildRecordV1(BuildIdentityProjectionV1):
    build_id: str = Field(pattern=_BUILD_RE.pattern)
    producer_attempt_id: str = Field(pattern=_ATTEMPT_RE.pattern)
    artifacts: BuildArtifactsV1

    def identity(self) -> BuildIdentityProjectionV1:
        # Derive the projection from the shared base fields so a new projection
        # field can never silently drop out of identity comparison.
        return BuildIdentityProjectionV1(
            **{name: getattr(self, name) for name in BuildIdentityProjectionV1.model_fields}
        )


class EvidenceRefV1(_StoredModel):
    schema_version: Literal[1] = 1
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class ProducerProvenanceV1(_StoredModel):
    """Immutable, pre-publication binding of one producer attempt to its build."""

    schema_version: Literal[1] = 1
    build_id: str = Field(pattern=_BUILD_RE.pattern)
    producer_attempt_id: str = Field(pattern=_ATTEMPT_RE.pattern)
    recipe_id: str = Field(pattern=r"^recipe-sha256:[0-9a-f]{64}$")
    artifact_set_id: str = Field(pattern=r"^artifact-set-sha256:[0-9a-f]{64}$")
    candidate_id: str = Field(pattern=r"^candidate-sha256:[0-9a-f]{64}$")
    snapshot_id: str = Field(pattern=r"^snapshot-sha256:[0-9a-f]{64}$")
    execution_class: Literal["built"]
    evidence: tuple[EvidenceRefV1, ...]


# How a canonical was first genuinely completed. A rehydration reuses an already
# attested canonical, so it is never an attestation class of its own.
AttestationClass = Literal["built", "recovered"]


class BuildAttestationV1(_StoredModel):
    """Post-finalization attestation that a canonical build was genuinely completed.

    A canonical record and its pre-publication producer provenance are written
    before the producing attempt finalizes; a crash in that window can leave a
    PRESENT root whose producer never claimed SUCCESS. This immutable attestation
    is the explicit post-finalization boundary: it names the finalized SUCCESS
    attempt (the producer itself for a normal build, or a distinct recovery attempt
    for a crash-forward completion) that authenticates the canonical digest and the
    successful/recovery outcome. A materialized root is only reusable as a cache HIT
    once such an attestation exists and its attestor is authenticated.
    """

    schema_version: Literal[1] = 1
    build_id: str = Field(pattern=_BUILD_RE.pattern)
    canonical_record_sha256: str = Field(pattern=_RECORD_RE.pattern)
    attestor_attempt_id: str = Field(pattern=_ATTEMPT_RE.pattern)
    execution_class: AttestationClass
    artifact_set_id: str = Field(pattern=r"^artifact-set-sha256:[0-9a-f]{64}$")
    # Exact content-addresses of the immutable producer and attestor attempt records,
    # authenticated at publication and required to match at reuse, so a
    # self-consistent replacement of either record fails closed.
    producer_record_sha256: str = Field(pattern=_RECORD_RE.pattern)
    attestor_record_sha256: str = Field(pattern=_RECORD_RE.pattern)


class BuildIndexV1(_StoredModel):
    schema_version: Literal[1] = 1
    build_id: str = Field(pattern=_BUILD_RE.pattern)
    canonical_record_sha256: str = Field(pattern=_RECORD_RE.pattern)
    producer_attempt_id: str = Field(pattern=_ATTEMPT_RE.pattern)


class MaterializationEvidenceV1(_StoredModel):
    """The materializing attempt's own validated artifact evidence for a root.

    A cache hit, inspection, or cleanup verifies the current root against *this*
    evidence, not the canonical producer's, so a rehydration whose non-identity
    observations (CMake cache, compile database, inspections, capture tools) differ
    from the original producer still authenticates its own materialized root. The
    canonical record and its immutable producer provenance are never mutated; the
    artifact-set ID here always equals the canonical's (enforced at publish).
    """

    schema_version: Literal[1] = 1
    build_id: str = Field(pattern=_BUILD_RE.pattern)
    attempt_id: str = Field(pattern=_ATTEMPT_RE.pattern)
    canonical_record_sha256: str = Field(pattern=_RECORD_RE.pattern)
    artifacts: BuildArtifactsV1


class MaterializationEventV1(_StoredModel):
    schema_version: Literal[1] = 1
    build_id: str = Field(pattern=_BUILD_RE.pattern)
    attempt_id: str = Field(pattern=_ATTEMPT_RE.pattern)
    sequence: int = Field(gt=0)
    previous_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    from_state: MaterializationState | None
    to_state: MaterializationState
    canonical_record_sha256: str | None = Field(default=None, pattern=_RECORD_RE.pattern)
    staging_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    materialization_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    root_device: int | None = Field(default=None, ge=0)
    root_inode: int | None = Field(default=None, gt=0)


class MaterializationRegistryV1(_StoredModel):
    schema_version: Literal[1] = 1
    build_id: str = Field(pattern=_BUILD_RE.pattern)
    attempt_id: str = Field(pattern=_ATTEMPT_RE.pattern)
    state: MaterializationState
    sequence: int = Field(gt=0)
    last_event_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_record_sha256: str | None = Field(default=None, pattern=_RECORD_RE.pattern)
    staging_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    materialization_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    root_device: int | None = Field(default=None, ge=0)
    root_inode: int | None = Field(default=None, gt=0)


@dataclass(frozen=True, slots=True)
class CacheLookup:
    classification: CacheClassification
    canonical: CanonicalBuildRecordV1 | None
    canonical_record_sha256: str | None
    owner: BuildRootOwnerV1 | None = None
    # A HIT whose PRESENT root is fully verified but whose canonical is not yet
    # attested (a crash-forward completion): the reusing attempt must publish a
    # recovery attestation before treating the build as a genuine cache hit.
    needs_attestation: bool = False


@dataclass(frozen=True, slots=True)
class BuildInspection:
    build_id: str
    state: MaterializationState
    root: Path | None
    canonical: CanonicalBuildRecordV1
    canonical_record_sha256: str
    # False for a PRESENT crash-forward root whose canonical is not yet attested:
    # observable, but explicitly not a fully verified, reusable, or cleanable build.
    attested: bool


@dataclass(frozen=True, slots=True)
class BuildCleanupResult:
    build_id: str
    state: MaterializationState
    record: Path


@dataclass(frozen=True, slots=True)
class BuildLease:
    """A read-only, lock-holding handle to one PRESENT, attested canonical build.

    Analogous in shape to :class:`~strixlab.sources.SourceLease`: the exact existing
    per-build-ID lock is held for the whole lease context, so ``cleanup_build`` and
    any materialization transition cannot race a run that leased the build. The fields
    mirror the canonical build authenticated at acquisition, and :meth:`verify`
    re-authenticates it under the still-held lock via the locked inspection primitive,
    additionally binding the leased root's no-follow device+inode captured at
    acquisition so a symlink or directory replacement fails closed.
    """

    build_id: str
    root: Path
    canonical: CanonicalBuildRecordV1
    canonical_record_sha256: str
    verify_callback: Callable[[], None]

    def verify(self) -> None:
        """Re-authenticate the leased build under the held lock; fail closed on drift."""

        self.verify_callback()


@dataclass(frozen=True, slots=True)
class _Layout:
    home: Path
    root: Path
    materialized: Path
    success: Path
    indexes: Path
    journals: Path
    staging: Path
    locks: Path


@dataclass(slots=True)
class BuildCacheSession:
    layout: _Layout
    build_id: str
    attempt_id: str

    @property
    def root(self) -> Path:
        return self.layout.materialized / self.build_id

    def lookup(self, identity: BuildIdentityProjectionV1, *, home: Path) -> CacheLookup:
        registry = _load_verified_registry(self.layout, self.build_id)
        if registry is None:
            record, _payload, _digest, index = _load_canonical(self.layout, self.build_id)
            # lexists, not exists: a dangling-symlink root is corrupt state, not a MISS.
            if record is None and index is None and not os.path.lexists(self.root):
                return CacheLookup(CacheClassification.MISS, None, None)
            raise BuildCacheError("build cache has state without a materialization journal")
        if registry.state in _RECOVERABLE_STATES:
            registry = _recover_incomplete(self.layout, registry)
        if registry.state is MaterializationState.VACANT:
            record, _payload, _digest, index = _load_canonical(self.layout, self.build_id)
            if record is None and index is None and not os.path.lexists(self.root):
                return CacheLookup(CacheClassification.MISS, None, None)
            raise BuildCacheError("vacant build cache has retained materialization state")
        # After recovery a terminal materialization must have a consistent pair.
        canonical, record_sha = _load_canonical_pair(self.layout, self.build_id)
        if canonical is None or record_sha is None:
            raise BuildCacheError("materialized build has no canonical success record")
        if registry.canonical_record_sha256 != record_sha:
            raise BuildCacheError("materialization journal changed canonical record binding")
        if canonical.identity() != identity:
            raise BuildCacheError("cache identity diverged from the current invocation")
        # Complete canonical/provenance validation gates every reuse; the verified
        # producer record is reused below for a same-attempt attestor.
        verified_producer = _verify_canonical_producer(home, canonical)
        if registry.state is MaterializationState.CLEANED:
            if os.path.lexists(self.root):
                raise BuildCacheError("cleaned build unexpectedly has a materialized root")
            return CacheLookup(CacheClassification.REHYDRATE, canonical, record_sha)
        if registry.state is not MaterializationState.PRESENT:
            raise BuildCacheError("materialization journal has an impossible terminal state")
        # Preserve the exact journal-authenticated owner so success finalization can
        # revalidate against it (attempt/build/uid/dev/inode), not a self-consistent
        # marker.
        owner = _verify_owner_against_journal(self.root, registry)
        _verify_materialized_root(self.layout, self.root, registry, canonical)
        # A cache HIT is reusable only once a genuine completion is attested. A
        # missing attestation is a crash-forward completion (canonical PRESENT but
        # its producer never finalized SUCCESS): the reusing attempt must publish a
        # recovery attestation before success, so signal that here rather than
        # returning an ordinary reusable HIT prematurely.
        attestation = _load_attestation(self.layout, self.build_id)
        if attestation is None:
            return CacheLookup(
                CacheClassification.HIT, canonical, record_sha, owner, needs_attestation=True
            )
        attestor = _resolve_attestor(
            home, canonical, attestation.attestor_attempt_id, verified_producer
        )
        _authenticate_attestor(canonical, record_sha, attestation, verified_producer, attestor)
        return CacheLookup(CacheClassification.HIT, canonical, record_sha, owner)

    def begin_materialization(self, *, rehydrate: bool) -> None:
        registry = _load_registry(self.layout, self.build_id)
        state = MaterializationState.REHYDRATING if rehydrate else MaterializationState.BUILDING
        if rehydrate:
            if registry is None or registry.state is not MaterializationState.CLEANED:
                raise BuildCacheError("rehydration requires a cleaned materialization")
        elif registry is not None and registry.state is not MaterializationState.VACANT:
            raise BuildCacheError("new build requires vacant materialization state")
        _transition(self.layout, self.build_id, self.attempt_id, state)

    def bind_root(self, owner: BuildRootOwnerV1) -> None:
        registry = _required_registry(self.layout, self.build_id)
        if registry.state not in {
            MaterializationState.BUILDING,
            MaterializationState.REHYDRATING,
        }:
            raise BuildCacheError("build root cannot be bound in the current state")
        if owner.attempt_id != self.attempt_id or owner.build_id != self.build_id:
            raise BuildCacheError("build-root owner binding is inconsistent")
        _transition(
            self.layout,
            self.build_id,
            self.attempt_id,
            registry.state,
            root_device=owner.root_device,
            root_inode=owner.root_inode,
        )

    def publish(
        self,
        record: CanonicalBuildRecordV1,
        *,
        rehydrate: bool,
    ) -> str:
        if record.build_id != self.build_id:
            raise BuildCacheError("canonical record is bound to another build")
        if rehydrate:
            existing, existing_digest = _load_canonical_pair(self.layout, self.build_id)
            if existing is None or existing_digest is None:
                raise BuildCacheError("rehydration lost its canonical success record")
            # The approved rehydrate equivalence is exactly the artifact-set ID plus
            # the canonical requested-target/CMake-selection identity (carried by
            # identity()). Non-identity capture observations (CMake cache, compile
            # database, capture tools, inspection digests) are validated intrinsically
            # by capture and against the retained root below, and must not reject a
            # legitimate rehydrate.
            if (
                existing.identity() != record.identity()
                or existing.artifacts.artifact_set_id != record.artifacts.artifact_set_id
            ):
                raise BuildCacheError(
                    "rehydrated artifact identity diverged from canonical evidence"
                )
            # Verify the freshly rebuilt root against the REHYDRATING attempt's own
            # validated evidence (record.artifacts), not the original producer's
            # non-identity observations. The artifact-set ID equivalence above already
            # binds the actual binaries to the canonical; this materialization's own
            # CMake-cache/compile-db/inspection observations are what a later HIT will
            # enforce, via the durable materialization-evidence binding below.
            _verify_record_artifacts(self.root, record)
            registry = _required_registry(self.layout, self.build_id)
            materialization_sha256 = _write_materialization_evidence(
                self.layout, self.build_id, self.attempt_id, existing_digest, record.artifacts
            )
            _transition(
                self.layout,
                self.build_id,
                self.attempt_id,
                MaterializationState.PRESENT,
                canonical_record_sha256=existing_digest,
                materialization_sha256=materialization_sha256,
                root_device=registry.root_device,
                root_inode=registry.root_inode,
            )
            return existing_digest
        if record.producer_attempt_id != self.attempt_id:
            raise BuildCacheError("canonical record producer is not the publishing attempt")
        payload = canonical_json_bytes(record.model_dump(mode="json"))
        digest = _record_digest(payload)
        stage = _stage_path(self.layout, self.build_id, self.attempt_id)
        write_exclusive(stage, payload, 0o400)
        fsync_directory(stage.parent)
        registry = _required_registry(self.layout, self.build_id)
        _transition(
            self.layout,
            self.build_id,
            self.attempt_id,
            MaterializationState.PUBLISHING,
            staging_sha256=hashlib.sha256(payload).hexdigest(),
            root_device=registry.root_device,
            root_inode=registry.root_inode,
        )
        _publish_canonical(self.layout, record, payload, digest)
        materialization_sha256 = _write_materialization_evidence(
            self.layout, self.build_id, self.attempt_id, digest, record.artifacts
        )
        _transition(
            self.layout,
            self.build_id,
            self.attempt_id,
            MaterializationState.PRESENT,
            canonical_record_sha256=digest,
            staging_sha256=hashlib.sha256(payload).hexdigest(),
            materialization_sha256=materialization_sha256,
            root_device=registry.root_device,
            root_inode=registry.root_inode,
        )
        return digest


_LEGAL: dict[MaterializationState | None, frozenset[MaterializationState]] = {
    None: frozenset({MaterializationState.BUILDING}),
    MaterializationState.VACANT: frozenset({MaterializationState.BUILDING}),
    MaterializationState.BUILDING: frozenset(
        {
            MaterializationState.BUILDING,
            MaterializationState.PUBLISHING,
            MaterializationState.DISCARDING,
        }
    ),
    MaterializationState.PUBLISHING: frozenset(
        {MaterializationState.PRESENT, MaterializationState.DISCARDING}
    ),
    MaterializationState.PRESENT: frozenset({MaterializationState.CLEANING}),
    MaterializationState.CLEANING: frozenset({MaterializationState.CLEANED}),
    MaterializationState.CLEANED: frozenset({MaterializationState.REHYDRATING}),
    MaterializationState.REHYDRATING: frozenset(
        {
            MaterializationState.REHYDRATING,
            MaterializationState.PRESENT,
            MaterializationState.DISCARDING,
        }
    ),
    MaterializationState.DISCARDING: frozenset(
        {MaterializationState.VACANT, MaterializationState.CLEANED}
    ),
}

_RECOVERABLE_STATES = frozenset(
    {
        MaterializationState.BUILDING,
        MaterializationState.REHYDRATING,
        MaterializationState.PUBLISHING,
        MaterializationState.DISCARDING,
        MaterializationState.CLEANING,
    }
)


@contextlib.contextmanager
def build_cache_session(
    build_id: str, attempt_id: str, *, home: Path
) -> Iterator[BuildCacheSession]:
    """Hold the nested build-ID lock beneath the caller's recipe lock."""

    _validate_build_id(build_id)
    _validate_attempt_id(attempt_id)
    layout = _layout(home, create=True)
    lock = layout.locks / f"build-id-{build_id.removeprefix('build-sha256:')}.lock"
    with exclusive_lock(lock) as held:
        if not held.acquired:
            raise BuildCacheError(held.reason or "build cache lock is unavailable")
        yield BuildCacheSession(layout, build_id, attempt_id)


# The allowlisted required pre-publication evidence a genuine producer inventory
# must contain, derived from the build flow. Fixed logical items are matched by
# exact relative path; variable families (File API replies, per-role tool output,
# per-process metadata) are matched by a directory prefix requiring at least one
# member. compile_commands is present-or-explicitly-absent.
_REQUIRED_EVIDENCE_PATHS = frozenset(
    {
        "build/artifacts.json",
        "build/profile.resolved.json",
        "build/environment.json",
        "source/evidence.json",
        "source/snapshot.json",
    }
)
_REQUIRED_EVIDENCE_PREFIXES = ("cmake/file-api/", "tools/", "process/")
_REQUIRED_COMPILE_COMMANDS = ("cmake/compile_commands.json", "cmake/compile_commands.absent.json")


def _verify_producer_provenance(
    canonical: CanonicalBuildRecordV1, verified: VerifiedAttemptRecord
) -> None:
    """Authenticate the canonical build's producer provenance and required evidence.

    ``verified`` is the already-authenticated immutable producer attempt record
    (content-addressed, verified lock-free to avoid a recipe→build-ID lock
    inversion). Its ``provenance.json`` must bind the build/recipe/artifact-set/
    snapshot/candidate identity and ``built`` execution class, and its declared
    digest/size inventory must be validated against the record's authenticated
    manifest — never re-hashed. Every declared entry must be non-duplicate and match
    the manifest exactly (a safe, known path with an equal digest/size); the
    allowlisted required set (artifacts, profile, environment, source evidence,
    snapshot manifest, at least one configure cache, a compile-database observation,
    at least one File API reply, tool and process observations, and every canonical
    source reproducer byte) must be covered. The canonical artifacts and source
    references are bound to that authenticated inventory.

    This authenticates the producer's own durable pre-publication evidence only; the
    SUCCESS/recovery outcome and canonical-digest binding are established separately
    by the post-finalization :class:`BuildAttestationV1` (see ``_authenticate_
    attestor``), so a legitimate crash-forward canonical whose producer never
    finalized SUCCESS is still authenticated by a distinct recovery attestor.
    """

    provenance = _parse_provenance(verified.path / "build" / "provenance.json")
    if (
        provenance.build_id != canonical.build_id
        or provenance.producer_attempt_id != canonical.producer_attempt_id
        or provenance.recipe_id != canonical.recipe_id
        or provenance.artifact_set_id != canonical.artifacts.artifact_set_id
        or provenance.candidate_id != canonical.source.candidate_id
        or provenance.snapshot_id != canonical.source.snapshot_id
        or provenance.execution_class != "built"
    ):
        raise BuildCacheError("producer provenance does not bind this canonical build")
    # Validate each declared evidence entry against the record verifier's already
    # authenticated manifest — no blob is re-read. Duplicate declarations, unsafe or
    # unknown paths (absent from the manifest), and digest/size divergence all fail.
    authenticated = {entry.path: (entry.sha256, entry.size_bytes) for entry in verified.files}
    inventory: dict[str, tuple[str, int]] = {}
    for ref in provenance.evidence:
        if ref.relative_path in inventory:
            raise BuildCacheError("producer provenance inventory has a duplicate entry")
        if authenticated.get(ref.relative_path) != (ref.sha256, ref.size_bytes):
            raise BuildCacheError("producer evidence diverged from its provenance inventory")
        inventory[ref.relative_path] = (ref.sha256, ref.size_bytes)
    _require_complete_inventory(inventory)
    # Bind the exact canonical artifacts to the producer's authenticated evidence.
    artifacts_bytes = canonical_json_bytes(canonical.artifacts.model_dump(mode="json"))
    if inventory.get("build/artifacts.json") != (
        hashlib.sha256(artifacts_bytes).hexdigest(),
        len(artifacts_bytes),
    ):
        raise BuildCacheError("canonical artifacts are not in the producer inventory")
    source_refs: list[SourceBlobRefV1 | SourcePatchRefV1] = []
    if canonical.source.diff is not None:
        source_refs.append(canonical.source.diff)
    source_refs.extend(canonical.source.patches)
    for source_ref in source_refs:
        if inventory.get(source_ref.relative_path) != (source_ref.sha256, source_ref.size_bytes):
            raise BuildCacheError("canonical source reproducer is not in the producer inventory")


def _verify_canonical_producer(
    home: Path, canonical: CanonicalBuildRecordV1
) -> VerifiedAttemptRecord:
    """Authenticate the producer attempt record once and validate its provenance.

    Returns the verified record so a same-attempt attestor can reuse it without
    re-hashing the immutable record.
    """

    verified = verify_attempt_record(canonical.producer_attempt_id, home=home)
    _verify_producer_provenance(canonical, verified)
    return verified


def _require_complete_inventory(inventory: dict[str, tuple[str, int]]) -> None:
    """Reject a producer inventory missing any allowlisted required evidence."""

    missing = _REQUIRED_EVIDENCE_PATHS - inventory.keys()
    if missing:
        raise BuildCacheError("producer provenance inventory is missing required evidence")
    for prefix in _REQUIRED_EVIDENCE_PREFIXES:
        if not any(path.startswith(prefix) for path in inventory):
            raise BuildCacheError("producer provenance inventory is missing required evidence")
    if not any(path.startswith("cmake/") and path.endswith("-cache.txt") for path in inventory):
        raise BuildCacheError("producer provenance inventory is missing a configure cache")
    if sum(path in inventory for path in _REQUIRED_COMPILE_COMMANDS) != 1:
        raise BuildCacheError("producer provenance inventory lacks a compile-database observation")


def _resolve_attestor(
    home: Path,
    canonical: CanonicalBuildRecordV1,
    attestor_attempt_id: str,
    producer: VerifiedAttemptRecord,
) -> VerifiedAttemptRecord:
    """Resolve the attestor's verified record, reusing the producer when it is the
    attestor (normal build) and verifying a distinct recovery attestor exactly once.

    Threading the result through :func:`_authenticate_attestor` avoids re-hashing a
    distinct attestor record at both the resolve and authenticate steps.
    """

    if attestor_attempt_id == canonical.producer_attempt_id:
        return producer
    return verify_attempt_record(attestor_attempt_id, home=home)


def _authenticate_attestor(
    canonical: CanonicalBuildRecordV1,
    canonical_digest: str,
    attestation: BuildAttestationV1,
    producer: VerifiedAttemptRecord,
    attestor: VerifiedAttemptRecord,
) -> None:
    """Authenticate the finalized SUCCESS attempt that attests a canonical build.

    ``producer`` and ``attestor`` are already-verified immutable records (the
    attestor resolved by :func:`_resolve_attestor`; it is the producer itself for a
    normal build). The attestor record must be finalized ``SUCCESS`` for the expected
    build ID, its immutable ``build/result.json`` must bind the canonical digest,
    build ID, and execution class, and both record content-addresses must match the
    attestation's digest anchors — so a self-consistent replacement of either record
    fails closed.
    """

    if (
        attestation.build_id != canonical.build_id
        or attestation.canonical_record_sha256 != canonical_digest
        or attestation.artifact_set_id != canonical.artifacts.artifact_set_id
    ):
        raise BuildCacheError("build attestation does not bind this canonical build")
    # Digest-anchor the producer record: a self-consistent replacement of the
    # producer evidence and its manifest has a different content-address and fails.
    if attestation.producer_record_sha256 != producer.record_sha256:
        raise BuildCacheError("producer record digest diverged from its attestation anchor")
    # Digest-anchor the attestor record likewise, so a replaced recovery attestor
    # record fails closed even if it is internally self-consistent.
    if attestation.attestor_record_sha256 != attestor.record_sha256:
        raise BuildCacheError("attestor record digest diverged from its attestation anchor")
    if (
        attestor.registry.outcome is not AttemptOutcome.SUCCESS
        or attestor.registry.build_id != canonical.build_id
    ):
        raise BuildCacheError("build attestor attempt did not finalize SUCCESS for this build")
    result = _parse_attestor_result(attestor.path / "build" / "result.json")
    if (
        result.get("build_id") != canonical.build_id
        or result.get("canonical_record_sha256") != canonical_digest
        or result.get("execution_class") != attestation.execution_class
        or result.get("artifact_set_id") != canonical.artifacts.artifact_set_id
    ):
        raise BuildCacheError("build attestor result does not bind this canonical build")


def _parse_attestor_result(path: Path) -> dict[str, Any]:
    try:
        parsed: Any = json.loads(_read_bytes(path, _MODEL_LIMIT))
    except (OSError, ValueError) as exc:
        raise BuildCacheError("build attestor result is unreadable") from exc
    if not isinstance(parsed, dict):
        raise BuildCacheError("build attestor result is malformed")
    return parsed


def _parse_provenance(path: Path) -> ProducerProvenanceV1:
    try:
        return ProducerProvenanceV1.model_validate_json(
            _read_bytes(path, _MODEL_LIMIT), strict=True
        )
    except ValidationError as exc:
        raise BuildCacheError("producer provenance is invalid") from exc


@dataclass(frozen=True, slots=True)
class _InspectedBuild:
    registry: MaterializationRegistryV1
    canonical: CanonicalBuildRecordV1
    digest: str
    owner: BuildRootOwnerV1 | None
    attested: bool


def _inspect_build_locked(layout: _Layout, build_id: str) -> _InspectedBuild:
    """Verify one build's journal, record, owned root, and attestation once.

    Both inspectable states are post-canonical, so the journal's bound canonical
    digest must equal the loaded record digest. The attestation is authenticated
    when present; a PRESENT root without one is a crash-forward, recovery-pending
    state (``attested=False``) that callers must not treat as fully verified or
    destructively clean. A CLEANED build is always attested — a missing attestation
    there is corruption.
    """

    registry = _required_registry(layout, build_id)
    canonical, digest = _load_canonical_pair(layout, build_id)
    if canonical is None or digest is None:
        raise BuildCacheError("canonical build record is missing")
    if registry.canonical_record_sha256 != digest:
        raise BuildCacheError("materialization journal changed canonical record binding")
    verified_producer = _verify_canonical_producer(layout.home, canonical)
    owner: BuildRootOwnerV1 | None = None
    if registry.state is MaterializationState.PRESENT:
        root = layout.materialized / build_id
        owner = _verify_owner_against_journal(root, registry)
        _verify_materialized_root(layout, root, registry, canonical)
    elif registry.state is not MaterializationState.CLEANED:
        raise BuildCacheError("build materialization is not inspectable")
    attestation = _load_attestation(layout, build_id)
    if attestation is None and registry.state is MaterializationState.CLEANED:
        raise BuildCacheError("cleaned build is missing its attestation")
    if attestation is not None:
        attestor = _resolve_attestor(
            layout.home, canonical, attestation.attestor_attempt_id, verified_producer
        )
        _authenticate_attestor(canonical, digest, attestation, verified_producer, attestor)
    return _InspectedBuild(registry, canonical, digest, owner, attestation is not None)


def inspect_build(build_id: str, *, home: Path) -> BuildInspection:
    """Verify and inspect one exact canonical build."""

    _validate_build_id(build_id)
    layout = _layout(home, create=False)
    lock = layout.locks / f"build-id-{build_id.removeprefix('build-sha256:')}.lock"
    with exclusive_lock(lock) as held:
        if not held.acquired:
            raise BuildCacheError(held.reason or "build cache lock is unavailable")
        inspected = _inspect_build_locked(layout, build_id)
        root = (
            layout.materialized / build_id
            if inspected.registry.state is MaterializationState.PRESENT
            else None
        )
        return BuildInspection(
            build_id,
            inspected.registry.state,
            root,
            inspected.canonical,
            inspected.digest,
            inspected.attested,
        )


def _leased_root_identity(root: Path) -> tuple[int, int]:
    """Capture one leased PRESENT root's no-follow device+inode, failing closed.

    Opens the root ``O_DIRECTORY | O_NOFOLLOW`` so a symlink or non-directory cannot
    be followed, then rejects any foreign-owned or non-directory entry.
    """

    try:
        descriptor = os.open(root, _DIR_OPEN_FLAGS)
    except OSError as exc:
        raise BuildCacheError("leased build root is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if is_unsafe_directory(metadata):
        raise BuildCacheError("leased build root is unsafe")
    return metadata.st_dev, metadata.st_ino


def _require_present_attested(inspected: _InspectedBuild) -> None:
    if inspected.registry.state is not MaterializationState.PRESENT:
        raise BuildCacheError("leased build is not present")
    if not inspected.attested:
        raise BuildCacheError("leased build is recovery-pending and not attested")


@contextlib.contextmanager
def lease_build(build_id: str, *, home: Path) -> Iterator[BuildLease]:
    """Hold the build-ID lock while yielding one PRESENT, attested canonical build.

    Acquires the exact existing per-build lock, verifies the build through the locked
    inspection primitive (never public :func:`inspect_build`, which would try to
    reacquire the held lock), and yields only a fully verified, attested PRESENT build.
    The lock is held for the whole context, so ``cleanup_build`` and any
    materialization transition cannot race a run that holds the lease.
    """

    _validate_build_id(build_id)
    layout = _layout(home, create=False)
    lock = layout.locks / f"build-id-{build_id.removeprefix('build-sha256:')}.lock"
    with exclusive_lock(lock) as held:
        if not held.acquired:
            raise BuildCacheError(held.reason or "build cache lock is unavailable")
        inspected = _inspect_build_locked(layout, build_id)
        _require_present_attested(inspected)
        root = layout.materialized / build_id
        acquired_device, acquired_inode = _leased_root_identity(root)

        def verify() -> None:
            reinspected = _inspect_build_locked(layout, build_id)
            _require_present_attested(reinspected)
            if (
                reinspected.digest != inspected.digest
                or reinspected.canonical != inspected.canonical
            ):
                raise BuildCacheError("leased build canonical record changed")
            current_device, current_inode = _leased_root_identity(root)
            if current_device != acquired_device or current_inode != acquired_inode:
                raise BuildCacheError("leased build root identity changed")

        yield BuildLease(
            build_id=build_id,
            root=root,
            canonical=inspected.canonical,
            canonical_record_sha256=inspected.digest,
            verify_callback=verify,
        )


def cleanup_build(build_id: str, *, home: Path) -> BuildCleanupResult:
    """Remove one exact verified build root while retaining immutable evidence."""

    _validate_build_id(build_id)
    layout = _layout(home, create=False)
    lock = layout.locks / f"build-id-{build_id.removeprefix('build-sha256:')}.lock"
    with exclusive_lock(lock) as held:
        if not held.acquired:
            raise BuildCacheError(held.reason or "build cache lock is unavailable")
        inspected = _inspect_build_locked(layout, build_id)
        registry, digest, owner = inspected.registry, inspected.digest, inspected.owner
        if registry.state is MaterializationState.CLEANED:
            return BuildCleanupResult(build_id, registry.state, _record_path(layout, build_id))
        # A PRESENT-but-unattested root is a crash-forward completion pending a
        # recovery attestation; refuse to destructively remove it as if it were an
        # ordinary finalized build. The root and evidence are preserved.
        if not inspected.attested:
            raise BuildCacheError("refusing to clean a recovery-pending unattested build")
        assert owner is not None  # PRESENT builds always resolve an owner
        _transition(
            layout,
            build_id,
            owner.attempt_id,
            MaterializationState.CLEANING,
            canonical_record_sha256=digest,
            root_device=owner.root_device,
            root_inode=owner.root_inode,
        )
        shutil.rmtree(layout.materialized / build_id)
        fsync_directory(layout.materialized)
        _transition(
            layout,
            build_id,
            owner.attempt_id,
            MaterializationState.CLEANED,
            canonical_record_sha256=digest,
        )
        return BuildCleanupResult(
            build_id, MaterializationState.CLEANED, _record_path(layout, build_id)
        )


def write_build_root_owner(root: Path, attempt_id: str, build_id: str) -> BuildRootOwnerV1:
    metadata = root.lstat()
    if is_unsafe_directory(metadata):
        raise BuildCacheError(f"build root is unsafe: {root}")
    owner = BuildRootOwnerV1(
        attempt_id=attempt_id,
        build_id=build_id,
        root_device=metadata.st_dev,
        root_inode=metadata.st_ino,
    )
    write_exclusive(
        root / ".strixlab-owner.json",
        canonical_json_bytes(owner.model_dump(mode="json")),
        0o400,
    )
    return owner


def verify_build_root_owner(
    root: Path, expected: BuildRootOwnerV1 | None = None
) -> BuildRootOwnerV1:
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise BuildCacheError("owned build root is unavailable") from exc
    if is_unsafe_directory(metadata):
        raise BuildCacheError("build-root ownership changed")
    owner = _read_owner(root)
    if metadata.st_dev != owner.root_device or metadata.st_ino != owner.root_inode:
        raise BuildCacheError("build-root ownership changed")
    if expected is not None and owner != expected:
        raise BuildCacheError("build-root ownership changed")
    return owner


def remove_owned_build_root(root: Path | None, owner: BuildRootOwnerV1 | None) -> None:
    if root is None or owner is None:
        return
    verify_build_root_owner(root, owner)
    shutil.rmtree(root)
    fsync_directory(root.parent)


def _layout(home: Path, *, create: bool) -> _Layout:
    roots = build_storage_roots(home)
    layout = _Layout(
        roots.home,
        roots.root,
        roots.materialized,
        roots.success_records,
        roots.build_indexes,
        roots.materializations,
        roots.staging,
        roots.locks,
    )
    prepare_storage_tree(roots, create=create, validate=_validate_cache_directory)
    return layout


def _validate_cache_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise BuildCacheError("StrixLab build cache does not exist") from exc
    if is_unsafe_directory(metadata):
        raise BuildCacheError(f"unsafe build cache directory: {path}")


def _carry_forward[CarriedT: (str, int)](
    new: CarriedT | None, previous: MaterializationRegistryV1 | None, attr: str
) -> CarriedT | None:
    """Return ``new`` when set, else carry ``previous.attr`` forward (or None)."""

    if new is not None:
        return new
    if previous is None:
        return None
    carried: CarriedT | None = getattr(previous, attr)
    return carried


def _transition(
    layout: _Layout,
    build_id: str,
    attempt_id: str,
    state: MaterializationState,
    *,
    canonical_record_sha256: str | None = None,
    staging_sha256: str | None = None,
    materialization_sha256: str | None = None,
    root_device: int | None = None,
    root_inode: int | None = None,
) -> MaterializationRegistryV1:
    previous = _load_registry(layout, build_id)
    from_state = previous.state if previous is not None else None
    if state not in _LEGAL.get(from_state, frozenset()):
        raise BuildCacheError(f"illegal materialization transition: {from_state} -> {state}")
    sequence = 1 if previous is None else previous.sequence + 1
    event = MaterializationEventV1(
        build_id=build_id,
        attempt_id=attempt_id,
        sequence=sequence,
        previous_sha256=previous.last_event_sha256 if previous else None,
        from_state=from_state,
        to_state=state,
        canonical_record_sha256=_carry_forward(
            canonical_record_sha256, previous, "canonical_record_sha256"
        ),
        staging_sha256=_carry_forward(staging_sha256, previous, "staging_sha256"),
        materialization_sha256=_carry_forward(
            materialization_sha256, previous, "materialization_sha256"
        ),
        root_device=_carry_forward(root_device, previous, "root_device"),
        root_inode=_carry_forward(root_inode, previous, "root_inode"),
    )
    payload, digest = _event_bytes(event)
    journal = _journal(layout, build_id)
    events = _ensure_journal_events(journal)
    # Publish the event durably and atomically (temp + no-replace rename) so a
    # torn write can never leave a partial event; current.json follows.
    _publish_event(events, sequence, payload)
    registry = _registry_from_event(event, digest)
    _atomic_write(journal / "current.json", canonical_json_bytes(registry.model_dump(mode="json")))
    return registry


def _event_bytes(event: MaterializationEventV1) -> tuple[bytes, str]:
    payload = canonical_json_bytes(event.model_dump(mode="json", by_alias=True))
    return payload, hashlib.sha256(payload).hexdigest()


def _event_links(
    event: MaterializationEventV1,
    *,
    build_id: str,
    sequence: int,
    previous: str | None,
    from_state: MaterializationState | None,
) -> bool:
    """Authenticate that ``event`` is the exact next link in the journal chain."""

    return (
        event.build_id == build_id
        and event.sequence == sequence
        and event.previous_sha256 == previous
        and event.from_state is from_state
    )


def _registry_from_event(event: MaterializationEventV1, digest: str) -> MaterializationRegistryV1:
    return MaterializationRegistryV1(
        build_id=event.build_id,
        attempt_id=event.attempt_id,
        state=event.to_state,
        sequence=event.sequence,
        last_event_sha256=digest,
        canonical_record_sha256=event.canonical_record_sha256,
        staging_sha256=event.staging_sha256,
        materialization_sha256=event.materialization_sha256,
        root_device=event.root_device,
        root_inode=event.root_inode,
    )


def _publish_event(events: Path, sequence: int, payload: bytes) -> None:
    destination = events / f"{sequence:08d}.json"
    temporary = events / f".{sequence:08d}.json.{secrets.token_hex(8)}.tmp"
    try:
        write_exclusive(temporary, payload, 0o400)
        rename_noreplace(temporary, destination)
        fsync_directory(events)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _verify_event_chain_at(events_fd: int, registry: MaterializationRegistryV1) -> None:
    # Enumerate the directory and require exactly the contiguous committed chain
    # [1..sequence]. Any higher-sequence extra, gap, unexpected filename, or leftover
    # orphan (reconciliation adopts at most one, then this must see none) fails
    # closed. Orphan adoption runs before this in _load_registry.
    present = set(os.listdir(events_fd))
    expected = {f"{sequence:08d}.json" for sequence in range(1, registry.sequence + 1)}
    if present != expected:
        raise BuildCacheError("materialization event directory has unexpected entries")
    previous: str | None = None
    state: MaterializationState | None = None
    terminal: MaterializationEventV1 | None = None
    for sequence in range(1, registry.sequence + 1):
        event = _read_model_at(events_fd, f"{sequence:08d}.json", MaterializationEventV1)
        _payload, digest = _event_bytes(event)
        if not _event_links(
            event,
            build_id=registry.build_id,
            sequence=sequence,
            previous=previous,
            from_state=state,
        ):
            raise BuildCacheError("materialization event chain is inconsistent")
        if event.to_state not in _LEGAL.get(state, frozenset()):
            raise BuildCacheError("materialization event chain has an illegal transition")
        previous = digest
        state = event.to_state
        terminal = event
    if previous != registry.last_event_sha256 or state is not registry.state:
        raise BuildCacheError("materialization registry diverged from its event chain")
    if terminal is None:
        raise BuildCacheError("materialization registry has no events")
    # The terminal event's carried bindings must equal the registry's. Any orphan
    # beyond the terminal sequence was already rejected by the enumeration above.
    if (
        terminal.attempt_id != registry.attempt_id
        or terminal.canonical_record_sha256 != registry.canonical_record_sha256
        or terminal.staging_sha256 != registry.staging_sha256
        or terminal.materialization_sha256 != registry.materialization_sha256
        or terminal.root_device != registry.root_device
        or terminal.root_inode != registry.root_inode
    ):
        raise BuildCacheError("terminal event diverged from registry bindings")


def _verify_owner_against_journal(
    root: Path, registry: MaterializationRegistryV1
) -> BuildRootOwnerV1:
    """Authenticate the owned root against its full journal-bound owner identity."""

    if registry.root_device is None or registry.root_inode is None:
        raise BuildCacheError("build-root owner diverged from its journal binding")
    # BuildRootOwnerV1's fields are exactly the journal-bound owner identity, so the
    # full-model equality contract in verify_build_root_owner authenticates every
    # binding (attempt/build/device/inode) without a hand-maintained field list.
    expected = BuildRootOwnerV1(
        attempt_id=registry.attempt_id,
        build_id=registry.build_id,
        root_device=registry.root_device,
        root_inode=registry.root_inode,
    )
    return verify_build_root_owner(root, expected)


def _complete_publishing(
    layout: _Layout,
    registry: MaterializationRegistryV1,
    root: Path,
    record: CanonicalBuildRecordV1,
    payload: bytes,
    digest: str,
) -> MaterializationRegistryV1:
    """Finish an interrupted publication: verify the owned root and artifacts,
    publish the canonical record/index no-replace, and reach PRESENT.

    Each caller performs its own distinct input authentication (a present record
    is checked for a foreign producer and against the staged digest; a staged
    record is authenticated inside ``_load_staged_record``) before delegating this
    identical tail.
    """

    owner = _verify_owner_against_journal(root, registry)
    _verify_record_artifacts(root, record)
    _publish_canonical(layout, record, payload, digest)
    # The recovered root was produced by this canonical's own producer, so its
    # materialization evidence is the canonical artifacts. Record the durable
    # binding before PRESENT so later HIT/inspect verify against it uniformly.
    materialization_sha256 = _write_materialization_evidence(
        layout, registry.build_id, registry.attempt_id, digest, record.artifacts
    )
    return _transition(
        layout,
        registry.build_id,
        registry.attempt_id,
        MaterializationState.PRESENT,
        canonical_record_sha256=digest,
        materialization_sha256=materialization_sha256,
        root_device=owner.root_device,
        root_inode=owner.root_inode,
    )


def _recover_incomplete(
    layout: _Layout, registry: MaterializationRegistryV1
) -> MaterializationRegistryV1:
    root = layout.materialized / registry.build_id
    if registry.state is MaterializationState.PUBLISHING:
        record, payload, digest, index = _load_canonical(layout, registry.build_id)
        if record is not None and payload is not None and digest is not None:
            # A canonical record exists (with or without its index). Confirm the
            # record was produced by this journal attempt and is the journal-bound
            # staged payload, then complete publication (index if missing) to
            # PRESENT. REHYDRATING never enters PUBLISHING, so a PUBLISHING record
            # must be its journal attempt's own.
            if record.producer_attempt_id != registry.attempt_id:
                raise BuildCacheError("publishing canonical record has a foreign producer")
            if (
                registry.staging_sha256 is None
                or hashlib.sha256(payload).hexdigest() != registry.staging_sha256
            ):
                raise BuildCacheError("canonical record diverged from staged publication bytes")
            return _complete_publishing(layout, registry, root, record, payload, digest)
        if index is not None:
            raise BuildCacheError("build index published without its canonical record")
        staged = _load_staged_record(layout, registry)
        if staged is not None:
            # _load_staged_record already authenticated the producer and staging
            # digest; complete publication from those verified staged bytes.
            return _complete_publishing(layout, registry, root, *staged)
        # No canonical record and no usable staged bytes: publication cannot be
        # completed forward. A missing PUBLISHING root is an integrity failure, never
        # a silent discard to VACANT; an existing root falls through to the shared
        # discard tail below, which authenticates the owner (dangling/mis-owned root
        # fails closed there) before removing it.
        if not os.path.lexists(root):
            raise BuildCacheError("publishing materialization lost its root before recovery")
    if registry.state is MaterializationState.CLEANING:
        # Cleanup is always post-canonical: require the exact journal-bound canonical
        # record/index pair to still be intact BEFORE deleting or completing cleanup,
        # so a later rehydrate can never find the canonical missing.
        _require_journal_bound_canonical(layout, registry)
        # lexists, not exists: a dangling-symlink root must not read as absent.
        # verify_owner_against_journal then fails closed on any symlink/mismatch.
        if os.path.lexists(root):
            _verify_owner_against_journal(root, registry)
            shutil.rmtree(root)
        # Always fsync the materialized parent, including the already-absent case,
        # so the removal is durable before the CLEANED transition is recorded.
        fsync_directory(layout.materialized)
        return _transition(
            layout,
            registry.build_id,
            registry.attempt_id,
            MaterializationState.CLEANED,
            canonical_record_sha256=registry.canonical_record_sha256,
        )
    # REHYDRATING/DISCARDING recovery. Once a canonical digest has been journal-bound
    # (post-canonical: REHYDRATING resumes from CLEANED, or DISCARDING descends from a
    # post-canonical state), the journal-bound canonical record/index pair MUST still
    # be intact. A missing/corrupt/divergent pair is an integrity failure that
    # preserves the root and evidence — never a silent regression to VACANT. Only a
    # genuinely pre-canonical state discards fully to VACANT.
    if registry.canonical_record_sha256 is not None:
        _require_journal_bound_canonical(layout, registry)
        destination = MaterializationState.CLEANED
    else:
        destination = MaterializationState.VACANT
    # Enter DISCARDING unless already there — an interrupted discard resumes removal
    # from its own DISCARDING state (a DISCARDING→DISCARDING transition is illegal).
    already_discarding = registry.state is MaterializationState.DISCARDING
    if os.path.lexists(root):
        owner = _verify_owner_against_journal(root, registry)
        if not already_discarding:
            _transition(
                layout,
                registry.build_id,
                registry.attempt_id,
                MaterializationState.DISCARDING,
                canonical_record_sha256=registry.canonical_record_sha256,
                root_device=owner.root_device,
                root_inode=owner.root_inode,
            )
        shutil.rmtree(root)
        fsync_directory(layout.materialized)
    elif not already_discarding:
        _transition(
            layout,
            registry.build_id,
            registry.attempt_id,
            MaterializationState.DISCARDING,
            canonical_record_sha256=registry.canonical_record_sha256,
        )
    return _transition(
        layout,
        registry.build_id,
        registry.attempt_id,
        destination,
        canonical_record_sha256=registry.canonical_record_sha256,
    )


def _require_journal_bound_canonical(layout: _Layout, registry: MaterializationRegistryV1) -> None:
    """Fail closed unless the exact journal-bound canonical record/index pair exists.

    Used at post-canonical recovery boundaries (CLEANING, and REHYDRATING/DISCARDING
    once a canonical digest is journal-bound), so a missing, corrupt, or divergent
    canonical is an integrity failure preserving the root and evidence, never a
    destructive delete or a regression to VACANT.
    """

    canonical, canonical_digest = _load_canonical_pair(layout, registry.build_id)
    if canonical is None or canonical_digest != registry.canonical_record_sha256:
        raise BuildCacheError("post-canonical recovery lost its journal-bound canonical record")


def _publish_canonical(
    layout: _Layout,
    record: CanonicalBuildRecordV1,
    payload: bytes,
    digest: str,
) -> None:
    _publish_immutable_file(
        _record_path(layout, record.build_id),
        payload,
        layout.success,
        "canonical build record",
    )
    index = BuildIndexV1(
        build_id=record.build_id,
        canonical_record_sha256=digest,
        producer_attempt_id=record.producer_attempt_id,
    )
    index_payload = canonical_json_bytes(index.model_dump(mode="json"))
    _publish_immutable_file(
        _index_path(layout, record.build_id),
        index_payload,
        layout.indexes,
        "build index",
    )


def _publish_immutable_file(path: Path, payload: bytes, parent: Path, description: str) -> None:
    """Publish one immutable file crash-atomically with no-replace semantics.

    A fully-written and fsynced same-directory temporary is atomically renamed into
    place (``rename_noreplace``) and the parent is then fsynced, so a crash can never
    leave a visible partial final file — only inert ``.tmp`` residue, which no reader
    treats as authoritative. A byte-identical existing file (including one a
    concurrent publisher won the race to create) is accepted; a divergent one fails
    closed. lexists, not exists, so a symlinked destination is never followed.
    """

    if os.path.lexists(path):
        if _read_bytes(path, _MODEL_LIMIT) != payload:
            raise BuildCacheError(f"divergent {description} collision")
        # The final may already be renamed into place by a prior attempt (or a
        # concurrent publisher) that crashed or failed before fsyncing the parent, so
        # repeat the durability barrier: acceptance must always imply a durable
        # directory entry, never a rename that a later crash could still lose.
        fsync_directory(parent)
        return
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    try:
        write_exclusive(temporary, payload, 0o400)
        try:
            rename_noreplace(temporary, path)
        except FileExistsError:
            # A concurrent publisher created it first: accept only byte-identical, and
            # still fsync the parent so our return implies a durable directory entry.
            if _read_bytes(path, _MODEL_LIMIT) != payload:
                raise BuildCacheError(f"divergent {description} collision") from None
            fsync_directory(parent)
            return
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    fsync_directory(parent)


def _load_canonical(
    layout: _Layout, build_id: str
) -> tuple[CanonicalBuildRecordV1 | None, bytes | None, str | None, BuildIndexV1 | None]:
    """Load the canonical record and index independently, without pairing them.

    Recovery uses this so a record written before its index (or an orphan
    index) can be diagnosed and repaired instead of aborting on a cardinality
    mismatch.
    """

    record_path = _record_path(layout, build_id)
    index_path = _index_path(layout, build_id)
    record: CanonicalBuildRecordV1 | None = None
    payload: bytes | None = None
    digest: str | None = None
    index: BuildIndexV1 | None = None
    # lexists, not exists: a dangling or symlinked immutable path is corrupt state,
    # never an absent record. Presence routes to the nofollow read, which fails
    # closed on any non-regular entry rather than reading it as missing.
    if os.path.lexists(record_path):
        payload = _read_bytes(record_path, _MODEL_LIMIT)
        record = _parse_model(payload, CanonicalBuildRecordV1)
        if record.build_id != build_id:
            raise BuildCacheError("canonical build record is bound to another build")
        digest = _record_digest(payload)
    if os.path.lexists(index_path):
        index = _read_model(index_path, BuildIndexV1)
        if index.build_id != build_id:
            raise BuildCacheError("build index is bound to another build")
    return record, payload, digest, index


def _load_canonical_pair(
    layout: _Layout, build_id: str
) -> tuple[CanonicalBuildRecordV1 | None, str | None]:
    record, _payload, digest, index = _load_canonical(layout, build_id)
    if record is None and index is None:
        return None, None
    if record is None or index is None or digest is None:
        raise BuildCacheError("canonical record and build index cardinality diverged")
    if (
        index.canonical_record_sha256 != digest
        or index.producer_attempt_id != record.producer_attempt_id
    ):
        raise BuildCacheError("canonical record and build index binding diverged")
    return record, digest


_EVENT_TEMP_RE = re.compile(r"^\.(?P<sequence>\d{8})\.json\.[0-9a-f]{16}\.tmp$")


@contextlib.contextmanager
def _open_journal_fds(layout: _Layout, build_id: str) -> Iterator[tuple[int, int | None] | None]:
    """Hold the validated per-build journal (and events) directory descriptors for the
    duration of a read/recovery, with disciplined close ownership.

    Yields ``(journal_fd, events_fd)`` — ``events_fd`` is ``None`` when the events dir
    does not exist — or ``None`` when the journal has not been created (a fresh build).
    Every read/enumeration/write is done relative to these held descriptors, so a
    symlink swapped in for a per-build directory after validation cannot redirect
    anything.
    """

    journal_fd = _open_owned_journal_fd(layout, build_id)
    if journal_fd is None:
        yield None
        return
    events_fd: int | None = None
    try:
        events_fd = _open_owned_child_dir(
            journal_fd, "events", _journal(layout, build_id) / "events"
        )
        yield journal_fd, events_fd
    finally:
        if events_fd is not None:
            os.close(events_fd)
        os.close(journal_fd)


def _reconcile_locked(
    journal_fd: int, events_fd: int | None, build_id: str
) -> MaterializationRegistryV1 | None:
    # lexists (via fd-relative lstat), not exists: a dangling current.json symlink is
    # corruption (the nofollow read then fails), never absence.
    registry = (
        _read_model_at(journal_fd, "current.json", MaterializationRegistryV1)
        if _lexists_at(journal_fd, "current.json")
        else None
    )
    _reconcile_orphan_temp(events_fd, build_id, registry)
    return _reconcile_orphan_event(journal_fd, events_fd, build_id, registry)


def _load_registry(layout: _Layout, build_id: str) -> MaterializationRegistryV1 | None:
    """Load and orphan-reconcile the registry without verifying the event chain.

    Used by ``_transition`` to read the immediately-previous registry; callers that
    reuse a materialization use ``_load_verified_registry`` instead.
    """

    with _open_journal_fds(layout, build_id) as handle:
        if handle is None:
            return None
        journal_fd, events_fd = handle
        return _reconcile_locked(journal_fd, events_fd, build_id)


def _load_verified_registry(layout: _Layout, build_id: str) -> MaterializationRegistryV1 | None:
    """Load, orphan-reconcile, and verify the event chain under one held journal/events
    descriptor pair — no duplicate open/validate of the same directories."""

    with _open_journal_fds(layout, build_id) as handle:
        if handle is None:
            return None
        journal_fd, events_fd = handle
        registry = _reconcile_locked(journal_fd, events_fd, build_id)
        if registry is not None:
            if events_fd is None:
                raise BuildCacheError("materialization events directory is missing")
            _verify_event_chain_at(events_fd, registry)
        return registry


def _reconcile_orphan_temp(
    events_fd: int | None, build_id: str, registry: MaterializationRegistryV1 | None
) -> None:
    """Remove a single authenticated writer temp left by an interrupted _publish_event.

    All entries are enumerated and read relative to the held ``events_fd``.
    ``_publish_event`` writes ``.NNNNNNNN.json.<token>.tmp`` before renaming it into
    place, so a crash there can leave exactly that temp. Strict event cardinality
    must not brick the journal on it, but nothing is blindly deleted: only the exact
    temp shape this writer creates is considered, it must be an owned 0o400 regular
    file, and it must relate to the committed chain — a redundant byte-identical copy
    of an already-committed event, or the uncommitted next event (no committed
    destination). More than one temp, or a malformed/foreign/divergent/gap temp,
    fails closed; any other unexpected entry is left for _verify_event_chain.
    """

    if events_fd is None:
        return
    temps = [name for name in os.listdir(events_fd) if _EVENT_TEMP_RE.match(name)]
    if not temps:
        return
    if len(temps) > 1:
        raise BuildCacheError("multiple writer temp materialization events")
    temp_name = temps[0]
    match = _EVENT_TEMP_RE.match(temp_name)
    assert match is not None
    sequence = int(match.group("sequence"))
    metadata = os.lstat(temp_name, dir_fd=events_fd)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o400
    ):
        raise BuildCacheError("unsafe writer temp materialization event")
    next_sequence = 1 if registry is None else registry.sequence + 1
    destination = f"{sequence:08d}.json"
    if _lexists_at(events_fd, destination):
        # A committed event already holds this sequence: the temp is only a redundant
        # leftover if it is a byte-identical copy of that committed event.
        if _read_bytes_at(events_fd, temp_name, _MODEL_LIMIT) != _read_bytes_at(
            events_fd, destination, _MODEL_LIMIT
        ):
            raise BuildCacheError("divergent writer temp materialization event")
    elif sequence != next_sequence:
        # No committed event holds this sequence and it is not the next one to
        # commit: a gap or future temp is divergent, never silently removed.
        raise BuildCacheError("divergent writer temp materialization event")
    # Uncommitted next event (possibly a torn write) or an authenticated redundant
    # copy: removing it is safe — the interrupted transition simply retries.
    os.unlink(temp_name, dir_fd=events_fd)
    os.fsync(events_fd)


def _reconcile_orphan_event(
    journal_fd: int,
    events_fd: int | None,
    build_id: str,
    registry: MaterializationRegistryV1 | None,
) -> MaterializationRegistryV1 | None:
    """Complete a transition that durably wrote its event but not current.json.

    ``_transition`` publishes event N before current.json, so a crash there
    leaves exactly one orphan event beyond the registry (or an orphan first
    event with no registry at all). Adopt that single authenticated next event
    so the interrupted transition finishes; reject any divergent orphan. All reads
    and the current.json write are relative to the held ``events_fd``/``journal_fd``.
    """

    if events_fd is None:
        return registry
    next_sequence = 1 if registry is None else registry.sequence + 1
    orphan_name = f"{next_sequence:08d}.json"
    if not _lexists_at(events_fd, orphan_name):
        return registry
    event = _read_model_at(events_fd, orphan_name, MaterializationEventV1)
    expected_previous = registry.last_event_sha256 if registry is not None else None
    expected_from = registry.state if registry is not None else None
    if not _event_links(
        event,
        build_id=build_id,
        sequence=next_sequence,
        previous=expected_previous,
        from_state=expected_from,
    ) or event.to_state not in _LEGAL.get(expected_from, frozenset()):
        raise BuildCacheError("divergent orphan materialization event")
    _payload, digest = _event_bytes(event)
    reconciled = _registry_from_event(event, digest)
    # Ensure the adopted event's directory entry is durable BEFORE promoting
    # current.json to reference it: otherwise a crash could persist the reconciled
    # registry while losing the orphan event it points at.
    os.fsync(events_fd)
    _atomic_write_at(
        journal_fd, "current.json", canonical_json_bytes(reconciled.model_dump(mode="json"))
    )
    return reconciled


def _required_registry(layout: _Layout, build_id: str) -> MaterializationRegistryV1:
    registry = _load_verified_registry(layout, build_id)
    if registry is None:
        raise BuildCacheError("materialization registry is missing")
    return registry


def _read_owner(root: Path) -> BuildRootOwnerV1:
    marker = root / ".strixlab-owner.json"
    flags = readonly_open_flags()
    try:
        descriptor = os.open(marker, flags)
    except OSError as exc:
        raise BuildCacheError("build-root ownership marker is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o400:
            raise BuildCacheError("build-root ownership marker is unsafe")
        content = os.read(descriptor, _OWNER_LIMIT + 1)
        if len(content) > _OWNER_LIMIT:
            raise BuildCacheError("build-root ownership marker is oversized")
    finally:
        os.close(descriptor)
    return _parse_model(content, BuildRootOwnerV1)


def _atomic_write(path: Path, payload: bytes) -> None:
    # Create the temp at the final 0400 mode so os.replace publishes an already
    # read-only file (no post-publication chmod window), and always remove the
    # temp on any failure before the rename lands.
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    try:
        write_exclusive(temporary, payload, 0o400)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    fsync_directory(path.parent)


def _read_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    return _parse_model(_read_bytes(path, _MODEL_LIMIT), model)


def _parse_model[ModelT: BaseModel](payload: bytes, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(payload, strict=True)
    except ValidationError as exc:
        raise BuildCacheError("stored build cache model is invalid") from exc


def _read_descriptor(descriptor: int, limit: int) -> bytes:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
        raise BuildCacheError("build cache file is unsafe")
    content = bytearray()
    while len(content) <= limit:
        chunk = os.read(descriptor, min(64 * 1024, limit + 1 - len(content)))
        if not chunk:
            break
        content.extend(chunk)
    if len(content) > limit:
        raise BuildCacheError("build cache file is oversized")
    return bytes(content)


def _read_bytes(path: Path, limit: int) -> bytes:
    try:
        descriptor = os.open(path, readonly_open_flags())
    except OSError as exc:
        raise BuildCacheError(f"build cache file is unavailable: {path.name}") from exc
    try:
        return _read_descriptor(descriptor, limit)
    finally:
        os.close(descriptor)


def _read_bytes_at(dir_fd: int, name: str, limit: int) -> bytes:
    """Read one regular file relative to a held directory descriptor, no-follow.

    Anchoring the open at ``dir_fd`` means the read cannot be redirected by a symlink
    swapped in for the containing directory after it was validated.
    """

    try:
        descriptor = os.open(name, readonly_open_flags(), dir_fd=dir_fd)
    except OSError as exc:
        raise BuildCacheError(f"build cache file is unavailable: {name}") from exc
    try:
        return _read_descriptor(descriptor, limit)
    finally:
        os.close(descriptor)


def _read_model_at[ModelT: BaseModel](dir_fd: int, name: str, model: type[ModelT]) -> ModelT:
    return _parse_model(_read_bytes_at(dir_fd, name, _MODEL_LIMIT), model)


def _lexists_at(dir_fd: int, name: str) -> bool:
    """True if ``name`` exists relative to ``dir_fd`` (no-follow; a dangling symlink
    counts as present so the nofollow read then fails closed as corruption)."""

    try:
        os.lstat(name, dir_fd=dir_fd)
    except FileNotFoundError:
        return False
    return True


def _atomic_write_at(dir_fd: int, name: str, payload: bytes) -> None:
    """Atomically publish one file relative to a held directory descriptor.

    The temp is created, written, and fsynced, then renamed over ``name`` and the
    directory descriptor itself fsynced — all anchored at ``dir_fd``, so no symlink
    swapped in for the directory can redirect the write.
    """

    temporary = f".{name}.{secrets.token_hex(8)}.tmp"
    try:
        descriptor = os.open(temporary, exclusive_create_flags(), 0o400, dir_fd=dir_fd)
        try:
            write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=dir_fd)
    os.fsync(dir_fd)


def _record_digest(payload: bytes) -> str:
    return f"record-sha256:{hashlib.sha256(payload).hexdigest()}"


def _journal(layout: _Layout, build_id: str) -> Path:
    return layout.journals / build_id.removeprefix("build-sha256:")


def _record_path(layout: _Layout, build_id: str) -> Path:
    return layout.success / f"{build_id.removeprefix('build-sha256:')}.json"


def _index_path(layout: _Layout, build_id: str) -> Path:
    return layout.indexes / f"{build_id.removeprefix('build-sha256:')}.json"


def _attestation_path(layout: _Layout, build_id: str) -> Path:
    return layout.success / f"{build_id.removeprefix('build-sha256:')}.attestation.json"


def _publish_attestation(layout: _Layout, attestation: BuildAttestationV1) -> None:
    payload = canonical_json_bytes(attestation.model_dump(mode="json"))
    _publish_immutable_file(
        _attestation_path(layout, attestation.build_id),
        payload,
        layout.success,
        "build attestation",
    )


def _load_attestation(layout: _Layout, build_id: str) -> BuildAttestationV1 | None:
    path = _attestation_path(layout, build_id)
    # lexists, not exists: a dangling attestation symlink is corrupt state, not an
    # absent attestation that would be silently re-created as a recovery.
    if not os.path.lexists(path):
        return None
    attestation = _read_model(path, BuildAttestationV1)
    if attestation.build_id != build_id:
        raise BuildCacheError("build attestation is bound to another build")
    return attestation


def publish_build_attestation(
    build_id: str,
    *,
    attestor_attempt_id: str,
    canonical_record_sha256: str,
    execution_class: AttestationClass,
    artifact_set_id: str,
    home: Path,
) -> None:
    """Publish the post-finalization attestation for a genuinely completed build.

    Called under the caller's recipe lock after the attestor attempt finalized
    SUCCESS; re-acquires the nested build-ID lock (preserving recipe→build-ID
    order). The attestor is authenticated before the immutable attestation is
    written, so a divergent or unauthenticated attestation can never be published.
    """

    _validate_build_id(build_id)
    _validate_attempt_id(attestor_attempt_id)
    layout = _layout(home, create=False)
    lock = layout.locks / f"build-id-{build_id.removeprefix('build-sha256:')}.lock"
    with exclusive_lock(lock) as held:
        if not held.acquired:
            raise BuildCacheError(held.reason or "build cache lock is unavailable")
        canonical, digest = _load_canonical_pair(layout, build_id)
        if canonical is None or digest is None:
            raise BuildCacheError("cannot attest a build without a canonical record")
        if digest != canonical_record_sha256:
            raise BuildCacheError("attestation canonical digest diverged from the record")
        # Authenticate the whole canonical (producer provenance) and resolve the
        # attestor before publishing; reuse the verified producer when it is the
        # attestor. Their exact record digests are anchored into the attestation.
        producer = _verify_canonical_producer(home, canonical)
        # External anchor: this runs under the caller's held recipe lock, so bind the
        # producer's self-consistent record digest to the authoritative recipe-index
        # digest before an attestation exists. A self-consistently replaced producer
        # record (whose content-address then differs) is rejected here during
        # crash-forward recovery, not merely at later attested reuse.
        external_producer_digest = recipe_index_record_digest(
            canonical.recipe_id, canonical.producer_attempt_id, home=home
        )
        if producer.record_sha256 != external_producer_digest:
            raise BuildCacheError("producer record digest diverged from its recipe-index anchor")
        attestor = _resolve_attestor(home, canonical, attestor_attempt_id, producer)
        attestation = BuildAttestationV1(
            build_id=build_id,
            canonical_record_sha256=digest,
            attestor_attempt_id=attestor_attempt_id,
            execution_class=execution_class,
            artifact_set_id=artifact_set_id,
            producer_record_sha256=producer.record_sha256,
            attestor_record_sha256=attestor.record_sha256,
        )
        _authenticate_attestor(canonical, digest, attestation, producer, attestor)
        _publish_attestation(layout, attestation)


def _materialization_path(layout: _Layout, build_id: str) -> Path:
    return _journal(layout, build_id) / "materialization.json"


def _write_materialization_evidence(
    layout: _Layout,
    build_id: str,
    attempt_id: str,
    canonical_digest: str,
    artifacts: BuildArtifactsV1,
) -> str:
    """Durably record this materialization's own validated artifact evidence.

    Written and fsynced (atomic replace) before the PRESENT transition records its
    digest, so a crash can only leave the journal PUBLISHING/REHYDRATING (recovered
    forward) — never a PRESENT registry whose evidence file is absent.
    """

    evidence = MaterializationEvidenceV1(
        build_id=build_id,
        attempt_id=attempt_id,
        canonical_record_sha256=canonical_digest,
        artifacts=artifacts,
    )
    payload = canonical_json_bytes(evidence.model_dump(mode="json"))
    _atomic_write(_materialization_path(layout, build_id), payload)
    return hashlib.sha256(payload).hexdigest()


def _load_materialization_evidence(
    layout: _Layout, registry: MaterializationRegistryV1
) -> MaterializationEvidenceV1:
    """Load and authenticate the current materialization's artifact evidence."""

    if registry.materialization_sha256 is None or registry.canonical_record_sha256 is None:
        raise BuildCacheError("materialized build has no materialization evidence binding")
    payload = _read_bytes(_materialization_path(layout, registry.build_id), _MODEL_LIMIT)
    if hashlib.sha256(payload).hexdigest() != registry.materialization_sha256:
        raise BuildCacheError("materialization evidence diverged from its journal digest")
    evidence = _parse_model(payload, MaterializationEvidenceV1)
    if (
        evidence.build_id != registry.build_id
        or evidence.attempt_id != registry.attempt_id
        or evidence.canonical_record_sha256 != registry.canonical_record_sha256
    ):
        raise BuildCacheError("materialization evidence is not bound to its journal")
    return evidence


def _require_owned_directory(path: Path) -> None:
    """Fail closed unless ``path`` is a current-user-owned real directory."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise BuildCacheError(f"build cache directory is unavailable: {path}") from exc
    if is_unsafe_directory(metadata):
        raise BuildCacheError(f"build cache directory is unsafe: {path}")


_DIR_OPEN_FLAGS = directory_open_flags()


def _open_owned_child_dir(dir_fd: int, name: str, describe: Path) -> int | None:
    """openat one child directory no-follow, verifying it is an owned real directory.

    Returns the child descriptor (caller closes it) or ``None`` if the entry does not
    exist. A symlink, non-directory, or foreign-owned entry fails closed. Descriptor
    anchoring means a symlinked ancestor cannot redirect the traversal.
    """

    try:
        return try_open_owned_directory(name, dir_fd=dir_fd)
    except OSError as exc:
        raise BuildCacheError(f"build cache directory is unsafe: {describe}") from exc


def _open_owned_build_subdir(root: Path, child: Path) -> int | None:
    """Open ``child`` descriptor-anchored from its validated storage ``root``.

    Returns the child directory descriptor (caller closes) or ``None`` when absent; a
    symlinked, non-directory, or foreign-owned root or child fails closed. Shared by
    the structurally identical per-build journal and publication-staging openers.
    """

    try:
        root_fd = os.open(root, _DIR_OPEN_FLAGS)
    except OSError as exc:
        raise BuildCacheError(f"build cache directory is unsafe: {root}") from exc
    try:
        return _open_owned_child_dir(root_fd, child.name, child)
    finally:
        os.close(root_fd)


def _open_owned_journal_fd(layout: _Layout, build_id: str) -> int | None:
    """Open the validated per-build journal directory descriptor (``None`` if the
    journal has not been created — a fresh build)."""

    return _open_owned_build_subdir(layout.journals, _journal(layout, build_id))


def _open_owned_staging_fd(layout: _Layout, build_id: str) -> int | None:
    """Open the validated per-build publication-staging directory descriptor
    (``None`` if absent)."""

    return _open_owned_build_subdir(
        layout.staging, layout.staging / build_id.removeprefix("build-sha256:")
    )


def _ensure_journal_events(journal: Path) -> Path:
    """Create the per-build journal and its events dir, fsyncing each parent.

    A pre-existing per-build entry (ensure_directory_fsynced returned False) is
    validated as an owned real directory before use, so a symlink swapped in for a
    journal/events directory is never followed.
    """

    if not ensure_directory_fsynced(journal):
        _require_owned_directory(journal)
    events = journal / "events"
    if not ensure_directory_fsynced(events):
        _require_owned_directory(events)
    return events


def _stage_path(layout: _Layout, build_id: str, attempt_id: str) -> Path:
    root = layout.staging / build_id.removeprefix("build-sha256:")
    if not ensure_directory_fsynced(root):
        _require_owned_directory(root)
    return root / f"{attempt_id}.json"


def _load_staged_record(
    layout: _Layout, registry: MaterializationRegistryV1
) -> tuple[CanonicalBuildRecordV1, bytes, str] | None:
    """Return the authenticated pre-canonical staged record, or None if unusable.

    A missing or invalid staged payload is a safe pre-commit discard, not an
    integrity failure: nothing durable has been published from it yet.
    """

    if registry.staging_sha256 is None:
        return None
    # Read the staged record relative to the held, validated per-build staging
    # directory descriptor: a symlinked staging directory is never followed. An absent
    # staging directory is a safe pre-commit discard.
    stage_fd = _open_owned_staging_fd(layout, registry.build_id)
    if stage_fd is None:
        return None
    try:
        payload = _read_bytes_at(stage_fd, f"{registry.attempt_id}.json", _MODEL_LIMIT)
    except BuildCacheError:
        return None
    finally:
        os.close(stage_fd)
    if hashlib.sha256(payload).hexdigest() != registry.staging_sha256:
        return None
    try:
        record = _parse_model(payload, CanonicalBuildRecordV1)
    except BuildCacheError:
        return None
    if record.build_id != registry.build_id or record.producer_attempt_id != registry.attempt_id:
        return None
    return record, payload, _record_digest(payload)


def _validate_build_id(build_id: str) -> None:
    if _BUILD_RE.fullmatch(build_id) is None:
        raise ValueError("invalid machine-local build ID")


def _validate_attempt_id(attempt_id: str) -> None:
    if _ATTEMPT_RE.fullmatch(attempt_id) is None:
        raise ValueError("invalid build attempt ID")


def _identity_entries(values: tuple[IdentityEntryV1, ...]) -> tuple[IdentityEntry, ...]:
    return tuple(IdentityEntry(value.name, value.value) for value in values)


def _verify_root_artifacts(
    root: Path,
    artifacts: BuildArtifactsV1,
    *,
    selections: tuple[IdentityEntryV1, ...],
    toolchain_mode: Literal["host", "rocm"],
) -> None:
    """Rehash the materialized root against one artifact set and identity."""

    verify_artifact_capture(
        root,
        artifacts,
        selections=_identity_entries(selections),
        toolchain_mode=toolchain_mode,
    )


def _verify_record_artifacts(root: Path, record: CanonicalBuildRecordV1) -> None:
    """Rehash the materialized root against one canonical record's artifacts."""

    _verify_root_artifacts(
        root,
        record.artifacts,
        selections=record.selections,
        toolchain_mode=record.toolchain_mode,
    )


def _verify_materialized_root(
    layout: _Layout,
    root: Path,
    registry: MaterializationRegistryV1,
    canonical: CanonicalBuildRecordV1,
) -> None:
    """Verify a PRESENT root against this materialization's own artifact evidence.

    Identity (selections/toolchain_mode) comes from the canonical record; the
    artifacts come from the current materialization, so a rehydration's own
    non-identity observations are what a later HIT/inspect/cleanup enforces.
    """

    evidence = _load_materialization_evidence(layout, registry)
    if evidence.artifacts.artifact_set_id != canonical.artifacts.artifact_set_id:
        raise BuildCacheError("materialization artifact-set diverged from canonical evidence")
    _verify_root_artifacts(
        root,
        evidence.artifacts,
        selections=canonical.selections,
        toolchain_mode=canonical.toolchain_mode,
    )


def cache_environment_projection(
    environment: Mapping[str, str],
    *,
    home: Path,
    source_root: Path,
    build_root: Path,
    build_home: Path,
    build_tmp: Path,
) -> tuple[IdentityEntryV1, ...]:
    """Project owned runtime roots onto the four stable cache path domains.

    Only exact StrixLab-owned prefixes are replaced at component boundaries;
    every declared non-owned value stays byte-exact. Any residual owned path
    that no binding can represent fails closed.
    """

    bindings = sorted(
        (
            (str(source_root), ROOT_PLACEHOLDERS["SOURCE_ROOT"]),
            (str(build_root), ROOT_PLACEHOLDERS["BUILD_ROOT"]),
            (str(build_home), ROOT_PLACEHOLDERS["BUILD_HOME"]),
            (str(build_tmp), ROOT_PLACEHOLDERS["BUILD_TMP"]),
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    owned = str(build_storage_roots(home).root)

    def _project_component(component: str) -> str:
        for prefix, placeholder in bindings:
            if component == prefix:
                return placeholder
            if component.startswith(prefix + os.sep):
                return placeholder + component[len(prefix) :]
        return component

    entries: list[IdentityEntryV1] = []
    for name in sorted(environment):
        components = environment[name].split(os.pathsep)
        projected = os.pathsep.join(_project_component(component) for component in components)
        for component in projected.split(os.pathsep):
            if component == owned or component.startswith(owned + os.sep):
                raise BuildCacheError("cache environment projection leaks an owned StrixLab path")
        entries.append(IdentityEntryV1(name=name, value=projected))
    return tuple(entries)


def identity_models(values: tuple[IdentityEntry, ...]) -> tuple[IdentityEntryV1, ...]:
    return tuple(IdentityEntryV1(name=value.name, value=value.value) for value in values)


def tool_models(values: tuple[ToolObservation, ...]) -> tuple[ToolObservationV1, ...]:
    return tuple(
        ToolObservationV1(
            role=value.role,
            path=value.path,
            realpath=value.realpath,
            mode=value.mode,
            size_bytes=value.size_bytes,
            sha256=value.sha256,
            version_sha256=value.version_sha256,
            search_sha256=value.search_sha256,
        )
        for value in values
    )

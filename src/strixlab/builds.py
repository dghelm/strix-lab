"""Crash-safe control plane for reproducible build attempts."""

from __future__ import annotations

import contextlib
import os
import re
import secrets
import shutil
import stat
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from strixlab.build_identity import attempt_id
from strixlab.build_records import (
    BuildRecordError,
    RecordVerification,
    publish_record,
    record_source_digest,
    verify_record,
)
from strixlab.locks import exclusive_lock
from strixlab.secure_fs import (
    exclusive_create_flags,
    fsync_directory,
    rename_noreplace,
    write_exclusive,
)
from strixlab.serialization import canonical_json_bytes

_RECIPE_RE = re.compile(r"^recipe-sha256:[0-9a-f]{64}$")
_BUILD_RE = re.compile(r"^build-sha256:[0-9a-f]{64}$")
_ATTEMPT_RE = re.compile(r"^attempt-[0-9a-f]{24}-[0-9a-f]{32}$")
_RECORD_RE = re.compile(r"^record-sha256:[0-9a-f]{64}$")
_ALLOCATION_TEMP_RE = re.compile(r"^\.current\.json\.[0-9a-f]{16}\.tmp$")


class BuildStateError(RuntimeError):
    """Persisted build state is unsafe, inconsistent, or corrupt."""


class BuildBusyError(BuildStateError):
    """Another process owns the requested recipe transition."""


class AttemptOutcome(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    CACHE_HIT = "cache-hit"


class AttemptState(StrEnum):
    ALLOCATED = "allocated"
    ACTIVE = "active"
    FINALIZING = "finalizing"
    RECORD_PUBLISHED = "record-published"
    TORN_DOWN = "torn-down"


_LEGAL_TRANSITIONS: dict[AttemptState | None, frozenset[AttemptState]] = {
    None: frozenset({AttemptState.ALLOCATED}),
    AttemptState.ALLOCATED: frozenset({AttemptState.ACTIVE, AttemptState.FINALIZING}),
    AttemptState.ACTIVE: frozenset({AttemptState.FINALIZING}),
    AttemptState.FINALIZING: frozenset({AttemptState.RECORD_PUBLISHED}),
}
_OUTCOME_STATES = frozenset(
    {AttemptState.FINALIZING, AttemptState.RECORD_PUBLISHED, AttemptState.TORN_DOWN}
)
_RECORDED_STATES = frozenset({AttemptState.RECORD_PUBLISHED, AttemptState.TORN_DOWN})


class _StoredModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ProcessOwnerV1(_StoredModel):
    schema_version: Literal[1] = 1
    pid: int = Field(gt=0)
    boot_id: str = Field(min_length=1)
    process_start_ticks: int = Field(gt=0)


class AttemptRegistryV1(_StoredModel):
    schema_version: Literal[1] = 1
    attempt_id: str = Field(pattern=_ATTEMPT_RE.pattern)
    recipe_id: str = Field(pattern=_RECIPE_RE.pattern)
    nonce: str = Field(pattern=r"^[0-9a-f]{32}$")
    state: AttemptState
    outcome: AttemptOutcome | None
    build_id: str | None = Field(default=None, pattern=_BUILD_RE.pattern)
    record_sha256: str | None = Field(default=None, pattern=_RECORD_RE.pattern)
    owner: ProcessOwnerV1
    root_device: int = Field(ge=0)
    root_inode: int = Field(gt=0)
    created_at: str
    updated_at: str


class AttemptEventV1(_StoredModel):
    schema_version: Literal[1] = 1
    attempt_id: str = Field(pattern=_ATTEMPT_RE.pattern)
    sequence: int = Field(gt=0)
    timestamp: str
    from_state: AttemptState | None = Field(alias="from")
    to_state: AttemptState = Field(alias="to")


class AttemptIndexEntryV1(_StoredModel):
    schema_version: Literal[1] = 1
    attempt_id: str = Field(pattern=_ATTEMPT_RE.pattern)
    state: AttemptState
    outcome: AttemptOutcome | None
    build_id: str | None = Field(default=None, pattern=_BUILD_RE.pattern)
    record_sha256: str | None = Field(default=None, pattern=_RECORD_RE.pattern)


class RecipeIndexV1(_StoredModel):
    schema_version: Literal[1] = 1
    recipe_id: str = Field(pattern=_RECIPE_RE.pattern)
    attempts: tuple[AttemptIndexEntryV1, ...]


@dataclass(frozen=True, slots=True)
class AttemptResult:
    attempt_id: str
    recipe_id: str
    outcome: AttemptOutcome
    build_id: str | None
    record: Path
    record_sha256: str


@dataclass(slots=True)
class BuildAttemptSession:
    _layout: _BuildLayout
    registry: AttemptRegistryV1
    _finalized: bool = False
    result: AttemptResult | None = None

    @property
    def root(self) -> Path:
        return self._layout.attempts / self.registry.attempt_id

    @property
    def snapshots(self) -> Path:
        return self._layout.snapshots

    @property
    def materialized(self) -> Path:
        return self._layout.materialized

    @property
    def probe_root(self) -> Path:
        return self.root / "private" / "probe"

    def build_root(self, build_id: str) -> Path:
        if _BUILD_RE.fullmatch(build_id) is None:
            raise ValueError("invalid machine-local build ID")
        return self._layout.materialized / build_id

    def write_evidence(self, relative: str, content: bytes) -> Path:
        path = _safe_attempt_relative(relative)
        return _write_attempt_file(self.root, path, content)

    def mark_active(self) -> None:
        self.registry = _transition(self.root, self.registry, AttemptState.ACTIVE)
        _store_recipe_entry(
            self._layout,
            self.registry.recipe_id,
            AttemptIndexEntryV1(
                attempt_id=self.registry.attempt_id,
                state=AttemptState.ACTIVE,
                outcome=None,
            ),
        )

    def finalize(
        self,
        outcome: AttemptOutcome,
        *,
        build_id: str | None = None,
    ) -> AttemptResult:
        if self._finalized:
            raise BuildStateError("build attempt is already finalized")
        self.result = _finalize_attempt(self._layout, self.registry, outcome, build_id=build_id)
        self._finalized = True
        return self.result


@dataclass(frozen=True, slots=True)
class _BuildLayout:
    home: Path
    root: Path
    attempts: Path
    attempt_records: Path
    success_records: Path
    materialized: Path
    snapshots: Path
    recipe_indexes: Path
    build_indexes: Path
    locks: Path


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        path.mkdir(mode=0o700)
        return
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise BuildStateError(f"build state path is unsafe: {path}")


def _layout(home: Path, *, create: bool) -> _BuildLayout:
    if not home.is_absolute():
        raise ValueError("StrixLab home must be absolute")
    if home.exists() and home.is_symlink():
        raise BuildStateError("StrixLab home cannot be a symbolic link")
    if create and not home.exists():
        home.mkdir(mode=0o700, parents=True)
    root = home / "builds"
    layout = _BuildLayout(
        home=home,
        root=root,
        attempts=root / "attempts",
        attempt_records=root / "records" / "attempts",
        success_records=root / "records" / "success",
        materialized=root / "materialized",
        snapshots=root / "snapshots",
        recipe_indexes=root / "indexes" / "recipes",
        build_indexes=root / "indexes" / "builds",
        locks=home / "locks",
    )
    paths = (
        layout.home,
        layout.root,
        layout.attempts,
        layout.root / "records",
        layout.attempt_records,
        layout.success_records,
        layout.materialized,
        layout.snapshots,
        layout.root / "indexes",
        layout.recipe_indexes,
        layout.build_indexes,
        layout.locks,
    )
    if not create and any(not path.exists() for path in paths):
        raise BuildStateError("StrixLab build state does not exist")
    for path in paths:
        _ensure_directory(path)
    return layout


def _atomic_model(path: Path, model: BaseModel) -> None:
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    try:
        write_exclusive(temporary, canonical_json_bytes(model.model_dump(mode="json")))
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    flags = os.O_CLOEXEC | os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BuildStateError(f"build state is unavailable: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise BuildStateError(f"build state is not an owned regular file: {path}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 64 * 1024):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    try:
        return model.model_validate_json(b"".join(chunks))
    except ValueError as exc:
        raise BuildStateError(f"build state is invalid: {path}") from exc


def _boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise BuildStateError("cannot read the Linux boot identity") from exc
    if not value:
        raise BuildStateError("Linux boot identity is empty")
    return value


def _process_start_ticks(pid: int) -> int:
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        closing = value.rfind(")")
        fields = value[closing + 2 :].split()
        ticks = int(fields[19])
    except (OSError, ValueError, IndexError) as exc:
        raise BuildStateError(f"cannot read process identity for PID {pid}") from exc
    if ticks <= 0:
        raise BuildStateError("process start identity is invalid")
    return ticks


def _current_owner() -> ProcessOwnerV1:
    pid = os.getpid()
    return ProcessOwnerV1(
        pid=pid, boot_id=_boot_id(), process_start_ticks=_process_start_ticks(pid)
    )


def _owner_alive(owner: ProcessOwnerV1) -> bool:
    if owner.boot_id != _boot_id():
        return False
    try:
        return _process_start_ticks(owner.pid) == owner.process_start_ticks
    except BuildStateError:
        return False


def _safe_attempt_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or path.is_absolute()
        or path == PurePosixPath(".")
        or ".." in path.parts
        or path.as_posix() != value
        or path.parts[0] in {"current.json", "events", "terminal.json"}
    ):
        raise BuildStateError(f"unsafe attempt-relative evidence path: {value!r}")
    return path


def _write_attempt_file(root: Path, relative: PurePosixPath, content: bytes) -> Path:
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    descriptor = os.open(root, directory_flags)
    try:
        for part in relative.parts[:-1]:
            try:
                child = os.open(part, directory_flags, dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(part, 0o700, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(part, directory_flags, dir_fd=descriptor)
            metadata = os.fstat(child)
            if metadata.st_uid != os.geteuid() or not stat.S_ISDIR(metadata.st_mode):
                os.close(child)
                raise BuildStateError("attempt evidence parent is unsafe")
            os.close(descriptor)
            descriptor = child
        output = os.open(relative.name, exclusive_create_flags(), 0o600, dir_fd=descriptor)
        try:
            view = memoryview(content)
            while view:
                written = os.write(output, view)
                if written <= 0:
                    raise OSError("short attempt evidence write")
                view = view[written:]
            os.fsync(output)
        finally:
            os.close(output)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return root / relative


def _event_path(root: Path, sequence: int) -> Path:
    return root / "events" / f"{sequence:04d}.json"


def _write_event(
    root: Path,
    attempt_identifier: str,
    sequence: int,
    timestamp: str,
    from_state: AttemptState | None,
    to_state: AttemptState,
) -> None:
    event = {
        "attempt_id": attempt_identifier,
        "from": from_state,
        "schema_version": 1,
        "sequence": sequence,
        "timestamp": timestamp,
        "to": to_state,
    }
    events = root / "events"
    destination = _event_path(root, sequence)
    temporary = events / f".{destination.name}.{secrets.token_hex(8)}.tmp"
    try:
        write_exclusive(temporary, canonical_json_bytes(event))
        rename_noreplace(temporary, destination)
        fsync_directory(events)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_event_chain(root: Path, registry: AttemptRegistryV1) -> None:
    events = sorted((root / "events").glob("*.json"))
    if not events:
        raise BuildStateError("build attempt event chain is empty")
    previous: AttemptState | None = None
    for sequence, path in enumerate(events, start=1):
        if path.name != f"{sequence:04d}.json":
            raise BuildStateError("build attempt event sequence is not contiguous")
        event = _read_model(path, AttemptEventV1)
        if (
            event.attempt_id != registry.attempt_id
            or event.sequence != sequence
            or event.from_state is not previous
            or event.to_state not in _LEGAL_TRANSITIONS.get(event.from_state, frozenset())
        ):
            raise BuildStateError("build attempt event chain is inconsistent")
        previous = event.to_state
    if previous is not registry.state:
        raise BuildStateError("build attempt event chain does not reach the registry state")


def _reconcile_missing_event(root: Path, registry: AttemptRegistryV1) -> None:
    events = sorted((root / "events").glob("*.json"))
    if events:
        last = _read_model(events[-1], AttemptEventV1)
        if last.to_state is registry.state:
            return
        previous = last.to_state
        sequence = last.sequence + 1
    else:
        previous = None
        sequence = 1
    if registry.state not in _LEGAL_TRANSITIONS.get(previous, frozenset()):
        raise BuildStateError("build attempt journal cannot reconcile current state")
    _write_event(
        root,
        registry.attempt_id,
        sequence,
        registry.updated_at,
        previous,
        registry.state,
    )


def _transition(
    root: Path,
    registry: AttemptRegistryV1,
    state: AttemptState,
    *,
    changes: Mapping[str, Any] | None = None,
) -> AttemptRegistryV1:
    if state not in _LEGAL_TRANSITIONS.get(registry.state, frozenset()):
        raise BuildStateError(f"illegal build attempt transition: {registry.state} -> {state}")
    sequence = len(tuple((root / "events").glob("*.json"))) + 1
    now = _utc_now()
    updated = AttemptRegistryV1.model_validate(
        registry.model_copy(update={**dict(changes or {}), "state": state, "updated_at": now})
    )
    if state is AttemptState.RECORD_PUBLISHED and updated.record_sha256 is None:
        raise BuildStateError("record publication transition requires a record digest")
    _atomic_model(root / "current.json", updated)
    _write_event(root, registry.attempt_id, sequence, now, registry.state, state)
    return updated


def _recipe_index_path(layout: _BuildLayout, recipe_id: str) -> Path:
    return layout.recipe_indexes / f"{recipe_id.removeprefix('recipe-sha256:')}.json"


def _validate_state_cardinality(
    state: AttemptState,
    outcome: AttemptOutcome | None,
    record_sha256: str | None,
    *,
    subject: str,
) -> None:
    if (state in _OUTCOME_STATES) != (outcome is not None):
        raise BuildStateError(f"{subject} state/outcome combination is invalid")
    if (state in _RECORDED_STATES) != (record_sha256 is not None):
        raise BuildStateError(f"{subject} record cardinality is invalid")


def _load_recipe_index(layout: _BuildLayout, recipe_id: str) -> RecipeIndexV1:
    path = _recipe_index_path(layout, recipe_id)
    if not path.exists():
        return RecipeIndexV1(recipe_id=recipe_id, attempts=())
    index = _read_model(path, RecipeIndexV1)
    if index.recipe_id != recipe_id:
        raise BuildStateError("recipe index identity changed")
    identifiers = [entry.attempt_id for entry in index.attempts]
    if len(identifiers) != len(set(identifiers)):
        raise BuildStateError("recipe index contains duplicate attempts")
    for entry in index.attempts:
        _validate_state_cardinality(
            entry.state,
            entry.outcome,
            entry.record_sha256,
            subject="recipe index",
        )
    return index


def _store_recipe_entry(layout: _BuildLayout, recipe_id: str, entry: AttemptIndexEntryV1) -> None:
    index = _load_recipe_index(layout, recipe_id)
    values = {value.attempt_id: value for value in index.attempts}
    values[entry.attempt_id] = entry
    updated = RecipeIndexV1(
        recipe_id=recipe_id,
        attempts=tuple(values[name] for name in sorted(values)),
    )
    _atomic_model(_recipe_index_path(layout, recipe_id), updated)


def _allocate_attempt(
    layout: _BuildLayout,
    recipe_id: str,
    nonce_factory: Callable[[], bytes] | None,
) -> BuildAttemptSession:
    owner = _current_owner()
    make_nonce = nonce_factory or (lambda: secrets.token_bytes(16))
    for _ in range(32):
        nonce = make_nonce()
        if not isinstance(nonce, bytes) or len(nonce) != 16:
            raise ValueError("attempt nonce factory must return exactly 16 bytes")
        identifier = attempt_id(recipe_id, nonce)
        root = layout.attempts / identifier
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            if nonce_factory is not None:
                raise BuildStateError("build attempt identifier collision") from None
            continue
        (root / "events").mkdir(mode=0o700)
        metadata = root.lstat()
        now = _utc_now()
        registry = AttemptRegistryV1(
            attempt_id=identifier,
            recipe_id=recipe_id,
            nonce=nonce.hex(),
            state=AttemptState.ALLOCATED,
            outcome=None,
            owner=owner,
            root_device=metadata.st_dev,
            root_inode=metadata.st_ino,
            created_at=now,
            updated_at=now,
        )
        _atomic_model(root / "current.json", registry)
        _write_event(root, identifier, 1, now, None, AttemptState.ALLOCATED)
        fsync_directory(root)
        fsync_directory(layout.attempts)
        _store_recipe_entry(
            layout,
            recipe_id,
            AttemptIndexEntryV1(
                attempt_id=identifier,
                state=AttemptState.ALLOCATED,
                outcome=None,
            ),
        )
        return BuildAttemptSession(layout, registry)
    raise BuildStateError("unable to allocate a unique build attempt")


def _validate_attempt_root(root: Path, registry: AttemptRegistryV1) -> None:
    metadata = root.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_dev != registry.root_device
        or metadata.st_ino != registry.root_inode
        or root.name != registry.attempt_id
    ):
        raise BuildStateError("build attempt ownership identity changed")
    _validate_state_cardinality(
        registry.state,
        registry.outcome,
        registry.record_sha256,
        subject="attempt registry",
    )
    _reconcile_missing_event(root, registry)
    _validate_event_chain(root, registry)


def _validate_attempt_identity(registry: AttemptRegistryV1) -> None:
    expected = attempt_id(registry.recipe_id, bytes.fromhex(registry.nonce))
    if expected != registry.attempt_id:
        raise BuildStateError("attempt ID is not derived from its recipe and nonce")


def _verify_record(path: Path, *, error: str) -> RecordVerification:
    try:
        return verify_record(path)
    except BuildRecordError as exc:
        raise BuildStateError(error) from exc


def _assert_finalized_record_binding(
    root: Path,
    registry: AttemptRegistryV1,
    *,
    recipe_id: str,
    attempt_identifier: str,
    outcome: AttemptOutcome,
    build_id: str | None,
    error: str,
) -> None:
    _validate_attempt_identity(registry)
    if (
        registry.recipe_id != recipe_id
        or registry.attempt_id != attempt_identifier
        or registry.state is not AttemptState.FINALIZING
        or registry.outcome is not outcome
        or registry.build_id != build_id
        or registry.record_sha256 is not None
    ):
        raise BuildStateError(error)
    _validate_event_chain(root, registry)


def _remove_preallocation_orphans(layout: _BuildLayout) -> None:
    """Remove only allocator-shaped roots while the global allocation lock is held."""

    for root in sorted(layout.attempts.iterdir()):
        if _ATTEMPT_RE.fullmatch(root.name) is None or (root / "current.json").exists():
            continue
        metadata = root.lstat()
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
        ):
            continue
        removable = True
        for entry in root.iterdir():
            entry_metadata = entry.lstat()
            if entry.name == "events":
                removable = (
                    not stat.S_ISLNK(entry_metadata.st_mode)
                    and stat.S_ISDIR(entry_metadata.st_mode)
                    and entry_metadata.st_uid == os.geteuid()
                    and not any(entry.iterdir())
                )
            elif _ALLOCATION_TEMP_RE.fullmatch(entry.name) is not None:
                removable = (
                    stat.S_ISREG(entry_metadata.st_mode) and entry_metadata.st_uid == os.geteuid()
                )
            else:
                removable = False
            if not removable:
                break
        if removable:
            shutil.rmtree(root)
            fsync_directory(layout.attempts)


def _finalize_attempt(
    layout: _BuildLayout,
    registry: AttemptRegistryV1,
    outcome: AttemptOutcome,
    *,
    build_id: str | None,
) -> AttemptResult:
    if build_id is not None and _BUILD_RE.fullmatch(build_id) is None:
        raise ValueError("invalid machine-local build ID")
    if outcome in {AttemptOutcome.SUCCESS, AttemptOutcome.CACHE_HIT} and build_id is None:
        raise BuildStateError("successful attempt outcomes require a build ID")
    root = layout.attempts / registry.attempt_id
    _validate_attempt_root(root, registry)
    _remove_private_attempt_state(root)
    if registry.state is AttemptState.RECORD_PUBLISHED:
        destination = layout.attempt_records / registry.attempt_id
        verification = _verify_record(
            destination, error="published attempt record verification failed"
        )
        if registry.record_sha256 != verification.record_sha256:
            raise BuildStateError("published attempt record digest changed")
        recorded_registry = _read_model(destination / "current.json", AttemptRegistryV1)
        terminal = recorded_registry
    elif registry.state is AttemptState.FINALIZING:
        if registry.outcome is not outcome or registry.build_id != build_id:
            raise BuildStateError("finalizing attempt outcome changed during recovery")
        terminal = registry
    else:
        terminal = _transition(
            root,
            registry,
            AttemptState.FINALIZING,
            changes={"outcome": outcome, "build_id": build_id},
        )
        _store_recipe_entry(
            layout,
            registry.recipe_id,
            AttemptIndexEntryV1(
                attempt_id=registry.attempt_id,
                state=AttemptState.FINALIZING,
                outcome=outcome,
                build_id=build_id,
            ),
        )
    if registry.state is not AttemptState.RECORD_PUBLISHED:
        destination = layout.attempt_records / registry.attempt_id
        if destination.exists():
            verification = _verify_record(
                destination, error="existing attempt record verification failed"
            )
            if record_source_digest(root) != verification.record_sha256:
                raise BuildStateError("divergent immutable attempt record collision")
        else:
            verification = publish_record(root, destination)
        _transition(
            root,
            terminal,
            AttemptState.RECORD_PUBLISHED,
            changes={"record_sha256": verification.record_sha256},
        )
        recorded_registry = _read_model(destination / "current.json", AttemptRegistryV1)
        if recorded_registry != terminal:
            raise BuildStateError("published attempt record is not bound to the terminal registry")
    _assert_finalized_record_binding(
        destination,
        recorded_registry,
        recipe_id=registry.recipe_id,
        attempt_identifier=registry.attempt_id,
        outcome=outcome,
        build_id=build_id,
        error="published attempt record has inconsistent terminal state",
    )
    _store_recipe_entry(
        layout,
        registry.recipe_id,
        AttemptIndexEntryV1(
            attempt_id=registry.attempt_id,
            state=AttemptState.RECORD_PUBLISHED,
            outcome=outcome,
            build_id=build_id,
            record_sha256=verification.record_sha256,
        ),
    )
    shutil.rmtree(root)
    fsync_directory(layout.attempts)
    _store_recipe_entry(
        layout,
        registry.recipe_id,
        AttemptIndexEntryV1(
            attempt_id=registry.attempt_id,
            state=AttemptState.TORN_DOWN,
            outcome=outcome,
            build_id=build_id,
            record_sha256=verification.record_sha256,
        ),
    )
    return AttemptResult(
        attempt_id=terminal.attempt_id,
        recipe_id=terminal.recipe_id,
        outcome=outcome,
        build_id=build_id,
        record=destination,
        record_sha256=verification.record_sha256,
    )


def _remove_private_attempt_state(root: Path) -> None:
    private = root / "private"
    try:
        metadata = private.lstat()
    except FileNotFoundError:
        return
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise BuildStateError(f"unsafe private build-attempt state: {private}")
    shutil.rmtree(private)
    fsync_directory(root)


def _recover_stale_attempts(layout: _BuildLayout, recipe_id: str) -> tuple[AttemptResult, ...]:
    results: list[AttemptResult] = []
    for root in sorted(layout.attempts.iterdir()):
        try:
            metadata = root.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise BuildStateError(f"unsafe build attempt entry: {root}")
        if not (root / "current.json").exists():
            continue
        registry = _read_model(root / "current.json", AttemptRegistryV1)
        _validate_attempt_identity(registry)
        if registry.recipe_id != recipe_id:
            continue
        _validate_attempt_root(root, registry)
        if _owner_alive(registry.owner):
            raise BuildBusyError(f"build attempt is still owned by PID {registry.owner.pid}")
        recovered_outcome = (
            registry.outcome
            if registry.state in {AttemptState.FINALIZING, AttemptState.RECORD_PUBLISHED}
            and registry.outcome is not None
            else AttemptOutcome.INTERRUPTED
        )
        results.append(
            _finalize_attempt(
                layout,
                registry,
                recovered_outcome,
                build_id=registry.build_id,
            )
        )
    return tuple(results)


def _recover_index_teardowns(layout: _BuildLayout, recipe_id: str) -> None:
    index = _load_recipe_index(layout, recipe_id)
    for entry in index.attempts:
        if entry.state is not AttemptState.RECORD_PUBLISHED:
            continue
        if entry.record_sha256 is None or entry.outcome is None:
            raise BuildStateError("record-published attempt index is incomplete")
        record = layout.attempt_records / entry.attempt_id
        verification = _verify_record(record, error="record-published attempt verification failed")
        if verification.record_sha256 != entry.record_sha256:
            raise BuildStateError("record-published attempt digest changed")
        recorded = _read_model(
            record / "current.json",
            AttemptRegistryV1,
        )
        _assert_finalized_record_binding(
            record,
            recorded,
            recipe_id=recipe_id,
            attempt_identifier=entry.attempt_id,
            outcome=entry.outcome,
            build_id=entry.build_id,
            error="record-published attempt is not bound to its index",
        )
        root = layout.attempts / entry.attempt_id
        if root.exists():
            registry = _read_model(root / "current.json", AttemptRegistryV1)
            _validate_attempt_root(root, registry)
            if (
                registry.state is not AttemptState.RECORD_PUBLISHED
                or registry.recipe_id != recipe_id
                or registry.attempt_id != entry.attempt_id
                or registry.outcome is not entry.outcome
                or registry.build_id != entry.build_id
                or registry.record_sha256 != entry.record_sha256
            ):
                raise BuildStateError("attempt teardown state is inconsistent")
            shutil.rmtree(root)
            fsync_directory(layout.attempts)
        _store_recipe_entry(
            layout,
            recipe_id,
            entry.model_copy(update={"state": AttemptState.TORN_DOWN}),
        )


def build_snapshot_directory(*, home: Path) -> Path:
    """Return the owned shared snapshot directory for build-attempt leases."""

    return _layout(home, create=True).snapshots


@contextlib.contextmanager
def begin_build_attempt(
    recipe_id: str,
    *,
    home: Path,
    nonce_factory: Callable[[], bytes] | None = None,
) -> Iterator[BuildAttemptSession]:
    """Hold the recipe lock through one fresh, never-resumed build attempt."""

    if _RECIPE_RE.fullmatch(recipe_id) is None:
        raise ValueError("invalid portable build recipe ID")
    layout = _layout(home, create=True)
    lock_path = layout.locks / f"build-recipe-{recipe_id.removeprefix('recipe-sha256:')}.lock"
    with exclusive_lock(lock_path) as lock:
        if not lock.acquired:
            raise BuildBusyError(lock.reason or "build recipe lock is unavailable")
        allocation_lock_path = layout.locks / "build-attempt-allocation.lock"
        with exclusive_lock(allocation_lock_path) as allocation_lock:
            if not allocation_lock.acquired:
                raise BuildBusyError(
                    allocation_lock.reason or "build attempt allocation lock is unavailable"
                )
            _remove_preallocation_orphans(layout)
        _recover_stale_attempts(layout, recipe_id)
        _recover_index_teardowns(layout, recipe_id)
        with exclusive_lock(allocation_lock_path) as allocation_lock:
            if not allocation_lock.acquired:
                raise BuildBusyError(
                    allocation_lock.reason or "build attempt allocation lock is unavailable"
                )
            session = _allocate_attempt(layout, recipe_id, nonce_factory)
        try:
            yield session
        finally:
            if not session._finalized:
                if session.root.exists():
                    durable = _read_model(session.root / "current.json", AttemptRegistryV1)
                    session.registry = durable
                    if durable.state in {
                        AttemptState.FINALIZING,
                        AttemptState.RECORD_PUBLISHED,
                    }:
                        if durable.outcome is None:
                            raise BuildStateError("durable finalization has no outcome")
                        session.result = _finalize_attempt(
                            layout,
                            durable,
                            durable.outcome,
                            build_id=durable.build_id,
                        )
                        session._finalized = True
                    else:
                        _write_attempt_file(
                            session.root,
                            PurePosixPath("terminal.json"),
                            canonical_json_bytes(
                                {"outcome": AttemptOutcome.FAILED, "schema_version": 1}
                            ),
                        )
                        session.finalize(AttemptOutcome.FAILED)
                else:
                    _recover_index_teardowns(layout, recipe_id)


def _inspect_recipe_locked(recipe_id: str, layout: _BuildLayout) -> RecipeIndexV1:
    index = _load_recipe_index(layout, recipe_id)
    for entry in index.attempts:
        if entry.record_sha256 is None:
            root = layout.attempts / entry.attempt_id
            if not root.is_dir():
                raise BuildStateError("active attempt index has no mutable attempt root")
            registry = _read_model(root / "current.json", AttemptRegistryV1)
            _validate_attempt_identity(registry)
            if (
                registry.recipe_id != recipe_id
                or registry.attempt_id != entry.attempt_id
                or registry.state is not entry.state
                or registry.outcome is not None
                or registry.build_id is not None
            ):
                raise BuildStateError("active attempt index is not bound to its registry")
            _validate_attempt_root(root, registry)
            continue
        if entry.state is not AttemptState.TORN_DOWN:
            raise BuildStateError("finalized attempt index has incomplete teardown state")
        record = layout.attempt_records / entry.attempt_id
        verification = _verify_record(record, error="attempt record digest verification failed")
        if verification.record_sha256 != entry.record_sha256:
            raise BuildStateError("attempt record digest does not match recipe index")
        registry = _read_model(
            record / "current.json",
            AttemptRegistryV1,
        )
        if entry.outcome is None:
            raise BuildStateError("finalized attempt index has no outcome")
        _assert_finalized_record_binding(
            record,
            registry,
            recipe_id=recipe_id,
            attempt_identifier=entry.attempt_id,
            outcome=entry.outcome,
            build_id=entry.build_id,
            error="attempt record is not bound to its recipe index",
        )
    return index


def inspect_recipe(recipe_id: str, *, home: Path) -> RecipeIndexV1:
    """Verify and return the durable attempt index for one portable recipe."""

    if _RECIPE_RE.fullmatch(recipe_id) is None:
        raise ValueError("invalid portable build recipe ID")
    layout = _layout(home, create=False)
    lock_path = layout.locks / f"build-recipe-{recipe_id.removeprefix('recipe-sha256:')}.lock"
    with exclusive_lock(lock_path) as lock:
        if not lock.acquired:
            raise BuildBusyError(lock.reason or "build recipe lock is unavailable")
        return _inspect_recipe_locked(recipe_id, layout)

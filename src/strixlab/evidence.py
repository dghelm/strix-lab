"""Trustworthy, reusable run-evidence boundary: allocation, state, and records.

A run is allocated as a complete, fsynced, no-replace staged tree under ``active/``,
advances through an append-only authenticated event journal, and finalizes — on both
success and failure — into an immutable, content-addressed run record with canonical
``checksums.sha256`` and a bound index. Recovery is crash-forward and never regresses
or overwrites a divergent record. Run code is deliberately independent of the build
recipe/materialization machinery.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from strixlab.locks import LockStatus, exclusive_lock
from strixlab.manifests import DASH_ID_PATTERN
from strixlab.records import (
    RecordError,
    RecordVerification,
    hash_owned_tree,
    publish_record,
    verify_record,
)
from strixlab.run_paths import RunStorageRoots, prepare_run_storage, run_storage_roots
from strixlab.secret_policy import (
    RedactionContext,
    SensitiveInterpolationError,
    UnsafeOutputError,
    reject_sensitive_interpolations,
)
from strixlab.secure_fs import (
    UnownedDirectoryError,
    directory_open_flags,
    exclusive_create_flags,
    fsync_directory,
    fsync_tree,
    open_owned_directory,
    readonly_open_flags,
    rename_noreplace,
    rename_noreplace_at,
    try_open_owned_directory,
    write_all,
    write_exclusive,
)
from strixlab.serialization import canonical_json_bytes, canonical_yaml_bytes
from strixlab.source_identity import length_frame

_SHA256 = r"^[0-9a-f]{64}$"
_MODEL_LIMIT = 64 * 1024 * 1024
_MAX_PATH_BYTES = 255
_MAX_PORTABLE_ENTRIES = 1024
_MAX_MEMBER_BYTES = 16 * 1024 * 1024
_MAX_AGGREGATE_BYTES = 64 * 1024 * 1024
_MAX_TOTAL_FILES = 4096
_CHECKSUMS_NAME = "checksums.sha256"
_STATUS_NAME = "status.json"
_DESCRIPTOR_NAME = "run.json"
_INPUT_MANIFEST = "manifest.input.yaml"
_RESOLVED_MANIFEST = "manifest.resolved.yaml"
_RECORD_MANIFEST = "record-manifest.json"
_RESERVED_CONTROL_FILES = frozenset(
    {
        _DESCRIPTOR_NAME,
        _STATUS_NAME,
        _INPUT_MANIFEST,
        _RESOLVED_MANIFEST,
        _CHECKSUMS_NAME,
        _RECORD_MANIFEST,
    }
)
_RESERVED_CONTROL_PREFIXES = ("events/", "portable/")
_RUN_ID_RE = re.compile(r"^run-[0-9]{8}T[0-9]{6}Z-[a-z][a-z0-9]*(?:-[a-z0-9]+)*-[0-9a-f]{32}$")
_STAGE_TEMP_RE = re.compile(r"^\.(?P<name>run-[0-9A-Za-z-]+)\.[0-9a-f]{16}\.tmp$")
# A committed event or portable-entry file: an 8-digit zero-padded sequence plus
# ``.json``. The two share one grammar; the temp forms add the writer-temp suffix.
_NUMBERED_JSON_RE = re.compile(r"^(?P<sequence>[0-9]{8})\.json$")
_NUMBERED_JSON_TEMP_RE = re.compile(r"^\.(?P<sequence>[0-9]{8})\.json\.[0-9a-f]{16}\.tmp$")
_EVENT_NAME_RE = _NUMBERED_JSON_RE
_EVENT_TEMP_RE = _NUMBERED_JSON_TEMP_RE
_ENTRY_NAME_RE = _NUMBERED_JSON_RE
_ENTRY_TEMP_RE = _NUMBERED_JSON_TEMP_RE
_BLOB_NAME_RE = re.compile(r"^[0-9a-f]{64}$")
_BLOB_TEMP_RE = re.compile(r"^\.(?P<digest>[0-9a-f]{64})\.[0-9a-f]{16}\.tmp$")
# Strict namespaces for the run-root atomic-writer temps (status projection and the
# checksum file), plus a catch-all for any other stray writer temp at the run root.
_STATUS_TEMP_RE = re.compile(r"^\.status\.json\.[0-9a-f]{16}\.tmp$")
_CHECKSUMS_TEMP_RE = re.compile(r"^\.checksums\.sha256\.[0-9a-f]{16}\.tmp$")
_ROOT_TEMP_RE = re.compile(r"^\..*\.tmp$")

Clock = Callable[[], datetime]
TokenFactory = Callable[[], bytes]

_ROLES = frozenset(
    {
        "environment",
        "source",
        "build",
        "correctness",
        "samples",
        "profiler-summary",
        "comparison",
        "summary",
    }
)
_MEDIA_TYPES = frozenset(
    {
        "application/json",
        "application/x-ndjson",
        "application/yaml",
        "text/plain",
        "text/csv",
        "text/markdown",
        "text/x-diff",
    }
)

# Public views of the closed portable policy, shared with bundle verification so the
# limits, role/media enums, and numbered-file grammar are declared exactly once.
PORTABLE_ROLES = _ROLES
PORTABLE_MEDIA_TYPES = _MEDIA_TYPES
MAX_PATH_BYTES = _MAX_PATH_BYTES
MAX_MEMBER_BYTES = _MAX_MEMBER_BYTES
MAX_AGGREGATE_BYTES = _MAX_AGGREGATE_BYTES
MAX_TOTAL_FILES = _MAX_TOTAL_FILES
MAX_PORTABLE_ENTRIES = _MAX_PORTABLE_ENTRIES
NUMBERED_JSON_NAME_RE = _NUMBERED_JSON_RE
BLOB_NAME_RE = _BLOB_NAME_RE


class RunError(RuntimeError):
    """Run evidence state is unsafe, inconsistent, corrupt, or divergent."""


@dataclass(frozen=True, slots=True)
class RunBusy:
    """A run is currently owned by a live process and cannot be recovered."""

    run_id: str


class RunState(StrEnum):
    ALLOCATED = "allocated"
    ACTIVE = "active"
    TERMINAL = "terminal"


class RunOutcome(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    INTERRUPTED = "interrupted"


class _StoredModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _check_terminal_metadata(
    is_terminal: bool, outcome: RunOutcome | None, reason: str | None
) -> None:
    """Shared legality of ``outcome``/``reason`` against terminality.

    A terminal state carries an outcome (its reason stays optional); a nonterminal
    state must carry neither an outcome nor a reason. Centralized so every producer
    and reader — transitions, committed-chain verification, orphan/temp adoption, and
    bundle verification — enforces the identical predicate.
    """

    if is_terminal:
        if outcome is None:
            raise ValueError("a terminal run state requires an outcome")
    else:
        if outcome is not None:
            raise ValueError("a nonterminal run state must not declare an outcome")
        if reason is not None:
            raise ValueError("a nonterminal run state must not declare a reason")


class RunOwnerV1(_StoredModel):
    schema_version: Literal[1] = 1
    pid: int = Field(gt=0)
    boot_id: str = Field(min_length=1)
    process_start_ticks: int = Field(gt=0)
    uid: int = Field(ge=0)


class RunDescriptorV1(_StoredModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=_RUN_ID_RE.pattern)
    experiment_id: str
    created_at: str
    input_manifest_sha256: str = Field(pattern=_SHA256)
    resolved_manifest_sha256: str = Field(pattern=_SHA256)
    owner: RunOwnerV1
    stage_device: int = Field(ge=0)
    stage_inode: int = Field(gt=0)


class RunEventV1(_StoredModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=_RUN_ID_RE.pattern)
    sequence: int = Field(gt=0)
    previous_sha256: str | None = Field(default=None, pattern=_SHA256)
    from_state: RunState | None
    to_state: RunState
    timestamp: str
    outcome: RunOutcome | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _validate_metadata(self) -> RunEventV1:
        _check_terminal_metadata(self.to_state is RunState.TERMINAL, self.outcome, self.reason)
        return self


class RunStatusV1(_StoredModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=_RUN_ID_RE.pattern)
    state: RunState
    sequence: int = Field(gt=0)
    last_event_sha256: str = Field(pattern=_SHA256)
    outcome: RunOutcome | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _validate_metadata(self) -> RunStatusV1:
        _check_terminal_metadata(self.state is RunState.TERMINAL, self.outcome, self.reason)
        return self


class RunIndexV1(_StoredModel):
    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=_RUN_ID_RE.pattern)
    outcome: RunOutcome
    record_sha256: str = Field(pattern=r"^record-sha256:[0-9a-f]{64}$")
    terminal_event_sha256: str = Field(pattern=_SHA256)
    checksums_sha256: str = Field(pattern=_SHA256)
    active_device: int = Field(ge=0)
    active_inode: int = Field(gt=0)


class PortableEvidenceV1(_StoredModel):
    schema_version: Literal[1] = 1
    sequence: int = Field(gt=0)
    logical_path: str
    media_type: str
    role: str
    blob_sha256: str = Field(pattern=_SHA256)
    size_bytes: int = Field(ge=0)


@dataclass(frozen=True, slots=True)
class RunInspection:
    run_id: str
    state: RunState
    outcome: RunOutcome
    record: Path
    record_sha256: str
    checksums_sha256: str


# --------------------------------------------------------------------------- paths


_EXPERIMENT_ID_RE = re.compile(DASH_ID_PATTERN)


def _validate_experiment_id(experiment_id: str) -> str:
    if _EXPERIMENT_ID_RE.fullmatch(experiment_id) is None or len(experiment_id) > 64:
        raise RunError(f"invalid experiment id: {experiment_id!r}")
    return experiment_id


def run_relative(value: str) -> PurePosixPath:
    """Validate one run-relative path *syntax* shared by every reader and writer.

    Rejects empty, backslash, absolute, ``..``, control/newline, noncanonical
    separator, and overlong (> 255 UTF-8 byte) paths; accepts the run's legitimate
    reserved control paths (the adapter-write policy layers on top separately).
    """

    if (
        not value
        or "\\" in value
        or len(value.encode("utf-8")) > _MAX_PATH_BYTES
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise RunError(f"unsafe run-relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
        raise RunError(f"unsafe run-relative path: {value!r}")
    if path.as_posix() != value:
        raise RunError(f"noncanonical run-relative path: {value!r}")
    return path


def _reject_reserved_control(path: PurePosixPath) -> None:
    text = path.as_posix()
    if (
        text in _RESERVED_CONTROL_FILES
        or (len(path.parts) == 1 and _ROOT_TEMP_RE.fullmatch(path.name) is not None)
        or any(
            text == prefix.rstrip("/") or text.startswith(prefix)
            for prefix in _RESERVED_CONTROL_PREFIXES
        )
    ):
        raise RunError(f"adapter evidence cannot write a reserved control path: {text}")


# ------------------------------------------------------------------ secret scanning


def _scan_text_safe(context: RedactionContext, value: str) -> None:
    try:
        reject_sensitive_interpolations(value)
        context.assert_text_safe(value)
    except SensitiveInterpolationError as exc:
        raise RunError("evidence text references a sensitive environment variable") from exc
    except UnsafeOutputError as exc:
        raise RunError("evidence text discloses a sensitive value") from exc


def _scan_payload_safe(context: RedactionContext, payload: bytes) -> None:
    try:
        context.assert_payload_safe(payload)
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            return
        reject_sensitive_interpolations(text)
    except SensitiveInterpolationError as exc:
        raise RunError("evidence payload references a sensitive environment variable") from exc
    except UnsafeOutputError as exc:
        raise RunError("evidence payload discloses a sensitive value") from exc


# ----------------------------------------------------------------------- owner facts


def _boot_id() -> str:
    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise RunError("cannot read the Linux boot identity") from exc
    if not value:
        raise RunError("Linux boot identity is empty")
    return value


def _process_start_ticks(pid: int) -> int:
    try:
        value = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        closing = value.rfind(")")
        ticks = int(value[closing + 2 :].split()[19])
    except (OSError, ValueError, IndexError) as exc:
        raise RunError(f"cannot read process identity for PID {pid}") from exc
    if ticks <= 0:
        raise RunError("process start identity is invalid")
    return ticks


def _current_owner() -> RunOwnerV1:
    pid = os.getpid()
    return RunOwnerV1(
        pid=pid,
        boot_id=_boot_id(),
        process_start_ticks=_process_start_ticks(pid),
        uid=os.geteuid(),
    )


def _owner_alive(owner: RunOwnerV1) -> bool:
    if owner.uid != os.geteuid() or owner.boot_id != _boot_id():
        return False
    try:
        return _process_start_ticks(owner.pid) == owner.process_start_ticks
    except RunError:
        return False


# ------------------------------------------------------------------- layout helpers


def _validate_run_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise RunError("StrixLab run storage does not exist") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RunError(f"unsafe run storage directory: {path}")


def _layout(home: Path, *, create: bool) -> RunStorageRoots:
    roots = run_storage_roots(home)
    prepare_run_storage(roots, create=create, validate=_validate_run_directory)
    return roots


def _validate_run_id(run_id: str) -> None:
    if _RUN_ID_RE.fullmatch(run_id) is None:
        raise RunError(f"invalid run id: {run_id!r}")


def _active_root(roots: RunStorageRoots, run_id: str) -> Path:
    return roots.active / run_id


def _record_root(roots: RunStorageRoots, run_id: str) -> Path:
    return roots.records / run_id


def _index_path(roots: RunStorageRoots, run_id: str) -> Path:
    return roots.indexes / f"{run_id}.json"


def _lock_path(roots: RunStorageRoots, run_id: str) -> Path:
    return roots.locks / f"{run_id}.lock"


# ------------------------------------------------------------------- event framing


def _event_bytes(event: RunEventV1) -> tuple[bytes, str]:
    payload = canonical_json_bytes(event.model_dump(mode="json"))
    digest = hashlib.sha256(
        length_frame("strixlab.run.event.v1", (("event", payload),))
    ).hexdigest()
    return payload, digest


def _event_links(
    event: RunEventV1,
    *,
    run_id: str,
    sequence: int,
    previous: str | None,
    from_state: RunState | None,
) -> bool:
    return (
        event.run_id == run_id
        and event.sequence == sequence
        and event.previous_sha256 == previous
        and event.from_state is from_state
    )


_LEGAL: dict[RunState | None, frozenset[RunState]] = {
    None: frozenset({RunState.ALLOCATED}),
    RunState.ALLOCATED: frozenset({RunState.ACTIVE, RunState.TERMINAL}),
    RunState.ACTIVE: frozenset({RunState.TERMINAL}),
    RunState.TERMINAL: frozenset(),
}


def validate_event_chain(payloads: Sequence[bytes], *, run_id: str, status: RunStatusV1) -> None:
    """Authenticate an ordered event payload chain and bind it to ``status``.

    The single shared algorithm used by both committed-chain recovery (which reads
    the ordered payloads from a held directory descriptor) and bundle verification
    (which reads them from an in-memory member snapshot): identical linking,
    transition legality, and exact projection onto the recorded status. Callers own
    reading and enumeration; this owns the chain semantics.
    """

    if len(payloads) != status.sequence:
        raise RunError("run event chain length diverged from its status")
    previous: str | None = None
    state: RunState | None = None
    last: RunEventV1 | None = None
    for index, payload in enumerate(payloads, start=1):
        event = _parse_model(payload, RunEventV1)
        _payload, digest = _event_bytes(event)
        if not _event_links(
            event, run_id=run_id, sequence=index, previous=previous, from_state=state
        ):
            raise RunError("run event chain is inconsistent")
        if event.to_state not in _LEGAL.get(state, frozenset()):
            raise RunError("run event chain has an illegal transition")
        previous, state, last = digest, event.to_state, event
    if last is None:
        raise RunError("run event chain is empty")
    if (
        previous != status.last_event_sha256
        or state is not status.state
        or last.outcome is not status.outcome
        or last.reason != status.reason
    ):
        raise RunError("run status diverged from its event chain")


# ------------------------------------------------------------------------ low-level


def _read_bytes(path: Path, limit: int = _MODEL_LIMIT) -> bytes:
    try:
        descriptor = os.open(path, readonly_open_flags())
    except OSError as exc:
        raise RunError(f"run file is unavailable: {path.name}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
            raise RunError("run file is unsafe")
        content = os.read(descriptor, limit + 1)
        if len(content) > limit:
            raise RunError("run file is oversized")
        return content
    finally:
        os.close(descriptor)


def _read_model[ModelT: BaseModel](path: Path, model: type[ModelT]) -> ModelT:
    return _parse_model(_read_bytes(path), model)


def _parse_model[ModelT: BaseModel](payload: bytes, model: type[ModelT]) -> ModelT:
    try:
        return model.model_validate_json(payload, strict=True)
    except ValidationError as exc:
        raise RunError("stored run model is invalid") from exc


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    try:
        write_exclusive(temporary, payload, 0o600)
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    fsync_directory(path.parent)


def _write_no_replace(path: Path, payload: bytes, mode: int) -> None:
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    try:
        write_exclusive(temporary, payload, mode)
        rename_noreplace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    fsync_directory(path.parent)


def _atomic_write_at(active: Path, name: str, payload: bytes) -> None:
    """Descriptor-anchored atomic replace of a direct child ``name`` under ``active``."""

    root_fd = _open_owned_directory_fd(active)
    try:
        temporary = f".{name}.{secrets.token_hex(8)}.tmp"
        fd = os.open(temporary, exclusive_create_flags(), 0o600, dir_fd=root_fd)
        try:
            write_all(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.rename(temporary, name, src_dir_fd=root_fd, dst_dir_fd=root_fd)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=root_fd)
            raise
        os.fsync(root_fd)
    finally:
        os.close(root_fd)


_DIR_OPEN_FLAGS = directory_open_flags()


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


def _hash_owned_fd(descriptor: int) -> tuple[int, str]:
    """Size and sha256 of an already-open owned regular file, race-checked."""

    digest = hashlib.sha256()
    size = 0
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid():
        raise RunError("portable blob is not an owned regular file")
    while chunk := os.read(descriptor, 64 * 1024):
        digest.update(chunk)
        size += len(chunk)
    after = os.fstat(descriptor)
    if before.st_dev != after.st_dev or before.st_ino != after.st_ino or size != before.st_size:
        raise RunError("portable blob changed while hashing")
    return size, digest.hexdigest()


# ------------------------------------------------------- descriptor-anchored writes


def _open_owned_directory_fd(path: Path) -> int:
    """Open ``path`` as an owned, non-symlink directory and return its descriptor."""

    try:
        return open_owned_directory(path)
    except UnownedDirectoryError as exc:
        raise RunError(f"unsafe run directory: {path.name}") from exc
    except OSError as exc:
        raise RunError(f"run directory is unavailable: {path.name}") from exc


def _open_owned_child_dir(parent_fd: int, name: str) -> int:
    """Open child directory ``name`` no-follow under ``parent_fd``; require it present.

    A thin wrapper over :func:`_try_open_owned_child_dir` that additionally requires
    the child to exist.
    """

    child = _try_open_owned_child_dir(parent_fd, name)
    if child is None:
        raise RunError(f"run subdirectory is unavailable: {name}")
    return child


def _ensure_owned_child_dir(parent_fd: int, name: str) -> int:
    """Create (if absent) then open child directory ``name`` under ``parent_fd``."""

    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    else:
        os.fsync(parent_fd)
    return _open_owned_child_dir(parent_fd, name)


def _descend_owned(root_fd: int, parts: tuple[str, ...], *, create: bool) -> int:
    """Walk ``parts`` under ``root_fd``, returning the final owned directory fd.

    Intermediate descriptors are closed; ``root_fd`` is left open for the caller.
    When ``create`` is true, each missing component is created ``0o700``.
    """

    current = root_fd
    try:
        for name in parts:
            nxt = (
                _ensure_owned_child_dir(current, name)
                if create
                else _open_owned_child_dir(current, name)
            )
            if current is not root_fd:
                os.close(current)
            current = nxt
    except BaseException:
        if current is not root_fd:
            os.close(current)
        raise
    return current


def _write_child_no_replace_at(dir_fd: int, name: str, content: bytes, mode: int) -> None:
    """Crash-atomically create ``name`` under ``dir_fd`` without replacing it.

    A fsynced writer temp is renamed no-replace within the same directory
    descriptor, then the directory is fsynced. ``FileExistsError`` propagates so
    the caller can surface a no-overwrite violation.
    """

    temporary = f".{name}.{secrets.token_hex(8)}.tmp"
    fd = os.open(temporary, exclusive_create_flags(), mode, dir_fd=dir_fd)
    try:
        write_all(fd, content)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        rename_noreplace_at(dir_fd, temporary, dir_fd, name)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary, dir_fd=dir_fd)
        raise
    os.fsync(dir_fd)


def _anchored_write(active: Path, relative: PurePosixPath, content: bytes, mode: int) -> None:
    """Descriptor-anchored, no-follow, no-replace write of one run-relative file."""

    *parents, filename = relative.parts
    root_fd = _open_owned_directory_fd(active)
    try:
        dir_fd = _descend_owned(root_fd, tuple(parents), create=True)
        try:
            _write_child_no_replace_at(dir_fd, filename, content, mode)
        finally:
            if dir_fd != root_fd:
                os.close(dir_fd)
    finally:
        os.close(root_fd)


def _try_open_owned_child_dir(parent_fd: int, name: str) -> int | None:
    """Open child directory ``name`` no-follow, or ``None`` when it does not exist.

    Any existing-but-unsafe entry (symlink, non-directory, foreign owner) fails
    closed rather than being followed or treated as absent.
    """

    try:
        return try_open_owned_directory(name, dir_fd=parent_fd)
    except UnownedDirectoryError as exc:
        raise RunError(f"unsafe run subdirectory: {name}") from exc
    except OSError as exc:
        raise RunError(f"run subdirectory is unavailable: {name}") from exc


def _read_owned_regular_at(dir_fd: int, name: str, limit: int = _MODEL_LIMIT) -> bytes:
    """Read one owned, 0600, non-symlink regular file relative to ``dir_fd``."""

    try:
        descriptor = os.open(name, readonly_open_flags(), dir_fd=dir_fd)
    except OSError as exc:
        raise RunError(f"run file is unavailable: {name}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_size > limit
        ):
            raise RunError(f"unsafe run file: {name}")
        content = os.read(descriptor, limit + 1)
        after = os.fstat(descriptor)
        if (
            len(content) > limit
            or after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
        ):
            raise RunError(f"run file changed while reading: {name}")
        return content
    finally:
        os.close(descriptor)


def _child_exists(dir_fd: int, name: str) -> bool:
    try:
        os.lstat(name, dir_fd=dir_fd)
    except FileNotFoundError:
        return False
    return True


# ------------------------------------------------------------------- event journal


def _events_dir(active: Path) -> Path:
    return active / "events"


def _load_status(active: Path) -> RunStatusV1 | None:
    # Read the status projection descriptor-anchored under a no-follow open of the run
    # root itself, so a symlinked or foreign ``active/<run-id>`` fails closed here — at
    # the first access of a recovery — rather than being followed via an intermediate
    # symlink by a path-based read.
    root_fd = _open_owned_directory_fd(active)
    try:
        if not _child_exists(root_fd, _STATUS_NAME):
            return None
        return _parse_model(_read_owned_regular_at(root_fd, _STATUS_NAME), RunStatusV1)
    finally:
        os.close(root_fd)


def _publish_event(active: Path, event: RunEventV1) -> RunStatusV1:
    payload, digest = _event_bytes(event)
    _anchored_write(active, PurePosixPath(f"events/{event.sequence:08d}.json"), payload, 0o600)
    status = RunStatusV1(
        run_id=event.run_id,
        state=event.to_state,
        sequence=event.sequence,
        last_event_sha256=digest,
        outcome=event.outcome,
        reason=event.reason,
    )
    _atomic_write_at(active, _STATUS_NAME, canonical_json_bytes(status.model_dump(mode="json")))
    return status


def _transition(
    active: Path,
    previous: RunStatusV1 | None,
    to_state: RunState,
    *,
    run_id: str,
    clock: Clock,
    outcome: RunOutcome | None = None,
    reason: str | None = None,
) -> RunStatusV1:
    from_state = previous.state if previous is not None else None
    if to_state not in _LEGAL.get(from_state, frozenset()):
        raise RunError(f"illegal run transition: {from_state} -> {to_state}")
    sequence = 1 if previous is None else previous.sequence + 1
    event = RunEventV1(
        run_id=run_id,
        sequence=sequence,
        previous_sha256=previous.last_event_sha256 if previous is not None else None,
        from_state=from_state,
        to_state=to_state,
        timestamp=_iso(clock()),
        outcome=outcome,
        reason=reason,
    )
    return _publish_event(active, event)


def _reconcile_orphan_event(active: Path, status: RunStatusV1) -> RunStatusV1:
    """Adopt at most one authenticated orphan event beyond the recorded status."""

    _reconcile_event_temp(active, status)
    events = _events_dir(active)
    orphan = events / f"{status.sequence + 1:08d}.json"
    if not os.path.lexists(orphan):
        return status
    event = _read_model(orphan, RunEventV1)
    if not _event_links(
        event,
        run_id=status.run_id,
        sequence=status.sequence + 1,
        previous=status.last_event_sha256,
        from_state=status.state,
    ) or event.to_state not in _LEGAL.get(status.state, frozenset()):
        raise RunError("divergent orphan run event")
    _payload, digest = _event_bytes(event)
    fsync_directory(events)
    adopted = RunStatusV1(
        run_id=event.run_id,
        state=event.to_state,
        sequence=event.sequence,
        last_event_sha256=digest,
        outcome=event.outcome,
        reason=event.reason,
    )
    _atomic_write_at(active, _STATUS_NAME, canonical_json_bytes(adopted.model_dump(mode="json")))
    return adopted


def _reconcile_event_temp(active: Path, status: RunStatusV1) -> None:
    """Authenticate and remove at most one event writer temp, parsing/linking it.

    The temp must be either the byte-identical committed event or the exact
    uncommitted next event (parsing and chaining onto the recorded status). The
    event directory is enumerated strictly: any non-``NNNNNNNN.json`` and
    non-writer-temp entry fails closed.
    """

    root_fd = _open_owned_directory_fd(active)
    try:
        events_fd = _try_open_owned_child_dir(root_fd, "events")
        if events_fd is None:
            return
        try:
            temps: list[str] = []
            for name in os.listdir(events_fd):
                if _EVENT_TEMP_RE.match(name):
                    temps.append(name)
                elif _EVENT_NAME_RE.match(name) is None:
                    raise RunError(f"unexpected run event directory member: {name}")
            if not temps:
                return
            if len(temps) > 1:
                raise RunError("multiple writer temp run events")
            name = temps[0]
            match = _EVENT_TEMP_RE.match(name)
            assert match is not None
            sequence = int(match.group("sequence"))
            data = _read_owned_regular_at(events_fd, name)
            committed_name = f"{sequence:08d}.json"
            if _child_exists(events_fd, committed_name):
                if data != _read_owned_regular_at(events_fd, committed_name):
                    raise RunError("divergent writer temp run event")
            else:
                if sequence != status.sequence + 1:
                    raise RunError("divergent writer temp run event")
                event = _parse_model(data, RunEventV1)
                if not _event_links(
                    event,
                    run_id=status.run_id,
                    sequence=sequence,
                    previous=status.last_event_sha256,
                    from_state=status.state,
                ) or event.to_state not in _LEGAL.get(status.state, frozenset()):
                    raise RunError("divergent writer temp run event")
            os.unlink(name, dir_fd=events_fd)
            os.fsync(events_fd)
        finally:
            os.close(events_fd)
    finally:
        os.close(root_fd)


def _verify_event_chain(active: Path, status: RunStatusV1) -> None:
    root_fd = _open_owned_directory_fd(active)
    try:
        events_fd = _open_owned_child_dir(root_fd, "events")
        try:
            present = set(os.listdir(events_fd))
            expected = {f"{sequence:08d}.json" for sequence in range(1, status.sequence + 1)}
            if present != expected:
                raise RunError("run event directory has unexpected entries")
            payloads = [
                _read_owned_regular_at(events_fd, f"{sequence:08d}.json")
                for sequence in range(1, status.sequence + 1)
            ]
        finally:
            os.close(events_fd)
    finally:
        os.close(root_fd)
    validate_event_chain(payloads, run_id=status.run_id, status=status)


def _required_status(active: Path, run_id: str) -> RunStatusV1:
    status = _load_status(active)
    if status is None:
        raise RunError("run has active state without a status projection")
    if status.run_id != run_id:
        raise RunError("run status is bound to another run")
    status = _reconcile_orphan_event(active, status)
    if status.run_id != run_id:
        raise RunError("run status is bound to another run")
    _verify_event_chain(active, status)
    return status


# ---------------------------------------------------------------------- allocation

_MAX_ALLOC_RETRIES = 8


def _run_id_taken(roots: RunStorageRoots, run_id: str) -> bool:
    """A run ID is taken if any active tree, immutable record, or index exists."""

    return (
        os.path.lexists(_active_root(roots, run_id))
        or os.path.lexists(_record_root(roots, run_id))
        or os.path.lexists(_index_path(roots, run_id))
    )


def _stage_run(
    roots: RunStorageRoots,
    run_id: str,
    *,
    experiment_id: str,
    created_iso: str,
    input_bytes: bytes,
    resolved_bytes: bytes,
    input_sha: str,
    resolved_sha: str,
    owner: RunOwnerV1,
) -> bool:
    """Build the complete initial tree and rename it no-replace into ``active/``.

    Returns ``False`` on an ID-collision rename so the caller retries a new suffix.
    """

    active = _active_root(roots, run_id)
    stage = roots.allocation_staging / f".{run_id}.{secrets.token_hex(8)}.tmp"
    try:
        stage.mkdir(mode=0o700)
        metadata = stage.lstat()
        descriptor = RunDescriptorV1(
            run_id=run_id,
            experiment_id=experiment_id,
            created_at=created_iso,
            input_manifest_sha256=input_sha,
            resolved_manifest_sha256=resolved_sha,
            owner=owner,
            stage_device=metadata.st_dev,
            stage_inode=metadata.st_ino,
        )
        write_exclusive(stage / _INPUT_MANIFEST, input_bytes, 0o600)
        write_exclusive(stage / _RESOLVED_MANIFEST, resolved_bytes, 0o600)
        write_exclusive(
            stage / _DESCRIPTOR_NAME,
            canonical_json_bytes(descriptor.model_dump(mode="json")),
            0o600,
        )
        (stage / "events").mkdir(mode=0o700)
        allocated = RunEventV1(
            run_id=run_id,
            sequence=1,
            previous_sha256=None,
            from_state=None,
            to_state=RunState.ALLOCATED,
            timestamp=created_iso,
        )
        payload, digest = _event_bytes(allocated)
        write_exclusive(stage / "events" / "00000001.json", payload, 0o600)
        status = RunStatusV1(
            run_id=run_id, state=RunState.ALLOCATED, sequence=1, last_event_sha256=digest
        )
        write_exclusive(
            stage / _STATUS_NAME, canonical_json_bytes(status.model_dump(mode="json")), 0o600
        )
        fsync_tree(stage)
        try:
            rename_noreplace(stage, active)
        except FileExistsError:
            shutil.rmtree(stage, ignore_errors=True)
            return False
        fsync_directory(roots.active)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return True


def begin_run(
    experiment_id: str,
    manifest_input: bytes,
    *,
    resolved: Mapping[str, Any],
    home: Path,
    environ: Mapping[str, str],
    clock: Clock | None = None,
    token_factory: TokenFactory | None = None,
) -> RunSession:
    """Allocate a new run and return its ACTIVE session, holding the per-run lock.

    Both manifests are captured (exact input bytes, deterministically serialized
    resolved YAML), secret-scanned, and content-addressed before the run appears.
    """

    _validate_experiment_id(experiment_id)
    now: Clock = clock if clock is not None else (lambda: datetime.now(UTC))
    token: TokenFactory = (
        token_factory if token_factory is not None else (lambda: secrets.token_bytes(16))
    )
    context = RedactionContext.from_environ(environ)
    input_bytes = bytes(manifest_input)
    resolved_bytes = canonical_yaml_bytes(resolved)
    _scan_payload_safe(context, input_bytes)
    _scan_payload_safe(context, resolved_bytes)
    try:
        reject_sensitive_interpolations(resolved)
    except SensitiveInterpolationError as exc:
        raise RunError("resolved manifest references a sensitive environment variable") from exc
    input_sha = hashlib.sha256(input_bytes).hexdigest()
    resolved_sha = hashlib.sha256(resolved_bytes).hexdigest()
    roots = _layout(home, create=True)
    owner = _current_owner()
    created = now().astimezone(UTC)
    created_iso = _iso(created)
    basic = created.strftime("%Y%m%dT%H%M%SZ")
    for _attempt in range(_MAX_ALLOC_RETRIES):
        run_id = f"run-{basic}-{experiment_id}-{token().hex()}"
        _validate_run_id(run_id)
        if _run_id_taken(roots, run_id):
            continue
        stack = contextlib.ExitStack()
        try:
            held = stack.enter_context(exclusive_lock(_lock_path(roots, run_id)))
            if not held.acquired:
                raise RunError(held.reason or "run lock is unavailable")
            if _run_id_taken(roots, run_id):
                stack.close()
                continue
            staged = _stage_run(
                roots,
                run_id,
                experiment_id=experiment_id,
                created_iso=created_iso,
                input_bytes=input_bytes,
                resolved_bytes=resolved_bytes,
                input_sha=input_sha,
                resolved_sha=resolved_sha,
                owner=owner,
            )
            if not staged:
                stack.close()
                continue
            active = _active_root(roots, run_id)
            status = _transition(
                active, _load_status(active), RunState.ACTIVE, run_id=run_id, clock=now
            )
            return RunSession(roots, run_id, context, now, stack, status)
        except BaseException:
            stack.close()
            raise
    raise RunError("run id allocation exhausted its collision retries")


# ------------------------------------------------------------------------- session


@dataclass(slots=True)
class RunSession:
    """Live, lock-holding handle to an ACTIVE run for adapters and tests."""

    roots: RunStorageRoots
    run_id: str
    context: RedactionContext
    clock: Clock
    _stack: contextlib.ExitStack
    _status: RunStatusV1
    _finalized: bool = False

    @property
    def active(self) -> Path:
        return _active_root(self.roots, self.run_id)

    def __enter__(self) -> RunSession:
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> Literal[False]:
        try:
            if self._finalized:
                return False
            if exc is not None:
                # Finalize FAILURE for an escaping exception without masking it.
                with contextlib.suppress(Exception):
                    self._run_finalization(RunOutcome.FAILURE, self._exception_reason(exc))
            else:
                self._run_finalization(RunOutcome.FAILURE, "run-context-exited-without-outcome")
        finally:
            self._finalized = True
            self._stack.close()
        return False

    def write_evidence(self, relative: str, content: bytes) -> Path:
        """Write one local evidence file with the checksum-compatible path grammar."""

        self._require_active()
        path = run_relative(relative)
        _reject_reserved_control(path)
        _scan_text_safe(self.context, path.as_posix())
        _scan_payload_safe(self.context, content)
        try:
            _anchored_write(self.active, path, content, 0o600)
        except FileExistsError as exc:
            raise RunError(f"evidence path already exists: {path.as_posix()}") from exc
        return self.active / path

    def write_portable(
        self, logical_path: str, content: bytes, *, media_type: str, role: str
    ) -> PortableEvidenceV1:
        """Publish one durable, content-addressed portable evidence entry."""

        self._require_active()
        if role not in _ROLES:
            raise RunError(f"unknown portable role: {role!r}")
        if media_type not in _MEDIA_TYPES:
            raise RunError(f"unknown portable media type: {media_type!r}")
        path = run_relative(logical_path)
        _reject_reserved_control(path)
        _scan_text_safe(self.context, logical_path)
        validate_portable_payload(content, media_type)
        _scan_payload_safe(self.context, content)
        if len(content) > _MAX_MEMBER_BYTES:
            raise RunError("portable payload exceeds the per-member limit")
        entries = _load_portable_entries(self.active)
        if any(entry.logical_path == logical_path for entry in entries):
            raise RunError(f"duplicate portable logical path: {logical_path}")
        if len(entries) >= _MAX_PORTABLE_ENTRIES:
            raise RunError("portable evidence exceeds the per-run entry limit")
        blob_sha = hashlib.sha256(content).hexdigest()
        # A blob deduplicated across entries must carry exactly one media type.
        if any(
            entry.blob_sha256 == blob_sha and entry.media_type != media_type for entry in entries
        ):
            raise RunError("portable blob is shared under conflicting media types")
        # Enforce the aggregate-payload and total-file limits at write time, over the
        # deduplicated portable blob set (bundle export/verify re-enforce them too).
        blob_sizes = {entry.blob_sha256: entry.size_bytes for entry in entries}
        if blob_sha not in blob_sizes:
            blob_sizes[blob_sha] = len(content)
        if sum(blob_sizes.values()) > _MAX_AGGREGATE_BYTES:
            raise RunError("portable evidence exceeds the aggregate payload limit")
        if len(blob_sizes) + len(entries) + 1 > _MAX_TOTAL_FILES:
            raise RunError("portable evidence exceeds the total-file limit")
        _publish_blob(self.active, blob_sha, content)
        entry = PortableEvidenceV1(
            sequence=len(entries) + 1,
            logical_path=logical_path,
            media_type=media_type,
            role=role,
            blob_sha256=blob_sha,
            size_bytes=len(content),
        )
        _publish_portable_entry(self.active, entry)
        return entry

    def succeed(self) -> RunInspection:
        return self._finalize_and_release(RunOutcome.SUCCESS, None)

    def fail(self, reason: str) -> RunInspection:
        return self._finalize_and_release(RunOutcome.FAILURE, self._explicit_reason(reason))

    def _finalize_and_release(self, outcome: RunOutcome, reason: str | None) -> RunInspection:
        if self._finalized:
            raise RunError("run is already finalized")
        try:
            return self._run_finalization(outcome, reason)
        finally:
            self._finalized = True
            self._stack.close()

    def _run_finalization(self, outcome: RunOutcome, reason: str | None) -> RunInspection:
        self._status = _transition(
            self.active,
            self._status,
            RunState.TERMINAL,
            run_id=self.run_id,
            clock=self.clock,
            outcome=outcome,
            reason=reason,
        )
        return _complete_finalization(self.roots, self.run_id)

    def _require_active(self) -> None:
        if self._finalized:
            raise RunError("cannot write evidence after the run terminates")

    def _explicit_reason(self, reason: str) -> str:
        reason = reason[:4096]
        _scan_text_safe(self.context, reason)
        return reason

    def _exception_reason(self, exc: BaseException) -> str:
        text = str(exc)[:4096]
        try:
            _scan_text_safe(self.context, text)
        except RunError:
            return "run-failed-with-sensitive-error"
        return text or "run-failed-with-sensitive-error"


def validate_portable_payload(content: bytes, media_type: str) -> None:
    """Enforce the closed portable payload policy (UTF-8, controls, structure).

    Shared by ``write_portable`` and bundle verification so both apply the exact
    same content policy to portable evidence bytes.
    """

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RunError("portable payload is not valid UTF-8") from exc
    if any(
        (ord(character) < 32 and character not in "\t\r\n") or ord(character) == 127
        for character in text
    ):
        raise RunError("portable payload contains disallowed control bytes")
    try:
        if media_type == "application/json":
            json.loads(text)
        elif media_type == "application/x-ndjson":
            for line in text.splitlines():
                if line.strip():
                    json.loads(line)
        elif media_type == "application/yaml":
            yaml.safe_load(text)
    except (ValueError, yaml.YAMLError) as exc:
        raise RunError(f"portable payload is not valid {media_type}") from exc


# -------------------------------------------------------------------------- portable


def _publish_blob(active: Path, blob_sha: str, content: bytes) -> None:
    root_fd = _open_owned_directory_fd(active)
    try:
        blobs_fd = _descend_owned(root_fd, ("portable", "blobs"), create=True)
        try:
            try:
                existing = os.open(blob_sha, readonly_open_flags(), dir_fd=blobs_fd)
            except FileNotFoundError:
                existing = None
            if existing is not None:
                try:
                    size, digest = _hash_owned_fd(existing)
                finally:
                    os.close(existing)
                if digest != blob_sha or size != len(content):
                    raise RunError("existing portable blob diverged from its content address")
                os.fsync(blobs_fd)
                return
            _write_child_no_replace_at(blobs_fd, blob_sha, content, 0o600)
        finally:
            os.close(blobs_fd)
    finally:
        os.close(root_fd)


def _publish_portable_entry(active: Path, entry: PortableEvidenceV1) -> None:
    root_fd = _open_owned_directory_fd(active)
    try:
        entries_fd = _descend_owned(root_fd, ("portable", "entries"), create=True)
        try:
            _write_child_no_replace_at(
                entries_fd,
                f"{entry.sequence:08d}.json",
                canonical_json_bytes(entry.model_dump(mode="json")),
                0o600,
            )
        finally:
            os.close(entries_fd)
    finally:
        os.close(root_fd)


def _validate_entry_policy(entry: PortableEvidenceV1, sequence: int, seen: set[str]) -> None:
    """Enforce the closed portable policy on one parsed entry."""

    if entry.sequence != sequence:
        raise RunError("portable entry sequence mismatch")
    run_relative(entry.logical_path)
    if entry.role not in _ROLES or entry.media_type not in _MEDIA_TYPES:
        raise RunError("portable entry has an out-of-policy role or media type")
    if entry.logical_path in seen:
        raise RunError("duplicate portable logical path")


def _verify_entry_blob(blobs_fd: int | None, entry: PortableEvidenceV1) -> None:
    """Require the entry's referenced blob to exist, be owned, and match digest/size."""

    if blobs_fd is None:
        raise RunError("portable entry references a missing blob directory")
    try:
        descriptor = os.open(entry.blob_sha256, readonly_open_flags(), dir_fd=blobs_fd)
    except OSError as exc:
        raise RunError("portable entry references a missing blob") from exc
    try:
        size, digest = _hash_owned_fd(descriptor)
    finally:
        os.close(descriptor)
    if digest != entry.blob_sha256 or size != entry.size_bytes:
        raise RunError("portable entry diverged from its blob")


def _load_portable_entries(active: Path) -> tuple[PortableEvidenceV1, ...]:
    root_fd = _open_owned_directory_fd(active)
    try:
        portable_fd = _try_open_owned_child_dir(root_fd, "portable")
        if portable_fd is None:
            return ()
        try:
            entries_fd = _try_open_owned_child_dir(portable_fd, "entries")
            if entries_fd is None:
                return ()
            blobs_fd = _try_open_owned_child_dir(portable_fd, "blobs")
            try:
                return _load_entries_locked(entries_fd, blobs_fd)
            finally:
                if blobs_fd is not None:
                    os.close(blobs_fd)
                os.close(entries_fd)
        finally:
            os.close(portable_fd)
    finally:
        os.close(root_fd)


def _load_entries_locked(entries_fd: int, blobs_fd: int | None) -> tuple[PortableEvidenceV1, ...]:
    committed: set[str] = set()
    temps: list[str] = []
    for name in os.listdir(entries_fd):
        if _ENTRY_NAME_RE.match(name):
            committed.add(name)
        elif _ENTRY_TEMP_RE.match(name):
            temps.append(name)
        else:
            raise RunError(f"unexpected portable entry directory member: {name}")
    entries: list[PortableEvidenceV1] = []
    seen: set[str] = set()
    for index, name in enumerate(sorted(committed), start=1):
        if int(name[:8]) != index:
            raise RunError("noncontiguous portable entries")
        entry = _parse_model(_read_owned_regular_at(entries_fd, name), PortableEvidenceV1)
        _validate_entry_policy(entry, index, seen)
        _verify_entry_blob(blobs_fd, entry)
        seen.add(entry.logical_path)
        entries.append(entry)
    _reconcile_entry_temp(entries_fd, blobs_fd, temps, committed, entries, seen)
    _require_single_blob_media(entries)
    return tuple(entries)


def _require_single_blob_media(entries: list[PortableEvidenceV1]) -> None:
    """Every deduplicated blob must be referenced under exactly one media type."""

    media_by_blob: dict[str, str] = {}
    for entry in entries:
        prior = media_by_blob.setdefault(entry.blob_sha256, entry.media_type)
        if prior != entry.media_type:
            raise RunError("portable blob is shared under conflicting media types")


def _reconcile_entry_temp(
    entries_fd: int,
    blobs_fd: int | None,
    temps: list[str],
    committed: set[str],
    entries: list[PortableEvidenceV1],
    seen: set[str],
) -> None:
    """Authenticate and remove at most one portable-entry writer temp."""

    if not temps:
        return
    if len(temps) > 1:
        raise RunError("multiple writer temp portable entries")
    name = temps[0]
    match = _ENTRY_TEMP_RE.match(name)
    assert match is not None
    sequence = int(match.group("sequence"))
    data = _read_owned_regular_at(entries_fd, name)
    committed_name = f"{sequence:08d}.json"
    if committed_name in committed:
        if data != _read_owned_regular_at(entries_fd, committed_name):
            raise RunError("divergent writer temp portable entry")
    else:
        if sequence != len(entries) + 1:
            raise RunError("divergent writer temp portable entry")
        entry = _parse_model(data, PortableEvidenceV1)
        _validate_entry_policy(entry, sequence, seen)
        _verify_entry_blob(blobs_fd, entry)
    os.unlink(name, dir_fd=entries_fd)
    os.fsync(entries_fd)


def _recover_portable(active: Path) -> tuple[PortableEvidenceV1, ...]:
    """Validate portable entries, then reconcile blob orphans and one blob temp."""

    entries = _load_portable_entries(active)
    root_fd = _open_owned_directory_fd(active)
    try:
        portable_fd = _try_open_owned_child_dir(root_fd, "portable")
        if portable_fd is None:
            return entries
        try:
            blobs_fd = _try_open_owned_child_dir(portable_fd, "blobs")
            if blobs_fd is None:
                return entries
            try:
                _reconcile_blobs(blobs_fd, {entry.blob_sha256 for entry in entries})
            finally:
                os.close(blobs_fd)
        finally:
            os.close(portable_fd)
    finally:
        os.close(root_fd)
    return entries


def _reconcile_blobs(blobs_fd: int, referenced: set[str]) -> None:
    committed: list[str] = []
    temp: str | None = None
    for name in os.listdir(blobs_fd):
        if _BLOB_NAME_RE.match(name):
            committed.append(name)
        elif _BLOB_TEMP_RE.match(name):
            if temp is not None:
                raise RunError("multiple writer temp portable blobs")
            temp = name
        else:
            raise RunError(f"unexpected portable blob directory member: {name}")
    for name in committed:
        if name not in referenced:
            _remove_orphan_blob(blobs_fd, name)
    if temp is not None:
        _reconcile_blob_temp(blobs_fd, temp)


def _authenticated_owned_regular(dir_fd: int, name: str, describe: str) -> None:
    metadata = os.lstat(name, dir_fd=dir_fd)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise RunError(describe)


def _remove_orphan_blob(blobs_fd: int, name: str) -> None:
    _authenticated_owned_regular(blobs_fd, name, f"unsafe orphan portable blob: {name}")
    descriptor = os.open(name, readonly_open_flags(), dir_fd=blobs_fd)
    try:
        _size, digest = _hash_owned_fd(descriptor)
    finally:
        os.close(descriptor)
    if digest != name:
        raise RunError("orphan portable blob failed its content-address check")
    os.unlink(name, dir_fd=blobs_fd)
    os.fsync(blobs_fd)


def _reconcile_blob_temp(blobs_fd: int, name: str) -> None:
    """Authenticate one blob writer temp against its own name-derived digest."""

    match = _BLOB_TEMP_RE.match(name)
    assert match is not None
    expected = match.group("digest")
    _authenticated_owned_regular(blobs_fd, name, "unsafe writer temp portable blob")
    descriptor = os.open(name, readonly_open_flags(), dir_fd=blobs_fd)
    try:
        _size, digest = _hash_owned_fd(descriptor)
    finally:
        os.close(descriptor)
    if digest != expected:
        raise RunError("divergent writer temp portable blob")
    os.unlink(name, dir_fd=blobs_fd)
    os.fsync(blobs_fd)


# ------------------------------------------------------------------------ checksums


def _checksum_lines(active: Path) -> bytes:
    files = hash_owned_tree(active, skip=lambda relative: relative == _CHECKSUMS_NAME)
    for entry in files:
        run_relative(entry.path)
    ordered = sorted(files, key=lambda entry: entry.path.encode("utf-8"))
    return "".join(f"{entry.sha256}  {entry.path}\n" for entry in ordered).encode("utf-8")


def _reconcile_control_temps(active: Path) -> None:
    """Remove the run-root atomic-writer temps before checksum generation.

    A crash after a ``status.json`` or ``checksums.sha256`` writer temp is fsynced but
    before it is renamed leaves a strict-namespace temp at the run root; without this,
    checksum generation would hash the stray temp into the immutable record. At most one
    temp per control file is tolerated; a duplicate, or any other unexpected writer temp
    at the run root, fails closed. The authoritative committed status (from the event
    chain) and the freshly regenerated checksums supersede the temps, so they are
    removed rather than adopted.
    """

    root_fd = _open_owned_directory_fd(active)
    try:
        status_temps: list[str] = []
        checksum_temps: list[str] = []
        for name in os.listdir(root_fd):
            if _STATUS_TEMP_RE.match(name):
                status_temps.append(name)
            elif _CHECKSUMS_TEMP_RE.match(name):
                checksum_temps.append(name)
            elif _ROOT_TEMP_RE.match(name):
                raise RunError(f"unexpected run root writer temp: {name}")
        removed = False
        for temps, describe in ((status_temps, _STATUS_NAME), (checksum_temps, _CHECKSUMS_NAME)):
            if len(temps) > 1:
                raise RunError(f"multiple {describe} writer temps")
            for name in temps:
                _authenticated_owned_regular(root_fd, name, f"unsafe {describe} writer temp")
                os.unlink(name, dir_fd=root_fd)
                removed = True
        if removed:
            os.fsync(root_fd)
    finally:
        os.close(root_fd)


def _ensure_checksums(active: Path) -> None:
    path = active / _CHECKSUMS_NAME
    if os.path.lexists(path):
        _verify_checksums(active)
        return
    _write_no_replace(path, _checksum_lines(active), 0o600)


def _verify_checksums(active: Path) -> None:
    declared = _read_bytes(active / _CHECKSUMS_NAME)
    if declared != _checksum_lines(active):
        raise RunError("checksums.sha256 diverged from the run payload")


def parse_checksums(content: bytes) -> dict[str, str]:
    """Parse a canonical ``sha256  path`` checksum file, failing closed on deviation."""

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RunError("checksums file is not valid UTF-8") from exc
    if text and not text.endswith("\n"):
        raise RunError("checksums file lacks a trailing newline")
    result: dict[str, str] = {}
    ordering: list[bytes] = []
    for line in text.splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise RunError("noncanonical checksum line")
        digest, path = line[:64], line[66:]
        if re.fullmatch(_SHA256[1:-1], digest) is None:
            raise RunError("checksum digest is not lowercase SHA-256")
        run_relative(path)
        if path in result:
            raise RunError("duplicate checksum declaration")
        result[path] = digest
        ordering.append(path.encode("utf-8"))
    if ordering != sorted(ordering):
        raise RunError("checksums are not in canonical byte order")
    return result


# --------------------------------------------------------------------- finalization


@dataclass(frozen=True, slots=True)
class _VerifiedRecord:
    descriptor: RunDescriptorV1
    status: RunStatusV1
    verification: RecordVerification
    checksums_sha256: str


def _verify_finalized_record(
    record: Path,
    run_id: str,
    *,
    verification: RecordVerification | None = None,
) -> _VerifiedRecord:
    """Completely verify one immutable finalized record for ``run_id``.

    The single verifier shared by index rebuild and inspection. Beyond the generic
    :func:`verify_record` (which authenticates the exact file set and each file's
    digest against the record manifest), this binds the record's *semantics*: the
    descriptor and status must name this run and be terminal with an outcome; the
    captured manifest digests must match the bundled manifest bytes; the complete
    committed event chain must authenticate and project exactly onto ``status``; and
    ``checksums.sha256`` must declare exactly the record payload set (minus itself)
    with digests matching the authenticated record files. A record whose manifest is
    self-consistent (so generic ``verify_record`` passes) but whose semantics are
    corrupt fails closed here. A fresh publication may supply the
    :class:`RecordVerification` returned by :func:`publish_record`; recovery of an
    existing record omits it and always re-verifies the tree from disk.
    """

    if verification is None:
        verification = verify_record(record)
    descriptor = _read_model(record / _DESCRIPTOR_NAME, RunDescriptorV1)
    if descriptor.run_id != run_id:
        raise RunError("run record descriptor is bound to another run")
    status = _read_model(record / _STATUS_NAME, RunStatusV1)
    if status.run_id != run_id or status.state is not RunState.TERMINAL or status.outcome is None:
        raise RunError("run record is not a terminal record for this run")
    input_bytes = _read_bytes(record / _INPUT_MANIFEST)
    resolved_bytes = _read_bytes(record / _RESOLVED_MANIFEST)
    if (
        descriptor.input_manifest_sha256 != hashlib.sha256(input_bytes).hexdigest()
        or descriptor.resolved_manifest_sha256 != hashlib.sha256(resolved_bytes).hexdigest()
    ):
        raise RunError("run record manifests diverged from its descriptor")
    _verify_event_chain(record, status)
    checksums_bytes = _read_bytes(record / _CHECKSUMS_NAME)
    _verify_record_checksums(verification, checksums_bytes)
    return _VerifiedRecord(
        descriptor=descriptor,
        status=status,
        verification=verification,
        checksums_sha256=hashlib.sha256(checksums_bytes).hexdigest(),
    )


def _verify_record_checksums(verification: RecordVerification, checksums_bytes: bytes) -> None:
    declared = parse_checksums(checksums_bytes)
    record_files = {entry.path: entry.sha256 for entry in verification.files}
    if set(declared) != set(record_files) - {_CHECKSUMS_NAME}:
        raise RunError("run record checksums do not cover the exact payload set")
    for path, digest in declared.items():
        if record_files.get(path) != digest:
            raise RunError("run record checksum diverged from its payload")


def _build_index(verified: _VerifiedRecord) -> RunIndexV1:
    assert verified.status.outcome is not None  # guaranteed by _verify_finalized_record
    return RunIndexV1(
        run_id=verified.descriptor.run_id,
        outcome=verified.status.outcome,
        record_sha256=verified.verification.record_sha256,
        terminal_event_sha256=verified.status.last_event_sha256,
        checksums_sha256=verified.checksums_sha256,
        active_device=verified.descriptor.stage_device,
        active_inode=verified.descriptor.stage_inode,
    )


def _publish_index(index_path: Path, index: RunIndexV1) -> None:
    payload = canonical_json_bytes(index.model_dump(mode="json"))
    if os.path.lexists(index_path):
        if _read_bytes(index_path) != payload:
            raise RunError("divergent run index collision")
        return
    _write_no_replace(index_path, payload, 0o600)


def _rmtree_fd(dir_fd: int) -> None:
    """Recursively delete the CONTENTS of the directory held by ``dir_fd``.

    Every entry is removed relative to a held, no-follow directory descriptor and each
    subdirectory is descended through its own held descriptor, so no ancestor name is
    re-resolved: a same-uid rename of an ancestor between operations cannot redirect
    the deletion to another tree. The directory itself is left empty for the caller.
    """

    for name in os.listdir(dir_fd):
        metadata = os.lstat(name, dir_fd=dir_fd)
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            child_fd = os.open(name, _DIR_OPEN_FLAGS, dir_fd=dir_fd)
            try:
                _rmtree_fd(child_fd)
            finally:
                os.close(child_fd)
            os.rmdir(name, dir_fd=dir_fd)
        else:
            os.unlink(name, dir_fd=dir_fd)


def _remove_authenticated_subdir(
    parent_fd: int, name: str, *, expect_dev: int, expect_ino: int
) -> None:
    """Delete owned child directory ``name`` bound to an expected device+inode.

    Opens ``name`` no-follow, requires its owner and ``(device, inode)`` to equal the
    authenticated identity, and empties it through the held descriptor — the recursive
    content deletion is therefore fully bound to the authenticated inode and can never
    touch another tree even under a concurrent same-UID rename. The now-empty directory
    is then removed by name after re-confirming the name still resolves to that inode.

    POSIX has no ``rmdir`` by descriptor, so the final ``rmdir`` inherently follows a
    name lookup: a same-UID swap in the sub-instruction window between the re-confirm
    and the ``rmdir`` can at worst make it remove a *different empty* directory (a swap
    to a non-empty directory fails ``ENOTEMPTY`` and is refused). No data is ever
    deleted from a wrong tree — that deletion already happened, inode-bound, above. A
    missing entry is a completed prior teardown, not an error.
    """

    try:
        child_fd = os.open(name, _DIR_OPEN_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        return
    try:
        metadata = os.fstat(child_fd)
        if (
            metadata.st_dev != expect_dev
            or metadata.st_ino != expect_ino
            or metadata.st_uid != os.geteuid()
        ):
            raise RunError("run directory diverged from its authenticated identity")
        _rmtree_fd(child_fd)
    finally:
        os.close(child_fd)
    final = os.lstat(name, dir_fd=parent_fd)
    if final.st_dev != expect_dev or final.st_ino != expect_ino:
        raise RunError("run directory was replaced before removal")
    os.rmdir(name, dir_fd=parent_fd)
    os.fsync(parent_fd)


def _teardown_active(roots: RunStorageRoots, run_id: str, index: RunIndexV1) -> None:
    try:
        active_dir_fd = os.open(roots.active, _DIR_OPEN_FLAGS)
    except OSError as exc:
        raise RunError("run active root is unavailable") from exc
    try:
        _remove_authenticated_subdir(
            active_dir_fd,
            run_id,
            expect_dev=index.active_device,
            expect_ino=index.active_inode,
        )
    finally:
        os.close(active_dir_fd)


def _inspection(roots: RunStorageRoots, index: RunIndexV1) -> RunInspection:
    return RunInspection(
        run_id=index.run_id,
        state=RunState.TERMINAL,
        outcome=index.outcome,
        record=_record_root(roots, index.run_id),
        record_sha256=index.record_sha256,
        checksums_sha256=index.checksums_sha256,
    )


def _finish_from_index(roots: RunStorageRoots, run_id: str) -> RunInspection:
    index = _read_model(_index_path(roots, run_id), RunIndexV1)
    if index.run_id != run_id:
        raise RunError("run index is bound to another run")
    record = _record_root(roots, run_id)
    verified = _verify_finalized_record(record, run_id)
    if (
        verified.verification.record_sha256 != index.record_sha256
        or verified.checksums_sha256 != index.checksums_sha256
        or verified.status.last_event_sha256 != index.terminal_event_sha256
        or verified.status.outcome is not index.outcome
        or verified.descriptor.stage_device != index.active_device
        or verified.descriptor.stage_inode != index.active_inode
    ):
        raise RunError("run record diverged from its index")
    # Both the record and the index were published (and accepted here) by a prior
    # finalization that may have crashed between a rename and its parent fsync. Make
    # both directory entries durable before deleting the only remaining active copy.
    fsync_directory(record.parent)
    fsync_directory(_index_path(roots, run_id).parent)
    _teardown_active(roots, run_id, index)
    return _inspection(roots, index)


def _complete_finalization(roots: RunStorageRoots, run_id: str) -> RunInspection:
    """Crash-forward finalize a TERMINAL run: checksums, record, index, teardown."""

    index_path = _index_path(roots, run_id)
    if os.path.lexists(index_path):
        return _finish_from_index(roots, run_id)
    active = _active_root(roots, run_id)
    _reconcile_control_temps(active)
    _recover_portable(active)
    _ensure_checksums(active)
    record = _record_root(roots, run_id)
    verification: RecordVerification | None = None
    if not os.path.lexists(record):
        try:
            verification = publish_record(active, record)
        except RecordError as exc:
            raise RunError(f"run record publication failed: {exc}") from exc
    else:
        # Accept a record published by a prior crashed finalization, but make its
        # rename durable before deleting the active evidence: a crash between that
        # prior rename and its parent fsync would otherwise leave the record's
        # directory entry non-durable while we tear down the only durable copy.
        fsync_directory(record.parent)
    verified = _verify_finalized_record(record, run_id, verification=verification)
    index = _build_index(verified)
    _publish_index(index_path, index)
    _teardown_active(roots, run_id, index)
    return _inspection(roots, index)


# ------------------------------------------------------------------------- recovery


def _reconcile_allocation_staging(roots: RunStorageRoots) -> None:
    """Reconcile ``allocation-staging/`` under a held no-follow directory descriptor.

    Each stage is reclaimed only while holding its own per-run lock: a live allocator
    holds that lock for the whole of its staging (including the brief pre-descriptor
    empty-directory window), so a contended lock means the stage belongs to a live
    allocator and is left untouched rather than deleted out from under it.
    """

    try:
        staging_fd = os.open(roots.allocation_staging, _DIR_OPEN_FLAGS)
    except OSError as exc:
        raise RunError("run allocation staging is unavailable") from exc
    try:
        for name in os.listdir(staging_fd):
            match = _STAGE_TEMP_RE.match(name)
            if match is None or _RUN_ID_RE.fullmatch(match.group("name")) is None:
                raise RunError(f"unexpected run allocation-staging entry: {name}")
            run_id = match.group("name")
            with exclusive_lock(_lock_path(roots, run_id)) as held:
                if held.status is LockStatus.CONTENDED:
                    continue  # a live allocator holds the lock; leave its stage
                if not held.acquired:
                    raise RunError(held.reason or f"run lock is unavailable: {run_id}")
                _reconcile_one_stage(staging_fd, name, run_id)
    finally:
        os.close(staging_fd)


def _reconcile_one_stage(staging_fd: int, name: str, run_id: str) -> None:
    metadata = os.lstat(name, dir_fd=staging_fd)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise RunError(f"unsafe run allocation-staging entry: {name}")
    try:
        stage_fd = os.open(name, _DIR_OPEN_FLAGS, dir_fd=staging_fd)
    except OSError as exc:
        raise RunError(f"unsafe run allocation-staging entry: {name}") from exc
    reclaim = False
    try:
        stage_meta = os.fstat(stage_fd)
        if stage_meta.st_dev != metadata.st_dev or stage_meta.st_ino != metadata.st_ino:
            raise RunError(f"unsafe run allocation-staging entry: {name}")
        members = os.listdir(stage_fd)
        if not members:
            # An empty stage is the mkdir-before-first-write crash; the per-run lock we
            # hold proves no live allocator owns it, so it may be reclaimed.
            reclaim = True
        else:
            if _DESCRIPTOR_NAME not in members:
                raise RunError(f"descriptorless run allocation-staging entry: {name}")
            descriptor = _parse_model(
                _read_owned_regular_at(stage_fd, _DESCRIPTOR_NAME), RunDescriptorV1
            )
            if (
                descriptor.run_id != run_id
                or descriptor.stage_device != stage_meta.st_dev
                or descriptor.stage_inode != stage_meta.st_ino
            ):
                raise RunError(f"run allocation-staging identity mismatch: {name}")
            # A descriptor owned by another uid is a foreign artifact, never our dead
            # crash: fail closed rather than reclaiming (and deleting) someone else's stage.
            if descriptor.owner.uid != os.geteuid():
                raise RunError(f"foreign-owned run allocation-staging entry: {name}")
            reclaim = not _owner_alive(descriptor.owner)
    finally:
        os.close(stage_fd)
    if reclaim:
        _remove_authenticated_subdir(
            staging_fd, name, expect_dev=metadata.st_dev, expect_ino=metadata.st_ino
        )


def _authenticate_active_root(active: Path, run_id: str) -> RunDescriptorV1:
    """Authenticate ``active/<run-id>`` at the recovery entry boundary.

    Opens the root no-follow (rejecting a symlinked, special, or foreign directory),
    then binds its held device+inode and its ``run.json`` descriptor to the requested
    ``run_id``: the descriptor must name this run, and the live root inode must equal
    the descriptor's recorded staged-root inode (preserved by the no-replace rename
    into ``active/``). Detected cross-run or inode divergence fails closed. The owning
    UID remains trusted and may not race later recovery operations; see the documented
    local-storage trust boundary.
    """

    root_fd = _open_owned_directory_fd(active)
    try:
        descriptor = _parse_model(
            _read_owned_regular_at(root_fd, _DESCRIPTOR_NAME), RunDescriptorV1
        )
        metadata = os.fstat(root_fd)
        if (
            descriptor.run_id != run_id
            or metadata.st_dev != descriptor.stage_device
            or metadata.st_ino != descriptor.stage_inode
        ):
            raise RunError(f"active run root diverged from its descriptor: {run_id}")
        return descriptor
    finally:
        os.close(root_fd)


def _recover_locked(roots: RunStorageRoots, run_id: str) -> RunInspection | RunBusy:
    index_path = _index_path(roots, run_id)
    record = _record_root(roots, run_id)
    active = _active_root(roots, run_id)
    if os.path.lexists(index_path):
        return _finish_from_index(roots, run_id)
    if not os.path.lexists(active):
        if os.path.lexists(record):
            raise RunError("run record exists without an authenticated index")
        raise RunError(f"unknown run: {run_id}")
    descriptor = _authenticate_active_root(active, run_id)
    status = _required_status(active, run_id)
    if status.state is RunState.TERMINAL:
        return _complete_finalization(roots, run_id)
    if _owner_alive(descriptor.owner):
        return RunBusy(run_id)
    _transition(
        active,
        status,
        RunState.TERMINAL,
        run_id=run_id,
        clock=lambda: datetime.now(UTC),
        outcome=RunOutcome.INTERRUPTED,
        reason="run-interrupted-owner-dead",
    )
    return _complete_finalization(roots, run_id)


def recover_run(run_id: str, *, home: Path) -> RunInspection | RunBusy:
    """Perform one exact recovery for ``run_id`` under its run lock."""

    _validate_run_id(run_id)
    roots = _layout(home, create=False)
    with exclusive_lock(_lock_path(roots, run_id)) as held:
        if held.status is LockStatus.CONTENDED:
            # Another live process owns the run: report busy, never force it.
            return RunBusy(run_id)
        if not held.acquired:
            raise RunError(held.reason or "run lock is unavailable")
        return _recover_locked(roots, run_id)


def recover_runs(*, home: Path) -> tuple[RunInspection, ...]:
    """Recover every non-live run under ``active/`` and reconcile allocation staging."""

    roots = _layout(home, create=True)
    _reconcile_allocation_staging(roots)
    recovered: list[RunInspection] = []
    for name in sorted(os.listdir(roots.active)):
        if _RUN_ID_RE.fullmatch(name) is None:
            raise RunError(f"unexpected active run entry: {name}")
        with exclusive_lock(_lock_path(roots, name)) as held:
            if held.status is LockStatus.CONTENDED:
                continue  # a live owner holds the run; skip it this pass
            if not held.acquired:
                raise RunError(held.reason or f"run lock is unavailable: {name}")
            outcome = _recover_locked(roots, name)
        if isinstance(outcome, RunInspection):
            recovered.append(outcome)
    return tuple(recovered)


def inspect_run(run_id: str, *, home: Path) -> RunInspection:
    """Recover, then require and verify the immutable record for one run."""

    outcome = recover_run(run_id, home=home)
    if isinstance(outcome, RunBusy):
        raise RunError(f"run is busy: {run_id}")
    return outcome

"""Crash-recoverable Git source preparation and evidence publication."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import posixpath
import re
import secrets
import shutil
import stat
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from strixlab.git_boundary import GitBoundary, GitBoundaryError, SshTrust
from strixlab.locks import exclusive_lock
from strixlab.manifests import GitUrl, SourceLockV1
from strixlab.process import ProcessResult
from strixlab.secure_fs import fsync_directory, write_exclusive
from strixlab.serialization import canonical_json_bytes
from strixlab.source_identity import (
    PatchIdentity,
    SubmoduleIdentity,
    candidate_id,
    content_tree_id,
    locator_class,
    preparation_identity,
    request_digest,
)

MAX_PATCH_COUNT = 64
MAX_PATCH_AGGREGATE_BYTES = 64 * 1024 * 1024
MAX_SINGLE_PATCH_BYTES = 64 * 1024 * 1024
MAX_SUBMODULE_DEPTH = 16
MAX_SUBMODULE_COUNT = 256
MAX_SUBMODULE_CHECKOUT_BYTES = 8 * 1024 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PREPARATION_ID_RE = re.compile(r"^prep-[a-z][a-z0-9-]*-[0-9a-f]{24}$")
_GIT_URL_ADAPTER = TypeAdapter(GitUrl)
_GIT_OUTPUT_PREFIX_BYTES = 256 * 1024
_LEGACY_UNINITIALIZED_SUBMODULE_SHA256 = hashlib.sha256(b"uninitialized").hexdigest()
_ZERO_OID = "0" * 40


class SourceError(RuntimeError):
    """Base error for source lifecycle failures."""


class SourceBusyError(SourceError):
    """Another source operation owns the source lock."""


class SourceCommandError(SourceError):
    """A hermetic Git command failed."""


class SourcePolicyError(SourceError):
    """Source material violates a preparation policy."""


class SourceDivergedError(SourceError):
    """Cleanup cannot prove ownership or candidate equality."""


class SourceTransitionInterrupt(BaseException):
    """Test-only crash analogue raised by a transition fault hook."""


class RegistryState(StrEnum):
    ALLOCATED = "allocated"
    MIRROR_READY = "mirror_ready"
    WORKTREE_CREATED = "worktree_created"
    CANDIDATE_READY = "candidate_ready"
    PUBLISHED = "published"
    FAILED = "failed"
    CLEANUP_STARTED = "cleanup_started"
    CLEANED = "cleaned"


class _StoredModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PatchEvidenceV1(_StoredModel):
    order: int = Field(ge=1)
    sha256: str = Field(pattern=_SHA256_RE.pattern)
    size_bytes: int = Field(ge=0)
    record_file: str


class _SubmoduleEvidence(_StoredModel):
    path: str
    commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    locator: str | None


class SubmoduleEvidenceV1(_SubmoduleEvidence):
    locator_sha256: str = Field(pattern=_SHA256_RE.pattern)


class SubmoduleEvidenceV2(_SubmoduleEvidence):
    locator_sha256: str | None = Field(pattern=_SHA256_RE.pattern)


class _SourceEvidence(_StoredModel):
    preparation_id: str = Field(pattern=_PREPARATION_ID_RE.pattern)
    request_digest: str = Field(pattern=_SHA256_RE.pattern)
    source_id: str
    source_locator: str | None
    source_locator_sha256: str = Field(pattern=_SHA256_RE.pattern)
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    branch_hint: str | None
    adapter: str
    submodules_enabled: bool
    patches: tuple[PatchEvidenceV1, ...]
    root_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    content_tree_id: str = Field(pattern=r"^content-tree-sha256:[0-9a-f]{64}$")
    candidate_id: str = Field(pattern=r"^candidate-sha256:[0-9a-f]{64}$")
    diff_file: str
    diff_sha256: str = Field(pattern=_SHA256_RE.pattern)
    diff_size_bytes: int = Field(ge=0)
    status: tuple[str, ...]
    created_at: str


class SourceEvidenceV1(_SourceEvidence):
    schema_version: Literal[1] = 1
    submodules: tuple[SubmoduleEvidenceV1, ...]


class SourceEvidenceV2(_SourceEvidence):
    schema_version: Literal[2] = 2
    submodules: tuple[SubmoduleEvidenceV2, ...]


SourceEvidence = SourceEvidenceV1 | SourceEvidenceV2
_SOURCE_EVIDENCE_ADAPTER: TypeAdapter[SourceEvidence] = TypeAdapter(SourceEvidence)


class OwnershipV1(_StoredModel):
    worktree_device: int
    worktree_inode: int
    admin_path: str
    admin_device: int
    admin_inode: int
    registered_path: str
    detached_head: str
    lock_reason: str


class SourceRegistryV1(_StoredModel):
    schema_version: Literal[1] = 1
    preparation_id: str = Field(pattern=_PREPARATION_ID_RE.pattern)
    source_id: str
    source_url: str
    request_digest: str = Field(pattern=_SHA256_RE.pattern)
    nonce: str = Field(pattern=r"^[0-9a-f]{32}$")
    attempt_digest: str = Field(pattern=_SHA256_RE.pattern)
    state: RegistryState
    last_completed_state: RegistryState | None
    failure_code: str | None
    mirror_path: str
    mirror_temporary_path: str
    worktree_path: str
    stage_path: str
    record_path: str
    registry_path: str
    scratch_path: str
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    ownership: OwnershipV1 | None = None
    evidence_sha256: str | None = Field(default=None, pattern=_SHA256_RE.pattern)
    created_at: str
    updated_at: str


class RequestRecordV1(_StoredModel):
    schema_version: Literal[1] = 1
    source_lock: dict[str, Any]
    resolved_locator: str
    patches: tuple[PatchEvidenceV1, ...]
    request_digest: str = Field(pattern=_SHA256_RE.pattern)


@dataclass(frozen=True, slots=True)
class SourcePreparation:
    evidence: SourceEvidence
    worktree: Path
    record: Path


@dataclass(frozen=True, slots=True)
class SourceInspection:
    registry: SourceRegistryV1
    evidence: SourceEvidence | None
    worktree_exists: bool
    record_exists: bool


@dataclass(frozen=True, slots=True)
class SourceCleanup:
    preparation_id: str
    state: RegistryState
    record: Path


@dataclass(frozen=True, slots=True)
class SourceLease:
    preparation_id: str
    source_id: str
    worktree: Path
    record: Path
    evidence: SourceEvidence
    verify_callback: Callable[[], None]

    def verify(self) -> None:
        """Revalidate the leased worktree against its published evidence."""

        self.verify_callback()


@dataclass(frozen=True, slots=True)
class _Layout:
    home: Path
    root: Path
    mirrors: Path
    worktrees: Path
    records: Path
    registry: Path
    locks: Path
    git_home: Path


@dataclass(frozen=True, slots=True)
class _PatchInput:
    identity: PatchIdentity
    content: bytes


@dataclass(slots=True)
class _SubmoduleBudget:
    count: int = 0
    bytes: int = 0


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _refuse_root() -> None:
    if os.geteuid() == 0:
        raise SourcePolicyError("source preparation refuses to run as root")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _portable_locator(locator: str) -> str | None:
    return None if locator_class(locator) in {"local-path", "file"} else locator


def _fsync_tree(root: Path) -> None:
    directories: list[Path] = []
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        directories.append(current)
        for name in files:
            path = current / name
            if path.is_symlink():
                continue
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        names[:] = [name for name in names if not (current / name).is_symlink()]
    for fsync_path in reversed(directories):
        fsync_directory(fsync_path)


def _write_atomic(path: Path, content: bytes) -> None:
    temporary = path.parent / f".{path.name}.tmp-{secrets.token_hex(12)}"
    write_exclusive(temporary, content)
    try:
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_model(value: BaseModel) -> bytes:
    return canonical_json_bytes(value.model_dump(mode="json"))


def _read_model_bytes(path: Path, model: type[BaseModel]) -> tuple[BaseModel, bytes]:
    try:
        content = path.read_bytes()
        value = model.model_validate(json.loads(content))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise SourcePolicyError(f"invalid source record: {path.name}") from exc
    if content != _canonical_model(value):
        raise SourcePolicyError(f"noncanonical source record: {path.name}")
    return value, content


def _read_model(path: Path, model: type[BaseModel]) -> BaseModel:
    value, _content = _read_model_bytes(path, model)
    return value


def _read_evidence(path: Path) -> tuple[SourceEvidence, bytes]:
    try:
        content = path.read_bytes()
        value = _SOURCE_EVIDENCE_ADAPTER.validate_json(content)
    except (OSError, ValidationError) as exc:
        raise SourcePolicyError(f"invalid source record: {path.name}") from exc
    if content != _canonical_model(value):
        raise SourcePolicyError(f"noncanonical source record: {path.name}")
    return value, content


def _ensure_directory(path: Path) -> None:
    if path.exists():
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise SourcePolicyError(f"source state path is not a directory: {path}")
        if metadata.st_uid != os.geteuid():
            raise SourcePolicyError(f"source state path is owned by another user: {path}")
        return
    path.mkdir(mode=0o700)


def _require_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise SourcePolicyError("StrixLab source state does not exist") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SourcePolicyError(f"source state path is not a directory: {path}")
    if metadata.st_uid != os.geteuid():
        raise SourcePolicyError(f"source state path is owned by another user: {path}")


def _layout(home: Path, *, create: bool) -> _Layout:
    if not home.is_absolute():
        raise ValueError("StrixLab home must be absolute")
    if home.exists() and home.is_symlink():
        raise SourcePolicyError("StrixLab home cannot be a symbolic link")
    if create and not home.exists():
        home.mkdir(mode=0o700, parents=True)
    root = home / "sources"
    layout = _Layout(
        home,
        root,
        root / "mirrors",
        root / "worktrees",
        root / "records",
        root / "registry",
        home / "locks",
        root / "git-home",
    )
    paths = (
        layout.home,
        layout.root,
        layout.mirrors,
        layout.worktrees,
        layout.records,
        layout.registry,
        layout.locks,
        layout.git_home,
    )
    if create:
        for path in paths:
            _ensure_directory(path)
    else:
        for path in paths:
            _require_directory(path)
    return layout


def _resolve_locator(locator: str) -> str:
    if locator.startswith("/"):
        return str(Path(locator).resolve(strict=True))
    return locator


def _read_patch(path: Path, order: int) -> _PatchInput:
    flags = os.O_CLOEXEC | os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if path.is_symlink():
            raise SourcePolicyError(f"patch is not a regular file: {path}") from exc
        raise SourcePolicyError(f"patch is unavailable: {path.name}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise SourcePolicyError(f"patch is not a regular file: {path}")
        if metadata.st_size > MAX_SINGLE_PATCH_BYTES:
            raise SourcePolicyError(f"patch exceeds {MAX_SINGLE_PATCH_BYTES} bytes: {path.name}")
        chunks = []
        while chunk := os.read(descriptor, 64 * 1024):
            chunks.append(chunk)
        content = b"".join(chunks)
        final_metadata = os.fstat(descriptor)
        if len(content) != metadata.st_size or final_metadata.st_size != metadata.st_size:
            raise SourcePolicyError(f"patch changed while being read: {path.name}")
    finally:
        os.close(descriptor)
    return _PatchInput(PatchIdentity(order, len(content), _sha256(content)), content)


def _patch_inputs(paths: Sequence[Path]) -> tuple[_PatchInput, ...]:
    if len(paths) > MAX_PATCH_COUNT:
        raise SourcePolicyError(f"patch count exceeds {MAX_PATCH_COUNT}")
    patches = tuple(_read_patch(path, index) for index, path in enumerate(paths, start=1))
    if sum(patch.identity.size_bytes for patch in patches) > MAX_PATCH_AGGREGATE_BYTES:
        raise SourcePolicyError(f"aggregate patch bytes exceed {MAX_PATCH_AGGREGATE_BYTES}")
    return patches


def _registry_directory(layout: _Layout, preparation_id: str) -> Path:
    if not _PREPARATION_ID_RE.fullmatch(preparation_id):
        raise SourcePolicyError("invalid preparation ID")
    return layout.registry / preparation_id


def _current_path(registry_directory: Path) -> Path:
    return registry_directory / "current.json"


def _random_nonce() -> str:
    return secrets.token_hex(16)


def _transition(
    registry: SourceRegistryV1,
    state: RegistryState,
    *,
    hook: Callable[[RegistryState], None] | None = None,
    details: Mapping[str, Any] | None = None,
    **changes: Any,
) -> SourceRegistryV1:
    directory = Path(registry.registry_path)
    events = directory / "events"
    sequence = len(tuple(events.glob("*.json"))) + 1
    previous = registry.state
    updated = SourceRegistryV1.model_validate(
        registry.model_copy(
            update={
                **changes,
                "last_completed_state": previous,
                "state": state,
                "updated_at": _utc_now(),
            }
        ).model_dump(mode="json")
    )
    event = {
        "details": dict(details or {}),
        "from": previous,
        "preparation_id": registry.preparation_id,
        "schema_version": 1,
        "sequence": sequence,
        "timestamp": updated.updated_at,
        "to": state,
    }
    write_exclusive(events / f"{sequence:04d}.json", canonical_json_bytes(event))
    fsync_directory(events)
    _write_atomic(_current_path(directory), _canonical_model(updated))
    if hook is not None:
        hook(state)
    return updated


def _allocate_registry(
    layout: _Layout,
    lock: SourceLockV1,
    resolved_locator: str,
    patches: tuple[_PatchInput, ...],
    nonce_factory: Callable[[], str] | None,
) -> SourceRegistryV1:
    patch_identities = tuple(patch.identity for patch in patches)
    digest = request_digest(lock, resolved_locator=resolved_locator, patches=patch_identities)
    make_nonce = nonce_factory or _random_nonce
    for _ in range(32):
        raw_nonce = make_nonce()
        if not re.fullmatch(r"[0-9a-f]{32}", raw_nonce):
            raise ValueError("nonce factory must return 32 lowercase hexadecimal characters")
        preparation_id, attempt = preparation_identity(lock.id, digest, bytes.fromhex(raw_nonce))
        registry_directory = layout.registry / preparation_id
        mirror_temporary = layout.mirrors / f".{lock.id}.{preparation_id}.tmp"
        paths = (
            registry_directory,
            layout.worktrees / preparation_id,
            layout.records / preparation_id,
            layout.records / f".{preparation_id}.tmp",
            mirror_temporary,
        )
        if any(path.exists() or path.is_symlink() for path in paths):
            if nonce_factory is not None:
                raise SourcePolicyError("preparation identifier collision")
            continue
        now = _utc_now()
        registry = SourceRegistryV1(
            preparation_id=preparation_id,
            source_id=lock.id,
            source_url=resolved_locator,
            request_digest=digest,
            nonce=raw_nonce,
            attempt_digest=attempt,
            state=RegistryState.ALLOCATED,
            last_completed_state=None,
            failure_code=None,
            mirror_path=str(layout.mirrors / f"{lock.id}.git"),
            mirror_temporary_path=str(mirror_temporary),
            worktree_path=str(layout.worktrees / preparation_id),
            stage_path=str(layout.records / f".{preparation_id}.tmp"),
            record_path=str(layout.records / preparation_id),
            registry_path=str(registry_directory),
            scratch_path=str(layout.records / f".{preparation_id}.tmp" / "scratch"),
            base_commit=lock.commit,
            created_at=now,
            updated_at=now,
        )
        patch_records = tuple(
            PatchEvidenceV1(
                order=patch.identity.order,
                sha256=patch.identity.sha256,
                size_bytes=patch.identity.size_bytes,
                record_file=f"patch-{patch.identity.order:03d}.patch",
            )
            for patch in patches
        )
        request = RequestRecordV1(
            source_lock=lock.model_dump(mode="json"),
            resolved_locator=resolved_locator,
            patches=patch_records,
            request_digest=digest,
        )
        registry_directory.mkdir(mode=0o700)
        (registry_directory / "events").mkdir(mode=0o700)
        event: dict[str, Any] = {
            "details": {},
            "from": None,
            "preparation_id": preparation_id,
            "schema_version": 1,
            "sequence": 1,
            "timestamp": now,
            "to": RegistryState.ALLOCATED,
        }
        write_exclusive(registry_directory / "events" / "0001.json", canonical_json_bytes(event))
        write_exclusive(registry_directory / "request.json", _canonical_model(request))
        fsync_directory(registry_directory / "events")
        write_exclusive(_current_path(registry_directory), _canonical_model(registry))
        fsync_directory(registry_directory)
        return registry
    raise SourcePolicyError("unable to allocate a unique preparation ID")


def _git_result(
    git: GitBoundary,
    arguments: Sequence[str],
    *,
    cwd: Path,
    stdout_spool: Path | None = None,
    output_limit_bytes: int | None = _GIT_OUTPUT_PREFIX_BYTES,
    allowed_returncodes: frozenset[int] = frozenset({0}),
) -> ProcessResult:
    try:
        return git.run(
            arguments,
            cwd=cwd,
            stdout_spool=stdout_spool,
            output_limit_bytes=output_limit_bytes,
            allowed_returncodes=allowed_returncodes,
        )
    except GitBoundaryError as exc:
        raise SourceCommandError(str(exc)) from exc


def _git_network_result(
    git: GitBoundary,
    arguments: Sequence[str],
    *,
    cwd: Path,
    locator: str,
    stdout_spool: Path | None = None,
    output_limit_bytes: int | None = _GIT_OUTPUT_PREFIX_BYTES,
    allowed_returncodes: frozenset[int] = frozenset({0}),
) -> ProcessResult:
    try:
        return git.run_network(
            arguments,
            cwd=cwd,
            locator=locator,
            stdout_spool=stdout_spool,
            output_limit_bytes=output_limit_bytes,
            allowed_returncodes=allowed_returncodes,
        )
    except GitBoundaryError as exc:
        raise SourceCommandError(str(exc)) from exc


def _git_run(
    git: GitBoundary,
    arguments: Sequence[str],
    *,
    cwd: Path,
    allowed_returncodes: frozenset[int] = frozenset({0}),
) -> str:
    return _git_result(
        git,
        arguments,
        cwd=cwd,
        allowed_returncodes=allowed_returncodes,
    ).stdout


def _git_network_run(
    git: GitBoundary,
    arguments: Sequence[str],
    *,
    cwd: Path,
    locator: str,
    allowed_returncodes: frozenset[int] = frozenset({0}),
) -> str:
    return _git_network_result(
        git,
        arguments,
        cwd=cwd,
        locator=locator,
        allowed_returncodes=allowed_returncodes,
    ).stdout


def _validate_mirror(git: GitBoundary, mirror: Path, locator: str, cwd: Path) -> None:
    if mirror.is_symlink() or not mirror.is_dir():
        raise SourcePolicyError("source mirror path is unsafe")
    bare = _git_run(git, ["--git-dir", str(mirror), "rev-parse", "--is-bare-repository"], cwd=cwd)
    if bare.strip() != "true":
        raise SourcePolicyError("source mirror is not bare")
    object_format = _git_run(
        git, ["--git-dir", str(mirror), "rev-parse", "--show-object-format"], cwd=cwd
    )
    if object_format.strip() != "sha1":
        raise SourcePolicyError("source-lock v1 requires SHA-1 Git objects")
    try:
        git.validate_mirror_config(mirror, locator)
    except GitBoundaryError as exc:
        raise SourcePolicyError(str(exc)) from exc


def _promote_verified_ref(git: GitBoundary, mirror: Path, commit: str, cwd: Path) -> None:
    verified = f"refs/strixlab/verified/{commit}"
    result = _git_result(
        git,
        ["--git-dir", str(mirror), "show-ref", "--verify", "--hash", verified],
        cwd=cwd,
        allowed_returncodes=frozenset({0, 1, 128}),
    )
    existing = result.stdout.strip() if result.returncode == 0 else None
    if existing is None:
        _git_run(
            git,
            ["--git-dir", str(mirror), "update-ref", verified, commit, _ZERO_OID],
            cwd=cwd,
        )
    elif existing != commit:
        raise SourcePolicyError("verified source ref is corrupt")


def _delete_quarantine(git: GitBoundary, mirror: Path, preparation_id: str, cwd: Path) -> None:
    for leaf in ("raw", "branch"):
        with contextlib.suppress(SourceCommandError):
            _git_run(
                git,
                [
                    "--git-dir",
                    str(mirror),
                    "update-ref",
                    "-d",
                    f"refs/strixlab/quarantine/{preparation_id}/{leaf}",
                ],
                cwd=cwd,
            )


def _fetch_and_verify(
    git: GitBoundary,
    mirror: Path,
    lock: SourceLockV1,
    resolved_locator: str,
    preparation_id: str,
    cwd: Path,
) -> None:
    raw_ref = f"refs/strixlab/quarantine/{preparation_id}/raw"
    fetch_base = [
        "--git-dir",
        str(mirror),
        "fetch",
        "--no-tags",
        "--no-write-fetch-head",
        "--no-recurse-submodules",
    ]
    try:
        _git_network_result(
            git,
            [*fetch_base, "origin", f"{lock.commit}:{raw_ref}"],
            cwd=cwd,
            locator=resolved_locator,
        )
    except SourceCommandError as exact_error:
        if lock.branch_hint is None:
            raise
        _git_run(git, ["check-ref-format", "--branch", lock.branch_hint], cwd=cwd)
        branch_ref = f"refs/strixlab/quarantine/{preparation_id}/branch"
        try:
            _git_network_run(
                git,
                [
                    *fetch_base,
                    "origin",
                    f"+refs/heads/{lock.branch_hint}:{branch_ref}",
                ],
                cwd=cwd,
                locator=resolved_locator,
            )
        except SourceCommandError as fallback_error:
            raise SourceCommandError(
                f"{exact_error}; branch fallback failed: {fallback_error}"
            ) from fallback_error
    object_type = _git_result(
        git,
        ["--git-dir", str(mirror), "cat-file", "-t", lock.commit],
        cwd=cwd,
        allowed_returncodes=frozenset({0, 1, 128}),
    )
    if object_type.returncode != 0 or object_type.stdout.strip() != "commit":
        raise SourcePolicyError("the pinned Git object is not an exact commit")
    resolved = _git_run(
        git, ["--git-dir", str(mirror), "rev-parse", f"{lock.commit}^{{commit}}"], cwd=cwd
    ).strip()
    if resolved != lock.commit:
        raise SourcePolicyError("the fetched commit does not match the source lock")
    _promote_verified_ref(git, mirror, lock.commit, cwd)
    _delete_quarantine(git, mirror, preparation_id, cwd)


def _prepare_mirror(
    git: GitBoundary,
    registry: SourceRegistryV1,
    lock: SourceLockV1,
    layout: _Layout,
) -> Path:
    mirror = Path(registry.mirror_path)
    temporary = Path(registry.mirror_temporary_path)
    if mirror.exists():
        _validate_mirror(git, mirror, registry.source_url, layout.root)
        try:
            _fetch_and_verify(
                git, mirror, lock, registry.source_url, registry.preparation_id, layout.root
            )
        finally:
            _delete_quarantine(git, mirror, registry.preparation_id, layout.root)
        _validate_mirror(git, mirror, registry.source_url, layout.root)
        return mirror
    temporary.mkdir(mode=0o700)
    try:
        _git_run(
            git,
            ["init", "--bare", "--object-format=sha1", str(temporary)],
            cwd=layout.root,
        )
        _git_run(
            git,
            ["--git-dir", str(temporary), "config", "remote.origin.url", registry.source_url],
            cwd=layout.root,
        )
        _validate_mirror(git, temporary, registry.source_url, layout.root)
        _fetch_and_verify(
            git, temporary, lock, registry.source_url, registry.preparation_id, layout.root
        )
        _validate_mirror(git, temporary, registry.source_url, layout.root)
        _fsync_tree(temporary)
        temporary.rename(mirror)
        fsync_directory(layout.mirrors)
        return mirror
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _safe_repo_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or ".git" in path.parts
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise SourcePolicyError(f"unsafe repository-relative path: {value!r}")
    return path


def _reject_symlinked_ancestors(repository: Path, path: PurePosixPath) -> None:
    ancestor = repository
    for part in path.parts[:-1]:
        ancestor /= part
        if ancestor.is_symlink():
            raise SourcePolicyError("repository path has a symlinked ancestor")


def _gitlinks(repository: Path, git: GitBoundary) -> dict[str, str]:
    content = git.bytes(["-C", str(repository), "ls-tree", "-r", "-z", "HEAD"], cwd=repository)
    links: dict[str, str] = {}
    for entry in content.rstrip(b"\0").split(b"\0") if content else ():
        try:
            header, raw_path = entry.split(b"\t", 1)
            mode, object_type, object_id = header.decode("ascii").split()
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise SourcePolicyError("unsupported Git tree entry encoding") from exc
        if mode == "160000":
            if object_type != "commit":
                raise SourcePolicyError("invalid gitlink object type")
            _safe_repo_path(path)
            links[path] = object_id
    return links


def _resolve_relative_locator(parent: str, child: str) -> str:
    if not child.startswith(("./", "../")):
        return _GIT_URL_ADAPTER.validate_python(child, strict=True)
    kind = locator_class(parent)
    if kind == "local-path":
        resolved = str((Path(parent) / child).resolve(strict=False))
    elif kind in {"file", "https", "ssh"}:
        resolved = urljoin(parent.rstrip("/") + "/", child)
    else:
        host, parent_path = parent.split(":", 1)
        resolved = f"{host}:{posixpath.normpath(parent_path + '/' + child)}"
    return _GIT_URL_ADAPTER.validate_python(resolved, strict=True)


def _submodule_config(
    repository: Path,
    gitlinks: Mapping[str, str],
    parent_locator: str,
    git: GitBoundary,
) -> dict[str, tuple[str, str]]:
    metadata = repository / ".gitmodules"
    if not gitlinks:
        return {}
    if metadata.is_symlink() or not metadata.is_file():
        raise SourcePolicyError("gitlinks require a regular .gitmodules file")
    content = git.bytes(
        ["-C", str(repository), "config", "--file", str(metadata), "--null", "--list"],
        cwd=repository,
    )
    entries: dict[str, dict[str, str]] = {}
    for raw in content.rstrip(b"\0").split(b"\0") if content else ():
        try:
            key_raw, value_raw = raw.split(b"\n", 1)
            key = key_raw.decode("utf-8")
            value = value_raw.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise SourcePolicyError("invalid .gitmodules encoding") from exc
        match = re.fullmatch(r"submodule\.(.+)\.(path|url|branch|update)", key)
        if match is None:
            raise SourcePolicyError(f"unsupported .gitmodules key: {key}")
        name, field = match.groups()
        if not name or any(ord(character) < 32 or ord(character) == 127 for character in name):
            raise SourcePolicyError("unsafe submodule name")
        if any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise SourcePolicyError("unsafe submodule metadata value")
        fields = entries.setdefault(name, {})
        if field in fields:
            raise SourcePolicyError("duplicate .gitmodules key")
        fields[field] = value
    by_path: dict[str, tuple[str, str]] = {}
    for name, fields in entries.items():
        if "path" not in fields or "url" not in fields:
            raise SourcePolicyError("submodule entry lacks path or URL")
        path = fields["path"]
        safe_path = _safe_repo_path(path)
        _reject_symlinked_ancestors(repository, safe_path)
        update = fields.get("update")
        if update not in {None, "checkout"}:
            raise SourcePolicyError("unsupported submodule update policy")
        if path in by_path:
            raise SourcePolicyError("duplicate submodule path")
        by_path[path] = (name, _resolve_relative_locator(parent_locator, fields["url"]))
    if set(by_path) != set(gitlinks):
        raise SourcePolicyError(".gitmodules paths do not match immediate gitlinks")
    return by_path


def _detached_head(repository: Path, git: GitBoundary) -> str:
    head = _git_run(git, ["-C", str(repository), "rev-parse", "HEAD"], cwd=repository).strip()
    symbolic = _git_result(
        git,
        ["-C", str(repository), "symbolic-ref", "-q", "HEAD"],
        cwd=repository,
        allowed_returncodes=frozenset({0, 1}),
    )
    if symbolic.returncode == 0:
        raise SourcePolicyError("source checkout is not detached")
    return head


def _tree_bytes(root: Path) -> int:
    total = 0
    stack = [root]
    while stack:
        directory = stack.pop()
        for entry in os.scandir(directory):
            if entry.name == ".git":
                continue
            metadata = entry.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                continue
            if stat.S_ISDIR(metadata.st_mode):
                stack.append(Path(entry.path))
            elif stat.S_ISREG(metadata.st_mode):
                total += metadata.st_size
    return total


def _clear_submodule_config(git: GitBoundary, repository: Path, name: str) -> None:
    for suffix in ("active", "url"):
        _git_result(
            git,
            ["-C", str(repository), "config", "--unset-all", f"submodule.{name}.{suffix}"],
            cwd=repository,
            allowed_returncodes=frozenset({0, 1, 5}),
        )


def _materialize_submodules(
    repository: Path,
    parent_locator: str,
    git: GitBoundary,
    budget: _SubmoduleBudget,
    *,
    prefix: PurePosixPath | None = None,
    depth: int = 0,
    ancestry: set[tuple[str, str]] | None = None,
) -> tuple[SubmoduleEvidenceV2, ...]:
    if depth >= MAX_SUBMODULE_DEPTH:
        raise SourcePolicyError("submodule depth limit exceeded")
    active = set() if ancestry is None else ancestry
    resolved_prefix = PurePosixPath() if prefix is None else prefix
    links = _gitlinks(repository, git)
    configurations = _submodule_config(repository, links, parent_locator, git)
    if budget.count + len(links) > MAX_SUBMODULE_COUNT:
        raise SourcePolicyError("submodule count limit exceeded")
    budget.count += len(links)
    evidence = []
    for path in sorted(links, key=lambda value: value.encode("utf-8")):
        commit = links[path]
        name, locator = configurations[path]
        identity = (_sha256(locator.encode()), commit)
        if identity in active:
            raise SourcePolicyError("recursive submodule identity detected")
        active.add(identity)
        _git_network_run(
            git,
            [
                "-c",
                f"submodule.{name}.url={locator}",
                "-C",
                str(repository),
                "submodule",
                "update",
                "--init",
                "--checkout",
                "--",
                path,
            ],
            cwd=repository,
            locator=locator,
        )
        _clear_submodule_config(git, repository, name)
        child = repository.joinpath(*PurePosixPath(path).parts)
        object_format = _git_run(
            git, ["-C", str(child), "rev-parse", "--show-object-format"], cwd=child
        ).strip()
        if object_format != "sha1" or _detached_head(child, git) != commit:
            raise SourcePolicyError("submodule checkout identity mismatch")
        if _status(child, git):
            raise SourcePolicyError("submodule checkout is dirty")
        budget.bytes += _tree_bytes(child)
        if budget.bytes > MAX_SUBMODULE_CHECKOUT_BYTES:
            raise SourcePolicyError("submodule checkout byte limit exceeded")
        full_path = resolved_prefix / path
        evidence.append(
            SubmoduleEvidenceV2(
                path=str(full_path),
                commit=commit,
                locator=_portable_locator(locator),
                locator_sha256=_sha256(locator.encode()),
            )
        )
        evidence.extend(
            _materialize_submodules(
                child,
                locator,
                git,
                budget,
                prefix=full_path,
                depth=depth + 1,
                ancestry=active,
            )
        )
        active.remove(identity)
    return tuple(evidence)


def _raw_status(worktree: Path, git: GitBoundary) -> bytes:
    return git.bytes(
        [
            "-C",
            str(worktree),
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        cwd=worktree,
    )


def _decode_status(content: bytes) -> tuple[str, ...]:
    try:
        return tuple(entry.decode("utf-8") for entry in content.rstrip(b"\0").split(b"\0"))
    except UnicodeDecodeError as exc:
        raise SourcePolicyError("source status contains a non-UTF-8 path") from exc


def _status(worktree: Path, git: GitBoundary) -> tuple[str, ...]:
    content = _raw_status(worktree, git)
    return () if not content else _decode_status(content)


def _validate_candidate_paths(worktree: Path, base_commit: str, git: GitBoundary) -> None:
    content = git.bytes(
        [
            "-C",
            str(worktree),
            "diff",
            "--raw",
            "-z",
            "--full-index",
            "--cached",
            "--no-ext-diff",
            "--no-textconv",
            base_commit,
            "--",
        ],
        cwd=worktree,
    )
    values = content.split(b"\0")
    index = 0
    while index < len(values) and values[index]:
        try:
            header = values[index].decode("ascii")
            old_mode, new_mode, _old, _new, status_code = header[1:].split()
            path_count = 2 if status_code.startswith(("R", "C")) else 1
            paths = [values[index + offset + 1].decode("utf-8") for offset in range(path_count)]
        except (IndexError, UnicodeDecodeError, ValueError) as exc:
            raise SourcePolicyError("invalid staged raw diff") from exc
        if old_mode == "160000" or new_mode == "160000":
            raise SourcePolicyError("patches may not change gitlinks")
        for value in paths:
            path = _safe_repo_path(value)
            if path == PurePosixPath(".gitmodules"):
                raise SourcePolicyError("patches may not change .gitmodules")
            ancestor = worktree
            for part in path.parts[:-1]:
                ancestor /= part
                if ancestor.is_symlink():
                    raise SourcePolicyError("candidate path has a symlinked ancestor")
        index += path_count + 1


def _validate_index_state(worktree: Path, git: GitBoundary) -> tuple[str, ...]:
    status = _status(worktree, git)
    for line in status:
        if len(line) < 3 or line.startswith("??") or line[1] != " ":
            raise SourcePolicyError("source contains unstaged or untracked changes")
    return status


def _capture_diff(
    worktree: Path, base_commit: str, destination: Path, git: GitBoundary
) -> tuple[str, int]:
    temporary = git.scratch / "candidate.diff"
    result = _git_result(
        git,
        [
            "-C",
            str(worktree),
            "diff",
            "--binary",
            "--full-index",
            "--cached",
            "--no-ext-diff",
            "--no-textconv",
            base_commit,
            "--",
        ],
        cwd=worktree,
        stdout_spool=temporary,
        output_limit_bytes=0,
    )
    os.replace(temporary, destination)
    fsync_directory(destination.parent)
    return result.stdout_sha256, result.stdout_bytes


def _capture_ownership(
    worktree: Path,
    mirror: Path,
    base_commit: str,
    lock_reason: str,
    git: GitBoundary,
) -> OwnershipV1:
    if _detached_head(worktree, git) != base_commit:
        raise SourcePolicyError("worktree HEAD does not match the pinned commit")
    admin_text = _git_run(git, ["-C", str(worktree), "rev-parse", "--git-dir"], cwd=worktree)
    admin = Path(admin_text.strip()).resolve(strict=True)
    expected_parent = (mirror / "worktrees").resolve(strict=True)
    if admin.parent != expected_parent:
        raise SourcePolicyError("worktree administrative entry is outside the mirror")
    worktree_metadata = worktree.lstat()
    admin_metadata = admin.lstat()
    return OwnershipV1(
        worktree_device=worktree_metadata.st_dev,
        worktree_inode=worktree_metadata.st_ino,
        admin_path=str(admin),
        admin_device=admin_metadata.st_dev,
        admin_inode=admin_metadata.st_ino,
        registered_path=str(worktree),
        detached_head=base_commit,
        lock_reason=lock_reason,
    )


def _lock_reason(registry: SourceRegistryV1) -> str:
    return f"strixlab-source-v1:{registry.preparation_id}:{registry.nonce}"


def _publish_failure(
    registry: SourceRegistryV1,
    error: Exception,
    hook: Callable[[RegistryState], None] | None,
) -> SourceRegistryV1:
    return _transition(
        registry,
        RegistryState.FAILED,
        hook=hook,
        failure_code=type(error).__name__,
    )


def prepare_source(
    lock: SourceLockV1,
    *,
    home: Path,
    patches: Sequence[Path] = (),
    ssh_trust: SshTrust | None = None,
    nonce_factory: Callable[[], str] | None = None,
    transition_hook: Callable[[RegistryState], None] | None = None,
) -> SourcePreparation:
    """Prepare one detached source candidate and publish immutable evidence."""

    _refuse_root()
    layout = _layout(home, create=True)
    resolved_locator = _resolve_locator(lock.url)
    patch_inputs = _patch_inputs(patches)
    lock_path = layout.locks / f"source-{lock.id}.lock"
    with exclusive_lock(lock_path) as attempt:
        if not attempt.acquired:
            raise SourceBusyError(attempt.reason or "source lock is unavailable")
        registry = _allocate_registry(layout, lock, resolved_locator, patch_inputs, nonce_factory)
        registry_directory = Path(registry.registry_path)
        stage = Path(registry.stage_path)
        stage.mkdir(mode=0o700)
        scratch = Path(registry.scratch_path)
        scratch.mkdir(mode=0o700)
        git: GitBoundary | None = None
        try:
            git = GitBoundary.create(
                git_home=layout.git_home,
                scratch=scratch,
                locator=resolved_locator,
                ssh_trust=ssh_trust,
            )
            patch_evidence = []
            patch_files = []
            for patch in patch_inputs:
                name = f"patch-{patch.identity.order:03d}.patch"
                target = stage / name
                write_exclusive(target, patch.content)
                patch_files.append(target)
                patch_evidence.append(
                    PatchEvidenceV1(
                        order=patch.identity.order,
                        sha256=patch.identity.sha256,
                        size_bytes=patch.identity.size_bytes,
                        record_file=name,
                    )
                )
                del patch
            patch_identities = tuple(value.identity for value in patch_inputs)
            patch_inputs = ()
            mirror = _prepare_mirror(git, registry, lock, layout)
            registry = _transition(registry, RegistryState.MIRROR_READY, hook=transition_hook)
            verified_ref = f"refs/strixlab/verified/{lock.commit}"
            lock_reason = _lock_reason(registry)
            worktree = Path(registry.worktree_path)
            _git_run(
                git,
                [
                    "--git-dir",
                    str(mirror),
                    "worktree",
                    "add",
                    "--lock",
                    "--reason",
                    lock_reason,
                    "--detach",
                    str(worktree),
                    verified_ref,
                ],
                cwd=layout.root,
            )
            ownership = _capture_ownership(worktree, mirror, lock.commit, lock_reason, git)
            write_exclusive(registry_directory / "owner.json", _canonical_model(ownership))
            registry = _transition(
                registry,
                RegistryState.WORKTREE_CREATED,
                hook=transition_hook,
                ownership=ownership,
            )
            if _status(worktree, git):
                raise SourcePolicyError("base source worktree is dirty")
            if lock.submodules:
                submodules = _materialize_submodules(
                    worktree,
                    resolved_locator,
                    git,
                    _SubmoduleBudget(),
                )
                git.validate_mirror_config(mirror, resolved_locator)
            else:
                root_links = _gitlinks(worktree, git)
                submodules = tuple(
                    SubmoduleEvidenceV2(
                        path=path,
                        commit=commit,
                        locator=None,
                        locator_sha256=None,
                    )
                    for path, commit in sorted(root_links.items())
                )
            for patch_file in patch_files:
                _git_run(
                    git,
                    [
                        "-C",
                        str(worktree),
                        "apply",
                        "--check",
                        "--index",
                        "--",
                        str(patch_file),
                    ],
                    cwd=worktree,
                )
                _git_run(
                    git,
                    ["-C", str(worktree), "apply", "--index", "--", str(patch_file)],
                    cwd=worktree,
                )
            _validate_candidate_paths(worktree, lock.commit, git)
            status = _validate_index_state(worktree, git)
            tree = _git_run(git, ["-C", str(worktree), "write-tree"], cwd=worktree).strip()
            if not re.fullmatch(r"[0-9a-f]{40}", tree):
                raise SourcePolicyError("invalid root tree identity")
            diff_file = "candidate.diff"
            diff_sha256, diff_size = _capture_diff(worktree, lock.commit, stage / diff_file, git)
            submodule_identities = tuple(
                SubmoduleIdentity(value.path, value.commit) for value in submodules
            )
            content_id = content_tree_id(
                tree, patches=patch_identities, submodules=submodule_identities
            )
            candidate = candidate_id(lock.commit, content_id, submodules=lock.submodules)
            evidence = SourceEvidenceV2(
                preparation_id=registry.preparation_id,
                request_digest=registry.request_digest,
                source_id=lock.id,
                source_locator=_portable_locator(resolved_locator),
                source_locator_sha256=_sha256(resolved_locator.encode()),
                base_commit=lock.commit,
                branch_hint=lock.branch_hint,
                adapter=lock.adapter,
                submodules_enabled=lock.submodules,
                patches=tuple(patch_evidence),
                submodules=submodules,
                root_tree=tree,
                content_tree_id=content_id,
                candidate_id=candidate,
                diff_file=diff_file,
                diff_sha256=diff_sha256,
                diff_size_bytes=diff_size,
                status=status,
                created_at=_utc_now(),
            )
            evidence_bytes = _canonical_model(evidence)
            write_exclusive(stage / "evidence.json", evidence_bytes)
            fsync_directory(stage)
            registry = _transition(registry, RegistryState.CANDIDATE_READY, hook=transition_hook)
            shutil.rmtree(scratch)
            fsync_directory(stage)
            record = Path(registry.record_path)
            stage.rename(record)
            fsync_directory(layout.records)
            registry = _transition(
                registry,
                RegistryState.PUBLISHED,
                hook=transition_hook,
                evidence_sha256=_sha256(evidence_bytes),
            )
            return SourcePreparation(evidence, worktree, record)
        except Exception as exc:
            if git is not None:
                _delete_quarantine(
                    git, Path(registry.mirror_path), registry.preparation_id, layout.root
                )
            _publish_failure(registry, exc, transition_hook)
            if isinstance(exc, GitBoundaryError):
                raise SourceCommandError(str(exc)) from exc
            raise


def _load_registry(layout: _Layout, preparation_id: str) -> SourceRegistryV1:
    directory = _registry_directory(layout, preparation_id)
    loaded = _read_model(_current_path(directory), SourceRegistryV1)
    assert isinstance(loaded, SourceRegistryV1)
    expected = {
        "mirror_path": str(layout.mirrors / f"{loaded.source_id}.git"),
        "mirror_temporary_path": str(layout.mirrors / f".{loaded.source_id}.{preparation_id}.tmp"),
        "worktree_path": str(layout.worktrees / preparation_id),
        "stage_path": str(layout.records / f".{preparation_id}.tmp"),
        "record_path": str(layout.records / preparation_id),
        "registry_path": str(directory),
        "scratch_path": str(layout.records / f".{preparation_id}.tmp" / "scratch"),
    }
    if loaded.preparation_id != preparation_id or any(
        getattr(loaded, name) != value for name, value in expected.items()
    ):
        raise SourcePolicyError("source registry paths do not match the owned layout")
    _verify_event_log(directory, loaded)
    return loaded


def _verify_event_log(directory: Path, registry: SourceRegistryV1) -> None:
    events = directory / "events"
    _require_directory(events)
    paths = sorted(events.iterdir())
    if not paths or [path.name for path in paths] != [
        f"{sequence:04d}.json" for sequence in range(1, len(paths) + 1)
    ]:
        raise SourcePolicyError("source transition log is not contiguous")
    previous: RegistryState | None = None
    for sequence, path in enumerate(paths, start=1):
        try:
            content = path.read_bytes()
            event = json.loads(content)
            state = RegistryState(event["to"])
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            raise SourcePolicyError("invalid source transition event") from exc
        if content != canonical_json_bytes(event):
            raise SourcePolicyError("noncanonical source transition event")
        if (
            event.get("schema_version") != 1
            or event.get("sequence") != sequence
            or event.get("preparation_id") != registry.preparation_id
            or event.get("from") != previous
            or not isinstance(event.get("details"), dict)
            or not isinstance(event.get("timestamp"), str)
        ):
            raise SourcePolicyError("source transition event chain is invalid")
        previous = state
    if previous is not registry.state:
        raise SourcePolicyError("source transition log disagrees with current state")


def _verify_evidence(registry: SourceRegistryV1) -> SourceEvidence | None:
    request_value = _read_model(Path(registry.registry_path) / "request.json", RequestRecordV1)
    assert isinstance(request_value, RequestRecordV1)
    lock = SourceLockV1.model_validate(request_value.source_lock)
    patch_ids = tuple(
        PatchIdentity(value.order, value.size_bytes, value.sha256)
        for value in request_value.patches
    )
    expected_request = request_digest(
        lock,
        resolved_locator=request_value.resolved_locator,
        patches=patch_ids,
    )
    if (
        expected_request != registry.request_digest
        or expected_request != request_value.request_digest
    ):
        raise SourcePolicyError("source request identity mismatch")
    evidence_path = Path(registry.record_path) / "evidence.json"
    if not evidence_path.exists():
        return None
    evidence_value, evidence_bytes = _read_evidence(evidence_path)
    if registry.evidence_sha256 is not None and _sha256(evidence_bytes) != registry.evidence_sha256:
        raise SourcePolicyError("portable evidence digest mismatch")
    legacy_inconsistent = isinstance(evidence_value, SourceEvidenceV1) and any(
        (submodule.locator_sha256 != _LEGACY_UNINITIALIZED_SUBMODULE_SHA256)
        != evidence_value.submodules_enabled
        for submodule in evidence_value.submodules
    )
    current_inconsistent = isinstance(evidence_value, SourceEvidenceV2) and any(
        (submodule.locator_sha256 is not None) != evidence_value.submodules_enabled
        for submodule in evidence_value.submodules
    )
    if legacy_inconsistent or current_inconsistent:
        raise SourcePolicyError("submodule initialization evidence is inconsistent")
    record = Path(registry.record_path)
    for patch in evidence_value.patches:
        content = (record / patch.record_file).read_bytes()
        if len(content) != patch.size_bytes or _sha256(content) != patch.sha256:
            raise SourcePolicyError("preserved patch integrity mismatch")
    diff = (record / evidence_value.diff_file).read_bytes()
    if len(diff) != evidence_value.diff_size_bytes or _sha256(diff) != evidence_value.diff_sha256:
        raise SourcePolicyError("preserved diff integrity mismatch")
    submodule_ids = tuple(
        SubmoduleIdentity(value.path, value.commit) for value in evidence_value.submodules
    )
    expected_content = content_tree_id(
        evidence_value.root_tree,
        patches=patch_ids,
        submodules=submodule_ids,
    )
    expected_candidate = candidate_id(
        evidence_value.base_commit,
        expected_content,
        submodules=evidence_value.submodules_enabled,
    )
    if (
        evidence_value.preparation_id != registry.preparation_id
        or evidence_value.request_digest != registry.request_digest
        or evidence_value.content_tree_id != expected_content
        or evidence_value.candidate_id != expected_candidate
    ):
        raise SourcePolicyError("portable source identity mismatch")
    return evidence_value


def inspect_source(preparation_id: str, *, home: Path) -> SourceInspection:
    """Independently verify registry, request, and portable evidence."""

    if not _PREPARATION_ID_RE.fullmatch(preparation_id):
        raise SourcePolicyError("invalid preparation ID")
    layout = _layout(home, create=False)
    registry = _load_registry(layout, preparation_id)
    evidence = _verify_evidence(registry)
    if registry.state in {RegistryState.PUBLISHED, RegistryState.CLEANED} and evidence is None:
        raise SourcePolicyError("published source preparation lacks evidence")
    return SourceInspection(
        registry,
        evidence,
        Path(registry.worktree_path).is_dir(),
        Path(registry.record_path).is_dir(),
    )


def _allocate_lease_scratch(layout: _Layout, preparation_id: str) -> Path:
    for _ in range(32):
        scratch = layout.records / f".lease-{preparation_id}-{secrets.token_hex(16)}"
        try:
            scratch.mkdir(mode=0o700)
        except FileExistsError:
            continue
        metadata = scratch.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise SourcePolicyError("source lease scratch ownership is unsafe")
        fsync_directory(layout.records)
        return scratch
    raise SourcePolicyError("unable to allocate unique source lease scratch")


@contextlib.contextmanager
def lease_source(preparation_id: str, *, home: Path) -> Iterator[SourceLease]:
    """Hold the source lock while yielding one authenticated published candidate."""

    layout = _layout(home, create=False)
    registry = _load_registry(layout, preparation_id)
    lock_path = layout.locks / f"source-{registry.source_id}.lock"
    with exclusive_lock(lock_path) as attempt:
        if not attempt.acquired:
            raise SourceBusyError(attempt.reason or "source lock is unavailable")
        registry = _load_registry(layout, preparation_id)
        if registry.state is not RegistryState.PUBLISHED:
            raise SourcePolicyError("source preparation is not published")
        evidence = _verify_evidence(registry)
        if evidence is None:
            raise SourcePolicyError("published source preparation lacks evidence")
        owner = registry.ownership
        if owner is None:
            raise SourceDivergedError("published source preparation lacks ownership")
        _validate_ownership_binding(registry, owner)
        worktree = Path(registry.worktree_path)
        if not _directory_exists_nofollow(worktree, "worktree directory"):
            raise SourceDivergedError("published source worktree is missing")
        _validate_inode(worktree, owner.worktree_device, owner.worktree_inode, "worktree directory")
        admin_exists, admin_locked = _admin_entry_state(owner, allow_missing_lock=False)
        if not admin_exists or not admin_locked:
            raise SourceDivergedError("published source worktree ownership is incomplete")
        scratch = _allocate_lease_scratch(layout, preparation_id)
        try:
            git = GitBoundary.create(
                git_home=layout.git_home,
                scratch=scratch,
                locator=registry.mirror_path,
            )

            def verify_candidate() -> None:
                _verify_candidate_for_cleanup(registry, evidence, git, require_match=True)

            verify_candidate()
            yield SourceLease(
                preparation_id,
                registry.source_id,
                worktree,
                Path(registry.record_path),
                evidence,
                verify_candidate,
            )
        finally:
            shutil.rmtree(scratch)
            fsync_directory(layout.records)


def _validate_inode(path: Path, device: int, inode: int, description: str) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or metadata.st_dev != device or metadata.st_ino != inode:
        raise SourceDivergedError(f"{description} ownership identity changed")


def _directory_exists_nofollow(path: Path, description: str) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise SourceDivergedError(f"{description} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SourceDivergedError(f"{description} is unsafe")
    return True


def _validated_submodule_path(worktree: Path, path: str) -> tuple[Path, bool]:
    child = worktree
    child_exists = True
    for part in _safe_repo_path(path).parts:
        child /= part
        try:
            metadata = child.lstat()
        except FileNotFoundError:
            child_exists = False
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise SourceDivergedError("submodule path was replaced by a symbolic link")
        if not stat.S_ISDIR(metadata.st_mode):
            child_exists = False
            break
    return child, child_exists


def _verify_candidate_for_cleanup(
    registry: SourceRegistryV1,
    evidence: SourceEvidence,
    git: GitBoundary,
    *,
    require_match: bool,
) -> dict[str, Any]:
    worktree = Path(registry.worktree_path)
    for submodule in evidence.submodules:
        _validated_submodule_path(worktree, submodule.path)
    head = _detached_head(worktree, git)
    tree = _git_run(git, ["-C", str(worktree), "write-tree"], cwd=worktree).strip()
    status_raw = _raw_status(worktree, git)
    status = () if not status_raw else _decode_status(status_raw)
    temporary = git.scratch / "cleanup.diff"
    digest, size = _capture_diff(worktree, evidence.base_commit, temporary, git)
    temporary.unlink(missing_ok=True)
    matches = not (
        head != evidence.base_commit
        or tree != evidence.root_tree
        or status != evidence.status
        or digest != evidence.diff_sha256
        or size != evidence.diff_size_bytes
    )
    if require_match and not matches:
        raise SourceDivergedError("worktree candidate state diverged from evidence")
    submodule_observations = []
    for submodule in evidence.submodules:
        child, child_exists = _validated_submodule_path(worktree, submodule.path)
        if not evidence.submodules_enabled:
            continue
        child_head = None if not child_exists else _detached_head(child, git)
        child_status_raw = b"" if not child_exists else _raw_status(child, git)
        child_status = () if not child_status_raw else _decode_status(child_status_raw)
        child_matches = child_exists and child_head == submodule.commit and not child_status
        submodule_observations.append(
            {
                "exists": child_exists,
                "head": child_head,
                "matches_evidence": child_matches,
                "path": submodule.path,
                "status_bytes": len(child_status_raw),
                "status_preview": "" if not child_status_raw else "[CONTENT OMITTED]",
                "status_sha256": _sha256(child_status_raw),
            }
        )
        matches = matches and child_matches
        if require_match and not child_matches:
            raise SourceDivergedError("submodule state diverged from evidence")
    return {
        "diff_bytes": size,
        "diff_preview": "" if size == 0 else "[CONTENT OMITTED]",
        "diff_sha256": digest,
        "head": head,
        "matches_evidence": matches,
        "status_bytes": len(status_raw),
        "status_preview": "" if not status_raw else "[CONTENT OMITTED]",
        "status_sha256": _sha256(status_raw),
        "submodules": submodule_observations,
        "tree": tree,
    }


def _admin_entry_state(owner: OwnershipV1, *, allow_missing_lock: bool) -> tuple[bool, bool]:
    admin = Path(owner.admin_path)
    if not _directory_exists_nofollow(admin, "worktree admin entry"):
        return False, False
    _validate_inode(admin, owner.admin_device, owner.admin_inode, "worktree admin entry")
    if admin.parent.name != "worktrees":
        raise SourceDivergedError("worktree admin entry is not a direct child")
    lock_file = admin / "locked"
    try:
        gitdir = (admin / "gitdir").read_text(encoding="utf-8").rstrip("\n")
        head = (admin / "HEAD").read_text(encoding="ascii").rstrip("\n")
    except OSError as exc:
        raise SourceDivergedError("worktree admin ownership metadata is incomplete") from exc
    try:
        lock_reason = lock_file.read_text(encoding="utf-8").rstrip("\n")
    except FileNotFoundError:
        if not allow_missing_lock:
            raise SourceDivergedError("worktree lock reason is missing") from None
        lock_present = False
    except OSError as exc:
        raise SourceDivergedError("worktree lock reason is unreadable") from exc
    else:
        lock_present = True
        if lock_reason != owner.lock_reason:
            raise SourceDivergedError("worktree lock reason changed")
    if gitdir != str(Path(owner.registered_path) / ".git"):
        raise SourceDivergedError("registered worktree path changed")
    if head != owner.detached_head:
        raise SourceDivergedError("worktree admin HEAD changed")
    return True, lock_present


def _remove_owned_tree(path: Path, device: int, inode: int, description: str) -> None:
    _validate_inode(path, device, inode, description)
    shutil.rmtree(path)


def _validate_ownership_binding(registry: SourceRegistryV1, owner: OwnershipV1) -> None:
    expected_worktree = Path(registry.worktree_path)
    mirror = Path(registry.mirror_path)
    expected_admin_parent = mirror / "worktrees"
    admin = Path(owner.admin_path)
    expected_lock_reason = _lock_reason(registry)
    if (
        Path(owner.registered_path) != expected_worktree
        or admin.parent != expected_admin_parent
        or admin.name != registry.preparation_id
        or owner.detached_head != registry.base_commit
        or owner.lock_reason != expected_lock_reason
    ):
        raise SourceDivergedError("persisted ownership is not bound to the preparation")
    for path, description in (
        (mirror, "source mirror"),
        (expected_admin_parent, "worktree admin parent"),
    ):
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise SourceDivergedError(f"{description} is unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise SourceDivergedError(f"{description} is unsafe")
    marker = Path(registry.registry_path) / "owner.json"
    marker_value = _read_model(marker, OwnershipV1)
    if marker_value != owner:
        raise SourceDivergedError("ownership marker changed")


def cleanup_source(
    preparation_id: str,
    *,
    home: Path,
    force_changed: bool = False,
) -> SourceCleanup:
    """Recover or remove exactly one authenticated worktree preparation."""

    if not _PREPARATION_ID_RE.fullmatch(preparation_id):
        raise SourcePolicyError("invalid preparation ID")
    layout = _layout(home, create=False)
    registry = _load_registry(layout, preparation_id)
    record = Path(registry.record_path)
    if registry.state is RegistryState.CLEANED:
        return SourceCleanup(preparation_id, RegistryState.CLEANED, record)
    evidence = _verify_evidence(registry)
    lock_path = layout.locks / f"source-{registry.source_id}.lock"
    with exclusive_lock(lock_path) as attempt:
        if not attempt.acquired:
            raise SourceBusyError(attempt.reason or "source lock is unavailable")
        stage = Path(registry.stage_path)
        scratch = Path(registry.scratch_path)
        stage_exists = _directory_exists_nofollow(stage, "source stage")
        if not _directory_exists_nofollow(scratch, "cleanup scratch"):
            scratch_parent = stage if stage_exists else Path(registry.registry_path)
            scratch = scratch_parent / f"cleanup-scratch-{secrets.token_hex(8)}"
            scratch.mkdir(mode=0o700)
        git = GitBoundary.create(
            git_home=layout.git_home,
            scratch=scratch,
            locator=registry.mirror_path,
        )
        owner = registry.ownership
        worktree = Path(registry.worktree_path)
        if owner is not None:
            _validate_ownership_binding(registry, owner)
        directory_exists = _directory_exists_nofollow(worktree, "worktree directory")
        admin_exists, admin_locked = (
            (False, False)
            if owner is None
            else _admin_entry_state(
                owner,
                allow_missing_lock=registry.state is RegistryState.CLEANUP_STARTED,
            )
        )
        observed: dict[str, Any] = {
            "admin_exists": admin_exists,
            "admin_locked": admin_locked,
            "directory_exists": directory_exists,
            "force_changed": force_changed,
        }
        if owner is None:
            if directory_exists or admin_exists:
                raise SourceDivergedError("unowned source state cannot be removed")
        else:
            if directory_exists:
                _validate_inode(
                    worktree, owner.worktree_device, owner.worktree_inode, "worktree directory"
                )
            if directory_exists and admin_exists and evidence is not None:
                observed.update(
                    _verify_candidate_for_cleanup(
                        registry,
                        evidence,
                        git,
                        require_match=not force_changed,
                    )
                )
        stranded_paths = (Path(registry.stage_path), Path(registry.mirror_temporary_path))
        stranded_existing = {
            stranded: _directory_exists_nofollow(stranded, "stranded source path")
            for stranded in stranded_paths
        }
        for stranded, exists in stranded_existing.items():
            if exists and stranded.parent not in {layout.records, layout.mirrors}:
                raise SourceDivergedError("stranded source path is unsafe")
        registry = _transition(
            registry,
            RegistryState.CLEANUP_STARTED,
            details=observed,
        )
        if owner is not None:
            if directory_exists and admin_exists:
                if admin_locked:
                    _git_run(
                        git,
                        [
                            "--git-dir",
                            registry.mirror_path,
                            "worktree",
                            "unlock",
                            registry.worktree_path,
                        ],
                        cwd=layout.root,
                    )
                _git_run(
                    git,
                    [
                        "--git-dir",
                        registry.mirror_path,
                        "worktree",
                        "remove",
                        "--force",
                        registry.worktree_path,
                    ],
                    cwd=layout.root,
                )
            elif admin_exists:
                _remove_owned_tree(
                    Path(owner.admin_path), owner.admin_device, owner.admin_inode, "admin entry"
                )
            elif directory_exists:
                _remove_owned_tree(
                    worktree, owner.worktree_device, owner.worktree_inode, "worktree directory"
                )
        for stranded, exists in stranded_existing.items():
            if exists:
                shutil.rmtree(stranded)
        final_directory = _directory_exists_nofollow(worktree, "worktree directory")
        final_admin = (
            False
            if owner is None
            else _directory_exists_nofollow(Path(owner.admin_path), "worktree admin entry")
        )
        if final_directory or final_admin:
            raise SourceDivergedError("cleanup did not remove both ownership sides")
        observed.update(admin_exists_after=final_admin, directory_exists_after=final_directory)
        registry = _transition(
            registry,
            RegistryState.CLEANED,
            details=observed,
            failure_code=None,
        )
        return SourceCleanup(preparation_id, registry.state, record)

"""Immutable, content-addressed source snapshots for build attempts."""

from __future__ import annotations

import contextlib
import hashlib
import os
import secrets
import shutil
import stat
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from strixlab.locks import LockStatus, exclusive_lock
from strixlab.secure_fs import (
    fsync_directory,
    readonly_open_flags,
    rename_noreplace,
    write_all,
    write_exclusive,
)
from strixlab.serialization import canonical_json_bytes
from strixlab.source_identity import length_frame

_HASH_PATTERN = r"^[0-9a-f]{64}$"
_SNAPSHOT_PATTERN = r"^snapshot-sha256:[0-9a-f]{64}$"
_CANDIDATE_PATTERN = r"^candidate-sha256:[0-9a-f]{64}$"
_CONTENT_TREE_PATTERN = r"^content-tree-sha256:[0-9a-f]{64}$"
_MANIFEST_NAME = "snapshot.json"
_LOCKS_DIRNAME = ".locks"
_LOCK_RETRY_SECONDS = 0.05
_LOCK_TIMEOUT_SECONDS = 300.0


class SnapshotError(RuntimeError):
    """A source tree or persisted snapshot is unsafe, unstable, or corrupt."""


class SnapshotEntryV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    path: str
    kind: Literal["file", "symlink"]
    mode: int = Field(ge=0, le=0o7777)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=_HASH_PATTERN)
    link_target: str | None


class SnapshotManifestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    snapshot_id: str = Field(pattern=_SNAPSHOT_PATTERN)
    candidate_id: str = Field(pattern=_CANDIDATE_PATTERN)
    content_tree_id: str = Field(pattern=_CONTENT_TREE_PATTERN)
    entries: tuple[SnapshotEntryV1, ...]


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    snapshot_id: str
    root: Path
    source: Path
    manifest: SnapshotManifestV1


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
        or path.is_absolute()
        or path == PurePosixPath(".")
        or ".." in path.parts
        or path.as_posix() != value
        or path.parts[0] == ".git"
    ):
        raise SnapshotError(f"unsafe snapshot-relative path: {value!r}")
    return path


def _owned_directory(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise SnapshotError(f"snapshot directory is missing: {path}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise SnapshotError(f"snapshot directory is unsafe: {path}")
    return metadata


def _hash_regular(path: Path) -> tuple[int, int, str]:
    try:
        descriptor = os.open(path, readonly_open_flags())
    except OSError as exc:
        raise SnapshotError(f"snapshot input is unavailable: {path}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid():
            raise SnapshotError(f"snapshot input is not an owned regular file: {path}")
        while chunk := os.read(descriptor, 64 * 1024):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or size != before.st_size
        ):
            raise SnapshotError(f"snapshot input changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return stat.S_IMODE(before.st_mode), size, digest.hexdigest()


def _safe_link(root: Path, path: Path, target: str) -> None:
    if not target or "\x00" in target or PurePosixPath(target).is_absolute():
        raise SnapshotError(f"snapshot symlink is unsafe: {path}")
    resolved_root = root.resolve()
    resolved_target = (path.parent / target).resolve(strict=False)
    if not resolved_target.is_relative_to(resolved_root):
        raise SnapshotError(f"snapshot symlink escapes the source root: {path}")


def _symlink_entry(
    root: Path, child: Path, relative: str, metadata: os.stat_result
) -> SnapshotEntryV1:
    target = os.readlink(child)
    _safe_link(root, child, target)
    return SnapshotEntryV1(
        path=relative,
        kind="symlink",
        mode=stat.S_IMODE(metadata.st_mode),
        size_bytes=len(os.fsencode(target)),
        sha256=hashlib.sha256(os.fsencode(target)).hexdigest(),
        link_target=target,
    )


def _scan_source(root: Path) -> tuple[SnapshotEntryV1, ...]:
    _owned_directory(root)
    entries: list[SnapshotEntryV1] = []
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        _owned_directory(current)
        names[:] = sorted(name for name in names if not (current == root and name == ".git"))
        files.sort()
        for name in tuple(names):
            child = current / name
            metadata = child.lstat()
            relative = child.relative_to(root).as_posix()
            if stat.S_ISLNK(metadata.st_mode):
                names.remove(name)
                entries.append(_symlink_entry(root, child, relative, metadata))
            elif not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
                raise SnapshotError(f"snapshot tree contains an unsafe directory: {child}")
        for name in files:
            child = current / name
            relative = child.relative_to(root).as_posix()
            if current == root and name == ".git":
                continue
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                entries.append(_symlink_entry(root, child, relative, metadata))
                continue
            mode, size, digest = _hash_regular(child)
            entries.append(
                SnapshotEntryV1(
                    path=relative,
                    kind="file",
                    mode=mode & 0o555,
                    size_bytes=size,
                    sha256=digest,
                    link_target=None,
                )
            )
    return tuple(sorted(entries, key=lambda entry: entry.path))


def _snapshot_id(
    candidate_id: str, content_tree_id: str, entries: tuple[SnapshotEntryV1, ...]
) -> str:
    entry_bytes = canonical_json_bytes([entry.model_dump(mode="json") for entry in entries])
    framed = length_frame(
        "strixlab.build.source-snapshot.v1",
        (
            ("candidate-id", candidate_id.encode("ascii")),
            ("content-tree-id", content_tree_id.encode("ascii")),
            ("entries", entry_bytes),
        ),
    )
    return "snapshot-sha256:" + hashlib.sha256(framed).hexdigest()


def _copy_entry(source: Path, destination: Path, entry: SnapshotEntryV1) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if entry.kind == "symlink":
        if entry.link_target is None:
            raise SnapshotError("snapshot symlink has no target")
        destination.symlink_to(entry.link_target)
        return
    source_descriptor = os.open(source, readonly_open_flags())
    output = os.open(destination, os.O_CLOEXEC | os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        while chunk := os.read(source_descriptor, 64 * 1024):
            write_all(output, chunk)
        os.fchmod(output, entry.mode)
        os.fsync(output)
    finally:
        os.close(output)
        os.close(source_descriptor)


def _fsync_and_seal(root: Path) -> None:
    directories = [Path(directory) for directory, _names, _files in os.walk(root)]
    for directory in reversed(directories):
        fsync_directory(directory)
        os.chmod(directory, 0o500)


def _remove_tree(root: Path) -> None:
    if not root.exists():
        return
    for directory, _names, _files in os.walk(root, topdown=True, followlinks=False):
        Path(directory).chmod(0o700)
    shutil.rmtree(root)


def _remove_destination(destination: Path) -> None:
    """Delete whatever occupies a published path, symlink or sealed tree alike."""

    try:
        metadata = destination.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
        _remove_tree(destination)
    else:
        destination.unlink()
    fsync_directory(destination.parent)


def _read_manifest(root: Path) -> SnapshotManifestV1:
    try:
        return SnapshotManifestV1.model_validate_json((root / _MANIFEST_NAME).read_bytes())
    except (OSError, ValueError) as exc:
        raise SnapshotError(f"snapshot manifest is invalid: {root}") from exc


def _verify_content(root: Path) -> SourceSnapshot:
    _owned_directory(root)
    manifest = _read_manifest(root)
    source = root / "source"
    entries = _scan_source(source)
    expected_id = _snapshot_id(manifest.candidate_id, manifest.content_tree_id, entries)
    if entries != manifest.entries or expected_id != manifest.snapshot_id:
        raise SnapshotError("snapshot content does not match its manifest")
    return SourceSnapshot(manifest.snapshot_id, root, source, manifest)


def verify_snapshot(root: Path) -> SourceSnapshot:
    """Verify a persisted immutable source snapshot and return its source root."""

    snapshot = _verify_content(root)
    if root.name != snapshot.snapshot_id:
        raise SnapshotError("snapshot content does not match its manifest")
    return snapshot


@contextlib.contextmanager
def _snapshot_lock(snapshots: Path, identifier: str) -> Iterator[None]:
    locks = snapshots / _LOCKS_DIRNAME
    with contextlib.suppress(FileExistsError):
        locks.mkdir(mode=0o700)
    _owned_directory(locks)
    lock_path = locks / f"{identifier.removeprefix('snapshot-sha256:')}.lock"
    deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
    while True:
        with exclusive_lock(lock_path) as lock:
            if lock.acquired:
                yield
                return
            reason = lock.reason or "snapshot publication lock is unavailable"
            if lock.status is not LockStatus.CONTENDED or time.monotonic() >= deadline:
                raise SnapshotError(reason)
        time.sleep(_LOCK_RETRY_SECONDS)


def _retire_snapshot(destination: Path, expected: SourceSnapshot) -> None:
    try:
        before = destination.lstat()
    except FileNotFoundError as exc:
        raise SnapshotError("leased snapshot disappeared before retirement") from exc
    current = verify_snapshot(destination)
    after = destination.lstat()
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
        or before.st_uid != os.geteuid()
        or before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or current.manifest != expected.manifest
    ):
        raise SnapshotError("snapshot ownership changed before retirement")
    _remove_destination(destination)


def _healthy_destination(destination: Path) -> SourceSnapshot | None:
    try:
        destination.lstat()
    except FileNotFoundError:
        return None
    try:
        return _verified_published(destination)
    except SnapshotError:
        return None


def _verified_published(destination: Path) -> SourceSnapshot:
    try:
        return verify_snapshot(destination)
    except SnapshotError:
        _remove_destination(destination)
        raise


def _publish_snapshot(
    source: Path,
    snapshots: Path,
    destination: Path,
    identifier: str,
    *,
    candidate_id: str,
    content_tree_id: str,
    before: tuple[SnapshotEntryV1, ...],
) -> SourceSnapshot:
    stage = snapshots / f".{identifier}.{secrets.token_hex(8)}.tmp"
    try:
        stage.mkdir(mode=0o700)
        stage_source = stage / "source"
        stage_source.mkdir(mode=0o700)
        for entry in before:
            relative = _safe_relative(entry.path)
            _copy_entry(source / relative, stage_source / relative, entry)
        after = _scan_source(source)
        if after != before:
            raise SnapshotError("source tree changed while snapshotting")
        manifest = SnapshotManifestV1(
            snapshot_id=identifier,
            candidate_id=candidate_id,
            content_tree_id=content_tree_id,
            entries=before,
        )
        write_exclusive(
            stage / _MANIFEST_NAME,
            canonical_json_bytes(manifest.model_dump(mode="json")),
            0o400,
        )
        _fsync_and_seal(stage)
        if _verify_content(stage).snapshot_id != identifier:
            raise SnapshotError("staged snapshot does not match its intended identity")
        try:
            rename_noreplace(stage, destination)
            fsync_directory(snapshots)
        except FileExistsError:
            _remove_tree(stage)
    except BaseException:
        _remove_tree(stage)
        raise
    return _verified_published(destination)


@contextlib.contextmanager
def lease_snapshot(
    source: Path,
    snapshots: Path,
    *,
    candidate_id: str,
    content_tree_id: str,
    retire: bool = True,
) -> Iterator[SourceSnapshot]:
    """Hold the per-snapshot lock and retire it after its consumer completes."""

    _owned_directory(source)
    _owned_directory(snapshots)
    before = _scan_source(source)
    identifier = _snapshot_id(candidate_id, content_tree_id, before)
    destination = snapshots / identifier
    with _snapshot_lock(snapshots, identifier):
        snapshot = _healthy_destination(destination)
        if snapshot is None:
            snapshot = _publish_snapshot(
                source,
                snapshots,
                destination,
                identifier,
                candidate_id=candidate_id,
                content_tree_id=content_tree_id,
                before=before,
            )
        completed = False
        try:
            yield snapshot
            completed = True
        finally:
            if retire:
                if completed:
                    _retire_snapshot(destination, snapshot)
                else:
                    with contextlib.suppress(SnapshotError, OSError):
                        _retire_snapshot(destination, snapshot)


def materialize_snapshot(
    source: Path,
    snapshots: Path,
    *,
    candidate_id: str,
    content_tree_id: str,
) -> SourceSnapshot:
    """Copy and atomically publish one stable, read-only source snapshot."""

    with lease_snapshot(
        source,
        snapshots,
        candidate_id=candidate_id,
        content_tree_id=content_tree_id,
        retire=False,
    ) as snapshot:
        return snapshot

"""Immutable record publication and content verification."""

from __future__ import annotations

import hashlib
import os
import secrets
import shutil
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from strixlab.secure_fs import (
    exclusive_create_flags,
    fsync_directory,
    rename_noreplace,
    write_exclusive,
)
from strixlab.serialization import canonical_json_bytes
from strixlab.source_identity import length_frame

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_MANIFEST_NAME = "record-manifest.json"


class RecordError(RuntimeError):
    """An immutable build record is unsafe, corrupt, or divergent."""


class RecordFileV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    path: str
    mode: int = Field(ge=0, le=0o7777)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)


class RecordManifestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    files: tuple[RecordFileV1, ...]


@dataclass(frozen=True, slots=True)
class RecordVerification:
    path: Path
    record_sha256: str
    device: int
    inode: int
    files: tuple[RecordFileV1, ...]


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path == PurePosixPath(".")
        or ".." in path.parts
        or path.as_posix() != value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise RecordError(f"unsafe record-relative path: {value!r}")
    return path


def _record_digest(manifest_bytes: bytes) -> str:
    framed = length_frame(
        "strixlab.build.record-manifest.v1",
        (("manifest", manifest_bytes),),
    )
    return "record-sha256:" + hashlib.sha256(framed).hexdigest()


def record_manifest_digest(manifest_bytes: bytes) -> str:
    """Public record digest for canonical ``record-manifest.json`` bytes.

    Narrowly reusable by run/bundle verification to re-derive and bind the immutable
    record digest without re-copying the tree.
    """

    return _record_digest(manifest_bytes)


def _owned_directory(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise RecordError(f"record directory is missing: {path}") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise RecordError(f"record directory is unsafe: {path}")
    return metadata


def _open_owned_regular(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_CLOEXEC | os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RecordError(f"record input is unavailable: {path}") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
        os.close(descriptor)
        raise RecordError(f"record input is not an owned regular file: {path}")
    return descriptor, metadata


def _copy_regular(source: Path, destination: Path, relative: str) -> RecordFileV1:
    descriptor, before = _open_owned_regular(source)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    output = os.open(destination, exclusive_create_flags(), stat.S_IMODE(before.st_mode))
    digest = hashlib.sha256()
    size = 0
    try:
        while content := os.read(descriptor, 64 * 1024):
            digest.update(content)
            size += len(content)
            view = memoryview(content)
            while view:
                written = os.write(output, view)
                if written <= 0:
                    raise OSError("short record write")
                view = view[written:]
        after = os.fstat(descriptor)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or size != before.st_size
        ):
            raise RecordError(f"record input changed while copying: {source}")
        os.fchmod(output, stat.S_IMODE(before.st_mode))
        os.fsync(output)
    finally:
        os.close(output)
        os.close(descriptor)
    return RecordFileV1(
        path=relative,
        mode=stat.S_IMODE(before.st_mode),
        size_bytes=size,
        sha256=digest.hexdigest(),
    )


def _walk_owned_tree(root: Path) -> tuple[tuple[str, Path], ...]:
    _owned_directory(root)
    values: list[tuple[str, Path]] = []
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        current = Path(directory)
        _owned_directory(current)
        names.sort()
        files.sort()
        for name in tuple(names):
            child = current / name
            metadata = child.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise RecordError(f"record tree contains an unsafe directory: {child}")
        for name in files:
            child = current / name
            relative = child.relative_to(root).as_posix()
            _safe_relative(relative)
            values.append((relative, child))
    return tuple(values)


def _fsync_tree(root: Path) -> None:
    directories = [Path(directory) for directory, _names, _files in os.walk(root)]
    for directory in reversed(directories):
        fsync_directory(directory)


def hash_owned_tree(
    source: Path, *, skip: Callable[[str], bool] | None = None
) -> tuple[RecordFileV1, ...]:
    """Deterministically hash every owned regular file under one evidence tree.

    Uses the same nofollow, ownership-checked, descriptor-based stable-hashing
    machinery as immutable record verification, so a caller building a
    prepublication inventory cannot drift from the record verifier. ``skip`` (given
    a record-relative POSIX path) excludes entries from the returned inventory while
    still walking the whole tree with the same safety checks; skipped subtrees are
    not hashed. The result is sorted by path for a canonical inventory.
    """

    files: list[RecordFileV1] = []
    for relative, path in _walk_owned_tree(source):
        if skip is not None and skip(relative):
            continue
        size, digest, metadata = _hash_regular(path)
        files.append(
            RecordFileV1(
                path=relative,
                mode=stat.S_IMODE(metadata.st_mode),
                size_bytes=size,
                sha256=digest,
            )
        )
    return tuple(sorted(files, key=lambda item: item.path))


def record_source_digest(source: Path) -> str:
    """Compute the record digest for an owned evidence tree without copying it."""

    files = hash_owned_tree(source)
    if any(entry.path == _MANIFEST_NAME for entry in files):
        raise RecordError("record input cannot supply record-manifest.json")
    manifest = RecordManifestV1(files=files)
    return _record_digest(canonical_json_bytes(manifest.model_dump(mode="json")))


def publish_record(source: Path, destination: Path) -> RecordVerification:
    """Copy one owned evidence tree and publish it immutably with no replacement."""

    _owned_directory(source)
    _owned_directory(destination.parent)
    stage = destination.parent / f".{destination.name}.{secrets.token_hex(16)}.tmp"
    try:
        stage.mkdir(mode=0o700)
        source_files = _walk_owned_tree(source)
        if any(relative == _MANIFEST_NAME for relative, _path in source_files):
            raise RecordError("record input cannot supply record-manifest.json")
        files = tuple(
            _copy_regular(path, stage / PurePosixPath(relative), relative)
            for relative, path in source_files
        )
        manifest = RecordManifestV1(files=tuple(sorted(files, key=lambda item: item.path)))
        manifest_bytes = canonical_json_bytes(manifest.model_dump(mode="json"))
        write_exclusive(stage / _MANIFEST_NAME, manifest_bytes)
        _fsync_tree(stage)
        rename_noreplace(stage, destination)
        fsync_directory(destination.parent)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return verify_record(destination)


def _read_regular(path: Path) -> tuple[bytes, os.stat_result]:
    descriptor, before = _open_owned_regular(path)
    chunks: list[bytes] = []
    try:
        while chunk := os.read(descriptor, 64 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if before.st_dev != after.st_dev or before.st_ino != after.st_ino:
            raise RecordError(f"record file changed while reading: {path}")
    finally:
        os.close(descriptor)
    return b"".join(chunks), before


def _hash_regular(path: Path) -> tuple[int, str, os.stat_result]:
    descriptor, before = _open_owned_regular(path)
    digest = hashlib.sha256()
    size = 0
    try:
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
            raise RecordError(f"record file changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return size, digest.hexdigest(), before


def verify_record(path: Path) -> RecordVerification:
    """Verify every immutable record file against its canonical manifest."""

    root_metadata = _owned_directory(path)
    manifest_bytes, _metadata = _read_regular(path / _MANIFEST_NAME)
    try:
        manifest = RecordManifestV1.model_validate_json(manifest_bytes)
    except ValueError as exc:
        raise RecordError("record manifest is invalid") from exc
    expected = {entry.path: entry for entry in manifest.files}
    if len(expected) != len(manifest.files):
        raise RecordError("record manifest contains duplicate paths")
    actual_paths = {
        relative for relative, _child in _walk_owned_tree(path) if relative != _MANIFEST_NAME
    }
    if actual_paths != set(expected):
        raise RecordError("record payload set does not match its manifest")
    for relative, entry in expected.items():
        size, digest, metadata = _hash_regular(path / _safe_relative(relative))
        if (
            stat.S_IMODE(metadata.st_mode) != entry.mode
            or size != entry.size_bytes
            or digest != entry.sha256
        ):
            raise RecordError(f"record payload integrity mismatch: {relative}")
    return RecordVerification(
        path=path,
        record_sha256=_record_digest(manifest_bytes),
        device=root_metadata.st_dev,
        inode=root_metadata.st_ino,
        files=manifest.files,
    )

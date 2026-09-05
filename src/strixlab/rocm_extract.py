"""Synthetic structural quarantine; observed-empty metadata is never admission.

Caller-owned trees must remain quiescent. Named metadata probes and guarded
binding checks are not atomic against same-UID writers or transient mount/ABA
replacement. Failures leave a new quarantine in place, without cleanup or reuse.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict

from strixlab.rocm_archive import (
    ArchiveEntryV1,
    ArchiveError,
    ArchiveManifestV1,
    _consume_archive,
    _ProvisionalChunk,
    _ProvisionalEnd,
    _ProvisionalEvent,
    _ProvisionalStart,
    inspect_archive,
)
from strixlab.rocm_metadata import (
    InodeIdentityV1,
    InodeMetadataObservationV1,
    MetadataError,
    _identity,
    observe_inode_metadata,
)
from strixlab.rocm_prefix import PrefixError, PrefixInventoryV1, _openat2, inspect_prefix

_MAX_DEPTH = 240
_MAX_PATH_BYTES = 4096
_CHUNK_BYTES = 65536

type _Created = tuple[int, int]


class QuarantineError(ValueError):
    """Bounded reason and, only after successful mkdir, retained-leaf evidence."""

    def __init__(
        self,
        reason: str,
        quarantine_name: str | None = None,
        root_identity: _Created | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.quarantine_name = quarantine_name
        self.root_identity = root_identity


class QuarantineResultV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    validation: Literal["complete"] = "complete"
    scope: Literal["structural-quarantine-only"] = "structural-quarantine-only"
    metadata_coverage: Literal["unknown"] = "unknown"
    link_closure: Literal["not-checked"] = "not-checked"
    archive: ArchiveManifestV1
    inventory: PrefixInventoryV1


def _text(value: str, maximum: int) -> None:
    if not isinstance(value, str) or len(value) > maximum:
        raise QuarantineError("quarantine-text-limit")
    try:
        data = value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise QuarantineError("quarantine-text-utf8") from exc
    if len(data) > maximum or b"\0" in data:
        raise QuarantineError("quarantine-text-limit")


def _leaf(name: str) -> None:
    _text(name, 255)
    if not name or name in {".", ".."} or "/" in name:
        raise QuarantineError("quarantine-leaf")


def _preflight(manifest: ArchiveManifestV1) -> None:
    for entry in manifest.entries:
        _text(entry.path, _MAX_PATH_BYTES)
        parts = entry.path.split("/")
        if len(parts) > _MAX_DEPTH:
            raise QuarantineError("quarantine-depth-limit")
        for part in parts:
            _leaf(part)
        if entry.kind == "file" and not entry.mode & 0o400:
            raise QuarantineError("quarantine-final-owner-permissions")
        if entry.kind == "directory" and entry.mode & 0o500 != 0o500:
            raise QuarantineError("quarantine-final-owner-permissions")


def _created(identity: InodeIdentityV1) -> _Created:
    return identity.dev, identity.ino


def _stable_parent(identity: InodeIdentityV1) -> tuple[int, ...]:
    return identity.dev, identity.ino, identity.mode, identity.uid, identity.gid


def _check_inode(
    identity: InodeIdentityV1, owner: tuple[int, int], kind: str, created: _Created | None = None
) -> None:
    correct_type = {
        "directory": stat.S_ISDIR,
        "file": stat.S_ISREG,
        "symlink": stat.S_ISLNK,
    }[kind]
    if (
        not correct_type(identity.mode)
        or (identity.uid, identity.gid) != owner
        or (kind != "directory" and identity.nlink != 1)
        or (created is not None and _created(identity) != created)
    ):
        raise QuarantineError("quarantine-inode-mismatch")


def _empty_metadata(observation: InodeMetadataObservationV1, identity: InodeIdentityV1) -> None:
    if (
        observation.validation != "complete"
        or observation.coverage != "unknown"
        or observation.list_status != "observed"
        or observation.list_errno is not None
        or observation.name_list_size_bytes != 0
        or observation.names_bytes_escaped != ()
        or any(
            value != identity
            for value in (
                observation.leaf_before,
                observation.leaf_opened,
                observation.leaf_after,
                observation.leaf_named_after,
            )
        )
    ):
        raise QuarantineError("quarantine-metadata-not-observed-empty")


@contextmanager
def _opened(parent: int, name: str, flags: int) -> Iterator[int]:
    fd = _openat2(parent, name, flags)
    try:
        yield fd
    finally:
        os.close(fd)


def _observe(fd: int, parent: int, name: str) -> InodeIdentityV1:
    identity = _identity(os.fstat(fd))
    parent_before = _identity(os.fstat(parent))
    observation = observe_inode_metadata(parent, name)
    _empty_metadata(observation, identity)
    if (
        observation.parent_before != parent_before
        or observation.parent_after != parent_before
        or _identity(os.fstat(parent)) != parent_before
    ):
        raise QuarantineError("quarantine-parent-drift")
    with _opened(parent, name, os.O_PATH) as named:
        if _identity(os.fstat(named)) != identity:
            raise QuarantineError("quarantine-binding-drift")
    if _identity(os.fstat(fd)) != identity:
        raise QuarantineError("quarantine-inode-drift")
    return identity


@dataclass
class _Tree:
    parent: int
    name: str
    owner: tuple[int, int]
    created: dict[str, _Created] = field(default_factory=dict)

    @contextmanager
    def directory(self, path: str) -> Iterator[int]:
        # One chain at a time; close the previous descriptor on every descent.
        fd = _openat2(self.parent, self.name, os.O_RDONLY | os.O_DIRECTORY)
        try:
            _check_inode(_identity(os.fstat(fd)), self.owner, "directory", self.created[""])
            prefix = ""
            for part in path.split("/") if path else ():
                child = _openat2(fd, part, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    prefix = f"{prefix}/{part}" if prefix else part
                    _check_inode(
                        _identity(os.fstat(child)), self.owner, "directory", self.created[prefix]
                    )
                except BaseException:
                    os.close(child)
                    raise
                os.close(fd)
                fd = child
            yield fd
        finally:
            os.close(fd)

    def new_directory(self, parent: int, name: str, path: str) -> None:
        # mkdir is performed by the caller so root error evidence is set immediately.
        with _opened(parent, name, os.O_PATH) as held:
            identity = _identity(os.fstat(held))
            self.created[path] = _created(identity)
            _check_inode(identity, self.owner, "directory")
            if stat.S_IMODE(identity.mode) & 0o700 != 0o700:
                raise QuarantineError("quarantine-creation-owner-permissions")
            with _opened(parent, name, os.O_RDONLY | os.O_DIRECTORY) as readable:
                if _identity(os.fstat(readable)) != identity:
                    raise QuarantineError("quarantine-binding-drift")
                _observe(readable, parent, name)
                os.fchmod(readable, 0o700)
                _observe(readable, parent, name)

    def final_mode(self, entry: ArchiveEntryV1) -> None:
        parent_path, _, name = entry.path.rpartition("/")
        with self.directory(parent_path) as parent:
            flags = os.O_RDONLY | (os.O_DIRECTORY if entry.kind == "directory" else os.O_NONBLOCK)
            with _opened(parent, name, flags) as fd:
                _check_inode(
                    _identity(os.fstat(fd)), self.owner, entry.kind, self.created[entry.path]
                )
                _observe(fd, parent, name)
                os.fchmod(fd, entry.mode)
                identity = _observe(fd, parent, name)
                if stat.S_IMODE(identity.mode) != entry.mode:
                    raise QuarantineError("quarantine-final-mode")


@dataclass
class _Writer:
    tree: _Tree
    expected: dict[str, ArchiveEntryV1]
    active: ArchiveEntryV1 | None = None
    fd: int | None = None
    written: int = 0
    completed: set[str] = field(default_factory=set)

    def close(self) -> None:
        if self.fd is not None:
            fd, self.fd = self.fd, None
            os.close(fd)

    def consume(self, event: _ProvisionalEvent) -> None:
        if isinstance(event, _ProvisionalStart):
            header = event.header
            expected = self.expected.get(header.path)
            if (
                self.active is not None
                or expected is None
                or header.path in self.completed
                or header.entry(None) != expected.model_copy(update={"sha256": None})
            ):
                raise QuarantineError("quarantine-second-header-mismatch")
            self.active, self.written = expected, 0
            if expected.kind == "file":
                parent_path, _, name = expected.path.rpartition("/")
                with self.tree.directory(parent_path) as parent:
                    self.fd = os.open(
                        name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                        0o600,
                        dir_fd=parent,
                    )
                    identity = _identity(os.fstat(self.fd))
                    self.tree.created[expected.path] = _created(identity)
                    _check_inode(identity, self.tree.owner, "file")
                    if stat.S_IMODE(identity.mode) & 0o600 != 0o600:
                        raise QuarantineError("quarantine-creation-owner-permissions")
                    _observe(self.fd, parent, name)
                    os.fchmod(self.fd, 0o600)
                    _observe(self.fd, parent, name)
        elif isinstance(event, _ProvisionalChunk):
            if (
                self.active is None
                or self.active.kind != "file"
                or self.fd is None
                or not 0 < len(event.data) <= _CHUNK_BYTES
                or self.written + len(event.data) > self.active.payload_size_bytes
            ):
                raise QuarantineError("quarantine-second-chunk-mismatch")
            pending = memoryview(event.data)
            while pending:
                count = os.write(self.fd, pending)
                if not 0 < count <= len(pending):
                    raise QuarantineError("quarantine-write-progress")
                self.written += count
                pending = pending[count:]
        elif isinstance(event, _ProvisionalEnd):
            if self.active is None or event.entry != self.active:
                raise QuarantineError("quarantine-second-entry-mismatch")
            if self.active.kind == "file":
                if self.fd is None or self.written != self.active.payload_size_bytes:
                    raise QuarantineError("quarantine-second-size-mismatch")
                parent_path, _, name = self.active.path.rpartition("/")
                with self.tree.directory(parent_path) as parent:
                    identity = _observe(self.fd, parent, name)
                    _check_inode(
                        identity, self.tree.owner, "file", self.tree.created[self.active.path]
                    )
                    if identity.size != self.written:
                        raise QuarantineError("quarantine-written-size-mismatch")
                self.close()
            self.completed.add(self.active.path)
            self.active = None


def _project(manifest: ArchiveManifestV1, inventory: PrefixInventoryV1, tree: _Tree) -> None:
    expected = {entry.path: entry for entry in manifest.entries}
    if (
        inventory.member_count != len(expected)
        or len(inventory.entries) != len(expected)
        or {entry.path for entry in inventory.entries} != expected.keys()
        or inventory.metadata_coverage != "unknown"
        or inventory.link_closure != "not-checked"
        or inventory.root.path != ""
    ):
        raise QuarantineError("quarantine-final-member-mismatch")
    for item in (inventory.root, *inventory.entries):
        archive = expected.get(item.path)
        kind = archive.kind if archive else "directory"
        mode = archive.mode if archive else 0o700
        length = (
            archive.payload_size_bytes
            if archive and kind == "file"
            else len(archive.link_target.encode("utf-8"))
            if archive and archive.link_target is not None
            else None
        )
        _check_inode(item.identity, tree.owner, kind, tree.created[item.path])
        _empty_metadata(item.metadata, item.identity)
        if (
            item.kind != kind
            or item.mode != mode
            or (item.uid, item.gid) != tree.owner
            or item.byte_length != length
            or item.sha256 != (archive.sha256 if archive else None)
            or item.link_target != (archive.link_target if archive else None)
            or item.nlink != (None if kind == "directory" else 1)
        ):
            raise QuarantineError("quarantine-final-entry-mismatch")


def extract_quarantine(
    archive_parent_fd: int,
    archive_name: str,
    destination_grandparent_fd: int,
    destination_parent_name: str,
    quarantine_name: str,
) -> QuarantineResultV1:
    """Create one new structural quarantine; every post-mkdir failure retains it."""

    retained: str | None = None
    tree: _Tree | None = None
    try:
        for name in (archive_name, destination_parent_name, quarantine_name):
            _leaf(name)
        with ExitStack() as resources:
            archive_parent = os.dup(archive_parent_fd)
            resources.callback(os.close, archive_parent)
            grandparent = os.dup(destination_grandparent_fd)
            resources.callback(os.close, grandparent)
            first = inspect_archive(archive_parent, archive_name)
            _preflight(first)
            owner = os.geteuid(), os.getegid()
            parent_held = resources.enter_context(
                _opened(grandparent, destination_parent_name, os.O_PATH)
            )
            before = _identity(os.fstat(parent_held))
            _check_inode(before, owner, "directory")
            if stat.S_IMODE(before.mode) != 0o700:
                raise QuarantineError("quarantine-parent-mode")
            parent = resources.enter_context(
                _opened(grandparent, destination_parent_name, os.O_RDONLY | os.O_DIRECTORY)
            )
            if _identity(os.fstat(parent)) != before:
                raise QuarantineError("quarantine-binding-drift")
            _observe(parent, grandparent, destination_parent_name)
            tree = _Tree(parent, quarantine_name, owner)
            os.mkdir(quarantine_name, 0o700, dir_fd=parent)
            retained = quarantine_name
            tree.new_directory(parent, quarantine_name, "")
            directories = sorted(
                (entry for entry in first.entries if entry.kind == "directory"),
                key=lambda entry: (entry.path.count("/"), entry.path.encode("utf-8")),
            )
            for entry in directories:
                parent_path, _, name = entry.path.rpartition("/")
                with tree.directory(parent_path) as directory:
                    os.mkdir(name, 0o700, dir_fd=directory)
                    tree.new_directory(directory, name, entry.path)
            writer = _Writer(tree, {entry.path: entry for entry in first.entries})
            try:
                second = _consume_archive(archive_parent, archive_name, writer.consume)
            finally:
                writer.close()
            if (
                writer.active is not None
                or writer.completed != writer.expected.keys()
                or second.canonical_bytes() != first.canonical_bytes()
            ):
                raise QuarantineError("quarantine-second-manifest-mismatch")
            for entry in first.entries:
                if entry.kind == "symlink":
                    assert entry.link_target is not None
                    parent_path, _, name = entry.path.rpartition("/")
                    with tree.directory(parent_path) as directory:
                        os.symlink(entry.link_target, name, dir_fd=directory)
                        with _opened(directory, name, os.O_PATH) as fd:
                            identity = _identity(os.fstat(fd))
                            tree.created[entry.path] = _created(identity)
                            _check_inode(identity, owner, "symlink")
                            _observe(fd, directory, name)
            for entry in first.entries:
                if entry.kind == "file":
                    tree.final_mode(entry)
            for entry in reversed(directories):
                tree.final_mode(entry)
            inventory = inspect_prefix(parent, quarantine_name)
            _project(first, inventory, tree)
            with _opened(parent, quarantine_name, os.O_PATH) as root:
                if _identity(os.fstat(root)) != inventory.root.identity:
                    raise QuarantineError("quarantine-binding-drift")
            after = _observe(parent, grandparent, destination_parent_name)
            if _stable_parent(after) != _stable_parent(before):
                raise QuarantineError("quarantine-parent-drift")
            return QuarantineResultV1(archive=first, inventory=inventory)
    except BaseException as exc:
        if isinstance(exc, QuarantineError):
            reason = exc.reason
        elif isinstance(exc, ArchiveError):
            reason = "quarantine-archive-failed"
        elif isinstance(exc, (MetadataError, PrefixError)):
            reason = "quarantine-observation-failed"
        elif isinstance(exc, OSError):
            reason = "quarantine-io"
        else:
            reason = "quarantine-interrupted"
        retained_identity = tree.created.get("") if tree is not None else None
        raise QuarantineError(reason, retained, retained_identity) from exc

"""Bounded, unqualified observations of stored xattr names on one Linux inode.

No values, qualification, approval, extraction, or privilege changes. Name probes
address a leaf beneath a held parent, not an atomic inode snapshot. Identity
brackets detect observed drift; they cannot exclude hostile same-UID writers.
"""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import stat
from typing import Literal

from pydantic import BaseModel, ConfigDict

from strixlab.serialization import canonical_json_bytes

_MAX_NAME_BYTES = 65536
_AT_SYMLINK_NOFOLLOW = 0x100
# Linux native x86_64 ABI: arch/x86/entry/syscalls/syscall_64.tbl, Linux 6.13+.
# listxattrat(int dfd, const char *path, unsigned int flags, char *list, size_t size).
_SYS_LISTXATTRAT = 465
_LIBC = ctypes.CDLL(None, use_errno=True)
type _ListStatus = Literal["observed", "error", "unsupported", "resource-limit", "malformed"]


class MetadataError(ValueError):
    """Invalid input or unstable observation; no completed report is returned."""


class InodeIdentityV1(BaseModel):
    """Stat evidence for observed drift, not a filesystem snapshot or approval."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    dev: int
    ino: int
    mode: int
    uid: int
    gid: int
    nlink: int
    size: int
    mtime_ns: int
    ctime_ns: int


def _identity(value: os.stat_result) -> InodeIdentityV1:
    return InodeIdentityV1(
        dev=value.st_dev,
        ino=value.st_ino,
        mode=value.st_mode,
        uid=value.st_uid,
        gid=value.st_gid,
        nlink=value.st_nlink,
        size=value.st_size,
        mtime_ns=value.st_mtime_ns,
        ctime_ns=value.st_ctime_ns,
    )


class InodeMetadataObservationV1(BaseModel):
    """A completed probe whose namespace coverage is ALWAYS unknown."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    observer_id: Literal["linux-x86_64-listxattrat-v1"] = "linux-x86_64-listxattrat-v1"
    validation: Literal["complete"] = "complete"
    coverage: Literal["unknown"] = "unknown"
    scope: Literal["stored-xattr-names-only"] = "stored-xattr-names-only"
    name_bytes_escaped: str
    kind: Literal["file", "directory", "symlink"]
    parent_before: InodeIdentityV1
    parent_after: InodeIdentityV1
    leaf_before: InodeIdentityV1
    leaf_opened: InodeIdentityV1
    leaf_after: InodeIdentityV1
    leaf_named_after: InodeIdentityV1
    list_status: _ListStatus
    list_errno: int | None
    name_list_size_bytes: int | None
    names_bytes_escaped: tuple[str, ...] | None

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def _escaped(value: bytes) -> str:
    return "".join(f"\\x{byte:02x}" for byte in value)


def _supported_abi() -> bool:
    return (
        platform.system() == "Linux"
        and platform.machine() == "x86_64"
        and ctypes.sizeof(ctypes.c_void_p) == 8
        and ctypes.sizeof(ctypes.c_long) == 8
        and ctypes.sizeof(ctypes.c_size_t) == 8
        and hasattr(os, "O_PATH")
        and hasattr(os, "O_NOFOLLOW")
        and hasattr(_LIBC, "syscall")
    )


def _listxattrat(parent_fd: int, name: bytes, size: int) -> tuple[int, bytes]:
    """One bounded name-only syscall. Caller has checked the native ABI."""

    buffer = ctypes.create_string_buffer(size) if size else None
    syscall = _LIBC.syscall
    syscall.restype = ctypes.c_long
    # syscall is variadic: explicitly type every argument, including size_t.
    result = int(
        syscall(
            ctypes.c_long(_SYS_LISTXATTRAT),
            ctypes.c_int(parent_fd),
            ctypes.c_char_p(name),
            ctypes.c_uint(_AT_SYMLINK_NOFOLLOW),
            ctypes.cast(buffer, ctypes.c_void_p) if buffer is not None else ctypes.c_void_p(),
            ctypes.c_size_t(size),
        )
    )
    if result == -1:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result, buffer.raw[:result] if buffer is not None and 0 <= result <= size else b""


def _names(
    parent_fd: int, name: bytes
) -> tuple[_ListStatus, int | None, int | None, tuple[str, ...] | None]:
    for attempt in range(2):
        try:
            size, _ = _listxattrat(parent_fd, name, 0)
            if size < 0:
                return "malformed", None, None, None
            if size > _MAX_NAME_BYTES:
                return "resource-limit", None, size, None
            if size == 0:
                return "observed", None, 0, ()
            used, data = _listxattrat(parent_fd, name, size)
        except OSError as exc:
            if exc.errno == errno.ERANGE and attempt == 0:
                continue
            if exc.errno in {errno.ENOSYS, errno.ENOTSUP}:
                return "unsupported", exc.errno, None, None
            if exc.errno == errno.E2BIG:
                return "resource-limit", exc.errno, None, None
            return "error", exc.errno, None, None
        if used < 0 or used > size or len(data) != used:
            return "malformed", None, None, None
        if used != size:
            # A size query followed by a shorter list is observed drift.
            raise MetadataError("metadata-names-changed")
        if not data.endswith(b"\0"):
            return "malformed", None, used, None
        names = data[:-1].split(b"\0")
        if any(not item for item in names) or len(names) != len(set(names)):
            return "malformed", None, used, None
        return "observed", None, used, tuple(_escaped(item) for item in sorted(names))
    raise AssertionError("bounded retry exhausted without result")  # pragma: no cover


def observe_inode_metadata(parent_fd: int, name: str) -> InodeMetadataObservationV1:
    """Observe one leaf without following it or claiming metadata absence.

    Caller holds a directory FD and provides a single UTF-8 leaf. Files,
    directories and symlinks are supported; special inodes are never opened for
    I/O. The parent FD is duplicated for the call. No owner/mode admission policy
    is applied. Syscall errors are recorded distinctly from empty observations;
    unsupported ABI and identity/I/O failures raise MetadataError.
    """

    if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name or "\0" in name:
        raise MetadataError("metadata-name")
    try:
        encoded = name.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise MetadataError("metadata-name-utf8") from exc
    if len(encoded) > 255:
        raise MetadataError("metadata-name-limit")
    if not _supported_abi():
        raise MetadataError("metadata-abi-unsupported")
    directory: int | None = None
    leaf: int | None = None
    try:
        directory = os.dup(parent_fd)
        parent_before = _identity(os.fstat(directory))
        if not stat.S_ISDIR(parent_before.mode):
            raise MetadataError("metadata-parent-not-directory")
        before = _identity(os.stat(encoded, dir_fd=directory, follow_symlinks=False))
        if stat.S_ISREG(before.mode):
            kind: Literal["file", "directory", "symlink"] = "file"
        elif stat.S_ISDIR(before.mode):
            kind = "directory"
        elif stat.S_ISLNK(before.mode):
            kind = "symlink"
        else:
            raise MetadataError("metadata-leaf-type")
        leaf = os.open(encoded, os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=directory)
        opened = _identity(os.fstat(leaf))
        if before != opened:
            raise MetadataError("metadata-leaf-changed")
        status, code, size, names = _names(directory, encoded)
        after = _identity(os.fstat(leaf))
        named_after = _identity(os.stat(encoded, dir_fd=directory, follow_symlinks=False))
        parent_after = _identity(os.fstat(directory))
        if opened != after or after != named_after:
            raise MetadataError("metadata-leaf-changed")
        if parent_before != parent_after:
            raise MetadataError("metadata-parent-changed")
        return InodeMetadataObservationV1(
            name_bytes_escaped=_escaped(encoded),
            kind=kind,
            parent_before=parent_before,
            parent_after=parent_after,
            leaf_before=before,
            leaf_opened=opened,
            leaf_after=after,
            leaf_named_after=named_after,
            list_status=status,
            list_errno=code,
            name_list_size_bytes=size,
            names_bytes_escaped=names,
        )
    except OSError as exc:
        raise MetadataError("metadata-io") from exc
    finally:
        if leaf is not None:
            os.close(leaf)
        if directory is not None:
            os.close(directory)

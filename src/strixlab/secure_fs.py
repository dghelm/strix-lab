"""Symlink-safe, crash-safe filesystem primitives for persisted state."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path

_EXCLUSIVE_CREATE_FLAGS = os.O_CLOEXEC | os.O_CREAT | os.O_EXCL | os.O_WRONLY
if hasattr(os, "O_NOFOLLOW"):
    _EXCLUSIVE_CREATE_FLAGS |= os.O_NOFOLLOW
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_LIBC = ctypes.CDLL(None, use_errno=True)
_RENAMEAT2 = getattr(_LIBC, "renameat2", None)
if _RENAMEAT2 is not None:
    _RENAMEAT2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    _RENAMEAT2.restype = ctypes.c_int


def exclusive_create_flags() -> int:
    """Open flags for an exclusive, no-follow, write-only file creation."""

    return _EXCLUSIVE_CREATE_FLAGS


def write_exclusive(path: Path, content: bytes, mode: int = 0o600) -> None:
    """Create ``path`` exclusively without following symlinks, then fsync.

    The file must not already exist. Every byte is written before the
    descriptor is flushed so a crash cannot publish a partial record.
    """

    descriptor = os.open(path, _EXCLUSIVE_CREATE_FLAGS, mode)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    """Flush a directory's entries to stable storage."""

    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one path without replacing an existing entry."""

    if _RENAMEAT2 is None:
        raise OSError(errno.ENOSYS, "renameat2 is required for no-replace publication")
    result = _RENAMEAT2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)

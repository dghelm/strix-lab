"""Symlink-safe, crash-safe filesystem primitives for persisted state."""

from __future__ import annotations

import ctypes
import errno
import os
import stat
from pathlib import Path

_EXCLUSIVE_CREATE_FLAGS = os.O_CLOEXEC | os.O_CREAT | os.O_EXCL | os.O_WRONLY
if hasattr(os, "O_NOFOLLOW"):
    _EXCLUSIVE_CREATE_FLAGS |= os.O_NOFOLLOW
_READONLY_OPEN_FLAGS = os.O_CLOEXEC | os.O_RDONLY
if hasattr(os, "O_NOFOLLOW"):
    _READONLY_OPEN_FLAGS |= os.O_NOFOLLOW
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


def readonly_open_flags() -> int:
    """Open flags for a no-follow, read-only file open."""

    return _READONLY_OPEN_FLAGS


def write_all(descriptor: int, content: bytes) -> None:
    """Write every byte of ``content`` to ``descriptor``, refusing short writes."""

    view = memoryview(content)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def write_exclusive(path: Path, content: bytes, mode: int = 0o600) -> None:
    """Create ``path`` exclusively without following symlinks, then fsync.

    The file must not already exist. Every byte is written before the
    descriptor is flushed so a crash cannot publish a partial record.
    """

    descriptor = os.open(path, _EXCLUSIVE_CREATE_FLAGS, mode)
    try:
        write_all(descriptor, content)
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


def ensure_directory_fsynced(path: Path, *, mode: int = 0o700) -> bool:
    """Exclusively create one directory when absent, fsyncing its parent on creation.

    Returns ``True`` when this call created the directory (its parent has been
    flushed so the new entry is durable) and ``False`` when an entry already
    existed. Creation is a single atomic ``mkdir``: a concurrent creator or any
    pre-existing entry (including a symlink) raises ``FileExistsError`` internally
    and yields ``False``, so the primitive never follows or replaces an existing
    entry. Validating an existing entry (ownership, type) is the caller's
    responsibility, since the safe/unsafe policy differs per caller.
    """

    try:
        os.mkdir(path, mode)
    except FileExistsError:
        return False
    fsync_directory(path.parent)
    return True


def fsync_tree(root: Path) -> None:
    """Durably flush every regular file and directory under ``root``, plus its parent.

    ``os.walk`` with ``followlinks=False`` never descends through a symlinked
    directory. Each regular file is opened relative to its directory descriptor
    with ``O_NOFOLLOW`` and validated as an owned regular file before being
    flushed, so a symlink or special file swapped in for a race fails closed
    rather than being followed. Files and then directories are flushed
    bottom-up, and finally the parent directory, so the tree is durable on
    return.
    """

    dir_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    if hasattr(os, "O_NOFOLLOW"):
        dir_flags |= os.O_NOFOLLOW
        file_flags |= os.O_NOFOLLOW
    euid = os.geteuid()
    directories: list[Path] = []
    for current, _dirs, files in os.walk(root, followlinks=False):
        directory = Path(current)
        directories.append(directory)
        dir_fd = os.open(directory, dir_flags)
        try:
            for name in files:
                # lstat the name, then open it nofollow and confirm the opened
                # descriptor is the very same regular inode: a symlink swapped in
                # fails O_NOFOLLOW, and a regular file swapped in between the
                # lstat and the open is caught by the device/inode comparison.
                pre = os.lstat(name, dir_fd=dir_fd)
                if not stat.S_ISREG(pre.st_mode) or pre.st_uid != euid:
                    raise OSError(f"unsafe file in fsync tree: {directory / name}")
                descriptor = os.open(name, file_flags, dir_fd=dir_fd)
                try:
                    post = os.fstat(descriptor)
                    if (
                        not stat.S_ISREG(post.st_mode)
                        or post.st_uid != euid
                        or post.st_dev != pre.st_dev
                        or post.st_ino != pre.st_ino
                    ):
                        raise OSError(f"unsafe file replaced during fsync: {directory / name}")
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        finally:
            os.close(dir_fd)
    for directory in reversed(directories):
        descriptor = os.open(directory, dir_flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    fsync_directory(root.parent)


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

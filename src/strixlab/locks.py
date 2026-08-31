"""Non-blocking, symlink-safe Linux file locks."""

from __future__ import annotations

import os
import stat
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class LockStatus(StrEnum):
    ACQUIRED = "acquired"
    CONTENDED = "contended"
    UNSAFE = "unsafe"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class LockAttempt:
    status: LockStatus
    path: Path
    reason: str | None = None

    @property
    def acquired(self) -> bool:
        return self.status is LockStatus.ACQUIRED


def _open_lock(path: Path) -> int:
    if sys.platform != "linux" or not hasattr(os, "O_NOFOLLOW"):
        raise NotImplementedError("secure Linux file locking is unavailable")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"lock parent does not exist: {path.parent}")
    flags = os.O_CLOEXEC | os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW
    return os.open(path, flags, 0o600)


def _validate_descriptor(fd: int) -> None:
    metadata = os.fstat(fd)
    if not stat.S_ISREG(metadata.st_mode):
        raise PermissionError("lock path is not a regular file")
    if metadata.st_uid != os.geteuid():
        raise PermissionError("lock file is owned by another user")
    mode = stat.S_IMODE(metadata.st_mode)
    if mode & ~0o600:
        raise PermissionError(f"lock file mode {mode:#o} is broader than 0o600")


@contextmanager
def exclusive_lock(path: Path) -> Iterator[LockAttempt]:
    """Attempt a safe exclusive lock without waiting.

    The lock file is deliberately retained after release so another inode cannot
    be substituted while cooperating processes still reference the path.
    """

    normalized = path.absolute()
    fd: int | None = None
    try:
        try:
            fd = _open_lock(normalized)
            _validate_descriptor(fd)
        except (FileNotFoundError, NotImplementedError, OSError) as exc:
            yield LockAttempt(LockStatus.UNAVAILABLE, normalized, str(exc))
            return

        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield LockAttempt(
                LockStatus.CONTENDED,
                normalized,
                "lock is held by another process",
            )
            return
        yield LockAttempt(LockStatus.ACQUIRED, normalized)
    finally:
        if fd is not None:
            os.close(fd)


@contextmanager
def wait_for_exclusive_lock(
    path: Path, *, timeout: float, poll_interval: float = 0.05
) -> Iterator[LockAttempt]:
    """Retry :func:`exclusive_lock` until acquired or a monotonic deadline expires.

    Only a *contended* lock is retried: an unsafe or unavailable lock (a missing
    parent, foreign owner, wrong type, or an unsupported platform) is reported
    immediately without spinning. On success the lock is held for the duration of the
    caller's ``with`` body exactly as :func:`exclusive_lock` holds it. On deadline
    expiry a ``CONTENDED`` attempt is yielded so the caller can fail closed without
    ever having hashed or mutated state.
    """

    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    deadline = time.monotonic() + timeout
    while True:
        with exclusive_lock(path) as attempt:
            if attempt.acquired or attempt.status is not LockStatus.CONTENDED:
                yield attempt
                return
        if time.monotonic() >= deadline:
            yield LockAttempt(LockStatus.CONTENDED, path.absolute(), "lock wait timed out")
            return
        time.sleep(poll_interval)

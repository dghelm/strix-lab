"""Shared executable-identity primitive for verified-binary callers.

Extracted verbatim from ADAPTER-001's binary integrity check: it stream-hashes a
non-symlink regular executable no-follow and asserts complete pre/post
descriptor-metadata stability, so a swapped, truncated, or mutated file cannot be
attested. Each adapter supplies its own integrity-exception factory and subject
noun, so this shared primitive never imposes a common exception type — adapter
specific exception translation is preserved. Repeated path-identity checks narrow
but do not eliminate the small check-to-exec race, because the child is launched by
path rather than descriptor.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from strixlab.secure_fs import readonly_open_flags

_READ_CHUNK_BYTES = 64 * 1024

# Each adapter injects its own integrity-exception constructor; both are RuntimeError
# subclasses, so this shared primitive never imposes a common exception type.
IntegrityErrorFactory = Callable[[str], RuntimeError]


@dataclass(frozen=True, slots=True)
class ExecutableIdentity:
    """Complete descriptor identity and content digest of one verified executable."""

    dev: int
    ino: int
    size: int
    mode: int
    mtime_ns: int
    ctime_ns: int
    sha256: str


def hash_executable(
    path: Path, *, error: IntegrityErrorFactory, subject: str
) -> ExecutableIdentity:
    """Stream-hash a non-symlink regular executable with pre/post metadata stability.

    ``error`` builds the adapter's own integrity exception from a message; ``subject``
    names the executable in that message. Metadata is captured no-follow from the open
    descriptor and compared before and after the read so a mid-hash swap fails closed.
    """

    try:
        descriptor = os.open(path, readonly_open_flags())
    except OSError as exc:
        raise error(f"{subject} is unavailable: {path}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise error(f"{subject} is not a regular file: {path}")
        if not before.st_mode & 0o111:
            raise error(f"{subject} is not executable: {path}")
        while chunk := os.read(descriptor, _READ_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mode != after.st_mode
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or size != before.st_size
        ):
            raise error(f"{subject} changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return ExecutableIdentity(
        dev=before.st_dev,
        ino=before.st_ino,
        size=size,
        mode=stat.S_IMODE(before.st_mode),
        mtime_ns=before.st_mtime_ns,
        ctime_ns=before.st_ctime_ns,
        sha256=digest.hexdigest(),
    )


def require_stable_executable(
    path: Path, expected: ExecutableIdentity, *, error: IntegrityErrorFactory, subject: str
) -> ExecutableIdentity:
    """Re-hash ``path`` and require it to equal ``expected`` or raise the adapter error."""

    current = hash_executable(path, error=error, subject=subject)
    if current != expected:
        raise error(f"{subject} drifted across the run window: {path}")
    return current

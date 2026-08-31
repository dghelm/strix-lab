"""Safe subprocess execution with structured, optionally bounded capture."""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import secrets
import signal
import stat
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, Protocol

from strixlab.naming import ENV_NAME_RE
from strixlab.secure_fs import exclusive_create_flags, fsync_directory
from strixlab.serialization import canonical_json_bytes

TERMINATION_GRACE_SECONDS = 2.0
_READ_CHUNK_BYTES = 64 * 1024
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class _Digest(Protocol):
    def update(self, value: bytes) -> None: ...

    def hexdigest(self) -> str: ...


class ProcessOutcome(StrEnum):
    """High-level process outcome independent of the return code."""

    EXITED = "exited"
    TIMED_OUT = "timed_out"
    SPAWN_FAILED = "spawn_failed"
    CAPTURE_FAILED = "capture_failed"


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Complete structured result from :func:`run_process`."""

    outcome: ProcessOutcome
    argv: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    started_at: float
    ended_at: float
    duration: float
    error: str | None
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_sha256: str = _EMPTY_SHA256
    stderr_sha256: str = _EMPTY_SHA256
    stdout_spool: Path | None = None
    stderr_spool: Path | None = None
    capture_error: str | None = None


def _validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)) or not argv:
        raise ValueError("argv must be a non-empty sequence of strings")
    normalized = tuple(argv)
    if any(not isinstance(value, str) for value in normalized):
        raise ValueError("argv entries must be strings")
    if not normalized[0]:
        raise ValueError("argv[0] must not be empty")
    if any("\x00" in value for value in normalized):
        raise ValueError("argv entries must not contain NUL bytes")
    return normalized


def _validate_timeout(timeout: float | None) -> None:
    if timeout is None:
        return
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a finite positive number or None")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number or None")


def _validate_byte_limit(value: int | None, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a nonnegative integer or None")
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer or None")


def _validate_pass_fds(pass_fds: Sequence[int]) -> tuple[int, ...]:
    if isinstance(pass_fds, (str, bytes)):
        raise TypeError("pass_fds must be a sequence of file descriptors")
    normalized = tuple(pass_fds)
    for descriptor in normalized:
        if isinstance(descriptor, bool) or not isinstance(descriptor, int) or descriptor < 0:
            raise ValueError("pass_fds entries must be nonnegative integers")
    return normalized


def _prepare_environment(
    overrides: Mapping[str, str | None] | None,
    *,
    inherit: bool,
    base: Mapping[str, str] | None,
) -> dict[str, str]:
    # An explicit ``base`` is always the baseline; ``inherit`` selects ``os.environ`` only
    # when no base is given, so a caller can supply a complete constructed environment
    # under ``inherit_env=False``. ``overrides`` still overlays or removes on top.
    if base is not None:
        environment = dict(base)
    elif inherit:
        environment = dict(os.environ)
    else:
        environment = {}
    if overrides is None:
        return environment
    for name, value in overrides.items():
        if not isinstance(name, str) or not ENV_NAME_RE.fullmatch(name):
            raise ValueError(f"invalid environment variable name: {name!r}")
        if value is not None and not isinstance(value, str):
            raise ValueError(f"environment variable {name!r} must be a string or None")
        if value is not None and "\x00" in value:
            raise ValueError(f"environment variable {name!r} must not contain NUL bytes")
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    return environment


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    while time.monotonic() < deadline:
        if not _process_group_exists(process.pid):
            break
        time.sleep(0.01)
    if _process_group_exists(process.pid):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=TERMINATION_GRACE_SECONDS)


@dataclass(slots=True)
class _Spool:
    final_path: Path
    temporary_path: Path | None = None
    stream: BinaryIO | None = None
    published_path: Path | None = None
    error: str | None = None


@dataclass(slots=True)
class _Capture:
    data: bytearray = field(default_factory=bytearray)
    digest: _Digest = field(default_factory=hashlib.sha256)
    byte_count: int = 0
    truncated: bool = False
    spool: _Spool | None = None
    error: str | None = None


def _validate_spool_directory(path: Path) -> None:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"spool directory is unsafe: {path}")
    if metadata.st_uid != os.geteuid():
        raise PermissionError(f"spool directory is owned by another user: {path}")


def _open_spool(path: Path, spool_root: Path) -> _Spool:
    root = spool_root.absolute()
    normalized = path.absolute()
    _validate_spool_directory(root)
    try:
        relative = normalized.relative_to(root)
    except ValueError as exc:
        raise ValueError("spool path must remain beneath spool_root") from exc
    if not relative.parts:
        raise ValueError("spool path must name a file beneath spool_root")
    parent = root
    for part in relative.parts[:-1]:
        parent /= part
        _validate_spool_directory(parent)
    if normalized.exists() or normalized.is_symlink():
        raise FileExistsError(normalized)
    spool = _Spool(final_path=normalized)
    temporary = normalized.parent / f".{normalized.name}.tmp-{secrets.token_hex(12)}"
    try:
        descriptor = os.open(temporary, exclusive_create_flags(), 0o600)
    except OSError:
        spool.error = "spool-open-failed"
        return spool
    spool.temporary_path = temporary
    spool.stream = os.fdopen(descriptor, "wb")
    return spool


def _discard_spool_temporary(spool: _Spool) -> None:
    if spool.temporary_path is not None:
        with contextlib.suppress(OSError):
            spool.temporary_path.unlink()
        spool.temporary_path = None


def _abort_spool(spool: _Spool | None) -> None:
    if spool is None:
        return
    if spool.stream is not None:
        with contextlib.suppress(OSError):
            spool.stream.close()
        spool.stream = None
    _discard_spool_temporary(spool)


def _publish_spool(spool: _Spool | None) -> None:
    if spool is None:
        return
    if spool.stream is not None:
        try:
            if spool.error is None:
                spool.stream.flush()
                os.fsync(spool.stream.fileno())
        except OSError:
            spool.error = spool.error or "spool-fsync-failed"
        try:
            spool.stream.close()
        except OSError:
            spool.error = spool.error or "spool-close-failed"
        spool.stream = None
    if spool.error is not None or spool.temporary_path is None:
        _discard_spool_temporary(spool)
        return
    linked = False
    try:
        os.link(spool.temporary_path, spool.final_path, follow_symlinks=False)
        linked = True
        spool.temporary_path.unlink()
        fsync_directory(spool.final_path.parent)
        spool.published_path = spool.final_path
    except OSError:
        spool.error = "spool-publish-failed"
        if linked:
            with contextlib.suppress(OSError):
                spool.final_path.unlink()
        _discard_spool_temporary(spool)


def _drain(stream: BinaryIO, capture: _Capture, limit: int | None, hard_limit: int | None) -> None:
    try:
        while chunk := stream.read(_READ_CHUNK_BYTES):
            capture.byte_count += len(chunk)
            capture.digest.update(chunk)
            spool = capture.spool
            if spool is not None and spool.stream is not None and spool.error is None:
                try:
                    written = spool.stream.write(chunk)
                    if written != len(chunk):
                        spool.error = "spool-short-write"
                except OSError:
                    spool.error = "spool-write-failed"
            if limit is None:
                capture.data.extend(chunk)
            else:
                remaining = max(0, limit - len(capture.data))
                capture.data.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    capture.truncated = True
            # A hard total-byte ceiling aborts the stream once the complete chunk that
            # crossed it has been counted and hashed. The partial spool is discarded so
            # a truncated capture can never be published, and the surfaced capture error
            # drives the poll loop to terminate the process group as CAPTURE_FAILED.
            if hard_limit is not None and capture.byte_count > hard_limit:
                capture.error = capture.error or "hard-limit-exceeded"
                _abort_spool(capture.spool)
                return
    except OSError:
        capture.error = "stream-drain-failed"


def _capture_error(stdout: _Capture, stderr: _Capture) -> str | None:
    errors = []
    if stdout.error is not None:
        errors.append(f"stdout:{stdout.error}")
    if stdout.spool is not None and stdout.spool.error is not None:
        errors.append(f"stdout:{stdout.spool.error}")
    if stderr.error is not None:
        errors.append(f"stderr:{stderr.error}")
    if stderr.spool is not None and stderr.spool.error is not None:
        errors.append(f"stderr:{stderr.spool.error}")
    return ";".join(errors) or None


def run_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout: float | None = None,
    env_overrides: Mapping[str, str | None] | None = None,
    inherit_env: bool = True,
    base_env: Mapping[str, str] | None = None,
    output_limit_bytes: int | None = None,
    stdout_total_limit_bytes: int | None = None,
    stderr_total_limit_bytes: int | None = None,
    stdout_spool: Path | None = None,
    stderr_spool: Path | None = None,
    spool_root: Path | None = None,
    pass_fds: Sequence[int] = (),
) -> ProcessResult:
    """Run a command without a shell and return a structured result.

    Each output stream is always drained, counted, and hashed over its exact
    bytes. ``output_limit_bytes`` bounds only the decoded in-memory prefix.
    ``stdout_total_limit_bytes`` and ``stderr_total_limit_bytes`` are optional hard
    ceilings on total bytes a stream may emit: the complete chunk that crosses one is
    still counted and hashed, then the process group is terminated and the outcome is
    ``CAPTURE_FAILED`` with ``capture_error`` ``stdout:hard-limit-exceeded`` or
    ``stderr:hard-limit-exceeded``; any partial spool is aborted rather than published.
    Optional spool paths preserve full byte streams through atomic publication
    beneath an explicit, owned ``spool_root``. ``pass_fds`` names extra descriptors the
    child inherits (e.g. a pre-opened model handle exposed as ``/proc/self/fd/<fd>``).
    """

    normalized_argv = _validate_argv(argv)
    _validate_timeout(timeout)
    _validate_byte_limit(output_limit_bytes, "output_limit_bytes")
    _validate_byte_limit(stdout_total_limit_bytes, "stdout_total_limit_bytes")
    _validate_byte_limit(stderr_total_limit_bytes, "stderr_total_limit_bytes")
    normalized_pass_fds = _validate_pass_fds(pass_fds)
    if not cwd.exists():
        raise FileNotFoundError(cwd)
    if not cwd.is_dir():
        raise NotADirectoryError(cwd)
    if (stdout_spool is not None or stderr_spool is not None) and spool_root is None:
        raise ValueError("spool_root is required when a spool path is requested")
    if (
        stdout_spool is not None
        and stderr_spool is not None
        and stdout_spool.absolute() == stderr_spool.absolute()
    ):
        raise ValueError("stdout_spool and stderr_spool must be different paths")
    environment = _prepare_environment(env_overrides, inherit=inherit_env, base=base_env)

    stdout_target: _Spool | None = None
    try:
        stdout_target = (
            None if stdout_spool is None else _open_spool(stdout_spool, spool_root)  # type: ignore[arg-type]
        )
        stderr_target = (
            None if stderr_spool is None else _open_spool(stderr_spool, spool_root)  # type: ignore[arg-type]
        )
    except BaseException:
        _abort_spool(stdout_target)
        raise
    stdout_capture = _Capture(spool=stdout_target)
    stderr_capture = _Capture(spool=stderr_target)

    started_at = time.monotonic()
    try:
        process = subprocess.Popen(
            normalized_argv,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            pass_fds=normalized_pass_fds,
        )
    except OSError as exc:
        _publish_spool(stdout_target)
        _publish_spool(stderr_target)
        ended_at = time.monotonic()
        return ProcessResult(
            outcome=ProcessOutcome.SPAWN_FAILED,
            argv=normalized_argv,
            returncode=None,
            stdout="",
            stderr="",
            started_at=started_at,
            ended_at=ended_at,
            duration=ended_at - started_at,
            error=f"{type(exc).__name__}: {exc}",
            stdout_spool=None if stdout_target is None else stdout_target.published_path,
            stderr_spool=None if stderr_target is None else stderr_target.published_path,
            capture_error=_capture_error(stdout_capture, stderr_capture),
        )

    assert process.stdout is not None
    assert process.stderr is not None
    threads = (
        threading.Thread(
            target=_drain,
            args=(process.stdout, stdout_capture, output_limit_bytes, stdout_total_limit_bytes),
            daemon=True,
        ),
        threading.Thread(
            target=_drain,
            args=(process.stderr, stderr_capture, output_limit_bytes, stderr_total_limit_bytes),
            daemon=True,
        ),
    )
    for thread in threads:
        thread.start()

    outcome = ProcessOutcome.EXITED
    deadline = None if timeout is None else time.monotonic() + timeout
    try:
        while process.poll() is None:
            if _capture_error(stdout_capture, stderr_capture) is not None:
                outcome = ProcessOutcome.CAPTURE_FAILED
                _terminate_process_group(process)
                break
            if deadline is not None and time.monotonic() >= deadline:
                outcome = ProcessOutcome.TIMED_OUT
                _terminate_process_group(process)
                break
            time.sleep(0.01)
        if process.poll() is None:
            process.wait()
    finally:
        for thread in threads:
            thread.join()
        process.stdout.close()
        process.stderr.close()
        # A child can exceed a hard ceiling and exit before the poll loop observes the
        # drain error (e.g. a fast burst then immediate exit). The drain threads have now
        # joined, so promote a still-EXITED outcome to CAPTURE_FAILED when a *drain*
        # capture error exists. Spool-publication errors are raised below and keep the
        # existing late-failure semantics (outcome stays EXITED with capture_error set).
        if outcome is ProcessOutcome.EXITED and (
            stdout_capture.error is not None or stderr_capture.error is not None
        ):
            outcome = ProcessOutcome.CAPTURE_FAILED
        _publish_spool(stdout_target)
        _publish_spool(stderr_target)

    ended_at = time.monotonic()
    return ProcessResult(
        outcome=outcome,
        argv=normalized_argv,
        returncode=process.returncode,
        stdout=bytes(stdout_capture.data).decode("utf-8", errors="replace"),
        stderr=bytes(stderr_capture.data).decode("utf-8", errors="replace"),
        started_at=started_at,
        ended_at=ended_at,
        duration=ended_at - started_at,
        error=None,
        stdout_truncated=stdout_capture.truncated,
        stderr_truncated=stderr_capture.truncated,
        stdout_bytes=stdout_capture.byte_count,
        stderr_bytes=stderr_capture.byte_count,
        stdout_sha256=stdout_capture.digest.hexdigest(),
        stderr_sha256=stderr_capture.digest.hexdigest(),
        stdout_spool=None if stdout_target is None else stdout_target.published_path,
        stderr_spool=None if stderr_target is None else stderr_target.published_path,
        capture_error=_capture_error(stdout_capture, stderr_capture),
    )


def process_result_digest(result: ProcessResult) -> str:
    """Return a stable digest of a process result, including its capture error."""

    return hashlib.sha256(
        canonical_json_bytes(
            {
                "argv": result.argv,
                "capture_error": result.capture_error,
                "error": result.error,
                "outcome": result.outcome,
                "returncode": result.returncode,
                "stderr_sha256": result.stderr_sha256,
                "stdout_sha256": result.stdout_sha256,
            }
        )
    ).hexdigest()

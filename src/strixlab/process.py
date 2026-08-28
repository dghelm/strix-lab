"""Safe subprocess execution with structured, optionally bounded capture."""

from __future__ import annotations

import contextlib
import math
import os
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO

from strixlab.naming import ENV_NAME_RE

TERMINATION_GRACE_SECONDS = 2.0
_READ_CHUNK_BYTES = 64 * 1024


class ProcessOutcome(StrEnum):
    """High-level process outcome independent of the return code."""

    EXITED = "exited"
    TIMED_OUT = "timed_out"
    SPAWN_FAILED = "spawn_failed"


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


def _validate_output_limit(output_limit_bytes: int | None) -> None:
    if output_limit_bytes is None:
        return
    if isinstance(output_limit_bytes, bool) or not isinstance(output_limit_bytes, int):
        raise TypeError("output_limit_bytes must be a nonnegative integer or None")
    if output_limit_bytes < 0:
        raise ValueError("output_limit_bytes must be a nonnegative integer or None")


def _prepare_environment(
    overrides: Mapping[str, str | None] | None,
    *,
    inherit: bool,
    base: Mapping[str, str] | None,
) -> dict[str, str]:
    environment = dict(os.environ if base is None else base) if inherit else {}
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
class _Capture:
    data: bytearray
    truncated: bool = False


def _drain(stream: BinaryIO, capture: _Capture, limit: int | None) -> None:
    while chunk := stream.read(_READ_CHUNK_BYTES):
        if limit is None:
            capture.data.extend(chunk)
            continue
        remaining = max(0, limit - len(capture.data))
        capture.data.extend(chunk[:remaining])
        if len(chunk) > remaining:
            capture.truncated = True


def run_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout: float | None = None,
    env_overrides: Mapping[str, str | None] | None = None,
    inherit_env: bool = True,
    base_env: Mapping[str, str] | None = None,
    output_limit_bytes: int | None = None,
) -> ProcessResult:
    """Run a command without a shell and return a structured result.

    When ``output_limit_bytes`` is set, each output stream is drained fully while
    only its leading bytes are retained. This prevents deadlocks and unbounded
    memory use without changing the historical unbounded default.
    """

    normalized_argv = _validate_argv(argv)
    _validate_timeout(timeout)
    _validate_output_limit(output_limit_bytes)
    if not cwd.exists():
        raise FileNotFoundError(cwd)
    if not cwd.is_dir():
        raise NotADirectoryError(cwd)
    environment = _prepare_environment(env_overrides, inherit=inherit_env, base=base_env)
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
        )
    except OSError as exc:
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
        )

    assert process.stdout is not None
    assert process.stderr is not None
    stdout_capture = _Capture(bytearray())
    stderr_capture = _Capture(bytearray())
    threads = (
        threading.Thread(
            target=_drain,
            args=(process.stdout, stdout_capture, output_limit_bytes),
            daemon=True,
        ),
        threading.Thread(
            target=_drain,
            args=(process.stderr, stderr_capture, output_limit_bytes),
            daemon=True,
        ),
    )
    for thread in threads:
        thread.start()

    outcome = ProcessOutcome.EXITED
    error: str | None = None
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        outcome = ProcessOutcome.TIMED_OUT
        _terminate_process_group(process)
    finally:
        for thread in threads:
            thread.join()
        process.stdout.close()
        process.stderr.close()

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
        error=error,
        stdout_truncated=stdout_capture.truncated,
        stderr_truncated=stderr_capture.truncated,
    )

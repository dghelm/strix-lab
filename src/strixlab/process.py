"""Deterministic Linux child-process execution."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from strixlab.naming import ENV_NAME_RE

TERMINATION_GRACE_SECONDS = 2.0


class ProcessOutcome(StrEnum):
    EXITED = "exited"
    TIMED_OUT = "timed_out"
    SPAWN_FAILED = "spawn_failed"


@dataclass(frozen=True, slots=True)
class ProcessResult:
    outcome: ProcessOutcome
    argv: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    started_at: float
    ended_at: float
    duration: float
    error: str | None


def _validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)) or not argv:
        raise ValueError("argv must be a nonempty sequence of strings")
    result = tuple(argv)
    if any(not isinstance(value, str) for value in result):
        raise ValueError("argv entries must be strings")
    if not result[0]:
        raise ValueError("argv[0] cannot be empty")
    if any("\x00" in value for value in result):
        raise ValueError("argv entries cannot contain NUL bytes")
    return result


def _validate_timeout(timeout: float | None) -> None:
    if timeout is None:
        return
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a positive finite number or None")
    if timeout <= 0 or not math.isfinite(timeout):
        raise ValueError("timeout must be a positive finite number or None")


def _prepare_environment(
    overrides: Mapping[str, str | None] | None,
    *,
    inherit: bool,
) -> dict[str, str] | None:
    if overrides is None:
        return None if inherit else {}
    environment = dict(os.environ) if inherit else {}
    for key, value in overrides.items():
        if not isinstance(key, str) or ENV_NAME_RE.fullmatch(key) is None:
            raise ValueError(f"invalid environment override key: {key!r}")
        if value is not None and not isinstance(value, str):
            raise ValueError(f"environment override must be a string or None: {key}")
        if value is not None and "\x00" in value:
            raise ValueError("environment overrides cannot contain NUL bytes")
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
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
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + TERMINATION_GRACE_SECONDS
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_exists(process_group):
            return
        time.sleep(0.02)

    with suppress(ProcessLookupError):
        os.killpg(process_group, signal.SIGKILL)


def run_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout: float | None = None,
    env_overrides: Mapping[str, str | None] | None = None,
    inherit_env: bool = True,
) -> ProcessResult:
    """Run a non-interactive process and retain every ordinary failure as data."""

    arguments = _validate_argv(argv)
    _validate_timeout(timeout)
    working_directory = Path(cwd)
    if not working_directory.exists():
        raise FileNotFoundError(working_directory)
    if not working_directory.is_dir():
        raise NotADirectoryError(working_directory)
    environment = _prepare_environment(env_overrides, inherit=inherit_env)

    started_at = time.monotonic()
    try:
        process = subprocess.Popen(
            arguments,
            cwd=working_directory,
            env=environment,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        ended_at = time.monotonic()
        return ProcessResult(
            outcome=ProcessOutcome.SPAWN_FAILED,
            argv=arguments,
            returncode=None,
            stdout="",
            stderr="",
            started_at=started_at,
            ended_at=ended_at,
            duration=ended_at - started_at,
            error=f"{type(exc).__name__}: {exc}",
        )

    outcome = ProcessOutcome.EXITED
    try:
        stdout_bytes, stderr_bytes = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        outcome = ProcessOutcome.TIMED_OUT
        _terminate_process_group(process)
        # communicate() may be safely retried after TimeoutExpired. Its final
        # result is authoritative and already contains earlier captured bytes.
        stdout_bytes, stderr_bytes = process.communicate()

    ended_at = time.monotonic()
    return ProcessResult(
        outcome=outcome,
        argv=arguments,
        returncode=process.returncode,
        stdout=stdout_bytes.decode("utf-8", errors="replace"),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
        started_at=started_at,
        ended_at=ended_at,
        duration=ended_at - started_at,
        error=None,
    )

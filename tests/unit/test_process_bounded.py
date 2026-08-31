from __future__ import annotations

import sys
from pathlib import Path

import pytest

from strixlab.process import ProcessOutcome, run_process


def command(source: str) -> list[str]:
    return [sys.executable, "-c", source]


def test_bounded_capture_drains_both_streams(tmp_path: Path) -> None:
    source = (
        "import sys; "
        "sys.stdout.write('a' * 200000); sys.stdout.flush(); "
        "sys.stderr.write('b' * 200000); sys.stderr.flush()"
    )
    result = run_process(command(source), cwd=tmp_path, output_limit_bytes=1024)

    assert result.outcome is ProcessOutcome.EXITED
    assert result.returncode == 0
    assert result.stdout == "a" * 1024
    assert result.stderr == "b" * 1024
    assert result.stdout_truncated is True
    assert result.stderr_truncated is True


def test_zero_limit_and_frozen_base_environment(tmp_path: Path) -> None:
    result = run_process(
        command("import os; print(os.environ['ONLY'])"),
        cwd=tmp_path,
        inherit_env=True,
        base_env={"ONLY": "frozen"},
        output_limit_bytes=0,
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert result.stdout_truncated is True


def test_utf8_cut_is_replaced(tmp_path: Path) -> None:
    result = run_process(
        command("import sys; sys.stdout.buffer.write('é'.encode())"),
        cwd=tmp_path,
        output_limit_bytes=1,
    )

    assert result.stdout == "�"
    assert result.stdout_truncated is True


@pytest.mark.parametrize("limit", [-1, True, 1.5, "1"])
def test_invalid_output_limit(limit: object, tmp_path: Path) -> None:
    with pytest.raises((TypeError, ValueError), match="output_limit_bytes"):
        run_process(
            command("pass"),
            cwd=tmp_path,
            output_limit_bytes=limit,  # type: ignore[arg-type]
        )


def test_pass_fds_hands_a_descriptor_to_the_child(tmp_path: Path) -> None:
    import os

    payload = b"receipt-bound-bytes"
    target = tmp_path / "model.bin"
    target.write_bytes(payload)
    descriptor = os.open(target, os.O_RDONLY | os.O_CLOEXEC)
    try:
        source = "import sys; sys.stdout.buffer.write(open(sys.argv[1], 'rb').read())"
        result = run_process(
            [sys.executable, "-c", source, f"/proc/self/fd/{descriptor}"],
            cwd=tmp_path,
            pass_fds=(descriptor,),
        )
    finally:
        os.close(descriptor)
    assert result.outcome is ProcessOutcome.EXITED
    assert result.stdout.encode() == payload


def test_stdout_hard_limit_terminates_and_reports_capture_failed(tmp_path: Path) -> None:
    source = (
        "import sys, time\nsys.stdout.write('x' * 100000)\nsys.stdout.flush()\ntime.sleep(30)\n"
    )
    result = run_process(
        command(source),
        cwd=tmp_path,
        output_limit_bytes=16,
        stdout_total_limit_bytes=1024,
    )
    assert result.outcome is ProcessOutcome.CAPTURE_FAILED
    assert result.capture_error == "stdout:hard-limit-exceeded"
    # The complete chunk that crossed the ceiling is still counted and hashed.
    assert result.stdout_bytes >= 1024
    assert result.stdout_spool is None


def test_stderr_hard_limit_reports_stderr_channel(tmp_path: Path) -> None:
    source = (
        "import sys, time\nsys.stderr.write('y' * 100000)\nsys.stderr.flush()\ntime.sleep(30)\n"
    )
    result = run_process(
        command(source),
        cwd=tmp_path,
        stderr_total_limit_bytes=1024,
    )
    assert result.outcome is ProcessOutcome.CAPTURE_FAILED
    assert result.capture_error == "stderr:hard-limit-exceeded"


def test_hard_limit_not_exceeded_is_normal(tmp_path: Path) -> None:
    result = run_process(
        command("import sys; sys.stdout.write('z' * 100)"),
        cwd=tmp_path,
        stdout_total_limit_bytes=1024,
        stderr_total_limit_bytes=1024,
    )
    assert result.outcome is ProcessOutcome.EXITED
    assert result.stdout == "z" * 100
    assert result.capture_error is None


def test_hard_limit_aborts_partial_spool(tmp_path: Path) -> None:
    spool = tmp_path / "spool.bin"
    source = (
        "import sys, time\nsys.stdout.write('x' * 100000)\nsys.stdout.flush()\ntime.sleep(30)\n"
    )
    result = run_process(
        command(source),
        cwd=tmp_path,
        stdout_total_limit_bytes=1024,
        stdout_spool=spool,
        spool_root=tmp_path,
    )
    assert result.outcome is ProcessOutcome.CAPTURE_FAILED
    assert result.capture_error == "stdout:hard-limit-exceeded"
    assert result.stdout_spool is None
    assert not spool.exists()


@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_invalid_hard_limit_is_rejected(tmp_path: Path, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="stdout_total_limit_bytes"):
        run_process(
            command("pass"),
            cwd=tmp_path,
            stdout_total_limit_bytes=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [(-1,), (True,), ("x",)])
def test_invalid_pass_fds_is_rejected(tmp_path: Path, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="pass_fds"):
        run_process(command("pass"), cwd=tmp_path, pass_fds=value)  # type: ignore[arg-type]


def test_fast_exit_hard_limit_is_capture_failed(tmp_path: Path) -> None:
    # A small burst (under the pipe buffer) followed by an immediate exit: the drain
    # flags the hard limit only after the child has already exited, so the poll loop
    # never observes it. The post-join promotion must still surface CAPTURE_FAILED.
    source = "import sys\nsys.stdout.write('x' * 4096)\nsys.stdout.flush()\n"
    result = run_process(command(source), cwd=tmp_path, stdout_total_limit_bytes=64)
    assert result.outcome is ProcessOutcome.CAPTURE_FAILED
    assert result.capture_error == "stdout:hard-limit-exceeded"
    assert result.stdout_bytes >= 64

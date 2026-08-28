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

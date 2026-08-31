from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from strixlab.process import ProcessOutcome, run_process


def python_command(source: str, *arguments: str) -> list[str]:
    return [sys.executable, "-c", source, *arguments]


def test_success_and_empty_later_argument(tmp_path: Path) -> None:
    result = run_process(
        python_command("import sys; print(repr(sys.argv[1]))", ""),
        cwd=tmp_path,
    )

    assert result.outcome is ProcessOutcome.EXITED
    assert result.returncode == 0
    assert result.stdout == "''\n"
    assert result.stderr == ""
    assert result.error is None
    assert result.duration >= 0
    assert result.ended_at >= result.started_at


def test_nonzero_exit_and_output_are_data(tmp_path: Path) -> None:
    result = run_process(
        python_command(
            "import sys; print('stdout'); print('stderr', file=sys.stderr); raise SystemExit(7)"
        ),
        cwd=tmp_path,
    )

    assert result.outcome is ProcessOutcome.EXITED
    assert result.returncode == 7
    assert result.stdout == "stdout\n"
    assert result.stderr == "stderr\n"


def test_output_decodes_invalid_utf8_with_replacement(tmp_path: Path) -> None:
    result = run_process(
        python_command("import os; os.write(1, b'bad: \\xff\\n')"),
        cwd=tmp_path,
    )
    assert result.stdout == "bad: �\n"


def test_spawn_failure_is_structured(tmp_path: Path) -> None:
    result = run_process(["/definitely/not/a/command"], cwd=tmp_path)

    assert result.outcome is ProcessOutcome.SPAWN_FAILED
    assert result.returncode is None
    assert result.stdout == result.stderr == ""
    assert result.error is not None and "FileNotFoundError" in result.error


def test_stdin_is_devnull(tmp_path: Path) -> None:
    result = run_process(python_command("import sys; print(repr(sys.stdin.read()))"), cwd=tmp_path)
    assert result.stdout == "''\n"


def test_environment_overlay_removal_and_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRIXLAB_INHERITED", "inherited")
    monkeypatch.setenv("WEIRD.KEY", "allowed-when-inherited")
    source = (
        "import json, os; print(json.dumps({key: os.environ.get(key) for key in "
        "['STRIXLAB_INHERITED', 'STRIXLAB_ADDED']}))"
    )

    overlay = run_process(
        python_command(source),
        cwd=tmp_path,
        env_overrides={"STRIXLAB_INHERITED": None, "STRIXLAB_ADDED": "added"},
    )
    isolated = run_process(
        python_command(source),
        cwd=tmp_path,
        inherit_env=False,
        env_overrides={"STRIXLAB_ADDED": "isolated"},
    )

    assert json.loads(overlay.stdout) == {
        "STRIXLAB_INHERITED": None,
        "STRIXLAB_ADDED": "added",
    }
    assert json.loads(isolated.stdout) == {
        "STRIXLAB_INHERITED": None,
        "STRIXLAB_ADDED": "isolated",
    }


@pytest.mark.parametrize(
    "argv",
    [[], [""], ["bad\x00command"], [sys.executable, 3]],
)
def test_invalid_argv_is_rejected(argv: list[object], tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_process(argv, cwd=tmp_path)  # type: ignore[arg-type]


@pytest.mark.parametrize("timeout", [0, -1, math.inf, math.nan, True, "one"])
def test_invalid_timeout_is_rejected(timeout: object, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="timeout"):
        run_process([sys.executable], cwd=tmp_path, timeout=timeout)  # type: ignore[arg-type]


def test_cwd_must_be_an_existing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_process([sys.executable], cwd=tmp_path / "missing")
    file_path = tmp_path / "file"
    file_path.write_text("not a directory", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        run_process([sys.executable], cwd=file_path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"BAD.KEY": "value"},
        {"GOOD_KEY": 3},
        {"GOOD_KEY": "bad\x00value"},
    ],
)
def test_invalid_environment_overrides_are_rejected(
    overrides: dict[str, object], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="environment"):
        run_process(
            [sys.executable],
            cwd=tmp_path,
            env_overrides=overrides,  # type: ignore[arg-type]
        )


def test_timeout_preserves_partial_output_exactly_once(tmp_path: Path) -> None:
    result = run_process(
        python_command("import time; print('once', flush=True); time.sleep(30)"),
        cwd=tmp_path,
        timeout=0.05,
    )

    assert result.outcome is ProcessOutcome.TIMED_OUT
    assert result.returncode == -15
    assert result.stdout == "once\n"
    assert result.error is None


def test_timeout_kills_descendant_that_ignores_sigterm(tmp_path: Path) -> None:
    source = """
import subprocess
import sys
import time

child = subprocess.Popen([
    sys.executable,
    "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
])
print(f"child={child.pid}", flush=True)
time.sleep(30)
"""

    result = run_process(python_command(source), cwd=tmp_path, timeout=0.05)

    assert result.outcome is ProcessOutcome.TIMED_OUT
    assert result.returncode == -15
    assert result.stdout.count("child=") == 1
    assert result.duration >= 2.0
    assert result.duration < 5.0


def test_explicit_base_env_is_baseline_without_inheritance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An explicit base_env under inherit_env=False is the child's environment baseline
    # (regression: it was previously dropped), and env_overrides still overlay/remove.
    monkeypatch.setenv("STRIXLAB_LEAKED", "must-not-appear")
    source = "import json, os; print(json.dumps(dict(os.environ)))"
    result = run_process(
        python_command(source),
        cwd=tmp_path,
        inherit_env=False,
        base_env={"LANG": "C", "KEEP": "kept", "DROP": "gone"},
        env_overrides={"DROP": None, "ADD": "added"},
    )
    env = json.loads(result.stdout)
    # The base_env is delivered (KEEP), overrides overlay (ADD) and remove (DROP), and
    # nothing is inherited (LEAKED absent). LC_CTYPE is added by CPython locale coercion.
    assert env["LANG"] == "C"
    assert env["KEEP"] == "kept"
    assert env["ADD"] == "added"
    assert "DROP" not in env
    assert "STRIXLAB_LEAKED" not in env

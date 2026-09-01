from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import strixlab.cli as cli
from strixlab.capsule_runs import CapsuleExecutionError, CapsuleRunError
from strixlab.evidence import RunOutcome

runner = CliRunner()
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _plain_terminal_text(value: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", value)


def _write_manifests(tmp_path: Path) -> tuple[Path, Path]:
    capsule = tmp_path / "capsule.yaml"
    capsule.write_text(
        """\
schema_version: 1
id: topk-capsule
candidate: candidate-a
machine: strix-halo-128g
build:
  source_id: topk-source
  source_commit: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
  toolchain_mode: host
  gfx_target: gfx1151
  target: topk-capsule
contract:
  protocol: native-capsule-v1
  scenario_sha256: bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
timeouts:
  describe_seconds: 1.0
  correctness_seconds: 1.0
  benchmark_seconds: 1.0
""",
        encoding="utf-8",
    )
    machine = tmp_path / "machine.yaml"
    machine.write_text(
        f"""\
schema_version: 1
id: strix-halo-128g
expect:
  gpu_arch: gfx1151
  integrated_gpu: true
  memory_gib_min: 64
exclusive_lock:
  path: {tmp_path / "machine.lock"}
telemetry:
  amd_smi: disabled
  sample_interval_ms: 100
validity:
  require_ac_power: false
  max_background_gpu_busy_pct: 100
  min_available_memory_gib: 0
  temperature_warn_c: 100
""",
        encoding="utf-8",
    )
    return capsule, machine


def _args(capsule: Path, machine: Path, home: Path) -> list[str]:
    return [
        "run",
        "capsule",
        str(capsule),
        "--machine",
        str(machine),
        "--build",
        "build-sha256:" + "a" * 64,
        "--home",
        str(home),
    ]


def test_capsule_cli_surface() -> None:
    root = runner.invoke(cli.app, ["--help"])
    group = runner.invoke(cli.app, ["run", "--help"])
    command = runner.invoke(cli.app, ["run", "capsule", "--help"])
    root_help = _plain_terminal_text(root.stdout)
    group_help = _plain_terminal_text(group.stdout)
    command_help = _plain_terminal_text(command.stdout)

    assert root.exit_code == group.exit_code == command.exit_code == 0
    assert "capsule" not in root_help
    assert "capsule" in group_help
    assert "run capsule" in command_help
    assert "--machine" in command_help
    assert "--build" in command_help
    assert "--home" in command_help
    assert "model-receipt" not in command_help


def test_capsule_cli_success_and_structured_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule, machine = _write_manifests(tmp_path)
    success = SimpleNamespace(
        run_id="run-20260901T000000Z-topk-capsule-" + "0" * 32,
        outcome=RunOutcome.SUCCESS,
        inspection=SimpleNamespace(record=tmp_path / "record"),
        result=SimpleNamespace(status="passed", reason="passed"),
    )
    monkeypatch.setattr(cli, "run_capsule", lambda *_a, **_k: success)

    passed = runner.invoke(cli.app, _args(capsule, machine, tmp_path / "home"))
    assert passed.exit_code == 0
    assert f"run: {success.run_id}" in passed.stdout
    assert "capsule: passed (passed)" in passed.stdout

    failed = SimpleNamespace(
        **{
            **success.__dict__,
            "outcome": RunOutcome.FAILURE,
            "result": SimpleNamespace(status="failed", reason="correctness-failed"),
        }
    )
    monkeypatch.setattr(cli, "run_capsule", lambda *_a, **_k: failed)
    rejected = runner.invoke(cli.app, _args(capsule, machine, tmp_path / "home"))
    assert rejected.exit_code == 1
    assert "capsule: failed (correctness-failed)" in rejected.stderr


def test_capsule_cli_reports_fixed_safe_pre_and_postallocation_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule, machine = _write_manifests(tmp_path)
    secret = "do-not-print-secret-123456"
    monkeypatch.setattr(
        cli,
        "run_capsule",
        lambda *_a, **_k: (_ for _ in ()).throw(CapsuleRunError(secret + " ${TOKEN}")),
    )
    pre = runner.invoke(
        cli.app,
        _args(capsule, machine, tmp_path / "home"),
        env={"AWS_SECRET_ACCESS_KEY": secret},
    )
    assert pre.exit_code == 1
    assert pre.stderr.strip() == "capsule run failed before allocating a run"
    assert secret not in pre.output and "${TOKEN}" not in pre.output

    execution = CapsuleExecutionError(
        run_id="run-20260901T000000Z-topk-capsule-" + "1" * 32,
        record=tmp_path / "record",
    )
    execution.__cause__ = RuntimeError(secret + " child output")
    monkeypatch.setattr(cli, "run_capsule", lambda *_a, **_k: (_ for _ in ()).throw(execution))
    post = runner.invoke(
        cli.app,
        _args(capsule, machine, tmp_path / "home"),
        env={"AWS_SECRET_ACCESS_KEY": secret},
    )
    assert post.exit_code == 1
    assert execution.run_id in post.stderr
    assert "capsule run failed before producing a structured result" in post.stderr
    assert secret not in post.output


def test_capsule_cli_rejects_sensitive_interpolation_without_echo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capsule, machine = _write_manifests(tmp_path)
    capsule.write_text(capsule.read_text().replace("candidate-a", "${AWS_SECRET_ACCESS_KEY}"))
    secret = "valid-but-secret-candidate"
    called = False

    def tripwire(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("must not execute")

    monkeypatch.setattr(cli, "run_capsule", tripwire)
    result = runner.invoke(
        cli.app,
        _args(capsule, machine, tmp_path / "home"),
        env={"AWS_SECRET_ACCESS_KEY": secret},
    )

    assert result.exit_code == 1
    assert "sensitive environment interpolation is forbidden" in result.stderr
    assert secret not in result.output
    assert not called


def test_capsule_cli_validation_never_echoes_invalid_input(tmp_path: Path) -> None:
    capsule, machine = _write_manifests(tmp_path)
    secretish = "secretish-invalid-value"
    capsule.write_text(capsule.read_text().replace("candidate-a", secretish + "!"))

    result = runner.invoke(cli.app, _args(capsule, machine, tmp_path / "home"))

    assert result.exit_code == 1
    assert "invalid capsule invocation" in result.stderr
    assert secretish not in result.output

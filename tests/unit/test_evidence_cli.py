from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

import strixlab.cli as cli
from strixlab.cli import app

runner = CliRunner()

_ENV = {"PATH": "/usr/bin"}


def _finalized_run(home: Path) -> str:
    import strixlab.evidence as ev

    with ev.begin_run("exp-cli", b"suite: c\n", resolved={"a": 1}, home=home, environ=_ENV) as run:
        run.write_portable(
            "env.json", b'{"k":1}\n', media_type="application/json", role="environment"
        )
        run.succeed()
    return run.run_id


def test_run_inspect_prints_canonical_json(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run_id = _finalized_run(home)
    result = runner.invoke(app, ["run", "inspect", run_id, "--home", str(home)])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["run_id"] == run_id
    assert payload["outcome"] == "success"
    assert payload["state"] == "terminal"
    assert payload["record_sha256"].startswith("record-sha256:")


def test_run_inspect_missing_run_exits_one(tmp_path: Path) -> None:
    result = runner.invoke(app, ["run", "inspect", "run-nope", "--home", str(tmp_path / "empty")])
    assert result.exit_code == 1
    assert "run inspect failed" in result.output


def test_run_inspect_invalid_home_exits_one() -> None:
    result = runner.invoke(app, ["run", "inspect", "run-x", "--home", "relative/home"])
    assert result.exit_code == 1
    assert "run inspect failed" in result.output


def test_bundle_export_and_verify_roundtrip(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run_id = _finalized_run(home)
    destination = tmp_path / "bundle"
    with mock.patch.dict(os.environ, _ENV, clear=True):
        export = runner.invoke(
            app, ["bundle", "export", run_id, str(destination), "--home", str(home)]
        )
    assert export.exit_code == 0
    assert str(destination) in export.stdout
    assert destination.is_dir()

    verify = runner.invoke(app, ["bundle", "verify", str(destination)])
    assert verify.exit_code == 0
    payload = json.loads(verify.stdout)
    assert payload["run_id"] == run_id
    assert payload["outcome"] == "success"
    assert payload["member_count"] >= 1


def test_bundle_export_missing_run_exits_one(tmp_path: Path) -> None:
    with mock.patch.dict(os.environ, _ENV, clear=True):
        result = runner.invoke(
            app,
            ["bundle", "export", "run-nope", str(tmp_path / "b"), "--home", str(tmp_path / "e")],
        )
    assert result.exit_code == 1
    assert "bundle export failed" in result.output


def test_bundle_export_secret_in_environment_exits_one(tmp_path: Path) -> None:
    import strixlab.evidence as ev

    home = tmp_path / "home"
    with ev.begin_run("exp-leak", b"suite: l\n", resolved={"a": 1}, home=home, environ=_ENV) as run:
        run_id = run.run_id
        run.write_portable(
            "env.json",
            b'{"m":"leakme123"}\n',
            media_type="application/json",
            role="environment",
        )
        run.succeed()
    with mock.patch.dict(os.environ, {"API_TOKEN": "leakme123"}, clear=True):
        result = runner.invoke(
            app, ["bundle", "export", run_id, str(tmp_path / "b"), "--home", str(home)]
        )
    assert result.exit_code == 1
    assert "bundle export failed" in result.output


def test_bundle_verify_missing_directory_exits_one(tmp_path: Path) -> None:
    result = runner.invoke(app, ["bundle", "verify", str(tmp_path / "nope")])
    assert result.exit_code == 1
    assert "bundle verify failed" in result.output


def test_evidence_commands_do_not_echo_sensitive_dynamic_output(tmp_path: Path) -> None:
    token = "EVIDENCE_SENTINEL_9f84"
    sensitive_environment = {"API_TOKEN": token, "PATH": "/usr/bin"}

    with mock.patch.dict(os.environ, sensitive_environment, clear=True):
        inspect = runner.invoke(app, ["run", "inspect", token, "--home", str(tmp_path)])
        verify = runner.invoke(app, ["bundle", "verify", str(tmp_path / token)])

    for result in (inspect, verify):
        assert result.exit_code == 1
        assert token not in result.output
        assert "unable to safely render terminal output" in result.output


def test_bundle_export_does_not_echo_sensitive_destination(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run_id = _finalized_run(home)
    token = "EVIDENCE_SENTINEL_9f84"
    destination = tmp_path / token

    with mock.patch.dict(os.environ, {"API_TOKEN": token, "PATH": "/usr/bin"}, clear=True):
        result = runner.invoke(
            app, ["bundle", "export", run_id, str(destination), "--home", str(home)]
        )

    assert destination.is_dir()
    assert result.exit_code == 1
    assert token not in result.output
    assert "unable to safely render terminal output" in result.output


def test_bundle_verify_tampered_exits_one(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run_id = _finalized_run(home)
    destination = tmp_path / "bundle"
    with mock.patch.dict(os.environ, _ENV, clear=True):
        cli.export_bundle(run_id, destination, home=home, environ=_ENV)
    target = destination / "run" / "status.json"
    target.chmod(0o600)
    target.write_bytes(b"{}")
    result = runner.invoke(app, ["bundle", "verify", str(destination)])
    assert result.exit_code == 1
    assert "bundle verify failed" in result.output


def test_run_inspect_usage_error_exits_two() -> None:
    result = runner.invoke(app, ["run", "inspect"])
    assert result.exit_code == 2


def test_bundle_export_usage_error_exits_two() -> None:
    result = runner.invoke(app, ["bundle", "export"])
    assert result.exit_code == 2

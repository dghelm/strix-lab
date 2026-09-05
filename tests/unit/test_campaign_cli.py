from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from typer.testing import CliRunner

import strixlab.campaign_cli as campaign_cli
from strixlab.campaigns import CampaignError, CampaignPhase, CampaignState
from strixlab.cli import app
from strixlab.secret_policy import UnsafeOutputError

runner = CliRunner()
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CORE_DIGEST = "a" * 64
_CORE_STATUSES = (
    ("ready", 0),
    ("completed", 0),
    ("running", 1),
    ("blocked", 1),
    ("interrupted", 1),
    ("budget_exhausted", 1),
)


def _plain(value: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", value)


def _invoke(args: list[str], *, env: dict[str, str] | None = None) -> Any:
    return runner.invoke(app, args, env=env)


def _state(**overrides: Any) -> CampaignState:
    values: dict[str, Any] = {
        "id": "campaign-fixture",
        "frozen_sha256": _CORE_DIGEST,
        "status": "ready",
        "reason": "created; no execution admitted",
        "max_suite_runs": 10,
    }
    values.update(overrides)
    return CampaignState(**values)


def test_campaign_cli_surface() -> None:
    root = _invoke(["--help"])
    group = _invoke(["campaign", "--help"])
    create = _invoke(["campaign", "create", "--help"])
    resume = _invoke(["campaign", "resume", "--help"])
    inspect = _invoke(["campaign", "inspect", "--help"])
    report = _invoke(["campaign", "report", "--help"])

    assert (
        root.exit_code
        == group.exit_code
        == create.exit_code
        == resume.exit_code
        == inspect.exit_code
        == report.exit_code
        == 0
    )
    root_help = _plain(root.stdout)
    group_help = _plain(group.stdout)
    assert "campaign" in root_help
    assert "create" in group_help
    assert "resume" in group_help
    assert "inspect" in group_help
    assert "report" in group_help
    for result in (create, resume, inspect, report):
        assert "--home" in _plain(result.stdout)


@pytest.mark.parametrize("command", ["create", "resume", "inspect", "report"])
def test_campaign_commands_usage_error_exits_two(command: str) -> None:
    result = _invoke(["campaign", command])
    assert result.exit_code == 2


def test_create_prints_canonical_json_and_forwards_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = tmp_path / "plan.yaml"
    plan.write_text("schema_version: 1\n", encoding="utf-8")
    home = tmp_path / "home"
    captured: dict[str, Any] = {}
    state = _state()

    def fake_create(plan_path: Path, *, home: Path, environ: Any) -> CampaignState:
        captured.update(plan_path=plan_path, home=home, environ=dict(environ))
        return state

    monkeypatch.setattr(campaign_cli, "create_campaign", fake_create)
    result = _invoke(["campaign", "create", str(plan), "--home", str(home)])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == state.model_dump(mode="json")
    assert captured["plan_path"] == plan
    assert captured["home"] == home
    assert isinstance(captured["environ"], dict)


def test_resume_completed_with_failed_candidates_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(
        status="completed",
        reason="finite candidate list evaluated",
        phases=[
            CampaignPhase(
                candidate_id="candidate-a",
                phase="screening",
                reserved_suite_runs=4,
                status="failed",
                stage="terminal",
                decision="objective_not_met",
            )
        ],
    )
    captured: dict[str, Any] = {}

    def fake_resume(campaign_id: str, *, home: Path, environ: Any) -> CampaignState:
        captured.update(campaign_id=campaign_id, home=home, environ=dict(environ))
        return state

    monkeypatch.setattr(campaign_cli, "resume_campaign", fake_resume)
    home = tmp_path / "home"
    result = _invoke(["campaign", "resume", "campaign-fixture", "--home", str(home)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert payload["phases"][0]["status"] == "failed"
    assert captured["campaign_id"] == "campaign-fixture"
    assert captured["home"] == home


@pytest.mark.parametrize("status", ["blocked", "interrupted", "budget_exhausted"])
def test_resume_unsuccessful_status_prints_json_and_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    state = _state(status=status, reason=status)
    monkeypatch.setattr(campaign_cli, "resume_campaign", lambda *_args, **_kwargs: state)
    result = _invoke(["campaign", "resume", "campaign-fixture", "--home", str(tmp_path / "home")])
    assert result.exit_code == 1
    assert json.loads(result.stdout) == state.model_dump(mode="json")


def test_inspect_prints_canonical_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    state = _state()

    def fake_inspect(campaign_id: str, *, home: Path) -> CampaignState:
        captured.update(campaign_id=campaign_id, home=home)
        return state

    monkeypatch.setattr(campaign_cli, "inspect_campaign", fake_inspect)
    home = tmp_path / "home"
    result = _invoke(["campaign", "inspect", "campaign-fixture", "--home", str(home)])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == state.model_dump(mode="json")
    assert captured == {"campaign_id": "campaign-fixture", "home": home}
    assert "environ" not in captured


@pytest.mark.parametrize("status", ["blocked", "running"])
def test_inspect_non_success_status_prints_json_and_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    state = _state(status=status, reason=status)
    monkeypatch.setattr(campaign_cli, "inspect_campaign", lambda *_args, **_kwargs: state)
    result = _invoke(["campaign", "inspect", "campaign-fixture", "--home", str(tmp_path / "home")])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == status


def test_inspect_forwards_core_campaign_state_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(
        id="historical-mmvq-demo",
        objective_cases=["pp512", "tg128"],
    )
    monkeypatch.setattr(campaign_cli, "inspect_campaign", lambda *_args, **_kwargs: state)
    result = _invoke(
        ["campaign", "inspect", "historical-mmvq-demo", "--home", str(tmp_path / "home")]
    )
    assert result.exit_code == 0
    assert json.loads(result.stdout) == state.model_dump(mode="json")


def test_report_prints_human_readable_actionable_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(status="completed", reason="finite candidate list evaluated")
    captured: dict[str, Any] = {}

    def fake_inspect(campaign_id: str, *, home: Path) -> CampaignState:
        captured["inspect"] = {"campaign_id": campaign_id, "home": home}
        return state

    def fake_render(rendered: CampaignState) -> str:
        captured["state"] = rendered
        return (
            "Campaign campaign-fixture is completed.\n"
            "Next: retain candidate-b evidence; reject candidate-a (failed).\n"
        )

    monkeypatch.setattr(campaign_cli, "inspect_campaign", fake_inspect)
    monkeypatch.setattr(campaign_cli, "render_campaign_report", fake_render)
    home = tmp_path / "home"
    result = _invoke(["campaign", "report", "campaign-fixture", "--home", str(home)])

    assert result.exit_code == 0
    assert "Campaign campaign-fixture is completed." in result.stdout
    assert "reject candidate-a (failed)" in result.stdout
    assert captured["inspect"] == {"campaign_id": "campaign-fixture", "home": home}
    assert captured["state"] is state


def test_report_prints_core_renderer_comparison_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(status="completed", reason="finite candidate list evaluated")
    monkeypatch.setattr(campaign_cli, "inspect_campaign", lambda *_a, **_k: state)

    def fake_render(rendered: CampaignState) -> str:
        assert rendered is state
        return (
            "# Campaign campaign-fixture\n"
            "Status: completed\n"
            "Judge: `mixed`; objective met: `false`\n"
            "- `pp512`: improved; change 1.250%; interval [0.100, 2.400]%\n"
        )

    monkeypatch.setattr(campaign_cli, "render_campaign_report", fake_render)
    result = _invoke(["campaign", "report", "campaign-fixture", "--home", str(tmp_path / "home")])
    assert result.exit_code == 0
    assert "Judge: `mixed`" in result.stdout
    assert "`pp512`: improved" in result.stdout


def test_report_budget_exhausted_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        campaign_cli,
        "inspect_campaign",
        lambda *_args, **_kwargs: _state(status="budget_exhausted", reason="budget exhausted"),
    )
    monkeypatch.setattr(
        campaign_cli,
        "render_campaign_report",
        lambda _state: "Campaign stopped: budget exhausted. Do not replay spent phases.",
    )
    result = _invoke(["campaign", "report", "campaign-fixture", "--home", str(tmp_path / "home")])
    assert result.exit_code == 1
    assert "budget exhausted" in result.stdout


def test_create_campaign_error_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        campaign_cli,
        "create_campaign",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(CampaignError("invalid campaign plan")),
    )
    result = _invoke(
        ["campaign", "create", str(tmp_path / "plan.yaml"), "--home", str(tmp_path / "home")]
    )
    assert result.exit_code == 1
    assert "campaign create failed: invalid campaign plan" in result.stderr


def test_resume_campaign_error_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        campaign_cli,
        "resume_campaign",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(CampaignError("unknown campaign ID")),
    )
    result = _invoke(["campaign", "resume", "campaign-missing", "--home", str(tmp_path / "home")])
    assert result.exit_code == 1
    assert "campaign resume failed: unknown campaign ID" in result.stderr


def test_inspect_campaign_error_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        campaign_cli,
        "inspect_campaign",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(CampaignError("unknown campaign ID")),
    )
    result = _invoke(["campaign", "inspect", "campaign-missing", "--home", str(tmp_path / "home")])
    assert result.exit_code == 1
    assert "campaign inspect failed: unknown campaign ID" in result.stderr


def test_create_does_not_use_typer_path_exists_check(tmp_path: Path) -> None:
    missing = tmp_path / "gone.yaml"
    result = _invoke(["campaign", "create", str(missing), "--home", str(tmp_path / "home")])
    assert result.exit_code == 1
    assert result.exit_code != 2
    assert "Invalid value" not in result.output
    assert "does not exist" not in result.output


def test_create_missing_plan_path_is_secret_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "supersecretvalue"
    missing = tmp_path / secret / "plan.yaml"

    def fake_create(plan_path: Path, *, home: Path, environ: Any) -> CampaignState:
        raise FileNotFoundError(f"plan not found: {plan_path}")

    monkeypatch.setattr(campaign_cli, "create_campaign", fake_create)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)
    result = _invoke(
        ["campaign", "create", str(missing), "--home", str(tmp_path / "home")],
        env={"AWS_SECRET_ACCESS_KEY": secret},
    )

    assert result.exit_code == 1
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert "unable to safely render terminal output" in result.stderr


def test_json_payload_secret_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "supersecretvalue"
    monkeypatch.setattr(
        campaign_cli,
        "inspect_campaign",
        lambda *_args, **_kwargs: _state(reason=secret),
    )
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)
    result = _invoke(
        ["campaign", "inspect", "campaign-fixture", "--home", str(tmp_path / "home")],
        env={"AWS_SECRET_ACCESS_KEY": secret},
    )
    assert result.exit_code == 1
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert "unable to safely render terminal output" in result.stderr


def test_report_secret_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "supersecretvalue"
    monkeypatch.setattr(
        campaign_cli,
        "inspect_campaign",
        lambda *_args, **_kwargs: _state(status="completed", reason="completed"),
    )
    monkeypatch.setattr(
        campaign_cli,
        "render_campaign_report",
        lambda _state: f"leak {secret} in report",
    )
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)
    result = _invoke(
        ["campaign", "report", "campaign-fixture", "--home", str(tmp_path / "home")],
        env={"AWS_SECRET_ACCESS_KEY": secret},
    )
    assert result.exit_code == 1
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert "unable to safely render terminal output" in result.stderr


@pytest.mark.parametrize(
    ("args", "target"),
    [
        (["campaign", "create", "plan.yaml"], "create_campaign"),
        (["campaign", "resume", "campaign-fixture"], "resume_campaign"),
        (["campaign", "inspect", "campaign-fixture"], "inspect_campaign"),
        (["campaign", "report", "campaign-fixture"], "inspect_campaign"),
    ],
)
def test_core_unsafe_output_error_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: list[str],
    target: str,
) -> None:
    monkeypatch.setattr(
        campaign_cli,
        target,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(UnsafeOutputError("leaked")),
    )
    result = _invoke([*args, "--home", str(tmp_path / "home")])
    assert result.exit_code == 1
    assert "unable to safely render terminal output" in result.stderr
    assert "leaked" not in result.output


def test_report_campaign_error_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        campaign_cli,
        "inspect_campaign",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(CampaignError("unknown campaign ID")),
    )
    result = _invoke(["campaign", "report", "campaign-missing", "--home", str(tmp_path / "home")])
    assert result.exit_code == 1
    assert "campaign report failed: unknown campaign ID" in result.stderr


def test_relative_home_exits_one_without_leaking_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "supersecretvalue"
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)
    result = _invoke(
        ["campaign", "inspect", "campaign-fixture", "--home", f"relative/{secret}"],
        env={"AWS_SECRET_ACCESS_KEY": secret},
    )
    assert result.exit_code == 1
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert "campaign inspect failed" in result.stderr or (
        "unable to safely render terminal output" in result.stderr
    )


@pytest.mark.parametrize(("status", "exit_code"), _CORE_STATUSES)
def test_inspect_real_campaign_state_status_exit_codes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, exit_code: int
) -> None:
    state = _state(status=status, reason="fixture")
    monkeypatch.setattr(campaign_cli, "inspect_campaign", lambda *_a, **_k: state)
    with mock.patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True):
        result = _invoke(
            ["campaign", "inspect", "campaign-fixture", "--home", str(tmp_path / "home")],
            env={"PATH": "/usr/bin"},
        )
    assert result.exit_code == exit_code
    payload = json.loads(result.stdout)
    assert payload == state.model_dump(mode="json")
    assert payload["status"] == status
    assert payload["phases"] == []
    assert payload["objective_cases"] == []
    assert payload["protected_regression_margin_percent"] == 0.0
    assert payload["baseline"] is None


def test_report_uses_exported_render_campaign_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state()
    monkeypatch.setattr(campaign_cli, "inspect_campaign", lambda *_a, **_k: state)
    with mock.patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True):
        result = _invoke(
            ["campaign", "report", "campaign-fixture", "--home", str(tmp_path / "home")],
            env={"PATH": "/usr/bin"},
        )
    assert result.exit_code == 0
    assert "# Campaign campaign-fixture" in result.stdout
    assert "Status: ready" in result.stdout
    assert "Protected regression margin: 0%" in result.stdout
    assert "Objective cases:" in result.stdout


def test_inspect_unknown_id_uses_core(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with mock.patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True):
        result = _invoke(
            ["campaign", "inspect", "campaign-does-not-exist", "--home", str(home)],
            env={"PATH": "/usr/bin"},
        )
    assert result.exit_code == 1
    assert "campaign inspect failed" in result.stderr
    assert "campaign-does-not-exist" not in result.stdout


def test_create_invalid_plan_uses_core(tmp_path: Path) -> None:
    plan = tmp_path / "plan.yaml"
    plan.write_text("schema_version: 1\nid: INVALID\n", encoding="utf-8")
    with mock.patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True):
        result = _invoke(
            ["campaign", "create", str(plan), "--home", str(tmp_path / "home")],
            env={"PATH": "/usr/bin"},
        )
    assert result.exit_code == 1
    assert "campaign create failed" in result.stderr
    assert "INVALID" not in result.stdout

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
from strixlab.cli import app
from strixlab.secret_policy import UnsafeOutputError

runner = CliRunner()
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class CampaignError(ValueError):
    """Test double matching the core CampaignError(ValueError) contract."""


class FakeState:
    def __init__(self, status: str, payload: dict[str, Any] | None = None) -> None:
        self.status = status
        self.payload = payload or {}

    def model_dump(self, *, mode: str) -> dict[str, Any]:
        assert mode == "json"
        return {"status": self.status, **self.payload}


def _plain(value: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", value)


def _invoke(args: list[str], *, env: dict[str, str] | None = None) -> Any:
    return runner.invoke(app, args, env=env)


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
    state = FakeState(
        "ready",
        {"campaign_id": "campaign-fixture", "candidates": []},
    )

    def fake_create(plan_path: Path, *, home: Path, environ: Any) -> FakeState:
        captured.update(plan_path=plan_path, home=home, environ=dict(environ))
        return state

    monkeypatch.setattr(campaign_cli, "create_campaign", fake_create)
    result = _invoke(["campaign", "create", str(plan), "--home", str(home)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload == {"campaign_id": "campaign-fixture", "candidates": [], "status": "ready"}
    assert captured["plan_path"] == plan
    assert captured["home"] == home
    assert isinstance(captured["environ"], dict)


def test_resume_completed_with_failed_candidates_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = FakeState(
        "completed",
        {
            "campaign_id": "campaign-fixture",
            "candidates": [
                {"id": "candidate-a", "status": "failed"},
                {"id": "candidate-b", "status": "retained"},
            ],
        },
    )
    captured: dict[str, Any] = {}

    def fake_resume(campaign_id: str, *, home: Path, environ: Any) -> FakeState:
        captured.update(campaign_id=campaign_id, home=home, environ=dict(environ))
        return state

    monkeypatch.setattr(campaign_cli, "resume_campaign", fake_resume)
    home = tmp_path / "home"
    result = _invoke(["campaign", "resume", "campaign-fixture", "--home", str(home)])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert payload["candidates"][0]["status"] == "failed"
    assert captured["campaign_id"] == "campaign-fixture"
    assert captured["home"] == home


@pytest.mark.parametrize("status", ["blocked", "interrupted", "budget_exhausted"])
def test_resume_unsuccessful_status_prints_json_and_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    monkeypatch.setattr(
        campaign_cli,
        "resume_campaign",
        lambda *_args, **_kwargs: FakeState(status, {"campaign_id": "campaign-fixture"}),
    )
    result = _invoke(["campaign", "resume", "campaign-fixture", "--home", str(tmp_path / "home")])
    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["status"] == status


def test_inspect_prints_canonical_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_inspect(campaign_id: str, *, home: Path) -> FakeState:
        captured.update(campaign_id=campaign_id, home=home)
        return FakeState("ready", {"campaign_id": campaign_id})

    monkeypatch.setattr(campaign_cli, "inspect_campaign", fake_inspect)
    home = tmp_path / "home"
    result = _invoke(["campaign", "inspect", "campaign-fixture", "--home", str(home)])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"campaign_id": "campaign-fixture", "status": "ready"}
    assert captured == {"campaign_id": "campaign-fixture", "home": home}
    assert "environ" not in captured


def test_inspect_blocked_status_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        campaign_cli,
        "inspect_campaign",
        lambda *_args, **_kwargs: FakeState("blocked", {"campaign_id": "campaign-fixture"}),
    )
    result = _invoke(["campaign", "inspect", "campaign-fixture", "--home", str(tmp_path / "home")])
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "blocked"


def test_report_prints_human_readable_actionable_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = FakeState(
        "completed",
        {
            "campaign_id": "campaign-fixture",
            "candidates": [{"id": "candidate-a", "status": "failed"}],
        },
    )
    captured: dict[str, Any] = {}

    def fake_inspect(campaign_id: str, *, home: Path) -> FakeState:
        captured["inspect"] = {"campaign_id": campaign_id, "home": home}
        return state

    def fake_render(rendered: FakeState) -> str:
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


def test_report_budget_exhausted_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        campaign_cli,
        "inspect_campaign",
        lambda *_args, **_kwargs: FakeState("budget_exhausted"),
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

    def fake_create(plan_path: Path, *, home: Path, environ: Any) -> FakeState:
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
        lambda *_args, **_kwargs: FakeState("ready", {"note": secret}),
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
        lambda *_args, **_kwargs: FakeState("completed"),
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


def test_inspect_unknown_id_uses_core_when_available(tmp_path: Path) -> None:
    pytest.importorskip("strixlab.campaigns")
    home = tmp_path / "home"
    with mock.patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True):
        result = _invoke(
            ["campaign", "inspect", "campaign-does-not-exist", "--home", str(home)],
            env={"PATH": "/usr/bin"},
        )
    assert result.exit_code == 1
    assert "campaign inspect failed" in result.stderr
    assert "campaign-does-not-exist" not in result.stdout


def test_create_invalid_plan_uses_core_when_available(tmp_path: Path) -> None:
    pytest.importorskip("strixlab.campaigns")
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


def test_wrappers_delegate_to_core_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    calls: dict[str, Any] = {}
    state = FakeState("ready", {"campaign_id": "campaign-core"})

    def fake_create(plan_path: Path, *, home: Path, environ: Any) -> FakeState:
        calls["create"] = {"plan_path": plan_path, "home": home, "environ": dict(environ)}
        return state

    def fake_resume(campaign_id: str, *, home: Path, environ: Any) -> FakeState:
        calls["resume"] = {"campaign_id": campaign_id, "home": home, "environ": dict(environ)}
        return state

    def fake_inspect(campaign_id: str, *, home: Path) -> FakeState:
        calls["inspect"] = {"campaign_id": campaign_id, "home": home}
        return state

    def fake_render(rendered: FakeState) -> str:
        calls["report"] = rendered
        return "ok\n"

    module = types.ModuleType("strixlab.campaigns")
    module.create_campaign = fake_create  # type: ignore[attr-defined]
    module.resume_campaign = fake_resume  # type: ignore[attr-defined]
    module.inspect_campaign = fake_inspect  # type: ignore[attr-defined]
    module.render_campaign_report = fake_render  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "strixlab.campaigns", module)

    plan = tmp_path / "plan.yaml"
    home = tmp_path / "home"
    environ = {"PATH": "/usr/bin"}
    assert campaign_cli.create_campaign(plan, home=home, environ=environ) is state
    assert campaign_cli.resume_campaign("campaign-core", home=home, environ=environ) is state
    assert campaign_cli.inspect_campaign("campaign-core", home=home) is state
    assert campaign_cli.render_campaign_report(state) == "ok\n"
    assert calls["create"] == {"plan_path": plan, "home": home, "environ": environ}
    assert calls["resume"]["campaign_id"] == "campaign-core"
    assert calls["inspect"] == {"campaign_id": "campaign-core", "home": home}
    assert calls["report"] is state
    module.render_campaign_report = lambda _state: None  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="must return str"):
        campaign_cli.render_campaign_report(state)


def test_status_enum_value_is_used_for_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Status:
        value = "ready"

        def __str__(self) -> str:
            return "CampaignStatus.ready"

    class EnumState:
        status = Status()

        def model_dump(self, *, mode: str) -> dict[str, str]:
            return {"status": "ready"}

    monkeypatch.setattr(campaign_cli, "inspect_campaign", lambda *_a, **_k: EnumState())
    result = _invoke(["campaign", "inspect", "campaign-fixture", "--home", str(tmp_path / "home")])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["status"] == "ready"

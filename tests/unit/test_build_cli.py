from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from typer.testing import CliRunner

import strixlab.cli as cli
from strixlab.cli import app

runner = CliRunner()

_BUILD = "build-sha256:" + "aa" * 32
_RECIPE = "recipe-sha256:" + "bb" * 32
_ATTEMPT = "attempt-" + "0" * 24 + "-" + "c" * 32


def _manifest(tmp_path: Path, build_value: dict[str, Any]) -> Path:
    path = tmp_path / "build.yaml"
    path.write_text(yaml.safe_dump(build_value), encoding="utf-8")
    return path


def test_build_prepare_success(
    tmp_path: Path, build_value: dict[str, Any], monkeypatch: Any
) -> None:
    manifest = _manifest(tmp_path, build_value)
    fake = SimpleNamespace(
        build_id=_BUILD,
        execution_class="built",
        attempt=SimpleNamespace(record=tmp_path / "record"),
    )
    monkeypatch.setattr(cli, "execute_cmake_build", lambda *a, **k: fake)
    result = runner.invoke(
        app, ["build", "prepare", "prep-x", str(manifest), "--home", str(tmp_path / "home")]
    )
    assert result.exit_code == 0
    assert _BUILD in result.stdout
    assert "execution: built" in result.stdout


def test_build_prepare_domain_failure_exits_one(
    tmp_path: Path, build_value: dict[str, Any], monkeypatch: Any
) -> None:
    manifest = _manifest(tmp_path, build_value)

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise cli.CMakeBuildError("configure failed")

    monkeypatch.setattr(cli, "execute_cmake_build", boom)
    result = runner.invoke(app, ["build", "prepare", "prep-x", str(manifest)])
    assert result.exit_code == 1
    assert "build prepare failed" in result.output


def test_build_prepare_invalid_manifest_exits_one(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.yaml"
    manifest.write_text("schema_version: 1\nid: x\n", encoding="utf-8")
    result = runner.invoke(app, ["build", "prepare", "prep-x", str(manifest)])
    assert result.exit_code == 1
    assert "invalid build profile" in result.output


def test_build_prepare_usage_error_exits_two() -> None:
    result = runner.invoke(app, ["build", "prepare"])
    assert result.exit_code == 2


def test_build_inspect_dispatches_by_prefix(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        cli,
        "inspect_recipe",
        lambda identifier, *, home: SimpleNamespace(model_dump=lambda mode: {"kind": "recipe"}),
    )
    recipe = runner.invoke(app, ["build", "inspect", _RECIPE])
    assert recipe.exit_code == 0
    assert json.loads(recipe.stdout)["kind"] == "recipe"

    monkeypatch.setattr(
        cli,
        "inspect_build",
        lambda identifier, *, home: SimpleNamespace(
            attested=True,
            build_id=_BUILD,
            canonical=SimpleNamespace(model_dump=lambda mode: {"kind": "canonical"}),
            canonical_record_sha256="record-sha256:" + "aa" * 32,
            root=tmp_path / "root",
            state="present",
        ),
    )
    build = runner.invoke(app, ["build", "inspect", _BUILD])
    assert build.exit_code == 0
    payload = json.loads(build.stdout)
    assert payload["state"] == "present"
    assert payload["attested"] is True
    assert payload["canonical"]["kind"] == "canonical"

    monkeypatch.setattr(
        cli,
        "inspect_attempt",
        lambda identifier, *, home: SimpleNamespace(model_dump=lambda mode: {"kind": "attempt"}),
    )
    attempt = runner.invoke(app, ["build", "inspect", _ATTEMPT])
    assert attempt.exit_code == 0
    assert json.loads(attempt.stdout)["kind"] == "attempt"


def test_build_inspect_unrecognized_id_exits_one(tmp_path: Path) -> None:
    result = runner.invoke(app, ["build", "inspect", "weird-id", "--home", str(tmp_path / "h")])
    assert result.exit_code == 1
    assert "build inspect failed" in result.output


def test_build_inspect_missing_build_exits_one(tmp_path: Path) -> None:
    result = runner.invoke(app, ["build", "inspect", _BUILD, "--home", str(tmp_path / "empty")])
    assert result.exit_code == 1


def test_build_cleanup_success_and_failure(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setattr(
        cli,
        "cleanup_build",
        lambda build_id, *, home: SimpleNamespace(
            build_id=build_id, state="cleaned", record=tmp_path / "rec"
        ),
    )
    ok = runner.invoke(app, ["build", "cleanup", _BUILD])
    assert ok.exit_code == 0
    assert "cleaned" in ok.stdout

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise cli.BuildCacheError("integrity failure")

    monkeypatch.setattr(cli, "cleanup_build", boom)
    bad = runner.invoke(app, ["build", "cleanup", _BUILD])
    assert bad.exit_code == 1
    assert "build cleanup failed" in bad.output


def test_build_inspect_invalid_home_exits_one() -> None:
    # A relative --home makes resolve_home raise ValueError, which the exit-code
    # contract maps to 1 (not an uncaught traceback).
    result = runner.invoke(app, ["build", "inspect", _RECIPE, "--home", "relative/home"])
    assert result.exit_code == 1
    assert "build inspect failed" in result.output


def test_build_prepare_wrong_model_propagates_as_programmer_error(
    tmp_path: Path, build_value: dict[str, Any], monkeypatch: Any
) -> None:
    # A wrong-model invariant is a programmer error, not a domain error: it must
    # propagate (surface as an uncaught exception), not be mapped to exit 1.
    manifest = _manifest(tmp_path, build_value)
    monkeypatch.setattr(
        cli, "resolve_and_validate_manifest", lambda *a, **k: SimpleNamespace(kind="not-a-profile")
    )
    result = runner.invoke(app, ["build", "prepare", "prep-x", str(manifest)])
    assert isinstance(result.exception, TypeError)


def test_build_prepare_source_error_exits_one(
    tmp_path: Path, build_value: dict[str, Any], monkeypatch: Any
) -> None:
    # A SourceError (e.g. an unauthenticated/mispolicied lease) is a domain error
    # that must map to exit 1, unlike the wrong-model TypeError.
    from strixlab.sources import SourcePolicyError

    manifest = _manifest(tmp_path, build_value)

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise SourcePolicyError("source lease is not authorized")

    monkeypatch.setattr(cli, "execute_cmake_build", boom)
    result = runner.invoke(app, ["build", "prepare", "prep-x", str(manifest)])
    assert result.exit_code == 1
    assert "build prepare failed" in result.output

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from strixlab import __version__
from strixlab.cli import app
from strixlab.sources import SourcePolicyError

runner = CliRunner()


def test_help_and_version() -> None:
    help_result = runner.invoke(app, ["--help"])
    version_result = runner.invoke(app, ["--version"])

    assert help_result.exit_code == 0
    assert "Evidence-first optimization" in help_result.stdout
    assert version_result.exit_code == 0
    assert version_result.stdout.strip() == __version__ == "0.1.0"


def test_schema_show_prints_packaged_schema() -> None:
    result = runner.invoke(app, ["schema", "show", "source-lock"])

    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert schema["$id"] == "urn:strixlab:schema:source-lock:1"


def test_schema_show_rejects_unknown_kind() -> None:
    result = runner.invoke(app, ["schema", "show", "challenge"])

    assert result.exit_code == 2
    assert "unknown manifest kind" in result.output


def test_manifest_validate_is_pure_and_accepts_unresolved_values(tmp_path: Path) -> None:
    path = tmp_path / "build.yaml"
    path.write_text(
        """\
schema_version: 1
id: hip-rocm10-gfx1151
source: strix-llama
generator: Ninja
build_type: Release
environment:
  ROCM_PATH: ${ROCM10_PATH}
cmake:
  GGML_HIP: true
targets: [llama-bench]
post_build_capture: [binary_hashes]
""",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["manifest", "validate", "build", str(path)])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"valid raw build manifest: {path}"


def test_manifest_validate_reports_load_and_validation_failures(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("schema_version: 1\nid: first\nid: second\n", encoding="utf-8")

    invalid = runner.invoke(app, ["manifest", "validate", "source-lock", str(path)])
    missing = runner.invoke(app, ["manifest", "validate", "source-lock", str(path) + ".gone"])

    assert invalid.exit_code == 1
    assert "duplicate key" in invalid.stderr
    assert missing.exit_code == 1
    assert "invalid raw source-lock manifest" in missing.stderr


def test_manifest_validate_usage_failure_is_exit_two() -> None:
    result = runner.invoke(app, ["manifest", "validate"])
    assert result.exit_code == 2


def test_source_command_group_exposes_lifecycle_commands() -> None:
    result = runner.invoke(app, ["source", "--help"])

    assert result.exit_code == 0
    assert "prepare" in result.stdout
    assert "inspect" in result.stdout
    assert "cleanup" in result.stdout


def test_source_inspect_rejects_an_invalid_preparation_id(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["source", "inspect", "not-a-preparation", "--home", str(tmp_path / "home")],
    )

    assert result.exit_code == 1
    assert "invalid preparation ID" in result.stderr


def test_source_prepare_command_publishes_owned_paths(tmp_path: Path, monkeypatch: object) -> None:
    manifest = tmp_path / "source.yaml"
    manifest.write_text(
        """schema_version: 1
id: fixture
kind: git
url: /srv/git/fixture
commit: 0123456789abcdef0123456789abcdef01234567
branch_hint: main
submodules: false
adapter: llama_cpp
allowed_dirty_state: false
""",
        encoding="utf-8",
    )
    patch = tmp_path / "candidate.patch"
    patch.write_text("patch", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_prepare(lock: object, *, home: Path, patches: object, ssh_trust: object) -> object:
        captured.update(lock=lock, home=home, patches=patches, ssh_trust=ssh_trust)
        return SimpleNamespace(
            evidence=SimpleNamespace(preparation_id="prep-fixture-0123456789abcdef01234567"),
            worktree=home / "sources" / "worktrees" / "prepared",
            record=home / "sources" / "records" / "prepared",
        )

    monkeypatch.setattr("strixlab.cli.prepare_source", fake_prepare)  # type: ignore[attr-defined]
    home = tmp_path / "home"
    result = runner.invoke(
        app,
        [
            "source",
            "prepare",
            str(manifest),
            "--patch",
            str(patch),
            "--home",
            str(home),
        ],
    )

    assert result.exit_code == 0
    assert "prep-fixture-0123456789abcdef01234567" in result.stdout
    assert f"worktree: {home}" in result.stdout
    assert captured["home"] == home
    assert captured["patches"] == [patch]
    assert captured["ssh_trust"] is None


def test_source_prepare_command_reports_structured_validation_errors(tmp_path: Path) -> None:
    manifest = tmp_path / "source.yaml"
    manifest.write_text("schema_version: 1\nid: INVALID\n", encoding="utf-8")

    result = runner.invoke(app, ["source", "prepare", str(manifest), "--home", str(tmp_path)])

    assert result.exit_code == 1
    assert "invalid source lock" in result.stderr
    assert "id:" in result.stderr
    assert "url:" in result.stderr


def test_source_prepare_requires_known_hosts_for_ssh_inputs(tmp_path: Path) -> None:
    manifest = tmp_path / "source.yaml"
    manifest.write_text(
        """schema_version: 1
id: fixture
kind: git
url: ssh://git@example.test/repository.git
commit: 0123456789abcdef0123456789abcdef01234567
submodules: false
adapter: llama_cpp
allowed_dirty_state: false
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "source",
            "prepare",
            str(manifest),
            "--home",
            str(tmp_path / "home"),
            "--ssh-private-key",
            str(tmp_path / "key"),
        ],
    )

    assert result.exit_code == 1
    assert "--ssh-known-hosts is required" in result.stderr


def test_source_inspect_and_cleanup_commands_render_results(
    tmp_path: Path, monkeypatch: object
) -> None:
    preparation_id = "prep-fixture-0123456789abcdef01234567"
    dump = SimpleNamespace(model_dump=lambda **_kwargs: {"preparation_id": preparation_id})
    inspection = SimpleNamespace(
        evidence=dump,
        record_exists=True,
        registry=dump,
        worktree_exists=True,
    )
    cleanup = SimpleNamespace(
        preparation_id=preparation_id,
        state="cleaned",
        record=tmp_path / "record",
    )
    cleanup_arguments: dict[str, object] = {}
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "strixlab.cli.inspect_source", lambda *_args, **_kwargs: inspection
    )

    def fake_cleanup(*_args: object, **kwargs: object) -> object:
        cleanup_arguments.update(kwargs)
        return cleanup

    monkeypatch.setattr("strixlab.cli.cleanup_source", fake_cleanup)  # type: ignore[attr-defined]

    inspected = runner.invoke(app, ["source", "inspect", preparation_id, "--home", str(tmp_path)])
    cleaned = runner.invoke(
        app,
        [
            "source",
            "cleanup",
            preparation_id,
            "--home",
            str(tmp_path),
            "--force-changed",
        ],
    )

    assert inspected.exit_code == 0
    assert json.loads(inspected.stdout)["worktree_exists"] is True
    assert cleaned.exit_code == 0
    assert f"{preparation_id}: cleaned" in cleaned.stdout
    assert "record retained" in cleaned.stdout
    assert cleanup_arguments["force_changed"] is True


def test_source_commands_report_lifecycle_errors(tmp_path: Path, monkeypatch: object) -> None:
    manifest = tmp_path / "source.yaml"
    manifest.write_text(
        """schema_version: 1
id: fixture
kind: git
url: /srv/git/fixture
commit: 0123456789abcdef0123456789abcdef01234567
submodules: false
adapter: llama_cpp
allowed_dirty_state: false
""",
        encoding="utf-8",
    )

    def fail(*_args: object, **_kwargs: object) -> None:
        raise SourcePolicyError("policy blocker")

    monkeypatch.setattr("strixlab.cli.prepare_source", fail)  # type: ignore[attr-defined]
    monkeypatch.setattr("strixlab.cli.inspect_source", fail)  # type: ignore[attr-defined]
    monkeypatch.setattr("strixlab.cli.cleanup_source", fail)  # type: ignore[attr-defined]
    prepared = runner.invoke(app, ["source", "prepare", str(manifest), "--home", str(tmp_path)])
    inspected = runner.invoke(
        app,
        ["source", "inspect", "prep-fixture-0123456789abcdef01234567", "--home", str(tmp_path)],
    )
    cleaned = runner.invoke(
        app,
        ["source", "cleanup", "prep-fixture-0123456789abcdef01234567", "--home", str(tmp_path)],
    )

    assert prepared.exit_code == inspected.exit_code == cleaned.exit_code == 1
    assert "policy blocker" in prepared.stderr
    assert "policy blocker" in inspected.stderr
    assert "policy blocker" in cleaned.stderr


def test_python_module_entrypoint_matches_console_version() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "strixlab", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == __version__


def test_bare_source_version_fallback() -> None:
    source_root = Path(__file__).parents[2] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    result = subprocess.run(
        [sys.executable, "-S", "-c", "import strixlab; print(strixlab.__version__)"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        cwd=source_root,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "0+unknown"

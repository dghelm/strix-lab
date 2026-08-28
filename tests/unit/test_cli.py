from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from strixlab import __version__
from strixlab.cli import app

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

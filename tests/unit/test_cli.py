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
toolchain:
  mode: rocm
  prefixes:
    system: /usr
    rocm: "/opt/${ROCM_VERSION}"
  cmake: "${SYSTEM_PREFIX}/bin/cmake"
  ninja: /usr/bin/ninja
  c_compiler: "/opt/${ROCM_VERSION}/bin/amdclang"
  cxx_compiler: "/opt/${ROCM_VERSION}/bin/amdclang++"
  hip_compiler: "/opt/${ROCM_VERSION}/bin/amdclang++"
  rocm_prefix: "/opt/${ROCM_VERSION}"
  path: ["/opt/${ROCM_VERSION}/bin", /usr/bin]
execution:
  jobs: 8
  timeouts:
    discovery_seconds: 30.0
    configure_seconds: 300.0
    build_seconds: 1800.0
    inspection_seconds: 30.0
    capability_seconds: 30.0
environment:
  path_lists:
    ROCM_PATH: "/opt/${ROCM_VERSION}"
  literals:
    SOURCE_DATE_EPOCH: "0"
cmake:
  GGML_HIP: true
targets: [llama-bench]
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


# --- model verify -------------------------------------------------------------

import hashlib  # noqa: E402

from _model_fixtures import _DUMP, FakeEvidence, FakeLease, patch_lease_source  # noqa: E402

from strixlab import models  # noqa: E402

_MODEL_FIXTURE_DIR = (
    Path(__file__).resolve().parent.parent / "fixtures" / "models" / "gguf-dump-ca94157"
)
_TINY_GGUF = _MODEL_FIXTURE_DIR / "tiny-qwen35.gguf"
_MODEL_PREP_ID = "prep-strix-llama-" + "a" * 24


def _model_manifest_yaml(local_path: str, size_bytes: int, sha256: str) -> str:
    return f"""schema_version: 1
id: qwen35-2b-smoke
registry_status: registered
base_model:
  repository: Qwen/Qwen3.5-2B
  revision: 15852e8c16360a2fea060d615a32b45270f8a8fc
  license: Apache-2.0
architecture:
  family: qwen3_5
  moe: false
  gated_deltanet: true
  full_attention: true
  qsa: false
  mtp: true
  vision: true
artifact:
  format: gguf
  file:
    repository: bartowski/Qwen_Qwen3.5-2B-GGUF
    revision: 7d26695454df6de5fbcce2e58681e62dae06ce43
    filename: model.gguf
    local_path: {local_path}
    size_bytes: {size_bytes}
    sha256: "{sha256}"
  metadata_predicates:
    - key: general.architecture
      value_type: STRING
      scalar_value: qwen35
quantization:
  format_family: Q4_K
  storage_format: gguf
  tensor_policy_id: unknown
  tensor_policy_source: unknown
  calibration_method: unknown
  calibration_source: unknown
  calibration_hash: unknown
execution:
  verification_status: unverified
"""


def _prepare_model_verify(
    tmp_path: Path,
    monkeypatch: object,
    *,
    evidence: FakeEvidence | None = None,
    model_path: Path | None = None,
) -> tuple[Path, Path]:
    """Materialize a home, leased worktree, interpreter stub, and model manifest."""

    home = tmp_path / "home"
    home.mkdir()
    worktree = tmp_path / "worktree"
    scripts = worktree / "gguf-py" / "gguf" / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "gguf_dump.py").write_text(
        "# pinned inspector script placeholder\n", encoding="utf-8"
    )
    python = tmp_path / "python-stub"
    python.write_text(
        f"#!/usr/bin/python3\nimport sys\nsys.stdout.buffer.write({_DUMP!r})\n", encoding="utf-8"
    )
    python.chmod(0o755)
    gguf = tmp_path / "model.gguf" if model_path is None else model_path
    if model_path is None:
        gguf.write_bytes(_TINY_GGUF.read_bytes())

    lease = FakeLease(worktree=worktree, evidence=evidence or FakeEvidence())
    patch_lease_source(monkeypatch, lease, expected_preparation_id=_MODEL_PREP_ID)  # type: ignore[arg-type]
    monkeypatch.setattr(  # type: ignore[attr-defined]
        "strixlab.models._default_inspector_interpreter", lambda: python
    )

    size = gguf.stat().st_size if gguf.exists() else 0
    sha = hashlib.sha256(gguf.read_bytes()).hexdigest() if gguf.exists() else "0" * 64
    manifest = tmp_path / "model.yaml"
    manifest.write_text(_model_manifest_yaml(str(gguf), size, sha), encoding="utf-8")
    return home, manifest


def test_model_verify_prints_receipt_sha256(tmp_path: Path, monkeypatch: object) -> None:
    home, manifest = _prepare_model_verify(tmp_path, monkeypatch)

    result = runner.invoke(
        app, ["model", "verify", str(manifest), "--source", _MODEL_PREP_ID, "--home", str(home)]
    )

    assert result.exit_code == 0
    digest = result.stdout.strip()
    assert len(digest) == 64 and all(character in "0123456789abcdef" for character in digest)
    # The printed digest is exactly the id ``run suite`` re-authenticates.
    reloaded = models.load_model_receipt("qwen35-2b-smoke", digest, home=home)
    assert models.receipt_registry_sha256(reloaded) == digest


def test_model_verify_is_repeatable(tmp_path: Path, monkeypatch: object) -> None:
    home, manifest = _prepare_model_verify(tmp_path, monkeypatch)
    args = ["model", "verify", str(manifest), "--source", _MODEL_PREP_ID, "--home", str(home)]

    first = runner.invoke(app, args)
    second = runner.invoke(app, args)

    assert first.exit_code == second.exit_code == 0
    assert first.stdout.strip() == second.stdout.strip()


def test_model_verify_reports_wrong_source_id(tmp_path: Path, monkeypatch: object) -> None:
    home, manifest = _prepare_model_verify(
        tmp_path, monkeypatch, evidence=FakeEvidence(source_id="other-source")
    )

    result = runner.invoke(
        app, ["model", "verify", str(manifest), "--source", _MODEL_PREP_ID, "--home", str(home)]
    )

    assert result.exit_code == 1
    assert "model verify failed" in result.stderr
    assert result.stdout == ""


def test_model_verify_reports_missing_model_artifact(tmp_path: Path, monkeypatch: object) -> None:
    absent = tmp_path / "absent.gguf"
    home, manifest = _prepare_model_verify(tmp_path, monkeypatch, model_path=absent)
    # Give the manifest a plausible size/sha so resolution passes and the file check fails.
    manifest.write_text(_model_manifest_yaml(str(absent), 4, "0" * 64), encoding="utf-8")

    result = runner.invoke(
        app, ["model", "verify", str(manifest), "--source", _MODEL_PREP_ID, "--home", str(home)]
    )

    assert result.exit_code == 1
    assert "model verify failed" in result.stderr


def test_model_verify_reports_invalid_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.yaml"
    manifest.write_text("schema_version: 1\nid: INVALID\n", encoding="utf-8")

    result = runner.invoke(
        app, ["model", "verify", str(manifest), "--source", _MODEL_PREP_ID, "--home", str(tmp_path)]
    )

    assert result.exit_code == 1
    assert "invalid model manifest" in result.stderr


def test_model_verify_missing_manifest_reports_command_body_error(tmp_path: Path) -> None:
    # No parser-level ``exists`` check: the missing path is surfaced by ``read_manifest``
    # inside the redaction-protected body, not by Typer before the safety check runs.
    result = runner.invoke(
        app,
        ["model", "verify", str(tmp_path / "gone.yaml"), "--source", _MODEL_PREP_ID],
    )
    assert result.exit_code == 1
    assert "model verify failed" in result.stderr


def test_model_verify_directory_manifest_reports_command_body_error(tmp_path: Path) -> None:
    # A directory input also reaches ``read_manifest`` (no parser ``dir_okay=False``).
    result = runner.invoke(
        app,
        ["model", "verify", str(tmp_path), "--source", _MODEL_PREP_ID],
    )
    assert result.exit_code == 1
    assert "model verify failed" in result.stderr


def test_model_verify_missing_manifest_path_is_secret_safe(
    tmp_path: Path, monkeypatch: object
) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "supersecretvalue")  # type: ignore[attr-defined]
    # The manifest path itself embeds the secret and does not exist. Without parser-level
    # validation this OSError is echoed only through the RedactionContext, which fails
    # closed to the generic safe line instead of disclosing the path.
    secret_dir = tmp_path / "supersecretvalue"
    secret_dir.mkdir()
    missing = secret_dir / "model.yaml"

    result = runner.invoke(
        app,
        ["model", "verify", str(missing), "--source", _MODEL_PREP_ID, "--home", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "supersecretvalue" not in result.stdout
    assert "supersecretvalue" not in result.stderr
    assert "unable to safely render terminal output" in result.stderr


def test_model_verify_requires_source_option(tmp_path: Path, monkeypatch: object) -> None:
    _home, manifest = _prepare_model_verify(tmp_path, monkeypatch)
    result = runner.invoke(app, ["model", "verify", str(manifest)])
    assert result.exit_code == 2


def test_model_verify_output_is_secret_safe(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "supersecretvalue")  # type: ignore[attr-defined]
    # A missing artifact whose absolute path literally embeds the secret would otherwise
    # echo it in the failure line; the redaction net must prevent that.
    secret_dir = tmp_path / "supersecretvalue"
    secret_dir.mkdir()
    absent = secret_dir / "model.gguf"
    home, manifest = _prepare_model_verify(tmp_path, monkeypatch, model_path=absent)
    manifest.write_text(_model_manifest_yaml(str(absent), 4, "0" * 64), encoding="utf-8")

    result = runner.invoke(
        app, ["model", "verify", str(manifest), "--source", _MODEL_PREP_ID, "--home", str(home)]
    )

    assert result.exit_code == 1
    assert "supersecretvalue" not in result.stderr
    assert "supersecretvalue" not in result.stdout


def test_model_verify_rejects_sensitive_interpolation(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "sekret")  # type: ignore[attr-defined]
    manifest = tmp_path / "model.yaml"
    manifest.write_text(
        _model_manifest_yaml("${AWS_SECRET_ACCESS_KEY}/model.gguf", 4, "0" * 64), encoding="utf-8"
    )

    result = runner.invoke(
        app,
        ["model", "verify", str(manifest), "--source", _MODEL_PREP_ID, "--home", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "sensitive environment interpolation is forbidden" in result.stderr
    assert "sekret" not in result.stderr


def test_model_verify_reports_non_mapping_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "model.yaml"
    manifest.write_text("- one\n- two\n", encoding="utf-8")

    result = runner.invoke(
        app,
        ["model", "verify", str(manifest), "--source", _MODEL_PREP_ID, "--home", str(tmp_path)],
    )

    assert result.exit_code == 1
    assert "model verify failed" in result.stderr

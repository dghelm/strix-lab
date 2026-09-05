from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import _suite_fixtures as fx
import pytest

from strixlab.build_artifacts import ArtifactV1, BuildArtifactsV1, TargetArtifactsV1
from strixlab.build_cache import IdentityEntryV1
from strixlab.build_runtime import (
    BuildRuntimeEnvironment,
    reconstruct_environment,
    resolve_target_artifact,
    resolve_target_executable,
)


class SeamError(RuntimeError):
    pass


def _artifacts_with(
    targets: tuple[TargetArtifactsV1, ...], artifacts: tuple[ArtifactV1, ...]
) -> BuildArtifactsV1:
    return BuildArtifactsV1(
        artifact_set_id="artifact-set-sha256:" + "cd" * 32,
        targets=targets,
        artifacts=artifacts,
        inspections=(),
        capture_tools=(),
        cmake_cache_sha256="cd" * 32,
        compile_commands_sha256=None,
    )


def _target(*, target_type: str = "EXECUTABLE") -> TargetArtifactsV1:
    return TargetArtifactsV1(
        name="llama-bench",
        target_id="target",
        target_type=target_type,
        artifacts=("bin/llama-bench",),
    )


def _artifact(
    *,
    path: str = "bin/llama-bench",
    kind: str = "elf",
    elf_type: str | None = "ET_EXEC",
    runtime_dependency: bool = False,
    sha256: str = "cd" * 32,
) -> ArtifactV1:
    return ArtifactV1.model_validate(
        {
            "path": path,
            "kind": kind,
            "elf_type": elf_type,
            "mode": 0o755,
            "size_bytes": 4,
            "sha256": sha256,
            "targets": ("llama-bench",),
            "runtime_dependency": runtime_dependency,
        }
    )


def test_public_target_resolution_preserves_relative_path_digest_and_no_io(
    tmp_path: Path,
) -> None:
    artifacts = fx._artifacts()
    missing_root = tmp_path / "not-created"

    assert resolve_target_artifact(artifacts, "llama-bench", error=SeamError) == (
        "bin/llama-bench",
        "cd" * 32,
    )
    assert resolve_target_executable(artifacts, "llama-bench", missing_root, error=SeamError) == (
        str(missing_root / "bin/llama-bench"),
        "cd" * 32,
    )
    assert not missing_root.exists()


@pytest.mark.parametrize(
    ("targets", "artifacts", "message"),
    [
        ((), (), "build target is missing or ambiguous: llama-bench"),
        (
            (_target(), _target()),
            (_artifact(),),
            "build target is missing or ambiguous: llama-bench",
        ),
        (
            (_target(target_type="SHARED_LIBRARY"),),
            (_artifact(),),
            "build target is not an executable: llama-bench",
        ),
        (
            (_target(),),
            (_artifact(kind="archive", elf_type=None),),
            "expected exactly one executable artifact for target: llama-bench",
        ),
        (
            (_target(),),
            (_artifact(elf_type="ET_REL"),),
            "expected exactly one executable artifact for target: llama-bench",
        ),
        (
            (_target(),),
            (_artifact(runtime_dependency=True),),
            "expected exactly one executable artifact for target: llama-bench",
        ),
        (
            (_target(),),
            (_artifact(), _artifact(path="bin/other", sha256="ab" * 32)),
            "expected exactly one executable artifact for target: llama-bench",
        ),
        (
            (_target(),),
            (_artifact(path="../escape"),),
            "build artifact escapes the leased root: llama-bench",
        ),
        (
            (_target(),),
            (_artifact(path="/absolute"),),
            "build artifact escapes the leased root: llama-bench",
        ),
    ],
)
def test_target_validation_uses_required_error_factory_with_exact_messages(
    targets: tuple[TargetArtifactsV1, ...],
    artifacts: tuple[ArtifactV1, ...],
    message: str,
) -> None:
    with pytest.raises(SeamError, match=f"^{message}$"):
        resolve_target_artifact(_artifacts_with(targets, artifacts), "llama-bench", error=SeamError)


def test_target_error_factory_is_required() -> None:
    with pytest.raises(TypeError, match="error"):
        resolve_target_artifact(fx._artifacts(), "llama-bench")  # type: ignore[call-arg]


def _env_without(name: str) -> tuple[IdentityEntryV1, ...]:
    return tuple(entry for entry in fx.default_environment() if entry.name != name)


def _env_with(name: str, value: str) -> tuple[IdentityEntryV1, ...]:
    return _env_without(name) + (IdentityEntryV1(name=name, value=value),)


def test_public_environment_reconstruction_is_canonical_mutable_and_caller_owned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AMBIENT_SENTINEL", "not-inherited")
    root = tmp_path / "leased-root"
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)

    runtime = reconstruct_environment(
        fx.canonical_record(environment=fx.default_environment()),
        root,
        scratch,
        error=SeamError,
    )

    assert isinstance(runtime, BuildRuntimeEnvironment)
    assert runtime.cwd == scratch / "tmp"
    assert runtime.scratch_root == scratch
    assert runtime.environment["HOME"] == str(scratch / "home")
    assert runtime.environment["TMPDIR"] == str(scratch / "tmp")
    assert runtime.environment["LD_LIBRARY_PATH"] == f"{root}/lib{os.pathsep}/usr/lib"
    assert runtime.environment["PATH"] == "/opt/rocm-10/bin:/usr/bin"
    assert "AMBIENT_SENTINEL" not in runtime.environment
    assert (scratch / "home").stat().st_mode & 0o777 == 0o700
    assert (scratch / "tmp").stat().st_mode & 0o777 == 0o700
    runtime.environment["MUTABLE"] = "yes"
    assert runtime.environment["MUTABLE"] == "yes"
    assert scratch.exists()
    with pytest.raises(FrozenInstanceError):
        runtime.cwd = root  # type: ignore[misc]
    assert not hasattr(runtime, "__dict__")


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        (_env_with("EXTRA", "{SOURCE_ROOT}/x"), "unknown placeholder"),
        (_env_with("EXTRA", "{BUILD_HOME}/x"), "unknown placeholder"),
        (_env_with("EXTRA", "{UNKNOWN}"), "unknown placeholder"),
        (_env_with("PATH", "/foo/{BUILD_ROOT}/bar"), "unknown placeholder"),
        (_env_with("EXTRA", "{BUILD_ROOT}/{UNKNOWN}"), "unexpected placeholder"),
        (_env_without("HOME"), "missing HOME"),
        (_env_without("TMPDIR"), "missing TMPDIR"),
        (_env_with("LANG", "en_US"), "unexpected LANG"),
        (_env_with("LC_ALL", "en_US"), "unexpected LC_ALL"),
        (_env_with("TZ", "PST"), "unexpected TZ"),
        (
            fx.default_environment() + (IdentityEntryV1(name="PATH", value="/dup"),),
            "duplicate name: 'PATH'",
        ),
        (_env_with("bad name", "x"), "invalid name: 'bad name'"),
        (_env_with("EXTRA", "x\x00y"), "contains a NUL byte"),
    ],
)
def test_environment_validation_uses_error_factory_and_preserves_creation_order(
    tmp_path: Path,
    environment: tuple[IdentityEntryV1, ...],
    message: str,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with pytest.raises(SeamError, match=message):
        reconstruct_environment(
            fx.canonical_record(environment=environment),
            tmp_path / "root",
            scratch,
            error=SeamError,
        )
    assert (scratch / "home").is_dir()
    assert (scratch / "tmp").is_dir()


def test_environment_filesystem_exceptions_are_not_wrapped(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "home").mkdir()

    with pytest.raises(FileExistsError):
        reconstruct_environment(
            fx.canonical_record(),
            tmp_path / "root",
            scratch,
            error=SeamError,
        )

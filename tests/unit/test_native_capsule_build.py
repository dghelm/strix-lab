"""Real host source/build/lease coverage for the fixed native fixture policy."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_sources import (
    GitRepository,
    create_repository,
    git,
    source_lock,
)
from test_sources import allow_root_source_tests as allow_root_source_tests

import strixlab.cmake_build as build
from strixlab.build_artifacts import BuildArtifactError
from strixlab.build_cache import BuildCacheError, cleanup_build, inspect_build, lease_build
from strixlab.build_identity import recipe_id
from strixlab.build_runtime import resolve_target_executable
from strixlab.capsule_runs import CapsuleRunError, _bind_build
from strixlab.manifests import BuildProfileV1, CapsuleManifestV1
from strixlab.process import ProcessOutcome, run_process
from strixlab.sources import SourcePreparation, lease_source, prepare_source

_TARGET = "topk_capsule_host_test"
_ADAPTER = "strixlab_native"
_CMAKE = """\
cmake_minimum_required(VERSION 3.20)
project(native_host_fixture LANGUAGES C CXX)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(STRIXLAB_NATIVE_BUILD_COMMIT "" CACHE STRING "source commit")
set(STRIXLAB_NATIVE_BUILD_NUMBER "" CACHE STRING "source build number")
add_library(topk_host_support SHARED support.c)
add_executable(topk_capsule_host_test main.cpp)
target_compile_features(topk_capsule_host_test PRIVATE cxx_std_17)
target_link_libraries(topk_capsule_host_test PRIVATE topk_host_support)
"""


def _profile() -> BuildProfileV1:
    tools: dict[str, str] = {}
    for role, command in (
        ("cmake", "cmake"),
        ("ninja", "ninja"),
        ("c_compiler", "cc"),
        ("cxx_compiler", "c++"),
    ):
        path = shutil.which(command)
        assert path is not None, f"native host lifecycle tests require {command}"
        tools[role] = path
    directories = sorted({str(Path(value).parent) for value in tools.values()} | {"/usr/bin"})
    return BuildProfileV1.model_validate(
        {
            "schema_version": 1,
            "id": "native-host-build",
            "source": "native-source",
            "generator": "Ninja",
            "build_type": "Release",
            "toolchain": {
                "mode": "host",
                "prefixes": {f"tools_{index}": value for index, value in enumerate(directories)},
                **tools,
                "hip_compiler": None,
                "rocm_prefix": None,
                "path": directories,
            },
            "execution": {
                "jobs": 2,
                "timeouts": {
                    "discovery_seconds": 30,
                    "configure_seconds": 60,
                    "build_seconds": 60,
                    "inspection_seconds": 30,
                    "capability_seconds": 30,
                },
            },
            "environment": {"path_lists": {}, "literals": {"SOURCE_DATE_EPOCH": "0"}},
            "cmake": {},
            "targets": [_TARGET],
        }
    )


def _source(tmp_path: Path, *, defect: str | None = None) -> tuple[Path, SourcePreparation]:
    repository = create_repository(tmp_path, "upstream")
    native = repository.path / "native" / "topk"
    native.mkdir(parents=True)
    (native / "CMakeLists.txt").write_text(_CMAKE)
    (native / "support.c").write_text("int fixture_value(void) { return 7; }\n")
    (native / "main.cpp").write_text(
        '#include <iostream>\nextern "C" int fixture_value(void);\n'
        'int main() { std::cout << fixture_value() << "\\n"; }\n'
    )
    if defect == "missing":
        (native / "CMakeLists.txt").unlink()
    elif defect in {"native-link", "topk-link"}:
        directory = native.parent if defect == "native-link" else native
        alternate = directory.with_name(directory.name + "-elsewhere")
        directory.rename(alternate)
        directory.symlink_to(alternate.name, target_is_directory=True)
    elif defect == "cmake-link":
        (native / "CMakeLists.txt").rename(native / "OtherCMake.txt")
        (native / "CMakeLists.txt").symlink_to("OtherCMake.txt")
    git(repository.path, "add", ".")
    git(repository.path, "commit", "-m", "native host fixture")
    repository = GitRepository(repository.path, git(repository.path, "rev-parse", "HEAD"))
    lock = source_lock(repository, source_id="native-source").model_copy(
        update={"adapter": _ADAPTER}
    )
    home = tmp_path / "home"
    return home, prepare_source(lock, home=home)


def _execute(home: Path, source: SourcePreparation, **kwargs: Any) -> build.CMakeBuildResult:
    return build.execute_cmake_build(
        source.evidence.preparation_id, _profile(), home=home, **kwargs
    )


def test_real_native_lifecycle_leases_closure_cache_and_host_admission(tmp_path: Path) -> None:
    home, source = _source(tmp_path)
    result = _execute(home, source)
    assert result.execution_class == "built"
    assert result.recipe_id == recipe_id(source.evidence.candidate_id, _ADAPTER, _profile())
    assert result.recipe_id != recipe_id(source.evidence.candidate_id, "llama_cpp", _profile())
    assert "hello.txt" in {entry.path for entry in result.snapshot.manifest.entries}
    cache = build.parse_cmake_cache((result.build_root / "CMakeCache.txt").read_bytes())
    assert cache["CMAKE_HOME_DIRECTORY"] == str(result.snapshot.source / "native" / "topk")
    assert cache["STRIXLAB_NATIVE_BUILD_COMMIT"] == source.evidence.base_commit
    assert cache["STRIXLAB_NATIVE_BUILD_NUMBER"] == "0"
    assert not any(key.startswith(("LLAMA_", "GGML_")) for key in cache)
    assert not any(entry.name in {"gfx_targets", "hip_compiler"} for entry in result.selections)
    assert {"c_compiler", "cxx_compiler", "linker", "archiver"} <= {
        entry.role for entry in result.tools
    }
    assert result.artifacts.compile_commands_sha256 is not None
    assert any(value.runtime_dependency for value in result.artifacts.artifacts)
    with lease_build(result.build_id, home=home) as lease:
        executable, digest = resolve_target_executable(
            lease.canonical.artifacts, _TARGET, lease.root, error=RuntimeError
        )
        assert hashlib.sha256(Path(executable).read_bytes()).hexdigest() == digest
        process = run_process(
            (executable,), cwd=lease.root, timeout=10, inherit_env=False, base_env={}
        )
        assert process.outcome == ProcessOutcome.EXITED
        assert process.returncode == 0 and process.stdout == "7\n"
        lease.verify()
        manifest = CapsuleManifestV1.model_validate(
            {
                "schema_version": 1,
                "id": "host-test",
                "candidate": "host-test",
                "machine": "host-test",
                "build": {
                    "source_id": source.evidence.source_id,
                    "source_commit": source.evidence.base_commit,
                    "toolchain_mode": "host",
                    "gfx_target": "gfx1151",
                    "target": _TARGET,
                },
                "contract": {
                    "protocol": "native-capsule-v1",
                    "scenario_sha256": "1" * 64,
                    "comparison": {
                        "policy": "paired-latency-log-bootstrap-v1",
                        "protected_regression_bps": None,
                        "permitted_arm_differences": ["candidate-id"],
                    },
                },
                "timeouts": {
                    "describe_seconds": 1,
                    "correctness_seconds": 1,
                    "benchmark_seconds": 1,
                },
            }
        )
        with pytest.raises(CapsuleRunError, match="does not record a gfx target"):
            _bind_build(lease, manifest)
    repeated = _execute(home, source)
    assert repeated.execution_class == "cache-hit"
    assert repeated.build_id == result.build_id
    assert repeated.canonical_record_sha256 == result.canonical_record_sha256
    cleanup_build(result.build_id, home=home)
    rehydrated = _execute(home, source)
    assert rehydrated.execution_class == "rehydrated"
    assert rehydrated.build_id == result.build_id
    assert inspect_build(result.build_id, home=home).canonical is not None


@pytest.mark.parametrize(
    "mutation", ["target", "extra-target", "mode", "adapter", "reserved", "gfx"]
)
def test_native_authorization_fails_before_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    home, source = _source(tmp_path)
    profile = _profile()
    if mutation in {"target", "extra-target"}:
        profile = profile.model_copy(
            update={
                "targets": ["llama-bench"] if mutation == "target" else [_TARGET, "llama-bench"]
            }
        )
    elif mutation == "mode":
        profile = profile.model_copy(
            update={"toolchain": profile.toolchain.model_copy(update={"mode": "rocm"})}
        )
    elif mutation in {"reserved", "gfx"}:
        key = "STRIXLAB_NATIVE_BUILD_COMMIT" if mutation == "reserved" else "AMDGPU_TARGETS"
        profile = profile.model_copy(update={"cmake": {key: "forged"}})
    monkeypatch.setattr(
        build, "begin_build_attempt", lambda *a, **k: pytest.fail("allocated attempt")
    )
    with lease_source(source.evidence.preparation_id, home=home) as lease:
        if mutation == "adapter":
            lease = replace(
                lease, evidence=lease.evidence.model_copy(update={"adapter": "unknown_native"})
            )
        with pytest.raises(build.CMakeBuildError):
            build._execute_leased_build(lease, profile, home=home)


@pytest.mark.parametrize("defect", ["missing", "native-link", "topk-link", "cmake-link"])
def test_fixed_source_directory_fails_closed(tmp_path: Path, defect: str) -> None:
    home, source = _source(tmp_path, defect=defect)
    with pytest.raises((build.CMakeBuildError, OSError)):
        _execute(home, source)


@pytest.mark.parametrize(
    "key,value",
    [
        ("STRIXLAB_NATIVE_BUILD_COMMIT", "wrong"),
        ("STRIXLAB_NATIVE_BUILD_NUMBER", "1"),
        ("CMAKE_HOME_DIRECTORY", "/tmp/wrong-source"),
        ("CMAKE_C_COMPILER", "/usr/bin/false"),
        ("CMAKE_HIP_ARCHITECTURES", "gfx1151"),
        ("LLAMA_BUILD_COMMIT", "forged"),
    ],
)
def test_actual_configure_selection_drift_is_rejected(tmp_path: Path, key: str, value: str) -> None:
    home, source = _source(tmp_path)

    def runner(argv: Any, **kwargs: Any) -> Any:
        result = run_process(argv, **kwargs)
        if "-B" in argv:
            cache_path = Path(argv[argv.index("-B") + 1]) / "CMakeCache.txt"
            _replace_cache(cache_path, key, value)
        return result

    with pytest.raises(build.CMakeBuildError):
        _execute(home, source, runner=runner)


def _replace_cache(path: Path, key: str, value: str) -> None:
    lines = [line for line in path.read_text().splitlines() if not line.startswith(key + ":")]
    path.chmod(0o600)
    path.write_text("\n".join([*lines, f"{key}:STRING={value}", ""]))


@pytest.mark.parametrize("mutation", ["version", "home", "compiler"])
def test_native_selection_revalidated_after_build(tmp_path: Path, mutation: str) -> None:
    home, source = _source(tmp_path)
    key, value = {
        "version": ("STRIXLAB_NATIVE_BUILD_NUMBER", "9"),
        "home": ("CMAKE_HOME_DIRECTORY", "/tmp/wrong-source"),
        "compiler": ("CMAKE_CXX_COMPILER", "/usr/bin/false"),
    }[mutation]

    def runner(argv: Any, **kwargs: Any) -> Any:
        result = run_process(argv, **kwargs)
        if "--build" in argv:
            _replace_cache(Path(argv[argv.index("--build") + 1]) / "CMakeCache.txt", key, value)
        return result

    with pytest.raises(build.CMakeBuildError):
        _execute(home, source, runner=runner)


@pytest.mark.parametrize("mutation", ["executable", "dependency", "cache", "source"])
def test_real_cache_and_lease_reject_mutation(tmp_path: Path, mutation: str) -> None:
    home, source = _source(tmp_path)
    result = _execute(home, source)
    if mutation == "cache":
        _replace_cache(result.build_root / "CMakeCache.txt", "STRIXLAB_NATIVE_BUILD_NUMBER", "9")
    elif mutation == "source":
        path = source.worktree / "hello.txt"
        path.chmod(0o600)
        path.write_text("drift\n")
    else:
        artifact = next(
            value
            for value in result.artifacts.artifacts
            if value.runtime_dependency == (mutation == "dependency")
        )
        path = result.build_root / artifact.path
        with lease_build(result.build_id, home=home) as lease:
            path.chmod(0o700)
            path.write_bytes(path.read_bytes() + b"drift")
            with pytest.raises((BuildCacheError, BuildArtifactError)):
                lease.verify()
    with pytest.raises((build.CMakeBuildError, BuildCacheError, BuildArtifactError, RuntimeError)):
        _execute(home, source)

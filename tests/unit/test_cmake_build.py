from __future__ import annotations

import json
import sys
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import strixlab.build_snapshot as snapshot_module
import strixlab.builds as builds_module
import strixlab.cmake_build as cmake_module
from strixlab.build_snapshot import SnapshotError
from strixlab.builds import AttemptOutcome, AttemptState, BuildAttemptSession
from strixlab.cmake_build import (
    CMakeBuildError,
    configure_command,
    parse_cmake_cache,
    probe_tools,
    selections_from_cache,
)
from strixlab.manifests import BuildProfileV1
from strixlab.process import ProcessResult, run_process
from strixlab.sources import SourceLease

_CANDIDATE = "candidate-sha256:" + "12" * 32
_CONTENT = "content-tree-sha256:" + "34" * 32


def _write_tool(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _toolchain(tmp_path: Path, *, drift: bool = False) -> Path:
    tools = tmp_path / "tools"
    tools.mkdir()
    common = "#!/bin/sh\nprintf '%s\\n' 'fake tool 1.0'\n"
    for name in ("ninja", "cc", "c++", "hipcc", "ld", "ld-drift", "ar"):
        _write_tool(tools / name, common)
    cmake = f"""#!{sys.executable}
import pathlib
import sys

args = sys.argv[1:]
if args == ["--version"]:
    print("cmake version 4.0")
    raise SystemExit(0)
if args and args[0] == "--build":
    root = pathlib.Path(args[1])
    targets = args[args.index("--target") + 1:args.index("--parallel")]
    output = root / "bin"
    output.mkdir(exist_ok=True)
    for target in targets:
        (output / target).write_text("binary:" + target)
    print("built", *targets)
    raise SystemExit(0)

build = pathlib.Path(args[args.index("-B") + 1])
defines = {{}}
for value in args:
    if value.startswith("-D"):
        key, item = value[2:].split("=", 1)
        defines[key] = item
build.mkdir(parents=True, exist_ok=True)
linker = {str(tools / "ld")!r}
if {drift!r} and "build-sha256" in str(build):
    linker += "-drift"
cache = {{
    "CMAKE_GENERATOR": "Ninja",
    "CMAKE_MAKE_PROGRAM": defines["CMAKE_MAKE_PROGRAM"],
    "CMAKE_C_COMPILER": defines["CMAKE_C_COMPILER"],
    "CMAKE_CXX_COMPILER": defines["CMAKE_CXX_COMPILER"],
    "CMAKE_HIP_COMPILER": defines["CMAKE_HIP_COMPILER"],
    "CMAKE_HIP_COMPILER_ROCM_ROOT": {str(tools)!r},
    "CMAKE_LINKER": linker,
    "CMAKE_AR": {str(tools / "ar")!r},
    "AMDGPU_TARGETS": defines["AMDGPU_TARGETS"],
    "GGML_BUILD_COMMIT": defines["GGML_BUILD_COMMIT"],
    "GGML_BUILD_NUMBER": defines["GGML_BUILD_NUMBER"],
    "LLAMA_BUILD_COMMIT": defines["LLAMA_BUILD_COMMIT"],
    "LLAMA_BUILD_NUMBER": defines["LLAMA_BUILD_NUMBER"],
}}
cache_lines = [f"{{key}}:STRING={{value}}\\n" for key, value in cache.items()]
(build / "CMakeCache.txt").write_text("".join(cache_lines))
(build / "build.ninja").write_text("# fake\\n")
print("configured", build)
"""
    _write_tool(tools / "cmake", cmake)
    return tools


def _profile(build_value: dict[str, Any], tools: Path) -> BuildProfileV1:
    value = deepcopy(build_value)
    value["toolchain"] = {
        "mode": "rocm",
        "prefixes": {"rocm": str(tools)},
        "cmake": str(tools / "cmake"),
        "ninja": str(tools / "ninja"),
        "c_compiler": str(tools / "cc"),
        "cxx_compiler": str(tools / "c++"),
        "hip_compiler": str(tools / "hipcc"),
        "rocm_prefix": str(tools),
        "path": [str(tools)],
    }
    value["environment"] = {
        "path_lists": {"ROCM_PATH": str(tools)},
        "literals": {"SOURCE_DATE_EPOCH": "0"},
    }
    return BuildProfileV1.model_validate(value)


def _preparation(
    tmp_path: Path,
    *,
    adapter: str = "llama_cpp",
    verify: Callable[[], None] | None = None,
    patched: bool = False,
) -> SourceLease:
    source = tmp_path / "source"
    source.mkdir()
    (source / "CMakeLists.txt").write_text("project(fake)\n", encoding="utf-8")
    (source / ".git").write_text("gitdir: authenticated-admin\n", encoding="utf-8")
    evidence_value = {
        "schema_version": 2,
        "preparation_id": "prep-test-source",
        "source_id": "strix-llama",
        "adapter": adapter,
        "base_commit": "ab" * 20,
        "candidate_id": _CANDIDATE,
        "content_tree_id": _CONTENT,
        "diff_size_bytes": 1 if patched else 0,
        "status": ("1 M. N... 100644 100644 100644 file.txt",) if patched else (),
    }
    evidence = SimpleNamespace(
        **evidence_value,
        model_dump=lambda mode: evidence_value,
    )
    return SourceLease(
        evidence_value["preparation_id"],
        evidence_value["source_id"],
        source,
        tmp_path / "source-record",
        cast(Any, evidence),
        verify or (lambda: None),
    )


def execute_cmake_build(
    preparation: SourceLease,
    profile: BuildProfileV1,
    *,
    home: Path,
    runner: Any = run_process,
):
    return cmake_module._execute_leased_build(
        preparation,
        profile,
        home=home,
        runner=runner,
    )


def test_two_phase_build_uses_stable_snapshot_and_identity_root(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    verification_count = 0

    def verify_source() -> None:
        nonlocal verification_count
        verification_count += 1

    preparation = _preparation(tmp_path, verify=verify_source)
    home = tmp_path / "home"

    result = execute_cmake_build(preparation, profile, home=home)

    assert result.build_id.startswith("build-sha256:")
    assert result.recipe_id.startswith("recipe-sha256:")
    assert not result.snapshot.root.exists()
    assert result.build_root.name == result.build_id
    assert result.build_root.parent == home / "builds" / "materialized"
    assert (result.build_root / ".strixlab-owner.json").is_file()
    assert {path.name for path in (result.build_root / "bin").iterdir()} == set(profile.targets)
    assert (result.build_root / "CMakeCache.txt").is_file()
    assert verification_count == 2
    assert not any(
        path.name == "private" for path in (home / "builds" / "records" / "attempts").rglob("*")
    )
    terminal = json.loads((result.attempt.record / "current.json").read_bytes())
    assert terminal["outcome"] == "success"
    assert json.loads((result.attempt.record / "build/profile.resolved.json").read_bytes()) == (
        profile.model_dump(mode="json")
    )
    environment = json.loads((result.attempt.record / "build/environment.json").read_bytes())
    assert "XDG_CACHE_HOME" not in environment
    assert environment["HOME"].endswith("/private/home")
    configure_cache = (result.attempt.record / "cmake/final-configure-cache.txt").read_text()
    assert f"LLAMA_BUILD_COMMIT:STRING={'ab' * 20}" in configure_cache
    assert "LLAMA_BUILD_NUMBER:STRING=0" in configure_cache


def test_patched_source_version_is_visible_to_cmake(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    result = execute_cmake_build(
        _preparation(tmp_path, patched=True), profile, home=tmp_path / "home"
    )

    configure_cache = (result.attempt.record / "cmake/final-configure-cache.txt").read_text()
    assert f"LLAMA_BUILD_COMMIT:STRING={'ab' * 20}-dirty" in configure_cache
    assert not result.snapshot.root.exists()


def test_probe_final_selection_drift_finalizes_failure_evidence(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    tools = _toolchain(tmp_path, drift=True)
    profile = _profile(build_value, tools)
    preparation = _preparation(tmp_path)
    home = tmp_path / "home"

    with pytest.raises(CMakeBuildError, match="drifted"):
        execute_cmake_build(preparation, profile, home=home)

    records = tuple((home / "builds" / "records" / "attempts").iterdir())
    assert len(records) == 1
    assert (records[0] / "failure.json").is_file()
    materialized = home / "builds" / "materialized"
    assert not list(materialized.rglob("CMakeCache.txt"))
    assert not list(materialized.glob("build-sha256:*"))


def test_final_tool_mutation_after_build_fails_closed(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    preparation = _preparation(tmp_path)
    home = tmp_path / "home"

    def mutating_runner(argv: Any, **kwargs: Any) -> ProcessResult:
        result = run_process(argv, **kwargs)
        if "--build" in tuple(argv):
            linker = tools / "ld"
            linker.chmod(0o755)
            with linker.open("a", encoding="utf-8") as handle:
                handle.write("# mutated after the build\n")
        return result

    with pytest.raises(CMakeBuildError, match="tool observations changed"):
        execute_cmake_build(preparation, profile, home=home, runner=mutating_runner)

    materialized = home / "builds" / "materialized"
    assert not list(materialized.rglob("CMakeCache.txt"))


def test_failure_evidence_backend_error_does_not_mask_build_failure(
    tmp_path: Path, build_value: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = _toolchain(tmp_path, drift=True)
    profile = _profile(build_value, tools)
    preparation = _preparation(tmp_path)
    home = tmp_path / "home"

    original = cmake_module.canonical_json_bytes

    def sabotaged(value: Any) -> bytes:
        if isinstance(value, dict) and value.get("code") == "cmake-build-failed":
            raise RuntimeError("evidence backend is unavailable")
        return original(value)

    monkeypatch.setattr(cmake_module, "canonical_json_bytes", sabotaged)

    with pytest.raises(CMakeBuildError, match="drifted"):
        execute_cmake_build(preparation, profile, home=home)


def test_setup_failure_still_finalizes_failure_evidence(
    tmp_path: Path, build_value: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    preparation = _preparation(tmp_path)
    home = tmp_path / "home"

    def fail_setup(_profile: BuildProfileV1, _root: Path) -> dict[str, str]:
        raise CMakeBuildError("private environment setup failed")

    monkeypatch.setattr(cmake_module, "_private_environment", fail_setup)

    with pytest.raises(CMakeBuildError, match="private environment setup failed"):
        execute_cmake_build(preparation, profile, home=home)

    records = tuple((home / "builds" / "records" / "attempts").iterdir())
    assert len(records) == 1
    assert (records[0] / "failure.json").is_file()
    terminal = json.loads((records[0] / "terminal.json").read_bytes())
    assert terminal["outcome"] == "failed"


def test_finalize_interrupt_preserves_successful_build_root(
    tmp_path: Path, build_value: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    preparation = _preparation(tmp_path)
    home = tmp_path / "home"

    def interrupt_after_success_transition(
        session: BuildAttemptSession,
        outcome: AttemptOutcome,
        *,
        build_id: str | None = None,
    ) -> None:
        assert outcome is AttemptOutcome.SUCCESS
        session.registry = builds_module._transition(
            session.root,
            session.registry,
            AttemptState.FINALIZING,
            changes={"outcome": outcome, "build_id": build_id},
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(BuildAttemptSession, "finalize", interrupt_after_success_transition)

    with pytest.raises(KeyboardInterrupt):
        execute_cmake_build(preparation, profile, home=home)

    build_roots = list((home / "builds" / "materialized").glob("build-sha256:*"))
    assert len(build_roots) == 1
    assert (build_roots[0] / "CMakeCache.txt").is_file()
    records = tuple((home / "builds" / "records" / "attempts").iterdir())
    assert len(records) == 1
    current = json.loads((records[0] / "current.json").read_bytes())
    assert current["outcome"] == "success"
    assert not (records[0] / "failure.json").exists()


def test_missing_snapshot_after_success_is_reported_with_durable_record(
    tmp_path: Path, build_value: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    preparation = _preparation(tmp_path)
    home = tmp_path / "home"
    original_finalize = BuildAttemptSession.finalize

    def finalize_then_remove_snapshot(
        session: BuildAttemptSession,
        outcome: AttemptOutcome,
        *,
        build_id: str | None = None,
    ):
        result = original_finalize(session, outcome, build_id=build_id)
        snapshots = list((home / "builds" / "snapshots").glob("snapshot-sha256:*"))
        assert len(snapshots) == 1
        snapshot_module._remove_destination(snapshots[0])
        return result

    monkeypatch.setattr(BuildAttemptSession, "finalize", finalize_then_remove_snapshot)

    with pytest.raises(SnapshotError, match="disappeared before retirement"):
        execute_cmake_build(preparation, profile, home=home)

    records = tuple((home / "builds" / "records" / "attempts").iterdir())
    assert len(records) == 1
    current = json.loads((records[0] / "current.json").read_bytes())
    assert current["outcome"] == "success"
    assert len(list((home / "builds" / "materialized").glob("build-sha256:*"))) == 1


def test_optional_cache_selections_reject_relative_paths(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    base = {
        "CMAKE_GENERATOR": "Ninja",
        "CMAKE_MAKE_PROGRAM": str(tools / "ninja"),
        "CMAKE_C_COMPILER": str(tools / "cc"),
        "CMAKE_CXX_COMPILER": str(tools / "c++"),
        "CMAKE_LINKER": str(tools / "ld"),
        "CMAKE_AR": str(tools / "ar"),
        "CMAKE_HIP_COMPILER": str(tools / "hipcc"),
        "CMAKE_HIP_COMPILER_ROCM_ROOT": str(tools),
        "AMDGPU_TARGETS": "gfx1151",
    }

    assert selections_from_cache(profile, base)
    with pytest.raises(CMakeBuildError, match="unexpected path"):
        selections_from_cache(profile, {**base, "CMAKE_C_COMPILER": str(tools / "c++")})

    with pytest.raises(CMakeBuildError, match="not supported"):
        selections_from_cache(profile, {**base, "CMAKE_TOOLCHAIN_FILE": "relative/toolchain.cmake"})
    with pytest.raises(CMakeBuildError, match="not supported"):
        selections_from_cache(profile, {**base, "CMAKE_SYSROOT": "/abs/sysroot"})


def test_cache_parser_and_adapter_owned_options_reject_ambiguity(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    with pytest.raises(CMakeBuildError, match="duplicate"):
        parse_cmake_cache(b"A:STRING=one\nA:STRING=two\n")
    with pytest.raises(CMakeBuildError, match="invalid entry"):
        parse_cmake_cache(b"not-a-cache-entry\n")
    with pytest.raises(CMakeBuildError, match="not UTF-8"):
        parse_cmake_cache(b"\xff")

    tools = _toolchain(tmp_path)
    profile_value = deepcopy(build_value)
    profile_value["cmake"]["CMAKE_C_COMPILER"] = "/other/cc"
    profile = _profile(profile_value, tools)
    with pytest.raises(CMakeBuildError, match="adapter-owned"):
        configure_command(
            profile,
            tmp_path / "source",
            tmp_path / "build",
            source_version=cmake_module._SourceVersion("ab" * 20, False),
        )

    profile_value["cmake"].pop("CMAKE_C_COMPILER")
    profile_value["cmake"]["CMAKE_LINKER"] = "/other/ld"
    profile = _profile(profile_value, tools)
    with pytest.raises(CMakeBuildError, match="adapter-owned"):
        configure_command(
            profile,
            tmp_path / "source",
            tmp_path / "build",
            source_version=cmake_module._SourceVersion("ab" * 20, False),
        )

    profile_value["cmake"].pop("CMAKE_LINKER")
    profile_value["cmake"]["LLAMA_BUILD_COMMIT"] = "untrusted"
    profile = _profile(profile_value, tools)
    with pytest.raises(CMakeBuildError, match="adapter-owned"):
        configure_command(
            profile,
            tmp_path / "source",
            tmp_path / "build",
            source_version=cmake_module._SourceVersion("ab" * 20, False),
        )

    with pytest.raises(CMakeBuildError, match="source-version metadata"):
        cmake_module._verify_source_version(
            {
                "GGML_BUILD_COMMIT": "wrong",
                "GGML_BUILD_NUMBER": "0",
                "LLAMA_BUILD_COMMIT": "ab" * 20,
                "LLAMA_BUILD_NUMBER": "0",
            },
            cmake_module._SourceVersion("ab" * 20, False),
        )


def test_tool_discovery_and_cache_selections_fail_closed(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    with pytest.raises(CMakeBuildError, match="PATH discovery"):
        probe_tools(
            profile,
            cwd=tmp_path,
            environment={"PATH": str(empty_path)},
        )

    cache = {
        "CMAKE_GENERATOR": "Ninja",
        "CMAKE_MAKE_PROGRAM": str(tools / "ninja"),
        "CMAKE_C_COMPILER": str(tools / "cc"),
        "CMAKE_CXX_COMPILER": str(tools / "c++"),
        "CMAKE_LINKER": str(tools / "ld"),
        "CMAKE_AR": str(tools / "ar"),
    }
    with pytest.raises(CMakeBuildError, match="gfx target"):
        selections_from_cache(profile, cache)
    with pytest.raises(CMakeBuildError, match="unexpected generator"):
        selections_from_cache(profile, {**cache, "CMAKE_GENERATOR": "Other"})
    with pytest.raises(CMakeBuildError, match="required selection"):
        selections_from_cache(profile, {})


def test_build_rejects_a_profile_for_another_source(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    tools = _toolchain(tmp_path)
    value = deepcopy(build_value)
    value["source"] = "other-source"
    profile = _profile(value, tools)

    with pytest.raises(CMakeBuildError, match="does not match"):
        execute_cmake_build(_preparation(tmp_path), profile, home=tmp_path / "home")


def test_public_build_holds_the_authenticated_source_lease(
    tmp_path: Path, build_value: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    preparation = _preparation(tmp_path)
    home = tmp_path / "home"
    active = False

    @contextmanager
    def authenticated_lease(preparation_id: str, *, home: Path):
        nonlocal active
        assert preparation_id == preparation.preparation_id
        active = True
        try:
            yield preparation
        finally:
            active = False

    def lease_asserting_runner(argv: Any, **kwargs: Any) -> ProcessResult:
        assert active
        return run_process(argv, **kwargs)

    monkeypatch.setattr(cmake_module, "lease_source", authenticated_lease)
    result = cmake_module.execute_cmake_build(
        preparation.preparation_id,
        profile,
        home=home,
        runner=lease_asserting_runner,
    )

    assert result.build_root.name == result.build_id
    assert not active


def test_source_is_revalidated_before_success(tmp_path: Path, build_value: dict[str, Any]) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    verification_count = 0

    def reject_second_verification() -> None:
        nonlocal verification_count
        verification_count += 1
        if verification_count == 2:
            raise CMakeBuildError("authenticated source changed")

    preparation = _preparation(tmp_path, verify=reject_second_verification)
    home = tmp_path / "home"

    with pytest.raises(CMakeBuildError, match="authenticated source changed"):
        execute_cmake_build(preparation, profile, home=home)

    assert verification_count == 2
    assert not list((home / "builds" / "materialized").glob("build-sha256:*"))
    assert not list((home / "builds" / "snapshots").glob("snapshot-sha256:*"))


def test_build_root_owner_tampering_refuses_success_and_deletion(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    preparation = _preparation(tmp_path)
    home = tmp_path / "home"

    def tampering_runner(argv: Any, **kwargs: Any) -> ProcessResult:
        arguments = tuple(argv)
        result = run_process(arguments, **kwargs)
        if "--build" in arguments:
            marker = Path(kwargs["cwd"]) / ".strixlab-owner.json"
            marker.chmod(0o600)
            marker.write_text("{}\n", encoding="utf-8")
            marker.chmod(0o400)
        return result

    with pytest.raises(CMakeBuildError, match="ownership marker is invalid"):
        execute_cmake_build(preparation, profile, home=home, runner=tampering_runner)

    build_roots = list((home / "builds" / "materialized").glob("build-sha256:*"))
    assert len(build_roots) == 1
    assert (build_roots[0] / ".strixlab-owner.json").read_text(encoding="utf-8") == "{}\n"


def test_snapshot_mutation_by_build_fails_before_success(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    preparation = _preparation(tmp_path)
    home = tmp_path / "home"
    snapshot_source: Path | None = None

    def mutating_runner(argv: Any, **kwargs: Any) -> ProcessResult:
        nonlocal snapshot_source
        arguments = tuple(argv)
        result = run_process(arguments, **kwargs)
        if "-S" in arguments:
            snapshot_source = Path(arguments[arguments.index("-S") + 1])
        if "--build" in arguments:
            assert snapshot_source is not None
            payload = snapshot_source / "CMakeLists.txt"
            payload.chmod(0o600)
            payload.write_text("project(corrupt)\n", encoding="utf-8")
        return result

    with pytest.raises(SnapshotError, match="does not match"):
        execute_cmake_build(preparation, profile, home=home, runner=mutating_runner)

    assert not list((home / "builds" / "materialized").glob("build-sha256:*"))


def test_build_authorizes_adapter_and_targets(tmp_path: Path, build_value: dict[str, Any]) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)

    with pytest.raises(CMakeBuildError, match="llama_cpp"):
        execute_cmake_build(
            _preparation(tmp_path, adapter="other"),
            profile,
            home=tmp_path / "home",
        )

    other_root = tmp_path / "other"
    other_root.mkdir()
    value = deepcopy(build_value)
    value["targets"] = ["untrusted-target"]
    unauthorized = _profile(value, tools)
    with pytest.raises(CMakeBuildError, match="unauthorized"):
        execute_cmake_build(
            _preparation(other_root),
            unauthorized,
            home=tmp_path / "other-home",
        )


def test_build_roots_and_cache_reads_reject_symlink_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    materialized = tmp_path / "materialized"
    materialized.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (materialized / "probes").symlink_to(outside, target_is_directory=True)
    with pytest.raises(CMakeBuildError, match="unsafe"):
        cmake_module._prepare_build_root(materialized, materialized / "probes" / "attempt")

    target = tmp_path / "target-cache"
    target.write_text("A:STRING=value\n", encoding="utf-8")
    link = tmp_path / "CMakeCache.txt"
    link.symlink_to(target)
    with pytest.raises(CMakeBuildError, match="safe CMakeCache"):
        cmake_module._read_cache(link)

    monkeypatch.setattr(cmake_module, "_CACHE_LIMIT", 1)
    with pytest.raises(CMakeBuildError, match="size limit"):
        cmake_module._read_cache(target)

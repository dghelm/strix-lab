from __future__ import annotations

import hashlib
import json
import shutil
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
from strixlab.build_artifacts import BuildArtifactError
from strixlab.build_cache import (
    BuildCacheError,
    MaterializationState,
    cache_environment_projection,
    cleanup_build,
    inspect_build,
)
from strixlab.build_snapshot import SnapshotError
from strixlab.builds import (
    AttemptOutcome,
    AttemptState,
    BuildAttemptSession,
    BuildStateError,
    inspect_attempt,
    inspect_recipe,
)
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
    for name in ("ldd", "readelf"):
        selected = shutil.which(name)
        assert selected is not None
        (tools / name).symlink_to(selected)
    cmake = f"""#!{sys.executable}
import hashlib
import json
import pathlib
import shutil
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
    executable = shutil.which("true")
    assert executable is not None
    for target in targets:
        shutil.copy2(executable, output / target)
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
reply = build / ".cmake" / "api" / "v1" / "reply"
reply.mkdir(parents=True, exist_ok=True)
targets = ["llama-bench", "llama-server", "test-backend-ops"]
for target in targets:
    (reply / f"target-{{target}}.json").write_text(json.dumps({{
        "artifacts": [{{"path": f"bin/{{target}}"}}],
        "id": f"{{target}}::@fake",
        "name": target,
        "type": "EXECUTABLE",
    }}))
(reply / "codemodel-v2.json").write_text(json.dumps({{
    "configurations": [{{
        "name": defines["CMAKE_BUILD_TYPE"],
        "targets": [
            {{"id": f"{{target}}::@fake", "jsonFile": f"target-{{target}}.json", "name": target}}
            for target in targets
        ],
    }}],
    "kind": "codemodel",
    "version": {{"major": 2, "minor": 0}},
}}))
response = {{
    "jsonFile": "codemodel-v2.json",
    "kind": "codemodel",
    "version": {{"major": 2, "minor": 0}},
}}
(reply / "index-test.json").write_text(json.dumps({{
    "reply": {{"client-strixlab": {{"query.json": {{"responses": [response]}}}}}}
}}))
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
    record = tmp_path / "source-record"
    record.mkdir(exist_ok=True)
    diff_bytes = b"--- patched diff\n" if patched else b""
    (record / "source.diff").write_bytes(diff_bytes)
    evidence_value: dict[str, Any] = {
        "schema_version": 2,
        "preparation_id": "prep-test-source",
        "source_id": "strix-llama",
        "adapter": adapter,
        "base_commit": "ab" * 20,
        "candidate_id": _CANDIDATE,
        "content_tree_id": _CONTENT,
        "diff_file": "source.diff",
        "diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
        "diff_size_bytes": len(diff_bytes),
        "status": ("1 M. N... 100644 100644 100644 file.txt",) if patched else (),
    }
    evidence = SimpleNamespace(
        **evidence_value,
        patches=(),
        model_dump=lambda mode: {**evidence_value, "patches": []},
    )
    return SourceLease(
        evidence_value["preparation_id"],
        evidence_value["source_id"],
        source,
        record,
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

    with pytest.raises(BuildCacheError, match="stored build cache model is invalid"):
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


def _count_runner() -> tuple[Callable[..., ProcessResult], dict[str, int]]:
    counts = {"build": 0, "configure": 0}

    def runner(argv: Any, **kwargs: Any) -> ProcessResult:
        arguments = tuple(str(value) for value in argv)
        if "--build" in arguments:
            counts["build"] += 1
        elif "-B" in arguments:
            counts["configure"] += 1
        return run_process(argv, **kwargs)

    return runner, counts


def test_second_build_is_a_cache_hit(tmp_path: Path, build_value: dict[str, Any]) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    runner, counts = _count_runner()
    preparation = _preparation(tmp_path)

    first = execute_cmake_build(preparation, profile, home=home, runner=runner)
    assert first.execution_class == "built"
    assert first.canonical_record_sha256.startswith("record-sha256:")
    builds_after_first = counts["build"]

    second = execute_cmake_build(preparation, profile, home=home, runner=runner)
    assert second.execution_class == "cache-hit"
    assert second.build_id == first.build_id
    assert second.canonical_record_sha256 == first.canonical_record_sha256
    assert counts["build"] == builds_after_first  # no final build ran on the hit
    result = json.loads((second.attempt.record / "build/result.json").read_bytes())
    assert result["execution_class"] == "cache-hit"
    assert result["canonical_record_sha256"] == first.canonical_record_sha256


def test_cache_hit_lookup_authenticates_producer_provenance(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    # Finding 3: the stronger provenance check gates cache reuse at lookup, not only
    # public inspection. A missing producer attempt record must fail the second
    # build's HIT lookup closed rather than silently returning a cache hit.
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    preparation = _preparation(tmp_path)

    first = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    shutil.rmtree(first.attempt.record)
    with pytest.raises(BuildStateError, match="producer attempt record is missing"):
        execute_cmake_build(preparation, profile, home=home, runner=run_process)


def test_cache_hit_lookup_rejects_tampered_provenance_evidence(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    # Finding 2: the digest-indexed evidence inventory is authenticated. Tampering a
    # required evidence file (here the canonical artifacts blob the inventory pins)
    # must fail the HIT lookup closed.
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    preparation = _preparation(tmp_path)

    first = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    artifacts_blob = first.attempt.record / "build" / "artifacts.json"
    tampered = json.loads(artifacts_blob.read_bytes())
    tampered["cmake_cache_sha256"] = "ab" * 32
    artifacts_blob.chmod(0o600)
    artifacts_blob.write_bytes(json.dumps(tampered).encode())
    with pytest.raises(BuildStateError, match="verification failed"):
        execute_cmake_build(preparation, profile, home=home, runner=run_process)


def _attestation_path(home: Path, build_id: str) -> Path:
    suffix = build_id.removeprefix("build-sha256:")
    return home / "builds" / "records" / "success" / f"{suffix}.attestation.json"


def _replace_record_file_self_consistently(record_dir: Path, relative: str, content: bytes) -> None:
    """Replace one immutable record file AND fix its manifest entry, so the record
    stays internally self-consistent but its content-address changes."""

    manifest_path = record_dir / "record-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    entry = next(item for item in manifest["files"] if item["path"] == relative)
    target = record_dir / relative
    target.chmod(0o600)
    target.write_bytes(content)
    target.chmod(entry["mode"])
    entry["sha256"] = hashlib.sha256(content).hexdigest()
    entry["size_bytes"] = len(content)
    manifest_path.chmod(0o600)
    manifest_path.write_bytes(json.dumps(manifest).encode())
    manifest_path.chmod(0o400)


def test_cache_hit_rejects_self_consistent_producer_record_replacement(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    # A wholesale, internally self-consistent replacement of the producer evidence
    # (and its manifest) changes the record content-address, so the attestation's
    # producer digest anchor fails the next HIT closed.
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    preparation = _preparation(tmp_path)
    first = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    _replace_record_file_self_consistently(
        first.attempt.record, "build/result.json", b'{"schema_version":1,"tampered":true}\n'
    )
    with pytest.raises(BuildCacheError, match="producer record digest diverged"):
        execute_cmake_build(preparation, profile, home=home, runner=run_process)


def test_cache_hit_rejects_self_consistent_attestor_record_replacement(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    # In a crash-forward recovery the attestor is a distinct attempt; a self-
    # consistent replacement of that recovery attestor record also fails closed.
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    preparation = _preparation(tmp_path)
    first = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    _attestation_path(home, first.build_id).unlink()  # force a recovery attestation
    recovered = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    assert recovered.execution_class == "recovered"
    assert recovered.attempt.attempt_id != first.attempt.attempt_id
    _replace_record_file_self_consistently(
        recovered.attempt.record, "build/result.json", b'{"schema_version":1,"tampered":true}\n'
    )
    with pytest.raises(BuildCacheError, match="attestor record digest diverged"):
        execute_cmake_build(preparation, profile, home=home, runner=run_process)


def test_recovery_attestation_rejects_self_consistent_producer_replacement(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    # Pre-attestation crash-forward window: before any attestation binds the producer
    # digest, a self-consistently replaced producer record must still fail closed —
    # the recovery attestation anchors the producer digest to the authoritative
    # recipe-index entry (read under the already-held recipe lock).
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    preparation = _preparation(tmp_path)
    first = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    _attestation_path(home, first.build_id).unlink()  # force a recovery attestation
    _replace_record_file_self_consistently(
        first.attempt.record, "build/result.json", b'{"schema_version":1,"tampered":true}\n'
    )
    with pytest.raises(BuildCacheError, match="recipe-index anchor"):
        execute_cmake_build(preparation, profile, home=home, runner=run_process)


def test_built_build_publishes_success_attestation(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    import strixlab.build_cache as cache_module

    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    result = execute_cmake_build(_preparation(tmp_path), profile, home=home, runner=run_process)
    # A normal build publishes a "built" attestation naming the producer itself as
    # the finalized SUCCESS attestor, binding the canonical digest.
    attestation = cache_module._read_model(
        _attestation_path(home, result.build_id), cache_module.BuildAttestationV1
    )
    assert attestation.execution_class == "built"
    assert attestation.attestor_attempt_id == result.attempt.attempt_id
    assert attestation.canonical_record_sha256 == result.canonical_record_sha256


def test_cache_hit_rejects_tampered_attestation(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    preparation = _preparation(tmp_path)
    first = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    path = _attestation_path(home, first.build_id)
    data = json.loads(path.read_bytes())
    data["canonical_record_sha256"] = "record-sha256:" + "00" * 32
    path.chmod(0o600)
    path.write_bytes(json.dumps(data).encode())
    # The HIT reuse boundary authenticates the attestation against the canonical.
    with pytest.raises(BuildCacheError, match="attestation does not bind"):
        execute_cmake_build(preparation, profile, home=home, runner=run_process)


def test_cache_hit_rejects_non_success_attestor(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    import strixlab.build_cache as cache_module

    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    preparation = _preparation(tmp_path)
    first = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    # Re-point the attestation at a non-existent attestor attempt: no finalized
    # SUCCESS record backs it, so the reuse boundary fails closed.
    path = _attestation_path(home, first.build_id)
    data = json.loads(path.read_bytes())
    data["attestor_attempt_id"] = "attempt-" + "0" * 24 + "-" + "e" * 32
    payload = cache_module.canonical_json_bytes(data)
    path.chmod(0o600)
    path.write_bytes(payload)
    with pytest.raises(BuildStateError, match="producer attempt record is missing"):
        execute_cmake_build(preparation, profile, home=home, runner=run_process)


def test_missing_attestation_triggers_recovery_attestation(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    preparation = _preparation(tmp_path)
    first = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    path = _attestation_path(home, first.build_id)
    path.unlink()  # simulate a crash between publish and the attestation boundary
    # The next attempt must not return an ordinary HIT; it recovers by re-verifying
    # the root and publishing a recovery attestation.
    recovered = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    assert recovered.execution_class == "recovered"
    assert path.exists()
    # Now genuinely attested, the following build is an ordinary cache hit.
    third = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    assert third.execution_class == "cache-hit"


def test_cleaned_build_rehydrates_into_the_same_root(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:

    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    preparation = _preparation(tmp_path)

    first = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    cleaned = cleanup_build(first.build_id, home=home)
    assert cleaned.state is MaterializationState.CLEANED
    assert not first.build_root.exists()
    assert inspect_build(first.build_id, home=home).state is MaterializationState.CLEANED

    second = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    assert second.execution_class == "rehydrated"
    assert second.build_id == first.build_id
    assert second.build_root == first.build_root
    assert second.build_root.exists()
    assert (second.build_root / "CMakeCache.txt").is_file()
    inspected = inspect_build(first.build_id, home=home)
    assert inspected.state is MaterializationState.PRESENT
    # cache-hit again after rehydration
    third = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    assert third.execution_class == "cache-hit"


def test_cache_hit_integrity_failure_when_artifact_is_tampered(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:

    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    preparation = _preparation(tmp_path)

    first = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    tampered = first.build_root / "bin" / "llama-bench"
    tampered.chmod(0o755)
    with tampered.open("ab") as handle:  # keep the ELF magic, change the hash
        handle.write(b"\x00tampered")

    with pytest.raises(BuildArtifactError, match="build artifact changed"):
        execute_cmake_build(preparation, profile, home=home, runner=run_process)


def test_cache_environment_projection_rejects_owned_leak(
    tmp_path: Path,
) -> None:

    home = tmp_path / "home"
    (home / "builds").mkdir(parents=True)
    source_root = home / "builds" / "snapshots" / "snap"
    build_root = home / "builds" / "materialized" / "b"
    build_home = home / "builds" / "attempts" / "a" / "private" / "home"
    build_tmp = home / "builds" / "attempts" / "a" / "private" / "tmp"

    projected = cache_environment_projection(
        {"HOME": str(build_home), "TMPDIR": str(build_tmp), "PATH": "/usr/bin"},
        home=home,
        source_root=source_root,
        build_root=build_root,
        build_home=build_home,
        build_tmp=build_tmp,
    )
    values = {entry.name: entry.value for entry in projected}
    assert values["HOME"] == "{BUILD_HOME}"
    assert values["TMPDIR"] == "{BUILD_TMP}"
    assert values["PATH"] == "/usr/bin"

    with pytest.raises(BuildCacheError, match="leaks an owned StrixLab path"):
        cache_environment_projection(
            {"STRAY": str(home / "builds" / "materialized" / "other")},
            home=home,
            source_root=source_root,
            build_root=build_root,
            build_home=build_home,
            build_tmp=build_tmp,
        )


def test_interrupted_publishing_recovers_forward_from_staging(
    tmp_path: Path, build_value: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import strixlab.build_cache as cache_module

    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    preparation = _preparation(tmp_path)
    original = cache_module._publish_canonical

    def failing(*args: Any, **kwargs: Any) -> None:
        raise cache_module.BuildCacheError("injected publication failure")

    monkeypatch.setattr(cache_module, "_publish_canonical", failing)
    with pytest.raises(cache_module.BuildCacheError, match="injected publication failure"):
        execute_cmake_build(preparation, profile, home=home)

    build_roots = list((home / "builds" / "materialized").glob("build-sha256:*"))
    assert len(build_roots) == 1  # the materialized root is preserved for forward recovery

    monkeypatch.setattr(cache_module, "_publish_canonical", original)
    recovered = execute_cmake_build(preparation, profile, home=home)
    # A crash-forward completion is not a false cache-hit: the recovering attempt
    # completes publication, finalizes SUCCESS, and publishes an explicit recovery
    # attestation (not a producer SUCCESS claim) before the root is reusable.
    assert recovered.execution_class == "recovered"
    assert recovered.build_root == build_roots[0]
    suffix = recovered.build_id.removeprefix("build-sha256:")
    attestation = cache_module._read_model(
        home / "builds" / "records" / "success" / f"{suffix}.attestation.json",
        cache_module.BuildAttestationV1,
    )
    assert attestation.execution_class == "recovered"
    assert attestation.attestor_attempt_id == recovered.attempt.attempt_id
    assert attestation.canonical_record_sha256 == recovered.canonical_record_sha256

    # Now that the recovered build is attested, the next build is a genuine hit.
    again = execute_cmake_build(preparation, profile, home=home)
    assert again.execution_class == "cache-hit"


def test_present_build_with_missing_root_is_integrity_failure(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:

    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    preparation = _preparation(tmp_path)

    first = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    shutil.rmtree(first.build_root)  # a PRESENT journal without its owned root

    with pytest.raises(BuildCacheError):
        execute_cmake_build(preparation, profile, home=home, runner=run_process)


def test_inspect_and_cleanup_are_disjoint_and_idempotent(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:

    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    preparation = _preparation(tmp_path)

    first = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    inspected = inspect_build(first.build_id, home=home)
    assert inspected.state is MaterializationState.PRESENT
    assert inspected.canonical_record_sha256 == first.canonical_record_sha256

    cleaned = cleanup_build(first.build_id, home=home)
    assert cleaned.state is MaterializationState.CLEANED
    again = cleanup_build(first.build_id, home=home)  # idempotent
    assert again.state is MaterializationState.CLEANED

    missing = "build-sha256:" + "0" * 64
    with pytest.raises(BuildCacheError):
        inspect_build(missing, home=home)
    with pytest.raises(ValueError, match="invalid machine-local build ID"):
        inspect_build("not-a-build-id", home=home)


def test_inspect_attempt_and_recipe_after_build(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:

    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    result = execute_cmake_build(_preparation(tmp_path), profile, home=home, runner=run_process)

    registry = inspect_attempt(result.attempt.attempt_id, home=home)
    assert registry.attempt_id == result.attempt.attempt_id
    assert registry.build_id == result.build_id
    index = inspect_recipe(result.recipe_id, home=home)
    assert any(entry.attempt_id == result.attempt.attempt_id for entry in index.attempts)

    with pytest.raises(ValueError, match="invalid build attempt ID"):
        inspect_attempt("not-an-attempt", home=home)
    missing = "attempt-" + "0" * 24 + "-" + "e" * 32
    with pytest.raises(BuildStateError, match="does not exist"):
        inspect_attempt(missing, home=home)


def test_build_cli_inspect_and_cleanup_end_to_end(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    from typer.testing import CliRunner

    from strixlab.cli import app

    cli_runner = CliRunner()
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    result = execute_cmake_build(_preparation(tmp_path), profile, home=home, runner=run_process)

    for identifier in (result.recipe_id, result.build_id, result.attempt.attempt_id):
        inspected = cli_runner.invoke(app, ["build", "inspect", identifier, "--home", str(home)])
        assert inspected.exit_code == 0, inspected.output

    cleaned = cli_runner.invoke(app, ["build", "cleanup", result.build_id, "--home", str(home)])
    assert cleaned.exit_code == 0
    assert "cleaned" in cleaned.stdout
    assert not result.build_root.exists()
    # idempotent second cleanup still exits 0
    again = cli_runner.invoke(app, ["build", "cleanup", result.build_id, "--home", str(home)])
    assert again.exit_code == 0


def test_inspect_attempt_refuses_active_and_unindexed(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    from strixlab.builds import _layout

    home = tmp_path / "home"
    layout = _layout(home, create=True)
    active_id = "attempt-" + "0" * 24 + "-" + "a" * 32
    (layout.attempts / active_id).mkdir()
    with pytest.raises(BuildStateError, match="active build attempt"):
        inspect_attempt(active_id, home=home)

    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    result = execute_cmake_build(_preparation(tmp_path), profile, home=home, runner=run_process)
    suffix = result.recipe_id.removeprefix("recipe-sha256:")
    (home / "builds" / "indexes" / "recipes" / f"{suffix}.json").unlink()
    with pytest.raises(BuildStateError, match="not indexed"):
        inspect_attempt(result.attempt.attempt_id, home=home)


def test_cache_hit_cmake_cache_tamper_fails_closed(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    preparation = _preparation(tmp_path)

    first = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    cache_txt = first.build_root / "CMakeCache.txt"
    cache_txt.chmod(0o600)
    with cache_txt.open("a", encoding="utf-8") as handle:
        handle.write("TAMPERED:STRING=1\n")

    with pytest.raises(BuildArtifactError, match="CMakeCache.txt changed"):
        execute_cmake_build(preparation, profile, home=home, runner=run_process)


def test_cache_hit_unexpected_compile_commands_fails_closed(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    preparation = _preparation(tmp_path)

    first = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    # The fake CMake emits no compile_commands.json, so one appearing is tampering.
    (first.build_root / "compile_commands.json").write_text("[]\n", encoding="utf-8")

    with pytest.raises(BuildArtifactError, match="compile_commands.json appeared"):
        execute_cmake_build(preparation, profile, home=home, runner=run_process)


def test_patched_build_copies_source_reproducer_bytes(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    preparation = _preparation(tmp_path, patched=True)

    result = execute_cmake_build(preparation, profile, home=home, runner=run_process)
    assert result.execution_class == "built"
    diff_copy = result.attempt.record / "source" / "diff.patch"
    assert diff_copy.read_bytes() == b"--- patched diff\n"
    record = json.loads((result.attempt.record / "build/result.json").read_bytes())
    assert record["execution_class"] == "built"


def test_cache_hit_snapshot_mutation_fails_before_success(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    preparation = _preparation(tmp_path)
    execute_cmake_build(preparation, profile, home=home, runner=run_process)  # populate cache

    snapshot_source: Path | None = None

    def mutating_runner(argv: Any, **kwargs: Any) -> ProcessResult:
        nonlocal snapshot_source
        args = tuple(str(value) for value in argv)
        if "-S" in args:
            snapshot_source = Path(args[args.index("-S") + 1])
        result = run_process(argv, **kwargs)
        if "-B" in args and "probe" in args[args.index("-B") + 1] and snapshot_source is not None:
            payload = snapshot_source / "CMakeLists.txt"
            payload.chmod(0o600)
            payload.write_text("project(mutated)\n", encoding="utf-8")
        return result

    # The hit revalidation re-verifies the snapshot before success publication.
    with pytest.raises(SnapshotError, match="does not match"):
        execute_cmake_build(preparation, profile, home=home, runner=mutating_runner)


def test_inspect_authenticates_producer_provenance(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    # patched=True so the canonical reproducer references a real diff blob.
    result = execute_cmake_build(
        _preparation(tmp_path, patched=True), profile, home=home, runner=run_process
    )
    inspect_build(result.build_id, home=home)  # provenance + source bytes authenticated
    assert (result.attempt.record / "build" / "provenance.json").is_file()

    # A missing producer attempt record fails inspection closed.
    shutil.rmtree(result.attempt.record)
    with pytest.raises(BuildStateError, match="producer attempt record is missing"):
        inspect_build(result.build_id, home=home)


def test_inspect_rejects_tampered_producer_provenance(
    tmp_path: Path, build_value: dict[str, Any]
) -> None:
    tools = _toolchain(tmp_path)
    profile = _profile(build_value, tools)
    home = tmp_path / "home"
    result = execute_cmake_build(_preparation(tmp_path), profile, home=home, runner=run_process)
    provenance = result.attempt.record / "build" / "provenance.json"
    data = json.loads(provenance.read_bytes())
    data["build_id"] = "build-sha256:" + "00" * 32
    provenance.chmod(0o600)
    provenance.write_bytes(json.dumps(data).encode())
    # Tampering any record file breaks the content-addressed attempt digest.
    with pytest.raises(BuildStateError, match="verification failed"):
        inspect_build(result.build_id, home=home)

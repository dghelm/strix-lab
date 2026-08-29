from __future__ import annotations

import dataclasses
from copy import deepcopy
from typing import Any

import pytest

from strixlab.build_identity import (
    ArtifactIdentity,
    IdentityEntry,
    ToolObservation,
    artifact_set_id,
    attempt_id,
    build_id,
    machine_identity_environment,
    normalize_prefix_path,
    recipe_id,
)
from strixlab.manifests import BuildProfileV1

_CANDIDATE = "candidate-sha256:" + "12" * 32


def _profile(value: dict[str, Any]) -> BuildProfileV1:
    return BuildProfileV1.model_validate(value)


def _tool(role: str, path: str) -> ToolObservation:
    return ToolObservation(
        role=role,
        path=path,
        realpath=path,
        mode=0o755,
        size_bytes=123,
        sha256="01" * 32,
        version_sha256="02" * 32,
        search_sha256="03" * 32,
    )


def _tools(
    profile: BuildProfileV1, *, cmake_path: str | None = None
) -> tuple[ToolObservation, ...]:
    paths = {
        "cmake": cmake_path or profile.toolchain.cmake,
        "ninja": profile.toolchain.ninja,
        "c_compiler": profile.toolchain.c_compiler,
        "cxx_compiler": profile.toolchain.cxx_compiler,
    }
    if profile.toolchain.hip_compiler is not None:
        paths["hip_compiler"] = profile.toolchain.hip_compiler
    return tuple(_tool(role, path) for role, path in paths.items())


def _selections(mode: str = "rocm") -> tuple[IdentityEntry, ...]:
    values = {
        "generator": "Ninja",
        "c_compiler": "/opt/rocm-10/bin/amdclang",
        "cxx_compiler": "/opt/rocm-10/bin/amdclang++",
        "linker": "/usr/bin/ld",
        "archiver": "/usr/bin/ar",
        "toolchain_files": "",
        "sysroot": "",
    }
    if mode == "rocm":
        values.update(
            hip_compiler="/opt/rocm-10/bin/amdclang++",
            rocm_prefix="/opt/rocm-10",
            gfx_targets="gfx1151",
        )
    return tuple(IdentityEntry(name, value) for name, value in values.items())


def test_recipe_is_portable_across_prefix_resolution_and_profile_order(
    build_value: dict[str, Any],
) -> None:
    first_value = deepcopy(build_value)
    second_value = deepcopy(build_value)
    first_value["targets"] = list(reversed(first_value["targets"]))
    first_value["cmake"] = dict(reversed(tuple(first_value["cmake"].items())))
    replacements = {
        "/usr": "/srv/toolchains/system",
        "/opt/rocm-10": "/srv/toolchains/rocm",
    }
    second_toolchain = second_value["toolchain"]
    for name, path in tuple(second_toolchain["prefixes"].items()):
        second_toolchain["prefixes"][name] = replacements[path]
    for field in ("cmake", "ninja", "c_compiler", "cxx_compiler", "hip_compiler"):
        path = second_toolchain[field]
        for old, new in replacements.items():
            if path == old or path.startswith(old + "/"):
                second_toolchain[field] = new + path[len(old) :]
                break
    second_toolchain["rocm_prefix"] = replacements["/opt/rocm-10"]
    second_toolchain["path"] = [
        next(new + path[len(old) :] for old, new in replacements.items() if path.startswith(old))
        for path in second_toolchain["path"]
    ]
    second_value["environment"]["path_lists"]["ROCM_PATH"] = replacements["/opt/rocm-10"]

    first = recipe_id(_CANDIDATE, "llama_cpp", _profile(first_value))
    second = recipe_id(_CANDIDATE, "llama_cpp", _profile(second_value))

    assert first == second


def test_recipe_includes_jobs_but_excludes_timeouts(build_value: dict[str, Any]) -> None:
    baseline = recipe_id(_CANDIDATE, "llama_cpp", _profile(build_value))
    changed_timeout = deepcopy(build_value)
    changed_timeout["execution"]["timeouts"]["build_seconds"] = 99.0
    changed_jobs = deepcopy(build_value)
    changed_jobs["execution"]["jobs"] = 9

    assert recipe_id(_CANDIDATE, "llama_cpp", _profile(changed_timeout)) == baseline
    assert recipe_id(_CANDIDATE, "llama_cpp", _profile(changed_jobs)) != baseline


def test_machine_build_identity_changes_with_actual_tool_location(
    build_value: dict[str, Any],
) -> None:
    recipe = recipe_id(_CANDIDATE, "llama_cpp", _profile(build_value))
    profile = _profile(build_value)

    assert build_id(
        recipe,
        profile=profile,
        tools=_tools(profile, cmake_path="/usr/bin/cmake"),
        selections=_selections(),
    ) != build_id(
        recipe,
        profile=profile,
        tools=_tools(profile, cmake_path="/srv/toolchains/system/bin/cmake"),
        selections=_selections(),
    )


def test_build_identity_chain_has_stable_vectors(build_value: dict[str, Any]) -> None:
    profile = _profile(build_value)
    recipe = recipe_id(_CANDIDATE, "llama_cpp", profile)
    machine = build_id(
        recipe,
        profile=profile,
        tools=_tools(profile),
        selections=_selections(),
    )
    artifacts = artifact_set_id(
        (ArtifactIdentity("bin/a", 0o755, 1, "01" * 32),),
        _selections(),
        toolchain_mode="rocm",
    )

    assert (
        recipe == "recipe-sha256:499406892dfb0d48fa144223981da7bd45fc4e8da954fc0d030fa92890ca4622"
    )
    assert (
        machine == "build-sha256:5e2ea70e13853572da2812ae4d8fc3b5a65323368df26e20bb5dc6eec58eb3eb"
    )
    assert artifacts == (
        "artifact-set-sha256:fd214dd6c028229e25bdb8a2a356c9e906e8bf2018bf7d2dd190fe92297ca9f3"
    )


def test_attempt_and_artifact_set_identities_are_ordered_and_stable() -> None:
    recipe = "recipe-sha256:" + "ab" * 32
    assert attempt_id(recipe, bytes.fromhex("01" * 16)) == (
        "attempt-" + "ab" * 12 + "-" + "01" * 16
    )
    artifacts = (
        ArtifactIdentity("bin/b", 0o755, 2, "02" * 32),
        ArtifactIdentity("bin/a", 0o755, 1, "01" * 32),
    )
    selections = _selections()
    assert artifact_set_id(artifacts, selections, toolchain_mode="rocm") == artifact_set_id(
        tuple(reversed(artifacts)), selections, toolchain_mode="rocm"
    )


def test_unrepresentable_environment_paths_fail_before_execution(
    build_value: dict[str, Any],
) -> None:
    build_value["environment"]["path_lists"]["ROCM_PATH"] = "/unowned/sdk"

    with pytest.raises(ValueError, match="outside declared toolchain prefixes"):
        recipe_id(_CANDIDATE, "llama_cpp", _profile(build_value))
    with pytest.raises(ValueError, match="normalized absolute path"):
        normalize_prefix_path("/opt/rocm-10/../other", {"rocm": "/opt/rocm-10"})


def test_identity_inputs_reject_ambiguous_and_malformed_values(
    build_value: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="ambiguous toolchain prefix"):
        normalize_prefix_path("/opt/sdk/bin", {"first": "/opt/sdk", "second": "/opt/sdk"})

    empty_component = deepcopy(build_value)
    empty_component["environment"]["path_lists"]["ROCM_PATH"] = "/opt/rocm-10:"
    with pytest.raises(ValueError, match="empty component"):
        recipe_id(_CANDIDATE, "llama_cpp", _profile(empty_component))

    absolute_literal = deepcopy(build_value)
    absolute_literal["environment"]["literals"]["SOURCE_DATE_EPOCH"] = "/tmp/epoch"
    with pytest.raises(ValueError, match="literal environment entry contains an absolute path"):
        recipe_id(_CANDIDATE, "llama_cpp", _profile(absolute_literal))

    profile = _profile(build_value)
    with pytest.raises(ValueError, match="invalid source candidate ID"):
        recipe_id("source-sha256:" + "12" * 32, "llama_cpp", profile)
    with pytest.raises(ValueError, match="invalid build recipe ID"):
        build_id("not-a-recipe", profile=profile, tools=(), selections=())
    with pytest.raises(ValueError, match="16-byte nonce"):
        attempt_id("recipe-sha256:" + "ab" * 32, b"short")


def test_recipe_identity_distinguishes_numeric_cmake_scalar_types(
    build_value: dict[str, Any],
) -> None:
    integer = deepcopy(build_value)
    floating = deepcopy(build_value)
    integer["cmake"]["LEVEL"] = 1
    floating["cmake"]["LEVEL"] = 1.0

    assert recipe_id(_CANDIDATE, "llama_cpp", _profile(integer)) != recipe_id(
        _CANDIDATE, "llama_cpp", _profile(floating)
    )


def test_machine_projection_is_closed_and_uses_stable_root_placeholders(
    build_value: dict[str, Any],
) -> None:
    profile = _profile(build_value)
    environment = dict((entry.name, entry.value) for entry in machine_identity_environment(profile))

    assert environment["SOURCE_ROOT"] == "{SOURCE_ROOT}"
    assert environment["BUILD_ROOT"] == "{BUILD_ROOT}"
    assert environment["BUILD_HOME"] == "{BUILD_HOME}"
    assert environment["BUILD_TMP"] == "{BUILD_TMP}"
    assert not any("/home/" in value or "/tmp/" in value for value in environment.values())


def test_machine_identity_rejects_noncanonical_or_incomplete_projections(
    build_value: dict[str, Any],
) -> None:
    profile = _profile(build_value)
    recipe = recipe_id(_CANDIDATE, "llama_cpp", profile)
    tools = _tools(profile)

    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        recipe_id("candidate-sha256:00", "llama_cpp", profile)
    with pytest.raises(ValueError, match="64 lowercase hexadecimal"):
        build_id("recipe-sha256:00", profile=profile, tools=tools, selections=_selections())
    with pytest.raises(ValueError, match="duplicate tool role"):
        build_id(
            recipe,
            profile=profile,
            tools=(*tools, tools[0]),
            selections=_selections(),
        )
    with pytest.raises(ValueError, match="required projection"):
        build_id(recipe, profile=profile, tools=tools, selections=())
    malformed_hash = (*tools[:-1], dataclasses.replace(tools[-1], sha256="AB" * 32))
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        build_id(recipe, profile=profile, tools=malformed_hash, selections=_selections())


@pytest.mark.parametrize("path", ["/absolute/bin", "../escape", "bin/../escape", "bin//a", "."])
def test_artifact_identity_requires_normalized_root_relative_paths(path: str) -> None:
    artifact = ArtifactIdentity(path, 0o755, 1, "01" * 32)
    with pytest.raises(ValueError, match="normalized and root-relative"):
        artifact_set_id((artifact,), _selections(), toolchain_mode="rocm")


def test_artifact_identity_rejects_empty_and_duplicate_sets() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        artifact_set_id((), _selections(), toolchain_mode="rocm")
    artifact = ArtifactIdentity("bin/a", 0o755, 1, "01" * 32)
    with pytest.raises(ValueError, match="duplicate artifact path"):
        artifact_set_id((artifact, artifact), _selections(), toolchain_mode="rocm")


def test_identity_rejects_runtime_values_that_only_resemble_canonical_types(
    build_value: dict[str, Any],
) -> None:
    profile = _profile(build_value)
    recipe = recipe_id(_CANDIDATE, "llama_cpp", profile)
    tools = _tools(profile)

    for malformed in (
        dataclasses.replace(tools[0], mode=True),
        dataclasses.replace(tools[0], size_bytes=False),
        dataclasses.replace(tools[0], path="/usr/bin/cm\x00ake"),
        dataclasses.replace(tools[0], realpath="/usr/bin/cm\x00ake"),
    ):
        with pytest.raises(ValueError):
            build_id(
                recipe,
                profile=profile,
                tools=(malformed, *tools[1:]),
                selections=_selections(),
            )

    artifact = ArtifactIdentity("bin/a", 0o755, 1, "01" * 32)
    with pytest.raises(ValueError, match="toolchain mode"):
        artifact_set_id((artifact,), _selections(), toolchain_mode="invalid")  # type: ignore[arg-type]
    for malformed in (
        dataclasses.replace(artifact, mode=True),
        dataclasses.replace(artifact, size_bytes=False),
        dataclasses.replace(artifact, path="bin/a\x00suffix"),
    ):
        with pytest.raises(ValueError):
            artifact_set_id((malformed,), _selections(), toolchain_mode="rocm")

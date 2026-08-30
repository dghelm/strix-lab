"""Versioned identities for portable recipes and machine-local builds."""

from __future__ import annotations

import hashlib
import os
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from strixlab.manifests import BuildProfileV1, CMakeScalar
from strixlab.source_identity import length_frame

_BUILD_ADAPTER = "cmake-ninja-llama-cpp"
_BUILD_PROTOCOL_VERSION = 1
_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROLE_RE = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")
_ENTRY_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")
_BASE_TOOL_ROLES = frozenset({"cmake", "ninja", "c_compiler", "cxx_compiler"})
_BASE_SELECTION_NAMES = frozenset(
    {
        "generator",
        "c_compiler",
        "cxx_compiler",
        "linker",
        "archiver",
        "toolchain_files",
        "sysroot",
    }
)
_ROCM_SELECTION_NAMES = frozenset({"hip_compiler", "rocm_prefix", "gfx_targets"})


@dataclass(frozen=True, slots=True)
class ToolObservation:
    role: str
    path: str
    realpath: str
    mode: int
    size_bytes: int
    sha256: str
    version_sha256: str
    search_sha256: str


@dataclass(frozen=True, slots=True)
class IdentityEntry:
    name: str
    value: str


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    path: str
    mode: int
    size_bytes: int
    sha256: str


ROOT_PLACEHOLDERS = {
    "SOURCE_ROOT": "{SOURCE_ROOT}",
    "BUILD_ROOT": "{BUILD_ROOT}",
    "BUILD_HOME": "{BUILD_HOME}",
    "BUILD_TMP": "{BUILD_TMP}",
}
_EXECUTOR_CONSTANTS = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"}


def _u32(value: int) -> bytes:
    return struct.pack(">I", value)


def _u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def _sequence(values: tuple[bytes, ...]) -> bytes:
    return _u32(len(values)) + b"".join(_u64(len(value)) + value for value in values)


def _digest(domain: str, fields: tuple[tuple[str, bytes], ...]) -> str:
    return hashlib.sha256(length_frame(domain, fields)).hexdigest()


def _digest_value(value: str, description: str) -> bytes:
    if _HEX_SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be exactly 64 lowercase hexadecimal characters")
    return bytes.fromhex(value)


def _protocol_id(value: str, prefix: str, description: str) -> bytes:
    if not value.startswith(prefix):
        raise ValueError(f"invalid {description}")
    return _digest_value(value.removeprefix(prefix), description)


def _strict_nonnegative_int(value: int, description: str, *, maximum: int | None = None) -> None:
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError(f"invalid {description}")


def _validated_entries(
    values: tuple[IdentityEntry, ...],
    description: str,
    *,
    expected_names: frozenset[str] | None = None,
) -> tuple[IdentityEntry, ...]:
    names: set[str] = set()
    for value in values:
        if _ENTRY_NAME_RE.fullmatch(value.name) is None:
            raise ValueError(f"invalid {description} name: {value.name}")
        if value.name in names:
            raise ValueError(f"duplicate {description} name: {value.name}")
        if "\x00" in value.value:
            raise ValueError(f"{description} value cannot contain NUL bytes: {value.name}")
        names.add(value.name)
    if expected_names is not None and names != expected_names:
        raise ValueError(
            f"{description} names do not match the required projection: "
            f"expected {sorted(expected_names)}, got {sorted(names)}"
        )
    return values


def _scalar(value: CMakeScalar) -> bytes:
    if isinstance(value, bool):
        return b"b\x01" if value else b"b\x00"
    if isinstance(value, int):
        return b"i" + str(value).encode("ascii")
    if isinstance(value, float):
        return b"f" + value.hex().encode("ascii")
    return b"s" + value.encode("utf-8")


def _mapping_frame(domain: str, values: tuple[IdentityEntry, ...]) -> bytes:
    _validated_entries(values, domain)
    entries = tuple(
        length_frame(
            domain + ".entry",
            (("name", value.name.encode("utf-8")), ("value", value.value.encode("utf-8"))),
        )
        for value in sorted(values, key=lambda item: item.name.encode("utf-8"))
    )
    return _sequence(entries)


def normalize_prefix_path(path: str, prefixes: Mapping[str, str]) -> str:
    """Replace one local absolute prefix with its stable logical name."""

    candidate = Path(path)
    if not candidate.is_absolute() or os.path.normpath(path) != path:
        raise ValueError(f"build path is not a normalized absolute path: {path}")
    matches: list[tuple[int, str, PurePosixPath]] = []
    for name, raw_prefix in prefixes.items():
        prefix = Path(raw_prefix)
        try:
            relative = candidate.relative_to(prefix)
        except ValueError:
            continue
        matches.append((len(prefix.parts), name, PurePosixPath(relative.as_posix())))
    if not matches:
        raise ValueError(f"build path is outside declared toolchain prefixes: {path}")
    longest = max(length for length, _name, _relative in matches)
    winners = [(name, relative) for length, name, relative in matches if length == longest]
    if len(winners) != 1:
        raise ValueError(f"build path has an ambiguous toolchain prefix: {path}")
    prefix_name, relative_path = winners[0]
    suffix = "" if str(relative_path) == "." else f"/{relative_path}"
    return f"{{PREFIX:{prefix_name}}}{suffix}"


def normalized_profile_environment(profile: BuildProfileV1) -> tuple[IdentityEntry, ...]:
    prefixes = profile.toolchain.prefixes
    entries: list[IdentityEntry] = []
    for path_name, raw in profile.environment.path_lists.items():
        components = raw.split(":")
        if not components or any(not component for component in components):
            raise ValueError(
                f"path-list environment entry contains an empty component: {path_name}"
            )
        normalized = ":".join(
            normalize_prefix_path(component, prefixes) for component in components
        )
        entries.append(IdentityEntry(path_name, normalized))
    for literal_name, value in profile.environment.literals.items():
        if Path(value).is_absolute():
            raise ValueError(f"literal environment entry contains an absolute path: {literal_name}")
        entries.append(IdentityEntry(literal_name, value))
    return tuple(sorted(entries, key=lambda item: item.name.encode("utf-8")))


def recipe_id(candidate_id: str, source_adapter: str, profile: BuildProfileV1) -> str:
    candidate_digest = _protocol_id(candidate_id, "candidate-sha256:", "source candidate ID")
    if _ROLE_RE.fullmatch(source_adapter) is None:
        raise ValueError("invalid source adapter")
    cmake = tuple(
        length_frame(
            "strixlab.build.cmake-entry.v1",
            (("name", name.encode("utf-8")), ("value", _scalar(value))),
        )
        for name, value in sorted(profile.cmake.items(), key=lambda item: item[0].encode("utf-8"))
    )
    roles = {
        "cmake": profile.toolchain.cmake,
        "ninja": profile.toolchain.ninja,
        "c_compiler": profile.toolchain.c_compiler,
        "cxx_compiler": profile.toolchain.cxx_compiler,
    }
    if profile.toolchain.hip_compiler is not None:
        roles["hip_compiler"] = profile.toolchain.hip_compiler
    if profile.toolchain.rocm_prefix is not None:
        roles["rocm_prefix"] = profile.toolchain.rocm_prefix
    for index, value in enumerate(profile.toolchain.path):
        roles[f"path.{index}"] = value
    role_entries = tuple(
        IdentityEntry(name, normalize_prefix_path(value, profile.toolchain.prefixes))
        for name, value in roles.items()
    )
    digest = _digest(
        "strixlab.build.recipe.v1",
        (
            ("identity_version", _u32(_BUILD_PROTOCOL_VERSION)),
            ("candidate_digest", candidate_digest),
            ("source_adapter", source_adapter.encode("utf-8")),
            ("build_adapter", _BUILD_ADAPTER.encode("ascii")),
            ("generator", profile.generator.encode("ascii")),
            ("build_type", profile.build_type.encode("ascii")),
            ("cmake", _sequence(cmake)),
            (
                "targets",
                _sequence(
                    tuple(
                        target.encode("utf-8")
                        for target in sorted(
                            profile.targets, key=lambda value: value.encode("utf-8")
                        )
                    )
                ),
            ),
            (
                "environment",
                _mapping_frame(
                    "strixlab.build.environment.v1", normalized_profile_environment(profile)
                ),
            ),
            ("toolchain_mode", profile.toolchain.mode.encode("ascii")),
            ("tool_roles", _mapping_frame("strixlab.build.tool-role.v1", role_entries)),
            ("jobs", _u32(profile.execution.jobs)),
        ),
    )
    return f"recipe-sha256:{digest}"


def build_id(
    recipe: str,
    *,
    profile: BuildProfileV1,
    tools: tuple[ToolObservation, ...],
    selections: tuple[IdentityEntry, ...],
) -> str:
    recipe_digest = _protocol_id(recipe, "recipe-sha256:", "build recipe ID")
    required_tools = set(_BASE_TOOL_ROLES)
    required_selections = set(_BASE_SELECTION_NAMES)
    if profile.toolchain.mode == "rocm":
        required_tools.add("hip_compiler")
        required_selections.update(_ROCM_SELECTION_NAMES)
    tool_roles: set[str] = set()
    for tool in tools:
        if _ROLE_RE.fullmatch(tool.role) is None:
            raise ValueError(f"invalid tool role: {tool.role}")
        if tool.role in tool_roles:
            raise ValueError(f"duplicate tool role: {tool.role}")
        tool_roles.add(tool.role)
        for path_name, raw_path in (("path", tool.path), ("realpath", tool.realpath)):
            if (
                "\x00" in raw_path
                or not Path(raw_path).is_absolute()
                or os.path.normpath(raw_path) != raw_path
            ):
                raise ValueError(f"tool {path_name} is not a normalized absolute path: {tool.role}")
        _strict_nonnegative_int(tool.mode, f"mode for tool role: {tool.role}", maximum=0o7777)
        _strict_nonnegative_int(tool.size_bytes, f"size for tool role: {tool.role}")
        _digest_value(tool.sha256, f"tool SHA-256 for {tool.role}")
        _digest_value(tool.version_sha256, f"tool version SHA-256 for {tool.role}")
        _digest_value(tool.search_sha256, f"tool search SHA-256 for {tool.role}")
    if not required_tools.issubset(tool_roles):
        raise ValueError(f"missing required tool roles: {sorted(required_tools - tool_roles)}")
    _validated_entries(
        selections,
        "build selection",
        expected_names=frozenset(required_selections),
    )
    environment = machine_identity_environment(profile)
    tool_frames = tuple(
        length_frame(
            "strixlab.build.tool-observation.v1",
            (
                ("role", tool.role.encode("utf-8")),
                ("path", tool.path.encode("utf-8")),
                ("realpath", tool.realpath.encode("utf-8")),
                ("mode", _u32(tool.mode)),
                ("size_bytes", _u64(tool.size_bytes)),
                ("sha256", _digest_value(tool.sha256, f"tool SHA-256 for {tool.role}")),
                (
                    "version_sha256",
                    _digest_value(tool.version_sha256, f"tool version SHA-256 for {tool.role}"),
                ),
                (
                    "search_sha256",
                    _digest_value(tool.search_sha256, f"tool search SHA-256 for {tool.role}"),
                ),
            ),
        )
        for tool in sorted(tools, key=lambda item: item.role.encode("utf-8"))
    )
    digest = _digest(
        "strixlab.build.machine.v1",
        (
            ("recipe_digest", recipe_digest),
            ("tools", _sequence(tool_frames)),
            ("selections", _mapping_frame("strixlab.build.selection.v1", selections)),
            ("environment", _mapping_frame("strixlab.build.machine-environment.v1", environment)),
        ),
    )
    return f"build-sha256:{digest}"


def attempt_id(recipe: str, nonce: bytes) -> str:
    try:
        _protocol_id(recipe, "recipe-sha256:", "build recipe ID")
    except ValueError as exc:
        raise ValueError("attempt identity requires a recipe ID and 16-byte nonce") from exc
    if len(nonce) != 16:
        raise ValueError("attempt identity requires a recipe ID and 16-byte nonce")
    return f"attempt-{recipe.removeprefix('recipe-sha256:')[:24]}-{nonce.hex()}"


def artifact_set_id(
    artifacts: tuple[ArtifactIdentity, ...],
    selections: tuple[IdentityEntry, ...],
    *,
    toolchain_mode: Literal["host", "rocm"],
) -> str:
    if toolchain_mode not in {"host", "rocm"}:
        raise ValueError(f"invalid artifact toolchain mode: {toolchain_mode}")
    if not artifacts:
        raise ValueError("artifact set cannot be empty")
    expected_selections = set(_BASE_SELECTION_NAMES)
    if toolchain_mode == "rocm":
        expected_selections.update(_ROCM_SELECTION_NAMES)
    _validated_entries(
        selections,
        "artifact selection",
        expected_names=frozenset(expected_selections),
    )
    artifact_paths: set[str] = set()
    for artifact in artifacts:
        candidate = PurePosixPath(artifact.path)
        if (
            not artifact.path
            or "\x00" in artifact.path
            or candidate.is_absolute()
            or artifact.path != candidate.as_posix()
            or candidate == PurePosixPath(".")
            or ".." in candidate.parts
        ):
            raise ValueError(f"artifact path is not normalized and root-relative: {artifact.path}")
        if artifact.path in artifact_paths:
            raise ValueError(f"duplicate artifact path: {artifact.path}")
        artifact_paths.add(artifact.path)
        _strict_nonnegative_int(
            artifact.mode,
            f"mode for artifact: {artifact.path}",
            maximum=0o7777,
        )
        _strict_nonnegative_int(artifact.size_bytes, f"size for artifact: {artifact.path}")
        _digest_value(artifact.sha256, f"artifact SHA-256 for {artifact.path}")
    frames = tuple(
        length_frame(
            "strixlab.build.artifact.v1",
            (
                ("path", artifact.path.encode("utf-8")),
                ("mode", _u32(artifact.mode)),
                ("size_bytes", _u64(artifact.size_bytes)),
                ("sha256", _digest_value(artifact.sha256, f"artifact SHA-256 for {artifact.path}")),
            ),
        )
        for artifact in sorted(artifacts, key=lambda item: item.path.encode("utf-8"))
    )
    digest = _digest(
        "strixlab.build.artifact-set.v1",
        (
            ("artifacts", _sequence(frames)),
            ("selections", _mapping_frame("strixlab.build.selection.v1", selections)),
        ),
    )
    return f"artifact-set-sha256:{digest}"


def machine_identity_environment(profile: BuildProfileV1) -> tuple[IdentityEntry, ...]:
    """Build the exact normalized environment projection for machine identity."""

    values = list(normalized_profile_environment(profile))
    values.extend(IdentityEntry(name, value) for name, value in _EXECUTOR_CONSTANTS.items())
    values.extend(IdentityEntry(name, value) for name, value in ROOT_PLACEHOLDERS.items())
    entries = tuple(values)
    expected_names = frozenset(
        {
            *profile.environment.path_lists,
            *profile.environment.literals,
            *_EXECUTOR_CONSTANTS,
            *ROOT_PLACEHOLDERS,
        }
    )
    _validated_entries(entries, "machine environment", expected_names=expected_names)
    return tuple(sorted(entries, key=lambda item: item.name.encode("utf-8")))

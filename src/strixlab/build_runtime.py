"""Build-artifact and canonical-environment runtime reconstruction helpers."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from strixlab.build_artifacts import BuildArtifactsV1
from strixlab.build_cache import CanonicalBuildRecordV1
from strixlab.build_identity import ROOT_PLACEHOLDERS

__all__ = [
    "BuildRuntimeEnvironment",
    "RuntimeErrorFactory",
    "reconstruct_environment",
    "resolve_target_artifact",
    "resolve_target_executable",
]

RuntimeErrorFactory = Callable[[str], RuntimeError]

_BUILD_ROOT_PLACEHOLDER = ROOT_PLACEHOLDERS["BUILD_ROOT"]
_PLACEHOLDER_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REQUIRED_LOCALE = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"}


@dataclass(frozen=True, slots=True)
class BuildRuntimeEnvironment:
    """Mutable child environment and caller-owned runtime paths."""

    environment: dict[str, str]
    cwd: Path
    scratch_root: Path


def resolve_target_artifact(
    artifacts: BuildArtifactsV1,
    target_name: str,
    *,
    error: RuntimeErrorFactory,
) -> tuple[str, str]:
    """Resolve one required target to its unique relative path and recorded digest."""

    named = [target for target in artifacts.targets if target.name == target_name]
    if len(named) != 1:
        raise error(f"build target is missing or ambiguous: {target_name}")
    if named[0].target_type != "EXECUTABLE":
        raise error(f"build target is not an executable: {target_name}")
    candidates = [
        artifact
        for artifact in artifacts.artifacts
        if target_name in artifact.targets
        and artifact.kind == "elf"
        and artifact.elf_type in ("ET_EXEC", "ET_DYN")
        and not artifact.runtime_dependency
    ]
    if len(candidates) != 1:
        raise error(f"expected exactly one executable artifact for target: {target_name}")
    artifact = candidates[0]
    relative = PurePosixPath(artifact.path)
    if relative.is_absolute() or ".." in relative.parts:
        raise error(f"build artifact escapes the leased root: {target_name}")
    return artifact.path, artifact.sha256


def resolve_target_executable(
    artifacts: BuildArtifactsV1,
    target_name: str,
    root: Path,
    *,
    error: RuntimeErrorFactory,
) -> tuple[str, str]:
    """Resolve one required target to an absolute executable path and recorded digest."""

    relative, digest = resolve_target_artifact(artifacts, target_name, error=error)
    return str(root / PurePosixPath(relative)), digest


def reconstruct_environment(
    canonical: CanonicalBuildRecordV1,
    root: Path,
    scratch_root: Path,
    *,
    error: RuntimeErrorFactory,
) -> BuildRuntimeEnvironment:
    """Rebuild a child environment from a leased canonical build tuple.

    Never inherits ambient ``os.environ``. Component-boundary rehydration replaces an
    exact ``{BUILD_ROOT}`` component (or one beginning ``{BUILD_ROOT}`` + ``os.sep``)
    with the leased root; ``HOME`` and ``TMPDIR`` are replaced with fresh directories
    under one mode-0700 temporary root. A residual ``{SOURCE_ROOT}``/``{BUILD_HOME}``/
    ``{BUILD_TMP}`` or any other placeholder-shaped component, a NUL, a duplicate name,
    an invalid name, a missing ``HOME``/``TMPDIR``, or a wrong locale/time value fails
    closed.
    """

    scratch_home = scratch_root / "home"
    scratch_tmp = scratch_root / "tmp"
    for path in (scratch_home, scratch_tmp):
        path.mkdir(mode=0o700)

    seen: set[str] = set()
    environment: dict[str, str] = {}
    root_str = str(root)
    for entry in canonical.environment:
        if _ENV_NAME_RE.fullmatch(entry.name) is None:
            raise error(f"leased build environment has an invalid name: {entry.name!r}")
        if entry.name in seen:
            raise error(f"leased build environment has a duplicate name: {entry.name!r}")
        seen.add(entry.name)
        if "\x00" in entry.name or "\x00" in entry.value:
            raise error("leased build environment contains a NUL byte")
        if entry.name == "HOME":
            environment["HOME"] = str(scratch_home)
        elif entry.name == "TMPDIR":
            environment["TMPDIR"] = str(scratch_tmp)
        else:
            environment[entry.name] = _rehydrate_value(entry.value, root_str, error=error)

    for name in ("HOME", "TMPDIR"):
        if name not in seen:
            raise error(f"leased build environment is missing {name}")
    for name, expected in _REQUIRED_LOCALE.items():
        if environment.get(name) != expected:
            raise error(f"leased build environment has an unexpected {name}")
    return BuildRuntimeEnvironment(
        environment=environment,
        cwd=scratch_tmp,
        scratch_root=scratch_root,
    )


def _rehydrate_value(value: str, root: str, *, error: RuntimeErrorFactory) -> str:
    return os.pathsep.join(
        _rehydrate_component(component, root, error=error) for component in value.split(os.pathsep)
    )


def _rehydrate_component(component: str, root: str, *, error: RuntimeErrorFactory) -> str:
    if component == _BUILD_ROOT_PLACEHOLDER:
        return root
    if component.startswith(_BUILD_ROOT_PLACEHOLDER + os.sep):
        rest = component[len(_BUILD_ROOT_PLACEHOLDER) :]
        if _PLACEHOLDER_RE.search(rest):
            raise error("leased build environment component has an unexpected placeholder")
        return root + rest
    if _PLACEHOLDER_RE.search(component):
        raise error("leased build environment component has an unknown placeholder")
    return component

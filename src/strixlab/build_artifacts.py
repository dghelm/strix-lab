"""Exact CMake File API artifact discovery and post-build evidence capture."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from strixlab.build_identity import ArtifactIdentity, IdentityEntry, artifact_set_id
from strixlab.process import ProcessOutcome, ProcessResult, process_result_digest, run_process
from strixlab.secure_fs import readonly_open_flags, write_exclusive
from strixlab.serialization import canonical_json_bytes

ElfType = Literal["ET_REL", "ET_EXEC", "ET_DYN", "ET_CORE", "ET_OTHER"]

_FILE_LIMIT = 64 * 1024 * 1024
_OUTPUT_LIMIT = 256 * 1024
_STREAM_CHUNK = 1024 * 1024
_HEADER_BYTES = 64
_READ_CHUNK = 64 * 1024
_ELF_MAGIC = b"\x7fELF"
_AR_MAGIC = b"!<arch>\n"
_LDD_STATIC = ("not a dynamic executable", "statically linked")
_ELF_TYPES: dict[int, ElfType] = {1: "ET_REL", 2: "ET_EXEC", 3: "ET_DYN", 4: "ET_CORE"}
_DYNAMIC_ELF_TYPES: frozenset[ElfType] = frozenset({"ET_EXEC", "ET_DYN"})
_TARGET_TYPES = frozenset(
    {"EXECUTABLE", "SHARED_LIBRARY", "MODULE_LIBRARY", "STATIC_LIBRARY", "OBJECT_LIBRARY"}
)
# CMake target kind -> the artifact format(s) its mandatory output must have.
_TARGET_ELF_TYPES: dict[str, frozenset[ElfType]] = {
    "EXECUTABLE": frozenset({"ET_EXEC", "ET_DYN"}),
    "SHARED_LIBRARY": frozenset({"ET_DYN"}),
    "MODULE_LIBRARY": frozenset({"ET_DYN"}),
    "OBJECT_LIBRARY": frozenset({"ET_REL"}),
}
_ARCHIVE_TARGET_TYPES = frozenset({"STATIC_LIBRARY"})


class BuildArtifactError(RuntimeError):
    """Artifact discovery or verification failed closed."""


class _StoredModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ArtifactV1(_StoredModel):
    schema_version: Literal[1] = 1
    path: str
    kind: Literal["elf", "archive"]
    elf_type: ElfType | None = None
    mode: int = Field(ge=0, le=0o7777)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    targets: tuple[str, ...]
    runtime_dependency: bool = False


class TargetArtifactsV1(_StoredModel):
    schema_version: Literal[1] = 1
    name: str
    target_id: str
    target_type: str
    artifacts: tuple[str, ...]


class DynamicDependencyV1(_StoredModel):
    schema_version: Literal[1] = 1
    name: str
    path: str | None
    resolved: bool
    in_build_root: bool


class DynamicInspectionV1(_StoredModel):
    schema_version: Literal[1] = 1
    artifact: str
    elf_type: ElfType
    dynamic: bool
    static: bool
    needed: tuple[str, ...]
    soname: str | None = None
    rpath: tuple[str, ...] = ()
    runpath: tuple[str, ...] = ()
    dependencies: tuple[DynamicDependencyV1, ...]
    readelf_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ldd_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class CaptureToolV1(_StoredModel):
    schema_version: Literal[1] = 1
    name: Literal["ldd", "readelf"]
    path: str
    realpath: str
    mode: int = Field(ge=0, le=0o7777)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    version_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BuildArtifactsV1(_StoredModel):
    schema_version: Literal[1] = 1
    artifact_set_id: str = Field(pattern=r"^artifact-set-sha256:[0-9a-f]{64}$")
    targets: tuple[TargetArtifactsV1, ...]
    artifacts: tuple[ArtifactV1, ...]
    inspections: tuple[DynamicInspectionV1, ...]
    capture_tools: tuple[CaptureToolV1, ...]
    cmake_cache_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    compile_commands_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RawReply:
    name: str
    content: bytes


@dataclass(frozen=True, slots=True)
class ArtifactCapture:
    evidence: BuildArtifactsV1
    raw_replies: tuple[RawReply, ...]
    process_results: tuple[tuple[str, ProcessResult], ...]
    compile_commands: bytes | None


ProcessRunner = Callable[..., ProcessResult]


def prepare_file_api_query(build_root: Path) -> None:
    """Install the versioned StrixLab CMake File API query before configure."""

    query = build_root / ".cmake" / "api" / "v1" / "query" / "client-strixlab"
    query.mkdir(mode=0o700, parents=True)
    payload = canonical_json_bytes(
        {"requests": [{"kind": "codemodel", "version": {"major": 2, "minor": 0}}]}
    )
    write_exclusive(query / "query.json", payload, 0o600)


def capture_build_artifacts(
    build_root: Path,
    *,
    build_type: str,
    requested_targets: tuple[str, ...],
    selections: tuple[IdentityEntry, ...],
    toolchain_mode: Literal["host", "rocm"],
    environment: Mapping[str, str],
    discovery_timeout: float,
    capability_timeout: float,
    inspection_timeout: float,
    process_root: Path,
    runner: ProcessRunner = run_process,
) -> ArtifactCapture:
    """Discover mandatory outputs and capture their exact runtime closure.

    Timeout routing: capture-tool ``--version`` probes use the discovery timeout,
    ``ldd``/``readelf`` inspection uses the inspection timeout, and target
    ``--help``/``--version`` capability probes use the capability timeout.
    """

    replies, targets = _file_api_targets(build_root, build_type, requested_targets)
    artifacts = _initial_artifacts(build_root, targets)
    tools, version_results = _capture_tools(
        environment, build_root, discovery_timeout, process_root, runner
    )
    readelf = Path(next(tool.realpath for tool in tools if tool.name == "readelf"))
    ldd = Path(next(tool.realpath for tool in tools if tool.name == "ldd"))
    inspections: dict[str, DynamicInspectionV1] = {}
    processes = list(version_results)
    pending = deque(sorted(path for path, item in artifacts.items() if item.kind == "elf"))
    observed: set[str] = set()
    while pending:
        relative = pending.popleft()
        if relative in observed:
            continue
        observed.add(relative)
        artifact_elf_type = artifacts[relative].elf_type
        assert artifact_elf_type is not None  # only ELF artifacts enter the queue
        inspection, results = _inspect_dynamic(
            build_root / PurePosixPath(relative),
            relative,
            artifact_elf_type,
            build_root,
            readelf,
            ldd,
            environment,
            inspection_timeout,
            process_root,
            runner,
        )
        inspections[relative] = inspection
        processes.extend(results)
        for dependency in inspection.dependencies:
            if not dependency.in_build_root or dependency.path is None:
                continue
            dep_relative = _relative_file(build_root, Path(dependency.path))
            if dep_relative not in artifacts:
                artifacts[dep_relative] = _artifact(
                    build_root, dep_relative, targets=(), runtime_dependency=True
                )
            if artifacts[dep_relative].kind == "elf" and dep_relative not in observed:
                pending.append(dep_relative)
    for target in targets:
        if target.target_type != "EXECUTABLE":
            continue
        for relative in target.artifacts:
            artifact = build_root / PurePosixPath(relative)
            for flag in ("--help", "--version"):
                name = f"target-{target.name}-{flag.removeprefix('--')}"
                result = _run(
                    runner,
                    (str(artifact), flag),
                    cwd=build_root,
                    environment=environment,
                    timeout=capability_timeout,
                    process_root=process_root,
                    label=name,
                )
                _require_observation(result, name, allow_nonzero=True)
                processes.append((name, result))
    cache = _read_regular(build_root, build_root / "CMakeCache.txt", _FILE_LIMIT)
    compile_path = build_root / "compile_commands.json"
    # lexists, not exists: a dangling or symlinked entry is divergent filesystem
    # state, not an absent observation. Its presence routes to the nofollow read,
    # which fails closed on any non-regular entry.
    compile_commands = (
        _read_regular(build_root, compile_path, _FILE_LIMIT)
        if os.path.lexists(compile_path)
        else None
    )
    identities = tuple(
        ArtifactIdentity(item.path, item.mode, item.size_bytes, item.sha256)
        for item in sorted(artifacts.values(), key=lambda value: value.path.encode("utf-8"))
    )
    evidence = BuildArtifactsV1(
        artifact_set_id=artifact_set_id(identities, selections, toolchain_mode=toolchain_mode),
        targets=targets,
        artifacts=tuple(sorted(artifacts.values(), key=lambda value: value.path.encode("utf-8"))),
        inspections=tuple(inspections[key] for key in sorted(inspections)),
        capture_tools=tools,
        cmake_cache_sha256=hashlib.sha256(cache).hexdigest(),
        compile_commands_sha256=(
            hashlib.sha256(compile_commands).hexdigest() if compile_commands is not None else None
        ),
    )
    return ArtifactCapture(evidence, replies, tuple(processes), compile_commands)


def verify_artifact_capture(
    build_root: Path,
    expected: BuildArtifactsV1,
    *,
    selections: tuple[IdentityEntry, ...],
    toolchain_mode: Literal["host", "rocm"],
) -> None:
    """Rehash captured artifacts and inspection tools without trusting path names."""

    if not expected.artifacts:
        raise BuildArtifactError("artifact evidence is empty")
    identities: list[ArtifactIdentity] = []
    for artifact in expected.artifacts:
        observed = _artifact(
            build_root,
            artifact.path,
            targets=artifact.targets,
            runtime_dependency=artifact.runtime_dependency,
        )
        if observed != artifact:
            raise BuildArtifactError(f"build artifact changed: {artifact.path}")
        identities.append(
            ArtifactIdentity(observed.path, observed.mode, observed.size_bytes, observed.sha256)
        )
    observed_set = artifact_set_id(tuple(identities), selections, toolchain_mode=toolchain_mode)
    if observed_set != expected.artifact_set_id:
        raise BuildArtifactError("artifact-set identity changed")
    cache = _read_regular(build_root, build_root / "CMakeCache.txt", _FILE_LIMIT)
    if hashlib.sha256(cache).hexdigest() != expected.cmake_cache_sha256:
        raise BuildArtifactError("CMakeCache.txt changed")
    compile_path = build_root / "compile_commands.json"
    # lexists, not exists: a dangling symlink must read as divergent, not absent.
    present = os.path.lexists(compile_path)
    if expected.compile_commands_sha256 is None:
        if present:
            raise BuildArtifactError("compile_commands.json appeared after capture")
    else:
        if not present:
            raise BuildArtifactError("compile_commands.json disappeared after capture")
        compile_commands = _read_regular(build_root, compile_path, _FILE_LIMIT)
        if hashlib.sha256(compile_commands).hexdigest() != expected.compile_commands_sha256:
            raise BuildArtifactError("compile_commands.json changed")
    for tool in expected.capture_tools:
        selected = Path(tool.path)
        try:
            realpath = selected.resolve(strict=True)
        except OSError as exc:
            raise BuildArtifactError(f"capture tool is unavailable: {tool.name}") from exc
        if str(realpath) != tool.realpath:
            raise BuildArtifactError(f"capture tool selection changed: {tool.name}")
        content, metadata = _read_regular_with_stat(
            realpath.parent, realpath, _FILE_LIMIT, require_owner=False
        )
        if (
            stat.S_IMODE(metadata.st_mode) != tool.mode
            or metadata.st_size != tool.size_bytes
            or hashlib.sha256(content).hexdigest() != tool.sha256
        ):
            raise BuildArtifactError(f"capture tool changed: {tool.name}")


def _is_codemodel_v2(version: object) -> bool:
    """Require an exact codemodel major version of 2, matching the query."""

    return (
        isinstance(version, dict)
        and type(version.get("major")) is int
        and version.get("major") == 2
    )


def _file_api_targets(
    build_root: Path, build_type: str, requested: tuple[str, ...]
) -> tuple[tuple[RawReply, ...], tuple[TargetArtifactsV1, ...]]:
    reply_root = build_root / ".cmake" / "api" / "v1" / "reply"
    try:
        indexes = sorted(reply_root.glob("index-*.json"))
    except OSError as exc:
        raise BuildArtifactError("CMake File API reply cannot be listed") from exc
    if len(indexes) != 1:
        raise BuildArtifactError("CMake File API must publish exactly one index")
    raw: dict[str, bytes] = {}

    def load(name: str) -> dict[str, object]:
        if PurePosixPath(name).name != name or not name.endswith(".json"):
            raise BuildArtifactError("CMake File API referenced an unsafe reply name")
        # Anchor the read at build_root, not reply_root, so the whole
        # .cmake/api/v1/reply ancestry is traversed descriptor-anchored no-follow: a
        # symlinked .cmake, api, v1, or reply component fails closed.
        content = _read_regular(build_root, reply_root / name, _FILE_LIMIT)
        previous = raw.setdefault(name, content)
        if previous != content:
            raise BuildArtifactError("CMake File API reply changed while reading")
        try:
            value = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BuildArtifactError("CMake File API reply is invalid JSON") from exc
        if not isinstance(value, dict):
            raise BuildArtifactError("CMake File API reply is not an object")
        return value

    index = load(indexes[0].name)
    reply = index.get("reply")
    client = reply.get("client-strixlab") if isinstance(reply, dict) else None
    query = client.get("query.json") if isinstance(client, dict) else None
    responses = query.get("responses") if isinstance(query, dict) else None
    if not isinstance(responses, list):
        raise BuildArtifactError("CMake File API client responses are missing")
    codemodel_files = [
        response.get("jsonFile")
        for response in responses
        if isinstance(response, dict)
        and response.get("kind") == "codemodel"
        and _is_codemodel_v2(response.get("version"))
    ]
    if len(codemodel_files) != 1 or not isinstance(codemodel_files[0], str):
        raise BuildArtifactError("CMake File API codemodel response is ambiguous")
    codemodel = load(codemodel_files[0])
    if codemodel.get("kind") != "codemodel" or not _is_codemodel_v2(codemodel.get("version")):
        raise BuildArtifactError("CMake codemodel object has an unexpected kind or version")
    configurations = codemodel.get("configurations")
    if not isinstance(configurations, list):
        raise BuildArtifactError("CMake codemodel configurations are missing")
    matches = [
        config
        for config in configurations
        if isinstance(config, dict) and config.get("name") in {build_type, ""}
    ]
    if len(matches) != 1:
        raise BuildArtifactError("CMake codemodel configuration is ambiguous")
    entries = matches[0].get("targets")
    if not isinstance(entries, list):
        raise BuildArtifactError("CMake codemodel targets are missing")
    by_name: dict[str, list[dict[str, object]]] = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("name"), str):
            by_name.setdefault(entry["name"], []).append(entry)
    targets: list[TargetArtifactsV1] = []
    for name in sorted(requested):
        candidates = by_name.get(name, [])
        if len(candidates) != 1:
            raise BuildArtifactError(f"requested CMake target is absent or ambiguous: {name}")
        entry = candidates[0]
        target_file = entry.get("jsonFile")
        target_id = entry.get("id")
        if not isinstance(target_file, str) or not isinstance(target_id, str):
            raise BuildArtifactError(f"requested CMake target reply is incomplete: {name}")
        detail = load(target_file)
        # The target reply must self-identify as the codemodel reference that named
        # it; a mismatch means the File API replies are inconsistent or swapped.
        if detail.get("id") != target_id or detail.get("name") != name:
            raise BuildArtifactError(
                f"CMake target reply identity diverged from its codemodel reference: {name}"
            )
        target_type = detail.get("type")
        artifact_entries = detail.get("artifacts")
        if target_type not in _TARGET_TYPES or not isinstance(artifact_entries, list):
            raise BuildArtifactError(f"requested CMake target has no supported artifacts: {name}")
        paths: list[str] = []
        for artifact in artifact_entries:
            if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
                raise BuildArtifactError(f"requested CMake target artifact is malformed: {name}")
            paths.append(_normalize_relative(artifact["path"]))
        if not paths:
            raise BuildArtifactError(f"requested CMake target has no artifacts: {name}")
        targets.append(
            TargetArtifactsV1(
                name=name,
                target_id=target_id,
                target_type=target_type,
                artifacts=tuple(sorted(set(paths))),
            )
        )
    return tuple(RawReply(name, raw[name]) for name in sorted(raw)), tuple(targets)


def _initial_artifacts(
    build_root: Path, targets: tuple[TargetArtifactsV1, ...]
) -> dict[str, ArtifactV1]:
    associations: dict[str, set[str]] = {}
    for target in targets:
        for path in target.artifacts:
            associations.setdefault(path, set()).add(target.name)
    artifacts = {
        path: _artifact(build_root, path, targets=tuple(sorted(names)))
        for path, names in associations.items()
    }
    by_name = {target.name: target for target in targets}
    for artifact in artifacts.values():
        for name in artifact.targets:
            _enforce_target_format(by_name[name], artifact)
    return artifacts


def _enforce_target_format(target: TargetArtifactsV1, artifact: ArtifactV1) -> None:
    if target.target_type in _ARCHIVE_TARGET_TYPES:
        if artifact.kind != "archive":
            raise BuildArtifactError(
                f"target {target.name} expects an archive but produced {artifact.kind}"
            )
        return
    allowed = _TARGET_ELF_TYPES.get(target.target_type)
    if allowed is None:
        return
    if artifact.kind != "elf" or artifact.elf_type not in allowed:
        raise BuildArtifactError(
            f"target {target.name} ({target.target_type}) produced an incompatible "
            f"artifact: {artifact.kind}/{artifact.elf_type}"
        )


def _artifact(
    root: Path,
    relative: str,
    *,
    targets: tuple[str, ...],
    runtime_dependency: bool = False,
) -> ArtifactV1:
    path = root / PurePosixPath(_normalize_relative(relative))
    metadata, sha256, header = _hash_regular_streaming(root, path)
    if header.startswith(_ELF_MAGIC):
        kind: Literal["elf", "archive"] = "elf"
        elf_type: ElfType | None = _elf_type(header)
    elif header.startswith(_AR_MAGIC):
        kind = "archive"
        elf_type = None
    else:
        raise BuildArtifactError(f"mandatory build artifact has an unsupported format: {relative}")
    return ArtifactV1(
        path=_relative_file(root, path),
        kind=kind,
        elf_type=elf_type,
        mode=stat.S_IMODE(metadata.st_mode),
        size_bytes=metadata.st_size,
        sha256=sha256,
        targets=targets,
        runtime_dependency=runtime_dependency,
    )


def _capture_tools(
    environment: Mapping[str, str],
    cwd: Path,
    timeout: float,
    process_root: Path,
    runner: ProcessRunner,
) -> tuple[tuple[CaptureToolV1, ...], tuple[tuple[str, ProcessResult], ...]]:
    tools: list[CaptureToolV1] = []
    results: list[tuple[str, ProcessResult]] = []
    search_path = environment.get("PATH")
    if search_path is None:
        raise BuildArtifactError("build environment PATH is missing")
    for name in ("ldd", "readelf"):
        selected = shutil.which(name, path=search_path)
        if selected is None:
            raise BuildArtifactError(f"required artifact inspection tool is unavailable: {name}")
        realpath = Path(selected).resolve(strict=True)
        metadata, sha256, _header = _hash_regular_streaming(
            realpath.parent, realpath, require_owner=False
        )
        if metadata.st_mode & 0o111 == 0:
            raise BuildArtifactError(f"artifact inspection tool is not executable: {name}")
        result = _run(
            runner,
            (str(realpath), "--version"),
            cwd=cwd,
            environment=environment,
            timeout=timeout,
            process_root=process_root,
            label=f"capture-tool-{name}",
        )
        _require_observation(result, f"capture tool {name}", allow_nonzero=False)
        tools.append(
            CaptureToolV1(
                name=name,
                path=selected,
                realpath=str(realpath),
                mode=stat.S_IMODE(metadata.st_mode),
                size_bytes=metadata.st_size,
                sha256=sha256,
                version_sha256=process_result_digest(result),
            )
        )
        results.append((f"capture-tool-{name}", result))
    return tuple(tools), tuple(results)


def _inspect_dynamic(
    artifact: Path,
    relative: str,
    elf_type: ElfType,
    root: Path,
    readelf: Path,
    ldd: Path,
    environment: Mapping[str, str],
    timeout: float,
    process_root: Path,
    runner: ProcessRunner,
) -> tuple[DynamicInspectionV1, tuple[tuple[str, ProcessResult], ...]]:
    slug = hashlib.sha256(relative.encode()).hexdigest()[:16]
    readelf_result = _run(
        runner,
        (str(readelf), "-d", str(artifact)),
        cwd=root,
        environment=environment,
        timeout=timeout,
        process_root=process_root,
        label=f"readelf-{slug}",
    )
    _require_observation(readelf_result, f"readelf for {relative}", allow_nonzero=False)
    dynamic_section = _parse_readelf_dynamic(readelf_result, relative)
    results: list[tuple[str, ProcessResult]] = [(f"readelf-{slug}", readelf_result)]
    # ldd only applies to dynamic ELF executables/shared objects; running it on a
    # relocatable object (ET_REL) or archive is meaningless and never attempted.
    dynamic = elf_type in _DYNAMIC_ELF_TYPES
    dependencies: tuple[DynamicDependencyV1, ...] = ()
    static = False
    ldd_digest: str | None = None
    if dynamic:
        ldd_result = _run(
            runner,
            (str(ldd), str(artifact)),
            cwd=root,
            environment=environment,
            timeout=timeout,
            process_root=process_root,
            label=f"ldd-{slug}",
        )
        results.append((f"ldd-{slug}", ldd_result))
        ldd_digest = process_result_digest(ldd_result)
        # Require safe process completion BEFORE recognizing the static form: a
        # timeout, signal, spawn/capture failure, or truncated output must fail even
        # if a static marker happens to appear in the (possibly truncated) stderr.
        # allow_nonzero here only tolerates the static form's documented nonzero exit.
        _require_observation(ldd_result, f"ldd for {relative}", allow_nonzero=True)
        _reject_truncated(ldd_result, f"ldd for {relative}")
        static = any(marker in ldd_result.stderr.lower() for marker in _LDD_STATIC)
        if not static:
            # A genuinely dynamic ldd must exit zero; the static nonzero exit is only
            # accepted above once completion has been proven safe and untruncated.
            _require_observation(ldd_result, f"ldd for {relative}", allow_nonzero=False)
            dependencies = _parse_ldd(ldd_result, root)
    return (
        DynamicInspectionV1(
            artifact=relative,
            elf_type=elf_type,
            dynamic=dynamic,
            static=static,
            needed=dynamic_section.needed,
            soname=dynamic_section.soname,
            rpath=dynamic_section.rpath,
            runpath=dynamic_section.runpath,
            dependencies=dependencies,
            readelf_sha256=process_result_digest(readelf_result),
            ldd_sha256=ldd_digest,
        ),
        tuple(results),
    )


def _reject_truncated(result: ProcessResult, description: str) -> None:
    if result.stdout_truncated or result.stderr_truncated:
        raise BuildArtifactError(f"{description} output was truncated")


@dataclass(frozen=True, slots=True)
class _ReadelfDynamic:
    needed: tuple[str, ...]
    soname: str | None
    rpath: tuple[str, ...]
    runpath: tuple[str, ...]


def _readelf_bracketed(line: str, relative: str, tag: str) -> str:
    """Extract the ``[value]`` payload from one ``readelf -d`` dynamic entry."""

    start = line.find("[")
    end = line.find("]", start + 1)
    if start == -1 or end == -1:
        raise BuildArtifactError(f"readelf {tag} entry is malformed for {relative}")
    return line[start + 1 : end]


def _parse_readelf_dynamic(result: ProcessResult, relative: str) -> _ReadelfDynamic:
    """Parse the NEEDED, SONAME, RPATH, and RUNPATH entries from ``readelf -d``."""

    _reject_truncated(result, f"readelf for {relative}")
    needed: list[str] = []
    soname: str | None = None
    rpath: list[str] = []
    runpath: list[str] = []
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if "(NEEDED)" in line:
            needed.append(_readelf_bracketed(line, relative, "NEEDED"))
        elif "(SONAME)" in line:
            if soname is not None:
                raise BuildArtifactError(f"readelf reported multiple SONAME entries for {relative}")
            soname = _readelf_bracketed(line, relative, "SONAME")
        elif "(RPATH)" in line:
            rpath.extend(_split_search_path(_readelf_bracketed(line, relative, "RPATH")))
        elif "(RUNPATH)" in line:
            runpath.extend(_split_search_path(_readelf_bracketed(line, relative, "RUNPATH")))
    return _ReadelfDynamic(tuple(needed), soname, tuple(rpath), tuple(runpath))


def _split_search_path(value: str) -> list[str]:
    """Split a colon-separated RPATH/RUNPATH value, dropping empty segments."""

    return [segment for segment in value.split(":") if segment]


def _parse_ldd(result: ProcessResult, root: Path) -> tuple[DynamicDependencyV1, ...]:
    _reject_truncated(result, "ldd")
    dependencies: list[DynamicDependencyV1] = []
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if not line or line.startswith("linux-vdso"):
            continue
        if "=>" in line:
            name, value = (part.strip() for part in line.split("=>", 1))
        else:
            fields = line.split()
            if not fields or not fields[0].startswith("/"):
                raise BuildArtifactError("ldd output is malformed")
            name, value = Path(fields[0]).name, fields[0]
        if value == "not found":
            dependencies.append(
                DynamicDependencyV1(name=name, path=None, resolved=False, in_build_root=False)
            )
            continue
        resolved = Path(value.split(" ", 1)[0])
        if not resolved.is_absolute():
            raise BuildArtifactError("ldd dependency path is not absolute")
        try:
            canonical = resolved.resolve(strict=True)
        except OSError as exc:
            raise BuildArtifactError("ldd dependency cannot be resolved") from exc
        try:
            canonical.relative_to(root)
            inside = True
        except ValueError:
            inside = False
        dependencies.append(
            DynamicDependencyV1(name=name, path=str(canonical), resolved=True, in_build_root=inside)
        )
    if any(not dependency.resolved for dependency in dependencies):
        raise BuildArtifactError("ldd reported an unresolved dependency")
    return tuple(dependencies)


def _run(
    runner: ProcessRunner,
    argv: tuple[str, ...],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout: float,
    process_root: Path,
    label: str,
) -> ProcessResult:
    logs = process_root / "artifact-logs"
    logs.mkdir(mode=0o700, exist_ok=True)
    return runner(
        argv,
        cwd=cwd,
        timeout=timeout,
        inherit_env=False,
        base_env=environment,
        output_limit_bytes=_OUTPUT_LIMIT,
        stdout_spool=logs / f"{label}.stdout",
        stderr_spool=logs / f"{label}.stderr",
        spool_root=process_root,
    )


def _require_observation(result: ProcessResult, description: str, *, allow_nonzero: bool) -> None:
    if (
        result.outcome is not ProcessOutcome.EXITED
        or result.returncode is None
        or result.returncode < 0
        or result.capture_error is not None
        or (not allow_nonzero and result.returncode != 0)
    ):
        raise BuildArtifactError(f"{description} failed")


def _normalize_relative(value: str) -> str:
    candidate = PurePosixPath(value)
    if (
        not value
        or "\x00" in value
        or candidate.is_absolute()
        or candidate == PurePosixPath(".")
        or ".." in candidate.parts
        or candidate.as_posix() != value
    ):
        raise BuildArtifactError(f"artifact path is not normalized and relative: {value}")
    return value


def _relative_file(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise BuildArtifactError("artifact path escapes the build root") from exc
    return _normalize_relative(PurePosixPath(*relative.parts).as_posix())


def _read_regular(root: Path, path: Path, limit: int) -> bytes:
    return _read_regular_with_stat(root, path, limit)[0]


def _directory_open_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _open_regular_nofollow(
    root: Path, path: Path, *, require_owner: bool
) -> tuple[int, os.stat_result]:
    """Open ``path`` read-only with descriptor-anchored, no-follow containment.

    Each path component is opened relative to the previous directory descriptor
    with ``O_DIRECTORY | O_NOFOLLOW``, and the final component with ``O_NOFOLLOW``,
    so a symlink swapped in for any ancestor or the file itself between checks
    cannot redirect the open outside the trusted root — the kernel never resolves a
    symlink at any step. Ownership and type are verified on the opened descriptors,
    not on a racy full-path ``lstat``.
    """

    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise BuildArtifactError("evidence path escapes its trusted root") from exc
    parts = relative.parts
    if not parts:
        raise BuildArtifactError("evidence path resolves to its trusted root")
    euid = os.geteuid()
    dir_flags = _directory_open_flags()
    open_dirs: list[int] = []
    try:
        try:
            current_fd = os.open(root, dir_flags)
        except OSError as exc:
            raise BuildArtifactError("evidence root is not a trusted directory") from exc
        open_dirs.append(current_fd)
        for part in parts[:-1]:
            try:
                child_fd = os.open(part, dir_flags, dir_fd=current_fd)
            except OSError as exc:
                raise BuildArtifactError("evidence path crosses an unsafe directory") from exc
            open_dirs.append(child_fd)
            metadata = os.fstat(child_fd)
            if require_owner and metadata.st_uid != euid:
                raise BuildArtifactError("evidence path crosses an unowned directory")
            current_fd = child_fd
        try:
            descriptor = os.open(parts[-1], readonly_open_flags(), dir_fd=current_fd)
        except OSError as exc:
            raise BuildArtifactError(
                f"required evidence file cannot be opened: {parts[-1]}"
            ) from exc
    finally:
        for opened in open_dirs:
            os.close(opened)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or (require_owner and metadata.st_uid != euid):
        os.close(descriptor)
        raise BuildArtifactError("evidence file is not a trusted regular file")
    return descriptor, metadata


def _read_regular_with_stat(
    root: Path, path: Path, limit: int, *, require_owner: bool = True
) -> tuple[bytes, os.stat_result]:
    descriptor, metadata = _open_regular_nofollow(root, path, require_owner=require_owner)
    try:
        if metadata.st_size > limit:
            raise BuildArtifactError("evidence file exceeds the capture limit")
        content = bytearray()
        while True:
            chunk = os.read(descriptor, min(_READ_CHUNK, limit + 1 - len(content)))
            if not chunk:
                break
            content.extend(chunk)
            if len(content) > limit:
                raise BuildArtifactError("evidence file exceeds the capture limit")
        if _stable_stat(os.fstat(descriptor)) != _stable_stat(metadata):
            raise BuildArtifactError("evidence file changed while reading")
        return bytes(content), metadata
    finally:
        os.close(descriptor)


def _hash_regular_streaming(
    root: Path, path: Path, *, require_owner: bool = True
) -> tuple[os.stat_result, str, bytes]:
    """Stream-hash a regular file without a size cap or whole-file buffer.

    Only a small header is retained (for format/ELF-type detection); the digest
    is computed over bounded chunks, and pre/post descriptor metadata must match
    so a concurrent replacement fails closed.
    """

    descriptor, pre = _open_regular_nofollow(root, path, require_owner=require_owner)
    try:
        digest = hashlib.sha256()
        header = b""
        total = 0
        while chunk := os.read(descriptor, _STREAM_CHUNK):
            digest.update(chunk)
            if len(header) < _HEADER_BYTES:
                header += chunk[: _HEADER_BYTES - len(header)]
            total += len(chunk)
        post = os.fstat(descriptor)
        if _stable_stat(pre) != _stable_stat(post) or total != pre.st_size:
            raise BuildArtifactError("artifact changed while hashing")
        return pre, digest.hexdigest(), header
    finally:
        os.close(descriptor)


def _elf_type(header: bytes) -> ElfType:
    if len(header) < 18:
        raise BuildArtifactError("ELF header is truncated")
    encoding = header[5]
    if encoding == 1:
        byteorder: Literal["little", "big"] = "little"
    elif encoding == 2:
        byteorder = "big"
    else:
        raise BuildArtifactError("ELF header has an unknown data encoding")
    return _ELF_TYPES.get(int.from_bytes(header[16:18], byteorder), "ET_OTHER")


def _stable_stat(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_mode,
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )

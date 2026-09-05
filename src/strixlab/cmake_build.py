"""Two-phase CMake/Ninja build adapter with identity-checked selections."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import stat
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from strixlab.build_artifacts import (
    BuildArtifactsV1,
    capture_build_artifacts,
    prepare_file_api_query,
    verify_artifact_capture,
)
from strixlab.build_cache import (
    BuildCacheError,
    BuildIdentityProjectionV1,
    BuildRootOwnerV1,
    CacheClassification,
    CanonicalBuildRecordV1,
    EvidenceRefV1,
    IdentityEntryV1,
    ProducerProvenanceV1,
    SourceBlobRefV1,
    SourcePatchRefV1,
    SourceReproducerV1,
    build_cache_session,
    cache_environment_projection,
    identity_models,
    publish_build_attestation,
    remove_owned_build_root,
    tool_models,
    verify_build_root_owner,
    write_build_root_owner,
)
from strixlab.build_identity import (
    IdentityEntry,
    ToolObservation,
)
from strixlab.build_identity import (
    build_id as compute_build_id,
)
from strixlab.build_identity import (
    recipe_id as compute_recipe_id,
)
from strixlab.build_snapshot import SourceSnapshot, lease_snapshot, verify_snapshot
from strixlab.builds import (
    AttemptOutcome,
    AttemptResult,
    BuildAttemptSession,
    begin_build_attempt,
    build_snapshot_directory,
)
from strixlab.manifests import BuildProfileV1
from strixlab.process import ProcessOutcome, ProcessResult, process_result_digest, run_process
from strixlab.secure_fs import fsync_tree, readonly_open_flags
from strixlab.serialization import canonical_json_bytes
from strixlab.sources import SourceEvidence, SourceLease, lease_source

_VERSION_LIMIT = 256 * 1024
_CACHE_LIMIT = 16 * 1024 * 1024
_LLAMA_CPP_TARGETS = frozenset({"llama-bench", "llama-server", "test-backend-ops"})
_NATIVE_ADAPTER = "strixlab_native"
_NATIVE_TARGET = "topk_capsule_host_test"
_NATIVE_METADATA_KEYS = frozenset({"STRIXLAB_NATIVE_BUILD_COMMIT", "STRIXLAB_NATIVE_BUILD_NUMBER"})
_NATIVE_GPU_KEYS = frozenset(
    {
        "AMDGPU_TARGETS",
        "CMAKE_HIP_ARCHITECTURES",
        "CMAKE_HIP_COMPILER",
        "CMAKE_HIP_COMPILER_ROCM_ROOT",
    }
)
_RESERVED_CMAKE_KEYS = frozenset(
    {
        "CMAKE_AR",
        "CMAKE_BUILD_TYPE",
        "CMAKE_C_COMPILER",
        "CMAKE_CXX_COMPILER",
        "CMAKE_EXPORT_COMPILE_COMMANDS",
        "CMAKE_GENERATOR",
        "CMAKE_HIP_COMPILER",
        "CMAKE_LINKER",
        "CMAKE_MAKE_PROGRAM",
        "GGML_BUILD_COMMIT",
        "GGML_BUILD_NUMBER",
        "LLAMA_BUILD_COMMIT",
        "LLAMA_BUILD_NUMBER",
    }
)


class CMakeBuildError(RuntimeError):
    """A build tool, configure, selection, or build step failed."""


@dataclass(frozen=True, slots=True)
class ToolProbe:
    observations: tuple[ToolObservation, ...]
    results: tuple[tuple[str, ProcessResult], ...]


@dataclass(frozen=True, slots=True)
class _SourceVersion:
    base_commit: str
    patched: bool

    @property
    def build_commit(self) -> str:
        suffix = "-dirty" if self.patched else ""
        return f"{self.base_commit}{suffix}"


@dataclass(frozen=True, slots=True)
class CMakeBuildResult:
    recipe_id: str
    build_id: str
    snapshot: SourceSnapshot
    build_root: Path
    selections: tuple[IdentityEntry, ...]
    tools: tuple[ToolObservation, ...]
    artifacts: BuildArtifactsV1
    execution_class: Literal["built", "cache-hit", "rehydrated", "recovered"]
    canonical_record_sha256: str
    attempt: AttemptResult


ProcessRunner = Callable[..., ProcessResult]


def _sha256_file(path: Path) -> tuple[int, int, str]:
    descriptor = os.open(path, readonly_open_flags())
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CMakeBuildError(f"build tool is not a regular file: {path}")
        while chunk := os.read(descriptor, 64 * 1024):
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or size != before.st_size
        ):
            raise CMakeBuildError(f"build tool changed while hashing: {path}")
    finally:
        os.close(descriptor)
    return stat.S_IMODE(before.st_mode), size, digest.hexdigest()


def _tool_paths(profile: BuildProfileV1) -> tuple[tuple[str, Path], ...]:
    values = [
        ("cmake", Path(profile.toolchain.cmake)),
        ("ninja", Path(profile.toolchain.ninja)),
        ("c_compiler", Path(profile.toolchain.c_compiler)),
        ("cxx_compiler", Path(profile.toolchain.cxx_compiler)),
    ]
    if profile.toolchain.hip_compiler is not None:
        values.append(("hip_compiler", Path(profile.toolchain.hip_compiler)))
    return tuple(values)


def _private_environment(profile: BuildProfileV1, root: Path) -> dict[str, str]:
    private_home = root / "home"
    temporary = root / "tmp"
    cache = private_home / ".cache"
    for path in (private_home, temporary, cache):
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    environment = {
        "HOME": str(private_home),
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.pathsep.join(profile.toolchain.path),
        "TMPDIR": str(temporary),
        "TZ": "UTC",
    }
    for name, value in profile.environment.path_lists.items():
        environment[str(name)] = value
    for literal_name, literal_value in profile.environment.literals.items():
        environment[str(literal_name)] = literal_value
    if any("\x00" in name or "\x00" in value for name, value in environment.items()):
        raise CMakeBuildError("build environment contains a NUL byte")
    return environment


def _ensure_owned_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        path.mkdir(mode=0o700)
        metadata = path.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise CMakeBuildError(f"build directory is unsafe: {path}")


def _prepare_build_root(container: Path, root: Path) -> None:
    _ensure_owned_directory(container)
    try:
        relative = root.relative_to(container)
    except ValueError as exc:
        raise CMakeBuildError("build root escapes its owned container") from exc
    current = container
    for part in relative.parts[:-1]:
        current /= part
        _ensure_owned_directory(current)
    try:
        root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise CMakeBuildError(f"independent build root already exists: {root}") from exc


def _read_cache(path: Path) -> bytes:
    try:
        descriptor = os.open(path, readonly_open_flags())
    except OSError as exc:
        raise CMakeBuildError("configure did not produce a safe CMakeCache.txt") from exc
    chunks: list[bytes] = []
    size = 0
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise CMakeBuildError("CMake cache is not an owned regular file")
        while chunk := os.read(descriptor, 64 * 1024):
            size += len(chunk)
            if size > _CACHE_LIMIT:
                raise CMakeBuildError("CMake cache exceeds the size limit")
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def _require_success(result: ProcessResult, description: str) -> None:
    if (
        result.outcome is not ProcessOutcome.EXITED
        or result.returncode != 0
        or result.capture_error is not None
    ):
        raise CMakeBuildError(f"{description} failed")


def probe_tools(
    profile: BuildProfileV1,
    *,
    cwd: Path,
    environment: Mapping[str, str],
    runner: ProcessRunner = run_process,
) -> ToolProbe:
    """Hash exact configured tools and capture deterministic version probes."""

    observations: list[ToolObservation] = []
    results: list[tuple[str, ProcessResult]] = []
    search_path = environment["PATH"]
    for role, configured in _tool_paths(profile):
        try:
            realpath = configured.resolve(strict=True)
        except OSError as exc:
            raise CMakeBuildError(f"configured {role} is unavailable") from exc
        discovered = shutil.which(configured.name, path=search_path)
        if discovered is None or Path(discovered).resolve(strict=True) != realpath:
            raise CMakeBuildError(f"configured {role} does not match PATH discovery")
        mode, size, digest = _sha256_file(realpath)
        result = runner(
            (str(configured), "--version"),
            cwd=cwd,
            timeout=profile.execution.timeouts.discovery_seconds,
            inherit_env=False,
            base_env=environment,
            output_limit_bytes=_VERSION_LIMIT,
        )
        _require_success(result, f"{role} version probe")
        results.append((role, result))
        observations.append(
            ToolObservation(
                role=role,
                path=str(configured),
                realpath=str(realpath),
                mode=mode,
                size_bytes=size,
                sha256=digest,
                version_sha256=process_result_digest(result),
                search_sha256=hashlib.sha256(
                    canonical_json_bytes(
                        {"name": configured.name, "resolved": str(Path(discovered))}
                    )
                ).hexdigest(),
            )
        )
    return ToolProbe(tuple(observations), tuple(results))


def probe_selected_tools(
    cache: Mapping[str, str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout: float,
    runner: ProcessRunner = run_process,
) -> ToolProbe:
    """Hash and version-probe helper executables selected by CMake itself."""

    observations: list[ToolObservation] = []
    results: list[tuple[str, ProcessResult]] = []
    for role, key in (("linker", "CMAKE_LINKER"), ("archiver", "CMAKE_AR")):
        selected = Path(_required(cache, key))
        if not selected.is_absolute() or os.path.normpath(selected) != str(selected):
            raise CMakeBuildError(f"CMake cache selection is not normalized and absolute: {key}")
        try:
            realpath = selected.resolve(strict=True)
        except OSError as exc:
            raise CMakeBuildError(f"CMake selected an unavailable helper: {key}") from exc
        mode, size, digest = _sha256_file(realpath)
        result = runner(
            (str(realpath), "--version"),
            cwd=cwd,
            timeout=timeout,
            inherit_env=False,
            base_env=environment,
            output_limit_bytes=_VERSION_LIMIT,
        )
        _require_success(result, f"selected {role} version probe")
        results.append((role, result))
        observations.append(
            ToolObservation(
                role=role,
                path=str(selected),
                realpath=str(realpath),
                mode=mode,
                size_bytes=size,
                sha256=digest,
                version_sha256=process_result_digest(result),
                search_sha256=hashlib.sha256(
                    canonical_json_bytes({"cache_key": key, "resolved": str(realpath)})
                ).hexdigest(),
            )
        )
    return ToolProbe(tuple(observations), tuple(results))


def _cmake_scalar(value: bool | int | float | str) -> str:
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    return str(value)


def _source_version(preparation: SourceLease) -> _SourceVersion:
    evidence = preparation.evidence
    return _SourceVersion(
        evidence.base_commit,
        bool(evidence.status or evidence.diff_size_bytes),
    )


def configure_command(
    profile: BuildProfileV1,
    source: Path,
    build: Path,
    *,
    source_version: _SourceVersion,
    source_adapter: str = "llama_cpp",
) -> tuple[str, ...]:
    """Return the deterministic CMake configure command for one fresh root."""

    reserved = _RESERVED_CMAKE_KEYS
    if source_adapter == _NATIVE_ADAPTER:
        reserved |= _NATIVE_METADATA_KEYS | _NATIVE_GPU_KEYS | {"CMAKE_HOME_DIRECTORY"}
    overlap = reserved.intersection(profile.cmake)
    if overlap:
        raise CMakeBuildError(
            "build profile overrides adapter-owned CMake keys: " + ", ".join(sorted(overlap))
        )
    command = [
        profile.toolchain.cmake,
        "-S",
        str(source),
        "-B",
        str(build),
        "-G",
        profile.generator,
        f"-DCMAKE_BUILD_TYPE={profile.build_type}",
        f"-DCMAKE_MAKE_PROGRAM={profile.toolchain.ninja}",
        f"-DCMAKE_C_COMPILER={profile.toolchain.c_compiler}",
        f"-DCMAKE_CXX_COMPILER={profile.toolchain.cxx_compiler}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
    ]
    for prefix in _source_version_prefixes(source_adapter):
        command.extend(
            (f"-D{prefix}_BUILD_COMMIT={source_version.build_commit}", f"-D{prefix}_BUILD_NUMBER=0")
        )
    if profile.toolchain.hip_compiler is not None:
        command.append(f"-DCMAKE_HIP_COMPILER={profile.toolchain.hip_compiler}")
    command.extend(
        f"-D{name}={_cmake_scalar(value)}" for name, value in sorted(profile.cmake.items())
    )
    return tuple(command)


def parse_cmake_cache(content: bytes) -> dict[str, str]:
    """Parse unique non-comment CMake cache entries without interpreting values."""

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CMakeBuildError("CMake cache is not UTF-8") from exc
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line or line.startswith(("#", "//")):
            continue
        key_type, separator, value = line.partition("=")
        key, type_separator, _entry_type = key_type.partition(":")
        if not separator or not type_separator or not key or "\x00" in value:
            raise CMakeBuildError("CMake cache contains an invalid entry")
        if key in values:
            raise CMakeBuildError(f"CMake cache contains duplicate key: {key}")
        values[key] = value
    return values


def _required(cache: Mapping[str, str], key: str) -> str:
    value = cache.get(key)
    if value is None or not value:
        raise CMakeBuildError(f"CMake cache is missing required selection: {key}")
    return value


def _source_version_prefixes(source_adapter: str) -> tuple[str, ...]:
    if source_adapter == "llama_cpp":
        return ("GGML", "LLAMA")
    if source_adapter == _NATIVE_ADAPTER:
        return ("STRIXLAB_NATIVE",)
    raise CMakeBuildError("CMake build source adapter is not supported")


def _verify_source_version(
    cache: Mapping[str, str], expected: _SourceVersion, *, source_adapter: str = "llama_cpp"
) -> None:
    for prefix in _source_version_prefixes(source_adapter):
        if _required(cache, f"{prefix}_BUILD_COMMIT") != expected.build_commit:
            raise CMakeBuildError("CMake selected unexpected source-version metadata")
        if _required(cache, f"{prefix}_BUILD_NUMBER") != "0":
            raise CMakeBuildError("CMake selected unexpected source build number")


def _configuration_source(snapshot: SourceSnapshot, source_adapter: str) -> Path:
    if source_adapter != _NATIVE_ADAPTER:
        return snapshot.source
    source = snapshot.source
    for part in ("native", "topk", "CMakeLists.txt"):
        source = source / part
        metadata = source.lstat()
        expected_type = stat.S_ISREG if part == "CMakeLists.txt" else stat.S_ISDIR
        if not expected_type(metadata.st_mode):
            raise CMakeBuildError(
                "native CMake source must use real directories and CMakeLists.txt"
            )
    return source.parent


def _verify_native_configuration(cache: Mapping[str, str], source: Path) -> None:
    if _required(cache, "CMAKE_HOME_DIRECTORY") != str(source):
        raise CMakeBuildError(
            "native CMake source directory does not match the fixed snapshot path"
        )
    if any(cache.get(key, "") for key in _NATIVE_GPU_KEYS):
        raise CMakeBuildError("native host fixture cannot select a HIP compiler or gfx target")
    if any(
        cache.get(f"{prefix}_BUILD_{suffix}", "")
        for prefix in ("GGML", "LLAMA")
        for suffix in ("COMMIT", "NUMBER")
    ):
        raise CMakeBuildError("native host fixture cannot declare llama source metadata")


def _selected_realpath(cache: Mapping[str, str], key: str, configured: str | None = None) -> str:
    value = _required(cache, key)
    path = Path(value)
    if not path.is_absolute() or os.path.normpath(path) != value:
        raise CMakeBuildError(f"CMake cache selection is not normalized and absolute: {key}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CMakeBuildError(f"CMake cache selection is unavailable: {key}") from exc
    if configured is not None:
        try:
            expected = Path(configured).resolve(strict=True)
        except OSError as exc:
            raise CMakeBuildError(f"configured path is unavailable for selection: {key}") from exc
        if resolved != expected:
            raise CMakeBuildError(f"CMake selected an unexpected path: {key}")
    return str(resolved)


def _reject_unobserved_path(cache: Mapping[str, str], key: str) -> str:
    if cache.get(key, ""):
        raise CMakeBuildError(f"CMake selection is not supported for reproducible identity: {key}")
    return ""


def selections_from_cache(
    profile: BuildProfileV1, cache: Mapping[str, str]
) -> tuple[IdentityEntry, ...]:
    """Project authoritative CMake selections into the machine build identity."""

    generator = _required(cache, "CMAKE_GENERATOR")
    if generator != profile.generator:
        raise CMakeBuildError("CMake selected an unexpected generator")
    _selected_realpath(cache, "CMAKE_MAKE_PROGRAM", profile.toolchain.ninja)
    values = {
        "generator": generator,
        "c_compiler": _selected_realpath(cache, "CMAKE_C_COMPILER", profile.toolchain.c_compiler),
        "cxx_compiler": _selected_realpath(
            cache, "CMAKE_CXX_COMPILER", profile.toolchain.cxx_compiler
        ),
        "linker": _selected_realpath(cache, "CMAKE_LINKER"),
        "archiver": _selected_realpath(cache, "CMAKE_AR"),
        "toolchain_files": _reject_unobserved_path(cache, "CMAKE_TOOLCHAIN_FILE"),
        "sysroot": _reject_unobserved_path(cache, "CMAKE_SYSROOT"),
    }
    if profile.toolchain.mode == "rocm":
        if profile.toolchain.hip_compiler is None or profile.toolchain.rocm_prefix is None:
            raise CMakeBuildError("ROCm toolchain selections are incomplete")
        gfx_targets = cache.get("CMAKE_HIP_ARCHITECTURES") or cache.get("AMDGPU_TARGETS")
        if not gfx_targets:
            raise CMakeBuildError("CMake cache is missing the gfx target selection")
        values.update(
            hip_compiler=_selected_realpath(
                cache, "CMAKE_HIP_COMPILER", profile.toolchain.hip_compiler
            ),
            rocm_prefix=_selected_realpath(
                cache, "CMAKE_HIP_COMPILER_ROCM_ROOT", profile.toolchain.rocm_prefix
            ),
            gfx_targets=gfx_targets,
        )
    return tuple(IdentityEntry(name, value) for name, value in values.items())


def _process_metadata(result: ProcessResult) -> dict[str, Any]:
    value = asdict(result)
    value["outcome"] = result.outcome
    for key in ("stdout", "stderr", "error"):
        value.pop(key, None)
    for key in ("stdout_spool", "stderr_spool"):
        path = value[key]
        value[key] = None if path is None else Path(path).name
    return value


def _write_process_evidence(attempt: BuildAttemptSession, name: str, result: ProcessResult) -> None:
    attempt.write_evidence(f"process/{name}.json", canonical_json_bytes(_process_metadata(result)))


def _record_failure_evidence(attempt: BuildAttemptSession) -> None:
    """Capture failure evidence without ever masking the originating exception."""

    with contextlib.suppress(Exception):
        if attempt.root.is_dir() and not (attempt.root / "failure.json").exists():
            attempt.write_evidence(
                "failure.json",
                canonical_json_bytes({"code": "cmake-build-failed", "schema_version": 1}),
            )


def _run_configure(
    profile: BuildProfileV1,
    *,
    source: Path,
    source_version: _SourceVersion,
    source_adapter: str,
    build: Path,
    environment: Mapping[str, str],
    attempt: BuildAttemptSession,
    label: str,
    runner: ProcessRunner,
) -> bytes:
    command = configure_command(
        profile, source, build, source_version=source_version, source_adapter=source_adapter
    )
    result = runner(
        command,
        cwd=build,
        timeout=profile.execution.timeouts.configure_seconds,
        inherit_env=False,
        base_env=environment,
        output_limit_bytes=_VERSION_LIMIT,
        stdout_spool=attempt.root / "logs" / f"{label}.stdout",
        stderr_spool=attempt.root / "logs" / f"{label}.stderr",
        spool_root=attempt.root,
    )
    _write_process_evidence(attempt, label, result)
    _require_success(result, label)
    cache = _read_cache(build / "CMakeCache.txt")
    attempt.write_evidence(f"cmake/{label}-cache.txt", cache)
    return cache


def _authorize_build(preparation: SourceLease, profile: BuildProfileV1) -> None:
    evidence = preparation.evidence
    if profile.source != evidence.source_id:
        raise CMakeBuildError("build profile source does not match the preparation")
    if evidence.adapter == _NATIVE_ADAPTER:
        if profile.toolchain.mode != "host" or profile.targets != [_NATIVE_TARGET]:
            raise CMakeBuildError("native build requires the single host fixture target")
        reserved = _RESERVED_CMAKE_KEYS | _NATIVE_METADATA_KEYS | _NATIVE_GPU_KEYS
        if (reserved | {"CMAKE_HOME_DIRECTORY"}).intersection(profile.cmake):
            raise CMakeBuildError("native build profile overrides adapter-owned CMake keys")
        return
    if evidence.adapter != "llama_cpp":
        raise CMakeBuildError("CMake build requires a llama_cpp or strixlab_native source")
    unauthorized = set(profile.targets) - _LLAMA_CPP_TARGETS
    if unauthorized:
        raise CMakeBuildError(
            "build profile requests unauthorized llama_cpp targets: "
            + ", ".join(sorted(unauthorized))
        )


def _write_tool_evidence(attempt: BuildAttemptSession, tools: ToolProbe) -> None:
    for role, result in tools.results:
        attempt.write_evidence(f"tools/{role}.stdout", result.stdout.encode("utf-8"))
        attempt.write_evidence(f"tools/{role}.stderr", result.stderr.encode("utf-8"))
        _write_process_evidence(attempt, f"tool-{role}", result)
    attempt.write_evidence(
        "tools/observations.json",
        canonical_json_bytes([asdict(value) for value in tools.observations]),
    )


def _combine_tool_probes(*probes: ToolProbe) -> ToolProbe:
    return ToolProbe(
        tuple(value for probe in probes for value in probe.observations),
        tuple(value for probe in probes for value in probe.results),
    )


def _revalidate_before_finalize(
    preparation: SourceLease,
    profile: BuildProfileV1,
    *,
    cache: Mapping[str, str],
    snapshot: SourceSnapshot,
    root: Path,
    owner: BuildRootOwnerV1,
    expected_tools: ToolProbe,
    expected_artifacts: BuildArtifactsV1,
    environment: Mapping[str, str],
    attempt: BuildAttemptSession,
    runner: ProcessRunner,
    verify_artifacts: bool = True,
) -> None:
    if preparation.evidence.adapter == _NATIVE_ADAPTER:
        source = _configuration_source(snapshot, preparation.evidence.adapter)
        current_cache = parse_cmake_cache(_read_cache(root / "CMakeCache.txt"))
        _verify_source_version(
            current_cache, _source_version(preparation), source_adapter=preparation.evidence.adapter
        )
        _verify_native_configuration(current_cache, source)
        if selections_from_cache(profile, current_cache) != selections_from_cache(profile, cache):
            raise CMakeBuildError("native CMake selections changed before finalization")
    reverified_tools = _combine_tool_probes(
        probe_tools(profile, cwd=snapshot.source, environment=environment, runner=runner),
        probe_selected_tools(
            cache,
            cwd=snapshot.source,
            environment=environment,
            timeout=profile.execution.timeouts.discovery_seconds,
            runner=runner,
        ),
    )
    attempt.write_evidence(
        "tools/post-build-observations.json",
        canonical_json_bytes([asdict(value) for value in reverified_tools.observations]),
    )
    for role, result in reverified_tools.results:
        _write_process_evidence(attempt, f"post-build-tool-{role}", result)
    if reverified_tools.observations != expected_tools.observations:
        raise CMakeBuildError("build tool observations changed during the build")
    # A rehydrate skips re-hashing the freshly captured artifacts here because
    # cache.publish immediately reverifies the rebuilt root against the retained
    # canonical evidence (a strictly stronger check); a fresh build has no
    # canonical record yet, so it must verify its own captured artifacts.
    if verify_artifacts:
        verify_artifact_capture(
            root,
            expected_artifacts,
            selections=selections_from_cache(profile, cache),
            toolchain_mode=profile.toolchain.mode,
        )
    verify_snapshot(snapshot.root)
    preparation.verify()
    verify_build_root_owner(root, owner)


_DIFF_EVIDENCE_PATH = "source/diff.patch"
_PATCH_EVIDENCE_DIR = "source/patches"
_SOURCE_BLOB_LIMIT = 64 * 1024 * 1024
_SOURCE_BLOB_AGGREGATE = 128 * 1024 * 1024


def _json_normalized(value: Any) -> dict[str, Any]:
    """Round-trip a model dump through canonical JSON so tuples become lists.

    Fresh and reloaded reproducer blobs must compare equal, which fails if one
    still holds tuples where the persisted form holds JSON arrays.
    """

    normalized = json.loads(canonical_json_bytes(value))
    if not isinstance(normalized, dict):
        raise CMakeBuildError("source reproducer evidence is not a JSON object")
    return normalized


def _source_reproducer(evidence: SourceEvidence, snapshot: SourceSnapshot) -> SourceReproducerV1:
    """Build the portable, content-addressed reproducer for one leased source."""

    evidence_dict = _json_normalized(evidence.model_dump(mode="json"))
    diff: SourceBlobRefV1 | None = None
    if evidence.diff_size_bytes > 0:
        diff = SourceBlobRefV1(
            relative_path=_DIFF_EVIDENCE_PATH,
            sha256=evidence.diff_sha256,
            size_bytes=evidence.diff_size_bytes,
        )
    patches = tuple(
        SourcePatchRefV1(
            order=patch.order,
            relative_path=f"{_PATCH_EVIDENCE_DIR}/{patch.order:04d}.patch",
            sha256=patch.sha256,
            size_bytes=patch.size_bytes,
        )
        for patch in evidence.patches
    )
    return SourceReproducerV1(
        candidate_id=evidence.candidate_id,
        content_tree_id=evidence.content_tree_id,
        snapshot_id=snapshot.snapshot_id,
        source_evidence=evidence_dict,
        source_evidence_sha256=hashlib.sha256(canonical_json_bytes(evidence_dict)).hexdigest(),
        snapshot_manifest=_json_normalized(snapshot.manifest.model_dump(mode="json")),
        diff=diff,
        patches=patches,
    )


def _write_artifacts_evidence(attempt: BuildAttemptSession, artifacts: BuildArtifactsV1) -> bytes:
    """Serialize the artifact set once, write ``build/artifacts.json``, return bytes.

    The returned canonical bytes are reused for the provenance inventory digest so
    the fresh-build path never serializes the same artifact set twice.
    """

    artifacts_bytes = canonical_json_bytes(artifacts.model_dump(mode="json"))
    attempt.write_evidence("build/artifacts.json", artifacts_bytes)
    return artifacts_bytes


def _producer_provenance(
    *,
    build_id: str,
    producer_attempt_id: str,
    recipe: str,
    source: SourceReproducerV1,
    artifacts: BuildArtifactsV1,
    attempt: BuildAttemptSession,
) -> ProducerProvenanceV1:
    """Bind the producer attempt to its build with a complete digest-indexed
    inventory of the durable evidence it has written.

    The inventory is the attempt's accumulated ``write_evidence`` manifest — digest
    and size captured from the exact bytes at write time — so no pre-publication
    disk re-read/hash is needed. It covers the artifacts, profile, environment,
    source evidence/snapshot/reproducer bytes, configure caches, compile database,
    File API replies, and tool/process observations (every required item and source
    blob); provenance.json/result.json are written later and so are naturally absent.
    The immutable ``publish_record`` remains the independent on-disk copy/hash
    boundary, and a later tamper still fails through record verification.
    """

    evidence = tuple(
        EvidenceRefV1(relative_path=path, sha256=digest, size_bytes=size)
        for path, digest, size in attempt.evidence_manifest()
    )

    return ProducerProvenanceV1(
        build_id=build_id,
        producer_attempt_id=producer_attempt_id,
        recipe_id=recipe,
        artifact_set_id=artifacts.artifact_set_id,
        candidate_id=source.candidate_id,
        snapshot_id=source.snapshot_id,
        execution_class="built",
        evidence=evidence,
    )


def _build_identity(
    *,
    recipe: str,
    profile: BuildProfileV1,
    environment: tuple[IdentityEntryV1, ...],
    selections: tuple[IdentityEntry, ...],
    tools: tuple[ToolObservation, ...],
    source: SourceReproducerV1,
) -> BuildIdentityProjectionV1:
    """Assemble the reproducible identity every invocation of one build must match."""

    return BuildIdentityProjectionV1(
        recipe_id=recipe,
        profile_sha256=hashlib.sha256(
            canonical_json_bytes(profile.model_dump(mode="json"))
        ).hexdigest(),
        toolchain_mode=profile.toolchain.mode,
        environment=environment,
        requested_targets=tuple(sorted(profile.targets)),
        selections=identity_models(selections),
        tools=tool_models(tools),
        source=source,
    )


def _canonical_from_identity(
    identity: BuildIdentityProjectionV1,
    *,
    build_id: str,
    producer_attempt_id: str,
    artifacts: BuildArtifactsV1,
) -> CanonicalBuildRecordV1:
    """Bind one producer attempt and its artifacts to a reproducible identity."""

    # Carry every projection field forward from ``identity`` via the shared model
    # machinery so a new projection field cannot be dropped from the canonical record.
    projection = {name: getattr(identity, name) for name in BuildIdentityProjectionV1.model_fields}
    return CanonicalBuildRecordV1(
        **projection,
        build_id=build_id,
        producer_attempt_id=producer_attempt_id,
        artifacts=artifacts,
    )


def _read_source_blob(path: Path, sha256: str, size_bytes: int) -> bytes:
    """Read a verified diff/patch blob nofollow, failing closed on any drift."""

    if size_bytes > _SOURCE_BLOB_LIMIT:
        raise CMakeBuildError("source reproducer blob exceeds the size limit")
    try:
        descriptor = os.open(path, readonly_open_flags())
    except OSError as exc:
        raise CMakeBuildError("source reproducer blob is unavailable") from exc
    content = bytearray()
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_size != size_bytes
        ):
            raise CMakeBuildError("source reproducer blob is not a trusted regular file")
        while chunk := os.read(descriptor, 64 * 1024):
            content.extend(chunk)
            if len(content) > size_bytes:
                raise CMakeBuildError("source reproducer blob grew while reading")
    finally:
        os.close(descriptor)
    if len(content) != size_bytes or hashlib.sha256(content).hexdigest() != sha256:
        raise CMakeBuildError("source reproducer blob digest changed")
    return bytes(content)


def _copy_source_reproducer_bytes(
    attempt: BuildAttemptSession, preparation: SourceLease, reproducer: SourceReproducerV1
) -> None:
    """Copy the verified raw diff/patch bytes into immutable attempt evidence."""

    evidence = preparation.evidence
    record = preparation.record
    total = 0
    if reproducer.diff is not None:
        data = _read_source_blob(
            record / evidence.diff_file, reproducer.diff.sha256, reproducer.diff.size_bytes
        )
        total += len(data)
        attempt.write_evidence(reproducer.diff.relative_path, data)
    for patch, ref in zip(evidence.patches, reproducer.patches, strict=True):
        data = _read_source_blob(record / patch.record_file, ref.sha256, ref.size_bytes)
        total += len(data)
        if total > _SOURCE_BLOB_AGGREGATE:
            raise CMakeBuildError("source reproducer bytes exceed the aggregate limit")
        attempt.write_evidence(ref.relative_path, data)


@contextlib.contextmanager
def _build_lifecycle(
    preparation: SourceLease, recipe: str, *, home: Path
) -> Iterator[tuple[BuildAttemptSession, SourceSnapshot]]:
    evidence = preparation.evidence
    snapshots = build_snapshot_directory(home=home)
    with (
        lease_snapshot(
            preparation.worktree,
            snapshots,
            candidate_id=evidence.candidate_id,
            content_tree_id=evidence.content_tree_id,
        ) as snapshot,
        begin_build_attempt(recipe, home=home) as attempt,
    ):
        yield attempt, snapshot


def _execute_leased_build(
    preparation: SourceLease,
    profile: BuildProfileV1,
    *,
    home: Path,
    runner: ProcessRunner = run_process,
) -> CMakeBuildResult:
    evidence = preparation.evidence
    _authorize_build(preparation, profile)
    source_version = _source_version(preparation)
    recipe = compute_recipe_id(evidence.candidate_id, evidence.adapter, profile)
    with _build_lifecycle(preparation, recipe, home=home) as (attempt, snapshot):
        final_root: Path | None = None
        root_owner: BuildRootOwnerV1 | None = None
        artifact_evidence: BuildArtifactsV1 | None = None
        execution_class: Literal["built", "cache-hit", "rehydrated", "recovered"] = "built"
        canonical_record_sha256 = ""
        finalize_started = False
        try:
            attempt.mark_active()
            configure_source = _configuration_source(snapshot, evidence.adapter)
            (attempt.root / "logs").mkdir(mode=0o700)
            environment = _private_environment(profile, attempt.root / "private")
            attempt.write_evidence(
                "build/profile.resolved.json",
                canonical_json_bytes(profile.model_dump(mode="json")),
            )
            attempt.write_evidence(
                "build/environment.json",
                canonical_json_bytes(environment),
            )
            attempt.write_evidence(
                "source/evidence.json",
                canonical_json_bytes(evidence.model_dump(mode="json")),
            )
            preparation.verify()
            attempt.write_evidence(
                "source/snapshot.json",
                canonical_json_bytes(snapshot.manifest.model_dump(mode="json")),
            )
            declared_tools = probe_tools(
                profile, cwd=snapshot.source, environment=environment, runner=runner
            )
            materialized = attempt.materialized
            _prepare_build_root(attempt.root / "private", attempt.probe_root)
            probe_cache_bytes = _run_configure(
                profile,
                source=configure_source,
                source_version=source_version,
                source_adapter=evidence.adapter,
                build=attempt.probe_root,
                environment=environment,
                attempt=attempt,
                label="probe-configure",
                runner=runner,
            )
            probe_cache = parse_cmake_cache(probe_cache_bytes)
            _verify_source_version(probe_cache, source_version, source_adapter=evidence.adapter)
            if evidence.adapter == _NATIVE_ADAPTER:
                _verify_native_configuration(probe_cache, configure_source)
            probe_selections = selections_from_cache(profile, probe_cache)
            selected_tools = probe_selected_tools(
                probe_cache,
                cwd=snapshot.source,
                environment=environment,
                timeout=profile.execution.timeouts.discovery_seconds,
                runner=runner,
            )
            tools = _combine_tool_probes(declared_tools, selected_tools)
            _write_tool_evidence(attempt, tools)
            machine_build_id = compute_build_id(
                recipe,
                profile=profile,
                tools=tools.observations,
                selections=probe_selections,
            )
            candidate_root = attempt.build_root(machine_build_id)
            identity = _build_identity(
                recipe=recipe,
                profile=profile,
                environment=cache_environment_projection(
                    environment,
                    home=home,
                    source_root=snapshot.source,
                    build_root=candidate_root,
                    build_home=attempt.root / "private" / "home",
                    build_tmp=attempt.root / "private" / "tmp",
                ),
                selections=probe_selections,
                tools=tools.observations,
                source=_source_reproducer(evidence, snapshot),
            )
            with build_cache_session(
                machine_build_id, attempt.registry.attempt_id, home=home
            ) as cache:
                lookup = cache.lookup(identity, home=home)
                if lookup.classification is CacheClassification.HIT:
                    assert lookup.canonical is not None
                    assert lookup.canonical_record_sha256 is not None
                    assert lookup.owner is not None
                    final_root = cache.root
                    final_selections = probe_selections
                    artifact_evidence = lookup.canonical.artifacts
                    # An un-attested HIT is a crash-forward completion (PRESENT root
                    # verified, but its producer never finalized SUCCESS). This
                    # attempt genuinely re-verifies the root below and finalizes as
                    # the recovery attestor, publishing the missing attestation
                    # before the recovered cache is treated as reusable.
                    execution_class = "recovered" if lookup.needs_attestation else "cache-hit"
                    canonical_record_sha256 = lookup.canonical_record_sha256
                    _write_artifacts_evidence(attempt, artifact_evidence)
                    # A cache hit reuses the materialized root without rebuilding,
                    # so reverify the whole boundary immediately before success:
                    # source lease, snapshot, current tools, the retained canonical
                    # artifacts, and the exact journal-authenticated owner binding
                    # (attempt/build/uid/dev/inode) returned by lookup.
                    _revalidate_before_finalize(
                        preparation,
                        profile,
                        cache=probe_cache,
                        snapshot=snapshot,
                        root=final_root,
                        owner=lookup.owner,
                        expected_tools=tools,
                        expected_artifacts=artifact_evidence,
                        environment=environment,
                        attempt=attempt,
                        runner=runner,
                    )
                else:
                    rehydrate = lookup.classification is CacheClassification.REHYDRATE
                    cache.begin_materialization(rehydrate=rehydrate)
                    try:
                        _prepare_build_root(materialized, candidate_root)
                        final_root = candidate_root
                        root_owner = write_build_root_owner(
                            final_root, attempt.registry.attempt_id, machine_build_id
                        )
                        cache.bind_root(root_owner)
                        prepare_file_api_query(final_root)
                        final_cache_bytes = _run_configure(
                            profile,
                            source=configure_source,
                            source_version=source_version,
                            source_adapter=evidence.adapter,
                            build=final_root,
                            environment=environment,
                            attempt=attempt,
                            label="final-configure",
                            runner=runner,
                        )
                        final_cache = parse_cmake_cache(final_cache_bytes)
                        _verify_source_version(
                            final_cache, source_version, source_adapter=evidence.adapter
                        )
                        if evidence.adapter == _NATIVE_ADAPTER:
                            _verify_native_configuration(final_cache, configure_source)
                        final_selections = selections_from_cache(profile, final_cache)
                        if final_selections != probe_selections:
                            raise CMakeBuildError("final CMake selections drifted from the probe")
                        build_result = runner(
                            (
                                profile.toolchain.cmake,
                                "--build",
                                str(final_root),
                                "--target",
                                *sorted(profile.targets),
                                "--parallel",
                                str(profile.execution.jobs),
                            ),
                            cwd=final_root,
                            timeout=profile.execution.timeouts.build_seconds,
                            inherit_env=False,
                            base_env=environment,
                            output_limit_bytes=_VERSION_LIMIT,
                            stdout_spool=attempt.root / "logs" / "build.stdout",
                            stderr_spool=attempt.root / "logs" / "build.stderr",
                            spool_root=attempt.root,
                        )
                        _write_process_evidence(attempt, "build", build_result)
                        _require_success(build_result, "CMake build")
                        artifact_capture = capture_build_artifacts(
                            final_root,
                            build_type=profile.build_type,
                            requested_targets=tuple(sorted(profile.targets)),
                            selections=final_selections,
                            toolchain_mode=profile.toolchain.mode,
                            environment=environment,
                            discovery_timeout=profile.execution.timeouts.discovery_seconds,
                            capability_timeout=profile.execution.timeouts.capability_seconds,
                            inspection_timeout=profile.execution.timeouts.inspection_seconds,
                            process_root=attempt.root,
                            runner=runner,
                        )
                        artifact_evidence = artifact_capture.evidence
                        _write_artifacts_evidence(attempt, artifact_evidence)
                        for reply in artifact_capture.raw_replies:
                            attempt.write_evidence(f"cmake/file-api/{reply.name}", reply.content)
                        if artifact_capture.compile_commands is not None:
                            attempt.write_evidence(
                                "cmake/compile_commands.json", artifact_capture.compile_commands
                            )
                        else:
                            attempt.write_evidence(
                                "cmake/compile_commands.absent.json",
                                canonical_json_bytes({"absent": True, "schema_version": 1}),
                            )
                        for name, result in artifact_capture.process_results:
                            _write_process_evidence(attempt, name, result)
                        _revalidate_before_finalize(
                            preparation,
                            profile,
                            cache=final_cache,
                            snapshot=snapshot,
                            root=final_root,
                            owner=root_owner,
                            expected_tools=tools,
                            expected_artifacts=artifact_evidence,
                            environment=environment,
                            attempt=attempt,
                            runner=runner,
                            verify_artifacts=not rehydrate,
                        )
                        if rehydrate and lookup.canonical is not None:
                            producer_attempt_id = lookup.canonical.producer_attempt_id
                        else:
                            producer_attempt_id = attempt.registry.attempt_id
                            _copy_source_reproducer_bytes(attempt, preparation, identity.source)
                            # Bind this producer attempt to the build identity in its
                            # own immutable evidence *before* the canonical record is
                            # published, so inspection can authenticate provenance
                            # without a lock-order inversion or a circular digest. The
                            # inventory covers the File API, artifact, and source
                            # evidence written just above.
                            provenance = _producer_provenance(
                                build_id=machine_build_id,
                                producer_attempt_id=producer_attempt_id,
                                recipe=recipe,
                                source=identity.source,
                                artifacts=artifact_evidence,
                                attempt=attempt,
                            )
                            attempt.write_evidence(
                                "build/provenance.json",
                                canonical_json_bytes(provenance.model_dump(mode="json")),
                            )
                        record = _canonical_from_identity(
                            identity,
                            build_id=machine_build_id,
                            producer_attempt_id=producer_attempt_id,
                            artifacts=artifact_evidence,
                        )
                        # Durably flush the fully validated tree before publication so a
                        # crash after PUBLISHING can only recover a persisted build root.
                        fsync_tree(final_root)
                    except BaseException:
                        # Failures up to and including artifact capture leave the
                        # journal in BUILDING/REHYDRATING; discard the uncommitted
                        # root now so the next lookup recovers cleanly.
                        with contextlib.suppress(CMakeBuildError, BuildCacheError, OSError):
                            remove_owned_build_root(final_root, root_owner)
                        raise
                    # Once publication begins the journal advances to PUBLISHING with
                    # authenticated staging; a failure here is completed forward by
                    # recovery, never by discarding the materialized root.
                    canonical_record_sha256 = cache.publish(record, rehydrate=rehydrate)
                    execution_class = "rehydrated" if rehydrate else "built"
            attempt.write_evidence(
                "build/result.json",
                canonical_json_bytes(
                    {
                        "artifact_set_id": artifact_evidence.artifact_set_id,
                        "build_id": machine_build_id,
                        "build_root": str(final_root),
                        "canonical_record_sha256": canonical_record_sha256,
                        "execution_class": execution_class,
                        "recipe_id": recipe,
                        "schema_version": 1,
                        "selections": [asdict(value) for value in final_selections],
                        "snapshot_id": snapshot.snapshot_id,
                        "targets": sorted(profile.targets),
                    }
                ),
            )
            finalize_started = True
            finalized = attempt.finalize(AttemptOutcome.SUCCESS, build_id=machine_build_id)
            # Post-finalization attestation boundary. This finalized SUCCESS record
            # binds the canonical digest, so publish the immutable attestation that
            # first makes a canonical reusable as a cache HIT: "built" when this
            # attempt produced it, "recovered" when this attempt completed a
            # crash-forward one whose producer never attested. A "rehydrated" run
            # reuses a canonical that was already attested when first completed, and
            # a "cache-hit" reused an already-attested build; neither publishes.
            if execution_class == "built" or execution_class == "recovered":
                publish_build_attestation(
                    machine_build_id,
                    attestor_attempt_id=finalized.attempt_id,
                    canonical_record_sha256=canonical_record_sha256,
                    execution_class=execution_class,
                    artifact_set_id=artifact_evidence.artifact_set_id,
                    home=home,
                )
        except BaseException:
            if not finalize_started:
                _record_failure_evidence(attempt)
            raise
    assert final_root is not None and artifact_evidence is not None
    return CMakeBuildResult(
        recipe,
        machine_build_id,
        snapshot,
        final_root,
        final_selections,
        tools.observations,
        artifact_evidence,
        execution_class,
        canonical_record_sha256,
        finalized,
    )


def execute_cmake_build(
    preparation_id: str,
    profile: BuildProfileV1,
    *,
    home: Path,
    runner: ProcessRunner = run_process,
) -> CMakeBuildResult:
    """Authenticate a source lease, then configure and build it reproducibly."""

    with lease_source(preparation_id, home=home) as preparation:
        return _execute_leased_build(
            preparation,
            profile,
            home=home,
            runner=runner,
        )

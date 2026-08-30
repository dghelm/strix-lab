"""Two-phase CMake/Ninja build adapter with identity-checked selections."""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import stat
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
from strixlab.process import ProcessOutcome, ProcessResult, run_process
from strixlab.secure_fs import fsync_directory, readonly_open_flags, write_exclusive
from strixlab.serialization import canonical_json_bytes
from strixlab.sources import SourceLease, lease_source

_VERSION_LIMIT = 256 * 1024
_CACHE_LIMIT = 16 * 1024 * 1024
_OWNER_LIMIT = 4 * 1024
_LLAMA_CPP_TARGETS = frozenset({"llama-bench", "llama-server", "test-backend-ops"})
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


class _BuildRootOwnerV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    attempt_id: str = Field(pattern=r"^attempt-[0-9a-f]{24}-[0-9a-f]{32}$")
    build_id: str = Field(pattern=r"^build-sha256:[0-9a-f]{64}$")
    root_device: int = Field(ge=0)
    root_inode: int = Field(gt=0)


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


def _result_digest(result: ProcessResult) -> str:
    value = {
        "argv": result.argv,
        "capture_error": result.capture_error,
        "error": result.error,
        "outcome": result.outcome,
        "returncode": result.returncode,
        "stderr_sha256": result.stderr_sha256,
        "stdout_sha256": result.stdout_sha256,
    }
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


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
                version_sha256=_result_digest(result),
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
                version_sha256=_result_digest(result),
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
) -> tuple[str, ...]:
    """Return the deterministic CMake configure command for one fresh root."""

    overlap = _RESERVED_CMAKE_KEYS.intersection(profile.cmake)
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
        f"-DGGML_BUILD_COMMIT={source_version.build_commit}",
        "-DGGML_BUILD_NUMBER=0",
        f"-DLLAMA_BUILD_COMMIT={source_version.build_commit}",
        "-DLLAMA_BUILD_NUMBER=0",
    ]
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


def _verify_source_version(cache: Mapping[str, str], expected: _SourceVersion) -> None:
    for prefix in ("GGML", "LLAMA"):
        if _required(cache, f"{prefix}_BUILD_COMMIT") != expected.build_commit:
            raise CMakeBuildError("CMake selected unexpected source-version metadata")
        if _required(cache, f"{prefix}_BUILD_NUMBER") != "0":
            raise CMakeBuildError("CMake selected unexpected source build number")


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


def _write_build_root_owner(root: Path, attempt_id: str, build_id: str) -> _BuildRootOwnerV1:
    metadata = root.lstat()
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    ):
        raise CMakeBuildError(f"build root is unsafe: {root}")
    owner = _BuildRootOwnerV1(
        attempt_id=attempt_id,
        build_id=build_id,
        root_device=metadata.st_dev,
        root_inode=metadata.st_ino,
    )
    write_exclusive(
        root / ".strixlab-owner.json",
        canonical_json_bytes(owner.model_dump(mode="json")),
        0o400,
    )
    return owner


def _read_build_root_owner(root: Path) -> _BuildRootOwnerV1:
    path = root / ".strixlab-owner.json"
    try:
        descriptor = os.open(path, readonly_open_flags())
    except OSError as exc:
        raise CMakeBuildError("build-root ownership marker is unavailable") from exc
    chunks: list[bytes] = []
    size = 0
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o400
        ):
            raise CMakeBuildError("build-root ownership marker is unsafe")
        while chunk := os.read(descriptor, 4096):
            size += len(chunk)
            if size > _OWNER_LIMIT:
                raise CMakeBuildError("build-root ownership marker exceeds the size limit")
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    try:
        return _BuildRootOwnerV1.model_validate_json(b"".join(chunks))
    except ValidationError as exc:
        raise CMakeBuildError("build-root ownership marker is invalid") from exc


def _verify_build_root_owner(root: Path, expected: _BuildRootOwnerV1) -> None:
    try:
        metadata = root.lstat()
    except OSError as exc:
        raise CMakeBuildError("owned build root is unavailable") from exc
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_dev != expected.root_device
        or metadata.st_ino != expected.root_inode
    ):
        raise CMakeBuildError("build-root ownership changed during the build")
    if _read_build_root_owner(root) != expected:
        raise CMakeBuildError("build-root ownership changed during the build")


def _remove_failed_build_root(root: Path | None, owner: _BuildRootOwnerV1 | None) -> None:
    if root is None or owner is None:
        return
    _verify_build_root_owner(root, owner)
    shutil.rmtree(root)
    fsync_directory(root.parent)


def _run_configure(
    profile: BuildProfileV1,
    *,
    source: Path,
    source_version: _SourceVersion,
    build: Path,
    environment: Mapping[str, str],
    attempt: BuildAttemptSession,
    label: str,
    runner: ProcessRunner,
) -> bytes:
    command = configure_command(profile, source, build, source_version=source_version)
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
    if evidence.adapter != "llama_cpp":
        raise CMakeBuildError("CMake build adapter requires a llama_cpp source")
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
    owner: _BuildRootOwnerV1,
    expected_tools: ToolProbe,
    environment: Mapping[str, str],
    attempt: BuildAttemptSession,
    runner: ProcessRunner,
) -> None:
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
    verify_snapshot(snapshot.root)
    preparation.verify()
    _verify_build_root_owner(root, owner)


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
        root_owner: _BuildRootOwnerV1 | None = None
        finalize_started = False
        try:
            attempt.mark_active()
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
                source=snapshot.source,
                source_version=source_version,
                build=attempt.probe_root,
                environment=environment,
                attempt=attempt,
                label="probe-configure",
                runner=runner,
            )
            probe_cache = parse_cmake_cache(probe_cache_bytes)
            _verify_source_version(probe_cache, source_version)
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
            _prepare_build_root(materialized, candidate_root)
            final_root = candidate_root
            root_owner = _write_build_root_owner(
                final_root, attempt.registry.attempt_id, machine_build_id
            )
            final_cache_bytes = _run_configure(
                profile,
                source=snapshot.source,
                source_version=source_version,
                build=final_root,
                environment=environment,
                attempt=attempt,
                label="final-configure",
                runner=runner,
            )
            final_cache = parse_cmake_cache(final_cache_bytes)
            _verify_source_version(final_cache, source_version)
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
            assert final_root is not None and root_owner is not None
            _revalidate_before_finalize(
                preparation,
                profile,
                cache=final_cache,
                snapshot=snapshot,
                root=final_root,
                owner=root_owner,
                expected_tools=tools,
                environment=environment,
                attempt=attempt,
                runner=runner,
            )
            attempt.write_evidence(
                "build/result.json",
                canonical_json_bytes(
                    {
                        "build_id": machine_build_id,
                        "build_root": str(final_root),
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
        except BaseException:
            if not finalize_started:
                _record_failure_evidence(attempt)
                with contextlib.suppress(CMakeBuildError, OSError):
                    _remove_failed_build_root(final_root, root_owner)
            raise
    assert final_root is not None
    return CMakeBuildResult(
        recipe,
        machine_build_id,
        snapshot,
        final_root,
        final_selections,
        tools.observations,
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

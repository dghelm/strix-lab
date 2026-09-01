"""Production orchestration for one verified native capsule run.

This module owns the build lease, run lifecycle, machine lock, canonical runtime
environment, scratch directory, and enclosing portable result.  The fixed child
protocol remains entirely owned by :mod:`strixlab.capsules`.
"""

from __future__ import annotations

import contextlib
import hashlib
import shutil
import tempfile
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from strixlab.build_cache import BuildCacheError, BuildLease, CanonicalBuildRecordV1, lease_build
from strixlab.build_runtime import reconstruct_environment, resolve_target_executable
from strixlab.capsules import CapsuleProtocolResultV1, FailureReason, run_capsule_protocol
from strixlab.evidence import (
    Clock,
    RunError,
    RunInspection,
    RunOutcome,
    RunSession,
    TokenFactory,
    begin_run,
    inspect_run,
    list_portable_entries,
)
from strixlab.locks import LockAttempt, exclusive_lock
from strixlab.manifests import BuildTarget, CapsuleManifestV1, DashId, MachineProfileV1
from strixlab.secret_policy import (
    RedactionContext,
    is_sensitive_name,
    reject_sensitive_interpolations,
)
from strixlab.serialization import canonical_json_bytes, canonical_yaml_bytes

__all__ = [
    "CapsuleExecutionError",
    "CapsuleHooks",
    "CapsuleInputRefV1",
    "CapsuleResultV1",
    "CapsuleRunError",
    "CapsuleRunResult",
    "run_capsule",
]

_GFX_SELECTION_NAME = "gfx_targets"
_EXECUTION_ERROR_MESSAGE = "capsule run failed before producing a structured result"

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
RecordSha = Annotated[str, Field(pattern=r"^record-sha256:[0-9a-f]{64}$")]
BuildId = Annotated[str, Field(pattern=r"^build-sha256:[0-9a-f]{64}$")]


class CapsuleRunError(RuntimeError):
    """A capsule invocation failed before allocating a run."""


class CapsuleExecutionError(CapsuleRunError):
    """A run was allocated but no structured enclosing result was produced."""

    def __init__(self, *, run_id: str, record: Path | None) -> None:
        super().__init__(_EXECUTION_ERROR_MESSAGE)
        self.run_id = run_id
        self.record = record


class _CapsuleRunModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class CapsuleInputRefV1(_CapsuleRunModel):
    """One exact portable input snapshot bound by role, path, and digest."""

    role: Literal["build", "environment"]
    logical_path: Literal["capsule/build.json", "capsule/machine.json"]
    sha256: Sha256Hex

    @model_validator(mode="after")
    def _role_matches_path(self) -> Self:
        expected = {
            "build": "capsule/build.json",
            "environment": "capsule/machine.json",
        }[self.role]
        if self.logical_path != expected:
            raise ValueError("capsule input role does not match its fixed logical path")
        return self


class _BuildInputSnapshotV1(_CapsuleRunModel):
    schema_version: Literal[1] = 1
    build_id: BuildId
    canonical_record_sha256: RecordSha
    canonical: CanonicalBuildRecordV1


class _MachineInputSnapshotV1(_CapsuleRunModel):
    schema_version: Literal[1] = 1
    machine_id: DashId
    profile_sha256: Sha256Hex
    profile: MachineProfileV1


class CapsuleResultV1(_CapsuleRunModel):
    """Terminal enclosing result written once to ``capsule/result.json``."""

    schema_version: Literal[1] = 1
    capsule_id: DashId
    candidate: DashId
    machine_id: DashId
    build_id: BuildId
    canonical_record_sha256: RecordSha
    manifest_sha256: Sha256Hex
    scenario_sha256: Sha256Hex
    target: BuildTarget
    executable_sha256: Sha256Hex
    protocol_result_sha256: Sha256Hex | None
    status: Literal["passed", "failed"]
    reason: FailureReason | Literal["lock-unavailable"]
    inputs: tuple[CapsuleInputRefV1, ...]

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        expected_inputs = (
            ("build", "capsule/build.json"),
            ("environment", "capsule/machine.json"),
        )
        if tuple((value.role, value.logical_path) for value in self.inputs) != expected_inputs:
            raise ValueError("capsule inputs must be the exact ordered build and machine snapshots")
        if self.reason == "lock-unavailable":
            if self.status != "failed" or self.protocol_result_sha256 is not None:
                raise ValueError("a lock refusal cannot bind a protocol result")
            return self
        if self.protocol_result_sha256 is None:
            raise ValueError("an executed capsule result must bind its protocol result")
        if (self.status == "passed") != (self.reason == "passed"):
            raise ValueError("capsule status must equal the closed protocol reason")
        return self


MachineLockFactory = Callable[[Path], AbstractContextManager[LockAttempt]]
TempRootFactory = Callable[[], Path]
ProtocolRunner = Callable[..., CapsuleProtocolResultV1]


def _default_temp_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="strixlab-capsule-"))


@dataclass(frozen=True, slots=True)
class CapsuleHooks:
    """The four deterministic orchestration seams permitted for capsule tests."""

    clock: Clock | None = None
    token_factory: TokenFactory | None = None
    temp_root_factory: TempRootFactory = _default_temp_root
    machine_lock: MachineLockFactory = exclusive_lock
    protocol: ProtocolRunner = run_capsule_protocol


@dataclass(frozen=True, slots=True)
class CapsuleRunResult:
    """The finalized runtime result for an allocated capsule run."""

    run_id: str
    outcome: RunOutcome
    inspection: RunInspection
    result: CapsuleResultV1


@dataclass(frozen=True, slots=True)
class _BoundExecutable:
    path: Path
    sha256: str


def _require_gfx_target(canonical: CanonicalBuildRecordV1, gfx_target: str) -> None:
    selection = next(
        (entry.value for entry in canonical.selections if entry.name == _GFX_SELECTION_NAME), None
    )
    if selection is None:
        raise CapsuleRunError("leased build does not record a gfx target selection")
    targets = {value for value in selection.split(";") if value}
    if gfx_target not in targets:
        raise CapsuleRunError("leased build gfx target does not match the capsule requirement")


def _bind_build(lease: BuildLease, manifest: CapsuleManifestV1) -> _BoundExecutable:
    requirement = manifest.build
    source_evidence = lease.canonical.source.source_evidence
    if source_evidence.get("source_id") != requirement.source_id:
        raise CapsuleRunError("leased build source id does not match the capsule requirement")
    if source_evidence.get("base_commit") != requirement.source_commit:
        raise CapsuleRunError("leased build source commit does not match the capsule requirement")
    if lease.canonical.toolchain_mode != requirement.toolchain_mode:
        raise CapsuleRunError("leased build toolchain mode does not match the capsule requirement")
    _require_gfx_target(lease.canonical, requirement.gfx_target)
    executable, digest = resolve_target_executable(
        lease.canonical.artifacts,
        requirement.target,
        lease.root,
        error=CapsuleRunError,
    )
    return _BoundExecutable(Path(executable), digest)


def _publish_inputs(
    run: RunSession,
    lease: BuildLease,
    machine_profile: MachineProfileV1,
) -> tuple[CapsuleInputRefV1, ...]:
    build = _BuildInputSnapshotV1(
        build_id=lease.build_id,
        canonical_record_sha256=lease.canonical_record_sha256,
        canonical=lease.canonical,
    )
    profile_dump = machine_profile.model_dump(mode="json")
    machine = _MachineInputSnapshotV1(
        machine_id=machine_profile.id,
        profile_sha256=hashlib.sha256(canonical_json_bytes(profile_dump)).hexdigest(),
        profile=machine_profile,
    )
    snapshots: tuple[
        tuple[
            Literal["capsule/build.json", "capsule/machine.json"],
            bytes,
            Literal["build", "environment"],
        ],
        ...,
    ] = (
        (
            "capsule/build.json",
            canonical_json_bytes(build.model_dump(mode="json")),
            "build",
        ),
        (
            "capsule/machine.json",
            canonical_json_bytes(machine.model_dump(mode="json")),
            "environment",
        ),
    )
    secrets = set(run.context.secrets)
    secrets.update(
        entry.value
        for entry in lease.canonical.environment
        if entry.value and is_sensitive_name(entry.name)
    )
    context = RedactionContext(tuple(sorted(secrets, key=len, reverse=True)))
    # Preflight the complete pair before the first write so an unsafe second snapshot
    # cannot leave a partially published input set.
    reject_sensitive_interpolations(build.model_dump(mode="json"))
    reject_sensitive_interpolations(machine.model_dump(mode="json"))
    for _logical_path, payload, _role in snapshots:
        context.assert_payload_safe(payload)

    refs: list[CapsuleInputRefV1] = []
    for logical_path, payload, role in snapshots:
        entry = run.write_portable(
            logical_path,
            payload,
            media_type="application/json",
            role=role,
        )
        refs.append(
            CapsuleInputRefV1(
                role=role,
                logical_path=logical_path,
                sha256=entry.blob_sha256,
            )
        )
    return tuple(refs)


def _result(
    manifest: CapsuleManifestV1,
    lease: BuildLease,
    manifest_sha256: str,
    inputs: tuple[CapsuleInputRefV1, ...],
    executable: _BoundExecutable,
    protocol: CapsuleProtocolResultV1 | None,
    protocol_result_sha256: str | None,
) -> CapsuleResultV1:
    if protocol is None:
        status: Literal["passed", "failed"] = "failed"
        reason: FailureReason | Literal["lock-unavailable"] = "lock-unavailable"
    else:
        status = protocol.status
        reason = protocol.reason
    return CapsuleResultV1(
        capsule_id=manifest.id,
        candidate=manifest.candidate,
        machine_id=manifest.machine,
        build_id=lease.build_id,
        canonical_record_sha256=lease.canonical_record_sha256,
        manifest_sha256=manifest_sha256,
        scenario_sha256=manifest.contract.scenario_sha256,
        target=manifest.build.target,
        executable_sha256=executable.sha256,
        protocol_result_sha256=protocol_result_sha256,
        status=status,
        reason=reason,
        inputs=inputs,
    )


def _authenticate_protocol_result(
    run: RunSession,
    manifest: CapsuleManifestV1,
    manifest_sha256: str,
    executable: _BoundExecutable,
    protocol: CapsuleProtocolResultV1,
) -> str:
    if (
        protocol.capsule_id != manifest.id
        or protocol.candidate != manifest.candidate
        or protocol.scenario_sha256 != manifest.contract.scenario_sha256
        or protocol.manifest_sha256 != manifest_sha256
        or protocol.executable_sha256 != executable.sha256
    ):
        raise CapsuleRunError("capsule protocol result identity does not match the invocation")
    payload = canonical_json_bytes(protocol.model_dump(mode="json"))
    digest = hashlib.sha256(payload).hexdigest()
    entries = [
        entry
        for entry in list_portable_entries(run.active)
        if entry.logical_path == "capsule/protocol/result.json"
    ]
    if len(entries) != 1:
        raise CapsuleRunError("capsule protocol result evidence is missing or ambiguous")
    entry = entries[0]
    if (
        entry.blob_sha256 != digest
        or entry.size_bytes != len(payload)
        or entry.media_type != "application/json"
        or entry.role != "summary"
    ):
        raise CapsuleRunError("capsule protocol result evidence does not match the returned result")
    return digest


def _publish_result(run: RunSession, result: CapsuleResultV1) -> None:
    run.write_portable(
        "capsule/result.json",
        canonical_json_bytes(result.model_dump(mode="json")),
        media_type="application/json",
        role="summary",
    )


def _drive_locked(
    run: RunSession,
    manifest: CapsuleManifestV1,
    machine_profile: MachineProfileV1,
    lease: BuildLease,
    executable: _BoundExecutable,
    manifest_sha256: str,
    inputs: tuple[CapsuleInputRefV1, ...],
    hooks: CapsuleHooks,
) -> CapsuleResultV1:
    with hooks.machine_lock(Path(machine_profile.exclusive_lock.path)) as lock:
        if not lock.acquired:
            lease.verify()
            result = _result(manifest, lease, manifest_sha256, inputs, executable, None, None)
            _publish_result(run, result)
            return result

        scratch_root = hooks.temp_root_factory()
        try:
            runtime = reconstruct_environment(
                lease.canonical,
                lease.root,
                scratch_root,
                error=CapsuleRunError,
            )
            protocol = hooks.protocol(
                run,
                manifest,
                manifest_sha256=manifest_sha256,
                executable_path=executable.path,
                executable_sha256=executable.sha256,
                cwd=runtime.cwd,
                environment=runtime.environment,
                scratch_root=runtime.scratch_root,
                redaction_context=RedactionContext.from_environ(runtime.environment),
            )
            protocol_result_sha256 = _authenticate_protocol_result(
                run, manifest, manifest_sha256, executable, protocol
            )
        finally:
            shutil.rmtree(scratch_root)
        lease.verify()
        result = _result(
            manifest,
            lease,
            manifest_sha256,
            inputs,
            executable,
            protocol,
            protocol_result_sha256,
        )
        _publish_result(run, result)
        return result


def _execute_run(
    run: RunSession,
    manifest: CapsuleManifestV1,
    machine_profile: MachineProfileV1,
    lease: BuildLease,
    executable: _BoundExecutable,
    manifest_sha256: str,
    hooks: CapsuleHooks,
    home: Path,
) -> CapsuleRunResult:
    run_id = run.run_id
    try:
        with run:
            inputs = _publish_inputs(run, lease, machine_profile)
            result = _drive_locked(
                run,
                manifest,
                machine_profile,
                lease,
                executable,
                manifest_sha256,
                inputs,
                hooks,
            )
            outcome = RunOutcome.SUCCESS if result.status == "passed" else RunOutcome.FAILURE
            inspection = (
                run.succeed()
                if outcome is RunOutcome.SUCCESS
                else run.fail(f"capsule:{result.reason}")
            )
            return CapsuleRunResult(run_id, outcome, inspection, result)
    except Exception as exc:  # noqa: BLE001 - RunSession finalizes failure on every escape
        record: Path | None = None
        with contextlib.suppress(Exception):
            record = inspect_run(run_id, home=home).record
        raise CapsuleExecutionError(run_id=run_id, record=record) from exc


def run_capsule(
    manifest: CapsuleManifestV1,
    manifest_input: bytes,
    *,
    machine_profile: MachineProfileV1,
    build_id: str,
    home: Path,
    environ: Mapping[str, str],
    hooks: CapsuleHooks | None = None,
) -> CapsuleRunResult:
    """Run and finalize one capsule from a pre-existing authenticated build."""

    hooks = hooks or CapsuleHooks()
    if machine_profile.id != manifest.machine:
        raise CapsuleRunError("machine profile id does not match the capsule manifest")
    resolved = manifest.model_dump(mode="json")
    manifest_sha256 = hashlib.sha256(canonical_yaml_bytes(resolved)).hexdigest()
    try:
        with lease_build(build_id, home=home) as lease:
            executable = _bind_build(lease, manifest)
            lease.verify()
            try:
                run = begin_run(
                    manifest.id,
                    manifest_input,
                    resolved=resolved,
                    home=home,
                    environ=environ,
                    clock=hooks.clock,
                    token_factory=hooks.token_factory,
                )
            except (OSError, RunError) as exc:
                raise CapsuleRunError("capsule run allocation failed") from exc
            return _execute_run(
                run,
                manifest,
                machine_profile,
                lease,
                executable,
                manifest_sha256,
                hooks,
                home,
            )
    except BuildCacheError as exc:
        raise CapsuleRunError(f"build lease failed: {exc}") from exc

"""Authenticated, immutable views of finalized successful capsule runs.

This module is a read-only boundary.  It neither allocates runs nor decides whether
two capsule arms may be compared; it only replays every identity and evidence link
needed by a later comparison layer.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Never, cast

import yaml
from pydantic import BaseModel, ConfigDict

from strixlab.build_cache import CanonicalBuildRecordV1
from strixlab.build_runtime import resolve_target_artifact
from strixlab.capsule_contracts import CapsuleArmDifference, CapsuleComparisonContractV1
from strixlab.capsule_runs import CapsuleResultV1
from strixlab.capsules import (
    OPERATIONS,
    PROTOCOL,
    BenchmarkCoordinateV1,
    BenchmarkResponseV1,
    CapsuleCoordinateV1,
    CapsuleProcessV1,
    CapsuleProtocolResultV1,
    CapsuleRequestV1,
    CorrectnessResponseV1,
    DescribeResponseV1,
)
from strixlab.evidence import (
    PortableEvidenceV1,
    RunInspection,
    RunOutcome,
    inspect_run,
    list_portable_entries,
    read_record_member,
)
from strixlab.manifests import CapsuleManifestV1, MachineProfileV1, validate_manifest
from strixlab.secret_policy import (
    RedactionContext,
    is_sensitive_name,
    reject_sensitive_interpolations,
)
from strixlab.serialization import canonical_json_bytes, canonical_yaml_bytes

__all__ = [
    "CapsuleAlignmentProjection",
    "CapsuleSnapshotError",
    "FinalizedCapsuleSnapshot",
    "load_finalized_capsule_snapshot",
]

_ERROR_MESSAGE = "finalized capsule snapshot authentication failed"
_JSON = "application/json"
_TEXT = "text/plain"


class CapsuleSnapshotError(RuntimeError):
    """A fixed-safe failure to authenticate a finalized capsule snapshot."""

    def __init__(self, _detail: str | None = None) -> None:
        super().__init__(_ERROR_MESSAGE)


class _SnapshotModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class _BuildSnapshotV1(_SnapshotModel):
    schema_version: Literal[1] = 1
    build_id: str
    canonical_record_sha256: str
    canonical: CanonicalBuildRecordV1


class _MachineSnapshotV1(_SnapshotModel):
    schema_version: Literal[1] = 1
    machine_id: str
    profile_sha256: str
    profile: MachineProfileV1


@dataclass(frozen=True, slots=True)
class CapsuleAlignmentProjection:
    """Authenticated descriptive inputs; equality has no admission semantics."""

    protocol: str
    capsule_id: str
    candidate: str
    scenario_sha256: str
    manifest_sha256: str
    comparison: CapsuleComparisonContractV1
    comparison_sha256: str
    permitted_arm_differences: tuple[CapsuleArmDifference, ...]
    machine_id: str
    machine_profile_sha256: str
    source_id: str
    source_commit: str
    toolchain_mode: str
    gfx_target: str
    target: str
    coordinate_structure_sha256: str
    coordinate_ids: tuple[str, ...]
    training_coordinate_ids: tuple[str, ...]
    evaluation_coordinate_ids: tuple[str, ...]
    coordinate_keys: tuple[tuple[Literal["training", "evaluation"], str, str], ...]


@dataclass(frozen=True, slots=True)
class FinalizedCapsuleSnapshot:
    """A fully re-authenticated immutable projection of one passed capsule run."""

    run_id: str
    record: Path
    record_sha256: str
    resolved_manifest_bytes: bytes
    resolved_manifest_sha256: str
    manifest: CapsuleManifestV1
    result: CapsuleResultV1
    result_sha256: str
    build_snapshot: CanonicalBuildRecordV1
    build_snapshot_sha256: str
    build_record_sha256: str
    machine_snapshot: MachineProfileV1
    machine_snapshot_sha256: str
    machine_profile_sha256: str
    protocol: CapsuleProtocolResultV1
    protocol_sha256: str
    coordinates: tuple[CapsuleCoordinateV1, ...]
    training_coordinates: tuple[CapsuleCoordinateV1, ...]
    evaluation_coordinates: tuple[CapsuleCoordinateV1, ...]
    latency_seconds_by_coordinate: Mapping[str, tuple[float, ...]]
    workspace_bytes_by_coordinate: Mapping[str, int]
    alignment: CapsuleAlignmentProjection


@dataclass(frozen=True, slots=True)
class _PhaseEvidence:
    request: CapsuleRequestV1
    process: CapsuleProcessV1
    response: DescribeResponseV1 | CorrectnessResponseV1 | BenchmarkResponseV1
    response_sha256: str
    paths: tuple[str, ...]


def _fail() -> Never:
    raise CapsuleSnapshotError()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe(context: RedactionContext, content: bytes) -> None:
    context.assert_payload_safe(content)


def _blob(
    record: Path,
    entries: Mapping[str, PortableEvidenceV1],
    path: str,
    *,
    role: str,
    media_type: str,
    context: RedactionContext | None = None,
) -> tuple[PortableEvidenceV1, bytes]:
    entry = entries.get(path)
    if entry is None or entry.role != role or entry.media_type != media_type:
        _fail()
    content = read_record_member(record, f"portable/blobs/{entry.blob_sha256}")
    if len(content) != entry.size_bytes or _sha256(content) != entry.blob_sha256:
        _fail()
    if context is not None:
        _safe(context, content)
    return entry, content


def _canonical_json[ModelT: BaseModel](content: bytes, model: type[ModelT]) -> ModelT:
    value = model.model_validate_json(content, strict=True)
    if canonical_json_bytes(value.model_dump(mode="json")) != content:
        _fail()
    reject_sensitive_interpolations(value.model_dump(mode="json"))
    return value


def _manifest(inspection: RunInspection) -> tuple[bytes, str, CapsuleManifestV1]:
    content = read_record_member(inspection.record, "manifest.resolved.yaml")
    raw = yaml.safe_load(content)
    reject_sensitive_interpolations(raw)
    value = validate_manifest("capsule", raw)
    if not isinstance(value, CapsuleManifestV1):
        _fail()
    if canonical_yaml_bytes(value.model_dump(mode="json")) != content:
        _fail()
    return content, _sha256(content), value


def _bind_envelope(result: CapsuleResultV1, manifest: CapsuleManifestV1, manifest_sha: str) -> None:
    if (
        result.status != "passed"
        or result.reason != "passed"
        or result.protocol_result_sha256 is None
        or result.capsule_id != manifest.id
        or result.candidate != manifest.candidate
        or result.machine_id != manifest.machine
        or result.manifest_sha256 != manifest_sha
        or result.scenario_sha256 != manifest.contract.scenario_sha256
        or result.target != manifest.build.target
    ):
        _fail()


def _bind_inputs(
    record: Path,
    entries: Mapping[str, PortableEvidenceV1],
    result: CapsuleResultV1,
    manifest: CapsuleManifestV1,
) -> tuple[_BuildSnapshotV1, str, bytes, _MachineSnapshotV1, str, bytes]:
    build_entry, build_bytes = _blob(
        record,
        entries,
        "capsule/build.json",
        role="build",
        media_type=_JSON,
    )
    machine_entry, machine_bytes = _blob(
        record,
        entries,
        "capsule/machine.json",
        role="environment",
        media_type=_JSON,
    )
    build = _canonical_json(build_bytes, _BuildSnapshotV1)
    machine = _canonical_json(machine_bytes, _MachineSnapshotV1)
    if (
        result.inputs[0].sha256 != build_entry.blob_sha256
        or result.inputs[1].sha256 != machine_entry.blob_sha256
    ):
        _fail()

    canonical_sha = "record-sha256:" + _sha256(
        canonical_json_bytes(build.canonical.model_dump(mode="json"))
    )
    requirement = manifest.build
    source = build.canonical.source.source_evidence
    gfx = [entry.value for entry in build.canonical.selections if entry.name == "gfx_targets"]
    if len(gfx) != 1:
        _fail()
    gfx_targets = {value for value in gfx[0].split(";") if value}
    _relative, executable_sha = resolve_target_artifact(
        build.canonical.artifacts, requirement.target, error=CapsuleSnapshotError
    )
    if (
        build.build_id != result.build_id
        or build.canonical.build_id != build.build_id
        or build.canonical_record_sha256 != result.canonical_record_sha256
        or build.canonical_record_sha256 != canonical_sha
        or source.get("source_id") != requirement.source_id
        or source.get("base_commit") != requirement.source_commit
        or build.canonical.toolchain_mode != requirement.toolchain_mode
        or requirement.gfx_target not in gfx_targets
        or executable_sha != result.executable_sha256
    ):
        _fail()

    profile_sha = _sha256(canonical_json_bytes(machine.profile.model_dump(mode="json")))
    if (
        machine.machine_id != manifest.machine
        or machine.machine_id != result.machine_id
        or machine.profile.id != machine.machine_id
        or machine.profile_sha256 != profile_sha
    ):
        _fail()
    return (
        build,
        build_entry.blob_sha256,
        build_bytes,
        machine,
        machine_entry.blob_sha256,
        machine_bytes,
    )


def _intrinsic_context(build: _BuildSnapshotV1) -> RedactionContext:
    values = tuple(
        entry.value
        for entry in build.canonical.environment
        if entry.value and is_sensitive_name(entry.name)
    )
    return RedactionContext(tuple(sorted(values, key=len, reverse=True)))


def _bind_request(
    request: CapsuleRequestV1,
    operation: str,
    result: CapsuleResultV1,
    manifest: CapsuleManifestV1,
) -> None:
    if (
        request.operation != operation
        or request.protocol != PROTOCOL
        or request.capsule_id != result.capsule_id
        or request.candidate != result.candidate
        or request.scenario_sha256 != result.scenario_sha256
        or request.manifest_sha256 != result.manifest_sha256
        or request.executable_sha256 != result.executable_sha256
        or request.capsule_id != manifest.id
    ):
        _fail()


def _bind_response(
    response: DescribeResponseV1 | CorrectnessResponseV1 | BenchmarkResponseV1,
    request: CapsuleRequestV1,
    request_sha: str,
) -> None:
    if (
        response.operation != request.operation
        or response.request_sha256 != request_sha
        or response.protocol != request.protocol
        or response.capsule_id != request.capsule_id
        or response.candidate != request.candidate
        or response.scenario_sha256 != request.scenario_sha256
        or response.manifest_sha256 != request.manifest_sha256
        or response.executable_sha256 != request.executable_sha256
        or response.prior_response_sha256 != request.prior_response_sha256
        or response.scenario_contract_sha256 != request.scenario_contract_sha256
    ):
        _fail()


def _phase(
    record: Path,
    entries: Mapping[str, PortableEvidenceV1],
    operation: Literal["describe", "correctness", "benchmark"],
    result: CapsuleResultV1,
    manifest: CapsuleManifestV1,
    context: RedactionContext,
) -> _PhaseEvidence:
    root = f"capsule/protocol/{operation}"
    role = "samples" if operation == "benchmark" else "correctness"
    request_entry, request_bytes = _blob(
        record, entries, f"{root}/request.json", role=role, media_type=_JSON, context=context
    )
    process_entry, process_bytes = _blob(
        record, entries, f"{root}/process.json", role=role, media_type=_JSON, context=context
    )
    stdout_entry, stdout = _blob(
        record, entries, f"{root}/stdout.json", role=role, media_type=_JSON, context=context
    )
    request = _canonical_json(request_bytes, CapsuleRequestV1)
    process = _canonical_json(process_bytes, CapsuleProcessV1)
    response_model: type[DescribeResponseV1 | CorrectnessResponseV1 | BenchmarkResponseV1]
    if operation == "describe":
        response_model = DescribeResponseV1
    elif operation == "correctness":
        response_model = CorrectnessResponseV1
    else:
        response_model = BenchmarkResponseV1
    response = _canonical_json(stdout, response_model)
    request_sha = _sha256(request_bytes)
    response_sha = _sha256(stdout)
    _bind_request(request, operation, result, manifest)
    _bind_response(response, request, request_sha)
    if (
        process.outcome != "exited"
        or process.returncode != 0
        or process.category != "none"
        or not process.stdout_complete
        or not process.stderr_complete
        or process.stdout_truncated
        or process.stderr_truncated
        or process.capture_error
        or process.stdout_bytes != len(stdout)
        or process.stdout_sha256 != stdout_entry.blob_sha256
    ):
        _fail()

    paths = [
        request_entry.logical_path,
        process_entry.logical_path,
        stdout_entry.logical_path,
    ]
    stderr_path = f"{root}/stderr.txt"
    if process.stderr_bytes:
        stderr_entry, stderr = _blob(
            record, entries, stderr_path, role=role, media_type=_TEXT, context=context
        )
        stderr_text = stderr.decode("utf-8", errors="strict")
        reject_sensitive_interpolations(stderr_text)
        if process.stderr_bytes != len(stderr) or process.stderr_sha256 != stderr_entry.blob_sha256:
            _fail()
        paths.append(stderr_path)
    elif stderr_path in entries or process.stderr_sha256 != _sha256(b""):
        _fail()
    return _PhaseEvidence(request, process, response, response_sha, tuple(paths))


def _authenticate_protocol(
    record: Path,
    entries: Mapping[str, PortableEvidenceV1],
    result: CapsuleResultV1,
    manifest: CapsuleManifestV1,
    context: RedactionContext,
) -> tuple[CapsuleProtocolResultV1, str, tuple[str, ...]]:
    phases = tuple(
        _phase(
            record,
            entries,
            cast(Literal["describe", "correctness", "benchmark"], op),
            result,
            manifest,
            context,
        )
        for op in OPERATIONS
    )
    describe = cast(DescribeResponseV1, phases[0].response)
    correctness = cast(CorrectnessResponseV1, phases[1].response)
    benchmark = cast(BenchmarkResponseV1, phases[2].response)
    scenario_sha = _sha256(canonical_json_bytes(describe.scenario.model_dump(mode="json")))
    if (
        describe.scenario.comparison != manifest.contract.comparison
        or phases[0].request.prior_response_sha256 is not None
        or phases[0].request.scenario_contract_sha256 is not None
        or phases[0].request.scenario is not None
        or phases[1].request.prior_response_sha256 != phases[0].response_sha256
        or phases[2].request.prior_response_sha256 != phases[1].response_sha256
        or phases[1].request.scenario_contract_sha256 != scenario_sha
        or phases[2].request.scenario_contract_sha256 != scenario_sha
        or phases[1].request.scenario != describe.scenario
        or phases[2].request.scenario != describe.scenario
        or tuple(value.coordinate for value in correctness.coordinates)
        != describe.scenario.coordinates
        or not all(value.passed for value in correctness.coordinates)
        or tuple(value.coordinate for value in benchmark.coordinates)
        != describe.scenario.coordinates
    ):
        _fail()

    protocol_entry, protocol_bytes = _blob(
        record,
        entries,
        "capsule/protocol/result.json",
        role="summary",
        media_type=_JSON,
        context=context,
    )
    protocol = _canonical_json(protocol_bytes, CapsuleProtocolResultV1)
    if (
        protocol_entry.blob_sha256 != result.protocol_result_sha256
        or protocol.status != "passed"
        or protocol.reason != "passed"
        or protocol.protocol != PROTOCOL
        or protocol.capsule_id != result.capsule_id
        or protocol.candidate != result.candidate
        or protocol.scenario_sha256 != result.scenario_sha256
        or protocol.manifest_sha256 != result.manifest_sha256
        or protocol.executable_sha256 != result.executable_sha256
        or protocol.scenario != describe.scenario
        or protocol.correctness != correctness.coordinates
        or protocol.benchmark != benchmark.coordinates
        or tuple(phase.request_sha256 for phase in protocol.phases)
        != tuple(
            _sha256(canonical_json_bytes(value.request.model_dump(mode="json"))) for value in phases
        )
        or tuple(phase.process for phase in protocol.phases)
        != tuple(value.process for value in phases)
        or tuple(phase.response_sha256 for phase in protocol.phases)
        != tuple(value.response_sha256 for value in phases)
        or any(not phase.accepted or phase.failure != "none" for phase in protocol.phases)
    ):
        _fail()
    paths = tuple(path for phase in phases for path in phase.paths) + (
        "capsule/protocol/result.json",
    )
    return protocol, protocol_entry.blob_sha256, paths


def _load(run_id: str, *, home: Path) -> FinalizedCapsuleSnapshot:
    inspection = inspect_run(run_id, home=home)
    if inspection.outcome is not RunOutcome.SUCCESS:
        _fail()
    manifest_bytes, manifest_sha, manifest = _manifest(inspection)
    entries_list = list_portable_entries(inspection.record)
    entries = {entry.logical_path: entry for entry in entries_list}
    if len(entries) != len(entries_list):
        _fail()

    result_entry, result_bytes = _blob(
        inspection.record,
        entries,
        "capsule/result.json",
        role="summary",
        media_type=_JSON,
    )
    result = _canonical_json(result_bytes, CapsuleResultV1)
    _bind_envelope(result, manifest, manifest_sha)
    build, build_sha, build_bytes, machine, machine_sha, machine_bytes = _bind_inputs(
        inspection.record, entries, result, manifest
    )
    context = _intrinsic_context(build)
    for payload in (manifest_bytes, result_bytes, build_bytes, machine_bytes):
        _safe(context, payload)
    protocol, protocol_sha, protocol_paths = _authenticate_protocol(
        inspection.record, entries, result, manifest, context
    )
    expected_paths = (
        "capsule/build.json",
        "capsule/machine.json",
        *protocol_paths,
        "capsule/result.json",
    )
    if tuple(entry.logical_path for entry in entries_list) != expected_paths:
        _fail()

    scenario = protocol.scenario
    if scenario is None:
        _fail()
    coordinates = scenario.coordinates
    training = tuple(value for value in coordinates if value.case_set == "training")
    evaluation = tuple(value for value in coordinates if value.case_set == "evaluation")
    benchmark = cast(tuple[BenchmarkCoordinateV1, ...], protocol.benchmark)
    latency = MappingProxyType(
        {value.coordinate.coordinate_id: tuple(value.latency_seconds) for value in benchmark}
    )
    workspace = MappingProxyType(
        {value.coordinate.coordinate_id: value.workspace_bytes for value in benchmark}
    )
    structure_sha = _sha256(
        canonical_json_bytes([value.model_dump(mode="json") for value in coordinates])
    )
    comparison = manifest.contract.comparison
    comparison_sha = _sha256(canonical_json_bytes(comparison.model_dump(mode="json")))
    alignment = CapsuleAlignmentProjection(
        protocol=protocol.protocol,
        capsule_id=result.capsule_id,
        candidate=result.candidate,
        scenario_sha256=result.scenario_sha256,
        manifest_sha256=result.manifest_sha256,
        comparison=comparison,
        comparison_sha256=comparison_sha,
        permitted_arm_differences=comparison.permitted_arm_differences,
        machine_id=result.machine_id,
        machine_profile_sha256=machine.profile_sha256,
        source_id=manifest.build.source_id,
        source_commit=manifest.build.source_commit,
        toolchain_mode=manifest.build.toolchain_mode,
        gfx_target=manifest.build.gfx_target,
        target=manifest.build.target,
        coordinate_structure_sha256=structure_sha,
        coordinate_ids=tuple(value.coordinate_id for value in coordinates),
        training_coordinate_ids=tuple(value.coordinate_id for value in training),
        evaluation_coordinate_ids=tuple(value.coordinate_id for value in evaluation),
        coordinate_keys=tuple((value.case_set, value.case_id, value.mode) for value in coordinates),
    )
    return FinalizedCapsuleSnapshot(
        run_id=inspection.run_id,
        record=inspection.record,
        record_sha256=inspection.record_sha256,
        resolved_manifest_bytes=manifest_bytes,
        resolved_manifest_sha256=manifest_sha,
        manifest=manifest,
        result=result,
        result_sha256=result_entry.blob_sha256,
        build_snapshot=build.canonical,
        build_snapshot_sha256=build_sha,
        build_record_sha256=build.canonical_record_sha256,
        machine_snapshot=machine.profile,
        machine_snapshot_sha256=machine_sha,
        machine_profile_sha256=machine.profile_sha256,
        protocol=protocol,
        protocol_sha256=protocol_sha,
        coordinates=coordinates,
        training_coordinates=training,
        evaluation_coordinates=evaluation,
        latency_seconds_by_coordinate=latency,
        workspace_bytes_by_coordinate=workspace,
        alignment=alignment,
    )


def load_finalized_capsule_snapshot(run_id: str, *, home: Path) -> FinalizedCapsuleSnapshot:
    """Load and fully authenticate one terminal successful capsule run.

    All failures deliberately collapse to one fixed-safe exception message.
    """

    try:
        return _load(run_id, home=home)
    except CapsuleSnapshotError:
        raise
    except Exception as exc:
        raise CapsuleSnapshotError() from exc

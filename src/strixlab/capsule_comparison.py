"""Pure offline admission and directional comparison of finalized capsule runs."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal, Never, Self

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from strixlab.build_artifacts import (
    ArtifactV1,
    BuildArtifactsV1,
    CaptureToolV1,
    DynamicDependencyV1,
    DynamicInspectionV1,
    TargetArtifactsV1,
)
from strixlab.build_cache import (
    CanonicalBuildRecordV1,
    IdentityEntryV1,
    SourceBlobRefV1,
    SourcePatchRefV1,
    SourceReproducerV1,
    ToolObservationV1,
)
from strixlab.build_snapshot import SnapshotEntryV1, SnapshotManifestV1
from strixlab.capsule_contracts import CapsuleComparisonContractV1
from strixlab.capsule_runs import CapsuleInputRefV1, CapsuleResultV1
from strixlab.capsule_snapshots import (
    CapsuleAlignmentProjection,
    FinalizedCapsuleSnapshot,
    load_finalized_capsule_snapshot,
)
from strixlab.capsules import (
    BenchmarkCoordinateV1,
    BenchmarkResponseV1,
    CapsuleCoordinateV1,
    CapsulePhaseResultV1,
    CapsuleProcessV1,
    CapsuleProtocolResultV1,
    CapsuleRequestV1,
    CapsuleScenarioContractV1,
    CorrectnessCoordinateV1,
    CorrectnessResponseV1,
    DescribeResponseV1,
)
from strixlab.manifests import (
    CapsuleBuildRequirementV1,
    CapsuleContractV1,
    CapsuleManifestV1,
    CapsuleTimeoutsV1,
    DashId,
    ExclusiveLockV1,
    MachineExpectationV1,
    MachineProfileV1,
    MachineValidityV1,
    Sha256Lower,
    TelemetryV1,
)
from strixlab.serialization import canonical_json_bytes
from strixlab.source_identity import (
    PatchIdentity,
    SubmoduleIdentity,
    candidate_id,
    content_tree_id,
    length_frame,
)
from strixlab.sources import (
    PatchEvidenceV1,
    SourceEvidenceV1,
    SourceEvidenceV2,
    SubmoduleEvidenceV1,
    SubmoduleEvidenceV2,
)

__all__ = [
    "CapsuleComparisonAdmissionError",
    "CapsuleComparisonArmV1",
    "CapsuleComparisonCoordinateV1",
    "CapsuleComparisonLoadError",
    "CapsuleComparisonReportV1",
    "CapsuleComparisonResult",
    "CapsuleComparisonStatisticsError",
    "compare_finalized_capsule_runs",
]

BOOTSTRAP_REPLICATES: Final = 4096
MAX_TOTAL_BOOTSTRAP_DRAWS: Final = 16_777_216
_BOOTSTRAP_DOMAIN: Final = "strixlab.capsule.paired-latency-log-bootstrap.v1"
_POLICY_ID: Final = b"paired-latency-log-bootstrap-v1"
_LEGACY_UNINITIALIZED_SUBMODULE_SHA256: Final = hashlib.sha256(b"uninitialized").hexdigest()
_MANIFEST_CANDIDATE_SENTINEL: Final = "comparison-candidate"
_LOAD_MESSAGE: Final = "capsule comparison snapshot loading failed"
_ADMISSION_MESSAGE: Final = "capsule comparison admission failed"
_STATISTICS_MESSAGE: Final = "capsule comparison statistics failed"

RecordSha = Annotated[str, Field(pattern=r"^record-sha256:[0-9a-f]{64}$")]
BuildId = Annotated[str, Field(pattern=r"^build-sha256:[0-9a-f]{64}$")]
RunId = Annotated[
    str,
    Field(pattern=r"^run-[0-9]{8}T[0-9]{6}Z-[a-z][a-z0-9]*(?:-[a-z0-9]+)*-[0-9a-f]{32}$"),
]
SourceCandidateId = Annotated[str, Field(pattern=r"^candidate-sha256:[0-9a-f]{64}$")]
PositiveFinite = Annotated[StrictFloat, Field(gt=0)]
NonNegativeFinite = Annotated[StrictFloat, Field(ge=0)]
NonNegativeBytes = Annotated[StrictInt, Field(ge=0)]
CoordinateVerdict = Literal["improvement", "regression", "inconclusive"]
OverallVerdict = Literal["improvement", "regression", "inconclusive", "mixed"]


class CapsuleComparisonLoadError(RuntimeError):
    """Fixed-safe failure while loading either authenticated arm."""

    def __init__(self) -> None:
        super().__init__(_LOAD_MESSAGE)


class CapsuleComparisonAdmissionError(RuntimeError):
    """Fixed-safe failure of the closed two-arm admission contract."""

    def __init__(self) -> None:
        super().__init__(_ADMISSION_MESSAGE)


class CapsuleComparisonStatisticsError(RuntimeError):
    """Fixed-safe failure of deterministic paired statistics."""

    def __init__(self) -> None:
        super().__init__(_STATISTICS_MESSAGE)


class _AdmissionFailure(RuntimeError):
    pass


class _StatisticsFailure(RuntimeError):
    pass


class _ComparisonModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class CapsuleComparisonArmV1(_ComparisonModel):
    """Authenticated identities retained for one directional comparison arm."""

    label: Literal["baseline", "candidate"]
    run_id: RunId
    record_sha256: RecordSha
    candidate: DashId
    source_candidate_id: SourceCandidateId
    manifest_sha256: Sha256Lower
    result_sha256: Sha256Lower
    protocol_sha256: Sha256Lower
    build_snapshot_sha256: Sha256Lower
    machine_snapshot_sha256: Sha256Lower
    build_id: BuildId
    build_record_sha256: RecordSha
    machine_id: DashId
    machine_profile_sha256: Sha256Lower
    executable_sha256: Sha256Lower


def _coordinate_verdict(low: float, high: float, effect: float, noise: float) -> CoordinateVerdict:
    if low <= 0.0 <= high or abs(effect) <= noise:
        return "inconclusive"
    if low > 0.0 and high > 0.0:
        return "improvement"
    if low < 0.0 and high < 0.0:
        return "regression"
    raise ValueError("coordinate interval has no closed verdict")


class CapsuleComparisonCoordinateV1(_ComparisonModel):
    """One complete ordered coordinate and its directional paired statistics."""

    coordinate: CapsuleCoordinateV1
    baseline_median_seconds: PositiveFinite
    candidate_median_seconds: PositiveFinite
    mean_log_effect: StrictFloat
    baseline_over_candidate_ratio: PositiveFinite
    improvement_percent: StrictFloat
    log_ci_low: StrictFloat
    log_ci_high: StrictFloat
    baseline_noise_log: NonNegativeFinite
    baseline_workspace_bytes: NonNegativeBytes
    candidate_workspace_bytes: NonNegativeBytes
    workspace_delta_bytes: StrictInt
    verdict: CoordinateVerdict
    protected_regression: bool

    @model_validator(mode="after")
    def _relations(self) -> Self:
        try:
            ratio = math.exp(self.mean_log_effect)
            improvement = 100.0 * math.expm1(self.mean_log_effect)
        except (OverflowError, ValueError) as exc:
            raise ValueError("coordinate exponential relation is invalid") from exc
        if (
            not math.isfinite(ratio)
            or not math.isfinite(improvement)
            or self.baseline_over_candidate_ratio != ratio
            or self.improvement_percent != improvement
            or self.log_ci_low > self.log_ci_high
            or self.workspace_delta_bytes
            != self.candidate_workspace_bytes - self.baseline_workspace_bytes
            or self.verdict
            != _coordinate_verdict(
                self.log_ci_low,
                self.log_ci_high,
                self.mean_log_effect,
                self.baseline_noise_log,
            )
        ):
            raise ValueError("coordinate derived fields are inconsistent")
        return self


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _scenario_digest(
    comparison: CapsuleComparisonContractV1,
    coordinates: tuple[CapsuleComparisonCoordinateV1, ...],
) -> str:
    scenario = CapsuleScenarioContractV1(
        comparison=comparison,
        coordinates=tuple(value.coordinate for value in coordinates),
    )
    return _sha256(canonical_json_bytes(scenario.model_dump(mode="json")))


def _structure_digest(coordinates: tuple[CapsuleComparisonCoordinateV1, ...]) -> str:
    return _sha256(
        canonical_json_bytes([value.coordinate.model_dump(mode="json") for value in coordinates])
    )


def _protected(
    coordinate: CapsuleComparisonCoordinateV1,
    protected_regression_bps: int | None,
) -> bool:
    if protected_regression_bps is None:
        return False
    ratio = coordinate.candidate_median_seconds / coordinate.baseline_median_seconds
    if not math.isfinite(ratio):
        raise ValueError("protected ratio is non-finite")
    return (
        coordinate.log_ci_low < 0.0
        and coordinate.log_ci_high < 0.0
        and ratio > 1.0 + protected_regression_bps / 10_000
    )


def _aggregate(coordinates: tuple[CapsuleComparisonCoordinateV1, ...]) -> OverallVerdict:
    evaluation = tuple(
        value.verdict for value in coordinates if value.coordinate.case_set == "evaluation"
    )
    if not evaluation:
        raise ValueError("comparison has no evaluation coordinate")
    provisional: OverallVerdict = evaluation[0] if len(set(evaluation)) == 1 else "mixed"
    if provisional == "improvement" and any(value.protected_regression for value in coordinates):
        return "mixed"
    return provisional


class CapsuleComparisonReportV1(_ComparisonModel):
    """Canonical, evidence-free directional report over two authenticated arms."""

    schema_version: Literal[1] = 1
    comparison: CapsuleComparisonContractV1
    comparison_sha256: Sha256Lower
    capsule_id: DashId
    scenario_sha256: Sha256Lower
    described_scenario_sha256: Sha256Lower
    machine_id: DashId
    machine_profile_sha256: Sha256Lower
    coordinate_structure_sha256: Sha256Lower
    baseline: CapsuleComparisonArmV1
    candidate: CapsuleComparisonArmV1
    coordinates: tuple[CapsuleComparisonCoordinateV1, ...]
    overall_verdict: OverallVerdict

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        comparison_sha = _sha256(canonical_json_bytes(self.comparison.model_dump(mode="json")))
        if (
            self.comparison_sha256 != comparison_sha
            or self.baseline.label != "baseline"
            or self.candidate.label != "candidate"
            or self.baseline.run_id == self.candidate.run_id
            or self.baseline.record_sha256 == self.candidate.record_sha256
            or self.baseline.machine_id != self.machine_id
            or self.candidate.machine_id != self.machine_id
            or self.baseline.machine_profile_sha256 != self.machine_profile_sha256
            or self.candidate.machine_profile_sha256 != self.machine_profile_sha256
            or not self.coordinates
            or self.described_scenario_sha256 != _scenario_digest(self.comparison, self.coordinates)
            or self.coordinate_structure_sha256 != _structure_digest(self.coordinates)
            or any(
                value.protected_regression
                != _protected(value, self.comparison.protected_regression_bps)
                for value in self.coordinates
            )
            or self.overall_verdict != _aggregate(self.coordinates)
        ):
            raise ValueError("comparison report identities or derived fields disagree")
        return self


class CapsuleComparisonResult(_ComparisonModel):
    """A validated report together with its exact canonical bytes and digest."""

    report: CapsuleComparisonReportV1
    report_bytes: bytes
    report_sha256: Sha256Lower

    @model_validator(mode="after")
    def _canonical_report(self) -> Self:
        expected = canonical_json_bytes(self.report.model_dump(mode="json"))
        if self.report_bytes != expected or self.report_sha256 != _sha256(expected):
            raise ValueError("comparison result does not bind canonical report bytes")
        return self


@dataclass(frozen=True, slots=True)
class _AuthenticatedSource:
    reproducer: SourceReproducerV1
    evidence: SourceEvidenceV1 | SourceEvidenceV2
    snapshot: SnapshotManifestV1


@dataclass(frozen=True, slots=True)
class _StableSourceEvidence:
    schema_version: int
    source_id: str
    source_locator: str | None
    source_locator_sha256: str
    base_commit: str
    branch_hint: str | None
    adapter: str
    submodules_enabled: bool
    submodules: tuple[SubmoduleEvidenceV1, ...] | tuple[SubmoduleEvidenceV2, ...]


@dataclass(frozen=True, slots=True)
class _StableSource:
    schema_version: int
    evidence: _StableSourceEvidence


@dataclass(frozen=True, slots=True)
class _StableTarget:
    schema_version: int
    name: str
    target_type: str
    artifacts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _StableArtifact:
    schema_version: int
    path: str
    kind: str
    elf_type: str | None
    targets: tuple[str, ...]
    runtime_dependency: bool


@dataclass(frozen=True, slots=True)
class _StableBuildArtifacts:
    schema_version: int
    targets: tuple[_StableTarget, ...]
    artifacts: tuple[_StableArtifact, ...]


@dataclass(frozen=True, slots=True)
class _StableBuild:
    schema_version: int
    profile_sha256: str
    toolchain_mode: str
    environment: tuple[IdentityEntryV1, ...]
    requested_targets: tuple[str, ...]
    selections: tuple[IdentityEntryV1, ...]
    tools: tuple[ToolObservationV1, ...]
    source: SourceReproducerV1 | _StableSource
    artifacts: _StableBuildArtifacts


@dataclass(frozen=True, slots=True)
class _StableAlignment:
    protocol: str
    capsule_id: str
    scenario_sha256: str
    comparison: CapsuleComparisonContractV1
    comparison_sha256: str
    permitted_arm_differences: tuple[str, ...]
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
    coordinate_keys: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True, slots=True)
class _StableInput:
    role: str
    logical_path: str
    sha256: str | None


@dataclass(frozen=True, slots=True)
class _StableResult:
    schema_version: int
    capsule_id: str
    machine_id: str
    build_id: str | None
    canonical_record_sha256: str | None
    scenario_sha256: str
    target: str
    executable_sha256: str | None
    status: str
    reason: str
    inputs: tuple[_StableInput, ...]


@dataclass(frozen=True, slots=True)
class _StableProcess:
    schema_version: int
    outcome: str
    returncode: int | None
    stderr_bytes: int
    stderr_sha256: str
    stdout_complete: bool
    stderr_complete: bool
    stdout_truncated: bool
    stderr_truncated: bool
    capture_error: bool
    category: str


@dataclass(frozen=True, slots=True)
class _StablePhase:
    schema_version: int
    operation: str
    process: _StableProcess
    accepted: bool
    failure: str


@dataclass(frozen=True, slots=True)
class _StableProtocol:
    schema_version: int
    protocol: str
    capsule_id: str
    scenario_sha256: str
    executable_sha256: str | None
    status: str
    reason: str
    phases: tuple[_StablePhase, ...]
    scenario: CapsuleScenarioContractV1 | None
    correctness: tuple[CorrectnessCoordinateV1, ...] | None
    benchmark_coordinates: tuple[CapsuleCoordinateV1, ...] | None


@dataclass(frozen=True, slots=True)
class _Admission:
    baseline: FinalizedCapsuleSnapshot
    candidate: FinalizedCapsuleSnapshot
    comparison: CapsuleComparisonContractV1
    scenario: CapsuleScenarioContractV1
    baseline_source: _AuthenticatedSource
    candidate_source: _AuthenticatedSource


_MODEL_FIELDS: tuple[tuple[type[BaseModel], frozenset[str]], ...] = (
    (
        CapsuleComparisonContractV1,
        frozenset({"policy", "protected_regression_bps", "permitted_arm_differences"}),
    ),
    (
        CapsuleBuildRequirementV1,
        frozenset({"source_id", "source_commit", "toolchain_mode", "gfx_target", "target"}),
    ),
    (CapsuleContractV1, frozenset({"protocol", "scenario_sha256", "comparison"})),
    (
        CapsuleTimeoutsV1,
        frozenset({"describe_seconds", "correctness_seconds", "benchmark_seconds"}),
    ),
    (
        CapsuleManifestV1,
        frozenset(
            {"schema_version", "id", "candidate", "machine", "build", "contract", "timeouts"}
        ),
    ),
    (
        CapsuleCoordinateV1,
        frozenset(
            {
                "coordinate_id",
                "case_id",
                "case_set",
                "mode",
                "order",
                "input_id",
                "input_sha256",
                "warmup_count",
                "sample_count",
            }
        ),
    ),
    (CapsuleScenarioContractV1, frozenset({"schema_version", "comparison", "coordinates"})),
    (
        CapsuleRequestV1,
        frozenset(
            {
                "schema_version",
                "protocol",
                "operation",
                "capsule_id",
                "candidate",
                "scenario_sha256",
                "manifest_sha256",
                "executable_sha256",
                "prior_response_sha256",
                "scenario_contract_sha256",
                "scenario",
            }
        ),
    ),
    (
        DescribeResponseV1,
        frozenset(
            {
                "schema_version",
                "protocol",
                "operation",
                "request_sha256",
                "capsule_id",
                "candidate",
                "scenario_sha256",
                "manifest_sha256",
                "executable_sha256",
                "prior_response_sha256",
                "scenario_contract_sha256",
                "opaque_payload",
                "scenario",
            }
        ),
    ),
    (
        CorrectnessResponseV1,
        frozenset(
            {
                "schema_version",
                "protocol",
                "operation",
                "request_sha256",
                "capsule_id",
                "candidate",
                "scenario_sha256",
                "manifest_sha256",
                "executable_sha256",
                "prior_response_sha256",
                "scenario_contract_sha256",
                "opaque_payload",
                "coordinates",
            }
        ),
    ),
    (
        BenchmarkResponseV1,
        frozenset(
            {
                "schema_version",
                "protocol",
                "operation",
                "request_sha256",
                "capsule_id",
                "candidate",
                "scenario_sha256",
                "manifest_sha256",
                "executable_sha256",
                "prior_response_sha256",
                "scenario_contract_sha256",
                "opaque_payload",
                "coordinates",
            }
        ),
    ),
    (CapsuleInputRefV1, frozenset({"role", "logical_path", "sha256"})),
    (
        CapsuleResultV1,
        frozenset(
            {
                "schema_version",
                "capsule_id",
                "candidate",
                "machine_id",
                "build_id",
                "canonical_record_sha256",
                "manifest_sha256",
                "scenario_sha256",
                "target",
                "executable_sha256",
                "protocol_result_sha256",
                "status",
                "reason",
                "inputs",
            }
        ),
    ),
    (
        CapsuleProcessV1,
        frozenset(
            {
                "schema_version",
                "outcome",
                "returncode",
                "duration_seconds",
                "stdout_bytes",
                "stderr_bytes",
                "stdout_sha256",
                "stderr_sha256",
                "stdout_complete",
                "stderr_complete",
                "stdout_truncated",
                "stderr_truncated",
                "capture_error",
                "category",
            }
        ),
    ),
    (
        CapsulePhaseResultV1,
        frozenset(
            {
                "schema_version",
                "operation",
                "request_sha256",
                "process",
                "response_sha256",
                "accepted",
                "failure",
            }
        ),
    ),
    (CorrectnessCoordinateV1, frozenset({"coordinate", "passed"})),
    (BenchmarkCoordinateV1, frozenset({"coordinate", "latency_seconds", "workspace_bytes"})),
    (
        CapsuleProtocolResultV1,
        frozenset(
            {
                "schema_version",
                "protocol",
                "capsule_id",
                "candidate",
                "scenario_sha256",
                "manifest_sha256",
                "executable_sha256",
                "status",
                "reason",
                "phases",
                "scenario",
                "correctness",
                "benchmark",
            }
        ),
    ),
    (IdentityEntryV1, frozenset({"name", "value"})),
    (
        ToolObservationV1,
        frozenset(
            {
                "role",
                "path",
                "realpath",
                "mode",
                "size_bytes",
                "sha256",
                "version_sha256",
                "search_sha256",
            }
        ),
    ),
    (SourceBlobRefV1, frozenset({"schema_version", "relative_path", "sha256", "size_bytes"})),
    (
        SourcePatchRefV1,
        frozenset({"schema_version", "order", "relative_path", "sha256", "size_bytes"}),
    ),
    (
        SourceReproducerV1,
        frozenset(
            {
                "schema_version",
                "candidate_id",
                "content_tree_id",
                "snapshot_id",
                "source_evidence",
                "source_evidence_sha256",
                "snapshot_manifest",
                "diff",
                "patches",
            }
        ),
    ),
    (PatchEvidenceV1, frozenset({"order", "sha256", "size_bytes", "record_file"})),
    (SubmoduleEvidenceV1, frozenset({"path", "commit", "locator", "locator_sha256"})),
    (SubmoduleEvidenceV2, frozenset({"path", "commit", "locator", "locator_sha256"})),
    (
        SourceEvidenceV1,
        frozenset(
            {
                "preparation_id",
                "request_digest",
                "source_id",
                "source_locator",
                "source_locator_sha256",
                "base_commit",
                "branch_hint",
                "adapter",
                "submodules_enabled",
                "patches",
                "root_tree",
                "content_tree_id",
                "candidate_id",
                "diff_file",
                "diff_sha256",
                "diff_size_bytes",
                "status",
                "created_at",
                "schema_version",
                "submodules",
            }
        ),
    ),
    (
        SourceEvidenceV2,
        frozenset(
            {
                "preparation_id",
                "request_digest",
                "source_id",
                "source_locator",
                "source_locator_sha256",
                "base_commit",
                "branch_hint",
                "adapter",
                "submodules_enabled",
                "patches",
                "root_tree",
                "content_tree_id",
                "candidate_id",
                "diff_file",
                "diff_sha256",
                "diff_size_bytes",
                "status",
                "created_at",
                "schema_version",
                "submodules",
            }
        ),
    ),
    (SnapshotEntryV1, frozenset({"path", "kind", "mode", "size_bytes", "sha256", "link_target"})),
    (
        SnapshotManifestV1,
        frozenset({"schema_version", "snapshot_id", "candidate_id", "content_tree_id", "entries"}),
    ),
    (
        TargetArtifactsV1,
        frozenset({"schema_version", "name", "target_id", "target_type", "artifacts"}),
    ),
    (
        ArtifactV1,
        frozenset(
            {
                "schema_version",
                "path",
                "kind",
                "elf_type",
                "mode",
                "size_bytes",
                "sha256",
                "targets",
                "runtime_dependency",
            }
        ),
    ),
    (
        DynamicDependencyV1,
        frozenset({"schema_version", "name", "path", "resolved", "in_build_root"}),
    ),
    (
        DynamicInspectionV1,
        frozenset(
            {
                "schema_version",
                "artifact",
                "elf_type",
                "dynamic",
                "static",
                "needed",
                "soname",
                "rpath",
                "runpath",
                "dependencies",
                "readelf_sha256",
                "ldd_sha256",
            }
        ),
    ),
    (
        CaptureToolV1,
        frozenset(
            {
                "schema_version",
                "name",
                "path",
                "realpath",
                "mode",
                "size_bytes",
                "sha256",
                "version_sha256",
            }
        ),
    ),
    (
        BuildArtifactsV1,
        frozenset(
            {
                "schema_version",
                "artifact_set_id",
                "targets",
                "artifacts",
                "inspections",
                "capture_tools",
                "cmake_cache_sha256",
                "compile_commands_sha256",
            }
        ),
    ),
    (
        CanonicalBuildRecordV1,
        frozenset(
            {
                "schema_version",
                "recipe_id",
                "profile_sha256",
                "toolchain_mode",
                "environment",
                "requested_targets",
                "selections",
                "tools",
                "source",
                "build_id",
                "producer_attempt_id",
                "artifacts",
            }
        ),
    ),
    (MachineExpectationV1, frozenset({"gpu_arch", "integrated_gpu", "memory_gib_min"})),
    (ExclusiveLockV1, frozenset({"path"})),
    (TelemetryV1, frozenset({"amd_smi", "sample_interval_ms"})),
    (
        MachineValidityV1,
        frozenset(
            {
                "require_ac_power",
                "max_background_gpu_busy_pct",
                "min_available_memory_gib",
                "temperature_warn_c",
            }
        ),
    ),
    (
        MachineProfileV1,
        frozenset({"schema_version", "id", "expect", "exclusive_lock", "telemetry", "validity"}),
    ),
)


def _guard_model_fields() -> None:
    for model, expected in _MODEL_FIELDS:
        if frozenset(model.model_fields) != expected:
            raise _AdmissionFailure()


def _fail_admission() -> Never:
    raise _AdmissionFailure()


def _parse_source(reproducer: SourceReproducerV1) -> _AuthenticatedSource:
    evidence_bytes = canonical_json_bytes(reproducer.source_evidence)
    schema_version = reproducer.source_evidence.get("schema_version")
    evidence_type: type[SourceEvidenceV1] | type[SourceEvidenceV2]
    if schema_version == 1 and type(schema_version) is int:
        evidence_type = SourceEvidenceV1
    elif schema_version == 2 and type(schema_version) is int:
        evidence_type = SourceEvidenceV2
    else:
        _fail_admission()
    evidence = evidence_type.model_validate_json(evidence_bytes, strict=True)
    if (
        canonical_json_bytes(evidence.model_dump(mode="json")) != evidence_bytes
        or _sha256(evidence_bytes) != reproducer.source_evidence_sha256
    ):
        _fail_admission()

    snapshot_bytes = canonical_json_bytes(reproducer.snapshot_manifest)
    snapshot = SnapshotManifestV1.model_validate_json(snapshot_bytes, strict=True)
    if canonical_json_bytes(snapshot.model_dump(mode="json")) != snapshot_bytes:
        _fail_admission()

    patch_identities = tuple(
        PatchIdentity(value.order, value.size_bytes, value.sha256) for value in evidence.patches
    )
    submodule_identities = tuple(
        SubmoduleIdentity(value.path, value.commit) for value in evidence.submodules
    )
    expected_content = content_tree_id(
        evidence.root_tree,
        patches=patch_identities,
        submodules=submodule_identities,
    )
    expected_candidate = candidate_id(
        evidence.base_commit,
        expected_content,
        submodules=evidence.submodules_enabled,
    )
    legacy_inconsistent = isinstance(evidence, SourceEvidenceV1) and any(
        (submodule.locator_sha256 != _LEGACY_UNINITIALIZED_SUBMODULE_SHA256)
        != evidence.submodules_enabled
        for submodule in evidence.submodules
    )
    current_inconsistent = isinstance(evidence, SourceEvidenceV2) and any(
        (submodule.locator_sha256 is not None) != evidence.submodules_enabled
        for submodule in evidence.submodules
    )
    if legacy_inconsistent or current_inconsistent:
        _fail_admission()
    if (
        evidence.source_locator is not None
        and _sha256(evidence.source_locator.encode("utf-8")) != evidence.source_locator_sha256
    ) or any(
        value.locator is not None
        and (
            value.locator_sha256 is None
            or _sha256(value.locator.encode("utf-8")) != value.locator_sha256
        )
        for value in evidence.submodules
    ):
        _fail_admission()
    entry_bytes = canonical_json_bytes(
        [entry.model_dump(mode="json") for entry in snapshot.entries]
    )
    expected_snapshot = "snapshot-sha256:" + _sha256(
        length_frame(
            "strixlab.build.source-snapshot.v1",
            (
                ("candidate-id", snapshot.candidate_id.encode("ascii")),
                ("content-tree-id", snapshot.content_tree_id.encode("ascii")),
                ("entries", entry_bytes),
            ),
        )
    )
    if (
        tuple(value.order for value in evidence.patches)
        != tuple(range(1, len(evidence.patches) + 1))
        or tuple(value.record_file for value in evidence.patches)
        != tuple(f"patch-{value.order:03d}.patch" for value in evidence.patches)
        or tuple(entry.path for entry in snapshot.entries)
        != tuple(sorted(entry.path for entry in snapshot.entries))
        or len({entry.path for entry in snapshot.entries}) != len(snapshot.entries)
        or evidence.content_tree_id != expected_content
        or evidence.candidate_id != expected_candidate
        or reproducer.content_tree_id != expected_content
        or reproducer.candidate_id != expected_candidate
        or snapshot.content_tree_id != expected_content
        or snapshot.candidate_id != expected_candidate
        or snapshot.snapshot_id != expected_snapshot
        or reproducer.snapshot_id != expected_snapshot
    ):
        _fail_admission()

    expected_diff = None
    if evidence.diff_size_bytes > 0:
        expected_diff = SourceBlobRefV1(
            relative_path="source/diff.patch",
            sha256=evidence.diff_sha256,
            size_bytes=evidence.diff_size_bytes,
        )
    expected_patches = tuple(
        SourcePatchRefV1(
            order=value.order,
            relative_path=f"source/patches/{value.order:04d}.patch",
            sha256=value.sha256,
            size_bytes=value.size_bytes,
        )
        for value in evidence.patches
    )
    if evidence.diff_file != "candidate.diff" or reproducer.diff != expected_diff:
        _fail_admission()
    if reproducer.patches != expected_patches:
        _fail_admission()
    return _AuthenticatedSource(reproducer, evidence, snapshot)


def _stable_source(value: _AuthenticatedSource) -> _StableSource:
    evidence = value.evidence
    return _StableSource(
        schema_version=value.reproducer.schema_version,
        evidence=_StableSourceEvidence(
            schema_version=evidence.schema_version,
            source_id=evidence.source_id,
            source_locator=evidence.source_locator,
            source_locator_sha256=evidence.source_locator_sha256,
            base_commit=evidence.base_commit,
            branch_hint=evidence.branch_hint,
            adapter=evidence.adapter,
            submodules_enabled=evidence.submodules_enabled,
            submodules=evidence.submodules,
        ),
    )


def _stable_artifacts(value: BuildArtifactsV1) -> _StableBuildArtifacts:
    return _StableBuildArtifacts(
        schema_version=value.schema_version,
        targets=tuple(
            _StableTarget(
                schema_version=target.schema_version,
                name=target.name,
                target_type=target.target_type,
                artifacts=target.artifacts,
            )
            for target in value.targets
        ),
        artifacts=tuple(
            _StableArtifact(
                schema_version=artifact.schema_version,
                path=artifact.path,
                kind=artifact.kind,
                elf_type=artifact.elf_type,
                targets=artifact.targets,
                runtime_dependency=artifact.runtime_dependency,
            )
            for artifact in value.artifacts
        ),
    )


def _stable_build(
    value: CanonicalBuildRecordV1,
    source: _AuthenticatedSource,
    *,
    source_difference: bool,
) -> _StableBuild:
    return _StableBuild(
        schema_version=value.schema_version,
        profile_sha256=value.profile_sha256,
        toolchain_mode=value.toolchain_mode,
        environment=value.environment,
        requested_targets=value.requested_targets,
        selections=value.selections,
        tools=value.tools,
        source=_stable_source(source) if source_difference else source.reproducer,
        artifacts=_stable_artifacts(value.artifacts),
    )


def _stable_alignment(value: CapsuleAlignmentProjection) -> _StableAlignment:
    return _StableAlignment(
        protocol=value.protocol,
        capsule_id=value.capsule_id,
        scenario_sha256=value.scenario_sha256,
        comparison=value.comparison,
        comparison_sha256=value.comparison_sha256,
        permitted_arm_differences=value.permitted_arm_differences,
        machine_id=value.machine_id,
        machine_profile_sha256=value.machine_profile_sha256,
        source_id=value.source_id,
        source_commit=value.source_commit,
        toolchain_mode=value.toolchain_mode,
        gfx_target=value.gfx_target,
        target=value.target,
        coordinate_structure_sha256=value.coordinate_structure_sha256,
        coordinate_ids=value.coordinate_ids,
        training_coordinate_ids=value.training_coordinate_ids,
        evaluation_coordinate_ids=value.evaluation_coordinate_ids,
        coordinate_keys=value.coordinate_keys,
    )


def _stable_result(value: CapsuleResultV1, *, build_difference: bool) -> _StableResult:
    return _StableResult(
        schema_version=value.schema_version,
        capsule_id=value.capsule_id,
        machine_id=value.machine_id,
        build_id=None if build_difference else value.build_id,
        canonical_record_sha256=None if build_difference else value.canonical_record_sha256,
        scenario_sha256=value.scenario_sha256,
        target=value.target,
        executable_sha256=None if build_difference else value.executable_sha256,
        status=value.status,
        reason=value.reason,
        inputs=tuple(
            _StableInput(
                role=entry.role,
                logical_path=entry.logical_path,
                sha256=None if build_difference and entry.role == "build" else entry.sha256,
            )
            for entry in value.inputs
        ),
    )


def _stable_process(value: CapsuleProcessV1) -> _StableProcess:
    return _StableProcess(
        schema_version=value.schema_version,
        outcome=value.outcome,
        returncode=value.returncode,
        stderr_bytes=value.stderr_bytes,
        stderr_sha256=value.stderr_sha256,
        stdout_complete=value.stdout_complete,
        stderr_complete=value.stderr_complete,
        stdout_truncated=value.stdout_truncated,
        stderr_truncated=value.stderr_truncated,
        capture_error=value.capture_error,
        category=value.category,
    )


def _stable_phase(value: CapsulePhaseResultV1) -> _StablePhase:
    return _StablePhase(
        schema_version=value.schema_version,
        operation=value.operation,
        process=_stable_process(value.process),
        accepted=value.accepted,
        failure=value.failure,
    )


def _stable_protocol(value: CapsuleProtocolResultV1, *, build_difference: bool) -> _StableProtocol:
    return _StableProtocol(
        schema_version=value.schema_version,
        protocol=value.protocol,
        capsule_id=value.capsule_id,
        scenario_sha256=value.scenario_sha256,
        executable_sha256=None if build_difference else value.executable_sha256,
        status=value.status,
        reason=value.reason,
        phases=tuple(_stable_phase(phase) for phase in value.phases),
        scenario=value.scenario,
        correctness=value.correctness,
        benchmark_coordinates=None
        if value.benchmark is None
        else tuple(entry.coordinate for entry in value.benchmark),
    )


def _normalized_manifest(value: CapsuleManifestV1) -> CapsuleManifestV1:
    return value.model_copy(update={"candidate": _MANIFEST_CANDIDATE_SENTINEL})


def _validate_snapshot_shape(value: FinalizedCapsuleSnapshot) -> CapsuleScenarioContractV1:
    scenario = value.protocol.scenario
    if scenario is None:
        _fail_admission()
    ids = tuple(coordinate.coordinate_id for coordinate in scenario.coordinates)
    if (
        value.coordinates != scenario.coordinates
        or tuple(value.latency_seconds_by_coordinate) != ids
        or tuple(value.workspace_bytes_by_coordinate) != ids
        or any(
            len(value.latency_seconds_by_coordinate[coordinate.coordinate_id])
            != coordinate.sample_count
            for coordinate in scenario.coordinates
        )
    ):
        _fail_admission()
    return scenario


def _admit(
    baseline: FinalizedCapsuleSnapshot,
    candidate: FinalizedCapsuleSnapshot,
) -> _Admission:
    _guard_model_fields()
    if baseline.run_id == candidate.run_id or baseline.record_sha256 == candidate.record_sha256:
        _fail_admission()
    baseline_scenario = _validate_snapshot_shape(baseline)
    candidate_scenario = _validate_snapshot_shape(candidate)
    if baseline_scenario != candidate_scenario:
        _fail_admission()
    comparison = baseline_scenario.comparison
    if (
        candidate_scenario.comparison != comparison
        or _normalized_manifest(baseline.manifest) != _normalized_manifest(candidate.manifest)
        or baseline.machine_snapshot != candidate.machine_snapshot
        or baseline.machine_profile_sha256 != candidate.machine_profile_sha256
        or _stable_alignment(baseline.alignment) != _stable_alignment(candidate.alignment)
        or baseline.alignment.comparison != comparison
        or candidate.alignment.comparison != comparison
    ):
        _fail_admission()

    baseline_source = _parse_source(baseline.build_snapshot.source)
    candidate_source = _parse_source(candidate.build_snapshot.source)
    differences = comparison.permitted_arm_differences
    source_difference = "source-candidate" in differences
    build_difference = "build-output" in differences
    if _stable_result(baseline.result, build_difference=build_difference) != _stable_result(
        candidate.result, build_difference=build_difference
    ) or _stable_protocol(baseline.protocol, build_difference=build_difference) != _stable_protocol(
        candidate.protocol, build_difference=build_difference
    ):
        _fail_admission()
    if source_difference:
        if _stable_source(baseline_source) != _stable_source(candidate_source):
            _fail_admission()
    elif baseline_source.reproducer != candidate_source.reproducer:
        _fail_admission()

    if build_difference:
        if _stable_build(
            baseline.build_snapshot,
            baseline_source,
            source_difference=source_difference,
        ) != _stable_build(
            candidate.build_snapshot,
            candidate_source,
            source_difference=source_difference,
        ):
            _fail_admission()
    elif (
        baseline.build_snapshot != candidate.build_snapshot
        or baseline.build_snapshot_sha256 != candidate.build_snapshot_sha256
        or baseline.build_record_sha256 != candidate.build_record_sha256
        or baseline.result.build_id != candidate.result.build_id
        or baseline.result.executable_sha256 != candidate.result.executable_sha256
    ):
        _fail_admission()
    return _Admission(
        baseline,
        candidate,
        comparison,
        baseline_scenario,
        baseline_source,
        candidate_source,
    )


def _u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def _bootstrap_index(
    baseline_record_sha256: str,
    candidate_record_sha256: str,
    case_id: str,
    mode: str,
    replicate: int,
    draw: int,
    n: int,
) -> int:
    framed = length_frame(
        _BOOTSTRAP_DOMAIN,
        (
            ("policy_id", _POLICY_ID),
            ("baseline_record_sha256", baseline_record_sha256.encode("ascii")),
            ("candidate_record_sha256", candidate_record_sha256.encode("ascii")),
            ("case_id", case_id.encode("utf-8")),
            ("mode", mode.encode("utf-8")),
            ("replicate", _u64(replicate)),
            ("draw", _u64(draw)),
        ),
    )
    return int.from_bytes(hashlib.sha256(framed).digest()[:8], "big") % n


def _r7(sorted_values: tuple[float, ...], p: float) -> float:
    if not sorted_values or not 0.0 <= p <= 1.0:
        raise _StatisticsFailure()
    q = (len(sorted_values) - 1) * p
    if q <= 0.0:
        return sorted_values[0]
    if q >= len(sorted_values) - 1:
        return sorted_values[-1]
    index = math.floor(q)
    fraction = q - index
    value = sorted_values[index] + fraction * (sorted_values[index + 1] - sorted_values[index])
    if not math.isfinite(value):
        raise _StatisticsFailure()
    return value


def _median(values: tuple[float, ...]) -> float:
    if not values:
        raise _StatisticsFailure()
    ordered = tuple(sorted(values))
    middle = len(ordered) // 2
    value = (
        ordered[middle]
        if len(ordered) % 2
        else math.fsum((ordered[middle - 1], ordered[middle])) / 2
    )
    if not math.isfinite(value):
        raise _StatisticsFailure()
    return value


def _coordinate_statistics(
    coordinate: CapsuleCoordinateV1,
    baseline_samples: tuple[float, ...],
    candidate_samples: tuple[float, ...],
    baseline_workspace: int,
    candidate_workspace: int,
    comparison: CapsuleComparisonContractV1,
    baseline_record_sha256: str,
    candidate_record_sha256: str,
) -> CapsuleComparisonCoordinateV1:
    n = coordinate.sample_count
    if len(baseline_samples) != n or len(candidate_samples) != n:
        raise _StatisticsFailure()
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (*baseline_samples, *candidate_samples)
    ):
        raise _StatisticsFailure()
    baseline_logs = tuple(math.log(value) for value in baseline_samples)
    candidate_logs = tuple(math.log(value) for value in candidate_samples)
    deltas = tuple(
        baseline - candidate
        for baseline, candidate in zip(baseline_logs, candidate_logs, strict=True)
    )
    mean = math.fsum(deltas) / n
    ratio = math.exp(mean)
    improvement = 100.0 * math.expm1(mean)
    means: list[float] = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected = tuple(
            deltas[
                _bootstrap_index(
                    baseline_record_sha256,
                    candidate_record_sha256,
                    coordinate.case_id,
                    coordinate.mode,
                    replicate,
                    draw,
                    n,
                )
            ]
            for draw in range(n)
        )
        replicate_mean = math.fsum(selected) / n
        if not math.isfinite(replicate_mean):
            raise _StatisticsFailure()
        means.append(replicate_mean)
    ordered_means = tuple(sorted(means))
    low = _r7(ordered_means, 0.025)
    high = _r7(ordered_means, 0.975)
    baseline_log_median = _median(baseline_logs)
    noise = 1.4826 * _median(tuple(abs(value - baseline_log_median) for value in baseline_logs))
    baseline_median = _median(baseline_samples)
    candidate_median = _median(candidate_samples)
    derived = (
        mean,
        ratio,
        improvement,
        low,
        high,
        noise,
        baseline_median,
        candidate_median,
    )
    if any(not math.isfinite(value) for value in derived):
        raise _StatisticsFailure()
    verdict = _coordinate_verdict(low, high, mean, noise)
    candidate_ratio = candidate_median / baseline_median
    if not math.isfinite(candidate_ratio):
        raise _StatisticsFailure()
    bps = comparison.protected_regression_bps
    protected = (
        bps is not None and low < 0.0 and high < 0.0 and candidate_ratio > 1.0 + bps / 10_000
    )
    return CapsuleComparisonCoordinateV1(
        coordinate=coordinate,
        baseline_median_seconds=baseline_median,
        candidate_median_seconds=candidate_median,
        mean_log_effect=mean,
        baseline_over_candidate_ratio=ratio,
        improvement_percent=improvement,
        log_ci_low=low,
        log_ci_high=high,
        baseline_noise_log=noise,
        baseline_workspace_bytes=baseline_workspace,
        candidate_workspace_bytes=candidate_workspace,
        workspace_delta_bytes=candidate_workspace - baseline_workspace,
        verdict=verdict,
        protected_regression=protected,
    )


def _arm(
    label: Literal["baseline", "candidate"], value: FinalizedCapsuleSnapshot
) -> CapsuleComparisonArmV1:
    return CapsuleComparisonArmV1(
        label=label,
        run_id=value.run_id,
        record_sha256=value.record_sha256,
        candidate=value.manifest.candidate,
        source_candidate_id=value.build_snapshot.source.candidate_id,
        manifest_sha256=value.resolved_manifest_sha256,
        result_sha256=value.result_sha256,
        protocol_sha256=value.protocol_sha256,
        build_snapshot_sha256=value.build_snapshot_sha256,
        machine_snapshot_sha256=value.machine_snapshot_sha256,
        build_id=value.result.build_id,
        build_record_sha256=value.build_record_sha256,
        machine_id=value.machine_snapshot.id,
        machine_profile_sha256=value.machine_profile_sha256,
        executable_sha256=value.result.executable_sha256,
    )


def _check_draw_budget(coordinates: tuple[CapsuleCoordinateV1, ...]) -> None:
    total_samples = sum(value.sample_count for value in coordinates)
    if BOOTSTRAP_REPLICATES * total_samples > MAX_TOTAL_BOOTSTRAP_DRAWS:
        raise _StatisticsFailure()


def _compare(admission: _Admission) -> CapsuleComparisonResult:
    _check_draw_budget(admission.scenario.coordinates)
    coordinates = tuple(
        _coordinate_statistics(
            coordinate,
            admission.baseline.latency_seconds_by_coordinate[coordinate.coordinate_id],
            admission.candidate.latency_seconds_by_coordinate[coordinate.coordinate_id],
            admission.baseline.workspace_bytes_by_coordinate[coordinate.coordinate_id],
            admission.candidate.workspace_bytes_by_coordinate[coordinate.coordinate_id],
            admission.comparison,
            admission.baseline.record_sha256,
            admission.candidate.record_sha256,
        )
        for coordinate in admission.scenario.coordinates
    )
    comparison_sha = _sha256(canonical_json_bytes(admission.comparison.model_dump(mode="json")))
    report = CapsuleComparisonReportV1(
        comparison=admission.comparison,
        comparison_sha256=comparison_sha,
        capsule_id=admission.baseline.manifest.id,
        scenario_sha256=admission.baseline.manifest.contract.scenario_sha256,
        described_scenario_sha256=_sha256(
            canonical_json_bytes(admission.scenario.model_dump(mode="json"))
        ),
        machine_id=admission.baseline.machine_snapshot.id,
        machine_profile_sha256=admission.baseline.machine_profile_sha256,
        coordinate_structure_sha256=admission.baseline.alignment.coordinate_structure_sha256,
        baseline=_arm("baseline", admission.baseline),
        candidate=_arm("candidate", admission.candidate),
        coordinates=coordinates,
        overall_verdict=_aggregate(coordinates),
    )
    report_bytes = canonical_json_bytes(report.model_dump(mode="json"))
    return CapsuleComparisonResult(
        report=report,
        report_bytes=report_bytes,
        report_sha256=_sha256(report_bytes),
    )


def _try_load(run_id: str, home: Path) -> FinalizedCapsuleSnapshot | None:
    try:
        return load_finalized_capsule_snapshot(run_id, home=home)
    except Exception:
        return None


def _try_admit(
    baseline: FinalizedCapsuleSnapshot,
    candidate: FinalizedCapsuleSnapshot,
) -> _Admission | None:
    try:
        return _admit(baseline, candidate)
    except Exception:
        return None


def _try_compare(admission: _Admission) -> CapsuleComparisonResult | None:
    try:
        return _compare(admission)
    except Exception:
        return None


def compare_finalized_capsule_runs(
    baseline_run_id: str,
    candidate_run_id: str,
    *,
    home: Path,
) -> CapsuleComparisonResult:
    """Load, admit, and directionally compare two finalized successful capsule runs."""

    baseline = _try_load(baseline_run_id, home)
    if baseline is None:
        raise CapsuleComparisonLoadError()
    candidate = _try_load(candidate_run_id, home)
    if candidate is None:
        raise CapsuleComparisonLoadError()
    admission = _try_admit(baseline, candidate)
    if admission is None:
        raise CapsuleComparisonAdmissionError()
    result = _try_compare(admission)
    if result is None:
        raise CapsuleComparisonStatisticsError()
    return result

"""SUITE-001 deterministic smoke-suite executor.

This module composes the three existing ca94157 adapters (``test-backend-ops``,
``llama-server``, ``llama-bench``) into one immutable run. It is a thin orchestration
layer: the adapters keep owning every child process, capability probe, parser, raw
stream, stable-executable check, model lease, and per-case ``sample.json``. SUITE-001
adds no comparison statistics, ranking, candidate pairing, profiler integration,
generic workflow engine, adapter plugin registry, downloader, build creation, or model
verification.

The high-level executor owns ``begin_run``, the machine-profile exclusive lock, the
read-only build lease, the correctness-first protocol, the windowed-interleaved
performance schedule, the portable input/result snapshots, and run finalization. The
adapters retain their current caller-owned :class:`RunSession` contract and never
finalize.
"""

from __future__ import annotations

import contextlib
import hashlib
import shutil
import tempfile
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from strixlab.adapters import backend_ops, llama_bench, llama_server
from strixlab.adapters.backend_ops import (
    BackendOpsCaseV1,
    BackendOpsInputsV1,
    BackendOpsIntegrityError,
    BackendOpsSampleV1,
    run_backend_ops_case,
)
from strixlab.adapters.llama_bench import (
    LlamaBenchCaseV1,
    LlamaBenchInputsV1,
    LlamaBenchIntegrityError,
    LlamaBenchSampleV1,
    run_llama_bench_case,
)
from strixlab.adapters.llama_server import (
    LlamaServerCaseV1,
    LlamaServerInputsV1,
    LlamaServerIntegrityError,
    LlamaServerSampleV1,
    run_llama_server_case,
)
from strixlab.build_artifacts import BuildArtifactsV1
from strixlab.build_cache import BuildCacheError, BuildLease, CanonicalBuildRecordV1, lease_build
from strixlab.build_runtime import (
    BuildRuntimeEnvironment as _RuntimeEnvironment,
)
from strixlab.build_runtime import (
    reconstruct_environment as _build_reconstruct_environment,
)
from strixlab.build_runtime import (
    resolve_target_artifact as _build_resolve_target_artifact,
)
from strixlab.build_runtime import (
    resolve_target_executable as _build_resolve_target_executable,
)
from strixlab.evidence import (
    Clock,
    PortableEvidenceV1,
    RunInspection,
    RunOutcome,
    RunSession,
    TokenFactory,
    begin_run,
    inspect_run,
    list_portable_entries,
    read_record_member,
)
from strixlab.locks import LockAttempt, exclusive_lock
from strixlab.manifests import (
    DASH_ID_PATTERN,
    MachineProfileV1,
    SuiteManifestV1,
    SuitePerformanceCaseV1,
    SuitePerformanceV1,
    SuiteTimeoutsV1,
    suite_greedy_case_id,
    suite_measurement_case_id,
    suite_warmup_case_id,
    validate_manifest,
)
from strixlab.models import (
    ModelError,
    ModelReceiptEvidence,
    ModelReceiptEvidenceV2,
    ModelReceiptV1,
    load_model_receipt,
    receipt_evidence_digest,
    require_current_model,
)
from strixlab.serialization import canonical_json_bytes, canonical_yaml_bytes

__all__ = [
    "FinalizedSuiteSnapshot",
    "PlannedCase",
    "SuiteError",
    "SuiteExecutionError",
    "SuiteHooks",
    "SuiteResultV1",
    "SuiteRunResult",
    "load_finalized_suite_snapshot",
    "plan_performance",
    "run_suite",
]

# The three required build targets, resolved once from the leased canonical inventory.
TARGET_BACKEND_OPS = "test-backend-ops"
TARGET_LLAMA_SERVER = "llama-server"
TARGET_LLAMA_BENCH = "llama-bench"

# ROCm-mode gfx selection entry name.
_GFX_SELECTION_NAME = "gfx_targets"

_ADAPTER_INTEGRITY_ERRORS = (
    BackendOpsIntegrityError,
    LlamaServerIntegrityError,
    LlamaBenchIntegrityError,
    ModelError,
)


# --- Errors -------------------------------------------------------------------


class SuiteError(RuntimeError):
    """A suite could not be validated, bound, or allocated (no run was created)."""


class SuiteExecutionError(SuiteError):
    """A run existed but failed before a structured ``suite/result.json``.

    Carries the finalized run id and record so the caller can report them even though
    no truthful structured result could be published.
    """

    def __init__(self, message: str, *, run_id: str, record: Path | None) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.record = record


# --- Suite-facing closed status mapping ---------------------------------------

SuiteCategory = Literal[
    "success",
    "capability-failed",
    "capture-failed",
    "timed-out",
    "child-failed",
    "malformed-output",
    "hard-gate-failed",
    "server-failed",
    "integrity-failed",
    "adapter-failed",
]

_BACKEND_CATEGORY: dict[str, SuiteCategory] = {
    "passed": "success",
    "capability-failed": "capability-failed",
    "capture-failed": "capture-failed",
    "spawn-failed": "capture-failed",
    "timed-out": "timed-out",
    "oversized-output": "capture-failed",
    "encoding-failed": "capture-failed",
    "child-failed": "child-failed",
    "parse-failed": "malformed-output",
    "hard-gate-failed": "hard-gate-failed",
}
_SERVER_CATEGORY: dict[str, SuiteCategory] = {
    "success": "success",
    "capability-failed": "capability-failed",
    "capture-failed": "capture-failed",
    "spawn-failed": "capture-failed",
    "port-unavailable": "server-failed",
    "readiness-failed": "server-failed",
    "request-failed": "server-failed",
    "response-failed": "malformed-output",
    "isolation-failed": "server-failed",
    "shutdown-failed": "server-failed",
}
# llama-bench folds several process defects under two statuses; its closed ``reason``
# literal refines them into the suite categories.
_BENCH_STATUS_CATEGORY: dict[str, SuiteCategory] = {
    "success": "success",
    "capability-failed": "capability-failed",
    "parse-failed": "malformed-output",
}
_BENCH_REASON_CATEGORY: dict[str, SuiteCategory] = {
    "spawn-failed": "capture-failed",
    "timed-out": "timed-out",
    "capture-failed": "capture-failed",
    "output-oversized": "capture-failed",
    "encoding-failed": "capture-failed",
    "nonzero-exit": "child-failed",
}


def _backend_category(status: str) -> SuiteCategory:
    return _BACKEND_CATEGORY.get(status, "adapter-failed")


def _server_category(status: str) -> SuiteCategory:
    return _SERVER_CATEGORY.get(status, "adapter-failed")


def _bench_category(status: str, reason: str) -> SuiteCategory:
    if status in _BENCH_STATUS_CATEGORY:
        return _BENCH_STATUS_CATEGORY[status]
    if status in ("process-failed", "output-truncated"):
        return _BENCH_REASON_CATEGORY.get(reason, "adapter-failed")
    return "adapter-failed"


# --- Compact result models ----------------------------------------------------

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
RecordSha = Annotated[str, Field(pattern=r"^record-sha256:[0-9a-f]{64}$")]
BuildIdStr = Annotated[str, Field(pattern=r"^build-sha256:[0-9a-f]{64}$")]
DashIdStr = Annotated[str, Field(pattern=DASH_ID_PATTERN, max_length=64)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(ge=1)]
PositiveRate = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]

SuiteReason = Literal[
    "passed",
    "lock-unavailable",
    "backend-ops-failed",
    "greedy-parity-failed",
    "warmup-failed",
    "measurement-failed",
    "integrity-failed",
]
AdapterName = Literal["backend-ops", "llama-server", "llama-bench"]
SamplePhase = Literal["correctness-backend", "correctness-greedy", "warmup", "measurement"]


class _SuiteModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class SuiteInputRefV1(_SuiteModel):
    """An authenticated portable input snapshot bound by logical path and digest."""

    role: Literal["build", "environment"]
    logical_path: str
    sha256: Sha256Hex


class SampleReferenceV1(_SuiteModel):
    """A compact reference to one authenticated portable adapter ``sample.json``."""

    adapter: AdapterName
    phase: SamplePhase
    case_id: DashIdStr
    logical_path: str
    sample_sha256: Sha256Hex
    adapter_status: str
    category: SuiteCategory


class BackendOpsVerdictV1(_SuiteModel):
    sample: SampleReferenceV1
    gate_passed: bool
    passed: bool


class TokenSequenceV1(_SuiteModel):
    ordinal: Literal[1, 2]
    token_count: NonNegativeInt
    tokens_sha256: Sha256Hex


class GreedyPromptVerdictV1(_SuiteModel):
    prompt_id: DashIdStr
    sample: SampleReferenceV1
    responses: tuple[TokenSequenceV1, ...]
    tokens_equal: bool
    passed: bool


class GreedyVerdictV1(_SuiteModel):
    prompts: tuple[GreedyPromptVerdictV1, ...]
    passed: bool


class MeasurementProjectionV1(_SuiteModel):
    case_id: DashIdStr
    window: PositiveInt
    adapter_case_id: DashIdStr
    sample: SampleReferenceV1
    avg_ts: PositiveRate
    stddev_ts: NonNegativeFloat
    samples_ts: tuple[PositiveRate, ...]


class PerformanceScheduleV1(_SuiteModel):
    protocol: Literal["windowed-interleaved-v1"]
    planned_warmups: NonNegativeInt
    planned_measurements: NonNegativeInt
    completed_warmups: NonNegativeInt
    completed_measurements: NonNegativeInt


class SuiteResultV1(_SuiteModel):
    """The terminal, versioned suite result written once to ``suite/result.json``."""

    schema_version: Literal[1] = 1
    suite_id: DashIdStr
    machine_id: DashIdStr
    model_id: DashIdStr
    build_id: BuildIdStr
    canonical_record_sha256: RecordSha
    status: Literal["passed", "failed"]
    reason: SuiteReason
    inputs: tuple[SuiteInputRefV1, ...]
    backend_ops: BackendOpsVerdictV1 | None
    greedy: GreedyVerdictV1 | None
    schedule: PerformanceScheduleV1
    measurements: tuple[MeasurementProjectionV1, ...]
    samples: tuple[SampleReferenceV1, ...]


class _BuildInputSnapshotV1(_SuiteModel):
    schema_version: Literal[1] = 1
    build_id: BuildIdStr
    canonical_record_sha256: RecordSha
    canonical: CanonicalBuildRecordV1


class _ModelInputSnapshotV1(_SuiteModel):
    schema_version: Literal[1] = 1
    model_id: DashIdStr
    model_receipt_sha256: Sha256Hex
    evidence: ModelReceiptEvidenceV2


class _MachineInputSnapshotV1(_SuiteModel):
    schema_version: Literal[1] = 1
    machine_id: DashIdStr
    profile_sha256: Sha256Hex
    profile: MachineProfileV1


# --- Deterministic performance planner ----------------------------------------


@dataclass(frozen=True, slots=True)
class PlannedCase:
    """One planned adapter benchmark invocation with its distinct evidence namespace."""

    phase: Literal["warmup", "measurement"]
    index: int
    case: SuitePerformanceCaseV1
    adapter_case_id: str
    repetitions: int


def plan_performance(performance: SuitePerformanceV1) -> tuple[PlannedCase, ...]:
    """Return the pure ``windowed-interleaved-v1`` schedule for ordered cases.

    Warmup round 1 across all cases in order, warmup round 2, ..., then measurement
    window 1 across all cases in order, ..., through the final measurement window.
    Every warmup uses ``repetitions=1``; every measurement uses
    ``repetitions_per_window``.
    """

    planned: list[PlannedCase] = []
    for index in range(1, performance.warmup_runs + 1):
        for case in performance.cases:
            planned.append(
                PlannedCase("warmup", index, case, suite_warmup_case_id(index, case.id), 1)
            )
    for window in range(1, performance.measurement_windows + 1):
        for case in performance.cases:
            planned.append(
                PlannedCase(
                    "measurement",
                    window,
                    case,
                    suite_measurement_case_id(window, case.id),
                    performance.repetitions_per_window,
                )
            )
    return tuple(planned)


def _backend_case(manifest: SuiteManifestV1) -> BackendOpsCaseV1:
    spec = manifest.correctness.backend_ops
    return BackendOpsCaseV1(
        id=spec.id,
        operations=tuple(spec.operations),
        params_regex=spec.params_regex,
        backend=spec.backend,
    )


def _server_case(manifest: SuiteManifestV1, prompt_id: str) -> LlamaServerCaseV1:
    greedy = manifest.correctness.greedy
    prompt = next((value for value in greedy.prompts if value.id == prompt_id), None)
    if prompt is None:
        raise SuiteError(f"suite manifest has no greedy prompt: {prompt_id}")
    return LlamaServerCaseV1(
        id=suite_greedy_case_id(greedy.id, prompt.id),
        prompt=prompt.text,
        n_predict=greedy.output_tokens,
        seed=greedy.seed,
        context_size=greedy.context_size,
        gpu_layers=greedy.gpu_layers,
    )


def _bench_case(planned: PlannedCase) -> LlamaBenchCaseV1:
    return LlamaBenchCaseV1(
        id=planned.adapter_case_id,
        prompt_tokens=planned.case.prompt_tokens,
        generated_tokens=planned.case.generated_tokens,
        repetitions=planned.repetitions,
        metric_kind=("prompt-processing" if planned.case.prompt_tokens > 0 else "text-generation"),
    )


# --- Injectable dependencies --------------------------------------------------

BackendOpsRunner = Callable[..., BackendOpsSampleV1]
LlamaServerRunner = Callable[..., LlamaServerSampleV1]
LlamaBenchRunner = Callable[..., LlamaBenchSampleV1]
MachineLockFactory = Callable[[Path], AbstractContextManager[LockAttempt]]
TempRootFactory = Callable[[], Path]


def _default_temp_root() -> Path:
    return Path(tempfile.mkdtemp(prefix="strixlab-suite-"))


@dataclass(frozen=True, slots=True)
class SuiteHooks:
    """Narrow, explicit test seams; every field defaults to production behavior."""

    clock: Clock | None = None
    token_factory: TokenFactory | None = None
    temp_root_factory: TempRootFactory = _default_temp_root
    machine_lock: MachineLockFactory = exclusive_lock
    backend_ops: BackendOpsRunner = run_backend_ops_case
    llama_server: LlamaServerRunner = run_llama_server_case
    llama_bench: LlamaBenchRunner = run_llama_bench_case


@dataclass(frozen=True, slots=True)
class SuiteRunResult:
    """The outcome of one suite run whose run was allocated and finalized."""

    run_id: str
    outcome: RunOutcome
    inspection: RunInspection
    result: SuiteResultV1


# --- Build binding and target resolution --------------------------------------


def _require_gfx_target(canonical: CanonicalBuildRecordV1, gfx_target: str) -> None:
    selection = next(
        (entry.value for entry in canonical.selections if entry.name == _GFX_SELECTION_NAME), None
    )
    if selection is None:
        raise SuiteError("leased build does not record a gfx target selection")
    targets = {value for value in selection.split(";") if value}
    if gfx_target not in targets:
        raise SuiteError("leased build gfx target does not match the suite requirement")


def _validate_suite_input_eligibility(
    canonical: CanonicalBuildRecordV1,
    manifest: SuiteManifestV1,
    evidence: ModelReceiptEvidence,
) -> None:
    """Replay the suite's offline-verifiable build and model admission gates."""

    if evidence.manifest_id != manifest.model:
        raise SuiteError("model receipt does not name the suite model")
    if not isinstance(evidence, ModelReceiptEvidenceV2):
        raise SuiteError("model receipt evidence is legacy v1 and cannot prove requirements")
    if evidence.execution.required_sources or evidence.execution.required_features:
        raise SuiteError("model execution requirements are not supported in this smoke suite v1")

    build = manifest.build
    source_evidence = canonical.source.source_evidence
    if source_evidence.get("source_id") != build.source_id:
        raise SuiteError("leased build source id does not match the suite requirement")
    if source_evidence.get("base_commit") != build.source_commit:
        raise SuiteError("leased build source commit does not match the suite requirement")
    if canonical.toolchain_mode != build.toolchain_mode:
        raise SuiteError("leased build toolchain mode does not match the suite requirement")
    _require_gfx_target(canonical, build.gfx_target)


def _resolve_target_artifact(artifacts: BuildArtifactsV1, target_name: str) -> tuple[str, str]:
    return _build_resolve_target_artifact(artifacts, target_name, error=SuiteError)


def _resolve_target_executable(
    artifacts: BuildArtifactsV1, target_name: str, root: Path
) -> tuple[str, str]:
    return _build_resolve_target_executable(artifacts, target_name, root, error=SuiteError)


def _reconstruct_environment(
    canonical: CanonicalBuildRecordV1, root: Path, scratch_root: Path
) -> _RuntimeEnvironment:
    return _build_reconstruct_environment(canonical, root, scratch_root, error=SuiteError)


# --- Adapter sample authentication --------------------------------------------


def _sample_reference_digest(sample: BaseModel) -> str:
    return hashlib.sha256(canonical_json_bytes(sample.model_dump(mode="json"))).hexdigest()


def _authenticate_sample(run: RunSession, logical_path: str, reference_digest: str) -> bool:
    """Authenticate a persisted ``sample.json`` against its portable content-addressed blob.

    Every adapter publishes its terminal ``sample.json`` as a portable entry at the
    predictable logical path (``llama-server`` additionally keeps a local copy beside its
    binary response siblings). A sample with no matching portable entry — a local-only
    write — fails closed rather than being accepted.
    """

    for entry in list_portable_entries(run.active):
        if entry.logical_path == logical_path:
            return entry.blob_sha256 == reference_digest
    return False


@dataclass(frozen=True, slots=True)
class _AdapterOutcome:
    """The classified result of one adapter invocation and sample authentication."""

    category: SuiteCategory
    reference: SampleReferenceV1 | None
    sample: BaseModel | None


def _run_adapter[SampleT: BaseModel](
    run: RunSession,
    *,
    adapter: AdapterName,
    phase: SamplePhase,
    case_id: str,
    logical_path: str,
    invoke: Callable[[], SampleT],
    category_of: Callable[[SampleT], SuiteCategory],
    status_of: Callable[[SampleT], str],
) -> _AdapterOutcome:
    """Invoke one adapter, map thrown integrity errors, and authenticate its sample.

    A thrown adapter/model integrity exception, or a returned sample that does not
    authenticate at its predictable logical path, is an ``integrity-failed`` outcome
    with no completed sample reference. Any other exception propagates (evidence-store
    integrity or unexpected), so ``RunSession.__exit__`` performs its fail-safe failure
    finalization.
    """

    try:
        sample = invoke()
    except _ADAPTER_INTEGRITY_ERRORS:
        return _AdapterOutcome("integrity-failed", None, None)
    reference_digest = _sample_reference_digest(sample)
    if not _authenticate_sample(run, logical_path, reference_digest):
        return _AdapterOutcome("integrity-failed", None, None)
    category = category_of(sample)
    reference = SampleReferenceV1(
        adapter=adapter,
        phase=phase,
        case_id=case_id,
        logical_path=logical_path,
        sample_sha256=reference_digest,
        adapter_status=status_of(sample),
        category=category,
    )
    return _AdapterOutcome(category, reference, sample)


# --- Protocol drive -----------------------------------------------------------


@dataclass(slots=True)
class _ProtocolState:
    samples: list[SampleReferenceV1] = field(default_factory=list)
    backend_ops: BackendOpsVerdictV1 | None = None
    greedy: GreedyVerdictV1 | None = None
    measurements: list[MeasurementProjectionV1] = field(default_factory=list)
    completed_warmups: int = 0
    completed_measurements: int = 0
    status: Literal["passed", "failed"] = "passed"
    reason: SuiteReason = "passed"

    def fail(self, reason: SuiteReason) -> None:
        self.status = "failed"
        self.reason = reason


@dataclass(frozen=True, slots=True)
class _AdapterInputs:
    """Resolved adapter inputs shared across every case of one run."""

    build_id: str
    backend_binary: tuple[str, str]
    server_binary: tuple[str, str]
    bench_binary: tuple[str, str]
    model_id: str
    model_path: str
    model_sha256: str
    receipt_sha256: str


def _bind_adapter_inputs(
    lease: BuildLease, manifest: SuiteManifestV1, receipt: ModelReceiptV1
) -> _AdapterInputs:
    """Validate the build bindings and resolve the three target executables once.

    Fails closed unless the leased canonical build satisfies the suite's source,
    toolchain, and gfx requirements, and resolves each of ``test-backend-ops``,
    ``llama-server``, and ``llama-bench`` to exactly one regular ELF beneath the leased
    root. Called under the held build lease before a run is allocated.
    """

    canonical = lease.canonical
    _validate_suite_input_eligibility(canonical, manifest, receipt.evidence)
    artifacts = canonical.artifacts
    return _AdapterInputs(
        build_id=lease.build_id,
        backend_binary=_resolve_target_executable(artifacts, TARGET_BACKEND_OPS, lease.root),
        server_binary=_resolve_target_executable(artifacts, TARGET_LLAMA_SERVER, lease.root),
        bench_binary=_resolve_target_executable(artifacts, TARGET_LLAMA_BENCH, lease.root),
        model_id=receipt.evidence.manifest_id,
        model_path=receipt.primary.local_path,
        model_sha256=receipt.primary.sha256,
        receipt_sha256=receipt_evidence_digest(receipt.evidence),
    )


def _drive_protocol(
    run: RunSession,
    manifest: SuiteManifestV1,
    receipt: ModelReceiptV1,
    inputs: _AdapterInputs,
    runtime: _RuntimeEnvironment,
    hooks: SuiteHooks,
    server_port: int,
) -> _ProtocolState:
    state = _ProtocolState()
    if not _run_backend_ops(run, manifest, inputs, runtime, hooks, state):
        return state
    if not _run_greedy(run, manifest, receipt, inputs, runtime, hooks, server_port, state):
        return state
    _run_performance(run, manifest, receipt, inputs, runtime, hooks, state)
    return state


def _invoke_server_case(
    run: RunSession,
    hooks: SuiteHooks,
    case: LlamaServerCaseV1,
    adapter_inputs: LlamaServerInputsV1,
    receipt: ModelReceiptV1,
    runtime: _RuntimeEnvironment,
    timeouts: SuiteTimeoutsV1,
    server_port: int,
) -> _AdapterOutcome:
    """Run one greedy server case, authenticating its local ``sample.json``."""

    return _run_adapter(
        run,
        adapter="llama-server",
        phase="correctness-greedy",
        case_id=case.id,
        logical_path=f"{llama_server.EVIDENCE_ROOT}/{case.id}/sample.json",
        invoke=lambda: hooks.llama_server(
            case=case,
            inputs=adapter_inputs,
            receipt=receipt,
            run=run,
            environment=runtime.environment,
            cwd=runtime.cwd,
            port=server_port,
            capability_timeout=timeouts.capability_seconds,
            readiness_timeout=timeouts.server_readiness_seconds,
            request_timeout=timeouts.server_request_seconds,
            shutdown_timeout=timeouts.server_shutdown_seconds,
        ),
        category_of=lambda sample: _server_category(sample.status),
        status_of=lambda sample: sample.status,
    )


def _invoke_bench_case(
    run: RunSession,
    hooks: SuiteHooks,
    case: LlamaBenchCaseV1,
    adapter_inputs: LlamaBenchInputsV1,
    receipt: ModelReceiptV1,
    runtime: _RuntimeEnvironment,
    timeouts: SuiteTimeoutsV1,
    phase: SamplePhase,
) -> _AdapterOutcome:
    """Run one benchmark case, authenticating its portable ``sample.json``."""

    return _run_adapter(
        run,
        adapter="llama-bench",
        phase=phase,
        case_id=case.id,
        logical_path=f"{llama_bench.EVIDENCE_ROOT}/{case.id}/sample.json",
        invoke=lambda: hooks.llama_bench(
            case=case,
            inputs=adapter_inputs,
            receipt=receipt,
            run=run,
            environment=runtime.environment,
            cwd=runtime.cwd,
            capability_timeout=timeouts.capability_seconds,
            benchmark_timeout=timeouts.benchmark_seconds,
        ),
        category_of=lambda sample: _bench_category(sample.status, sample.reason),
        status_of=lambda sample: sample.status,
    )


def _run_backend_ops(
    run: RunSession,
    manifest: SuiteManifestV1,
    inputs: _AdapterInputs,
    runtime: _RuntimeEnvironment,
    hooks: SuiteHooks,
    state: _ProtocolState,
) -> bool:
    timeouts = manifest.timeouts
    case = _backend_case(manifest)
    binary_path, binary_sha256 = inputs.backend_binary
    adapter_inputs = BackendOpsInputsV1(
        build_id=inputs.build_id, binary_path=binary_path, binary_sha256=binary_sha256
    )
    logical_path = f"{backend_ops.EVIDENCE_ROOT}/{case.id}/sample.json"
    outcome = _run_adapter(
        run,
        adapter="backend-ops",
        phase="correctness-backend",
        case_id=case.id,
        logical_path=logical_path,
        invoke=lambda: hooks.backend_ops(
            case=case,
            inputs=adapter_inputs,
            run=run,
            environment=runtime.environment,
            cwd=runtime.cwd,
            capability_timeout=timeouts.capability_seconds,
            test_timeout=timeouts.backend_ops_seconds,
        ),
        category_of=lambda sample: _backend_category(sample.status),
        status_of=lambda sample: sample.status,
    )
    if outcome.reference is None:
        # A thrown adapter integrity error or an unauthenticated sample: no verdict.
        state.fail("integrity-failed")
        return False
    state.samples.append(outcome.reference)
    assert isinstance(outcome.sample, BackendOpsSampleV1)
    gate = outcome.sample.gate
    gate_passed = gate is not None and gate.passed
    # A backend case passes only when the adapter status maps to success AND the gate is
    # present and passed; a tampered success-status sample with a missing/failed gate fails.
    passed = outcome.category == "success" and gate_passed
    state.backend_ops = BackendOpsVerdictV1(
        sample=outcome.reference, gate_passed=gate_passed, passed=passed
    )
    if not passed:
        # An authenticated non-success sample is an ordinary backend gate failure.
        state.fail("backend-ops-failed")
        return False
    return True


def _run_greedy(
    run: RunSession,
    manifest: SuiteManifestV1,
    receipt: ModelReceiptV1,
    inputs: _AdapterInputs,
    runtime: _RuntimeEnvironment,
    hooks: SuiteHooks,
    server_port: int,
    state: _ProtocolState,
) -> bool:
    greedy = manifest.correctness.greedy
    timeouts = manifest.timeouts
    binary_path, binary_sha256 = inputs.server_binary
    # The server inputs bind the binary and model once; only the per-prompt case varies.
    adapter_inputs = LlamaServerInputsV1(
        build_id=inputs.build_id,
        binary_path=binary_path,
        binary_sha256=binary_sha256,
        model_id=inputs.model_id,
        model_path=inputs.model_path,
        model_sha256=inputs.model_sha256,
        model_receipt_sha256=inputs.receipt_sha256,
        model_receipt_evidence=receipt.evidence,
    )
    prompt_verdicts: list[GreedyPromptVerdictV1] = []
    for prompt in greedy.prompts:
        case = _server_case(manifest, prompt.id)
        outcome = _invoke_server_case(
            run, hooks, case, adapter_inputs, receipt, runtime, timeouts, server_port
        )
        if outcome.reference is not None:
            state.samples.append(outcome.reference)
        if outcome.category == "integrity-failed" or outcome.reference is None:
            state.greedy = GreedyVerdictV1(prompts=tuple(prompt_verdicts), passed=False)
            state.fail("integrity-failed")
            return False
        assert isinstance(outcome.sample, LlamaServerSampleV1)
        verdict = _greedy_prompt_verdict(
            prompt.id, outcome.reference, outcome.sample, outcome.category
        )
        prompt_verdicts.append(verdict)
        if not verdict.passed:
            state.greedy = GreedyVerdictV1(prompts=tuple(prompt_verdicts), passed=False)
            state.fail("greedy-parity-failed")
            return False
    state.greedy = GreedyVerdictV1(prompts=tuple(prompt_verdicts), passed=True)
    return True


def _greedy_prompt_verdict(
    prompt_id: str,
    reference: SampleReferenceV1,
    sample: LlamaServerSampleV1,
    category: SuiteCategory,
) -> GreedyPromptVerdictV1:
    responses: list[TokenSequenceV1] = []
    token_tuples: list[tuple[int, ...]] = []
    for attempt in sample.requests:
        if attempt.response is None or attempt.ordinal not in (1, 2):
            continue
        tokens = tuple(attempt.response.tokens)
        token_tuples.append(tokens)
        responses.append(
            TokenSequenceV1(
                ordinal=cast(Literal[1, 2], attempt.ordinal),
                token_count=len(tokens),
                tokens_sha256=hashlib.sha256(canonical_json_bytes(list(tokens))).hexdigest(),
            )
        )
    tokens_equal = len(token_tuples) == 2 and token_tuples[0] == token_tuples[1]
    passed = (
        category == "success"
        and len(token_tuples) == 2
        and all(len(tokens) >= 1 for tokens in token_tuples)
        and tokens_equal
    )
    return GreedyPromptVerdictV1(
        prompt_id=prompt_id,
        sample=reference,
        responses=tuple(responses),
        tokens_equal=tokens_equal,
        passed=passed,
    )


def _run_performance(
    run: RunSession,
    manifest: SuiteManifestV1,
    receipt: ModelReceiptV1,
    inputs: _AdapterInputs,
    runtime: _RuntimeEnvironment,
    hooks: SuiteHooks,
    state: _ProtocolState,
) -> None:
    timeouts = manifest.timeouts
    binary_path, binary_sha256 = inputs.bench_binary
    # The bench inputs bind the binary and model once; only the per-case case varies.
    adapter_inputs = LlamaBenchInputsV1(
        build_id=inputs.build_id,
        binary_path=binary_path,
        binary_sha256=binary_sha256,
        model_id=inputs.model_id,
        model_path=inputs.model_path,
        model_sha256=inputs.model_sha256,
        model_receipt_sha256=inputs.receipt_sha256,
        model_receipt_evidence=receipt.evidence,
    )
    for planned in plan_performance(manifest.performance):
        case = _bench_case(planned)
        outcome = _invoke_bench_case(
            run, hooks, case, adapter_inputs, receipt, runtime, timeouts, planned.phase
        )
        if outcome.reference is not None:
            state.samples.append(outcome.reference)
            if planned.phase == "warmup":
                state.completed_warmups += 1
            else:
                state.completed_measurements += 1
        if outcome.category == "integrity-failed" or outcome.reference is None:
            state.fail("integrity-failed")
            return
        if outcome.category != "success":
            state.fail("warmup-failed" if planned.phase == "warmup" else "measurement-failed")
            return
        if planned.phase == "measurement":
            assert isinstance(outcome.sample, LlamaBenchSampleV1)
            measurement = outcome.sample.measurement
            assert measurement is not None  # a success bench sample binds a measurement
            state.measurements.append(
                MeasurementProjectionV1(
                    case_id=planned.case.id,
                    window=planned.index,
                    adapter_case_id=planned.adapter_case_id,
                    sample=outcome.reference,
                    avg_ts=measurement.avg_ts,
                    stddev_ts=measurement.stddev_ts,
                    samples_ts=tuple(measurement.samples_ts),
                )
            )


# --- Portable input/result snapshots ------------------------------------------


def _publish_input_snapshots(
    run: RunSession,
    lease: BuildLease,
    receipt: ModelReceiptV1,
    machine_profile: MachineProfileV1,
) -> tuple[SuiteInputRefV1, ...]:
    """Write the three authenticated portable input snapshots after ``begin_run``."""

    build_payload = canonical_json_bytes(
        {
            "schema_version": 1,
            "build_id": lease.build_id,
            "canonical_record_sha256": lease.canonical_record_sha256,
            "canonical": lease.canonical.model_dump(mode="json"),
        }
    )
    model_payload = canonical_json_bytes(
        {
            "schema_version": 1,
            "model_id": receipt.evidence.manifest_id,
            "model_receipt_sha256": receipt_evidence_digest(receipt.evidence),
            "evidence": receipt.evidence.model_dump(mode="json"),
        }
    )
    profile_dump = machine_profile.model_dump(mode="json")
    machine_payload = canonical_json_bytes(
        {
            "schema_version": 1,
            "machine_id": machine_profile.id,
            "profile_sha256": hashlib.sha256(canonical_json_bytes(profile_dump)).hexdigest(),
            "profile": profile_dump,
        }
    )
    refs: list[SuiteInputRefV1] = []
    snapshots: tuple[tuple[str, bytes, Literal["build", "environment"]], ...] = (
        ("suite/build.json", build_payload, "build"),
        ("suite/model.json", model_payload, "environment"),
        ("suite/machine.json", machine_payload, "environment"),
    )
    for logical_path, payload, role in snapshots:
        entry = run.write_portable(logical_path, payload, media_type="application/json", role=role)
        refs.append(
            SuiteInputRefV1(role=role, logical_path=entry.logical_path, sha256=entry.blob_sha256)
        )
    return tuple(refs)


def _build_result(
    manifest: SuiteManifestV1,
    lease: BuildLease,
    receipt: ModelReceiptV1,
    inputs: tuple[SuiteInputRefV1, ...],
    state: _ProtocolState,
) -> SuiteResultV1:
    performance = manifest.performance
    planned_warmups = len(performance.cases) * performance.warmup_runs
    planned_measurements = len(performance.cases) * performance.measurement_windows
    return SuiteResultV1(
        suite_id=manifest.id,
        machine_id=manifest.machine,
        model_id=manifest.model,
        build_id=lease.build_id,
        canonical_record_sha256=lease.canonical_record_sha256,
        status=state.status,
        reason=state.reason,
        inputs=inputs,
        backend_ops=state.backend_ops,
        greedy=state.greedy,
        schedule=PerformanceScheduleV1(
            protocol=performance.protocol,
            planned_warmups=planned_warmups,
            planned_measurements=planned_measurements,
            completed_warmups=state.completed_warmups,
            completed_measurements=state.completed_measurements,
        ),
        measurements=tuple(state.measurements),
        samples=tuple(state.samples),
    )


def _lock_unavailable_result(
    manifest: SuiteManifestV1,
    lease: BuildLease,
    receipt: ModelReceiptV1,
    inputs: tuple[SuiteInputRefV1, ...],
) -> SuiteResultV1:
    state = _ProtocolState()
    state.fail("lock-unavailable")
    return _build_result(manifest, lease, receipt, inputs, state)


def _publish_result(run: RunSession, result: SuiteResultV1) -> None:
    run.write_portable(
        "suite/result.json",
        canonical_json_bytes(result.model_dump(mode="json")),
        media_type="application/json",
        role="summary",
    )


# --- Executor -----------------------------------------------------------------


def run_suite(
    manifest: SuiteManifestV1,
    manifest_input: bytes,
    *,
    machine_profile: MachineProfileV1,
    build_id: str,
    local_receipt_sha256: str,
    home: Path,
    server_port: int = 18080,
    environ: Mapping[str, str],
    hooks: SuiteHooks | None = None,
) -> SuiteRunResult:
    """Compose the three adapters into one immutable, finalized suite run.

    Validates the manifests and authenticates the model receipt, then acquires the
    read-only build lease, binds the build, allocates a run, acquires the machine's
    exclusive lock, runs the correctness-first protocol and windowed-interleaved
    performance schedule, writes one compact ``suite/result.json``, and finalizes the
    run as success or failure. Acquisition order is build lease, run, machine lock;
    release order is machine lock, run finalization, build lease.
    """

    hooks = hooks or SuiteHooks()
    if machine_profile.id != manifest.machine:
        raise SuiteError("machine profile id does not match the suite manifest")
    try:
        receipt = load_model_receipt(manifest.model, local_receipt_sha256, home=home)
    except ModelError as exc:
        raise SuiteError(f"model receipt authentication failed: {exc}") from exc
    try:
        require_current_model(receipt)
    except ModelError as exc:
        raise SuiteError(f"model receipt is no longer valid: {exc}") from exc

    resolved = manifest.model_dump(mode="json")
    try:
        with lease_build(build_id, home=home) as lease:
            inputs = _bind_adapter_inputs(lease, manifest, receipt)
            lease.verify()
            run = begin_run(
                manifest.id,
                manifest_input,
                resolved=resolved,
                home=home,
                environ=environ,
                clock=hooks.clock,
                token_factory=hooks.token_factory,
            )
            return _execute_run(
                run, manifest, machine_profile, lease, receipt, inputs, hooks, server_port, home
            )
    except BuildCacheError as exc:
        raise SuiteError(f"build lease failed: {exc}") from exc


def _execute_run(
    run: RunSession,
    manifest: SuiteManifestV1,
    machine_profile: MachineProfileV1,
    lease: BuildLease,
    receipt: ModelReceiptV1,
    inputs: _AdapterInputs,
    hooks: SuiteHooks,
    server_port: int,
    home: Path,
) -> SuiteRunResult:
    run_id = run.run_id
    try:
        with run:
            snapshots = _publish_input_snapshots(run, lease, receipt, machine_profile)
            result = _drive_under_machine_lock(
                run,
                manifest,
                machine_profile,
                lease,
                receipt,
                inputs,
                snapshots,
                hooks,
                server_port,
            )
            outcome = RunOutcome.SUCCESS if result.status == "passed" else RunOutcome.FAILURE
            inspection = (
                run.succeed()
                if outcome is RunOutcome.SUCCESS
                else run.fail(f"suite:{result.reason}")
            )
            return SuiteRunResult(run_id, outcome, inspection, result)
    except Exception as exc:  # noqa: BLE001 - run.__exit__ finalized FAILURE without a result
        record: Path | None = None
        with contextlib.suppress(Exception):
            record = inspect_run(run_id, home=home).record
        raise SuiteExecutionError(
            "suite run failed before producing a structured result", run_id=run_id, record=record
        ) from exc


def _drive_under_machine_lock(
    run: RunSession,
    manifest: SuiteManifestV1,
    machine_profile: MachineProfileV1,
    lease: BuildLease,
    receipt: ModelReceiptV1,
    inputs: _AdapterInputs,
    snapshots: tuple[SuiteInputRefV1, ...],
    hooks: SuiteHooks,
    server_port: int,
) -> SuiteResultV1:
    lock_path = Path(machine_profile.exclusive_lock.path)
    with hooks.machine_lock(lock_path) as lock:
        if not lock.acquired:
            lease.verify()
            result = _lock_unavailable_result(manifest, lease, receipt, snapshots)
            _publish_result(run, result)
            return result
        scratch_root = hooks.temp_root_factory()
        try:
            runtime = _reconstruct_environment(lease.canonical, lease.root, scratch_root)
            state = _drive_protocol(run, manifest, receipt, inputs, runtime, hooks, server_port)
        finally:
            # The scratch root is removed on every exit (including adapter failure). A
            # deletion failure is not swallowed: it escapes so the run finalizes failure
            # without publishing a successful suite/result.json.
            shutil.rmtree(scratch_root)
        lease.verify()
        result = _build_result(manifest, lease, receipt, snapshots, state)
        _publish_result(run, result)
        return result


# --- Authenticated finalized-suite snapshot -----------------------------------
#
# The one reusable, descriptor-anchored seam that a downstream comparison judge
# (JUDGE-001) uses to obtain an immutable, fully re-authenticated view of one
# finalized *successful* smoke-suite run without reopening any unchecked path. Every
# byte is read through ``read_record_member`` (owned, no-follow, identity-checked) and
# rebound to its content address, and every stored projection is recomputed from the
# authenticated terminal adapter samples and required to match exactly, so a canonical
# but semantically misbound record fails closed.

_INPUT_SNAPSHOT_SPECS: tuple[tuple[str, Literal["build", "environment"], type[BaseModel]], ...] = (
    ("suite/build.json", "build", _BuildInputSnapshotV1),
    ("suite/model.json", "environment", _ModelInputSnapshotV1),
    ("suite/machine.json", "environment", _MachineInputSnapshotV1),
)

_SAMPLE_MODELS: dict[AdapterName, type[BaseModel]] = {
    "backend-ops": BackendOpsSampleV1,
    "llama-server": LlamaServerSampleV1,
    "llama-bench": LlamaBenchSampleV1,
}


@dataclass(frozen=True, slots=True)
class FinalizedSuiteSnapshot:
    """An immutable, fully re-authenticated view of one finalized successful suite run.

    Every field is bound to the run's authenticated immutable record: the resolved
    manifest bytes and their digest, the strictly parsed manifest and result, the
    ``suite/result.json`` blob digest, the build/model/machine input snapshot digests,
    and the ordered per-case measurement samples used for paired comparison. The record
    digest is the authenticated ``record-sha256:`` identity returned by ``inspect_run``.
    """

    run_id: str
    record: Path
    record_sha256: str
    resolved_manifest_bytes: bytes
    resolved_manifest_sha256: str
    manifest: SuiteManifestV1
    result: SuiteResultV1
    result_sha256: str
    build_id: str
    build_record_sha256: str
    model_input_sha256: str
    machine_input_sha256: str
    case_order: tuple[str, ...]
    measurement_windows: int
    repetitions_per_window: int
    case_samples: dict[str, tuple[float, ...]]


@dataclass(frozen=True, slots=True)
class _AuthenticatedInputs:
    build: _BuildInputSnapshotV1
    model: _ModelInputSnapshotV1
    machine: _MachineInputSnapshotV1
    model_sha256: str
    machine_sha256: str


def _snapshot_blob(
    record: Path, entries: dict[str, PortableEvidenceV1], logical_path: str
) -> tuple[PortableEvidenceV1, bytes]:
    """Read and re-authenticate one portable blob from a finalized record by logical path."""

    entry = entries.get(logical_path)
    if entry is None:
        raise SuiteError(f"finalized run is missing portable evidence: {logical_path}")
    content = read_record_member(record, f"portable/blobs/{entry.blob_sha256}")
    if hashlib.sha256(content).hexdigest() != entry.blob_sha256 or len(content) != entry.size_bytes:
        raise SuiteError(f"portable blob diverged from its index entry: {logical_path}")
    return entry, content


def _parse_canonical_json[ModelT: BaseModel](
    content: bytes,
    model: type[ModelT],
    *,
    validation_error: str,
    canonical_error: str,
) -> ModelT:
    """Strictly parse canonical JSON into one closed evidence model."""

    try:
        value = model.model_validate_json(content)
    except ValidationError as exc:
        raise SuiteError(validation_error) from exc
    if canonical_json_bytes(value.model_dump(mode="json")) != content:
        raise SuiteError(canonical_error)
    return value


def _expected_sample_coordinates(
    manifest: SuiteManifestV1, planned_cases: tuple[PlannedCase, ...]
) -> tuple[tuple[AdapterName, SamplePhase, str, str], ...]:
    """The exact ordered ``(adapter, phase, case_id, logical_path)`` schedule of a passed run."""

    coordinates: list[tuple[AdapterName, SamplePhase, str, str]] = []
    backend_id = manifest.correctness.backend_ops.id
    coordinates.append(
        (
            "backend-ops",
            "correctness-backend",
            backend_id,
            f"{backend_ops.EVIDENCE_ROOT}/{backend_id}/sample.json",
        )
    )
    greedy = manifest.correctness.greedy
    for prompt in greedy.prompts:
        case_id = suite_greedy_case_id(greedy.id, prompt.id)
        coordinates.append(
            (
                "llama-server",
                "correctness-greedy",
                case_id,
                f"{llama_server.EVIDENCE_ROOT}/{case_id}/sample.json",
            )
        )
    for planned in planned_cases:
        case_id = planned.adapter_case_id
        coordinates.append(
            (
                "llama-bench",
                planned.phase,
                case_id,
                f"{llama_bench.EVIDENCE_ROOT}/{case_id}/sample.json",
            )
        )
    return tuple(coordinates)


def _bind_result_to_manifest(result: SuiteResultV1, manifest: SuiteManifestV1) -> None:
    """Require a canonical result to be a passed, complete projection of its manifest."""

    if (
        result.suite_id != manifest.id
        or result.machine_id != manifest.machine
        or result.model_id != manifest.model
    ):
        raise SuiteError("suite result identities do not match the resolved manifest")
    if result.status != "passed" or result.reason != "passed":
        raise SuiteError("suite result is not a passed run")
    if result.backend_ops is None or not result.backend_ops.passed:
        raise SuiteError("suite result has no passing backend-ops projection")
    if result.greedy is None or not result.greedy.passed:
        raise SuiteError("suite result has no passing greedy projection")
    if len(result.greedy.prompts) != len(manifest.correctness.greedy.prompts):
        raise SuiteError("greedy projection does not cover every manifest prompt")
    performance = manifest.performance
    schedule = result.schedule
    planned_warmups = len(performance.cases) * performance.warmup_runs
    planned_measurements = len(performance.cases) * performance.measurement_windows
    if (
        schedule.protocol != performance.protocol
        or schedule.planned_warmups != planned_warmups
        or schedule.planned_measurements != planned_measurements
        or schedule.completed_warmups != planned_warmups
        or schedule.completed_measurements != planned_measurements
    ):
        raise SuiteError("suite result schedule does not match the manifest or is incomplete")


def _bind_input_snapshots(
    record: Path,
    entries: dict[str, PortableEvidenceV1],
    manifest: SuiteManifestV1,
    result: SuiteResultV1,
) -> _AuthenticatedInputs:
    """Authenticate the three input snapshots and bind them to the result fields."""

    if len(result.inputs) != len(_INPUT_SNAPSHOT_SPECS):
        raise SuiteError("suite result does not carry exactly the three input snapshots")
    parsed: dict[str, BaseModel] = {}
    digests: dict[str, str] = {}
    for ref, (logical_path, role, model_type) in zip(
        result.inputs, _INPUT_SNAPSHOT_SPECS, strict=True
    ):
        if ref.logical_path != logical_path or ref.role != role:
            raise SuiteError("suite result input reference is misbound")
        entry, content = _snapshot_blob(record, entries, logical_path)
        if entry.role != role or entry.media_type != "application/json":
            raise SuiteError(f"input snapshot has the wrong role or media type: {logical_path}")
        if entry.blob_sha256 != ref.sha256:
            raise SuiteError(f"input reference digest diverged from its blob: {logical_path}")
        parsed[logical_path] = _parse_canonical_json(
            content,
            model_type,
            validation_error=f"input snapshot failed strict validation: {logical_path}",
            canonical_error=f"input snapshot is not canonical: {logical_path}",
        )
        digests[logical_path] = entry.blob_sha256

    build = cast(_BuildInputSnapshotV1, parsed["suite/build.json"])
    model = cast(_ModelInputSnapshotV1, parsed["suite/model.json"])
    machine = cast(_MachineInputSnapshotV1, parsed["suite/machine.json"])
    return _authenticate_input_models(
        build,
        model,
        machine,
        manifest,
        result,
        model_sha256=digests["suite/model.json"],
        machine_sha256=digests["suite/machine.json"],
    )


def _authenticate_input_models(
    build: _BuildInputSnapshotV1,
    model: _ModelInputSnapshotV1,
    machine: _MachineInputSnapshotV1,
    manifest: SuiteManifestV1,
    result: SuiteResultV1,
    *,
    model_sha256: str,
    machine_sha256: str,
) -> _AuthenticatedInputs:
    """Recompute embedded identities after strict canonical snapshot parsing."""

    _validate_suite_input_eligibility(build.canonical, manifest, model.evidence)
    canonical_digest = (
        "record-sha256:"
        + hashlib.sha256(canonical_json_bytes(build.canonical.model_dump(mode="json"))).hexdigest()
    )
    if (
        build.build_id != result.build_id
        or build.canonical.build_id != build.build_id
        or build.canonical_record_sha256 != result.canonical_record_sha256
        or build.canonical_record_sha256 != canonical_digest
    ):
        raise SuiteError("build snapshot does not authenticate the result build identity")
    if (
        model.model_id != result.model_id
        or model.evidence.manifest_id != model.model_id
        or model.model_receipt_sha256 != receipt_evidence_digest(model.evidence)
    ):
        raise SuiteError("model snapshot does not authenticate the result model identity")
    profile_digest = hashlib.sha256(
        canonical_json_bytes(machine.profile.model_dump(mode="json"))
    ).hexdigest()
    if (
        machine.machine_id != result.machine_id
        or machine.profile.id != machine.machine_id
        or machine.profile_sha256 != profile_digest
    ):
        raise SuiteError("machine snapshot does not authenticate the result machine identity")
    return _AuthenticatedInputs(
        build=build,
        model=model,
        machine=machine,
        model_sha256=model_sha256,
        machine_sha256=machine_sha256,
    )


def _reauthenticate_samples(
    record: Path,
    entries: dict[str, PortableEvidenceV1],
    manifest: SuiteManifestV1,
    result: SuiteResultV1,
    inputs: _AuthenticatedInputs,
) -> None:
    """Recompute every correctness and measurement projection from terminal samples.

    Requires ``result.samples`` to be the exact ordered schedule of a passed run, each
    reference to authenticate against its portable blob, and every backend-gate, greedy
    token, and measurement projection to recompute exactly from the authenticated
    terminal sample. A canonical result that claims a pass its terminal evidence does not
    support fails closed.
    """

    planned_cases = plan_performance(manifest.performance)
    expected = _expected_sample_coordinates(manifest, planned_cases)
    if len(result.samples) != len(expected):
        raise SuiteError("suite result sample set is not the exact passed-run schedule")
    parsed: list[BaseModel] = []
    for reference, (adapter, phase, case_id, logical_path) in zip(
        result.samples, expected, strict=True
    ):
        if (
            reference.adapter != adapter
            or reference.phase != phase
            or reference.case_id != case_id
            or reference.logical_path != logical_path
        ):
            raise SuiteError("suite result sample ordering is inauthentic")
        entry, content = _snapshot_blob(record, entries, logical_path)
        expected_role = "correctness" if adapter == "backend-ops" else "samples"
        if entry.role != expected_role or entry.media_type != "application/json":
            raise SuiteError(f"sample has the wrong role or media type: {logical_path}")
        if entry.blob_sha256 != reference.sample_sha256:
            raise SuiteError(f"sample reference digest is inauthentic: {logical_path}")
        sample = _parse_canonical_json(
            content,
            _SAMPLE_MODELS[adapter],
            validation_error=f"portable sample failed strict validation: {logical_path}",
            canonical_error=f"portable sample is not canonical: {logical_path}",
        )
        status, category = _sample_status_category(adapter, sample)
        if reference.adapter_status != status:
            raise SuiteError(f"sample status diverged from its reference: {logical_path}")
        if reference.category != category:
            raise SuiteError(f"sample category diverged from its reference: {logical_path}")
        parsed.append(sample)

    _bind_sample_contracts(manifest, inputs, parsed, planned_cases)
    _reauthenticate_backend(result, parsed[0])
    greedy = manifest.correctness.greedy
    for index, prompt in enumerate(greedy.prompts):
        _reauthenticate_greedy(result, index, prompt.id, parsed[1 + index])
    _reauthenticate_measurements(manifest, result, parsed, planned_cases)


def _bind_sample_contracts(
    manifest: SuiteManifestV1,
    inputs: _AuthenticatedInputs,
    parsed: list[BaseModel],
    planned_cases: tuple[PlannedCase, ...],
) -> None:
    """Bind every typed adapter case and input projection to authenticated suite inputs."""

    backend = cast(BackendOpsSampleV1, parsed[0])
    if _backend_category(backend.status) != "success":
        raise SuiteError("passed suite contains a non-success backend-ops sample")
    if backend.case != _backend_case(manifest):
        raise SuiteError("backend-ops sample case diverged from the suite manifest")
    _bind_adapter_sample_inputs(
        backend.inputs, manifest, inputs, target_name=TARGET_BACKEND_OPS, model_bound=False
    )

    offset = 1
    for prompt in manifest.correctness.greedy.prompts:
        server = cast(LlamaServerSampleV1, parsed[offset])
        offset += 1
        if _server_category(server.status) != "success":
            raise SuiteError("passed suite contains a non-success llama-server sample")
        if server.case != _server_case(manifest, prompt.id):
            raise SuiteError("llama-server sample case diverged from the suite manifest")
        _bind_adapter_sample_inputs(
            server.inputs, manifest, inputs, target_name=TARGET_LLAMA_SERVER, model_bound=True
        )

    for planned in planned_cases:
        bench = cast(LlamaBenchSampleV1, parsed[offset])
        offset += 1
        if _bench_category(bench.status, bench.reason) != "success":
            raise SuiteError("passed suite contains a non-success llama-bench sample")
        if bench.case != _bench_case(planned):
            raise SuiteError("llama-bench sample case diverged from the suite manifest")
        _bind_adapter_sample_inputs(
            bench.inputs, manifest, inputs, target_name=TARGET_LLAMA_BENCH, model_bound=True
        )


def _bind_adapter_sample_inputs(
    value: BackendOpsInputsV1 | LlamaServerInputsV1 | LlamaBenchInputsV1,
    manifest: SuiteManifestV1,
    inputs: _AuthenticatedInputs,
    *,
    target_name: str,
    model_bound: bool,
) -> None:
    """Bind an adapter input projection to build/model snapshots and target inventory."""

    relative_path, binary_sha256 = _resolve_target_artifact(
        inputs.build.canonical.artifacts, target_name
    )
    binary_parts = PurePosixPath(value.binary_path).parts
    relative_parts = PurePosixPath(relative_path).parts
    if (
        value.build_id != inputs.build.build_id
        or value.source_commit != manifest.build.source_commit
        or value.binary_sha256 != binary_sha256
        or binary_parts[-len(relative_parts) :] != relative_parts
    ):
        raise SuiteError("adapter sample inputs diverged from authenticated build evidence")
    if not model_bound:
        return
    model_value = cast(LlamaServerInputsV1 | LlamaBenchInputsV1, value)
    evidence = inputs.model.evidence
    if (
        model_value.model_id != inputs.model.model_id
        or model_value.model_path != evidence.primary_local_path
        or model_value.model_sha256 != evidence.primary_sha256
        or model_value.model_receipt_sha256 != inputs.model.model_receipt_sha256
        or model_value.model_receipt_evidence != evidence
    ):
        raise SuiteError("adapter sample inputs diverged from authenticated model evidence")


def _sample_status_category(adapter: AdapterName, sample: BaseModel) -> tuple[str, SuiteCategory]:
    """Return one authenticated adapter sample's status and its closed suite category."""

    if adapter == "backend-ops":
        backend = cast(BackendOpsSampleV1, sample)
        return backend.status, _backend_category(backend.status)
    if adapter == "llama-server":
        server = cast(LlamaServerSampleV1, sample)
        return server.status, _server_category(server.status)
    bench = cast(LlamaBenchSampleV1, sample)
    return bench.status, _bench_category(bench.status, bench.reason)


def _reauthenticate_backend(result: SuiteResultV1, sample: BaseModel) -> None:
    backend_sample = cast(BackendOpsSampleV1, sample)
    verdict = result.backend_ops
    assert verdict is not None  # bound by _bind_result_to_manifest
    gate = backend_sample.gate
    gate_passed = gate is not None and gate.passed
    passed = _backend_category(backend_sample.status) == "success" and gate_passed
    if (
        verdict.sample != result.samples[0]
        or verdict.gate_passed != gate_passed
        or verdict.passed != passed
        or not passed
    ):
        raise SuiteError("backend-ops projection diverged from its authenticated sample")


def _reauthenticate_greedy(
    result: SuiteResultV1, index: int, prompt_id: str, sample: BaseModel
) -> None:
    server_sample = cast(LlamaServerSampleV1, sample)
    greedy = result.greedy
    assert greedy is not None  # bound by _bind_result_to_manifest
    if index >= len(greedy.prompts):
        raise SuiteError("greedy projection is missing a prompt verdict")
    stored = greedy.prompts[index]
    reference = result.samples[1 + index]
    recomputed = _greedy_prompt_verdict(
        prompt_id, reference, server_sample, _server_category(server_sample.status)
    )
    if stored != recomputed or not stored.passed:
        raise SuiteError("greedy projection diverged from its authenticated sample")


def _reauthenticate_measurements(
    manifest: SuiteManifestV1,
    result: SuiteResultV1,
    parsed: list[BaseModel],
    planned_cases: tuple[PlannedCase, ...],
) -> None:
    performance = manifest.performance
    reps = performance.repetitions_per_window
    bench_start = 1 + len(manifest.correctness.greedy.prompts)
    projections = result.measurements
    projection_index = 0
    for offset, planned in enumerate(planned_cases):
        if planned.phase != "measurement":
            continue
        if projection_index >= len(projections):
            raise SuiteError("suite result is missing a measurement projection")
        projection = projections[projection_index]
        projection_index += 1
        reference = result.samples[bench_start + offset]
        bench_sample = cast(LlamaBenchSampleV1, parsed[bench_start + offset])
        measurement = bench_sample.measurement
        if measurement is None:
            raise SuiteError("measurement sample carries no measurement")
        if (
            projection.case_id != planned.case.id
            or projection.window != planned.index
            or projection.adapter_case_id != planned.adapter_case_id
            or projection.sample != reference
            or projection.avg_ts != measurement.avg_ts
            or projection.stddev_ts != measurement.stddev_ts
            or tuple(projection.samples_ts) != tuple(measurement.samples_ts)
            or len(projection.samples_ts) != reps
        ):
            raise SuiteError("measurement projection diverged from its authenticated sample")
    if projection_index != len(projections):
        raise SuiteError("suite result carries an unexpected measurement projection")


def _measurement_coordinates(
    manifest: SuiteManifestV1, result: SuiteResultV1
) -> dict[str, tuple[float, ...]]:
    """Flatten measurements into ordered per-case samples keyed by manifest case id.

    Requires exactly one measurement projection per ``(case_id, window)`` coordinate over
    the manifest's declared cases and windows, each carrying ``repetitions_per_window``
    samples, and folds them into a per-case sequence in ``(window, repetition_index)``
    order. Duplicate or missing coordinates and incidental cases fail closed.
    """

    performance = manifest.performance
    windows = performance.measurement_windows
    reps = performance.repetitions_per_window
    case_order = tuple(case.id for case in performance.cases)
    by_case: dict[str, dict[int, tuple[float, ...]]] = {}
    for projection in result.measurements:
        windows_map = by_case.setdefault(projection.case_id, {})
        if projection.window in windows_map:
            raise SuiteError("duplicate measurement coordinate")
        windows_map[projection.window] = tuple(projection.samples_ts)
    if set(by_case) != set(case_order):
        raise SuiteError("measurement projections reference unexpected cases")
    coordinates: dict[str, tuple[float, ...]] = {}
    for case_id in case_order:
        windows_map = by_case[case_id]
        if set(windows_map) != set(range(1, windows + 1)):
            raise SuiteError("measurement coordinates do not cover the planned windows")
        flat: list[float] = []
        for window in range(1, windows + 1):
            samples = windows_map[window]
            if len(samples) != reps:
                raise SuiteError("measurement coordinate has the wrong repetition count")
            flat.extend(samples)
        coordinates[case_id] = tuple(flat)
    return coordinates


def load_finalized_suite_snapshot(run_id: str, *, home: Path) -> FinalizedSuiteSnapshot:
    """Load and fully re-authenticate one finalized, successful smoke-suite run.

    Inspects the run (requiring ``RunOutcome.SUCCESS`` and retaining the authenticated
    record digest), then reads and rebinds — through descriptor-anchored, no-follow,
    content-addressed reads — the resolved manifest, the ``suite/result.json`` summary,
    the three input snapshots, and every correctness and measurement sample, recomputing
    each stored projection from the authenticated terminal samples. Every failure —
    inauthentic bytes, a non-successful outcome, or a canonical-but-misbound record — is
    raised as :class:`SuiteError`; the caller maps it to its own load-failure taxonomy.
    """

    inspection = inspect_run(run_id, home=home)
    if inspection.outcome is not RunOutcome.SUCCESS:
        raise SuiteError("finalized run did not finish successfully")
    record = inspection.record

    resolved_bytes = read_record_member(record, "manifest.resolved.yaml")
    try:
        manifest = validate_manifest("suite", yaml.safe_load(resolved_bytes))
    except (ValidationError, ValueError, yaml.YAMLError) as exc:
        raise SuiteError("resolved manifest failed strict suite validation") from exc
    if not isinstance(manifest, SuiteManifestV1):
        raise SuiteError("resolved manifest is not a suite manifest")
    if canonical_yaml_bytes(manifest.model_dump(mode="json")) != resolved_bytes:
        raise SuiteError("resolved manifest bytes are not canonical")

    entries_list = list_portable_entries(record)
    entries = {entry.logical_path: entry for entry in entries_list}
    if len(entries) != len(entries_list):
        raise SuiteError("finalized run has duplicate portable logical paths")

    result_entry, result_bytes = _snapshot_blob(record, entries, "suite/result.json")
    if result_entry.role != "summary" or result_entry.media_type != "application/json":
        raise SuiteError("suite result entry has the wrong role or media type")
    result = _parse_canonical_json(
        result_bytes,
        SuiteResultV1,
        validation_error="suite result failed strict validation",
        canonical_error="suite result bytes are not canonical",
    )

    _bind_result_to_manifest(result, manifest)
    inputs = _bind_input_snapshots(record, entries, manifest, result)
    _reauthenticate_samples(record, entries, manifest, result, inputs)
    coordinates = _measurement_coordinates(manifest, result)

    return FinalizedSuiteSnapshot(
        run_id=inspection.run_id,
        record=record,
        record_sha256=inspection.record_sha256,
        resolved_manifest_bytes=resolved_bytes,
        resolved_manifest_sha256=hashlib.sha256(resolved_bytes).hexdigest(),
        manifest=manifest,
        result=result,
        result_sha256=result_entry.blob_sha256,
        build_id=result.build_id,
        build_record_sha256=result.canonical_record_sha256,
        model_input_sha256=inputs.model_sha256,
        machine_input_sha256=inputs.machine_sha256,
        case_order=tuple(case.id for case in manifest.performance.cases),
        measurement_windows=manifest.performance.measurement_windows,
        repetitions_per_window=manifest.performance.repetitions_per_window,
        case_samples=coordinates,
    )

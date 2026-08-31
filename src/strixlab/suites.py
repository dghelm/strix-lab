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
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StrictInt

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
from strixlab.build_identity import ROOT_PLACEHOLDERS
from strixlab.evidence import (
    Clock,
    RunInspection,
    RunOutcome,
    RunSession,
    TokenFactory,
    begin_run,
    inspect_run,
    list_portable_entries,
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
)
from strixlab.models import (
    ModelError,
    ModelReceiptEvidenceV2,
    ModelReceiptV1,
    load_model_receipt,
    receipt_evidence_digest,
    require_current_model,
)
from strixlab.serialization import canonical_json_bytes

__all__ = [
    "PlannedCase",
    "SuiteError",
    "SuiteExecutionError",
    "SuiteHooks",
    "SuiteResultV1",
    "SuiteRunResult",
    "plan_performance",
    "run_suite",
]

# The three required build targets, resolved once from the leased canonical inventory.
TARGET_BACKEND_OPS = "test-backend-ops"
TARGET_LLAMA_SERVER = "llama-server"
TARGET_LLAMA_BENCH = "llama-bench"

# ROCm-mode gfx selection entry name and the sole rehydratable placeholder.
_GFX_SELECTION_NAME = "gfx_targets"
_BUILD_ROOT_PLACEHOLDER = ROOT_PLACEHOLDERS["BUILD_ROOT"]
_PLACEHOLDER_RE = re.compile(r"\{[A-Za-z_][A-Za-z0-9_]*\}")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REQUIRED_LOCALE = {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"}

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


def _resolve_target_executable(
    artifacts: BuildArtifactsV1, target_name: str, root: Path
) -> tuple[str, str]:
    """Resolve one required target to exactly one regular ELF executable beneath root.

    Returns the absolute binary path and the recorded artifact SHA-256; the adapters
    re-hash and recheck each executable.
    """

    named = [target for target in artifacts.targets if target.name == target_name]
    if len(named) != 1:
        raise SuiteError(f"build target is missing or ambiguous: {target_name}")
    if named[0].target_type != "EXECUTABLE":
        raise SuiteError(f"build target is not an executable: {target_name}")
    candidates = [
        artifact
        for artifact in artifacts.artifacts
        if target_name in artifact.targets
        and artifact.kind == "elf"
        and artifact.elf_type in ("ET_EXEC", "ET_DYN")
        and not artifact.runtime_dependency
    ]
    if len(candidates) != 1:
        raise SuiteError(f"expected exactly one executable artifact for target: {target_name}")
    artifact = candidates[0]
    relative = PurePosixPath(artifact.path)
    if relative.is_absolute() or ".." in relative.parts:
        raise SuiteError(f"build artifact escapes the leased root: {target_name}")
    return str(root / relative), artifact.sha256


# --- Hermetic runtime environment ---------------------------------------------


@dataclass(frozen=True, slots=True)
class _RuntimeEnvironment:
    environment: dict[str, str]
    cwd: Path
    scratch_root: Path


def _reconstruct_environment(
    canonical: CanonicalBuildRecordV1, root: Path, scratch_root: Path
) -> _RuntimeEnvironment:
    """Rebuild the adapter child environment from the leased canonical build tuple.

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
            raise SuiteError(f"leased build environment has an invalid name: {entry.name!r}")
        if entry.name in seen:
            raise SuiteError(f"leased build environment has a duplicate name: {entry.name!r}")
        seen.add(entry.name)
        if "\x00" in entry.name or "\x00" in entry.value:
            raise SuiteError("leased build environment contains a NUL byte")
        if entry.name == "HOME":
            environment["HOME"] = str(scratch_home)
        elif entry.name == "TMPDIR":
            environment["TMPDIR"] = str(scratch_tmp)
        else:
            environment[entry.name] = _rehydrate_value(entry.value, root_str)

    for name in ("HOME", "TMPDIR"):
        if name not in seen:
            raise SuiteError(f"leased build environment is missing {name}")
    for name, expected in _REQUIRED_LOCALE.items():
        if environment.get(name) != expected:
            raise SuiteError(f"leased build environment has an unexpected {name}")
    return _RuntimeEnvironment(environment=environment, cwd=scratch_tmp, scratch_root=scratch_root)


def _rehydrate_value(value: str, root: str) -> str:
    return os.pathsep.join(
        _rehydrate_component(component, root) for component in value.split(os.pathsep)
    )


def _rehydrate_component(component: str, root: str) -> str:
    if component == _BUILD_ROOT_PLACEHOLDER:
        return root
    if component.startswith(_BUILD_ROOT_PLACEHOLDER + os.sep):
        rest = component[len(_BUILD_ROOT_PLACEHOLDER) :]
        if _PLACEHOLDER_RE.search(rest):
            raise SuiteError("leased build environment component has an unexpected placeholder")
        return root + rest
    if _PLACEHOLDER_RE.search(component):
        raise SuiteError("leased build environment component has an unknown placeholder")
    return component


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
    build = manifest.build
    evidence = canonical.source.source_evidence
    if evidence.get("source_id") != build.source_id:
        raise SuiteError("leased build source id does not match the suite requirement")
    if evidence.get("base_commit") != build.source_commit:
        raise SuiteError("leased build source commit does not match the suite requirement")
    if canonical.toolchain_mode != build.toolchain_mode:
        raise SuiteError("leased build toolchain mode does not match the suite requirement")
    _require_gfx_target(canonical, build.gfx_target)
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
    spec = manifest.correctness.backend_ops
    timeouts = manifest.timeouts
    case = BackendOpsCaseV1(
        id=spec.id,
        operations=tuple(spec.operations),
        params_regex=spec.params_regex,
        backend=spec.backend,
    )
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
        case = LlamaServerCaseV1(
            id=suite_greedy_case_id(greedy.id, prompt.id),
            prompt=prompt.text,
            n_predict=greedy.output_tokens,
            seed=greedy.seed,
            context_size=greedy.context_size,
            gpu_layers=greedy.gpu_layers,
        )
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
        case = LlamaBenchCaseV1(
            id=planned.adapter_case_id,
            prompt_tokens=planned.case.prompt_tokens,
            generated_tokens=planned.case.generated_tokens,
            repetitions=planned.repetitions,
            metric_kind=(
                "prompt-processing" if planned.case.prompt_tokens > 0 else "text-generation"
            ),
        )
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
    if receipt.evidence.manifest_id != manifest.model:
        raise SuiteError("model receipt does not name the suite model")
    evidence = receipt.evidence
    if not isinstance(evidence, ModelReceiptEvidenceV2):
        # A legacy v1 receipt carries no authenticated execution projection, so it cannot
        # prove the model declares no execution requirements: it is ineligible before a
        # run is allocated.
        raise SuiteError("model receipt evidence is legacy v1 and cannot prove requirements")
    if evidence.execution.required_sources or evidence.execution.required_features:
        # This smoke v1 supports only models with no execution requirements. The
        # requirement set is authenticated by the receipt/evidence digests, so emptiness
        # is never inferred: a non-empty set fails closed before a run is allocated.
        raise SuiteError("model execution requirements are not supported in this smoke suite v1")
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

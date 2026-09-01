"""Fixed, library-only ``native-capsule-v1`` protocol adapter.

The caller owns every orchestration boundary: a live :class:`RunSession`, manifest
resolution and identity, build authentication, executable selection, the complete child
environment and working directory, scratch allocation, machine locking, and final run
outcome.  This module only executes the three fixed protocol children, validates their
closed responses, and publishes a portable ``capsule/protocol`` evidence subtree.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import math
import os
import secrets
import stat
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StrictFloat,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from strixlab.evidence import RunError, RunSession, list_portable_entries
from strixlab.executable_identity import (
    ExecutableIdentity,
    hash_executable,
    require_stable_executable,
)
from strixlab.manifests import CapsuleManifestV1, DashId, Sha256Lower
from strixlab.process import ProcessOutcome, ProcessResult, run_process
from strixlab.secret_policy import (
    RedactionContext,
    SensitiveInterpolationError,
    UnsafeOutputError,
    reject_sensitive_interpolations,
)
from strixlab.secure_fs import readonly_open_flags, write_all
from strixlab.serialization import canonical_json_bytes

__all__ = [
    "BenchmarkCoordinateV1",
    "BenchmarkResponseV1",
    "CapsuleCoordinateV1",
    "CapsuleIntegrityError",
    "CapsulePhaseResultV1",
    "CapsuleProcessV1",
    "CapsuleProtocolResultV1",
    "CapsuleRequestV1",
    "CapsuleScenarioContractV1",
    "CorrectnessCoordinateV1",
    "CorrectnessResponseV1",
    "DescribeResponseV1",
    "run_capsule_protocol",
]

PROTOCOL: Final = "native-capsule-v1"
OPERATIONS: Final = ("describe", "correctness", "benchmark")
EVIDENCE_ROOT: Final = "capsule/protocol"
STDOUT_LIMIT_BYTES = 1024 * 1024
STDERR_LIMIT_BYTES = 256 * 1024
MAX_COORDINATES = 128
MAX_SAMPLES_PER_COORDINATE = 4096
MAX_OPAQUE_PAYLOAD_BYTES = 256 * 1024

Operation = Literal["describe", "correctness", "benchmark"]
FailureReason = Literal[
    "passed",
    "describe-process-failed",
    "describe-response-invalid",
    "describe-unsafe-output",
    "correctness-process-failed",
    "correctness-response-invalid",
    "correctness-unsafe-output",
    "correctness-incomplete",
    "correctness-failed",
    "benchmark-process-failed",
    "benchmark-response-invalid",
    "benchmark-unsafe-output",
    "benchmark-incomplete",
]
ProcessCategory = Literal[
    "none",
    "capture-failed",
    "spawn-failed",
    "timed-out",
    "nonzero-exit",
    "incomplete-capture",
]
PhaseFailure = Literal["none", "process", "response", "unsafe"]

# Python builds need not expose Linux's memfd/seal constants even though the running
# kernel and libc support them. This adapter is already Linux-specific because the
# protocol path is /proc/self/fd/N, so keep the ABI constants private and explicit.
_MFD_CLOEXEC = 0x0001
_MFD_ALLOW_SEALING = 0x0002
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_F_SEAL_SEAL = 0x0001
_F_SEAL_SHRINK = 0x0002
_F_SEAL_GROW = 0x0004
_F_SEAL_WRITE = 0x0008
_REQUEST_SEALS = _F_SEAL_SEAL | _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_WRITE


class CapsuleIntegrityError(RuntimeError):
    """Executable or evidence identity no longer supports a truthful result."""


class _CapsuleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


PositiveLatency = Annotated[StrictFloat, Field(gt=0)]
NonNegativeBytes = Annotated[StrictInt, Field(ge=0)]
SampleCount = Annotated[StrictInt, Field(ge=1, le=MAX_SAMPLES_PER_COORDINATE)]
WarmupCount = Annotated[StrictInt, Field(ge=0, le=MAX_SAMPLES_PER_COORDINATE)]


class CapsuleCoordinateV1(_CapsuleModel):
    """One exact ordered comparison coordinate declared by ``describe``."""

    coordinate_id: DashId
    case_set: Literal["training", "evaluation"]
    mode: DashId
    order: Annotated[StrictInt, Field(ge=0, lt=MAX_COORDINATES)]
    input_id: DashId
    input_sha256: Sha256Lower
    warmup_count: WarmupCount
    sample_count: SampleCount
    metric: Literal["latency-seconds"] = "latency-seconds"
    direction: Literal["lower-is-better"] = "lower-is-better"
    policy: Literal["topk-paired-log-bootstrap-v1"] = "topk-paired-log-bootstrap-v1"


class CapsuleScenarioContractV1(_CapsuleModel):
    """The comparison-complete, scenario-pinned contract returned by ``describe``."""

    schema_version: Literal[1] = 1
    coordinates: Annotated[tuple[CapsuleCoordinateV1, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _ordered_unique(self) -> Self:
        if len(self.coordinates) > MAX_COORDINATES:
            raise ValueError("scenario exceeds the coordinate limit")
        ids = tuple(value.coordinate_id for value in self.coordinates)
        if len(ids) != len(set(ids)):
            raise ValueError("coordinate ids must be unique")
        if tuple(value.order for value in self.coordinates) != tuple(range(len(self.coordinates))):
            raise ValueError("coordinate order must be the exact zero-based tuple order")
        return self


class CapsuleRequestV1(_CapsuleModel):
    """Canonical request delivered through one inherited read-only descriptor."""

    schema_version: Literal[1] = 1
    protocol: Literal["native-capsule-v1"] = PROTOCOL
    operation: Operation
    capsule_id: DashId
    candidate: DashId
    scenario_sha256: Sha256Lower
    manifest_sha256: Sha256Lower
    executable_sha256: Sha256Lower
    prior_response_sha256: Sha256Lower | None
    scenario_contract_sha256: Sha256Lower | None
    scenario: CapsuleScenarioContractV1 | None

    @model_validator(mode="after")
    def _operation_shape(self) -> Self:
        later = self.operation != "describe"
        values_present = (
            self.prior_response_sha256 is not None
            and self.scenario_contract_sha256 is not None
            and self.scenario is not None
        )
        if later != values_present:
            raise ValueError("only later requests carry the accepted scenario and response chain")
        return self


class _ResponseBindingV1(_CapsuleModel):
    schema_version: Literal[1] = 1
    protocol: Literal["native-capsule-v1"] = PROTOCOL
    operation: Operation
    request_sha256: Sha256Lower
    capsule_id: DashId
    candidate: DashId
    scenario_sha256: Sha256Lower
    manifest_sha256: Sha256Lower
    executable_sha256: Sha256Lower
    prior_response_sha256: Sha256Lower | None
    scenario_contract_sha256: Sha256Lower | None
    opaque_payload: JsonValue | None = None

    @field_validator("opaque_payload")
    @classmethod
    def _bounded_opaque_payload(cls, value: JsonValue | None) -> JsonValue | None:
        if value is not None and len(canonical_json_bytes(value)) > MAX_OPAQUE_PAYLOAD_BYTES:
            raise ValueError("opaque payload exceeds the canonical byte limit")
        return value


class DescribeResponseV1(_ResponseBindingV1):
    operation: Literal["describe"]
    prior_response_sha256: None
    scenario_contract_sha256: None
    scenario: CapsuleScenarioContractV1


class CorrectnessCoordinateV1(_CapsuleModel):
    coordinate: CapsuleCoordinateV1
    passed: bool


class CorrectnessResponseV1(_ResponseBindingV1):
    operation: Literal["correctness"]
    prior_response_sha256: Sha256Lower
    scenario_contract_sha256: Sha256Lower
    coordinates: Annotated[tuple[CorrectnessCoordinateV1, ...], Field(min_length=1)]


class BenchmarkCoordinateV1(_CapsuleModel):
    coordinate: CapsuleCoordinateV1
    latency_seconds: Annotated[tuple[PositiveLatency, ...], Field(min_length=1)]
    workspace_bytes: NonNegativeBytes

    @model_validator(mode="after")
    def _declared_sample_count(self) -> Self:
        if len(self.latency_seconds) != self.coordinate.sample_count:
            raise ValueError("latency sample count does not equal the described count")
        return self


class BenchmarkResponseV1(_ResponseBindingV1):
    operation: Literal["benchmark"]
    prior_response_sha256: Sha256Lower
    scenario_contract_sha256: Sha256Lower
    coordinates: Annotated[tuple[BenchmarkCoordinateV1, ...], Field(min_length=1)]


class CapsuleProcessV1(_CapsuleModel):
    """Secret-free exact-stream projection of one bounded child."""

    schema_version: Literal[1] = 1
    outcome: Literal["exited", "timed_out", "spawn_failed", "capture_failed"]
    returncode: StrictInt | None
    duration_seconds: Annotated[float, Field(ge=0)]
    stdout_bytes: Annotated[StrictInt, Field(ge=0)]
    stderr_bytes: Annotated[StrictInt, Field(ge=0)]
    stdout_sha256: Sha256Lower
    stderr_sha256: Sha256Lower
    stdout_complete: bool
    stderr_complete: bool
    stdout_truncated: bool
    stderr_truncated: bool
    capture_error: bool
    category: ProcessCategory

    @model_validator(mode="after")
    def _exact_category(self) -> Self:
        if self.outcome == "spawn_failed":
            expected: ProcessCategory = "spawn-failed"
            if self.returncode is not None:
                raise ValueError("a spawn failure cannot carry a return code")
        elif self.outcome == "timed_out":
            expected = "timed-out"
            if self.returncode is None:
                raise ValueError("a timed-out child must carry its termination return code")
        elif self.outcome == "capture_failed":
            expected = "capture-failed"
            if self.returncode is None or not self.capture_error:
                raise ValueError("a capture failure must carry return code and capture error")
        else:
            if self.returncode is None:
                raise ValueError("an exited child must carry a return code")
            if self.returncode != 0:
                expected = "nonzero-exit"
            elif (
                self.capture_error
                or self.stdout_truncated
                or self.stderr_truncated
                or not self.stdout_complete
                or not self.stderr_complete
            ):
                expected = "incomplete-capture"
            else:
                expected = "none"
        if self.category != expected:
            raise ValueError("process category is not the exact structured outcome")
        return self


class CapsulePhaseResultV1(_CapsuleModel):
    schema_version: Literal[1] = 1
    operation: Operation
    request_sha256: Sha256Lower
    process: CapsuleProcessV1
    response_sha256: Sha256Lower | None
    accepted: bool
    failure: PhaseFailure

    @model_validator(mode="after")
    def _accepted_digest(self) -> Self:
        if self.failure == "none":
            valid_state = self.accepted and self.response_sha256 is not None
        else:
            valid_state = not self.accepted and self.response_sha256 is None
        if not valid_state:
            raise ValueError("accepted, response digest, and phase failure disagree")
        if self.accepted and self.process.category != "none":
            raise ValueError("an accepted phase must have a successful complete process")
        if self.failure == "process" and self.process.category == "none":
            raise ValueError("a process failure must have a failed process category")
        if self.failure in {"response", "unsafe"} and self.process.category != "none":
            raise ValueError("response and safety failures require a complete successful process")
        return self


class CapsuleProtocolResultV1(_CapsuleModel):
    """Strict terminal protocol projection; the caller still owns the run outcome."""

    schema_version: Literal[1] = 1
    protocol: Literal["native-capsule-v1"] = PROTOCOL
    capsule_id: DashId
    candidate: DashId
    scenario_sha256: Sha256Lower
    manifest_sha256: Sha256Lower
    executable_sha256: Sha256Lower
    status: Literal["passed", "failed"]
    reason: FailureReason
    phases: tuple[CapsulePhaseResultV1, ...]
    scenario: CapsuleScenarioContractV1 | None
    correctness: tuple[CorrectnessCoordinateV1, ...] | None
    benchmark: tuple[BenchmarkCoordinateV1, ...] | None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        operations = tuple(phase.operation for phase in self.phases)
        if not operations or len(operations) > len(OPERATIONS):
            raise ValueError("terminal result must contain one to three phase results")
        if operations != OPERATIONS[: len(operations)]:
            raise ValueError("phase results must be an exact operation prefix")
        if any(not phase.accepted for phase in self.phases[:-1]):
            raise ValueError("only the terminal phase may be rejected")
        correctness_matches = (
            self.scenario is not None
            and self.correctness is not None
            and tuple(value.coordinate for value in self.correctness) == self.scenario.coordinates
        )
        benchmark_matches = (
            self.scenario is not None
            and self.benchmark is not None
            and tuple(value.coordinate for value in self.benchmark) == self.scenario.coordinates
        )
        correctness_passed = (
            correctness_matches
            and self.correctness is not None
            and all(value.passed for value in self.correctness)
        )
        if (self.status == "passed") != (self.reason == "passed"):
            raise ValueError("terminal status and reason disagree")

        phase_failures: dict[Operation, dict[PhaseFailure, FailureReason]] = {
            operation: {
                "process": cast(FailureReason, f"{operation}-process-failed"),
                "response": cast(FailureReason, f"{operation}-response-invalid"),
                "unsafe": cast(FailureReason, f"{operation}-unsafe-output"),
                "none": "passed",
            }
            for operation_value in OPERATIONS
            for operation in (cast(Operation, operation_value),)
        }
        last = self.phases[-1]
        if last.failure != "none":
            expected_reason = phase_failures[last.operation][last.failure]
            if self.reason != expected_reason:
                raise ValueError("terminal reason does not match the rejected phase")
            if last.accepted:
                raise ValueError("a failed terminal phase cannot be accepted")
            if last.operation == "describe":
                valid_shape = (
                    self.scenario is None and self.correctness is None and self.benchmark is None
                )
            elif last.operation == "correctness":
                valid_shape = (
                    self.scenario is not None
                    and self.correctness is None
                    and self.benchmark is None
                )
            else:
                valid_shape = (
                    correctness_passed
                    and self.scenario is not None
                    and self.correctness is not None
                    and self.benchmark is None
                )
            if not valid_shape:
                raise ValueError("terminal payload shape does not match the rejected phase")
            return self

        if not all(phase.accepted for phase in self.phases):
            raise ValueError("a semantic terminal result requires accepted phases")
        if self.reason == "correctness-incomplete":
            valid = (
                operations == OPERATIONS[:2]
                and self.scenario is not None
                and self.correctness is not None
                and not correctness_matches
                and self.benchmark is None
            )
        elif self.reason == "correctness-failed":
            valid = (
                operations == OPERATIONS[:2]
                and correctness_matches
                and self.correctness is not None
                and not all(value.passed for value in self.correctness)
                and self.benchmark is None
            )
        elif self.reason == "benchmark-incomplete":
            valid = (
                operations == OPERATIONS
                and correctness_passed
                and self.benchmark is not None
                and not benchmark_matches
            )
        elif self.reason == "passed":
            valid = operations == OPERATIONS and correctness_passed and benchmark_matches
        else:
            valid = False
        if not valid:
            raise ValueError("terminal reason and accepted protocol state disagree")
        return self


@dataclass(frozen=True, slots=True)
class _CapturedChild:
    result: ProcessResult
    projection: CapsuleProcessV1
    stdout: bytes | None
    stderr: bytes | None


@dataclass(frozen=True, slots=True)
class _RequestHandle:
    descriptor: int
    identity: tuple[int, int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _ExecutedPhase:
    phase: CapsulePhaseResultV1
    response: DescribeResponseV1 | CorrectnessResponseV1 | BenchmarkResponseV1 | None
    invalid_kind: Literal["process", "response", "unsafe"] | None


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _validate_sha256(value: str, *, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be exactly 64 lowercase hexadecimal characters")
    return value


def _combined_redaction(run: RunSession, supplied: RedactionContext) -> RedactionContext:
    values = set(run.context.secrets)
    values.update(supplied.secrets)
    return RedactionContext(tuple(sorted(values, key=len, reverse=True)))


def _validate_scratch_root(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError("scratch_root must be absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CapsuleIntegrityError("capsule scratch root is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CapsuleIntegrityError("capsule scratch root is not a real directory")
    if metadata.st_uid != os.geteuid():
        raise CapsuleIntegrityError("capsule scratch root is owned by another user")


def _preflight_evidence(run: RunSession) -> None:
    try:
        entries = list_portable_entries(run.active)
    except (OSError, RunError) as exc:
        raise CapsuleIntegrityError("capsule evidence inventory is unavailable") from exc
    if any(
        entry.logical_path == EVIDENCE_ROOT or entry.logical_path.startswith(f"{EVIDENCE_ROOT}/")
        for entry in entries
    ):
        raise CapsuleIntegrityError("capsule protocol evidence subtree already exists")


def _hash_initial_executable(path: Path, expected_sha256: str) -> ExecutableIdentity:
    identity = hash_executable(path, error=CapsuleIntegrityError, subject="capsule executable")
    if identity.sha256 != expected_sha256:
        raise CapsuleIntegrityError("capsule executable digest does not match the trusted digest")
    return identity


def _require_executable(path: Path, identity: ExecutableIdentity) -> None:
    require_stable_executable(
        path, identity, error=CapsuleIntegrityError, subject="capsule executable"
    )


def _memfd_create(name: str) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    function = libc.memfd_create
    function.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    function.restype = ctypes.c_int
    descriptor = int(function(name.encode(), _MFD_CLOEXEC | _MFD_ALLOW_SEALING))
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return descriptor


def _request_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mode,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _write_request_fd(operation: Operation, payload: bytes) -> _RequestHandle:
    writer: int | None = None
    descriptor: int | None = None
    try:
        writer = _memfd_create(f"strixlab-capsule-{operation}-request")
        write_all(writer, payload)
        fcntl.fcntl(writer, _F_ADD_SEALS, _REQUEST_SEALS)
        descriptor = os.open(f"/proc/self/fd/{writer}", os.O_RDONLY | os.O_CLOEXEC)
        os.close(writer)
        writer = None
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        if writer is not None:
            os.close(writer)
        raise CapsuleIntegrityError("capsule request descriptor could not be prepared") from exc
    handle = _RequestHandle(descriptor, _request_identity(os.fstat(descriptor)))
    _read_request_fd(handle, payload)
    return handle


def _read_request_fd(handle: _RequestHandle, expected: bytes) -> None:
    descriptor = handle.descriptor
    try:
        access_mode = fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
        seals = fcntl.fcntl(descriptor, _F_GET_SEALS)
        before = os.fstat(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 64 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise CapsuleIntegrityError("capsule request descriptor became unreadable") from exc
    before_identity = _request_identity(before)
    after_identity = _request_identity(after)
    if (
        access_mode != os.O_RDONLY
        or seals != _REQUEST_SEALS
        or not stat.S_ISREG(before.st_mode)
        or before_identity != handle.identity
        or after_identity != handle.identity
        or b"".join(chunks) != expected
    ):
        raise CapsuleIntegrityError("capsule request evidence drifted across the child")


def _read_spool(path: Path, *, expected_size: int, expected_sha256: str) -> bytes:
    try:
        descriptor = os.open(path, readonly_open_flags())
    except OSError as exc:
        raise CapsuleIntegrityError("capsule process spool is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid():
            raise CapsuleIntegrityError("capsule process spool is not an owned regular file")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 64 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    stable = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mode,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mode,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if (
        not stable
        or len(payload) != expected_size
        or before.st_size != expected_size
        or _sha256(payload) != expected_sha256
    ):
        raise CapsuleIntegrityError("capsule process spool drifted from the process digest")
    return payload


def _capture_spool(
    reported: Path | None,
    requested: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> bytes | None:
    if reported is None:
        return None
    if reported.absolute() != requested.absolute():
        raise CapsuleIntegrityError("capsule runner returned an unexpected spool path")
    return _read_spool(requested, expected_size=expected_size, expected_sha256=expected_sha256)


def _process_category(
    result: ProcessResult, stdout: bytes | None, stderr: bytes | None
) -> ProcessCategory:
    if result.outcome is ProcessOutcome.CAPTURE_FAILED:
        return "capture-failed"
    if result.outcome is ProcessOutcome.SPAWN_FAILED:
        return "spawn-failed"
    if result.outcome is ProcessOutcome.TIMED_OUT:
        return "timed-out"
    if result.returncode != 0:
        return "nonzero-exit"
    if (
        result.capture_error is not None
        or result.stdout_truncated
        or result.stderr_truncated
        or stdout is None
        or stderr is None
    ):
        return "incomplete-capture"
    return "none"


def _run_bounded_child(
    *,
    executable_path: Path,
    identity: ExecutableIdentity,
    operation: Operation,
    request: bytes,
    cwd: Path,
    environment: Mapping[str, str],
    scratch_root: Path,
    timeout: float,
) -> _CapturedChild:
    _require_executable(executable_path, identity)
    request_handle = _write_request_fd(operation, request)
    descriptor = request_handle.descriptor
    stdout_path = scratch_root / f".capsule-{operation}-stdout-{secrets.token_hex(12)}"
    stderr_path = scratch_root / f".capsule-{operation}-stderr-{secrets.token_hex(12)}"
    argv = (
        str(executable_path),
        operation,
        "--request",
        f"/proc/self/fd/{descriptor}",
    )
    try:
        result = run_process(
            argv,
            cwd=cwd,
            timeout=timeout,
            inherit_env=False,
            base_env=environment,
            output_limit_bytes=STDOUT_LIMIT_BYTES,
            stdout_total_limit_bytes=STDOUT_LIMIT_BYTES,
            stderr_total_limit_bytes=STDERR_LIMIT_BYTES,
            stdout_spool=stdout_path,
            stderr_spool=stderr_path,
            spool_root=scratch_root,
            pass_fds=(descriptor,),
        )
        _read_request_fd(request_handle, request)
        _require_executable(executable_path, identity)
        if result.argv != argv:
            raise CapsuleIntegrityError("capsule runner reported a divergent argv")
        if result.capture_error is not None and "spool-" in result.capture_error:
            raise CapsuleIntegrityError("capsule runner could not preserve exact stream evidence")
        stdout = _capture_spool(
            result.stdout_spool,
            stdout_path,
            expected_size=result.stdout_bytes,
            expected_sha256=result.stdout_sha256,
        )
        stderr = _capture_spool(
            result.stderr_spool,
            stderr_path,
            expected_size=result.stderr_bytes,
            expected_sha256=result.stderr_sha256,
        )
        if result.outcome is not ProcessOutcome.CAPTURE_FAILED and (
            stdout is None or stderr is None
        ):
            raise CapsuleIntegrityError("capsule runner omitted exact bounded stream evidence")
    finally:
        os.close(descriptor)
        for path in (stdout_path, stderr_path):
            with suppress(OSError):
                path.unlink(missing_ok=True)
    category = _process_category(result, stdout, stderr)
    projection = CapsuleProcessV1(
        outcome=cast(Any, result.outcome.value),
        returncode=result.returncode,
        duration_seconds=result.duration,
        stdout_bytes=result.stdout_bytes,
        stderr_bytes=result.stderr_bytes,
        stdout_sha256=result.stdout_sha256,
        stderr_sha256=result.stderr_sha256,
        stdout_complete=stdout is not None,
        stderr_complete=stderr is not None,
        stdout_truncated=result.stdout_truncated,
        stderr_truncated=result.stderr_truncated,
        capture_error=result.capture_error is not None,
        category=category,
    )
    return _CapturedChild(result, projection, stdout, stderr)


def _strict_json(payload: bytes) -> tuple[JsonValue | None, str | None]:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None, None
    try:
        value = json.loads(
            text,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON token: {token}")
            ),
        )
    except (json.JSONDecodeError, RecursionError, ValueError):
        return None, text
    if not isinstance(value, (dict, list, str, int, float, bool)) and value is not None:
        return None, text
    if isinstance(value, float) and not math.isfinite(value):
        return None, text
    try:
        canonical = canonical_json_bytes(value)
    except (RecursionError, UnicodeEncodeError):
        return None, text
    if canonical != payload:
        return None, text
    return cast(JsonValue, value), text


def _safe(context: RedactionContext, payload: bytes | None) -> bool:
    if payload is None:
        return True
    try:
        context.assert_payload_safe(payload)
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return True
        reject_sensitive_interpolations(text)
    except (SensitiveInterpolationError, UnsafeOutputError):
        return False
    return True


def _response_model(
    operation: Operation, payload: bytes
) -> DescribeResponseV1 | CorrectnessResponseV1 | BenchmarkResponseV1:
    model: type[DescribeResponseV1] | type[CorrectnessResponseV1] | type[BenchmarkResponseV1]
    if operation == "describe":
        model = DescribeResponseV1
    elif operation == "correctness":
        model = CorrectnessResponseV1
    else:
        model = BenchmarkResponseV1
    return model.model_validate_json(payload)


def _binding_matches(
    response: _ResponseBindingV1,
    request: CapsuleRequestV1,
    request_sha256: str,
) -> bool:
    return (
        response.operation == request.operation
        and response.request_sha256 == request_sha256
        and response.capsule_id == request.capsule_id
        and response.candidate == request.candidate
        and response.scenario_sha256 == request.scenario_sha256
        and response.manifest_sha256 == request.manifest_sha256
        and response.executable_sha256 == request.executable_sha256
        and response.prior_response_sha256 == request.prior_response_sha256
        and response.scenario_contract_sha256 == request.scenario_contract_sha256
    )


def _publish_artifacts(
    run: RunSession,
    context: RedactionContext,
    operation: Operation,
    artifacts: list[tuple[str, bytes, str]],
) -> None:
    role = "samples" if operation == "benchmark" else "correctness"
    try:
        for _, payload, _ in artifacts:
            if not _safe(context, payload):
                raise UnsafeOutputError("phase artifact failed evidence-safety validation")
        for name, payload, media_type in artifacts:
            run.write_portable(
                f"{EVIDENCE_ROOT}/{operation}/{name}",
                payload,
                media_type=media_type,
                role=role,
            )
    except (OSError, RunError, UnsafeOutputError) as exc:
        raise CapsuleIntegrityError("capsule phase evidence could not be published") from exc


def _execute_phase(
    *,
    run: RunSession,
    context: RedactionContext,
    executable_path: Path,
    identity: ExecutableIdentity,
    operation: Operation,
    request: CapsuleRequestV1,
    cwd: Path,
    environment: Mapping[str, str],
    scratch_root: Path,
    timeout: float,
) -> _ExecutedPhase:
    request_bytes = canonical_json_bytes(request.model_dump(mode="json"))
    try:
        context.assert_payload_safe(request_bytes)
    except UnsafeOutputError as exc:
        raise CapsuleIntegrityError("capsule request failed secret-safety validation") from exc
    request_sha256 = _sha256(request_bytes)
    captured = _run_bounded_child(
        executable_path=executable_path,
        identity=identity,
        operation=operation,
        request=request_bytes,
        cwd=cwd,
        environment=environment,
        scratch_root=scratch_root,
        timeout=timeout,
    )

    stdout_safe = _safe(context, captured.stdout)
    stderr_safe = _safe(context, captured.stderr)
    unsafe = not stdout_safe or not stderr_safe
    value, stdout_text = (None, None)
    if captured.stdout is not None and stdout_safe:
        value, stdout_text = _strict_json(captured.stdout)

    artifacts: list[tuple[str, bytes, str]] = [
        ("request.json", request_bytes, "application/json"),
        (
            "process.json",
            canonical_json_bytes(captured.projection.model_dump(mode="json")),
            "application/json",
        ),
    ]
    if value is not None and captured.stdout is not None:
        artifacts.append(("stdout.json", captured.stdout, "application/json"))
    elif stdout_text is not None and captured.stdout is not None and stdout_safe:
        artifacts.append(("stdout.txt", captured.stdout, "text/plain"))
    else:
        reason = (
            "stdout withheld: secret-safety validation failed\n"
            if not stdout_safe
            else "stdout unavailable as complete canonical UTF-8 JSON\n"
        )
        artifacts.append(("stdout-unavailable.txt", reason.encode(), "text/plain"))
    if captured.stderr:
        try:
            captured.stderr.decode("utf-8", errors="strict")
            stderr_utf8 = True
        except UnicodeDecodeError:
            stderr_utf8 = False
        conflicts_with_json = any(
            captured.stderr == payload and media_type != "text/plain"
            for _, payload, media_type in artifacts
        )
        if stderr_safe and stderr_utf8 and not conflicts_with_json:
            artifacts.append(("stderr.txt", captured.stderr, "text/plain"))

    response: DescribeResponseV1 | CorrectnessResponseV1 | BenchmarkResponseV1 | None = None
    response_sha256: str | None = None
    invalid_kind: Literal["process", "response", "unsafe"] | None
    if captured.projection.category != "none":
        invalid_kind = "process"
    elif unsafe:
        invalid_kind = "unsafe"
    elif value is None or captured.stdout is None:
        invalid_kind = "response"
    else:
        try:
            candidate_response = _response_model(operation, captured.stdout)
        except ValidationError:
            invalid_kind = "response"
        else:
            if not _binding_matches(candidate_response, request, request_sha256):
                invalid_kind = "response"
            else:
                response = candidate_response
                response_sha256 = _sha256(captured.stdout)
                invalid_kind = None

    _publish_artifacts(run, context, operation, artifacts)
    return _ExecutedPhase(
        CapsulePhaseResultV1(
            operation=operation,
            request_sha256=request_sha256,
            process=captured.projection,
            response_sha256=response_sha256,
            accepted=response is not None,
            failure="none" if invalid_kind is None else invalid_kind,
        ),
        response,
        invalid_kind,
    )


def _request(
    manifest: CapsuleManifestV1,
    *,
    manifest_sha256: str,
    executable_sha256: str,
    operation: Operation,
    prior_response_sha256: str | None,
    scenario: CapsuleScenarioContractV1 | None,
) -> CapsuleRequestV1:
    contract_sha = (
        None
        if scenario is None
        else _sha256(canonical_json_bytes(scenario.model_dump(mode="json")))
    )
    return CapsuleRequestV1(
        operation=operation,
        capsule_id=manifest.id,
        candidate=manifest.candidate,
        scenario_sha256=manifest.contract.scenario_sha256,
        manifest_sha256=manifest_sha256,
        executable_sha256=executable_sha256,
        prior_response_sha256=prior_response_sha256,
        scenario_contract_sha256=contract_sha,
        scenario=scenario,
    )


def _phase_failure(operation: Operation, kind: str) -> FailureReason:
    if kind == "unsafe":
        suffix = "unsafe-output"
    elif kind == "process":
        suffix = "process-failed"
    else:
        suffix = "response-invalid"
    return cast(FailureReason, f"{operation}-{suffix}")


def _publish_result(
    run: RunSession,
    context: RedactionContext,
    executable_path: Path,
    identity: ExecutableIdentity,
    result: CapsuleProtocolResultV1,
) -> None:
    _require_executable(executable_path, identity)
    payload = canonical_json_bytes(result.model_dump(mode="json"))
    try:
        context.assert_payload_safe(payload)
        run.write_portable(
            f"{EVIDENCE_ROOT}/result.json",
            payload,
            media_type="application/json",
            role="summary",
        )
    except (OSError, RunError, UnsafeOutputError) as exc:
        raise CapsuleIntegrityError("capsule terminal evidence could not be published") from exc


def _terminal(
    manifest: CapsuleManifestV1,
    *,
    manifest_sha256: str,
    executable_sha256: str,
    phases: list[CapsulePhaseResultV1],
    reason: FailureReason,
    scenario: CapsuleScenarioContractV1 | None,
    correctness: tuple[CorrectnessCoordinateV1, ...] | None,
    benchmark: tuple[BenchmarkCoordinateV1, ...] | None,
) -> CapsuleProtocolResultV1:
    return CapsuleProtocolResultV1(
        capsule_id=manifest.id,
        candidate=manifest.candidate,
        scenario_sha256=manifest.contract.scenario_sha256,
        manifest_sha256=manifest_sha256,
        executable_sha256=executable_sha256,
        status="passed" if reason == "passed" else "failed",
        reason=reason,
        phases=tuple(phases),
        scenario=scenario,
        correctness=correctness,
        benchmark=benchmark,
    )


def run_capsule_protocol(
    run: RunSession,
    manifest: CapsuleManifestV1,
    *,
    manifest_sha256: str,
    executable_path: Path,
    executable_sha256: str,
    cwd: Path,
    environment: Mapping[str, str],
    scratch_root: Path,
    redaction_context: RedactionContext,
) -> CapsuleProtocolResultV1:
    """Execute and evidence exactly ``describe``, ``correctness``, ``benchmark``.

    Ordinary child, parsing, correctness, and completeness failures return a strict
    failed result after publishing truthful phase evidence and ``result.json`` last.
    Executable or evidence drift raises :class:`CapsuleIntegrityError` and never claims
    success.  The function neither allocates/finalizes a run nor acquires a lease or lock.
    """

    manifest_sha256 = _validate_sha256(manifest_sha256, name="manifest_sha256")
    executable_sha256 = _validate_sha256(executable_sha256, name="executable_sha256")
    if not executable_path.is_absolute():
        raise ValueError("executable_path must be absolute")
    if not cwd.is_absolute():
        raise ValueError("cwd must be absolute")
    _validate_scratch_root(scratch_root)
    _preflight_evidence(run)
    context = _combined_redaction(run, redaction_context)
    identity = _hash_initial_executable(executable_path, executable_sha256)
    phases: list[CapsulePhaseResultV1] = []

    describe = _execute_phase(
        run=run,
        context=context,
        executable_path=executable_path,
        identity=identity,
        operation="describe",
        request=_request(
            manifest,
            manifest_sha256=manifest_sha256,
            executable_sha256=executable_sha256,
            operation="describe",
            prior_response_sha256=None,
            scenario=None,
        ),
        cwd=cwd,
        environment=environment,
        scratch_root=scratch_root,
        timeout=manifest.timeouts.describe_seconds,
    )
    phases.append(describe.phase)
    if describe.response is None:
        result = _terminal(
            manifest,
            manifest_sha256=manifest_sha256,
            executable_sha256=executable_sha256,
            phases=phases,
            reason=_phase_failure("describe", cast(str, describe.invalid_kind)),
            scenario=None,
            correctness=None,
            benchmark=None,
        )
        _publish_result(run, context, executable_path, identity, result)
        return result
    describe_response = cast(DescribeResponseV1, describe.response)
    scenario = describe_response.scenario

    correctness = _execute_phase(
        run=run,
        context=context,
        executable_path=executable_path,
        identity=identity,
        operation="correctness",
        request=_request(
            manifest,
            manifest_sha256=manifest_sha256,
            executable_sha256=executable_sha256,
            operation="correctness",
            prior_response_sha256=describe.phase.response_sha256,
            scenario=scenario,
        ),
        cwd=cwd,
        environment=environment,
        scratch_root=scratch_root,
        timeout=manifest.timeouts.correctness_seconds,
    )
    phases.append(correctness.phase)
    if correctness.response is None:
        result = _terminal(
            manifest,
            manifest_sha256=manifest_sha256,
            executable_sha256=executable_sha256,
            phases=phases,
            reason=_phase_failure("correctness", cast(str, correctness.invalid_kind)),
            scenario=scenario,
            correctness=None,
            benchmark=None,
        )
        _publish_result(run, context, executable_path, identity, result)
        return result
    correctness_response = cast(CorrectnessResponseV1, correctness.response)
    expected_coordinates = scenario.coordinates
    observed_correctness = correctness_response.coordinates
    if tuple(value.coordinate for value in observed_correctness) != expected_coordinates:
        reason: FailureReason = "correctness-incomplete"
    elif not all(value.passed for value in observed_correctness):
        reason = "correctness-failed"
    else:
        reason = "passed"
    if reason != "passed":
        result = _terminal(
            manifest,
            manifest_sha256=manifest_sha256,
            executable_sha256=executable_sha256,
            phases=phases,
            reason=reason,
            scenario=scenario,
            correctness=observed_correctness,
            benchmark=None,
        )
        _publish_result(run, context, executable_path, identity, result)
        return result

    benchmark = _execute_phase(
        run=run,
        context=context,
        executable_path=executable_path,
        identity=identity,
        operation="benchmark",
        request=_request(
            manifest,
            manifest_sha256=manifest_sha256,
            executable_sha256=executable_sha256,
            operation="benchmark",
            prior_response_sha256=correctness.phase.response_sha256,
            scenario=scenario,
        ),
        cwd=cwd,
        environment=environment,
        scratch_root=scratch_root,
        timeout=manifest.timeouts.benchmark_seconds,
    )
    phases.append(benchmark.phase)
    if benchmark.response is None:
        result = _terminal(
            manifest,
            manifest_sha256=manifest_sha256,
            executable_sha256=executable_sha256,
            phases=phases,
            reason=_phase_failure("benchmark", cast(str, benchmark.invalid_kind)),
            scenario=scenario,
            correctness=observed_correctness,
            benchmark=None,
        )
        _publish_result(run, context, executable_path, identity, result)
        return result
    benchmark_response = cast(BenchmarkResponseV1, benchmark.response)
    observed_benchmark = benchmark_response.coordinates
    if tuple(value.coordinate for value in observed_benchmark) != expected_coordinates:
        reason = "benchmark-incomplete"
    else:
        reason = "passed"
    result = _terminal(
        manifest,
        manifest_sha256=manifest_sha256,
        executable_sha256=executable_sha256,
        phases=phases,
        reason=reason,
        scenario=scenario,
        correctness=observed_correctness,
        benchmark=observed_benchmark,
    )
    _publish_result(run, context, executable_path, identity, result)
    return result

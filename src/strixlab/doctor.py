"""Machine-doctor policy, report model, orchestration, and safe publication."""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager, suppress
from dataclasses import dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict

from strixlab.config import iter_environment_references, read_manifest
from strixlab.locks import LockAttempt, exclusive_lock
from strixlab.machine import (
    PROFILER_TOOLS,
    MachineProbe,
    MachineSnapshot,
    ToolFact,
)
from strixlab.manifests import DashId, MachineProfileV1, resolve_and_validate_manifest
from strixlab.paths import resolve_home
from strixlab.serialization import canonical_json_bytes

CheckStatus = Literal["pass", "warning", "blocker", "skipped"]
ReportStatus = Literal["ready", "blocked"]
SENSITIVE_NAME_RE = re.compile(
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|KEY|CREDENTIAL|AUTH|COOKIE|SESSION)",
    re.IGNORECASE,
)
ENVIRONMENT_ALLOWLIST = (
    "ROCM_PATH",
    "HIP_PATH",
    "HIP_PLATFORM",
    "HSA_PATH",
    "HSA_OVERRIDE_GFX_VERSION",
    "HSA_ENABLE_SDMA",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "CC",
    "CXX",
    "CMAKE_PREFIX_PATH",
    "CMAKE_BUILD_PARALLEL_LEVEL",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)
MAX_SENSITIVE_NAMES = 128
KNOWN_NONSECRET_SESSION_NAMES = frozenset(
    {"DBUS_SESSION_BUS_ADDRESS", "SESSION_MANAGER", "XDG_SESSION_ID"}
)


class ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class HostFactsV1(ReportModel):
    platform: str
    kernel: str
    distro: str | None
    cpu_model: str | None
    cpu_count: int | None
    memory_total_bytes: int | None
    memory_available_bytes: int | None
    swap_total_bytes: int | None
    swap_free_bytes: int | None
    ac_power: bool | None


class ToolFactV1(ReportModel):
    name: str
    path: str | None
    version: str | None
    outcome: str | None
    truncated: bool


class GPUIdentityV1(ReportModel):
    node: str
    arch: str
    integrated: bool | None
    render_node: str
    pci_bdf: str
    vendor_id: int
    device_id: int
    marketing_name: str | None


class TelemetrySampleV1(ReportModel):
    index: int
    source: str
    busy_pct: float | None
    temperature_c: float | None
    power_w: float | None
    sclk_hz: int | None
    mclk_hz: int | None
    error: str | None


class ProbeIssueV1(ReportModel):
    code: str
    message: str


class CheckV1(ReportModel):
    id: DashId
    status: CheckStatus
    message: str


class LockFactV1(ReportModel):
    status: str
    path: str
    reason: str | None


class RedactionV1(ReportModel):
    sensitive_name_count: int
    sensitive_names: tuple[str, ...]
    sensitive_names_truncated: bool


class DoctorReportV1(ReportModel):
    schema_version: Literal[1] = 1
    kind: Literal["doctor"] = "doctor"
    started_at: datetime
    ended_at: datetime
    machine: MachineProfileV1
    status: ReportStatus
    host: HostFactsV1
    tools: tuple[ToolFactV1, ...]
    gpu: GPUIdentityV1 | None
    gpu_candidates: tuple[GPUIdentityV1, ...]
    rocminfo_arches: tuple[str, ...]
    telemetry_source: str | None
    samples: tuple[TelemetrySampleV1, ...]
    lock: LockFactV1
    checks: tuple[CheckV1, ...]
    probe_errors: tuple[ProbeIssueV1, ...]
    environment: dict[str, str]
    redaction: RedactionV1


class UnsafeDiagnosticError(RuntimeError):
    """Raised when an outgoing artifact could disclose sensitive data."""


class ReportWriteError(RuntimeError):
    """Raised when an atomic report publication fails."""


class SensitiveInterpolationError(ValueError):
    """Raised before resolution when a profile references a secret-like name."""


@dataclass(frozen=True, slots=True)
class RedactionContext:
    """Precomputed secret values shared by every redaction and safety check.

    The secret set is fixed for a run, so it is discovered once and reused for
    designated free-text redaction, whole-payload verification, and terminal-
    sink verification instead of rescanning the environment at each site.
    """

    secrets: tuple[str, ...] = field(repr=False)

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> RedactionContext:
        return cls(_secret_values(environ))

    def redact(self, value: str | None) -> str | None:
        if value is None:
            return None
        for secret in self.secrets:
            value = value.replace(secret, "[REDACTED]")
        return value

    def assert_payload_safe(self, payload: bytes) -> None:
        """Fail closed if a serialized artifact still discloses a secret."""

        for secret in self.secrets:
            if secret.encode() in payload:
                raise UnsafeDiagnosticError("diagnostic output failed secret-safety validation")

    def assert_text_safe(self, value: str) -> None:
        """Fail closed if a terminal line would disclose a sensitive value."""

        self.assert_payload_safe(value.encode())


class DoctorRun(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    report: DoctorReportV1
    path: Path

    @property
    def ready(self) -> bool:
        return self.report.status == "ready"


def _check(check_id: str, status: CheckStatus, message: str) -> CheckV1:
    return CheckV1(id=check_id, status=status, message=message)


def _tool(snapshot: MachineSnapshot, name: str) -> ToolFact | None:
    return next((tool for tool in snapshot.tools if tool.name == name), None)


def evaluate_machine(
    profile: MachineProfileV1,
    snapshot: MachineSnapshot,
    lock: LockAttempt,
) -> tuple[CheckV1, ...]:
    """Evaluate every readiness predicate in stable order."""

    checks: list[CheckV1] = []
    checks.append(
        _check(
            "platform",
            "pass" if snapshot.host.platform == "linux" else "blocker",
            "Linux host detected"
            if snapshot.host.platform == "linux"
            else "Linux is required for AMD GPU observation",
        )
    )
    checks.append(
        _check(
            "exclusive-lock",
            "pass" if lock.acquired else "blocker",
            "exclusive GPU lock acquired"
            if lock.acquired
            else "exclusive GPU lock could not be safely acquired",
        )
    )

    for name in ("rocminfo", "hipcc", "cmake", "ninja", "git"):
        tool = _tool(snapshot, name)
        ok = tool is not None and tool.available
        if ok and name in {"hipcc", "cmake", "ninja", "git"}:
            assert tool is not None
            ok = tool.outcome == "ok"
        if name == "rocminfo" and lock.acquired:
            ok = ok and bool(snapshot.rocminfo_arches)
        checks.append(
            _check(
                f"tool-{name}",
                "pass" if ok else "blocker",
                f"{name} is usable" if ok else f"{name} is missing or unusable",
            )
        )

    for name in ("clang", *PROFILER_TOOLS):
        tool = _tool(snapshot, name)
        optional_available = tool is not None and tool.available
        checks.append(
            _check(
                f"tool-{name.replace('_', '-')}",
                "pass" if optional_available else "warning",
                f"optional tool {name} is available"
                if optional_available
                else f"optional tool {name} is unavailable",
            )
        )

    if not lock.acquired:
        for check_id in ("gpu-identity", "gpu-arch", "gpu-integrated", "gpu-busy"):
            checks.append(
                _check(
                    check_id,
                    "skipped",
                    "GPU observation skipped without lock ownership",
                )
            )
    else:
        arch_candidates = [
            value for value in snapshot.gpu_candidates if value.arch == profile.expect.gpu_arch
        ]
        matching = [
            value for value in arch_candidates if value.integrated is profile.expect.integrated_gpu
        ]
        identity_ok = len(matching) == 1 and snapshot.gpu is not None
        checks.append(
            _check(
                "gpu-identity",
                "pass" if identity_ok else "blocker",
                "exactly one profile-matching GPU identified"
                if identity_ok
                else f"expected one profile-matching GPU, observed {len(matching)}",
            )
        )
        arch_ok = bool(arch_candidates)
        observed_arches = sorted({value.arch for value in snapshot.gpu_candidates})
        checks.append(
            _check(
                "gpu-arch",
                "pass" if arch_ok else "blocker",
                f"GPU architecture is {profile.expect.gpu_arch}"
                if arch_ok
                else f"expected {profile.expect.gpu_arch}; observed {observed_arches or 'none'}",
            )
        )
        classification_known = bool(arch_candidates) and all(
            value.integrated is not None for value in arch_candidates
        )
        integrated_ok = len(matching) == 1 and classification_known
        checks.append(
            _check(
                "gpu-integrated",
                "pass" if integrated_ok else "blocker",
                f"integrated classification is {profile.expect.integrated_gpu}"
                if integrated_ok
                else "integrated/discrete classification is unknown or mismatched",
            )
        )
        busy_values = [sample.busy_pct for sample in snapshot.samples]
        busy_complete = len(busy_values) == 3 and all(value is not None for value in busy_values)
        maximum_busy = max(cast(list[float], busy_values)) if busy_complete else None
        busy_ok = (
            busy_complete
            and maximum_busy is not None
            and maximum_busy <= profile.validity.max_background_gpu_busy_pct
        )
        checks.append(
            _check(
                "gpu-busy",
                "pass" if busy_ok else "blocker",
                (
                    f"maximum GPU busy {maximum_busy:g}% is within "
                    f"{profile.validity.max_background_gpu_busy_pct:g}%"
                )
                if busy_ok and maximum_busy is not None
                else "three valid idle GPU samples are required",
            )
        )

    total_required = profile.expect.memory_gib_min * 2**30
    total = snapshot.host.memory_total_bytes
    total_ok = total is not None and total >= total_required
    checks.append(
        _check(
            "memory-total",
            "pass" if total_ok else "blocker",
            "total memory meets the profile minimum"
            if total_ok
            else "total memory is unavailable or below the profile minimum",
        )
    )
    available_required = profile.validity.min_available_memory_gib * 2**30
    memory_available = snapshot.host.memory_available_bytes
    available_ok = memory_available is not None and memory_available >= available_required
    checks.append(
        _check(
            "memory-available",
            "pass" if available_ok else "blocker",
            "available memory meets the validity minimum"
            if available_ok
            else "available memory is unavailable or below the validity minimum",
        )
    )
    ac_ok = not profile.validity.require_ac_power or snapshot.host.ac_power is True
    checks.append(
        _check(
            "ac-power",
            "pass" if ac_ok else "blocker",
            "AC-power requirement is satisfied"
            if ac_ok
            else "AC power is required but offline or unknown",
        )
    )

    temperatures = [
        sample.temperature_c for sample in snapshot.samples if sample.temperature_c is not None
    ]
    if not lock.acquired:
        checks.append(_check("gpu-temperature", "skipped", "temperature probe skipped"))
    elif not temperatures:
        checks.append(_check("gpu-temperature", "warning", "GPU temperature is unavailable"))
    else:
        maximum_temperature = max(temperatures)
        warn = maximum_temperature >= profile.validity.temperature_warn_c
        incomplete = len(temperatures) != 3
        status: CheckStatus = "warning" if warn or incomplete else "pass"
        checks.append(
            _check(
                "gpu-temperature",
                status,
                f"maximum GPU temperature was {maximum_temperature:g} C"
                + (" with incomplete telemetry" if incomplete else ""),
            )
        )

    mode = profile.telemetry.amd_smi
    if not lock.acquired:
        checks.append(_check("telemetry-source", "skipped", "telemetry probe skipped"))
    elif mode == "required":
        ok = (
            snapshot.telemetry_source == "amd-smi"
            and len(snapshot.samples) == 3
            and all(sample.busy_pct is not None for sample in snapshot.samples)
        )
        checks.append(
            _check(
                "telemetry-source",
                "pass" if ok else "blocker",
                "required amd-smi telemetry is usable"
                if ok
                else "amd-smi telemetry is required but unusable",
            )
        )
    elif mode == "auto":
        if snapshot.telemetry_source is None:
            checks.append(_check("telemetry-source", "blocker", "no telemetry source is usable"))
        elif snapshot.telemetry_source == "amd-smi":
            checks.append(_check("telemetry-source", "pass", "amd-smi telemetry selected"))
        else:
            checks.append(
                _check(
                    "telemetry-source",
                    "warning",
                    f"telemetry fell back to {snapshot.telemetry_source}",
                )
            )
    else:
        ok = snapshot.telemetry_source == "sysfs"
        checks.append(
            _check(
                "telemetry-source",
                "pass" if ok else "blocker",
                "SMI telemetry disabled; sysfs selected" if ok else "sysfs telemetry is unusable",
            )
        )

    issue_codes = {issue.code for issue in snapshot.issues}
    if "rocminfo-correlation" in issue_codes:
        checks.append(
            _check(
                "rocminfo-correlation",
                "warning" if snapshot.gpu is not None else "blocker",
                "rocminfo identity disagrees with KFD/DRM identity",
            )
        )
    return tuple(checks)


def _is_sensitive_name(name: str) -> bool:
    return name not in KNOWN_NONSECRET_SESSION_NAMES and SENSITIVE_NAME_RE.search(name) is not None


def _sensitive_names(environ: Mapping[str, str]) -> list[str]:
    return sorted(name for name, value in environ.items() if value and _is_sensitive_name(name))


def capture_environment(environ: Mapping[str, str]) -> tuple[dict[str, str], RedactionV1]:
    captured: dict[str, str] = {}
    for name in ENVIRONMENT_ALLOWLIST:
        if name not in environ:
            continue
        captured[name] = "[REDACTED]" if _is_sensitive_name(name) else environ[name]
    names = _sensitive_names(environ)
    return captured, RedactionV1(
        sensitive_name_count=len(names),
        sensitive_names=tuple(names[:MAX_SENSITIVE_NAMES]),
        sensitive_names_truncated=len(names) > MAX_SENSITIVE_NAMES,
    )


def reject_sensitive_interpolations(value: Any) -> None:
    if isinstance(value, str):
        if any(_is_sensitive_name(name) for name in iter_environment_references(value)):
            raise SensitiveInterpolationError(
                "machine profile may not interpolate a sensitive environment variable"
            )
    elif isinstance(value, Mapping):
        for child in value.values():
            reject_sensitive_interpolations(child)
    elif isinstance(value, list):
        for child in value:
            reject_sensitive_interpolations(child)


def _secret_values(environ: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {value for name, value in environ.items() if value and _is_sensitive_name(name)},
            key=len,
            reverse=True,
        )
    )


def _to_model[ModelT: ReportModel](model_cls: type[ModelT], value: Any) -> ModelT:
    """Convert an observation dataclass to its report model by field reflection."""

    return model_cls(**{item.name: getattr(value, item.name) for item in fields(value)})


def _safe_snapshot(snapshot: MachineSnapshot, context: RedactionContext) -> MachineSnapshot:
    """Redact every designated free-text field before the snapshot is serialized.

    Coverage spans all subprocess- and probe-derived free text: tool versions,
    telemetry errors, probe messages, and the rocminfo-supplied GPU marketing
    name. Enumerated values (identities, paths, clocks) are left intact; the
    final whole-payload scan remains the fail-closed backstop.
    """

    gpu = snapshot.gpu
    if gpu is not None and gpu.marketing_name is not None:
        gpu = replace(gpu, marketing_name=context.redact(gpu.marketing_name))
    return replace(
        snapshot,
        tools=tuple(replace(tool, version=context.redact(tool.version)) for tool in snapshot.tools),
        samples=tuple(
            replace(sample, error=context.redact(sample.error)) for sample in snapshot.samples
        ),
        issues=tuple(
            replace(issue, message=context.redact(issue.message) or "probe failed")
            for issue in snapshot.issues
        ),
        gpu=gpu,
    )


def build_report(
    profile: MachineProfileV1,
    snapshot: MachineSnapshot,
    lock: LockAttempt,
    environ: Mapping[str, str],
    started_at: datetime,
    ended_at: datetime,
) -> DoctorReportV1:
    """Build a report using redaction state derived from the supplied environment."""

    return _build_report(
        profile,
        snapshot,
        lock,
        environ,
        started_at,
        ended_at,
        context=RedactionContext.from_environ(environ),
    )


def _build_report(
    profile: MachineProfileV1,
    snapshot: MachineSnapshot,
    lock: LockAttempt,
    environ: Mapping[str, str],
    started_at: datetime,
    ended_at: datetime,
    *,
    context: RedactionContext,
) -> DoctorReportV1:
    safe_snapshot = _safe_snapshot(snapshot, context)
    checks = tuple(
        CheckV1(id=value.id, status=value.status, message=context.redact(value.message) or "")
        for value in evaluate_machine(profile, safe_snapshot, lock)
    )
    environment, redaction = capture_environment(environ)
    return DoctorReportV1(
        started_at=started_at,
        ended_at=ended_at,
        machine=profile,
        status="blocked" if any(value.status == "blocker" for value in checks) else "ready",
        host=_to_model(HostFactsV1, safe_snapshot.host),
        tools=tuple(_to_model(ToolFactV1, value) for value in safe_snapshot.tools),
        gpu=_to_model(GPUIdentityV1, safe_snapshot.gpu) if safe_snapshot.gpu else None,
        gpu_candidates=tuple(
            _to_model(GPUIdentityV1, value) for value in safe_snapshot.gpu_candidates
        ),
        rocminfo_arches=safe_snapshot.rocminfo_arches,
        telemetry_source=safe_snapshot.telemetry_source,
        samples=tuple(_to_model(TelemetrySampleV1, value) for value in safe_snapshot.samples),
        lock=LockFactV1(
            status=lock.status.value,
            path=str(lock.path),
            reason=context.redact(lock.reason),
        ),
        checks=checks,
        probe_errors=tuple(_to_model(ProbeIssueV1, value) for value in safe_snapshot.issues),
        environment=environment,
        redaction=redaction,
    )


def canonical_report_bytes(report: DoctorReportV1) -> bytes:
    return canonical_json_bytes(report.model_dump(mode="json"))


def _prepare_parent(path: Path) -> None:
    missing: list[Path] = []
    current = path.parent
    while not current.exists():
        missing.append(current)
        if current == current.parent:
            break
        current = current.parent
    for directory in reversed(missing):
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            continue


def _write_temp(path: Path, payload: bytes) -> Path:
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw_path)
    try:
        os.fchmod(descriptor, 0o600)
        stream = os.fdopen(descriptor, "wb", closefd=True)
        descriptor = -1
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if descriptor >= 0:
            with suppress(OSError):
                os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_authoritative(path: Path, payload: bytes) -> Path:
    temporary: Path | None = None
    try:
        _prepare_parent(path)
        temporary = _write_temp(path, payload)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except OSError as exc:
        raise ReportWriteError("unable to publish authoritative doctor report") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def publish_diagnostic(
    path: Path,
    payload: bytes,
    *,
    path_validator: Callable[[Path], None] | None = None,
) -> Path:
    temporary: Path | None = None
    unsafe_candidate = False
    try:
        _prepare_parent(path)
        temporary = _write_temp(path, payload)
        for _ in range(16):
            candidate = path.with_name(f"{path.stem}.diagnostic.{uuid.uuid4().hex}{path.suffix}")
            if path_validator is not None:
                try:
                    path_validator(candidate)
                except UnsafeDiagnosticError:
                    unsafe_candidate = True
                    continue
            try:
                os.link(temporary, candidate)
            except FileExistsError:
                continue
            _fsync_directory(path.parent)
            return candidate
        if unsafe_candidate:
            raise UnsafeDiagnosticError("no secret-safe diagnostic report path could be generated")
        raise ReportWriteError("unable to reserve a unique diagnostic report path")
    except OSError as exc:
        raise ReportWriteError("unable to publish diagnostic doctor report") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_doctor(
    machine_path: Path,
    *,
    home: str | os.PathLike[str] | None = None,
    output: Path | None = None,
    environ: Mapping[str, str] | None = None,
    probe: MachineProbe | None = None,
    lock_factory: Callable[[Path], AbstractContextManager[LockAttempt]] = exclusive_lock,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> DoctorRun:
    """Validate, observe, evaluate, and atomically publish one doctor report."""

    frozen_environ = dict(os.environ if environ is None else environ)
    context = RedactionContext.from_environ(frozen_environ)
    raw = read_manifest(machine_path)
    reject_sensitive_interpolations(raw)
    model = resolve_and_validate_manifest("machine", raw, frozen_environ)
    profile = cast(MachineProfileV1, model)
    destination = output or (
        resolve_home(home, environ=frozen_environ) / "doctor" / profile.id / "doctor.json"
    )
    active_probe = probe or MachineProbe(environ=frozen_environ)
    started_at = now()
    snapshot = active_probe.collect_prelock()
    lock_path = Path(profile.exclusive_lock.path)
    with lock_factory(lock_path) as lock:
        if lock.acquired:
            snapshot = active_probe.collect_postlock(snapshot, profile)
        else:
            snapshot = active_probe.mark_gpu_skipped(
                snapshot, "GPU-facing probes require exclusive lock ownership"
            )
        report = _build_report(
            profile, snapshot, lock, frozen_environ, started_at, now(), context=context
        )
        payload = canonical_report_bytes(report)
        context.assert_payload_safe(payload)
        context.assert_text_safe(str(destination))
        actual = (
            publish_authoritative(destination, payload)
            if lock.acquired
            else publish_diagnostic(
                destination,
                payload,
                path_validator=lambda candidate: context.assert_text_safe(str(candidate)),
            )
        )
    return DoctorRun(report=report, path=actual)

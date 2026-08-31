"""JUDGE-001: the first offline comparison judge over two finalized suite runs.

The judge authenticates two distinct, finalized, successful smoke-suite runs, pairs their
matched throughput samples conservatively, and finalizes one immutable comparison run
carrying canonical JSON and Markdown reports. It never touches hardware, reruns an
adapter, or mutates either arm. Its required no-op result is ``inconclusive`` — a
statistical ``regression``/``mixed``/``inconclusive`` report is still a successfully
executed judge run. A comparison bundle authenticates its source run IDs, record digests,
result digests, and resolved-manifest digest, but it does not copy either arm's evidence
tree: it is a portable derived report, not a standalone proof of its arms. Export the two
source-run bundles as well when independent offline verification is required.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from strixlab.evidence import (
    MAX_AGGREGATE_BYTES,
    MAX_MEMBER_BYTES,
    MAX_PORTABLE_ENTRIES,
    MAX_TOTAL_FILES,
    Clock,
    RunError,
    RunOutcome,
    RunSession,
    TokenFactory,
    begin_run,
    inspect_run,
    run_relative,
    validate_portable_payload,
)
from strixlab.records import RecordError
from strixlab.secret_policy import (
    RedactionContext,
    SensitiveInterpolationError,
    UnsafeOutputError,
    reject_sensitive_interpolations,
)
from strixlab.serialization import canonical_json_bytes, canonical_yaml_bytes
from strixlab.source_identity import length_frame
from strixlab.suites import (
    BuildIdStr,
    DashIdStr,
    FinalizedSuiteSnapshot,
    NonNegativeFloat,
    PositiveRate,
    RecordSha,
    Sha256Hex,
    SuiteError,
    load_finalized_suite_snapshot,
)

POLICY_ID: Final = "paired-log-bootstrap-v1"
MIN_PAIRED_SAMPLES = 5
BOOTSTRAP_REPLICATES = 4096
BOOTSTRAP_DOMAIN = "strixlab.judge.bootstrap.v1"
_PERCENTILE_LOW = 0.025
_PERCENTILE_HIGH = 0.975
_TOLERANCE = 1e-12

_REPORT_JSON_PATH = "comparison/report.json"
_REPORT_MD_PATH = "comparison/report.md"

CaseVerdict = Literal["improvement", "regression", "inconclusive"]
OverallVerdict = Literal["improvement", "regression", "inconclusive", "mixed"]

# --- Failure taxonomy ---------------------------------------------------------


class JudgeError(RuntimeError):
    """A comparison judge operation failed."""


class JudgeLoadError(JudgeError):
    """An arm could not be authenticated; no comparison run is allocated."""


class JudgeEquivalenceError(JudgeError):
    """The two arms are not a comparable pair; no comparison run is allocated."""


class JudgeStatisticsError(JudgeError):
    """Paired statistics or the derived report could not be computed; no run is allocated."""


class JudgeExecutionError(JudgeError):
    """A comparison run was allocated but failed before publishing its report.

    Carries the finalized run id and its immutable record when finalization produced one,
    so a caller can surface them even though no truthful report could be published.
    """

    def __init__(self, message: str, *, run_id: str, record: Path | None) -> None:
        super().__init__(message)
        self.run_id = run_id
        self.record = record


# --- Deterministic verdict projections ----------------------------------------


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=_TOLERANCE, abs_tol=_TOLERANCE)


def case_verdict(
    log_ci_low: float, log_ci_high: float, mean_log_delta: float, baseline_noise_log: float
) -> CaseVerdict:
    """The exact per-case verdict projection of a conservative paired comparison."""

    if log_ci_low <= 0.0 <= log_ci_high:
        return "inconclusive"
    if abs(mean_log_delta) <= baseline_noise_log:
        return "inconclusive"
    if log_ci_low > 0.0:
        return "improvement"
    return "regression"


def overall_verdict(verdicts: tuple[CaseVerdict, ...]) -> OverallVerdict:
    """The exact overall verdict projection of a nonempty set of per-case verdicts."""

    unique = set(verdicts)
    if unique == {"inconclusive"}:
        return "inconclusive"
    if unique == {"improvement"}:
        return "improvement"
    if unique == {"regression"}:
        return "regression"
    return "mixed"


# --- Strict comparison models -------------------------------------------------


class _JudgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class ComparisonRequestV1(_JudgeModel):
    """The authenticated request that scopes and names one comparison."""

    schema_version: Literal[1] = 1
    policy_id: Literal["paired-log-bootstrap-v1"] = POLICY_ID
    baseline_run_id: str = Field(min_length=1)
    baseline_record_sha256: RecordSha
    candidate_run_id: str = Field(min_length=1)
    candidate_record_sha256: RecordSha

    @model_validator(mode="after")
    def _distinct(self) -> ComparisonRequestV1:
        if self.baseline_run_id == self.candidate_run_id:
            raise ValueError("a comparison request must name two distinct runs")
        return self


class ComparisonArmV1(_JudgeModel):
    """One authenticated comparison arm bound to its finalized run and build."""

    label: Literal["baseline", "candidate"]
    run_id: str = Field(min_length=1)
    record_sha256: RecordSha
    suite_result_sha256: Sha256Hex
    build_id: BuildIdStr
    build_record_sha256: RecordSha


class CaseComparisonV1(_JudgeModel):
    """One suite performance case's conservative matched-sample comparison."""

    case_id: DashIdStr
    metric: Literal["samples_ts"] = "samples_ts"
    direction: Literal["higher-is-better"] = "higher-is-better"
    pair_count: int = Field(ge=MIN_PAIRED_SAMPLES)
    baseline_mean: PositiveRate
    candidate_mean: PositiveRate
    mean_log_delta: float
    speed_ratio: PositiveRate
    delta_percent: float
    log_ci_low: float
    log_ci_high: float
    percent_ci_low: float
    percent_ci_high: float
    baseline_noise_log: NonNegativeFloat
    baseline_noise_percent: NonNegativeFloat
    verdict: CaseVerdict

    @model_validator(mode="after")
    def _check(self) -> CaseComparisonV1:
        if self.log_ci_low > self.log_ci_high:
            raise ValueError("log confidence interval endpoints are not ordered")
        if not _close(self.speed_ratio, math.exp(self.mean_log_delta)):
            raise ValueError("speed_ratio does not equal exp(mean_log_delta)")
        if not _close(self.delta_percent, 100.0 * math.expm1(self.mean_log_delta)):
            raise ValueError("delta_percent does not equal its log form")
        if not _close(self.percent_ci_low, 100.0 * math.expm1(self.log_ci_low)):
            raise ValueError("percent_ci_low does not equal its log form")
        if not _close(self.percent_ci_high, 100.0 * math.expm1(self.log_ci_high)):
            raise ValueError("percent_ci_high does not equal its log form")
        if not _close(self.baseline_noise_percent, 100.0 * math.expm1(self.baseline_noise_log)):
            raise ValueError("baseline_noise_percent does not equal its log form")
        expected = case_verdict(
            self.log_ci_low, self.log_ci_high, self.mean_log_delta, self.baseline_noise_log
        )
        if self.verdict != expected:
            raise ValueError("case verdict is not the exact projection of its statistics")
        return self


class ComparisonReportV1(_JudgeModel):
    """The complete validated comparison report, canonicalized to JSON on disk."""

    schema_version: Literal[1] = 1
    policy_id: Literal["paired-log-bootstrap-v1"] = POLICY_ID
    suite_id: DashIdStr
    machine_id: DashIdStr
    model_id: DashIdStr
    resolved_manifest_sha256: Sha256Hex
    baseline: ComparisonArmV1
    candidate: ComparisonArmV1
    overall_verdict: OverallVerdict
    cases: tuple[CaseComparisonV1, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> ComparisonReportV1:
        if self.baseline.label != "baseline" or self.candidate.label != "candidate":
            raise ValueError("comparison arms are mislabeled")
        if self.baseline.run_id == self.candidate.run_id:
            raise ValueError("comparison arms must be two distinct runs")
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("comparison cases must be unique")
        expected = overall_verdict(tuple(case.verdict for case in self.cases))
        if self.overall_verdict != expected:
            raise ValueError("overall verdict is not the exact projection of the cases")
        return self


@dataclass(frozen=True, slots=True)
class ComparisonRunResult:
    """The finalized comparison run, its outcome, record, and validated report."""

    run_id: str
    outcome: RunOutcome
    record: Path
    report: ComparisonReportV1


@dataclass(frozen=True, slots=True)
class JudgeHooks:
    """Narrow, explicit test seams; every field defaults to production behavior."""

    clock: Clock | None = None
    token_factory: TokenFactory | None = None
    publish: PublishHook | None = None


# --- Deterministic paired statistics ------------------------------------------


def _u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def _bootstrap_index(
    baseline_record_sha256: str,
    candidate_record_sha256: str,
    case_id: str,
    replicate: int,
    draw: int,
    n: int,
) -> int:
    """Select one zero-based paired index for one bootstrap draw, deterministically."""

    framed = length_frame(
        BOOTSTRAP_DOMAIN,
        (
            ("baseline_record_sha256", baseline_record_sha256.encode("utf-8")),
            ("candidate_record_sha256", candidate_record_sha256.encode("utf-8")),
            ("case_id", case_id.encode("utf-8")),
            ("replicate", _u64(replicate)),
            ("draw", _u64(draw)),
        ),
    )
    digest = hashlib.sha256(framed).digest()
    return int.from_bytes(digest[:8], "big") % n


def _percentile(sorted_values: list[float], p: float) -> float:
    """R-7 linear-interpolation percentile of a nonempty ascending sequence."""

    count = len(sorted_values)
    h = (count - 1) * p
    lo = math.floor(h)
    hi = math.ceil(h)
    low_value = sorted_values[lo]
    high_value = sorted_values[hi]
    return math.fsum((low_value, (h - lo) * (high_value - low_value)))


def _median(sorted_values: list[float]) -> float:
    count = len(sorted_values)
    middle = count // 2
    if count % 2 == 1:
        return sorted_values[middle]
    return math.fsum((sorted_values[middle - 1], sorted_values[middle])) / 2


def _bootstrap_interval(
    deltas: list[float],
    *,
    baseline_record_sha256: str,
    candidate_record_sha256: str,
    case_id: str,
) -> tuple[float, float]:
    """The 95% percentile interval of the ``paired-log-bootstrap-v1`` replicate means."""

    n = len(deltas)
    means: list[float] = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        selected = [
            deltas[
                _bootstrap_index(
                    baseline_record_sha256, candidate_record_sha256, case_id, replicate, draw, n
                )
            ]
            for draw in range(n)
        ]
        means.append(math.fsum(selected) / n)
    means.sort()
    return _percentile(means, _PERCENTILE_LOW), _percentile(means, _PERCENTILE_HIGH)


def compare_case_samples(
    case_id: str,
    baseline_samples: tuple[float, ...],
    candidate_samples: tuple[float, ...],
    *,
    baseline_record_sha256: str,
    candidate_record_sha256: str,
) -> CaseComparisonV1:
    """Compute one case's deterministic conservative comparison from matched samples.

    Pairs positionally in manifest order, computes the paired log-delta mean, the
    ``paired-log-bootstrap-v1`` 95% interval, and baseline noise, and projects the
    conservative verdict. Every arithmetic domain/overflow failure, a nonpositive or
    non-finite input or derived value, and any report-model validation failure caused by
    the derived statistics is translated to :class:`JudgeStatisticsError`.
    """

    try:
        n = len(baseline_samples)
        if n != len(candidate_samples):
            raise JudgeStatisticsError("paired sample counts differ")
        if n < MIN_PAIRED_SAMPLES:
            raise JudgeStatisticsError("too few paired samples for a comparison")
        for baseline_value, candidate_value in zip(
            baseline_samples, candidate_samples, strict=True
        ):
            if not (
                math.isfinite(baseline_value)
                and baseline_value > 0.0
                and math.isfinite(candidate_value)
                and candidate_value > 0.0
            ):
                raise JudgeStatisticsError("paired samples must be positive and finite")
        log_baseline = [math.log(value) for value in baseline_samples]
        log_candidate = [math.log(value) for value in candidate_samples]
        deltas = [c - b for c, b in zip(log_candidate, log_baseline, strict=True)]
        mean_delta = math.fsum(deltas) / n
        baseline_mean = math.fsum(baseline_samples) / n
        candidate_mean = math.fsum(candidate_samples) / n
        speed_ratio = math.exp(mean_delta)
        if not (math.isfinite(speed_ratio) and speed_ratio > 0.0):
            raise JudgeStatisticsError("derived speed ratio is nonpositive or non-finite")
        delta_percent = 100.0 * math.expm1(mean_delta)
        log_ci_low, log_ci_high = _bootstrap_interval(
            deltas,
            baseline_record_sha256=baseline_record_sha256,
            candidate_record_sha256=candidate_record_sha256,
            case_id=case_id,
        )
        median_log = _median(sorted(log_baseline))
        noise_log = 1.4826 * _median(sorted(abs(value - median_log) for value in log_baseline))
        return CaseComparisonV1(
            case_id=case_id,
            pair_count=n,
            baseline_mean=baseline_mean,
            candidate_mean=candidate_mean,
            mean_log_delta=mean_delta,
            speed_ratio=speed_ratio,
            delta_percent=delta_percent,
            log_ci_low=log_ci_low,
            log_ci_high=log_ci_high,
            percent_ci_low=100.0 * math.expm1(log_ci_low),
            percent_ci_high=100.0 * math.expm1(log_ci_high),
            baseline_noise_log=noise_log,
            baseline_noise_percent=100.0 * math.expm1(noise_log),
            verdict=case_verdict(log_ci_low, log_ci_high, mean_delta, noise_log),
        )
    except (ValueError, OverflowError, ValidationError) as exc:
        # An explicit ``JudgeStatisticsError`` raised above is a RuntimeError, not one of
        # these arithmetic/validation types, so it propagates directly; only genuine
        # domain/overflow/model failures are translated here.
        raise JudgeStatisticsError("paired statistics could not be computed") from exc


# --- Arm equivalence ----------------------------------------------------------


def _check_equivalence(baseline: FinalizedSuiteSnapshot, candidate: FinalizedSuiteSnapshot) -> None:
    """Reject two arms that are not a comparable pair before any run is allocated."""

    if baseline.run_id == candidate.run_id:
        raise JudgeEquivalenceError("cannot compare a run to itself")
    if baseline.resolved_manifest_bytes != candidate.resolved_manifest_bytes:
        raise JudgeEquivalenceError("arms do not share identical resolved manifest bytes")
    if (
        baseline.result.suite_id != candidate.result.suite_id
        or baseline.result.machine_id != candidate.result.machine_id
        or baseline.result.model_id != candidate.result.model_id
    ):
        raise JudgeEquivalenceError("arms do not share suite, machine, and model identity")
    if baseline.model_input_sha256 != candidate.model_input_sha256:
        raise JudgeEquivalenceError("arms do not share the model input snapshot")
    if baseline.machine_input_sha256 != candidate.machine_input_sha256:
        raise JudgeEquivalenceError("arms do not share the machine input snapshot")
    if baseline.case_order != candidate.case_order:
        raise JudgeEquivalenceError("arms do not share the measurement case order")
    if (
        baseline.measurement_windows != candidate.measurement_windows
        or baseline.repetitions_per_window != candidate.repetitions_per_window
    ):
        raise JudgeEquivalenceError("arms do not share measurement window and repetition counts")
    for case_id in baseline.case_order:
        baseline_count = len(baseline.case_samples[case_id])
        candidate_count = len(candidate.case_samples[case_id])
        if baseline_count != candidate_count:
            raise JudgeEquivalenceError("arms do not share matched sample counts per case")
        if baseline_count < MIN_PAIRED_SAMPLES:
            raise JudgeEquivalenceError("a case carries fewer than the minimum paired samples")


# --- Report and Markdown ------------------------------------------------------


def _build_report(
    request: ComparisonRequestV1,
    baseline: FinalizedSuiteSnapshot,
    candidate: FinalizedSuiteSnapshot,
    cases: tuple[CaseComparisonV1, ...],
) -> ComparisonReportV1:
    """Instantiate and cross-bind the report to the request that created it."""

    baseline_arm = ComparisonArmV1(
        label="baseline",
        run_id=baseline.run_id,
        record_sha256=baseline.record_sha256,
        suite_result_sha256=baseline.result_sha256,
        build_id=baseline.build_id,
        build_record_sha256=baseline.build_record_sha256,
    )
    candidate_arm = ComparisonArmV1(
        label="candidate",
        run_id=candidate.run_id,
        record_sha256=candidate.record_sha256,
        suite_result_sha256=candidate.result_sha256,
        build_id=candidate.build_id,
        build_record_sha256=candidate.build_record_sha256,
    )
    report = ComparisonReportV1(
        suite_id=baseline.result.suite_id,
        machine_id=baseline.result.machine_id,
        model_id=baseline.result.model_id,
        resolved_manifest_sha256=baseline.resolved_manifest_sha256,
        baseline=baseline_arm,
        candidate=candidate_arm,
        overall_verdict=overall_verdict(tuple(case.verdict for case in cases)),
        cases=cases,
    )
    if (
        report.policy_id != request.policy_id
        or report.baseline.run_id != request.baseline_run_id
        or report.baseline.record_sha256 != request.baseline_record_sha256
        or report.candidate.run_id != request.candidate_run_id
        or report.candidate.record_sha256 != request.candidate_record_sha256
    ):
        raise ValueError("report arms or policy do not match the comparison request")
    return report


_SCOPE_WARNING = (
    "Offline, matched-sample, conservative comparison. Higher is better. No universal "
    "score. This report is not independently verifiable without both source-run bundles."
)


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def render_report_markdown(report: ComparisonReportV1) -> bytes:
    """Render a validated report to canonical Markdown ending in exactly one newline.

    A pure projection of the validated report: identities and digests, policy, the scope
    warning, the overall verdict, and one stable table row per case. No environment-derived
    free text is emitted.
    """

    lines: list[str] = []
    lines.append("# StrixLab comparison report")
    lines.append("")
    lines.append(f"- policy: `{report.policy_id}`")
    lines.append(f"- suite: `{report.suite_id}`")
    lines.append(f"- machine: `{report.machine_id}`")
    lines.append(f"- model: `{report.model_id}`")
    lines.append(f"- resolved-manifest-sha256: `{report.resolved_manifest_sha256}`")
    lines.append(f"- overall verdict: **{report.overall_verdict}**")
    lines.append("")
    for arm in (report.baseline, report.candidate):
        lines.append(f"## {arm.label}")
        lines.append(f"- run: `{arm.run_id}`")
        lines.append(f"- record: `{arm.record_sha256}`")
        lines.append(f"- suite-result: `{arm.suite_result_sha256}`")
        lines.append(f"- build: `{arm.build_id}`")
        lines.append(f"- build-record: `{arm.build_record_sha256}`")
        lines.append("")
    lines.append(f"> {_SCOPE_WARNING}")
    lines.append("")
    lines.append(
        "| case | metric | baseline mean | candidate mean | delta % | "
        "log CI | percent CI | noise % | pairs | verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for case in report.cases:
        log_ci = f"[{_format_float(case.log_ci_low)}, {_format_float(case.log_ci_high)}]"
        percent_ci = (
            f"[{_format_float(case.percent_ci_low)}, {_format_float(case.percent_ci_high)}]"
        )
        lines.append(
            f"| {case.case_id} | {case.metric} | {_format_float(case.baseline_mean)} | "
            f"{_format_float(case.candidate_mean)} | {_format_float(case.delta_percent)} | "
            f"{log_ci} | {percent_ci} | {_format_float(case.baseline_noise_percent)} | "
            f"{case.pair_count} | {case.verdict} |"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


# --- Preflight, immutable run, and publication --------------------------------


@dataclass(frozen=True, slots=True)
class _PortableOutput:
    logical_path: str
    content: bytes
    media_type: str
    role: str


# The post-allocation publication seam: the only injection a test needs to exercise a
# comparison run that fails after allocation. Production uses ``_default_publish``.
PublishHook = Callable[[RunSession, tuple[_PortableOutput, ...]], None]


def _preflight_outputs(outputs: tuple[_PortableOutput, ...], *, environ: Mapping[str, str]) -> None:
    """Preflight the exact portable outputs against the closed portable policy.

    Enforces the member, aggregate, entry, path, file-count, and single-media-per-blob
    rules over the deduplicated blob set before a run is allocated, so a comparison run's
    capacity is bounded and predictable. No arm payloads are copied.
    """

    context = RedactionContext.from_environ(environ)
    if len(outputs) > MAX_PORTABLE_ENTRIES:
        raise JudgeError("comparison outputs exceed the per-run entry limit")
    seen_paths: set[str] = set()
    blob_media: dict[str, str] = {}
    blob_sizes: dict[str, int] = {}
    for output in outputs:
        run_relative(output.logical_path)
        if output.logical_path in seen_paths:
            raise JudgeError(f"duplicate comparison output path: {output.logical_path}")
        seen_paths.add(output.logical_path)
        validate_portable_payload(output.content, output.media_type)
        try:
            context.assert_payload_safe(output.content)
            reject_sensitive_interpolations(output.content.decode("utf-8"))
        except (UnicodeDecodeError, SensitiveInterpolationError, UnsafeOutputError) as exc:
            raise JudgeError("comparison output failed secret-policy preflight") from exc
        if len(output.content) > MAX_MEMBER_BYTES:
            raise JudgeError("a comparison output exceeds the per-member limit")
        blob_sha = hashlib.sha256(output.content).hexdigest()
        if blob_media.setdefault(blob_sha, output.media_type) != output.media_type:
            raise JudgeError("a comparison blob is shared under conflicting media types")
        blob_sizes[blob_sha] = len(output.content)
    if sum(blob_sizes.values()) > MAX_AGGREGATE_BYTES:
        raise JudgeError("comparison outputs exceed the aggregate payload limit")
    if len(blob_sizes) + len(outputs) > MAX_TOTAL_FILES:
        raise JudgeError("comparison outputs exceed the total-file limit")


def _default_publish(run: RunSession, outputs: tuple[_PortableOutput, ...]) -> None:
    for output in outputs:
        run.write_portable(
            output.logical_path, output.content, media_type=output.media_type, role=output.role
        )


def _experiment_id(request: ComparisonRequestV1) -> str:
    digest = hashlib.sha256(canonical_json_bytes(request.model_dump(mode="json"))).hexdigest()
    return f"compare-{digest[:24]}"


def compare_runs(
    baseline_run_id: str,
    candidate_run_id: str,
    *,
    home: Path,
    environ: Mapping[str, str],
    hooks: JudgeHooks | None = None,
) -> ComparisonRunResult:
    """Compare two finalized successful suite runs into one immutable comparison run.

    Authenticates both arms, checks equivalence, computes deterministic paired statistics,
    instantiates and renders the report, preflights the two portable outputs, and only then
    allocates a run, publishes ``comparison/report.json`` and ``comparison/report.md``, and
    finalizes ``RunOutcome.SUCCESS``. Pre-allocation failures raise a :class:`JudgeError`
    subclass and create no run; a post-allocation publication failure finalizes failure and
    raises :class:`JudgeExecutionError` carrying the run id and record.
    """

    hooks = hooks or JudgeHooks()
    baseline = _load_arm(baseline_run_id, home=home)
    candidate = _load_arm(candidate_run_id, home=home)
    _check_equivalence(baseline, candidate)

    request = ComparisonRequestV1(
        baseline_run_id=baseline.run_id,
        baseline_record_sha256=baseline.record_sha256,
        candidate_run_id=candidate.run_id,
        candidate_record_sha256=candidate.record_sha256,
    )
    try:
        cases = tuple(
            compare_case_samples(
                case_id,
                baseline.case_samples[case_id],
                candidate.case_samples[case_id],
                baseline_record_sha256=request.baseline_record_sha256,
                candidate_record_sha256=request.candidate_record_sha256,
            )
            for case_id in baseline.case_order
        )
        report = _build_report(request, baseline, candidate, cases)
    except JudgeStatisticsError:
        raise
    except (ValueError, ValidationError) as exc:
        raise JudgeStatisticsError("comparison report could not be built") from exc

    report_json = canonical_json_bytes(report.model_dump(mode="json"))
    report_md = render_report_markdown(report)
    outputs = (
        _PortableOutput(_REPORT_JSON_PATH, report_json, "application/json", "comparison"),
        _PortableOutput(_REPORT_MD_PATH, report_md, "text/markdown", "comparison"),
    )
    _preflight_outputs(outputs, environ=environ)

    captured = canonical_yaml_bytes(request.model_dump(mode="json"))
    publish = hooks.publish or _default_publish
    run = begin_run(
        _experiment_id(request),
        captured,
        resolved=request.model_dump(mode="json"),
        home=home,
        environ=environ,
        clock=hooks.clock,
        token_factory=hooks.token_factory,
    )
    run_id = run.run_id
    try:
        with run:
            publish(run, outputs)
            inspection = run.succeed()
            return ComparisonRunResult(run_id, inspection.outcome, inspection.record, report)
    except Exception as exc:  # noqa: BLE001 - run.__exit__ finalized FAILURE without a report
        record: Path | None = None
        try:
            record = inspect_run(run_id, home=home).record
        except (RunError, OSError, ValueError):
            record = None
        raise JudgeExecutionError(
            "comparison run failed before publishing its report", run_id=run_id, record=record
        ) from exc


def _load_arm(run_id: str, *, home: Path) -> FinalizedSuiteSnapshot:
    try:
        return load_finalized_suite_snapshot(run_id, home=home)
    except (SuiteError, RunError, RecordError, OSError, ValueError, ValidationError) as exc:
        raise JudgeLoadError(f"comparison arm could not be authenticated: {exc}") from exc

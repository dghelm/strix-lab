"""Publish derived capsule comparisons using the existing immutable run lifecycle."""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from strixlab.capsule_comparison import (
    CapsuleComparisonReportV1,
    RecordSha,
    RunId,
    compare_finalized_capsule_runs,
)
from strixlab.capsule_snapshots import load_finalized_capsule_snapshot
from strixlab.evidence import (
    MAX_AGGREGATE_BYTES,
    MAX_MEMBER_BYTES,
    RunOutcome,
    begin_run,
    inspect_run,
    validate_portable_payload,
)
from strixlab.manifests import Sha256Lower
from strixlab.secret_policy import RedactionContext, reject_sensitive_interpolations
from strixlab.serialization import canonical_json_bytes, canonical_yaml_bytes


class CapsuleComparisonRunError(RuntimeError):
    """Fixed-safe failure before comparison run allocation."""

    def __init__(self) -> None:
        super().__init__("capsule comparison could not be published")


class CapsuleComparisonExecutionError(CapsuleComparisonRunError):
    """Fixed-safe failure after allocation, with available evidence identity."""

    def __init__(self, run_id: str, record: Path | None) -> None:
        super().__init__()
        self.run_id = run_id
        self.record = record


class CapsuleComparisonRequestV1(BaseModel):
    """Captured publication request binding ordered arms and exact report bytes."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    kind: Literal["capsule-comparison"] = "capsule-comparison"
    policy: Literal["paired-latency-log-bootstrap-v1"] = "paired-latency-log-bootstrap-v1"
    baseline_run_id: RunId
    baseline_record_sha256: RecordSha
    candidate_run_id: RunId
    candidate_record_sha256: RecordSha
    comparison_sha256: Sha256Lower
    report_sha256: Sha256Lower
    markdown_sha256: Sha256Lower


@dataclass(frozen=True, slots=True)
class CapsuleComparisonRunResult:
    run_id: str
    outcome: RunOutcome
    record: Path
    report: CapsuleComparisonReportV1


def render_capsule_report_markdown(report: CapsuleComparisonReportV1) -> bytes:
    """Render only authenticated report fields; never interpret opaque payloads."""

    lines = [
        "# StrixLab capsule comparison report",
        "",
        f"- capsule: `{report.capsule_id}`",
        f"- scenario: `{report.scenario_sha256}`",
        f"- policy: `{report.comparison.policy}`",
        f"- comparison contract: `{report.comparison_sha256}`",
        f"- machine: `{report.machine_id}`",
        f"- overall verdict: **{report.overall_verdict}**",
        "",
        "> Provisional derived report. Export both source-run bundles for independent",
        "> arm verification. Opaque capsule payloads have no generic comparison semantics.",
        "",
    ]
    for arm in (report.baseline, report.candidate):
        lines.extend(
            [
                f"## {arm.label}",
                f"- run: `{arm.run_id}`",
                f"- record: `{arm.record_sha256}`",
                f"- candidate: `{arm.candidate}`",
                f"- result: `{arm.result_sha256}`",
                f"- protocol: `{arm.protocol_sha256}`",
                f"- build: `{arm.build_id}`",
                f"- build record: `{arm.build_record_sha256}`",
                "",
            ]
        )
    lines.extend(
        [
            "| coordinate | set | case | mode | pairs | baseline median s | candidate median s | "
            "improvement % | log CI | baseline log MAD | workspace delta bytes | "
            "verdict | protected |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for value in report.coordinates:
        coordinate = value.coordinate
        lines.append(
            f"| {coordinate.coordinate_id} | {coordinate.case_set} | {coordinate.case_id} | "
            f"{coordinate.mode} | {coordinate.sample_count} | "
            f"{value.baseline_median_seconds:.17g} | {value.candidate_median_seconds:.17g} | "
            f"{value.improvement_percent:.17g} | "
            f"[{value.log_ci_low:.17g}, {value.log_ci_high:.17g}] | "
            f"{value.baseline_noise_log:.17g} | {value.workspace_delta_bytes} | "
            f"{value.verdict} | {str(value.protected_regression).lower()} |"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _preflight(report_json: bytes, report_md: bytes, environ: Mapping[str, str]) -> None:
    # Exactly two fixed, distinct paths and at most two blobs: entry/file limits are
    # intrinsically bounded. Count bytes conservatively even if blobs happen to coincide.
    context = RedactionContext.from_environ(environ)
    if len(report_json) + len(report_md) > MAX_AGGREGATE_BYTES:
        raise CapsuleComparisonRunError()
    if report_json == report_md:
        raise CapsuleComparisonRunError()  # one blob cannot have two media types
    for content, media_type in ((report_json, "application/json"), (report_md, "text/markdown")):
        if len(content) > MAX_MEMBER_BYTES:
            raise CapsuleComparisonRunError()
        validate_portable_payload(content, media_type)
        context.assert_payload_safe(content)
        reject_sensitive_interpolations(content.decode("utf-8"))


def _reauthenticate(report: CapsuleComparisonReportV1, home: Path) -> None:
    for arm in (report.baseline, report.candidate):
        snapshot = load_finalized_capsule_snapshot(arm.run_id, home=home)
        if snapshot.record_sha256 != arm.record_sha256:
            raise CapsuleComparisonRunError()


def compare_capsule_runs(
    baseline_run_id: str,
    candidate_run_id: str,
    *,
    home: Path,
    environ: Mapping[str, str],
) -> CapsuleComparisonRunResult:
    """Authenticate, compare, preflight, then publish one fresh derived evidence run.

    Repeated calls retain identical reports in distinct runs. Both source records are
    reauthenticated immediately before publication. RunSession owns locking, secret
    checks, immutable finalization and crash recovery; neither arm is mutated.
    """

    try:
        result = compare_finalized_capsule_runs(baseline_run_id, candidate_run_id, home=home)
        markdown = render_capsule_report_markdown(result.report)
        _preflight(result.report_bytes, markdown, environ)
        request = CapsuleComparisonRequestV1(
            baseline_run_id=result.report.baseline.run_id,
            baseline_record_sha256=result.report.baseline.record_sha256,
            candidate_run_id=result.report.candidate.run_id,
            candidate_record_sha256=result.report.candidate.record_sha256,
            comparison_sha256=result.report.comparison_sha256,
            report_sha256=result.report_sha256,
            markdown_sha256=hashlib.sha256(markdown).hexdigest(),
        )
        resolved = request.model_dump(mode="json")
        digest = hashlib.sha256(canonical_json_bytes(resolved)).hexdigest()
        run = begin_run(
            f"capsule-compare-{digest[:24]}",
            canonical_yaml_bytes(resolved),
            resolved=resolved,
            home=home,
            environ=environ,
        )
    except Exception:
        raise CapsuleComparisonRunError() from None

    try:
        with run:
            _reauthenticate(result.report, home)
            run.write_portable(
                "comparison/report.json",
                result.report_bytes,
                media_type="application/json",
                role="comparison",
            )
            run.write_portable(
                "comparison/report.md", markdown, media_type="text/markdown", role="comparison"
            )
            inspection = run.succeed()
            return CapsuleComparisonRunResult(
                run.run_id, inspection.outcome, inspection.record, result.report
            )
    except Exception:
        record: Path | None = None
        with contextlib.suppress(Exception):
            record = inspect_run(run.run_id, home=home).record
        raise CapsuleComparisonExecutionError(run.run_id, record) from None

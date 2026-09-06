from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_capsule_comparison import _stub_cache  # noqa: F401
from test_capsule_comparison import real_pair as real_pair
from typer.testing import CliRunner

import strixlab.capsule_comparison_runs as publication
import strixlab.cli as cli
from strixlab.bundles import export_bundle, verify_bundle
from strixlab.capsule_comparison import compare_finalized_capsule_runs
from strixlab.evidence import (
    RunOutcome,
    RunSession,
    begin_run,
    inspect_run,
    list_portable_entries,
    read_record_member,
)


def _publish(pair: Any, **kwargs: Any) -> publication.CapsuleComparisonRunResult:
    home, baseline, candidate = pair
    return publication.compare_capsule_runs(
        baseline.run_id, candidate.run_id, home=home, environ=kwargs.pop("environ", {}), **kwargs
    )


def _entries(record: Path) -> dict[str, bytes]:
    return {
        entry.logical_path: read_record_member(record, f"portable/blobs/{entry.blob_sha256}")
        for entry in list_portable_entries(record)
    }


def test_exact_reports_request_repeat_direction_and_bundle(real_pair: Any, tmp_path: Path) -> None:
    home, baseline, candidate = real_pair
    expected = compare_finalized_capsule_runs(baseline.run_id, candidate.run_id, home=home)
    before = [inspect_run(arm.run_id, home=home).record_sha256 for arm in (baseline, candidate)]
    first = _publish(real_pair)
    second = _publish(real_pair)
    assert first.run_id != second.run_id
    assert first.outcome == second.outcome == RunOutcome.SUCCESS
    assert first.report.overall_verdict == "inconclusive"
    entries = _entries(first.record)
    assert entries == _entries(second.record)
    assert set(entries) == {"comparison/report.json", "comparison/report.md"}
    assert entries["comparison/report.json"] == expected.report_bytes
    assert entries["comparison/report.md"].endswith(b"\n")
    assert not entries["comparison/report.md"].endswith(b"\n\n")
    assert (
        b"Opaque capsule payloads have no generic comparison semantics"
        in entries["comparison/report.md"]
    )
    assert all(entry.role == "comparison" for entry in list_portable_entries(first.record))
    request = yaml.safe_load(read_record_member(first.record, "manifest.resolved.yaml"))
    assert request["report_sha256"] == expected.report_sha256
    assert request["markdown_sha256"] == hashlib.sha256(entries["comparison/report.md"]).hexdigest()
    assert request["baseline_record_sha256"] == baseline.record_sha256
    assert request["candidate_record_sha256"] == candidate.record_sha256
    assert request["comparison_sha256"] == expected.report.comparison_sha256
    assert read_record_member(first.record, "manifest.input.yaml") == read_record_member(
        first.record, "manifest.resolved.yaml"
    )
    reverse = publication.compare_capsule_runs(
        candidate.run_id, baseline.run_id, home=home, environ={}
    )
    assert reverse.report.baseline.run_id == candidate.run_id
    assert _entries(reverse.record)["comparison/report.json"] != expected.report_bytes
    assert before == [
        inspect_run(arm.run_id, home=home).record_sha256 for arm in (baseline, candidate)
    ]
    bundle = export_bundle(first.run_id, tmp_path / "bundle", home=home, environ={})
    verified = verify_bundle(bundle)
    assert verified.run_id == first.run_id
    assert verified.outcome == RunOutcome.SUCCESS
    assert _entries(bundle / "run") == entries


@pytest.mark.parametrize("arm", ["missing", "same", "foreign"])
def test_invalid_arms_allocate_nothing(
    real_pair: Any, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    home, baseline, candidate = real_pair
    candidate_id = candidate.run_id
    if arm == "missing":
        candidate_id = "not-a-run"
    elif arm == "same":
        candidate_id = baseline.run_id
    else:
        # A valid finalized record of another kind is still ineligible.
        with begin_run("foreign", b"{}\n", resolved={}, home=home, environ={}) as run:
            foreign = run.succeed()
        candidate_id = foreign.run_id

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("preflight failure allocated a run")

    monkeypatch.setattr(publication, "begin_run", forbidden)
    with pytest.raises(publication.CapsuleComparisonRunError):
        publication.compare_capsule_runs(baseline.run_id, candidate_id, home=home, environ={})


@pytest.mark.parametrize("content", [b"private-secret-value\n", b"${API_TOKEN}\n", b"x" * 50])
def test_preflight_failure_allocates_nothing(
    real_pair: Any, monkeypatch: pytest.MonkeyPatch, content: bytes
) -> None:
    monkeypatch.setattr(publication, "render_capsule_report_markdown", lambda _: content)
    if content == b"x" * 50:
        monkeypatch.setattr(publication, "MAX_MEMBER_BYTES", 40)
    monkeypatch.setattr(publication, "begin_run", lambda *a, **k: pytest.fail("allocated"))
    with pytest.raises(publication.CapsuleComparisonRunError) as error:
        _publish(real_pair, environ={"API_TOKEN": "private-secret-value"})
    assert "private-secret-value" not in str(error.value)


@pytest.mark.parametrize("failure", ["load", "drift", "first-write", "second-write", "finalize"])
def test_publication_failure_preserves_evidence_and_safe_id(
    real_pair: Any, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    home, baseline, _ = real_pair
    if failure in {"load", "drift"}:

        def loader(*args: Any, **kwargs: Any) -> Any:
            if failure == "load":
                raise RuntimeError("private-secret-value")
            return replace(baseline, record_sha256="record-sha256:" + "0" * 64)

        monkeypatch.setattr(publication, "load_finalized_capsule_snapshot", loader)
    elif failure == "finalize":

        def fail_success(self: RunSession) -> Any:
            raise RuntimeError("private-secret-value")

        monkeypatch.setattr(RunSession, "succeed", fail_success)
    else:
        original = RunSession.write_portable

        def write(self: RunSession, path: str, *args: Any, **kwargs: Any) -> Any:
            if failure == "first-write" or path.endswith(".md"):
                raise RuntimeError("private-secret-value")
            return original(self, path, *args, **kwargs)

        monkeypatch.setattr(RunSession, "write_portable", write)
    with pytest.raises(publication.CapsuleComparisonExecutionError) as caught:
        _publish(real_pair, environ={"API_TOKEN": "private-secret-value"})
    error = caught.value
    assert "private-secret-value" not in str(error)
    assert error.record is not None
    inspection = inspect_run(error.run_id, home=home)
    assert inspection.outcome == RunOutcome.FAILURE
    assert set(_entries(error.record)) == (
        {"comparison/report.json"}
        if failure == "second-write"
        else {"comparison/report.json", "comparison/report.md"}
        if failure == "finalize"
        else set()
    )


def test_cli_real_capsule_and_unchanged_default(
    real_pair: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, baseline, candidate = real_pair
    args = ["compare", baseline.run_id, candidate.run_id, "--home", str(home)]
    result = CliRunner().invoke(cli.app, [*args, "--kind", "capsule"])
    assert result.exit_code == 0, result.output
    assert "verdict: inconclusive" in result.stdout
    assert "comparison: run-" in result.stdout
    calls = []

    def suite(*args: Any, **kwargs: Any) -> Any:
        calls.append(args)
        raise cli.JudgeError("suite-dispatch")

    monkeypatch.setattr(cli, "compare_runs", suite)
    for extra in ([], ["--kind", "suite"]):
        result = CliRunner().invoke(cli.app, [*args, *extra])
        assert result.exit_code == 1
        assert "suite-dispatch" in result.output
    assert len(calls) == 2
    result = CliRunner().invoke(cli.app, [*args, "--kind", "unknown"])
    assert result.exit_code == 2
    assert len(calls) == 2


@pytest.mark.parametrize("allocated", [False, True])
def test_cli_fixed_safe_errors(monkeypatch: pytest.MonkeyPatch, allocated: bool) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        if allocated:
            raise publication.CapsuleComparisonExecutionError("safe-run-id", None)
        raise publication.CapsuleComparisonRunError()

    monkeypatch.setattr(cli, "compare_capsule_runs", fail)
    result = CliRunner().invoke(cli.app, ["compare", "a", "b", "--kind", "capsule"])
    assert result.exit_code == 1
    assert ("comparison: safe-run-id" in result.output) == allocated
    assert "capsule comparison could not be published" in result.output

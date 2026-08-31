from __future__ import annotations

import contextlib
import hashlib
import json
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import _suite_fixtures as fx
import pytest
from pydantic import ValidationError

from strixlab.config import read_manifest
from strixlab.evidence import (
    PortableEvidenceV1,
    RunError,
    RunOutcome,
    RunSession,
    inspect_run,
    list_portable_entries,
)
from strixlab.judge import (
    BOOTSTRAP_REPLICATES,
    MIN_PAIRED_SAMPLES,
    POLICY_ID,
    CaseComparisonV1,
    ComparisonArmV1,
    ComparisonReportV1,
    ComparisonRequestV1,
    JudgeEquivalenceError,
    JudgeExecutionError,
    JudgeHooks,
    JudgeLoadError,
    JudgeStatisticsError,
    _bootstrap_index,
    _median,
    _percentile,
    _PortableOutput,
    _preflight_outputs,
    case_verdict,
    compare_case_samples,
    compare_runs,
    overall_verdict,
    render_report_markdown,
)
from strixlab.locks import LockAttempt, LockStatus
from strixlab.manifests import (
    ExclusiveLockV1,
    MachineExpectationV1,
    MachineProfileV1,
    MachineValidityV1,
    SuiteManifestV1,
    TelemetryV1,
    validate_manifest,
)
from strixlab.serialization import canonical_json_bytes
from strixlab.suites import (
    _SAMPLE_MODELS,
    SuiteError,
    SuiteHooks,
    SuiteResultV1,
    _authenticate_input_models,
    _bind_input_snapshots,
    _bind_result_to_manifest,
    _bind_sample_contracts,
    _measurement_coordinates,
    _parse_canonical_json,
    _reauthenticate_samples,
    _snapshot_blob,
    _validate_suite_input_eligibility,
    load_finalized_suite_snapshot,
    plan_performance,
    run_suite,
)

_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "suites" / "smoke-qwen35.yaml"
_RB = "record-sha256:" + "ab" * 32
_RC = "record-sha256:" + "cd" * 32


@pytest.fixture(autouse=True)
def _stub(monkeypatch: pytest.MonkeyPatch) -> None:
    fx.stub_cache_verification(monkeypatch)


# --- Suite-run harness (two finalized arms, hardware-free) --------------------


def _manifest() -> SuiteManifestV1:
    manifest = validate_manifest("suite", read_manifest(_CONFIG))
    assert isinstance(manifest, SuiteManifestV1)
    return manifest


def _machine(tmp_path: Path) -> MachineProfileV1:
    return MachineProfileV1(
        schema_version=1,
        id="strix-halo-128g",
        expect=MachineExpectationV1(gpu_arch="gfx1151", integrated_gpu=True, memory_gib_min=64),
        exclusive_lock=ExclusiveLockV1(path=str(tmp_path / "gpu.lock")),
        telemetry=TelemetryV1(amd_smi="auto", sample_interval_ms=100),
        validity=MachineValidityV1(
            require_ac_power=True,
            max_background_gpu_busy_pct=10,
            min_available_memory_gib=8,
            temperature_warn_c=90,
        ),
    )


def _acquiring_lock() -> Any:
    @contextlib.contextmanager
    def factory(lock_path: Path) -> Iterator[LockAttempt]:
        yield LockAttempt(LockStatus.ACQUIRED, lock_path)

    return factory


# A per-case throughput map: manifest case id -> base samples/second. The bench fake
# derives the manifest case id and window from the namespaced adapter case id and
# produces deterministic ordered ``samples_ts`` so paired statistics are exactly known.
CaseValues = dict[str, float]


def _bench_values(values: CaseValues, *, step: float = 0.1) -> Callable[[Any, Any], Any]:
    def bench(inputs: Any, case: Any) -> Any:
        sample = fx.bench_success(inputs, case)
        measurement = sample.measurement
        assert measurement is not None
        # ``case.id`` is ``measure-NN-<caseid>`` or ``warmup-NN-<caseid>``; recover the
        # manifest case id as the remainder after the two leading segments.
        manifest_case = case.id.split("-", 2)[2]
        base = values[manifest_case]
        samples = tuple(base + step * index for index in range(case.repetitions))
        updated = measurement.model_copy(
            update={
                "avg_ts": sum(samples) / len(samples),
                "stddev_ts": 0.5,
                "samples_ts": samples,
            }
        )
        return sample.model_copy(update={"measurement": updated})

    return bench


def _prepare_home(tmp_path: Path) -> tuple[Path, str]:
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o700)
    fx.make_present_build(home)
    receipt_sha = fx.publish_receipt(home, tmp_path / "scratch")
    return home, receipt_sha


def _run_suite(
    tmp_path: Path,
    home: Path,
    receipt_sha: str,
    values: CaseValues,
    *,
    token_factory: Any = None,
) -> Any:
    def temp_root() -> Path:
        return Path(tempfile.mkdtemp(dir=tmp_path))

    hooks = SuiteHooks(
        temp_root_factory=temp_root,
        token_factory=token_factory,
        machine_lock=_acquiring_lock(),
        backend_ops=fx.fake_backend(),
        llama_server=fx.fake_server(),
        llama_bench=fx.fake_bench(_bench_values(values)),
    )
    return run_suite(
        _manifest(),
        _CONFIG.read_bytes(),
        machine_profile=_machine(tmp_path),
        build_id=fx.BUILD_ID,
        local_receipt_sha256=receipt_sha,
        home=home,
        environ={"PATH": "/usr/bin"},
        hooks=hooks,
    )


_UNIFORM: CaseValues = {"pp512": 100.0, "pp2048": 40.0, "tg128": 20.0}


def _two_arms(
    tmp_path: Path, *, baseline: CaseValues, candidate: CaseValues
) -> tuple[Path, str, str]:
    home, receipt_sha = _prepare_home(tmp_path)
    base = _run_suite(tmp_path, home, receipt_sha, baseline)
    cand = _run_suite(tmp_path, home, receipt_sha, candidate)
    assert base.outcome is RunOutcome.SUCCESS and cand.outcome is RunOutcome.SUCCESS
    return home, base.run_id, cand.run_id


# --- Pure verdict projections -------------------------------------------------


def test_case_verdict_projection() -> None:
    assert case_verdict(-0.1, 0.1, 0.0, 0.0) == "inconclusive"  # interval includes zero
    assert case_verdict(0.0, 0.2, 0.1, 0.0) == "inconclusive"  # low endpoint is zero
    assert case_verdict(0.05, 0.2, 0.1, 0.2) == "inconclusive"  # below noise
    assert case_verdict(0.05, 0.2, 0.1, 0.01) == "improvement"  # wholly above, above noise
    assert case_verdict(-0.2, -0.05, -0.1, 0.01) == "regression"  # wholly below, above noise


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [
        (("inconclusive",), "inconclusive"),
        (("improvement", "improvement"), "improvement"),
        (("regression", "regression"), "regression"),
        (("improvement", "regression"), "mixed"),
        (("improvement", "inconclusive"), "mixed"),
        (("regression", "inconclusive"), "mixed"),
        (("improvement", "regression", "inconclusive"), "mixed"),
    ],
)
def test_overall_verdict_projection(verdicts: tuple[str, ...], expected: str) -> None:
    assert overall_verdict(verdicts) == expected  # type: ignore[arg-type]


# --- Pure statistics ----------------------------------------------------------


def _samples(*values: float) -> tuple[float, ...]:
    return tuple(values)


def _compare(baseline: tuple[float, ...], candidate: tuple[float, ...]) -> CaseComparisonV1:
    return compare_case_samples(
        "pp512", baseline, candidate, baseline_record_sha256=_RB, candidate_record_sha256=_RC
    )


def test_identical_samples_are_inconclusive() -> None:
    result = _compare(
        _samples(10.0, 10.0, 10.0, 10.0, 10.0), _samples(10.0, 10.0, 10.0, 10.0, 10.0)
    )
    assert result.verdict == "inconclusive"
    assert result.mean_log_delta == 0.0
    assert result.delta_percent == 0.0
    assert result.speed_ratio == 1.0
    assert result.log_ci_low == 0.0 and result.log_ci_high == 0.0
    assert result.baseline_noise_log == 0.0


def test_uniform_improvement_is_decisive() -> None:
    baseline = tuple(10.0 + 0.01 * i for i in range(8))
    candidate = tuple(value * 1.5 for value in baseline)
    result = _compare(baseline, candidate)
    assert result.verdict == "improvement"
    assert result.speed_ratio > 1.0
    assert result.log_ci_low > 0.0


def test_uniform_regression_is_decisive() -> None:
    baseline = tuple(10.0 + 0.01 * i for i in range(8))
    candidate = tuple(value * 1.5 for value in baseline)
    result = _compare(candidate, baseline)
    assert result.verdict == "regression"
    assert result.speed_ratio < 1.0
    assert result.log_ci_high < 0.0


def test_interval_crossing_zero_is_inconclusive() -> None:
    baseline = (10.0, 10.0, 10.0, 10.0, 10.0, 10.0)
    candidate = (12.0, 8.0, 11.0, 9.0, 13.0, 7.0)
    result = _compare(baseline, candidate)
    assert result.verdict == "inconclusive"
    assert result.log_ci_low < 0.0 < result.log_ci_high


def test_small_signal_below_noise_is_inconclusive() -> None:
    # A tiny, consistent uplift far below the noisy baseline's own MAD is inconclusive.
    baseline = (10.0, 30.0, 20.0, 40.0, 15.0, 35.0, 25.0)
    candidate = tuple(value * 1.001 for value in baseline)
    result = _compare(baseline, candidate)
    assert result.verdict == "inconclusive"
    assert abs(result.mean_log_delta) <= result.baseline_noise_log


def test_repeated_calls_are_deterministic() -> None:
    baseline = tuple(10.0 + 0.01 * i for i in range(9))
    candidate = tuple(value * 1.3 for value in baseline)
    first = _compare(baseline, candidate)
    second = _compare(baseline, candidate)
    assert first.model_dump() == second.model_dump()


def test_extreme_finite_inputs_are_handled() -> None:
    baseline = (1e-6, 1e-6, 1e-6, 1e-6, 1e-6)
    candidate = (1e6, 1e6, 1e6, 1e6, 1e6)
    result = _compare(baseline, candidate)
    assert result.verdict == "improvement"
    assert result.speed_ratio > 1.0


def test_overflowing_ratio_is_typed_statistics_error() -> None:
    baseline = tuple([1e-300] * MIN_PAIRED_SAMPLES)
    candidate = tuple([1e300] * MIN_PAIRED_SAMPLES)
    with pytest.raises(JudgeStatisticsError):
        _compare(baseline, candidate)


@pytest.mark.parametrize("count", [1, 4])
def test_below_minimum_pairs_rejected(count: int) -> None:
    values = tuple([10.0] * count)
    with pytest.raises(JudgeStatisticsError):
        _compare(values, values)


def test_five_pairs_accepted() -> None:
    result = _compare(
        _samples(10.0, 11.0, 12.0, 13.0, 14.0), _samples(10.0, 11.0, 12.0, 13.0, 14.0)
    )
    assert result.pair_count == MIN_PAIRED_SAMPLES


def test_mismatched_pair_counts_rejected() -> None:
    with pytest.raises(JudgeStatisticsError):
        _compare(_samples(10.0, 10.0, 10.0, 10.0, 10.0), _samples(10.0, 10.0, 10.0, 10.0))


@pytest.mark.parametrize(
    ("baseline", "candidate"),
    [
        ((10.0, 0.0, 10.0, 10.0, 10.0), (11.0, 11.0, 11.0, 11.0, 11.0)),
        ((10.0, 10.0, 10.0, 10.0, 10.0), (11.0, -1.0, 11.0, 11.0, 11.0)),
        ((10.0, float("inf"), 10.0, 10.0, 10.0), (11.0, 11.0, 11.0, 11.0, 11.0)),
    ],
)
def test_nonpositive_or_nonfinite_rejected(
    baseline: tuple[float, ...], candidate: tuple[float, ...]
) -> None:
    with pytest.raises(JudgeStatisticsError):
        _compare(baseline, candidate)


# --- Bootstrap and percentile golden vectors ----------------------------------


def test_bootstrap_index_golden_vector() -> None:
    indices = [
        _bootstrap_index(_RB, _RC, "pp512", r, d, 5)
        for r, d in [(0, 0), (0, 1), (0, 4), (1, 0), (4095, 4)]
    ]
    assert indices == [1, 0, 4, 2, 4]


def test_percentile_r7_golden_vector() -> None:
    values = [float(i) for i in range(BOOTSTRAP_REPLICATES)]
    assert _percentile(values, 0.025) == 102.375
    assert _percentile(values, 0.975) == 3992.625


def test_median_odd_and_even() -> None:
    assert _median([1.0, 2.0, 3.0]) == 2.0
    assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_bootstrap_interval_is_seed_bound() -> None:
    # Swapping the case id (a domain-separator field) changes the resampling, so the
    # interval is bound to the arms and case, not shared across cases.
    one = _compare(
        _samples(10.0, 20.0, 30.0, 40.0, 50.0, 60.0), _samples(11.0, 19.0, 33.0, 38.0, 55.0, 57.0)
    )
    other = compare_case_samples(
        "tg128",
        _samples(10.0, 20.0, 30.0, 40.0, 50.0, 60.0),
        _samples(11.0, 19.0, 33.0, 38.0, 55.0, 57.0),
        baseline_record_sha256=_RB,
        candidate_record_sha256=_RC,
    )
    assert (one.log_ci_low, one.log_ci_high) != (other.log_ci_low, other.log_ci_high)


# --- Strict model invariants --------------------------------------------------


def _valid_case() -> CaseComparisonV1:
    return compare_case_samples(
        "pp512",
        _samples(10.0, 11.0, 12.0, 13.0, 14.0),
        _samples(12.0, 13.2, 14.4, 15.6, 16.8),
        baseline_record_sha256=_RB,
        candidate_record_sha256=_RC,
    )


def _case_dict(**overrides: Any) -> dict[str, Any]:
    value = _valid_case().model_dump(mode="json")
    value.update(overrides)
    return value


@pytest.mark.parametrize(
    "overrides",
    [
        {"speed_ratio": 1.5},
        {"delta_percent": 5.0},
        {"percent_ci_low": -5.0},
        {"percent_ci_high": 999.0},
        {"baseline_noise_percent": 0.0},
        {"verdict": "regression"},
        {"log_ci_low": 0.5, "log_ci_high": 0.1},
        {"pair_count": 4},
        {"baseline_mean": -1.0},
        {"speed_ratio": 0.0},
    ],
)
def test_case_comparison_rejects_inconsistent_fields(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        CaseComparisonV1.model_validate_json(json.dumps(_case_dict(**overrides)))


def _arm_model(label: str, run_id: str) -> ComparisonArmV1:
    return ComparisonArmV1(
        label=label,  # type: ignore[arg-type]
        run_id=run_id,
        record_sha256=_RB if label == "baseline" else _RC,
        suite_result_sha256="ee" * 32,
        build_id="build-sha256:" + "12" * 32,
        build_record_sha256="record-sha256:" + "34" * 32,
    )


def _report_dict(**overrides: Any) -> dict[str, Any]:
    report = ComparisonReportV1(
        suite_id="smoke-qwen35",
        machine_id="strix-halo-128g",
        model_id="qwen35-4b-smoke",
        resolved_manifest_sha256="ab" * 32,
        baseline=_arm_model("baseline", "run-a"),
        candidate=_arm_model("candidate", "run-b"),
        overall_verdict="improvement",
        cases=(_valid_case(),),
    )
    value = report.model_dump(mode="json")
    value.update(overrides)
    return value


def test_report_accepts_a_valid_projection() -> None:
    report = ComparisonReportV1.model_validate_json(json.dumps(_report_dict()))
    assert report.overall_verdict == "improvement"
    assert report.policy_id == POLICY_ID == "paired-log-bootstrap-v1"


def test_report_rejects_mislabeled_arms() -> None:
    swapped = _report_dict()
    swapped["baseline"], swapped["candidate"] = swapped["candidate"], swapped["baseline"]
    with pytest.raises(ValidationError):
        ComparisonReportV1.model_validate_json(json.dumps(swapped))


def test_report_rejects_same_run_arms() -> None:
    value = _report_dict()
    value["candidate"]["run_id"] = value["baseline"]["run_id"]
    with pytest.raises(ValidationError):
        ComparisonReportV1.model_validate_json(json.dumps(value))


def test_report_rejects_wrong_overall_verdict() -> None:
    with pytest.raises(ValidationError):
        ComparisonReportV1.model_validate_json(
            json.dumps(_report_dict(overall_verdict="regression"))
        )


def test_report_rejects_duplicate_cases() -> None:
    value = _report_dict()
    value["cases"] = [value["cases"][0], value["cases"][0]]
    with pytest.raises(ValidationError):
        ComparisonReportV1.model_validate_json(json.dumps(value))


def test_report_rejects_empty_cases() -> None:
    with pytest.raises(ValidationError):
        ComparisonReportV1.model_validate_json(json.dumps(_report_dict(cases=[])))


def test_request_rejects_same_run() -> None:
    with pytest.raises(ValidationError):
        ComparisonRequestV1(
            baseline_run_id="run-x",
            baseline_record_sha256=_RB,
            candidate_run_id="run-x",
            candidate_record_sha256=_RC,
        )


# --- Markdown rendering -------------------------------------------------------


def _report_with(overall: str, cases: tuple[CaseComparisonV1, ...]) -> ComparisonReportV1:
    return ComparisonReportV1(
        suite_id="smoke-qwen35",
        machine_id="strix-halo-128g",
        model_id="qwen35-4b-smoke",
        resolved_manifest_sha256="ab" * 32,
        baseline=_arm_model("baseline", "run-a"),
        candidate=_arm_model("candidate", "run-b"),
        overall_verdict=overall,  # type: ignore[arg-type]
        cases=cases,
    )


def test_markdown_is_a_pure_rendering() -> None:
    report = _report_with("improvement", (_valid_case(),))
    rendered = render_report_markdown(report)
    text = rendered.decode("utf-8")
    assert rendered.endswith(b"\n") and not rendered.endswith(b"\n\n")
    assert "not independently verifiable without both source-run bundles" in text
    assert f"`{POLICY_ID}`" in text
    assert "| pp512 |" in text
    assert "**improvement**" in text
    # Deterministic: identical reports render byte-identically.
    assert render_report_markdown(report) == rendered


# --- Snapshot loader and full comparison --------------------------------------


def test_identical_arms_are_inconclusive(tmp_path: Path) -> None:
    home, baseline_id, candidate_id = _two_arms(tmp_path, baseline=_UNIFORM, candidate=_UNIFORM)
    result = compare_runs(baseline_id, candidate_id, home=home, environ={"PATH": "/usr/bin"})
    assert result.outcome is RunOutcome.SUCCESS
    assert result.report.overall_verdict == "inconclusive"
    assert all(case.verdict == "inconclusive" for case in result.report.cases)
    assert {case.pair_count for case in result.report.cases} == {15}


def test_uniform_faster_candidate_improves_and_reverse_regresses(tmp_path: Path) -> None:
    faster = {name: value * 1.5 for name, value in _UNIFORM.items()}
    home, baseline_id, candidate_id = _two_arms(tmp_path, baseline=_UNIFORM, candidate=faster)
    improved = compare_runs(baseline_id, candidate_id, home=home, environ={"PATH": "/usr/bin"})
    assert improved.report.overall_verdict == "improvement"
    regressed = compare_runs(candidate_id, baseline_id, home=home, environ={"PATH": "/usr/bin"})
    assert regressed.report.overall_verdict == "regression"


def test_per_case_divergence_is_mixed(tmp_path: Path) -> None:
    candidate = {"pp512": 150.0, "pp2048": 20.0, "tg128": 20.0}  # faster, slower, equal
    home, baseline_id, candidate_id = _two_arms(tmp_path, baseline=_UNIFORM, candidate=candidate)
    result = compare_runs(baseline_id, candidate_id, home=home, environ={"PATH": "/usr/bin"})
    verdicts = {case.case_id: case.verdict for case in result.report.cases}
    assert verdicts == {"pp512": "improvement", "pp2048": "regression", "tg128": "inconclusive"}
    assert result.report.overall_verdict == "mixed"


def test_snapshot_binds_authenticated_identities(tmp_path: Path) -> None:
    home, receipt_sha = _prepare_home(tmp_path)
    run = _run_suite(tmp_path, home, receipt_sha, _UNIFORM)
    snapshot = load_finalized_suite_snapshot(run.run_id, home=home)
    inspection = inspect_run(run.run_id, home=home)
    assert snapshot.record_sha256 == inspection.record_sha256
    assert snapshot.build_id == fx.BUILD_ID
    assert snapshot.case_order == ("pp512", "pp2048", "tg128")
    assert snapshot.measurement_windows == 5 and snapshot.repetitions_per_window == 3
    assert all(len(values) == 15 for values in snapshot.case_samples.values())


def test_comparison_output_contains_only_the_two_reports(tmp_path: Path) -> None:
    home, baseline_id, candidate_id = _two_arms(tmp_path, baseline=_UNIFORM, candidate=_UNIFORM)
    result = compare_runs(baseline_id, candidate_id, home=home, environ={"PATH": "/usr/bin"})
    inspection = inspect_run(result.run_id, home=home)
    entries = list_portable_entries(inspection.record)
    assert {entry.logical_path for entry in entries} == {
        "comparison/report.json",
        "comparison/report.md",
    }
    assert all(entry.role == "comparison" for entry in entries)


def test_comparison_composes_with_inspect_and_bundles(tmp_path: Path) -> None:
    from strixlab.bundles import export_bundle, verify_bundle

    home, baseline_id, candidate_id = _two_arms(tmp_path, baseline=_UNIFORM, candidate=_UNIFORM)
    result = compare_runs(baseline_id, candidate_id, home=home, environ={"PATH": "/usr/bin"})
    # run inspect composes with the comparison run.
    inspection = inspect_run(result.run_id, home=home)
    assert inspection.outcome is RunOutcome.SUCCESS
    # bundle export + verify compose with the comparison run.
    destination = tmp_path / "comparison-bundle"
    export_bundle(result.run_id, destination, home=home, environ={"PATH": "/usr/bin"})
    verified = verify_bundle(destination)
    assert verified.run_id == result.run_id
    assert verified.run_record_sha256 == inspection.record_sha256


def _bundle_entry(bundle_dir: Path, logical_path: str) -> PortableEvidenceV1:
    entries_dir = bundle_dir / "run" / "portable" / "entries"
    for entry_path in sorted(entries_dir.iterdir()):
        entry = PortableEvidenceV1.model_validate_json(entry_path.read_bytes())
        if entry.logical_path == logical_path:
            return entry
    raise KeyError(logical_path)


def test_three_bundle_offline_linkage(tmp_path: Path) -> None:
    from strixlab.bundles import export_bundle, verify_bundle
    from strixlab.judge import ComparisonReportV1

    home, baseline_id, candidate_id = _two_arms(tmp_path, baseline=_UNIFORM, candidate=_UNIFORM)
    result = compare_runs(baseline_id, candidate_id, home=home, environ={"PATH": "/usr/bin"})

    offline = tmp_path / "offline"
    offline.mkdir()
    baseline_bundle = offline / "baseline"
    candidate_bundle = offline / "candidate"
    comparison_bundle = offline / "comparison"
    for run_id, destination in (
        (baseline_id, baseline_bundle),
        (candidate_id, candidate_bundle),
        (result.run_id, comparison_bundle),
    ):
        export_bundle(run_id, destination, home=home, environ={"PATH": "/usr/bin"})

    baseline_verified = verify_bundle(baseline_bundle)
    candidate_verified = verify_bundle(candidate_bundle)
    verify_bundle(comparison_bundle)

    # Read the derived report back out of the comparison bundle, offline.
    report_entry = _bundle_entry(comparison_bundle, "comparison/report.json")
    report_bytes = (
        comparison_bundle / "run" / "portable" / "blobs" / report_entry.blob_sha256
    ).read_bytes()
    report = ComparisonReportV1.model_validate_json(report_bytes)

    # Each report arm matches its source bundle's run id, record digest, and suite-result
    # blob digest; both arms share the resolved-manifest digest.
    baseline_result_entry = _bundle_entry(baseline_bundle, "suite/result.json")
    candidate_result_entry = _bundle_entry(candidate_bundle, "suite/result.json")
    assert report.baseline.run_id == baseline_verified.run_id
    assert report.baseline.record_sha256 == baseline_verified.run_record_sha256
    assert report.baseline.suite_result_sha256 == baseline_result_entry.blob_sha256
    assert report.candidate.run_id == candidate_verified.run_id
    assert report.candidate.record_sha256 == candidate_verified.run_record_sha256
    assert report.candidate.suite_result_sha256 == candidate_result_entry.blob_sha256
    baseline_manifest = (baseline_bundle / "run" / "manifest.resolved.yaml").read_bytes()
    candidate_manifest = (candidate_bundle / "run" / "manifest.resolved.yaml").read_bytes()
    assert baseline_manifest == candidate_manifest
    manifest_sha256 = hashlib.sha256(baseline_manifest).hexdigest()
    assert manifest_sha256 == report.resolved_manifest_sha256


# --- Equivalence and load failures --------------------------------------------


def _count_runs(home: Path) -> int:
    records = home / "runs" / "records"
    return len(list(records.iterdir())) if records.exists() else 0


def test_same_run_rejected_without_allocation(tmp_path: Path) -> None:
    home, receipt_sha = _prepare_home(tmp_path)
    run = _run_suite(tmp_path, home, receipt_sha, _UNIFORM)
    before = _count_runs(home)
    with pytest.raises(JudgeEquivalenceError):
        compare_runs(run.run_id, run.run_id, home=home, environ={"PATH": "/usr/bin"})
    assert _count_runs(home) == before


def test_incomparable_manifests_rejected_without_allocation(tmp_path: Path) -> None:
    import dataclasses

    from strixlab.judge import _check_equivalence

    home, baseline_id, candidate_id = _two_arms(tmp_path, baseline=_UNIFORM, candidate=_UNIFORM)
    baseline = load_finalized_suite_snapshot(baseline_id, home=home)
    candidate = load_finalized_suite_snapshot(candidate_id, home=home)
    divergent = dataclasses.replace(
        candidate, resolved_manifest_bytes=candidate.resolved_manifest_bytes + b" "
    )
    with pytest.raises(JudgeEquivalenceError):
        _check_equivalence(baseline, divergent)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda snap: {"model_input_sha256": "00" * 32},
        lambda snap: {"machine_input_sha256": "00" * 32},
        lambda snap: {"case_order": ("pp512", "pp2048")},
        lambda snap: {"measurement_windows": snap.measurement_windows + 1},
        lambda snap: {"repetitions_per_window": snap.repetitions_per_window + 1},
        lambda snap: {
            "case_samples": {name: values[:4] for name, values in snap.case_samples.items()}
        },
    ],
)
def test_equivalence_rules(tmp_path: Path, mutate: Callable[[Any], dict[str, Any]]) -> None:
    import dataclasses

    from strixlab.judge import _check_equivalence

    home, receipt_sha = _prepare_home(tmp_path)
    run = _run_suite(tmp_path, home, receipt_sha, _UNIFORM)
    snapshot = load_finalized_suite_snapshot(run.run_id, home=home)
    candidate = dataclasses.replace(snapshot, run_id=snapshot.run_id + "x", **mutate(snapshot))
    with pytest.raises(JudgeEquivalenceError):
        _check_equivalence(snapshot, candidate)


def test_failed_arm_is_a_load_error(tmp_path: Path) -> None:
    home, receipt_sha = _prepare_home(tmp_path)

    def temp_root() -> Path:
        return Path(tempfile.mkdtemp(dir=tmp_path))

    hooks = SuiteHooks(
        temp_root_factory=temp_root,
        machine_lock=_acquiring_lock(),
        backend_ops=fx.fake_backend(fx.backend_gate_failed),
        llama_server=fx.fake_server(),
        llama_bench=fx.fake_bench(),
    )
    failed = run_suite(
        _manifest(),
        _CONFIG.read_bytes(),
        machine_profile=_machine(tmp_path),
        build_id=fx.BUILD_ID,
        local_receipt_sha256=receipt_sha,
        home=home,
        environ={"PATH": "/usr/bin"},
        hooks=hooks,
    )
    assert failed.outcome is RunOutcome.FAILURE
    good = _run_suite(tmp_path, home, receipt_sha, _UNIFORM)
    with pytest.raises(JudgeLoadError):
        compare_runs(failed.run_id, good.run_id, home=home, environ={"PATH": "/usr/bin"})
    with pytest.raises(SuiteError):
        load_finalized_suite_snapshot(failed.run_id, home=home)


def test_missing_run_is_a_load_error(tmp_path: Path) -> None:
    home, receipt_sha = _prepare_home(tmp_path)
    good = _run_suite(tmp_path, home, receipt_sha, _UNIFORM)
    with pytest.raises(JudgeLoadError):
        compare_runs("run-does-not-exist", good.run_id, home=home, environ={"PATH": "/usr/bin"})


def test_tampered_blob_fails_immutable_verification(tmp_path: Path) -> None:
    from strixlab.records import RecordError

    home, receipt_sha = _prepare_home(tmp_path)
    run = _run_suite(tmp_path, home, receipt_sha, _UNIFORM)
    good = _run_suite(tmp_path, home, receipt_sha, _UNIFORM)
    inspection = inspect_run(run.run_id, home=home)
    blobs = inspection.record / "portable" / "blobs"
    victim = next(iter(sorted(blobs.iterdir())))
    victim.chmod(0o600)
    victim.write_bytes(victim.read_bytes() + b"tamper")
    # The immutable record verifier rejects the tampered blob before the snapshot loads.
    with pytest.raises(RecordError):
        load_finalized_suite_snapshot(run.run_id, home=home)
    with pytest.raises(JudgeLoadError):
        compare_runs(run.run_id, good.run_id, home=home, environ={"PATH": "/usr/bin"})


# --- Canonical-but-semantically-invalid record detection ----------------------


@pytest.fixture(scope="module")
def _arm(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Any, Any]]:
    monkeypatch = pytest.MonkeyPatch()
    fx.stub_cache_verification(monkeypatch)
    tmp = tmp_path_factory.mktemp("arm")
    home = tmp / "home"
    home.mkdir()
    home.chmod(0o700)
    fx.make_present_build(home)
    receipt_sha = fx.publish_receipt(home, tmp / "scratch")
    run = _run_suite(tmp, home, receipt_sha, _UNIFORM)
    snapshot = load_finalized_suite_snapshot(run.run_id, home=home)
    entries = {entry.logical_path: entry for entry in list_portable_entries(snapshot.record)}
    yield snapshot, entries
    monkeypatch.undo()


def _tamper_result(result: SuiteResultV1, **update: Any) -> SuiteResultV1:
    return result.model_copy(update=update)


def _reauthenticate_arm(snapshot: Any, entries: dict[str, PortableEvidenceV1], result: Any) -> None:
    inputs = _bind_input_snapshots(snapshot.record, entries, snapshot.manifest, result)
    _reauthenticate_samples(snapshot.record, entries, snapshot.manifest, result, inputs)


def _parsed_arm_samples(snapshot: Any, entries: dict[str, PortableEvidenceV1]) -> list[Any]:
    parsed: list[Any] = []
    for reference in snapshot.result.samples:
        _entry, content = _snapshot_blob(snapshot.record, entries, reference.logical_path)
        parsed.append(
            _parse_canonical_json(
                content,
                _SAMPLE_MODELS[reference.adapter],
                validation_error="invalid sample",
                canonical_error="noncanonical sample",
            )
        )
    return parsed


def test_bind_result_accepts_authentic(_arm: tuple[Any, Any]) -> None:
    snapshot, _entries = _arm
    _bind_result_to_manifest(snapshot.result, snapshot.manifest)  # does not raise


@pytest.mark.parametrize(
    "update",
    [
        {"status": "failed"},
        {"reason": "integrity-failed"},
        {"backend_ops": None},
        {"greedy": None},
        {"suite_id": "other-suite"},
        {"machine_id": "other-machine"},
        {"model_id": "other-model"},
    ],
)
def test_bind_result_to_manifest_rejects(_arm: tuple[Any, Any], update: dict[str, Any]) -> None:
    snapshot, _entries = _arm
    with pytest.raises(SuiteError):
        _bind_result_to_manifest(_tamper_result(snapshot.result, **update), snapshot.manifest)


def test_bind_result_rejects_failed_backend_gate(_arm: tuple[Any, Any]) -> None:
    snapshot, _entries = _arm
    assert snapshot.result.backend_ops is not None
    lying = snapshot.result.backend_ops.model_copy(update={"passed": False})
    with pytest.raises(SuiteError):
        _bind_result_to_manifest(
            _tamper_result(snapshot.result, backend_ops=lying), snapshot.manifest
        )


def test_bind_result_rejects_wrong_greedy_count(_arm: tuple[Any, Any]) -> None:
    snapshot, _entries = _arm
    assert snapshot.result.greedy is not None
    # The smoke manifest declares one greedy prompt; a projection with a different prompt
    # count cannot be a faithful projection of it.
    doubled = snapshot.result.greedy.model_copy(
        update={"prompts": snapshot.result.greedy.prompts * 2}
    )
    with pytest.raises(SuiteError):
        _bind_result_to_manifest(_tamper_result(snapshot.result, greedy=doubled), snapshot.manifest)


def test_bind_result_rejects_incomplete_schedule(_arm: tuple[Any, Any]) -> None:
    snapshot, _entries = _arm
    broken = snapshot.result.schedule.model_copy(update={"completed_measurements": 0})
    with pytest.raises(SuiteError):
        _bind_result_to_manifest(
            _tamper_result(snapshot.result, schedule=broken), snapshot.manifest
        )


def test_reauthenticate_samples_accepts_authentic(_arm: tuple[Any, Any]) -> None:
    snapshot, entries = _arm
    _reauthenticate_arm(snapshot, entries, snapshot.result)


def test_reauthenticate_rejects_reordered_samples(_arm: tuple[Any, Any]) -> None:
    snapshot, entries = _arm
    samples = list(snapshot.result.samples)
    samples[0], samples[1] = samples[1], samples[0]
    tampered = _tamper_result(snapshot.result, samples=tuple(samples))
    with pytest.raises(SuiteError):
        _reauthenticate_arm(snapshot, entries, tampered)


def test_reauthenticate_rejects_forged_sample_digest(_arm: tuple[Any, Any]) -> None:
    snapshot, entries = _arm
    forged = snapshot.result.samples[0].model_copy(update={"sample_sha256": "00" * 32})
    tampered = _tamper_result(snapshot.result, samples=(forged, *snapshot.result.samples[1:]))
    with pytest.raises(SuiteError):
        _reauthenticate_arm(snapshot, entries, tampered)


def test_reauthenticate_rejects_backend_projection_drift(_arm: tuple[Any, Any]) -> None:
    snapshot, entries = _arm
    assert snapshot.result.backend_ops is not None
    drifted = snapshot.result.backend_ops.model_copy(update={"gate_passed": False, "passed": False})
    tampered = _tamper_result(snapshot.result, backend_ops=drifted)
    with pytest.raises(SuiteError):
        _reauthenticate_arm(snapshot, entries, tampered)


def test_reauthenticate_rejects_greedy_projection_drift(_arm: tuple[Any, Any]) -> None:
    snapshot, entries = _arm
    assert snapshot.result.greedy is not None
    prompts = list(snapshot.result.greedy.prompts)
    prompts[0] = prompts[0].model_copy(update={"tokens_equal": False, "passed": False})
    drifted = snapshot.result.greedy.model_copy(update={"prompts": tuple(prompts)})
    tampered = _tamper_result(snapshot.result, greedy=drifted)
    with pytest.raises(SuiteError):
        _reauthenticate_arm(snapshot, entries, tampered)


def test_reauthenticate_rejects_measurement_drift(_arm: tuple[Any, Any]) -> None:
    snapshot, entries = _arm
    measurements = list(snapshot.result.measurements)
    measurements[0] = measurements[0].model_copy(update={"avg_ts": measurements[0].avg_ts + 1.0})
    tampered = _tamper_result(snapshot.result, measurements=tuple(measurements))
    with pytest.raises(SuiteError):
        _reauthenticate_arm(snapshot, entries, tampered)


@pytest.mark.parametrize("field", ["case", "inputs"])
def test_sample_contracts_reject_forged_case_or_inputs(_arm: tuple[Any, Any], field: str) -> None:
    snapshot, entries = _arm
    inputs = _bind_input_snapshots(snapshot.record, entries, snapshot.manifest, snapshot.result)
    parsed = _parsed_arm_samples(snapshot, entries)
    backend = parsed[0]
    if field == "case":
        forged = backend.model_copy(
            update={"case": backend.case.model_copy(update={"params_regex": "forged"})}
        )
    else:
        forged = backend.model_copy(
            update={
                "inputs": backend.inputs.model_copy(
                    update={"build_id": "build-sha256:" + "99" * 32}
                )
            }
        )
    parsed[0] = forged
    with pytest.raises(SuiteError):
        _bind_sample_contracts(
            snapshot.manifest, inputs, parsed, plan_performance(snapshot.manifest.performance)
        )


def test_sample_contracts_reject_failed_warmup(_arm: tuple[Any, Any]) -> None:
    snapshot, entries = _arm
    inputs = _bind_input_snapshots(snapshot.record, entries, snapshot.manifest, snapshot.result)
    parsed = _parsed_arm_samples(snapshot, entries)
    warmup_index = next(
        index
        for index, reference in enumerate(snapshot.result.samples)
        if reference.phase == "warmup"
    )
    warmup = parsed[warmup_index]
    parsed[warmup_index] = warmup.model_copy(
        update={"status": "parse-failed", "reason": "parse-failed"}
    )
    with pytest.raises(SuiteError):
        _bind_sample_contracts(
            snapshot.manifest, inputs, parsed, plan_performance(snapshot.manifest.performance)
        )


@pytest.mark.parametrize(
    ("adapter", "field", "value"),
    [
        ("backend-ops", "role", "samples"),
        ("llama-server", "media_type", "text/plain"),
    ],
)
def test_reauthenticate_rejects_sample_entry_role_or_media(
    _arm: tuple[Any, Any], adapter: str, field: str, value: str
) -> None:
    snapshot, entries = _arm
    actual_path = next(
        reference.logical_path
        for reference in snapshot.result.samples
        if reference.adapter == adapter
    )
    forged = dict(entries)
    forged[actual_path] = entries[actual_path].model_copy(update={field: value})
    with pytest.raises(SuiteError):
        _reauthenticate_arm(snapshot, forged, snapshot.result)


def test_measurement_coordinates_reject_duplicate(_arm: tuple[Any, Any]) -> None:
    snapshot, _entries = _arm
    doubled = (*snapshot.result.measurements, snapshot.result.measurements[0])
    with pytest.raises(SuiteError):
        _measurement_coordinates(
            snapshot.manifest, _tamper_result(snapshot.result, measurements=doubled)
        )


def test_measurement_coordinates_reject_missing(_arm: tuple[Any, Any]) -> None:
    snapshot, _entries = _arm
    dropped = snapshot.result.measurements[1:]
    with pytest.raises(SuiteError):
        _measurement_coordinates(
            snapshot.manifest, _tamper_result(snapshot.result, measurements=dropped)
        )


def test_measurement_coordinates_reject_wrong_reps(_arm: tuple[Any, Any]) -> None:
    snapshot, _entries = _arm
    measurements = list(snapshot.result.measurements)
    measurements[0] = measurements[0].model_copy(
        update={"samples_ts": measurements[0].samples_ts[:1]}
    )
    with pytest.raises(SuiteError):
        _measurement_coordinates(
            snapshot.manifest, _tamper_result(snapshot.result, measurements=tuple(measurements))
        )


def test_measurement_coordinates_reject_unknown_case(_arm: tuple[Any, Any]) -> None:
    snapshot, _entries = _arm
    measurements = list(snapshot.result.measurements)
    measurements[0] = measurements[0].model_copy(update={"case_id": "ghost-case"})
    with pytest.raises(SuiteError):
        _measurement_coordinates(
            snapshot.manifest, _tamper_result(snapshot.result, measurements=tuple(measurements))
        )


@pytest.mark.parametrize(
    "update",
    [
        {"build_id": "build-sha256:" + "99" * 32},
        {"model_id": "other-model"},
        {"machine_id": "other-machine"},
    ],
)
def test_input_snapshots_reject_misbound_payload(
    _arm: tuple[Any, Any], update: dict[str, Any]
) -> None:
    snapshot, entries = _arm
    with pytest.raises(SuiteError):
        _bind_input_snapshots(
            snapshot.record,
            entries,
            snapshot.manifest,
            _tamper_result(snapshot.result, **update),
        )


def test_input_snapshots_reject_misbound_reference(_arm: tuple[Any, Any]) -> None:
    snapshot, entries = _arm
    inputs = list(snapshot.result.inputs)
    inputs[0] = inputs[0].model_copy(update={"sha256": "00" * 32})
    with pytest.raises(SuiteError):
        _bind_input_snapshots(
            snapshot.record,
            entries,
            snapshot.manifest,
            _tamper_result(snapshot.result, inputs=tuple(inputs)),
        )


@pytest.mark.parametrize(
    ("field", "value"), [("role", "environment"), ("media_type", "text/plain")]
)
def test_input_snapshots_reject_entry_role_or_media(
    _arm: tuple[Any, Any], field: str, value: str
) -> None:
    snapshot, entries = _arm
    forged = dict(entries)
    entry = entries["suite/build.json"]
    forged["suite/build.json"] = entry.model_copy(update={field: value})
    with pytest.raises(SuiteError):
        _bind_input_snapshots(snapshot.record, forged, snapshot.manifest, snapshot.result)


def test_input_snapshot_parser_rejects_noncanonical_extra_and_malformed(
    _arm: tuple[Any, Any],
) -> None:
    snapshot, entries = _arm
    _entry, content = _snapshot_blob(snapshot.record, entries, "suite/build.json")
    inputs = _bind_input_snapshots(snapshot.record, entries, snapshot.manifest, snapshot.result)
    model = type(inputs.build)
    payload = inputs.build.model_dump(mode="json")
    with pytest.raises(SuiteError):
        _parse_canonical_json(
            content.rstrip(b"\n"),
            model,
            validation_error="invalid snapshot",
            canonical_error="noncanonical snapshot",
        )
    with pytest.raises(SuiteError):
        _parse_canonical_json(
            canonical_json_bytes({**payload, "extra": True}),
            model,
            validation_error="invalid snapshot",
            canonical_error="noncanonical snapshot",
        )
    with pytest.raises(SuiteError):
        _parse_canonical_json(
            b"{",
            model,
            validation_error="invalid snapshot",
            canonical_error="noncanonical snapshot",
        )


@pytest.mark.parametrize("kind", ["build", "model", "machine"])
def test_input_snapshot_rejects_invalid_embedded_digest(_arm: tuple[Any, Any], kind: str) -> None:
    snapshot, entries = _arm
    inputs = _bind_input_snapshots(snapshot.record, entries, snapshot.manifest, snapshot.result)
    build, model, machine = inputs.build, inputs.model, inputs.machine
    if kind == "build":
        build = build.model_copy(update={"canonical_record_sha256": "record-sha256:" + "00" * 32})
    elif kind == "model":
        model = model.model_copy(update={"model_receipt_sha256": "00" * 32})
    else:
        machine = machine.model_copy(update={"profile_sha256": "00" * 32})
    with pytest.raises(SuiteError):
        _authenticate_input_models(
            build,
            model,
            machine,
            snapshot.manifest,
            snapshot.result,
            model_sha256=inputs.model_sha256,
            machine_sha256=inputs.machine_sha256,
        )


@pytest.mark.parametrize(
    "kind",
    ["source-id", "source-commit", "toolchain", "gfx", "required-source", "required-feature"],
)
def test_suite_input_eligibility_rejects_self_consistent_forgery(
    _arm: tuple[Any, Any], kind: str
) -> None:
    snapshot, entries = _arm
    inputs = _bind_input_snapshots(snapshot.record, entries, snapshot.manifest, snapshot.result)
    canonical = inputs.build.canonical
    evidence = inputs.model.evidence
    if kind in {"source-id", "source-commit"}:
        source_evidence = dict(canonical.source.source_evidence)
        source_evidence["source_id" if kind == "source-id" else "base_commit"] = "forged"
        source = canonical.source.model_copy(update={"source_evidence": source_evidence})
        canonical = canonical.model_copy(update={"source": source})
    elif kind == "toolchain":
        canonical = canonical.model_copy(update={"toolchain_mode": "host"})
    elif kind == "gfx":
        selections = tuple(
            entry.model_copy(update={"value": "gfx9999"}) if entry.name == "gfx_targets" else entry
            for entry in canonical.selections
        )
        canonical = canonical.model_copy(update={"selections": selections})
    else:
        update = (
            {"required_sources": ("forged",)}
            if kind == "required-source"
            else {"required_features": ("forged",)}
        )
        evidence = evidence.model_copy(
            update={"execution": evidence.execution.model_copy(update=update)}
        )
    with pytest.raises(SuiteError):
        _validate_suite_input_eligibility(canonical, snapshot.manifest, evidence)


def test_snapshot_blob_rejects_missing_and_diverged(_arm: tuple[Any, Any]) -> None:
    snapshot, entries = _arm
    with pytest.raises(SuiteError):
        _snapshot_blob(snapshot.record, {}, "suite/result.json")
    entry = entries["suite/result.json"]
    lying = entry.model_copy(update={"size_bytes": entry.size_bytes + 1})
    with pytest.raises(SuiteError):
        _snapshot_blob(snapshot.record, {"suite/result.json": lying}, "suite/result.json")


# --- Preflight and post-allocation publication failure ------------------------


def test_preflight_rejects_oversized_output() -> None:
    from strixlab.evidence import MAX_MEMBER_BYTES
    from strixlab.judge import JudgeError

    oversized = _PortableOutput(
        "comparison/report.json",
        b'"' + b"a" * (MAX_MEMBER_BYTES + 1) + b'"',
        "application/json",
        "comparison",
    )
    with pytest.raises(JudgeError):
        _preflight_outputs((oversized,), environ={})


def test_preflight_rejects_secret_value_and_interpolation() -> None:
    from strixlab.judge import JudgeError

    for content in (b'{"value":"supersecretvalue"}\n', b'{"value":"${API_TOKEN}"}\n'):
        output = _PortableOutput(
            "comparison/report.json", content, "application/json", "comparison"
        )
        with pytest.raises(JudgeError):
            _preflight_outputs((output,), environ={"AWS_SECRET_ACCESS_KEY": "supersecretvalue"})


def test_secret_preflight_allocates_no_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import strixlab.judge as judge

    home, baseline_id, candidate_id = _two_arms(tmp_path, baseline=_UNIFORM, candidate=_UNIFORM)
    before = _count_runs(home)
    monkeypatch.setattr(judge, "render_report_markdown", lambda report: b"supersecretvalue\n")
    with pytest.raises(judge.JudgeError):
        compare_runs(
            baseline_id,
            candidate_id,
            home=home,
            environ={"AWS_SECRET_ACCESS_KEY": "supersecretvalue"},
        )
    assert _count_runs(home) == before


def test_post_allocation_failure_finalizes_and_reports(tmp_path: Path) -> None:
    home, baseline_id, candidate_id = _two_arms(tmp_path, baseline=_UNIFORM, candidate=_UNIFORM)

    def failing_publish(run: RunSession, outputs: Any) -> None:
        raise RunError("synthetic publication failure")

    before = _count_runs(home)
    with pytest.raises(JudgeExecutionError) as excinfo:
        compare_runs(
            baseline_id,
            candidate_id,
            home=home,
            environ={"PATH": "/usr/bin"},
            hooks=JudgeHooks(publish=failing_publish),
        )
    error = excinfo.value
    assert error.run_id and error.record is not None
    # A run was allocated and finalized failure; it carries no report entry.
    assert _count_runs(home) == before + 1
    entries = {entry.logical_path for entry in list_portable_entries(error.record)}
    assert "comparison/report.json" not in entries


# --- CLI ----------------------------------------------------------------------


def test_cli_compare_success(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from strixlab.cli import app

    home, baseline_id, candidate_id = _two_arms(tmp_path, baseline=_UNIFORM, candidate=_UNIFORM)
    result = CliRunner().invoke(app, ["compare", baseline_id, candidate_id, "--home", str(home)])
    assert result.exit_code == 0, result.output
    assert "comparison: run-" in result.stdout
    assert "verdict: inconclusive" in result.stdout
    assert "record:" in result.stdout


def test_cli_compare_pre_allocation_failure_reveals_no_id(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from strixlab.cli import app

    home, receipt_sha = _prepare_home(tmp_path)
    run = _run_suite(tmp_path, home, receipt_sha, _UNIFORM)
    result = CliRunner().invoke(
        app, ["compare", "run-does-not-exist", run.run_id, "--home", str(home)]
    )
    assert result.exit_code == 1
    assert "compare failed:" in result.stderr
    assert "comparison:" not in result.stderr


def test_cli_compare_pre_allocation_path_is_secret_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    import strixlab.cli as cli

    secret = "supersecretvalue"

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise JudgeLoadError(f"missing record beneath {tmp_path / secret}")

    monkeypatch.setattr(cli, "compare_runs", fail)
    result = CliRunner().invoke(
        cli.app,
        ["compare", "baseline", "candidate", "--home", str(tmp_path)],
        env={"AWS_SECRET_ACCESS_KEY": secret},
    )
    assert result.exit_code == 1
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert "unable to safely render terminal output" in result.stderr


def test_cli_compare_post_allocation_path_is_secret_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    import strixlab.cli as cli

    secret = "supersecretvalue"

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise JudgeExecutionError(
            "synthetic failure", run_id="run-safe", record=tmp_path / secret / "record"
        )

    monkeypatch.setattr(cli, "compare_runs", fail)
    result = CliRunner().invoke(
        cli.app,
        ["compare", "baseline", "candidate", "--home", str(tmp_path)],
        env={"AWS_SECRET_ACCESS_KEY": secret},
    )
    assert result.exit_code == 1
    assert secret not in result.stdout
    assert secret not in result.stderr
    assert "unable to safely render terminal output" in result.stderr


def test_cli_compare_post_allocation_failure_reports_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from typer.testing import CliRunner

    import strixlab.judge as judge
    from strixlab.cli import app

    home, baseline_id, candidate_id = _two_arms(tmp_path, baseline=_UNIFORM, candidate=_UNIFORM)

    def failing_publish(run: RunSession, outputs: Any) -> None:
        raise RunError("synthetic publication failure")

    monkeypatch.setattr(judge, "_default_publish", failing_publish)
    result = CliRunner().invoke(app, ["compare", baseline_id, candidate_id, "--home", str(home)])
    assert result.exit_code == 1
    assert "comparison: run-" in result.stderr
    assert "record:" in result.stderr
    assert "compare failed:" in result.stderr

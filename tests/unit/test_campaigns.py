from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

import strixlab.campaigns as c
from strixlab.judge import compare_case_samples, overall_verdict
from strixlab.locks import exclusive_lock

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "configs/campaigns/historical-mmvq-demo.yaml"


@pytest.fixture
def plan(tmp_path: Path) -> Path:
    raw = yaml.safe_load(EXAMPLE.read_text())
    for name in c._KINDS:
        original = EXAMPLE.parent / raw[name]
        path = tmp_path / f"{name}.yaml"
        path.write_bytes(original.read_bytes())
        raw[name] = path.name
    patch = tmp_path / "candidate.patch"
    patch.write_bytes((EXAMPLE.parent / raw["candidates"][0]["patches"][0]).read_bytes())
    raw["candidates"][0]["patches"] = [patch.name]
    raw["id"] = "test-campaign"
    path = tmp_path / "plan.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def change_plan(plan: Path, **updates: Any) -> None:
    raw = yaml.safe_load(plan.read_text())
    raw.update(updates)
    plan.write_text(yaml.safe_dump(raw))


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    return {"MODELS": str(tmp_path / "models"), "PATH": "/usr/bin"}


class FakeNative:
    """Typed API seams only: no shell, GPU, weights, network or actual builds."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.root = root
        self.calls: list[str] = []
        self.runs: dict[str, Any] = {}
        self.suite_count = 0
        self.prepare_count = 0
        self.compare_count = 0
        self.fail_stage: str | None = None
        self.fail_suite_at: int | None = None
        self.tokens_differ_at: int | None = None
        self.token_count_diff_at: int | None = None
        self.values = [100, 100, 100, 120, 120, 100, 100, 120, 120, 100]
        monkeypatch.setattr(c, "prepare_source", self.prepare)
        monkeypatch.setattr(c, "execute_cmake_build", self.build)
        monkeypatch.setattr(c, "verify_model_at_source", self.model)
        monkeypatch.setattr(c, "receipt_registry_sha256", lambda receipt: "a" * 64)
        monkeypatch.setattr(c, "load_model_receipt", lambda *a, **k: None)
        monkeypatch.setattr(c, "require_current_model", lambda receipt: None)
        monkeypatch.setattr(c, "run_suite", self.suite)
        monkeypatch.setattr(c, "inspect_run", self.inspect)
        monkeypatch.setattr(c, "load_finalized_suite_snapshot", self.snapshot)
        monkeypatch.setattr(c, "compare_runs", self.compare)

    def record(self, identifier: str) -> Path:
        path = self.root / (identifier + ".json")
        path.write_text(json.dumps({"id": identifier}))
        return path

    def prepare(self, *args: Any, **kwargs: Any) -> Any:
        # Every native side effect must see a durable phase reservation already.
        state = c.CampaignState.model_validate_json(
            (self.root / "campaigns/test-campaign/state.json").read_bytes()
        )
        assert state.phases[-1].status == "running"
        assert state.reserved_suite_runs >= 2
        self.calls.append("source")
        if self.fail_stage == "source":
            raise RuntimeError("SECRET must never be persisted")
        self.prepare_count += 1
        identifier = f"prep-{self.prepare_count}"
        return SimpleNamespace(
            evidence=SimpleNamespace(preparation_id=identifier), record=self.record(identifier)
        )

    def build(self, preparation: str, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("build")
        if self.fail_stage == "build":
            raise RuntimeError("SECRET must never be persisted")
        identifier = f"build-{preparation}"
        return SimpleNamespace(
            build_id=identifier, attempt=SimpleNamespace(record=self.record(identifier))
        )

    def model(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("model")
        if self.fail_stage == "model":
            raise RuntimeError("SECRET must never be persisted")
        return None

    def suite(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs["build_id"])
        self.suite_count += 1
        identifier = f"run-{self.suite_count}"
        self.runs[identifier] = SimpleNamespace(
            record=self.record(identifier),
            value=self.values[(self.suite_count - 1) % len(self.values)],
            tokens="b" if self.tokens_differ_at == self.suite_count else "a",
            token_count=2 if self.token_count_diff_at == self.suite_count else 1,
        )
        return SimpleNamespace(
            run_id=identifier,
            result=SimpleNamespace(
                status="failed" if self.fail_suite_at == self.suite_count else "passed",
                reason="backend_ops_failed",
            ),
        )

    def inspect(self, identifier: str, **kwargs: Any) -> Any:
        path = self.runs[identifier].record
        return SimpleNamespace(
            record=path, record_sha256=c._digest(path.read_bytes()), checksums_sha256="c" * 64
        )

    def snapshot(self, identifier: str, **kwargs: Any) -> Any:
        run = self.runs[identifier]
        return SimpleNamespace(
            result=SimpleNamespace(
                greedy=SimpleNamespace(
                    passed=True,
                    prompts=[
                        SimpleNamespace(
                            prompt_id="prompt",
                            responses=[
                                SimpleNamespace(
                                    token_count=run.token_count, tokens_sha256=run.tokens
                                )
                            ],
                        )
                    ],
                )
            )
        )

    def compare(self, baseline: str, candidate: str, **kwargs: Any) -> Any:
        self.compare_count += 1
        identifier = f"comparison-{self.compare_count}"
        self.runs[identifier] = SimpleNamespace(record=self.record(identifier))
        cases = tuple(
            compare_case_samples(
                case,
                (float(self.runs[baseline].value),) * 20,
                (float(self.runs[candidate].value),) * 20,
                baseline_record_sha256="record-sha256:" + "1" * 64,
                candidate_record_sha256="record-sha256:" + "2" * 64,
            )
            for case in ["pp512", "pp2048", "tg128"]
        )
        return SimpleNamespace(
            run_id=identifier,
            report=SimpleNamespace(
                overall_verdict=overall_verdict(tuple(case.verdict for case in cases)), cases=cases
            ),
        )


@pytest.fixture
def native(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeNative:
    home = tmp_path / "home"
    home.mkdir()
    return FakeNative(home, monkeypatch)


def run(plan: Path, env: dict[str, str], native: FakeNative) -> c.CampaignState:
    c.create_campaign(plan, home=native.root, environ=env)
    return c.resume_campaign("test-campaign", home=native.root, environ=env)


def test_example_is_valid_and_create_never_executes(tmp_path: Path, env: dict[str, str]) -> None:
    state = c.create_campaign(EXAMPLE, home=tmp_path / "home", environ=env)
    assert state.status == "ready" and state.reserved_suite_runs == 0
    assert state.objective_cases == ["pp512", "pp2048", "tg128"]
    assert c.inspect_campaign(state.id, home=tmp_path / "home") == state


def test_balanced_screening_and_fresh_confirmation_resume_idempotence(
    plan: Path,
    env: dict[str, str],
    native: FakeNative,
) -> None:
    state = run(plan, env, native)
    assert state.status == "completed" and state.reserved_suite_runs == 10
    assert [p.decision for p in state.phases] == [
        "inconclusive",
        "objective_met_provisional",
        "objective_met_provisional",
    ]
    assert native.suite_count == 10 and native.compare_count == 5
    assert native.calls.count("model") == 1
    for phase in state.phases[1:]:
        assert [link.role for link in phase.evidence if link.kind == "suite"] == [
            "baseline",
            "candidate",
            "candidate",
            "baseline",
        ]
    all_ids = [link.id for p in state.phases for link in p.evidence if link.kind == "suite"]
    assert len(set(all_ids)) == 10
    assert c.resume_campaign(state.id, home=native.root, environ=env) == state
    assert native.suite_count == 10
    report = c.render_campaign_report(state)
    assert "objective_met_provisional" in report and "positional samples" in report
    assert "pp2048" in report and str(native.root) not in report


@pytest.mark.parametrize("budget", [0, 1, 2, 5, 6, 9])
def test_budget_admission_before_work(
    plan: Path,
    env: dict[str, str],
    native: FakeNative,
    budget: int,
) -> None:
    change_plan(plan, max_suite_runs=budget)
    state = run(plan, env, native)
    expected = 0 if budget < 2 else 2 if budget < 6 else 6
    assert state.status == "budget_exhausted" and "no work admitted" in state.reason
    assert state.reserved_suite_runs == expected and native.suite_count == expected
    if budget < 2:
        assert native.calls == []
    assert c.resume_campaign(state.id, home=native.root, environ=env) == state


@pytest.mark.parametrize("stage", ["source", "build", "model"])
def test_failed_preparation_has_explicit_missing_link_and_no_retry(
    plan: Path,
    env: dict[str, str],
    native: FakeNative,
    stage: str,
) -> None:
    native.fail_stage = stage
    state = run(plan, env, native)
    phase = state.phases[0]
    assert state.status == "blocked" and state.reserved_suite_runs == 2
    assert phase.status == "failed" and f":{stage}:RuntimeError" in phase.decision
    assert phase.failure_evidence_link_unavailable
    if stage == "build":
        assert [link.kind for link in phase.evidence] == ["source"]
    assert "SECRET" not in state.model_dump_json()
    assert "lower layer may have retained evidence" in c.render_campaign_report(state)
    before = list(native.calls)
    assert c.resume_campaign(state.id, home=native.root, environ=env) == state
    assert native.calls == before


@pytest.mark.parametrize("at", [2, 4, 8])
@pytest.mark.parametrize("difference", ["tokens_differ_at", "token_count_diff_at"])
def test_cross_arm_tokens_gate_every_phase(
    plan: Path,
    env: dict[str, str],
    native: FakeNative,
    at: int,
    difference: str,
) -> None:
    setattr(native, difference, at)
    state = run(plan, env, native)
    assert state.phases[-1].decision == "correctness_failed:cross_arm_tokens"
    assert state.status == ("blocked" if at == 2 else "completed")
    assert native.compare_count == {2: 0, 4: 1, 8: 3}[at]
    assert state.reserved_suite_runs == {2: 2, 4: 6, 8: 10}[at]


@pytest.mark.parametrize("at", [1, 3, 7])
def test_failed_suite_is_preserved_not_compared(
    plan: Path,
    env: dict[str, str],
    native: FakeNative,
    at: int,
) -> None:
    native.fail_suite_at = at
    state = run(plan, env, native)
    assert state.phases[-1].decision.startswith("suite_failed:")
    assert state.phases[-1].evidence[-1].id == f"run-{at}"
    assert native.compare_count == {1: 0, 3: 1, 7: 3}[at]
    assert state.reserved_suite_runs == {1: 2, 3: 6, 7: 10}[at]


def test_bad_calibration_and_inconclusive_screen_are_not_promoted(
    plan: Path,
    env: dict[str, str],
    native: FakeNative,
) -> None:
    native.values[1] = 120
    state = run(plan, env, native)
    assert state.status == "blocked" and native.suite_count == 2


def test_inconclusive_screen_is_completed_evidence(
    plan: Path,
    env: dict[str, str],
    native: FakeNative,
) -> None:
    native.values = [100] * 10
    state = run(plan, env, native)
    assert state.status == "completed" and native.suite_count == 6
    assert state.phases[-1].decision == "objective_not_met"
    assert all(r.overall_verdict == "inconclusive" for r in state.phases[-1].comparisons)


@pytest.mark.parametrize("checkpoint", ["reserved", "after_suite", "terminal"])
def test_crash_checkpoint_semantics(
    plan: Path,
    env: dict[str, str],
    native: FakeNative,
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: str,
) -> None:
    c.create_campaign(plan, home=native.root, environ=env)
    save = c._save
    crashed = False

    def crash(root: Path, state: c.CampaignState) -> None:
        nonlocal crashed
        phase = state.phases[-1] if state.phases else None
        if not crashed and phase is not None:
            if checkpoint == "after_suite" and any(e.kind == "suite" for e in phase.evidence):
                crashed = True
                raise KeyboardInterrupt  # Returned evidence not checkpointed: remains unknown.
            if checkpoint == "reserved" and phase.stage == "reserved":
                save(root, state)
                crashed = True
                raise KeyboardInterrupt
            if (
                checkpoint == "terminal"
                and phase.phase == "confirmation"
                and phase.stage == "terminal"
            ):
                save(root, state)
                crashed = True
                raise KeyboardInterrupt
        save(root, state)

    monkeypatch.setattr(c, "_save", crash)
    with pytest.raises(KeyboardInterrupt):
        c.resume_campaign("test-campaign", home=native.root, environ=env)
    before = native.suite_count
    state = c.resume_campaign("test-campaign", home=native.root, environ=env)
    assert native.suite_count == before
    if checkpoint == "terminal":
        assert state.status == "completed" and state.reserved_suite_runs == 10
    else:
        assert state.status == "interrupted" and state.reserved_suite_runs == 2
        assert state.phases[-1].status == "interrupted"
        assert c.resume_campaign(state.id, home=native.root, environ=env) == state


@pytest.mark.parametrize(
    "drift", ["plan", "suite", "patch", "copied_patch", "environment", "evaluator", "policy"]
)
def test_drift_rejected_before_execution(
    plan: Path,
    env: dict[str, str],
    native: FakeNative,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    c.create_campaign(plan, home=native.root, environ=env)
    if drift in {"plan", "suite", "patch"}:
        target = {
            "plan": plan,
            "suite": plan.parent / "suite.yaml",
            "patch": plan.parent / "candidate.patch",
        }[drift]
        target.write_bytes(target.read_bytes() + b"\n")
    elif drift == "copied_patch":
        path = next((native.root / "campaigns/test-campaign").glob("*.patch"))
        path.write_bytes(path.read_bytes() + b"\n")
    elif drift == "environment":
        env["MODELS"] += "-changed"
    elif drift == "evaluator":
        monkeypatch.setattr(c, "_evaluator_digest", lambda: "f" * 64)
    else:
        root = native.root / "campaigns/test-campaign"
        _, state = c._load(root)
        state.objective_cases = ["tg128"]
        c._save(root, state)
    with pytest.raises(c.CampaignError, match="drift"):
        c.resume_campaign("test-campaign", home=native.root, environ=env)
    assert native.calls == []


def test_evidence_tampering_rejected_on_resume(
    plan: Path, env: dict[str, str], native: FakeNative
) -> None:
    state = run(plan, env, native)
    path = Path(state.phases[0].evidence[-1].record)
    path.write_text("tampered")
    with pytest.raises(c.CampaignError, match="evidence drift"):
        c.resume_campaign(state.id, home=native.root, environ=env)


def test_lock_refuses_second_writer_but_inspect_is_live(
    plan: Path,
    env: dict[str, str],
    native: FakeNative,
) -> None:
    state = c.create_campaign(plan, home=native.root, environ=env)
    with exclusive_lock(native.root / "campaigns/.test-campaign.lock") as lock:
        assert lock.acquired
        assert c.inspect_campaign(state.id, home=native.root) == state
        with pytest.raises(c.CampaignError, match="locked"):
            c.resume_campaign(state.id, home=native.root, environ=env)
    assert native.calls == []


@pytest.mark.parametrize(
    "updates",
    [
        {"objective_cases": []},
        {"objective_cases": ["tg128", "tg128"]},
        {"objective_cases": ["unknown"]},
        {"protected_regression_margin_percent": -1},
        {"protected_regression_margin_percent": 100},
        {"protected_regression_margin_percent": float("nan")},
        {"max_suite_runs": True},
        {"command": "echo unsafe"},
    ],
)
def test_invalid_policy_rejected(
    plan: Path, env: dict[str, str], native: FakeNative, updates: dict[str, Any]
) -> None:
    change_plan(plan, **updates)
    with pytest.raises((ValidationError, c.CampaignError)):
        c.create_campaign(plan, home=native.root, environ=env)
    assert native.calls == []


def test_duplicate_content_not_just_paths_rejected(
    plan: Path, env: dict[str, str], native: FakeNative
) -> None:
    (plan.parent / "different.patch").write_bytes((plan.parent / "candidate.patch").read_bytes())
    change_plan(
        plan,
        candidates=[
            {"id": "one", "patches": ["candidate.patch"]},
            {"id": "two", "patches": ["different.patch"]},
        ],
        max_candidates=2,
    )
    with pytest.raises(c.CampaignError, match="duplicate ordered"):
        c.create_campaign(plan, home=native.root, environ=env)


@pytest.mark.parametrize(
    "path", ["tests/test.c", "ggml/src/CMakeLists.txt", "ggml/src/../tests/check.cpp"]
)
def test_patch_scope_guard(plan: Path, env: dict[str, str], native: FakeNative, path: str) -> None:
    (plan.parent / "candidate.patch").write_text(
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
    )
    with pytest.raises(c.CampaignError, match="scope"):
        c.create_campaign(plan, home=native.root, environ=env)


@pytest.mark.parametrize(
    ("low", "margin", "expected"),
    [
        (-1.01, 1.0, False),
        (-1.0, 1.0, True),
        (-0.01, 0.0, False),
        (0.0, 0.0, True),
    ],
)
def test_protected_interval_boundary(low: float, margin: float, expected: bool) -> None:
    report = SimpleNamespace(
        cases=[
            SimpleNamespace(case_id="target", verdict="improvement", percent_ci_low=5.0),
            SimpleNamespace(case_id="protected", verdict="inconclusive", percent_ci_low=low),
        ]
    )
    assert c._objective_met(report, ["target"], margin) is expected


def test_protected_explicit_loss_does_not_rewrite_raw_judge() -> None:
    report = SimpleNamespace(
        overall_verdict="mixed",
        cases=[
            SimpleNamespace(case_id="target", verdict="improvement", percent_ci_low=5.0),
            SimpleNamespace(case_id="protected", verdict="regression", percent_ci_low=-0.8),
        ],
    )
    assert c._objective_met(report, ["target"], 1.0)
    assert not c._objective_met(report, ["target", "protected"], 1.0)
    assert report.overall_verdict == "mixed"


@pytest.mark.parametrize("phase_name", ["screening", "confirmation"])
def test_interrupted_candidate_is_spent_but_next_candidate_runs(
    plan: Path,
    env: dict[str, str],
    native: FakeNative,
    monkeypatch: pytest.MonkeyPatch,
    phase_name: str,
) -> None:
    second = plan.parent / "second.patch"
    second.write_bytes((plan.parent / "candidate.patch").read_bytes() + b"\n")
    change_plan(
        plan,
        max_candidates=2,
        max_suite_runs=18,
        candidates=[
            {"id": "first", "patches": ["candidate.patch"]},
            {"id": "second", "patches": ["second.patch"]},
        ],
    )
    c.create_campaign(plan, home=native.root, environ=env)
    original = c._execute_phase
    crashed = False

    def execute(*args: Any, **kwargs: Any) -> str:
        nonlocal crashed
        phase = args[3]
        if phase.candidate_id == "first" and phase.phase == phase_name and not crashed:
            crashed = True
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(c, "_execute_phase", execute)
    with pytest.raises(KeyboardInterrupt):
        c.resume_campaign("test-campaign", home=native.root, environ=env)
    state = c.resume_campaign("test-campaign", home=native.root, environ=env)
    interrupted = next(
        p for p in state.phases if p.candidate_id == "first" and p.phase == phase_name
    )
    assert interrupted.status == "interrupted" and interrupted.reserved_suite_runs == 4
    assert any(p.candidate_id == "second" and p.phase == "screening" for p in state.phases)
    assert state.status == "completed"
    before = list(native.calls)
    assert c.resume_campaign(state.id, home=native.root, environ=env) == state
    assert native.calls == before


def test_confirmation_reuses_screened_build_without_rebuilding(
    plan: Path,
    env: dict[str, str],
    native: FakeNative,
) -> None:
    state = run(plan, env, native)
    assert native.prepare_count == 2  # baseline + candidate; no confirmation re-prepare
    screening, confirmation = state.phases[1:]
    screened = next(e for e in screening.evidence if e.kind == "build")
    confirmed = next(e for e in confirmation.evidence if e.kind == "build")
    assert screened == confirmed
    assert native.calls.count(screened.id) == 4


def test_error_after_run_allocation_preserves_identifier(
    plan: Path,
    env: dict[str, str],
    native: FakeNative,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExecutionFailure(RuntimeError):
        run_id = "run-1"

    original = native.suite

    def suite(*args: Any, **kwargs: Any) -> Any:
        original(*args, **kwargs)
        raise ExecutionFailure

    monkeypatch.setattr(c, "run_suite", suite)
    state = run(plan, env, native)
    assert state.phases[0].evidence[-1].id == "run-1"
    assert not state.phases[0].failure_evidence_link_unavailable
    assert state.status == "blocked"


def test_unverifiable_run_id_remains_explicit(
    plan: Path,
    env: dict[str, str],
    native: FakeNative,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExecutionFailure(RuntimeError):
        run_id = "allocated-but-incomplete"

    def suite(*args: Any, **kwargs: Any) -> Any:
        raise ExecutionFailure

    monkeypatch.setattr(c, "run_suite", suite)
    state = run(plan, env, native)
    assert state.phases[0].unresolved_run_ids == ["allocated-but-incomplete"]
    assert state.phases[0].failure_evidence_link_unavailable


def test_objectives_and_raw_mixed_are_preserved_through_all_phases(
    plan: Path,
    env: dict[str, str],
    native: FakeNative,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    change_plan(plan, objective_cases=["tg128"], protected_regression_margin_percent=1.0)
    original = native.compare

    def compare(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if native.compare_count > 1:
            result.report.overall_verdict = "mixed"
            result.report.cases = tuple(
                case.model_copy(
                    update={
                        "verdict": "regression",
                        "percent_ci_low": -0.8,
                    }
                )
                if case.case_id != "tg128"
                else case
                for case in result.report.cases
            )
        return result

    monkeypatch.setattr(c, "compare_runs", compare)
    state = run(plan, env, native)
    assert [p.decision for p in state.phases[1:]] == ["objective_met_provisional"] * 2
    assert all(r.overall_verdict == "mixed" for p in state.phases[1:] for r in p.comparisons)
    report = c.render_campaign_report(state)
    assert "Objective cases: tg128" in report and "Protected regression margin: 1%" in report
    assert "Judge: `mixed`" in report


@pytest.mark.parametrize("comparison_index", [2, 3, 4, 5])
def test_protected_failure_in_any_comparison_withholds_acceptance(
    plan: Path,
    env: dict[str, str],
    native: FakeNative,
    monkeypatch: pytest.MonkeyPatch,
    comparison_index: int,
) -> None:
    change_plan(plan, objective_cases=["tg128"], protected_regression_margin_percent=1.0)
    original = native.compare

    def compare(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if native.compare_count == comparison_index:
            result.report.cases = tuple(
                case.model_copy(
                    update={
                        "verdict": "inconclusive",
                        "percent_ci_low": -1.01,
                    }
                )
                if case.case_id == "pp512"
                else case
                for case in result.report.cases
            )
        return result

    monkeypatch.setattr(c, "compare_runs", compare)
    state = run(plan, env, native)
    assert state.phases[-1].decision != "objective_met_provisional"
    assert len(state.phases) == (2 if comparison_index < 4 else 3)


def test_create_id_collision_and_symlink_state_root(
    plan: Path,
    env: dict[str, str],
    native: FakeNative,
    tmp_path: Path,
) -> None:
    c.create_campaign(plan, home=native.root, environ=env)
    with pytest.raises(c.CampaignError, match="already exists"):
        c.create_campaign(plan, home=native.root, environ=env)
    other = tmp_path / "link-home"
    other.symlink_to(native.root, target_is_directory=True)
    with pytest.raises(c.CampaignError, match="symlinked"):
        c.inspect_campaign("test-campaign", home=other)
    with pytest.raises(c.CampaignError, match="invalid campaign id"):
        c.inspect_campaign("../escape", home=native.root)


def test_controller_with_real_finalized_suite_and_judge_evidence(
    plan: Path,
    env: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Integration at real suite/judge seams; only hardware/build/model work is fake."""
    import contextlib
    import tempfile

    import _suite_fixtures as fx

    from strixlab.models import load_model_receipt
    from strixlab.suites import SuiteHooks, run_suite

    home = tmp_path / "real-evidence-home"
    home.mkdir(mode=0o700)
    fx.stub_cache_verification(monkeypatch)
    fx.make_present_build(home)
    receipt_sha = fx.publish_receipt(home, tmp_path / "scratch")
    receipt = load_model_receipt("qwen35-4b-smoke", receipt_sha, home=home)

    def prepare(*args: Any, **kwargs: Any) -> Any:
        return "fixture-preparation", fx.BUILD_ID

    @contextlib.contextmanager
    def machine_lock(path: Path) -> Any:
        from strixlab.locks import LockAttempt, LockStatus

        yield LockAttempt(LockStatus.ACQUIRED, path)

    hooks = SuiteHooks(
        temp_root_factory=lambda: Path(tempfile.mkdtemp(dir=tmp_path)),
        machine_lock=machine_lock,
        backend_ops=fx.fake_backend(),
        llama_server=fx.fake_server(),
        llama_bench=fx.fake_bench(),
    )

    def suite(*args: Any, **kwargs: Any) -> Any:
        return run_suite(*args, **kwargs, hooks=hooks)

    monkeypatch.setattr(c, "_prepare", prepare)
    monkeypatch.setattr(c, "verify_model_at_source", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(c, "run_suite", suite)
    c.create_campaign(plan, home=home, environ=env)
    state = c.resume_campaign("test-campaign", home=home, environ=env)
    assert state.status == "completed"
    assert [p.decision for p in state.phases] == ["inconclusive", "objective_not_met"]
    assert (
        len([link for phase in state.phases for link in phase.evidence if link.kind == "suite"])
        == 6
    )
    assert c.inspect_campaign(state.id, home=home) == state
    # Authentication is real here, including records, checksums and suite snapshots.
    suite_link = next(link for link in state.phases[0].evidence if link.kind == "suite")
    assert c._tokens(c.load_finalized_suite_snapshot(suite_link.id, home=home))


def test_confirmation_cannot_reuse_prior_suite_evidence(
    plan: Path,
    env: dict[str, str],
    native: FakeNative,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = native.suite

    def suite(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        if native.suite_count == 7:
            result.run_id = "run-1"  # Authentic old evidence still is not fresh confirmation.
        return result

    monkeypatch.setattr(c, "run_suite", suite)
    state = run(plan, env, native)
    assert state.phases[-1].status == "failed"
    assert state.phases[-1].decision == "execution_failed:suite:CampaignError"
    assert native.compare_count == 3
    assert not any(e.kind == "suite" for e in state.phases[-1].evidence)

"""Finite, restart-safe evaluation of reviewed patches using the existing smoke suite.

AB/BA balances whole-suite order only. The unchanged judge's positional samples
are not true temporal pairs, and repeated screening is not a campaign confidence
interval. Exact cross-arm tokens restrict v1 to launch/layout-preserving changes.
The patch scope guard is not a sandbox against malicious native code.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import uuid
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from strixlab.cmake_build import execute_cmake_build
from strixlab.config import parse_manifest_text
from strixlab.evidence import inspect_run
from strixlab.judge import POLICY_ID, ComparisonReportV1, compare_runs
from strixlab.locks import exclusive_lock
from strixlab.manifests import (
    BuildProfileV1,
    MachineProfileV1,
    ModelManifestV1,
    SourceLockV1,
    SuiteManifestV1,
    resolve_and_validate_manifest,
)
from strixlab.models import (
    load_model_receipt,
    receipt_registry_sha256,
    require_current_model,
    verify_model_at_source,
)
from strixlab.secure_fs import fsync_directory, readonly_open_flags, write_exclusive
from strixlab.serialization import canonical_json_bytes
from strixlab.sources import prepare_source
from strixlab.suites import FinalizedSuiteSnapshot, load_finalized_suite_snapshot, run_suite

Id = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=64)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class CampaignError(ValueError):
    """Campaign admission, integrity or locking failed before further work."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class CampaignCandidateV1(_Model):
    id: Id
    patches: list[str] = Field(min_length=1, max_length=64)


class CampaignPlanV1(_Model):
    schema_version: Literal[1]
    id: Id
    suite: str
    machine: str
    build: str
    source: str
    model: str
    candidates: list[CampaignCandidateV1] = Field(min_length=1, max_length=100)
    objective_cases: list[Id] | None = Field(default=None, min_length=1)
    protected_regression_margin_percent: float = Field(default=0.0, ge=0, lt=100)
    max_candidates: int = Field(ge=1, le=100)
    max_suite_runs: int = Field(ge=0, le=10000)

    @model_validator(mode="after")
    def check_candidates(self) -> CampaignPlanV1:
        if self.objective_cases is not None and len(set(self.objective_cases)) != len(
            self.objective_cases
        ):
            raise ValueError("objective_cases must be unique")
        ids = [candidate.id for candidate in self.candidates]
        if len(ids) != len(set(ids)) or "baseline" in ids:
            raise ValueError("candidate ids must be unique and cannot be baseline")
        if len(ids) > self.max_candidates:
            raise ValueError("candidate list exceeds max_candidates")
        return self


class InputFile(_Model):
    path: str
    sha256: Digest


class FrozenCandidate(_Model):
    id: Id
    identity: Digest
    patches: list[InputFile]


class FrozenCampaign(_Model):
    schema_version: Literal[1] = 1
    plan: CampaignPlanV1
    inputs: dict[str, InputFile]
    resolved: dict[str, dict[str, Any]]
    candidates: list[FrozenCandidate]
    evaluator_sha256: Digest
    judge_policy: str


class EvidenceLink(_Model):
    kind: Literal["source", "build", "suite", "comparison"]
    id: str
    role: str
    record: str
    record_sha256: str
    checksums_sha256: str | None = None


class ComparisonDecision(_Model):
    run_id: str
    overall_verdict: str
    objective_met: bool
    cases: list[dict[str, Any]]


class CampaignPhase(_Model):
    candidate_id: str
    phase: Literal["calibration", "screening", "confirmation"]
    reserved_suite_runs: int
    status: Literal["running", "completed", "failed", "interrupted"] = "running"
    stage: str = "reserved"
    decision: str | None = None
    evidence: list[EvidenceLink] = Field(default_factory=list)
    comparisons: list[ComparisonDecision] = Field(default_factory=list)
    # Exceptions may name an allocated run which could not be authenticated.
    unresolved_run_ids: list[str] = Field(default_factory=list)
    failure_evidence_link_unavailable: bool = False


class Baseline(_Model):
    preparation_id: str
    build_id: str
    model_receipt_sha256: Digest


class CampaignState(_Model):
    schema_version: Literal[1] = 1
    id: Id
    frozen_sha256: Digest
    status: Literal["ready", "running", "completed", "blocked", "interrupted", "budget_exhausted"]
    reason: str
    max_suite_runs: int
    objective_cases: list[str] = Field(default_factory=list)
    protected_regression_margin_percent: float = 0.0
    reserved_suite_runs: int = 0
    baseline: Baseline | None = None
    phases: list[CampaignPhase] = Field(default_factory=list)


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read(path: Path) -> bytes:
    fd = os.open(path, readonly_open_flags())
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise CampaignError("campaign input is not a regular file")
        return stream.read()


def _directory(path: Path, *, create: bool = False) -> None:
    # Reject symlinked ancestors as well as the final directory. Local state is private.
    for part in reversed([path, *path.parents]):
        if part.is_symlink():
            raise CampaignError("symlinked campaign directory")
        if create and not part.exists():
            part.mkdir(mode=0o700)
            fsync_directory(part.parent)
    if not path.is_dir() or path.stat().st_uid != os.geteuid():
        raise CampaignError("campaign directory is missing or not owned by this user")


def _root(home: Path, campaign_id: str, *, create: bool = False) -> Path:
    if re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", campaign_id) is None or len(campaign_id) > 64:
        raise CampaignError("invalid campaign id")
    parent = home.absolute() / "campaigns"
    _directory(parent, create=create)
    return parent / campaign_id


def _save(root: Path, state: CampaignState) -> None:
    temporary = root / f".state-{uuid.uuid4().hex}.tmp"
    write_exclusive(temporary, canonical_json_bytes(state.model_dump(mode="json")))
    os.replace(temporary, root / "state.json")
    fsync_directory(root)


def _evaluator_digest() -> str:
    package = Path(__file__).parent
    files = sorted(p for p in package.rglob("*") if p.suffix in {".py", ".json"})
    return _digest(
        canonical_json_bytes({str(p.relative_to(package)): _digest(_read(p)) for p in files})
    )


def _patch_scope(content: bytes) -> None:
    """Accept ordinary textual modifications of ggml source files only.

    Renames, copies, binary patches, modes and quoted paths are deliberately not
    supported. Git still validates and applies the actual patch in prepare_source.
    """
    text = content.decode("utf-8")
    paths = []
    for line in text.splitlines():
        if line.startswith("diff --git "):
            match = re.fullmatch(r"diff --git a/([^\s]+) b/([^\s]+)", line)
            if match is None or match[1] != match[2]:
                raise CampaignError("patch must modify an existing unrenamed source file")
            paths.append(match[1])
        elif line.startswith(("--- ", "+++ ")):
            if not paths or line[4:] not in {"a/" + paths[-1], "b/" + paths[-1]}:
                raise CampaignError("patch file headers do not match")
        elif line.startswith(
            (
                "rename ",
                "copy ",
                "old mode",
                "new mode",
                "new file",
                "deleted file",
                "GIT binary",
                "Binary files",
            )
        ):
            raise CampaignError("patch operation is outside campaign scope")
    if not paths:
        raise CampaignError("patch has no git file headers")
    for name in paths:
        path = PurePosixPath(name)
        if (
            not name.startswith("ggml/src/")
            or ".." in path.parts
            or "\\" in name
            or path.suffix not in {".c", ".cc", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".hip"}
        ):
            raise CampaignError("candidate patch is outside ggml/src source-file scope")


_KINDS = {
    "suite": "suite",
    "machine": "machine",
    "build": "build",
    "source": "source-lock",
    "model": "model",
}


def _resolve_inputs(
    plan: CampaignPlanV1, base: Path, environ: Mapping[str, str]
) -> tuple[dict[str, InputFile], dict[str, dict[str, Any]]]:
    inputs = {}
    resolved = {}
    for name, kind in _KINDS.items():
        path = (base / getattr(plan, name)).absolute()
        content = _read(path)
        inputs[name] = InputFile(path=str(path), sha256=_digest(content))
        model = resolve_and_validate_manifest(kind, parse_manifest_text(content.decode()), environ)
        resolved[name] = model.model_dump(mode="json")
    suite = SuiteManifestV1.model_validate(resolved["suite"])
    source = SourceLockV1.model_validate(resolved["source"])
    case_ids = {case.id for case in suite.performance.cases}
    if plan.objective_cases is not None and not set(plan.objective_cases) <= case_ids:
        raise CampaignError("objective_cases contains unknown suite performance cases")
    if (
        suite.machine != resolved["machine"]["id"]
        or suite.model != resolved["model"]["id"]
        or suite.build.source_id != source.id
        or suite.build.source_commit != source.commit
    ):
        raise CampaignError("campaign manifests do not describe the same frozen scenario")
    return inputs, resolved


def create_campaign(plan_path: Path, *, home: Path, environ: Mapping[str, str]) -> CampaignState:
    """Freeze a finite reviewed plan; never prepare source, build or run here."""
    plan_bytes = _read(plan_path)
    plan = CampaignPlanV1.model_validate(parse_manifest_text(plan_bytes.decode()))
    inputs, resolved = _resolve_inputs(plan, plan_path.absolute().parent, environ)
    inputs["plan"] = InputFile(path=str(plan_path.absolute()), sha256=_digest(plan_bytes))
    candidates = []
    blobs: dict[str, bytes] = {}
    identities = set()
    for candidate in plan.candidates:
        patches = []
        for index, locator in enumerate(candidate.patches):
            path = (plan_path.absolute().parent / locator).absolute()
            content = _read(path)
            _patch_scope(content)
            digest = _digest(content)
            name = f"{candidate.id}-{index}.patch"
            blobs[name] = content
            patches.append(InputFile(path=name, sha256=digest))
            inputs[f"patch:{candidate.id}:{index}"] = InputFile(path=str(path), sha256=digest)
        identity = _digest(canonical_json_bytes([resolved["source"], [p.sha256 for p in patches]]))
        if identity in identities:
            raise CampaignError("duplicate ordered candidate patch content")
        identities.add(identity)
        candidates.append(FrozenCandidate(id=candidate.id, identity=identity, patches=patches))
    frozen = FrozenCampaign(
        plan=plan,
        inputs=inputs,
        resolved=resolved,
        candidates=candidates,
        evaluator_sha256=_evaluator_digest(),
        judge_policy=POLICY_ID,
    )
    frozen_bytes = canonical_json_bytes(frozen.model_dump(mode="json"))
    root = _root(home, plan.id, create=True)
    with exclusive_lock(root.parent / f".{plan.id}.lock") as lock:
        if not lock.acquired:
            raise CampaignError("campaign is locked")
        if root.exists() or root.is_symlink():
            raise CampaignError("campaign id already exists; use a new id")
        root.mkdir(mode=0o700)
        fsync_directory(root.parent)
        write_exclusive(root / "frozen.json", frozen_bytes)
        for name, blob in blobs.items():
            write_exclusive(root / name, blob)
        state = CampaignState(
            id=plan.id,
            frozen_sha256=_digest(frozen_bytes),
            status="ready",
            reason="created; no execution admitted",
            max_suite_runs=plan.max_suite_runs,
            objective_cases=plan.objective_cases
            or [c["id"] for c in resolved["suite"]["performance"]["cases"]],
            protected_regression_margin_percent=plan.protected_regression_margin_percent,
        )
        _save(root, state)
        return state


def _load(root: Path) -> tuple[FrozenCampaign, CampaignState]:
    _directory(root)
    state = CampaignState.model_validate_json(_read(root / "state.json"))
    raw = _read(root / "frozen.json")
    if _digest(raw) != state.frozen_sha256:
        raise CampaignError("frozen campaign digest mismatch")
    frozen = FrozenCampaign.model_validate_json(raw)
    if (
        state.id != root.name
        or state.id != frozen.plan.id
        or state.max_suite_runs != frozen.plan.max_suite_runs
    ):
        raise CampaignError("campaign state identity mismatch")
    objectives = frozen.plan.objective_cases or [
        case["id"] for case in frozen.resolved["suite"]["performance"]["cases"]
    ]
    if (
        state.objective_cases != objectives
        or state.protected_regression_margin_percent
        != frozen.plan.protected_regression_margin_percent
    ):
        raise CampaignError("campaign objective policy drift")
    if state.reserved_suite_runs > state.max_suite_runs or any(
        p.reserved_suite_runs != (2 if p.phase == "calibration" else 4) for p in state.phases
    ):
        raise CampaignError("campaign reservation limit mismatch")
    if len({(p.candidate_id, p.phase) for p in state.phases}) != len(state.phases):
        raise CampaignError("duplicate campaign phase")
    if state.reserved_suite_runs != sum(p.reserved_suite_runs for p in state.phases):
        raise CampaignError("campaign budget ledger mismatch")
    return frozen, state


def _verify_links(state: CampaignState, home: Path) -> None:
    for phase in state.phases:
        for link in phase.evidence:
            if link.kind in {"suite", "comparison"}:
                inspection = inspect_run(link.id, home=home)
                if (
                    inspection.record_sha256 != link.record_sha256
                    or inspection.checksums_sha256 != link.checksums_sha256
                ):
                    raise CampaignError("campaign run evidence drift")
            elif _digest(_read(Path(link.record))) != link.record_sha256:
                raise CampaignError("campaign preparation/build evidence drift")


def _check_drift(root: Path, frozen: FrozenCampaign, environ: Mapping[str, str]) -> None:
    if frozen.evaluator_sha256 != _evaluator_digest() or frozen.judge_policy != POLICY_ID:
        raise CampaignError("campaign evaluator drift; create a new campaign")
    for item in frozen.inputs.values():
        if _digest(_read(Path(item.path))) != item.sha256:
            raise CampaignError("campaign input file drift")
    _, resolved = _resolve_inputs(frozen.plan, Path(frozen.inputs["plan"].path).parent, environ)
    if resolved != frozen.resolved:
        raise CampaignError("campaign resolved input drift")
    for candidate in frozen.candidates:
        for patch in candidate.patches:
            if _digest(_read(root / patch.path)) != patch.sha256:
                raise CampaignError("campaign frozen patch drift")


def inspect_campaign(campaign_id: str, *, home: Path) -> CampaignState:
    """Authenticate persisted evidence without executing or rewriting a campaign."""
    root = _root(home, campaign_id)
    # Atomic state replacement allows progress inspection while the writer holds
    # its campaign lock. Only fully checkpointed immutable evidence is read.
    _, state = _load(root)
    _verify_links(state, home)
    return state


def _file_link(
    kind: Literal["source", "build"], identifier: str, role: str, record: Path
) -> EvidenceLink:
    return EvidenceLink(
        kind=kind,
        id=identifier,
        role=role,
        record=str(record),
        record_sha256=_digest(_read(record)),
    )


def _run_link(identifier: str, role: str, home: Path, *, comparison: bool = False) -> EvidenceLink:
    result = inspect_run(identifier, home=home)
    return EvidenceLink(
        kind="comparison" if comparison else "suite",
        id=identifier,
        role=role,
        record=str(result.record),
        record_sha256=result.record_sha256,
        checksums_sha256=result.checksums_sha256,
    )


def _tokens(snapshot: FinalizedSuiteSnapshot) -> dict[str, list[tuple[int, str]]]:
    greedy = snapshot.result.greedy
    if greedy is None or not greedy.passed or not greedy.prompts:
        raise CampaignError("missing successful greedy correctness evidence")
    return {
        prompt.prompt_id: [(r.token_count, r.tokens_sha256) for r in prompt.responses]
        for prompt in greedy.prompts
    }


def _prepare(
    root: Path,
    frozen: FrozenCampaign,
    state: CampaignState,
    phase: CampaignPhase,
    home: Path,
    candidate: FrozenCandidate | None,
) -> tuple[str, str]:
    role = candidate.id if candidate else "baseline"
    phase.stage = "source"
    _save(root, state)
    prepared = prepare_source(
        SourceLockV1.model_validate(frozen.resolved["source"]),
        home=home,
        patches=[root / p.path for p in candidate.patches] if candidate else [],
    )
    preparation_id = prepared.evidence.preparation_id
    phase.evidence.append(_file_link("source", preparation_id, role, prepared.record))
    phase.stage = "build"
    _save(root, state)
    built = execute_cmake_build(
        preparation_id, BuildProfileV1.model_validate(frozen.resolved["build"]), home=home
    )
    phase.evidence.append(_file_link("build", built.build_id, role, built.attempt.record))
    _save(root, state)
    return preparation_id, built.build_id


def _objective_met(report: ComparisonReportV1, objectives: list[str], margin: float) -> bool:
    return all(
        case.verdict == "improvement"
        if case.case_id in objectives
        else case.percent_ci_low >= -margin
        for case in report.cases
    ) and set(objectives) <= {case.case_id for case in report.cases}


def _execute_phase(
    root: Path,
    frozen: FrozenCampaign,
    state: CampaignState,
    phase: CampaignPhase,
    home: Path,
    environ: Mapping[str, str],
    candidate: FrozenCandidate | None,
) -> str:
    if state.baseline is None:
        preparation, build = _prepare(root, frozen, state, phase, home, None)
        phase.stage = "model"
        _save(root, state)
        receipt = verify_model_at_source(
            ModelManifestV1.model_validate(frozen.resolved["model"]), preparation, home=home
        )
        state.baseline = Baseline(
            preparation_id=preparation,
            build_id=build,
            model_receipt_sha256=receipt_registry_sha256(receipt),
        )
        _save(root, state)
    baseline = state.baseline
    candidate_build = baseline.build_id
    if candidate is not None:
        if phase.phase == "confirmation":
            screening = next(
                p for p in state.phases if p.candidate_id == candidate.id and p.phase == "screening"
            )
            link = next(e for e in screening.evidence if e.kind == "build")
            # Reuse the exact screened artifact. run_suite authenticates its build
            # lease on every execution; confirmation never silently rebuilds it.
            candidate_build = link.id
            phase.evidence.append(link.model_copy())
            _save(root, state)
        else:
            _, candidate_build = _prepare(root, frozen, state, phase, home, candidate)
    suite = SuiteManifestV1.model_validate(frozen.resolved["suite"])
    machine = MachineProfileV1.model_validate(frozen.resolved["machine"])
    # Reserve 2 for calibration, 4 for each candidate phase, including all failures.
    order = (
        ["baseline", "candidate"]
        if candidate is None
        else ["baseline", "candidate", "candidate", "baseline"]
    )
    runs = []
    for role in order:
        _check_drift(root, frozen, environ)
        phase.stage = "suite"
        _save(root, state)
        result = run_suite(
            suite,
            _read(Path(frozen.inputs["suite"].path)),
            machine_profile=machine,
            build_id=baseline.build_id if role == "baseline" else candidate_build,
            local_receipt_sha256=baseline.model_receipt_sha256,
            home=home,
            environ=environ,
        )
        if any(
            link.id == result.run_id
            for previous in state.phases
            for link in previous.evidence
            if link.kind == "suite"
        ):
            raise CampaignError("suite run id was reused; fresh evidence is required")
        phase.evidence.append(_run_link(result.run_id, role, home))
        _save(root, state)
        if result.result.status != "passed":
            return "suite_failed:" + str(result.result.reason)
        runs.append(result.run_id)
    if len(set(runs)) != len(runs):
        raise CampaignError("suite executions did not produce independent fresh run ids")
    verdicts = []
    for left, right in [(0, 1)] if candidate is None else [(0, 1), (3, 2)]:
        phase.stage = "comparison"
        _save(root, state)
        baseline_snapshot = load_finalized_suite_snapshot(runs[left], home=home)
        candidate_snapshot = load_finalized_suite_snapshot(runs[right], home=home)
        if _tokens(baseline_snapshot) != _tokens(candidate_snapshot):
            return "correctness_failed:cross_arm_tokens"
        compared = compare_runs(runs[left], runs[right], home=home, environ=environ)
        phase.evidence.append(
            _run_link(compared.run_id, "baseline-candidate", home, comparison=True)
        )
        _save(root, state)
        met = _objective_met(
            compared.report, state.objective_cases, state.protected_regression_margin_percent
        )
        phase.comparisons.append(
            ComparisonDecision(
                run_id=compared.run_id,
                overall_verdict=compared.report.overall_verdict,
                objective_met=met,
                cases=[c.model_dump(mode="json") for c in compared.report.cases],
            )
        )
        _save(root, state)
        verdicts.append(
            compared.report.overall_verdict
            if candidate is None
            else "objective_met_provisional"
            if met
            else "objective_not_met"
        )
    return verdicts[0] if len(set(verdicts)) == 1 else "mixed"


def _phase(
    root: Path,
    frozen: FrozenCampaign,
    state: CampaignState,
    home: Path,
    environ: Mapping[str, str],
    candidate: FrozenCandidate | None,
    name: Literal["calibration", "screening", "confirmation"],
) -> bool:
    count = 2 if name == "calibration" else 4
    if state.reserved_suite_runs + count > state.max_suite_runs:
        state.status = "budget_exhausted"
        state.reason = f"insufficient suite slots for {name}; no work admitted"
        _save(root, state)
        return False
    _check_drift(root, frozen, environ)
    phase = CampaignPhase(
        candidate_id=candidate.id if candidate else "baseline",
        phase=name,
        reserved_suite_runs=count,
    )
    state.phases.append(phase)
    state.reserved_suite_runs += count
    state.status = "running"
    state.reason = f"{name} reserved"
    _save(root, state)  # Durable reservation BEFORE every side effect.
    try:
        decision = _execute_phase(root, frozen, state, phase, home, environ, candidate)
    except Exception as exc:
        # Do not persist arbitrary exception text (may contain paths/secrets).
        run_id = getattr(exc, "run_id", None)
        if isinstance(run_id, str):
            try:
                phase.evidence.append(
                    _run_link(run_id, "exception", home, comparison=phase.stage == "comparison")
                )
            except Exception:
                phase.unresolved_run_ids.append(run_id)
        if not isinstance(run_id, str) or phase.unresolved_run_ids:
            phase.failure_evidence_link_unavailable = True
        decision = f"execution_failed:{phase.stage}:{type(exc).__name__}"
        phase.status = "failed"
    else:
        phase.status = "completed"
    phase.decision = decision
    phase.stage = "terminal"
    _save(root, state)  # Evidence and decision commit together before promotion.
    return True


def resume_campaign(campaign_id: str, *, home: Path, environ: Mapping[str, str]) -> CampaignState:
    """Admit bounded phases. Unknown interrupted work is spent, never replayed."""
    root = _root(home, campaign_id)
    with exclusive_lock(root.parent / f".{campaign_id}.lock") as lock:
        if not lock.acquired:
            raise CampaignError("campaign is locked")
        frozen, state = _load(root)
        _verify_links(state, home)
        _check_drift(root, frozen, environ)
        pending = [p for p in state.phases if p.status == "running"]
        if pending:
            for phase in pending:
                phase.status = "interrupted"
                phase.decision = "interrupted:unknown_outcome_slots_spent"
                phase.stage = "terminal"
            if any(p.phase == "calibration" for p in pending):
                state.status = "interrupted"
                state.reason = "calibration interrupted; start a new campaign for further attempts"
                _save(root, state)
                return state
            state.status = "running"
            state.reason = "interrupted candidate is spent; continuing untouched candidates"
            _save(root, state)
        if state.status in {"completed", "blocked", "interrupted", "budget_exhausted"}:
            return state
        if state.baseline is not None:
            receipt = load_model_receipt(
                frozen.resolved["model"]["id"], state.baseline.model_receipt_sha256, home=home
            )
            if receipt_registry_sha256(receipt) != state.baseline.model_receipt_sha256:
                raise CampaignError("model receipt drift")
            require_current_model(receipt)
        if not state.phases and not _phase(root, frozen, state, home, environ, None, "calibration"):
            return state
        if state.phases[0].decision != "inconclusive":
            state.status = "blocked"
            state.reason = "baseline calibration did not yield token-matching inconclusive evidence"
            _save(root, state)
            return state
        for candidate in frozen.candidates:
            screening = next(
                (
                    p
                    for p in state.phases
                    if p.candidate_id == candidate.id and p.phase == "screening"
                ),
                None,
            )
            if screening is None:
                if not _phase(root, frozen, state, home, environ, candidate, "screening"):
                    return state
                screening = state.phases[-1]
            if screening.decision != "objective_met_provisional":
                continue
            if not any(
                p.candidate_id == candidate.id and p.phase == "confirmation" for p in state.phases
            ) and not _phase(root, frozen, state, home, environ, candidate, "confirmation"):
                return state
        state.status = "completed"
        state.reason = "finite candidate list evaluated; improvements are provisional, never merged"
        _save(root, state)
        return state


def render_campaign_report(state: CampaignState) -> str:
    """Portable summary for planning the next separately reviewed campaign."""
    lines = [
        f"# Campaign {state.id}",
        "",
        f"Status: {state.status}",
        f"Reason: {state.reason}",
        f"Reserved suite slots: {state.reserved_suite_runs}/{state.max_suite_runs}",
        "Objective cases: " + ", ".join(state.objective_cases),
        f"Protected regression margin: {state.protected_regression_margin_percent:g}%",
        "",
        "| Candidate | Phase | Status | Decision |",
        "| --- | --- | --- | --- |",
    ]
    for phase in state.phases:
        lines.append(
            f"| {phase.candidate_id} | {phase.phase} | {phase.status} | "
            f"{phase.decision or 'pending'} |"
        )
    for phase in state.phases:
        lines.extend(["", f"## {phase.candidate_id}: {phase.phase}"])
        for comparison in phase.comparisons:
            lines.append(
                f"\nJudge: `{comparison.overall_verdict}`; "
                f"objective met: `{comparison.objective_met}`\n"
            )
            for case in comparison.cases:
                lines.append(
                    f"- `{case['case_id']}`: {case['verdict']}; "
                    f"change {case['delta_percent']:.3f}%; "
                    f"interval [{case['percent_ci_low']:.3f}, "
                    f"{case['percent_ci_high']:.3f}]%"
                )
        if phase.failure_evidence_link_unavailable:
            lines.append(
                "\nFailure evidence link unavailable: the lower layer may have "
                "retained evidence the controller cannot identify."
            )
        for link in phase.evidence:
            lines.append(f"\nEvidence: `{link.kind}` `{link.id}` record `{link.record_sha256}`")
    lines.extend(
        [
            "",
            "Only confirmation objective_met_provisional is a provisional candidate observation.",
            "Per-case intervals are not simultaneous/campaign confidence or noninferiority proof.",
            "AB/BA balances whole-suite order; positional samples are not temporal pairs.",
            "Use negative results to propose a reviewed patch list in a new campaign.",
        ]
    )
    return "\n".join(lines) + "\n"

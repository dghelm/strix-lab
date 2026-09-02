from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, Never

import _suite_fixtures as fx
import pytest
from pydantic import ValidationError
from test_capsule_runs import (
    _BUILD_ID,
    _FAKE,
    _hooks,
    _machine,
    _manifest,
    _manifest_bytes,
    _record,
)

import strixlab.capsule_comparison as comparison
from strixlab.build_artifacts import (
    CaptureToolV1,
    DynamicInspectionV1,
)
from strixlab.build_cache import SourceBlobRefV1, SourcePatchRefV1, SourceReproducerV1
from strixlab.build_snapshot import SnapshotEntryV1, SnapshotManifestV1
from strixlab.capsule_comparison import (
    CapsuleComparisonAdmissionError,
    CapsuleComparisonArmV1,
    CapsuleComparisonCoordinateV1,
    CapsuleComparisonLoadError,
    CapsuleComparisonReportV1,
    CapsuleComparisonResult,
    CapsuleComparisonStatisticsError,
    compare_finalized_capsule_runs,
)
from strixlab.capsule_contracts import CapsuleComparisonContractV1
from strixlab.capsule_runs import run_capsule
from strixlab.capsule_snapshots import FinalizedCapsuleSnapshot, load_finalized_capsule_snapshot
from strixlab.capsules import CapsuleCoordinateV1
from strixlab.serialization import canonical_json_bytes
from strixlab.source_identity import (
    PatchIdentity,
    SubmoduleIdentity,
    candidate_id,
    content_tree_id,
    length_frame,
)
from strixlab.sources import (
    PatchEvidenceV1,
    SourceEvidenceV1,
    SourceEvidenceV2,
    SubmoduleEvidenceV2,
)

_HEX_A = "a" * 64
_HEX_B = "b" * 64
_BASE_COMMIT = "a" * 40
_LOCATOR = "https://example.test/topk.git"


@pytest.fixture(scope="module", autouse=True)
def _stub_cache() -> Iterator[None]:
    patch = pytest.MonkeyPatch()
    fx.stub_cache_verification(patch)
    yield
    patch.undo()


def _snapshot_id(
    candidate: str,
    content: str,
    entries: tuple[SnapshotEntryV1, ...],
) -> str:
    payload = canonical_json_bytes([entry.model_dump(mode="json") for entry in entries])
    framed = length_frame(
        "strixlab.build.source-snapshot.v1",
        (
            ("candidate-id", candidate.encode("ascii")),
            ("content-tree-id", content.encode("ascii")),
            ("entries", payload),
        ),
    )
    return "snapshot-sha256:" + hashlib.sha256(framed).hexdigest()


def _source(
    *,
    root_tree: str = "1" * 40,
    preparation_id: str = "prep-topk-source-" + "2" * 24,
    request_digest: str = "3" * 64,
    source_id: str = "topk-source",
    source_locator: str | None = _LOCATOR,
    base_commit: str = _BASE_COMMIT,
    branch_hint: str | None = "main",
    adapter: str = "git",
    submodules_enabled: bool = True,
    patch_sha256: str = "4" * 64,
    diff_sha256: str = "5" * 64,
    diff_size_bytes: int = 7,
    status: tuple[str, ...] = ("M  src/topk.cpp",),
    created_at: str = "2026-01-01T00:00:00+00:00",
    entry_sha256: str = "6" * 64,
) -> SourceReproducerV1:
    patch = PatchEvidenceV1(
        order=1,
        sha256=patch_sha256,
        size_bytes=11,
        record_file="patch-001.patch",
    )
    submodule = SubmoduleEvidenceV2(
        path="vendor/hip",
        commit="7" * 40,
        locator="https://example.test/hip.git",
        locator_sha256=hashlib.sha256(b"https://example.test/hip.git").hexdigest(),
    )
    content = content_tree_id(
        root_tree,
        patches=(PatchIdentity(patch.order, patch.size_bytes, patch.sha256),),
        submodules=(SubmoduleIdentity(submodule.path, submodule.commit),),
    )
    candidate = candidate_id(base_commit, content, submodules=submodules_enabled)
    evidence = SourceEvidenceV2(
        preparation_id=preparation_id,
        request_digest=request_digest,
        source_id=source_id,
        source_locator=source_locator,
        source_locator_sha256=hashlib.sha256((source_locator or "local").encode()).hexdigest(),
        base_commit=base_commit,
        branch_hint=branch_hint,
        adapter=adapter,
        submodules_enabled=submodules_enabled,
        patches=(patch,),
        submodules=(submodule,),
        root_tree=root_tree,
        content_tree_id=content,
        candidate_id=candidate,
        diff_file="candidate.diff",
        diff_sha256=diff_sha256,
        diff_size_bytes=diff_size_bytes,
        status=status,
        created_at=created_at,
    )
    entry = SnapshotEntryV1(
        path="src/topk.cpp",
        kind="file",
        mode=0o444,
        size_bytes=17,
        sha256=entry_sha256,
        link_target=None,
    )
    snapshot = SnapshotManifestV1(
        snapshot_id=_snapshot_id(candidate, content, (entry,)),
        candidate_id=candidate,
        content_tree_id=content,
        entries=(entry,),
    )
    evidence_value = json.loads(canonical_json_bytes(evidence.model_dump(mode="json")))
    snapshot_value = json.loads(canonical_json_bytes(snapshot.model_dump(mode="json")))
    return SourceReproducerV1(
        candidate_id=candidate,
        content_tree_id=content,
        snapshot_id=snapshot.snapshot_id,
        source_evidence=evidence_value,
        source_evidence_sha256=hashlib.sha256(canonical_json_bytes(evidence_value)).hexdigest(),
        snapshot_manifest=snapshot_value,
        diff=SourceBlobRefV1(
            relative_path="source/diff.patch",
            sha256=diff_sha256,
            size_bytes=diff_size_bytes,
        ),
        patches=(
            SourcePatchRefV1(
                order=1,
                relative_path="source/patches/0001.patch",
                sha256=patch_sha256,
                size_bytes=11,
            ),
        ),
    )


@pytest.fixture(scope="module")
def real_pair(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[
    Path,
    FinalizedCapsuleSnapshot,
    FinalizedCapsuleSnapshot,
]:
    root = tmp_path_factory.mktemp("capsule-comparison")
    home = root / "home"
    home.mkdir(mode=0o700)
    state = root / "state.json"
    fake = _FAKE.read_bytes()
    canonical = _record(fake, state).model_copy(update={"source": _source()})
    fx.make_present_build(home, record=canonical)
    executable = home / "builds" / "materialized" / _BUILD_ID / "bin" / "topk-capsule"
    executable.parent.mkdir(mode=0o700)
    executable.write_bytes(fake)
    executable.chmod(0o700)
    manifest = _manifest()
    machine = _machine(root)

    def execute() -> Any:
        state.unlink(missing_ok=True)
        return run_capsule(
            manifest,
            _manifest_bytes(manifest),
            machine_profile=machine,
            build_id=_BUILD_ID,
            home=home,
            environ={},
            hooks=_hooks(root),
        )

    baseline_run = execute()
    candidate_run = execute()
    return (
        home,
        load_finalized_capsule_snapshot(baseline_run.run_id, home=home),
        load_finalized_capsule_snapshot(candidate_run.run_id, home=home),
    )


def _install_pair(
    monkeypatch: pytest.MonkeyPatch,
    baseline: FinalizedCapsuleSnapshot,
    candidate: FinalizedCapsuleSnapshot,
) -> None:
    values = {baseline.run_id: baseline, candidate.run_id: candidate}
    monkeypatch.setattr(
        comparison,
        "load_finalized_capsule_snapshot",
        lambda run_id, *, home: values[run_id],
    )


def _call_pair(
    monkeypatch: pytest.MonkeyPatch,
    baseline: FinalizedCapsuleSnapshot,
    candidate: FinalizedCapsuleSnapshot,
) -> CapsuleComparisonResult:
    _install_pair(monkeypatch, baseline, candidate)
    return compare_finalized_capsule_runs(baseline.run_id, candidate.run_id, home=Path("/unused"))


def _with_source(
    snapshot: FinalizedCapsuleSnapshot,
    source: SourceReproducerV1,
) -> FinalizedCapsuleSnapshot:
    build = snapshot.build_snapshot.model_copy(update={"source": source})
    return replace(snapshot, build_snapshot=build)


def _as_v1(source: SourceReproducerV1) -> SourceReproducerV1:
    raw = dict(source.source_evidence)
    raw["schema_version"] = 1
    evidence = SourceEvidenceV1.model_validate(raw)
    value = json.loads(canonical_json_bytes(evidence.model_dump(mode="json")))
    return source.model_copy(
        update={
            "source_evidence": value,
            "source_evidence_sha256": hashlib.sha256(canonical_json_bytes(value)).hexdigest(),
        }
    )


def _disabled_source(*, schema_version: int, valid: bool) -> SourceReproducerV1:
    source = _source(submodules_enabled=False)
    raw = dict(source.source_evidence)
    raw["schema_version"] = schema_version
    submodule = dict(raw["submodules"][0])
    if valid:
        submodule["locator"] = None
        submodule["locator_sha256"] = (
            hashlib.sha256(b"uninitialized").hexdigest() if schema_version == 1 else None
        )
    raw["submodules"] = [submodule]
    evidence_type = SourceEvidenceV1 if schema_version == 1 else SourceEvidenceV2
    evidence = evidence_type.model_validate(raw)
    value = json.loads(canonical_json_bytes(evidence.model_dump(mode="json")))
    return source.model_copy(
        update={
            "source_evidence": value,
            "source_evidence_sha256": hashlib.sha256(canonical_json_bytes(value)).hexdigest(),
        }
    )


def _with_build(
    snapshot: FinalizedCapsuleSnapshot,
    build: Any,
    *,
    executable_sha256: str | None = None,
) -> FinalizedCapsuleSnapshot:
    record_sha = (
        "record-sha256:"
        + hashlib.sha256(canonical_json_bytes(build.model_dump(mode="json"))).hexdigest()
    )
    executable = executable_sha256 or snapshot.result.executable_sha256
    result = snapshot.result.model_copy(
        update={
            "build_id": build.build_id,
            "canonical_record_sha256": record_sha,
            "executable_sha256": executable,
        }
    )
    protocol = snapshot.protocol.model_copy(update={"executable_sha256": executable})
    return replace(
        snapshot,
        build_snapshot=build,
        build_snapshot_sha256="9" * 64,
        build_record_sha256=record_sha,
        result=result,
        protocol=protocol,
    )


def _with_contract(
    snapshot: FinalizedCapsuleSnapshot,
    differences: tuple[str, ...],
) -> FinalizedCapsuleSnapshot:
    contract = CapsuleComparisonContractV1(
        policy="paired-latency-log-bootstrap-v1",
        protected_regression_bps=500,
        permitted_arm_differences=differences,
    )
    manifest = snapshot.manifest.model_copy(
        update={"contract": snapshot.manifest.contract.model_copy(update={"comparison": contract})}
    )
    assert snapshot.protocol.scenario is not None
    scenario = snapshot.protocol.scenario.model_copy(update={"comparison": contract})
    protocol = snapshot.protocol.model_copy(update={"scenario": scenario})
    contract_sha = hashlib.sha256(
        canonical_json_bytes(contract.model_dump(mode="json"))
    ).hexdigest()
    alignment = replace(
        snapshot.alignment,
        comparison=contract,
        comparison_sha256=contract_sha,
        permitted_arm_differences=contract.permitted_arm_differences,
    )
    return replace(snapshot, manifest=manifest, protocol=protocol, alignment=alignment)


def _with_candidate(
    snapshot: FinalizedCapsuleSnapshot,
    candidate_id: str,
) -> FinalizedCapsuleSnapshot:
    manifest = snapshot.manifest.model_copy(update={"candidate": candidate_id})
    result = snapshot.result.model_copy(update={"candidate": candidate_id})
    protocol = snapshot.protocol.model_copy(update={"candidate": candidate_id})
    alignment = replace(snapshot.alignment, candidate=candidate_id)
    return replace(
        snapshot,
        manifest=manifest,
        result=result,
        protocol=protocol,
        alignment=alignment,
        resolved_manifest_sha256="8" * 64,
    )


def _assert_admission_closed(
    monkeypatch: pytest.MonkeyPatch,
    baseline: FinalizedCapsuleSnapshot,
    candidate: FinalizedCapsuleSnapshot,
) -> None:
    _install_pair(monkeypatch, baseline, candidate)
    with pytest.raises(CapsuleComparisonAdmissionError) as raised:
        compare_finalized_capsule_runs(baseline.run_id, candidate.run_id, home=Path("/unused"))
    assert str(raised.value) == "capsule comparison admission failed"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


def test_real_public_api_same_candidate_is_directional_and_canonical(
    real_pair: tuple[Path, FinalizedCapsuleSnapshot, FinalizedCapsuleSnapshot],
) -> None:
    home, baseline, candidate = real_pair
    result = compare_finalized_capsule_runs(baseline.run_id, candidate.run_id, home=home)
    reverse = compare_finalized_capsule_runs(candidate.run_id, baseline.run_id, home=home)

    assert result.report.baseline.candidate == result.report.candidate.candidate == "candidate-a"
    assert result.report.baseline.run_id == baseline.run_id
    assert result.report.candidate.run_id == candidate.run_id
    assert result.report.baseline.build_snapshot_sha256 == baseline.build_snapshot_sha256
    assert result.report.candidate.build_snapshot_sha256 == candidate.build_snapshot_sha256
    assert result.report.baseline.machine_snapshot_sha256 == baseline.machine_snapshot_sha256
    assert result.report.candidate.machine_snapshot_sha256 == candidate.machine_snapshot_sha256
    assert reverse.report.baseline.run_id == candidate.run_id
    assert result.report_bytes == canonical_json_bytes(result.report.model_dump(mode="json"))
    assert result.report_sha256 == hashlib.sha256(result.report_bytes).hexdigest()
    assert result.report_sha256 != reverse.report_sha256
    assert result.report.overall_verdict == "inconclusive"


def test_distinct_run_and_record_identities_are_required(
    real_pair: tuple[Path, FinalizedCapsuleSnapshot, FinalizedCapsuleSnapshot],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _home, baseline, candidate = real_pair
    _assert_admission_closed(monkeypatch, baseline, replace(candidate, run_id=baseline.run_id))
    with monkeypatch.context() as scoped:
        _assert_admission_closed(
            scoped,
            baseline,
            replace(candidate, record_sha256=baseline.record_sha256),
        )


@pytest.mark.parametrize(
    "differences",
    [
        ("candidate-id",),
        ("candidate-id", "source-candidate", "build-output"),
    ],
)
def test_both_closed_difference_tuples_admit(
    real_pair: tuple[Path, FinalizedCapsuleSnapshot, FinalizedCapsuleSnapshot],
    monkeypatch: pytest.MonkeyPatch,
    differences: tuple[str, ...],
) -> None:
    _home, baseline, candidate = real_pair
    result = _call_pair(
        monkeypatch,
        _with_contract(baseline, differences),
        _with_contract(candidate, differences),
    )
    assert result.report.comparison.permitted_arm_differences == differences


def test_candidate_identity_may_differ_but_candidate_only_closes_source_and_build(
    real_pair: tuple[Path, FinalizedCapsuleSnapshot, FinalizedCapsuleSnapshot],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _home, baseline, candidate = real_pair
    baseline = _with_contract(baseline, ("candidate-id",))
    candidate = _with_candidate(
        _with_contract(candidate, ("candidate-id",)),
        "candidate-b",
    )
    result = _call_pair(monkeypatch, baseline, candidate)
    assert result.report.baseline.candidate == "candidate-a"
    assert result.report.candidate.candidate == "candidate-b"

    with monkeypatch.context() as scoped:
        _assert_admission_closed(
            scoped, baseline, _with_source(candidate, _source(root_tree="9" * 40))
        )
    changed_build = candidate.build_snapshot.model_copy(
        update={"recipe_id": "recipe-sha256:" + "9" * 64}
    )
    with monkeypatch.context() as scoped:
        _assert_admission_closed(scoped, baseline, _with_build(candidate, changed_build))
    with monkeypatch.context() as scoped:
        _assert_admission_closed(
            scoped,
            baseline,
            _with_build(candidate, candidate.build_snapshot, executable_sha256="9" * 64),
        )


@pytest.mark.parametrize(
    "source",
    [
        _source(preparation_id="prep-topk-source-" + "9" * 24),
        _source(request_digest="9" * 64),
        _source(root_tree="9" * 40),
        _source(patch_sha256="9" * 64),
        _source(diff_sha256="9" * 64),
        _source(status=("M  src/topk.cpp", "A  src/provider.cpp")),
        _source(created_at="2026-02-02T00:00:00+00:00"),
        _source(entry_sha256="9" * 64),
    ],
)
def test_every_candidate_derived_source_family_is_permitted(
    real_pair: tuple[Path, FinalizedCapsuleSnapshot, FinalizedCapsuleSnapshot],
    monkeypatch: pytest.MonkeyPatch,
    source: SourceReproducerV1,
) -> None:
    _home, baseline, candidate = real_pair
    result = _call_pair(monkeypatch, baseline, _with_source(candidate, source))
    assert result.report.candidate.source_candidate_id == source.candidate_id


def test_strict_source_evidence_v1_and_v2_are_both_authenticated(
    real_pair: tuple[Path, FinalizedCapsuleSnapshot, FinalizedCapsuleSnapshot],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _home, baseline, candidate = real_pair
    baseline_v1 = _with_source(baseline, _as_v1(baseline.build_snapshot.source))
    candidate_v1 = _with_source(candidate, _as_v1(candidate.build_snapshot.source))
    assert _call_pair(monkeypatch, baseline_v1, candidate_v1).report.coordinates


@pytest.mark.parametrize("schema_version", [1, 2])
def test_disabled_submodule_evidence_requires_version_specific_locator_sentinel(
    real_pair: tuple[Path, FinalizedCapsuleSnapshot, FinalizedCapsuleSnapshot],
    monkeypatch: pytest.MonkeyPatch,
    schema_version: int,
) -> None:
    _home, baseline, candidate = real_pair
    malformed = _disabled_source(schema_version=schema_version, valid=False)
    _assert_admission_closed(
        monkeypatch,
        _with_source(baseline, malformed),
        _with_source(candidate, malformed),
    )


@pytest.mark.parametrize("schema_version", [1, 2])
def test_disabled_submodule_evidence_accepts_version_specific_locator_sentinel(
    real_pair: tuple[Path, FinalizedCapsuleSnapshot, FinalizedCapsuleSnapshot],
    monkeypatch: pytest.MonkeyPatch,
    schema_version: int,
) -> None:
    _home, baseline, candidate = real_pair
    source = _disabled_source(schema_version=schema_version, valid=True)
    result = _call_pair(
        monkeypatch,
        _with_source(baseline, source),
        _with_source(candidate, source),
    )
    assert result.report.coordinates


def _build_mutations(snapshot: FinalizedCapsuleSnapshot) -> tuple[Any, ...]:
    build = snapshot.build_snapshot
    target = build.artifacts.targets[0]
    artifact = build.artifacts.artifacts[0]
    inspection = DynamicInspectionV1(
        artifact=artifact.path,
        elf_type="ET_EXEC",
        dynamic=False,
        static=True,
        needed=(),
        dependencies=(),
        readelf_sha256="9" * 64,
    )
    capture = CaptureToolV1(
        name="readelf",
        path="/usr/bin/readelf",
        realpath="/usr/bin/readelf",
        mode=0o755,
        size_bytes=12,
        sha256="9" * 64,
        version_sha256="8" * 64,
    )
    return (
        build.model_copy(update={"recipe_id": "recipe-sha256:" + "9" * 64}),
        build.model_copy(update={"build_id": "build-sha256:" + "9" * 64}),
        build.model_copy(update={"producer_attempt_id": "attempt-" + "9" * 24 + "-" + "8" * 32}),
        build.model_copy(
            update={
                "artifacts": build.artifacts.model_copy(
                    update={"artifact_set_id": "artifact-set-sha256:" + "9" * 64}
                )
            }
        ),
        build.model_copy(
            update={
                "artifacts": build.artifacts.model_copy(
                    update={"targets": (target.model_copy(update={"target_id": "new-id"}),)}
                )
            }
        ),
        build.model_copy(
            update={
                "artifacts": build.artifacts.model_copy(
                    update={
                        "artifacts": (
                            artifact.model_copy(
                                update={"mode": 0o755, "size_bytes": 99, "sha256": "9" * 64}
                            ),
                        )
                    }
                )
            }
        ),
        build.model_copy(
            update={"artifacts": build.artifacts.model_copy(update={"inspections": (inspection,)})}
        ),
        build.model_copy(
            update={"artifacts": build.artifacts.model_copy(update={"capture_tools": (capture,)})}
        ),
        build.model_copy(
            update={
                "artifacts": build.artifacts.model_copy(
                    update={
                        "cmake_cache_sha256": "9" * 64,
                        "compile_commands_sha256": "8" * 64,
                    }
                )
            }
        ),
    )


def test_every_build_output_family_is_permitted(
    real_pair: tuple[Path, FinalizedCapsuleSnapshot, FinalizedCapsuleSnapshot],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _home, baseline, candidate = real_pair
    for build in _build_mutations(candidate):
        with monkeypatch.context() as scoped:
            result = _call_pair(scoped, baseline, _with_build(candidate, build))
            assert result.report.candidate.build_id == build.build_id

    changed_digest = _with_build(candidate, candidate.build_snapshot, executable_sha256="9" * 64)
    with monkeypatch.context() as scoped:
        assert _call_pair(scoped, baseline, changed_digest).report.candidate.executable_sha256 == (
            "9" * 64
        )


@pytest.mark.parametrize(
    "source",
    [
        _source(source_id="other-source"),
        _source(source_locator="https://example.test/other.git"),
        _source(base_commit="9" * 40),
        _source(branch_hint="other"),
        _source(adapter="other"),
        _source(submodules_enabled=False),
    ],
)
def test_neighboring_stable_source_fields_are_rejected(
    real_pair: tuple[Path, FinalizedCapsuleSnapshot, FinalizedCapsuleSnapshot],
    monkeypatch: pytest.MonkeyPatch,
    source: SourceReproducerV1,
) -> None:
    _home, baseline, candidate = real_pair
    _assert_admission_closed(monkeypatch, baseline, _with_source(candidate, source))


def _stable_build_mutations(snapshot: FinalizedCapsuleSnapshot) -> tuple[Any, ...]:
    build = snapshot.build_snapshot
    target = build.artifacts.targets[0]
    artifact = build.artifacts.artifacts[0]
    return (
        build.model_copy(update={"profile_sha256": "9" * 64}),
        build.model_copy(update={"environment": build.environment[:-1]}),
        build.model_copy(update={"requested_targets": ("other-target",)}),
        build.model_copy(update={"selections": build.selections[:-1]}),
        build.model_copy(
            update={
                "artifacts": build.artifacts.model_copy(
                    update={
                        "targets": (target.model_copy(update={"target_type": "SHARED_LIBRARY"}),)
                    }
                )
            }
        ),
        build.model_copy(
            update={
                "artifacts": build.artifacts.model_copy(
                    update={
                        "artifacts": (artifact.model_copy(update={"path": "bin/renamed-capsule"}),)
                    }
                )
            }
        ),
    )


def test_neighboring_stable_build_and_topology_fields_are_rejected(
    real_pair: tuple[Path, FinalizedCapsuleSnapshot, FinalizedCapsuleSnapshot],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _home, baseline, candidate = real_pair
    for build in _stable_build_mutations(candidate):
        with monkeypatch.context() as scoped:
            _assert_admission_closed(scoped, baseline, _with_build(candidate, build))


def _invalid_source_variants(source: SourceReproducerV1) -> tuple[SourceReproducerV1, ...]:
    evidence = dict(source.source_evidence)
    evidence["unexpected"] = True
    snapshot = dict(source.snapshot_manifest)
    snapshot["candidate_id"] = "candidate-sha256:" + "9" * 64
    locator = dict(source.source_evidence)
    locator["source_locator_sha256"] = "9" * 64
    locator_sha = hashlib.sha256(canonical_json_bytes(locator)).hexdigest()
    diff = source.diff
    assert diff is not None
    return (
        source.model_copy(update={"source_evidence_sha256": "9" * 64}),
        source.model_copy(update={"candidate_id": "candidate-sha256:" + "9" * 64}),
        source.model_copy(update={"snapshot_id": "snapshot-sha256:" + "9" * 64}),
        source.model_copy(update={"source_evidence": evidence}),
        source.model_copy(update={"snapshot_manifest": snapshot}),
        source.model_copy(
            update={"source_evidence": locator, "source_evidence_sha256": locator_sha}
        ),
        source.model_copy(update={"diff": diff.model_copy(update={"sha256": "9" * 64})}),
        source.model_copy(update={"patches": (source.patches[0].model_copy(update={"order": 2}),)}),
    )


def test_source_digest_binding_diff_and_patch_failures_are_rejected(
    real_pair: tuple[Path, FinalizedCapsuleSnapshot, FinalizedCapsuleSnapshot],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _home, baseline, candidate = real_pair
    for source in _invalid_source_variants(candidate.build_snapshot.source):
        with monkeypatch.context() as scoped:
            _assert_admission_closed(scoped, baseline, _with_source(candidate, source))


def test_manifest_scenario_machine_and_coordinate_structure_must_match(
    real_pair: tuple[Path, FinalizedCapsuleSnapshot, FinalizedCapsuleSnapshot],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _home, baseline, candidate = real_pair
    changed_manifest = replace(
        candidate,
        manifest=candidate.manifest.model_copy(
            update={
                "timeouts": candidate.manifest.timeouts.model_copy(
                    update={"benchmark_seconds": 2.0}
                )
            }
        ),
    )
    _assert_admission_closed(monkeypatch, baseline, changed_manifest)
    assert candidate.protocol.scenario is not None
    coordinate = candidate.protocol.scenario.coordinates[0].model_copy(update={"warmup_count": 2})
    changed_scenario = candidate.protocol.scenario.model_copy(
        update={"coordinates": (coordinate, *candidate.protocol.scenario.coordinates[1:])}
    )
    with monkeypatch.context() as scoped:
        _assert_admission_closed(
            scoped,
            baseline,
            replace(
                candidate,
                protocol=candidate.protocol.model_copy(update={"scenario": changed_scenario}),
            ),
        )
    with monkeypatch.context() as scoped:
        _assert_admission_closed(
            scoped,
            baseline,
            replace(
                candidate,
                machine_snapshot=candidate.machine_snapshot.model_copy(
                    update={"id": "other-machine"}
                ),
            ),
        )


def test_result_protocol_phase_process_correctness_and_input_semantics_are_closed(
    real_pair: tuple[Path, FinalizedCapsuleSnapshot, FinalizedCapsuleSnapshot],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _home, baseline, candidate = real_pair
    result_variants = (
        candidate.result.model_copy(update={"target": "other-target"}),
        candidate.result.model_copy(
            update={
                "inputs": (
                    candidate.result.inputs[0],
                    candidate.result.inputs[1].model_copy(update={"sha256": "9" * 64}),
                )
            }
        ),
    )
    for result in result_variants:
        with monkeypatch.context() as scoped:
            _assert_admission_closed(scoped, baseline, replace(candidate, result=result))

    assert candidate.protocol.correctness is not None
    incorrect = candidate.protocol.correctness[0].model_copy(update={"passed": False})
    protocol_variants = (
        candidate.protocol.model_copy(update={"capsule_id": "other-capsule"}),
        candidate.protocol.model_copy(
            update={"correctness": (incorrect, *candidate.protocol.correctness[1:])}
        ),
    )
    for protocol in protocol_variants:
        with monkeypatch.context() as scoped:
            _assert_admission_closed(scoped, baseline, replace(candidate, protocol=protocol))

    process = candidate.protocol.phases[0].process.model_copy(update={"stderr_sha256": "9" * 64})
    phase = candidate.protocol.phases[0].model_copy(update={"process": process})
    protocol = candidate.protocol.model_copy(
        update={"phases": (phase, *candidate.protocol.phases[1:])}
    )
    _assert_admission_closed(monkeypatch, baseline, replace(candidate, protocol=protocol))


def test_process_timing_samples_and_workspace_are_not_admission_equality(
    real_pair: tuple[Path, FinalizedCapsuleSnapshot, FinalizedCapsuleSnapshot],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _home, baseline, candidate = real_pair
    process = candidate.protocol.phases[0].process.model_copy(
        update={"duration_seconds": candidate.protocol.phases[0].process.duration_seconds + 1.0}
    )
    phase = candidate.protocol.phases[0].model_copy(update={"process": process})
    latency = dict(candidate.latency_seconds_by_coordinate)
    workspace = dict(candidate.workspace_bytes_by_coordinate)
    coordinate_id = candidate.coordinates[-1].coordinate_id
    latency[coordinate_id] = tuple(value * 1.1 for value in latency[coordinate_id])
    workspace[coordinate_id] += 17
    assert candidate.protocol.benchmark is not None
    benchmark = candidate.protocol.benchmark[-1].model_copy(
        update={
            "latency_seconds": latency[coordinate_id],
            "workspace_bytes": workspace[coordinate_id],
        }
    )
    protocol = candidate.protocol.model_copy(
        update={
            "phases": (phase, *candidate.protocol.phases[1:]),
            "benchmark": (*candidate.protocol.benchmark[:-1], benchmark),
        }
    )
    changed = replace(
        candidate,
        protocol=protocol,
        latency_seconds_by_coordinate=latency,
        workspace_bytes_by_coordinate=workspace,
    )

    result = _call_pair(monkeypatch, baseline, changed)
    coordinate = result.report.coordinates[-1]
    assert coordinate.candidate_workspace_bytes - coordinate.baseline_workspace_bytes == 17
    assert coordinate.mean_log_effect < 0


def test_bootstrap_r7_and_median_golden_vectors() -> None:
    indexes = tuple(
        comparison._bootstrap_index(  # type: ignore[attr-defined]
            "record-sha256:" + "1" * 64,
            "record-sha256:" + "2" * 64,
            "eval-case",
            "reverse",
            replicate,
            draw,
            5,
        )
        for replicate, draw in ((0, 0), (0, 1), (1, 0), (4095, 4))
    )
    assert indexes == (1, 3, 4, 3)
    assert comparison._r7((1.0, 2.0, 3.0, 4.0), 0.0) == 1.0  # type: ignore[attr-defined]
    assert comparison._r7((1.0, 2.0, 3.0, 4.0), 0.25) == 1.75  # type: ignore[attr-defined]
    assert comparison._r7((1.0, 2.0, 3.0, 4.0), 1.0) == 4.0  # type: ignore[attr-defined]
    assert comparison._median((3.0, 1.0, 2.0)) == 2.0  # type: ignore[attr-defined]
    assert comparison._median((4.0, 1.0, 3.0, 2.0)) == 2.5  # type: ignore[attr-defined]


def _coordinate(
    *,
    case_set: str = "evaluation",
    order: int = 0,
    coordinate_id: str = "eval-coordinate",
) -> CapsuleCoordinateV1:
    return CapsuleCoordinateV1.model_validate(
        {
            "coordinate_id": coordinate_id,
            "case_id": f"case-{order}",
            "case_set": case_set,
            "mode": "mode-a",
            "order": order,
            "input_id": f"input-{order}",
            "input_sha256": str(order + 1) * 64,
            "warmup_count": 1,
            "sample_count": 5,
        }
    )


def _statistics(
    coordinate: CapsuleCoordinateV1,
    baseline: tuple[float, ...],
    candidate: tuple[float, ...],
    *,
    baseline_workspace: int = 100,
    candidate_workspace: int = 80,
) -> CapsuleComparisonCoordinateV1:
    return comparison._coordinate_statistics(  # type: ignore[attr-defined]
        coordinate,
        baseline,
        candidate,
        baseline_workspace,
        candidate_workspace,
        CapsuleComparisonContractV1(
            policy="paired-latency-log-bootstrap-v1",
            protected_regression_bps=500,
            permitted_arm_differences=("candidate-id",),
        ),
        "record-sha256:" + "1" * 64,
        "record-sha256:" + "2" * 64,
    )


def test_positive_effect_workspace_sign_and_training_protected_veto() -> None:
    evaluation = _statistics(_coordinate(), (2.0,) * 5, (1.0,) * 5)
    training = _statistics(
        _coordinate(case_set="training", order=1, coordinate_id="train-coordinate"),
        (1.0,) * 5,
        (2.0,) * 5,
        baseline_workspace=50,
        candidate_workspace=75,
    )
    assert evaluation.mean_log_effect > 0
    assert evaluation.baseline_over_candidate_ratio == 2.0
    assert evaluation.improvement_percent == 100.0
    assert evaluation.workspace_delta_bytes == -20
    assert training.verdict == "regression" and training.protected_regression
    assert training.workspace_delta_bytes == 25
    assert comparison._aggregate((evaluation, training)) == "mixed"  # type: ignore[attr-defined]


def test_strict_protection_boundary_and_arithmetic_overflow_fail_closed() -> None:
    exact = _statistics(_coordinate(), (20.0,) * 5, (21.0,) * 5)
    beyond = _statistics(_coordinate(), (20.0,) * 5, (21.01,) * 5)
    assert not exact.protected_regression
    assert beyond.protected_regression
    with pytest.raises((ValidationError, comparison._StatisticsFailure)):  # type: ignore[attr-defined]
        _statistics(_coordinate(), (5e-324,) * 5, (1e308,) * 5)


def test_draw_cap_boundary() -> None:
    first = _coordinate().model_copy(update={"sample_count": 2048})
    second = _coordinate(order=1, coordinate_id="eval-two").model_copy(
        update={"sample_count": 2048}
    )
    comparison._check_draw_budget((first, second))  # type: ignore[attr-defined]
    over = second.model_copy(update={"sample_count": 2049})
    with pytest.raises(comparison._StatisticsFailure):  # type: ignore[attr-defined]
        comparison._check_draw_budget((first, over))  # type: ignore[attr-defined]


def test_models_reject_forged_relations_and_result_bytes(
    real_pair: tuple[Path, FinalizedCapsuleSnapshot, FinalizedCapsuleSnapshot],
) -> None:
    home, baseline, candidate = real_pair
    result = compare_finalized_capsule_runs(baseline.run_id, candidate.run_id, home=home)
    coordinate = result.report.coordinates[0].model_dump(mode="json")
    coordinate["workspace_delta_bytes"] += 1
    with pytest.raises(ValidationError):
        CapsuleComparisonCoordinateV1.model_validate(coordinate)

    report = result.report.model_dump(mode="json")
    report["coordinates"][0]["protected_regression"] = True
    with pytest.raises(ValidationError):
        CapsuleComparisonReportV1.model_validate(report)
    with pytest.raises(ValidationError):
        CapsuleComparisonResult(
            report=result.report,
            report_bytes=result.report_bytes + b" ",
            report_sha256=result.report_sha256,
        )


def test_report_has_only_reviewed_fields_and_no_host_or_opaque_data(
    real_pair: tuple[Path, FinalizedCapsuleSnapshot, FinalizedCapsuleSnapshot],
) -> None:
    home, baseline, candidate = real_pair
    result = compare_finalized_capsule_runs(baseline.run_id, candidate.run_id, home=home)
    text = result.report_bytes.decode("utf-8")
    assert str(home) not in text
    assert _LOCATOR not in text
    assert "FAKE_CAPSULE_STATE" not in text
    assert "opaque_payload" not in text
    assert set(CapsuleComparisonCoordinateV1.model_fields) == {
        "coordinate",
        "baseline_median_seconds",
        "candidate_median_seconds",
        "mean_log_effect",
        "baseline_over_candidate_ratio",
        "improvement_percent",
        "log_ci_low",
        "log_ci_high",
        "baseline_noise_log",
        "baseline_workspace_bytes",
        "candidate_workspace_bytes",
        "workspace_delta_bytes",
        "verdict",
        "protected_regression",
    }
    assert not {"latency_mean", "percent_ci", "noise_percent", "pair_count"} & set(
        CapsuleComparisonCoordinateV1.model_fields
    )
    assert set(CapsuleComparisonArmV1.model_fields) == {
        "label",
        "run_id",
        "record_sha256",
        "candidate",
        "source_candidate_id",
        "manifest_sha256",
        "result_sha256",
        "protocol_sha256",
        "build_snapshot_sha256",
        "machine_snapshot_sha256",
        "build_id",
        "build_record_sha256",
        "machine_id",
        "machine_profile_sha256",
        "executable_sha256",
    }


def test_fixed_safe_load_admission_and_statistics_errors_have_no_context(
    real_pair: tuple[Path, FinalizedCapsuleSnapshot, FinalizedCapsuleSnapshot],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _home, baseline, candidate = real_pair

    def fail_load(*_args: Any, **_kwargs: Any) -> Never:
        raise OSError("/secret/path API_TOKEN=value")

    monkeypatch.setattr(comparison, "load_finalized_capsule_snapshot", fail_load)
    with pytest.raises(CapsuleComparisonLoadError) as loaded:
        compare_finalized_capsule_runs("baseline-secret", "candidate-secret", home=Path("/secret"))
    assert str(loaded.value) == "capsule comparison snapshot loading failed"
    assert loaded.value.__cause__ is loaded.value.__context__ is None

    with monkeypatch.context() as scoped:
        _install_pair(scoped, baseline, replace(candidate, record_sha256=baseline.record_sha256))
        with pytest.raises(CapsuleComparisonAdmissionError) as admitted:
            compare_finalized_capsule_runs(baseline.run_id, candidate.run_id, home=Path("/secret"))
        assert str(admitted.value) == "capsule comparison admission failed"
        assert admitted.value.__cause__ is admitted.value.__context__ is None

    with monkeypatch.context() as scoped:
        _install_pair(scoped, baseline, candidate)
        scoped.setattr(comparison, "_try_compare", lambda _admission: None)
        with pytest.raises(CapsuleComparisonStatisticsError) as statistics:
            compare_finalized_capsule_runs(baseline.run_id, candidate.run_id, home=Path("/secret"))
        assert str(statistics.value) == "capsule comparison statistics failed"
        assert statistics.value.__cause__ is statistics.value.__context__ is None

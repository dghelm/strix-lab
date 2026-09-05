from __future__ import annotations

import contextlib
import hashlib
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import _suite_fixtures as fx
import pytest
from pydantic import ValidationError

import strixlab.capsule_runs as capsule_runs
from strixlab.build_artifacts import ArtifactV1, BuildArtifactsV1, TargetArtifactsV1
from strixlab.build_cache import BuildCacheError, IdentityEntryV1
from strixlab.capsule_runs import (
    CapsuleExecutionError,
    CapsuleHooks,
    CapsuleResultV1,
    CapsuleRunError,
    run_capsule,
)
from strixlab.capsules import CapsulePhaseResultV1, CapsuleProcessV1, CapsuleProtocolResultV1
from strixlab.evidence import RunOutcome, list_portable_entries, read_record_member
from strixlab.locks import LockAttempt, LockStatus
from strixlab.manifests import (
    CapsuleManifestV1,
    ExclusiveLockV1,
    MachineExpectationV1,
    MachineProfileV1,
    MachineValidityV1,
    TelemetryV1,
)
from strixlab.serialization import canonical_yaml_bytes

_FAKE = Path(__file__).parents[1] / "fixtures" / "fake_capsule.py"
_BUILD_ID = fx.BUILD_ID


@pytest.fixture(autouse=True)
def _stub_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    fx.stub_cache_verification(monkeypatch)


def _manifest(**build_updates: str) -> CapsuleManifestV1:
    build = {
        "source_id": "topk-source",
        "source_commit": "a" * 40,
        "toolchain_mode": "host",
        "gfx_target": "gfx1151",
        "target": "topk-capsule",
        **build_updates,
    }
    return CapsuleManifestV1.model_validate(
        {
            "schema_version": 1,
            "id": "topk-capsule",
            "candidate": "candidate-a",
            "machine": "strix-halo-128g",
            "build": build,
            "contract": {
                "protocol": "native-capsule-v1",
                "scenario_sha256": "b" * 64,
                "comparison": {
                    "policy": "paired-latency-log-bootstrap-v1",
                    "protected_regression_bps": 500,
                    "permitted_arm_differences": [
                        "candidate-id",
                        "source-candidate",
                        "build-output",
                    ],
                },
            },
            "timeouts": {
                "describe_seconds": 1.0,
                "correctness_seconds": 1.0,
                "benchmark_seconds": 1.0,
            },
        }
    )


def _machine(tmp_path: Path, *, machine_id: str = "strix-halo-128g") -> MachineProfileV1:
    return MachineProfileV1(
        schema_version=1,
        id=machine_id,
        expect=MachineExpectationV1(gpu_arch="gfx1151", integrated_gpu=True, memory_gib_min=64),
        exclusive_lock=ExclusiveLockV1(path=str(tmp_path / "machine.lock")),
        telemetry=TelemetryV1(amd_smi="disabled", sample_interval_ms=100),
        validity=MachineValidityV1(
            require_ac_power=False,
            max_background_gpu_busy_pct=100,
            min_available_memory_gib=0,
            temperature_warn_c=100,
        ),
    )


def _record(fake: bytes, state_path: Path, *, mode: str = "success", **updates: Any) -> Any:
    digest = hashlib.sha256(fake).hexdigest()
    artifacts = BuildArtifactsV1(
        artifact_set_id="artifact-set-sha256:" + "c" * 64,
        targets=(
            TargetArtifactsV1(
                name="topk-capsule",
                target_id="id-topk-capsule",
                target_type="EXECUTABLE",
                artifacts=("bin/topk-capsule",),
            ),
        ),
        artifacts=(
            ArtifactV1(
                path="bin/topk-capsule",
                kind="elf",
                elf_type="ET_EXEC",
                mode=0o700,
                size_bytes=len(fake),
                sha256=digest,
                targets=("topk-capsule",),
            ),
        ),
        inspections=(),
        capture_tools=(),
        cmake_cache_sha256="d" * 64,
    )
    environment = fx.default_environment() + (
        IdentityEntryV1(name="FAKE_CAPSULE_MODE", value=mode),
        IdentityEntryV1(name="FAKE_CAPSULE_STATE", value=str(state_path)),
    )
    record = fx.canonical_record(
        environment=environment,
        toolchain_mode="host",
        artifacts=artifacts,
        source_id="topk-source",
        base_commit="a" * 40,
    ).model_copy(update={"requested_targets": ("topk-capsule",)})
    return record.model_copy(update=updates)


def _prepare(
    tmp_path: Path, *, mode: str = "success", record_updates: dict[str, Any] | None = None
) -> tuple[Path, CapsuleManifestV1, MachineProfileV1]:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    fake = _FAKE.read_bytes()
    record = _record(fake, tmp_path / "state.json", mode=mode, **(record_updates or {}))
    fx.make_present_build(home, record=record)
    executable = home / "builds" / "materialized" / _BUILD_ID / "bin" / "topk-capsule"
    executable.parent.mkdir(mode=0o700)
    executable.write_bytes(fake)
    executable.chmod(0o700)
    return home, _manifest(), _machine(tmp_path)


def _manifest_bytes(manifest: CapsuleManifestV1) -> bytes:
    return canonical_yaml_bytes(manifest.model_dump(mode="json"))


def _acquiring_lock(_path: Path) -> Any:
    @contextlib.contextmanager
    def factory(path: Path) -> Iterator[LockAttempt]:
        yield LockAttempt(LockStatus.ACQUIRED, path)

    return factory


def _refusing_lock(_path: Path) -> Any:
    @contextlib.contextmanager
    def factory(path: Path) -> Iterator[LockAttempt]:
        yield LockAttempt(LockStatus.CONTENDED, path, "secret child detail")

    return factory


def _hooks(tmp_path: Path, **updates: Any) -> CapsuleHooks:
    defaults: dict[str, Any] = {
        "temp_root_factory": lambda: Path(tempfile.mkdtemp(dir=tmp_path)),
        "machine_lock": _acquiring_lock(tmp_path),
    }
    defaults.update(updates)
    return CapsuleHooks(**defaults)


def _run(tmp_path: Path, *, mode: str = "success", hooks: CapsuleHooks | None = None) -> Any:
    home, manifest, machine = _prepare(tmp_path, mode=mode)
    return run_capsule(
        manifest,
        _manifest_bytes(manifest),
        machine_profile=machine,
        build_id=_BUILD_ID,
        home=home,
        environ={},
        hooks=hooks or _hooks(tmp_path),
    )


def test_host_fake_end_to_end_success_and_exact_enclosing_evidence(tmp_path: Path) -> None:
    result = _run(tmp_path)

    assert result.outcome is RunOutcome.SUCCESS
    assert result.result.status == result.result.reason == "passed"
    entries = list(list_portable_entries(result.inspection.record))
    paths = [entry.logical_path for entry in entries]
    assert paths[:2] == ["capsule/build.json", "capsule/machine.json"]
    assert paths[-2:] == ["capsule/protocol/result.json", "capsule/result.json"]
    assert [entry.role for entry in entries[:2]] == ["build", "environment"]
    assert all(str(tmp_path / "home" / "builds") not in path for path in paths)
    by_path = {entry.logical_path: entry for entry in entries}
    assert all(by_path[ref.logical_path].blob_sha256 == ref.sha256 for ref in result.result.inputs)
    assert (
        result.result.protocol_result_sha256 == by_path["capsule/protocol/result.json"].blob_sha256
    )
    assert result.result.target == "topk-capsule"
    assert result.result.executable_sha256 == hashlib.sha256(_FAKE.read_bytes()).hexdigest()
    assert "protocol" not in result.result.model_dump(mode="json")
    resolved_bytes = read_record_member(result.inspection.record, "manifest.resolved.yaml")
    assert result.result.manifest_sha256 == hashlib.sha256(resolved_bytes).hexdigest()


def test_success_never_inherits_ambient_environment_and_always_cleans_scratch(
    tmp_path: Path,
) -> None:
    home, manifest, machine = _prepare(tmp_path)
    scratch_roots: list[Path] = []

    def temp_root() -> Path:
        value = Path(tempfile.mkdtemp(dir=tmp_path))
        scratch_roots.append(value)
        return value

    result = run_capsule(
        manifest,
        _manifest_bytes(manifest),
        machine_profile=machine,
        build_id=_BUILD_ID,
        home=home,
        environ={"AWS_SECRET_ACCESS_KEY": "ambient-secret-123456"},
        hooks=_hooks(tmp_path, temp_root_factory=temp_root),
    )

    assert result.outcome is RunOutcome.SUCCESS
    assert len(scratch_roots) == 1 and not scratch_roots[0].exists()


def test_ordinary_protocol_failure_finalizes_a_failed_enclosing_result(tmp_path: Path) -> None:
    result = _run(tmp_path, mode="correctness-fail")

    assert result.outcome is RunOutcome.FAILURE
    assert result.result.reason == "correctness-failed"
    assert result.result.protocol_result_sha256 is not None


def test_lock_refusal_publishes_structured_failure_without_protocol(tmp_path: Path) -> None:
    hooks = _hooks(tmp_path, machine_lock=_refusing_lock(tmp_path), protocol=lambda *_a, **_k: None)
    result = _run(tmp_path, hooks=hooks)

    assert result.outcome is RunOutcome.FAILURE
    assert result.result.reason == "lock-unavailable"
    assert result.result.protocol_result_sha256 is None
    assert [entry.logical_path for entry in list_portable_entries(result.inspection.record)] == [
        "capsule/build.json",
        "capsule/machine.json",
        "capsule/result.json",
    ]


@pytest.mark.parametrize(
    ("record_updates", "manifest_updates", "message"),
    [
        (
            {
                "source": fx.canonical_record().source.model_copy(
                    update={"source_evidence": {"source_id": "wrong", "base_commit": "a" * 40}}
                )
            },
            {},
            "source id",
        ),
        (
            {
                "source": fx.canonical_record().source.model_copy(
                    update={
                        "source_evidence": {"source_id": "topk-source", "base_commit": "c" * 40}
                    }
                )
            },
            {},
            "source commit",
        ),
        ({"toolchain_mode": "rocm"}, {}, "toolchain mode"),
        ({"selections": ()}, {}, "does not record"),
        ({"selections": (IdentityEntryV1(name="gfx_targets", value="gfx1100"),)}, {}, "gfx target"),
        ({}, {"target": "missing-target"}, "missing or ambiguous"),
    ],
)
def test_coordinate_and_target_mismatches_fail_before_run(
    tmp_path: Path,
    record_updates: dict[str, Any],
    manifest_updates: dict[str, str],
    message: str,
) -> None:
    home, manifest, machine = _prepare(tmp_path, record_updates=record_updates)
    manifest = _manifest(**manifest_updates)

    with pytest.raises(CapsuleRunError, match=message):
        run_capsule(
            manifest,
            _manifest_bytes(manifest),
            machine_profile=machine,
            build_id=_BUILD_ID,
            home=home,
            environ={},
        )
    assert not (home / "runs").exists()


def test_machine_mismatch_and_missing_build_allocate_no_run(tmp_path: Path) -> None:
    home, manifest, _machine_profile = _prepare(tmp_path)
    with pytest.raises(CapsuleRunError, match="machine profile"):
        run_capsule(
            manifest,
            _manifest_bytes(manifest),
            machine_profile=_machine(tmp_path, machine_id="other-machine"),
            build_id=_BUILD_ID,
            home=home,
            environ={},
        )
    with pytest.raises(CapsuleRunError, match="build lease failed"):
        run_capsule(
            manifest,
            _manifest_bytes(manifest),
            machine_profile=_machine(tmp_path),
            build_id="build-sha256:" + "9" * 64,
            home=home,
            environ={},
        )
    assert not (home / "runs").exists()


def test_preallocation_lease_verify_failure_creates_no_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, manifest, machine = _prepare(tmp_path)
    monkeypatch.setattr(
        capsule_runs.BuildLease,
        "verify",
        lambda _lease: (_ for _ in ()).throw(BuildCacheError("drift")),
    )
    with pytest.raises(CapsuleRunError, match="build lease failed"):
        run_capsule(
            manifest,
            _manifest_bytes(manifest),
            machine_profile=machine,
            build_id=_BUILD_ID,
            home=home,
            environ={},
        )
    assert not (home / "runs").exists()


def test_begin_run_oserror_is_a_preallocation_capsule_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, manifest, machine = _prepare(tmp_path)
    monkeypatch.setattr(
        capsule_runs,
        "begin_run",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("allocation storage failed")),
    )

    with pytest.raises(CapsuleRunError, match="allocation failed"):
        run_capsule(
            manifest,
            _manifest_bytes(manifest),
            machine_profile=machine,
            build_id=_BUILD_ID,
            home=home,
            environ={},
        )
    assert not (home / "runs").exists()


def test_injected_protocol_result_without_matching_evidence_fails_postallocation(
    tmp_path: Path,
) -> None:
    def protocol_without_evidence(
        _run: object, manifest: CapsuleManifestV1, **kwargs: Any
    ) -> CapsuleProtocolResultV1:
        process = CapsuleProcessV1(
            outcome="exited",
            returncode=1,
            duration_seconds=0.0,
            stdout_bytes=0,
            stderr_bytes=0,
            stdout_sha256="0" * 64,
            stderr_sha256="0" * 64,
            stdout_complete=True,
            stderr_complete=True,
            stdout_truncated=False,
            stderr_truncated=False,
            capture_error=False,
            category="nonzero-exit",
        )
        return CapsuleProtocolResultV1(
            capsule_id=manifest.id,
            candidate=manifest.candidate,
            scenario_sha256=manifest.contract.scenario_sha256,
            manifest_sha256=kwargs["manifest_sha256"],
            executable_sha256=kwargs["executable_sha256"],
            status="failed",
            reason="describe-process-failed",
            phases=(
                CapsulePhaseResultV1(
                    operation="describe",
                    request_sha256="1" * 64,
                    process=process,
                    response_sha256=None,
                    accepted=False,
                    failure="process",
                ),
            ),
            scenario=None,
            correctness=None,
            benchmark=None,
        )

    with pytest.raises(CapsuleExecutionError) as excinfo:
        _run(tmp_path, hooks=_hooks(tmp_path, protocol=protocol_without_evidence))
    assert excinfo.value.record is not None
    paths = {entry.logical_path for entry in list_portable_entries(excinfo.value.record)}
    assert "capsule/result.json" not in paths
    assert "capsule/protocol/result.json" not in paths


@pytest.mark.parametrize("failure", ["publication", "environment", "protocol", "cleanup", "drift"])
def test_postallocation_failures_are_fixed_safe_execution_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    home, manifest, machine = _prepare(tmp_path)
    hooks = _hooks(tmp_path)
    if failure == "publication":
        monkeypatch.setattr(
            capsule_runs,
            "_publish_inputs",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("secret publication")),
        )
    elif failure == "environment":
        monkeypatch.setattr(
            capsule_runs,
            "reconstruct_environment",
            lambda *_a, **_k: (_ for _ in ()).throw(CapsuleRunError("secret env")),
        )
    elif failure == "protocol":
        hooks = _hooks(
            tmp_path, protocol=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("secret child"))
        )
    elif failure == "cleanup":
        monkeypatch.setattr(
            shutil, "rmtree", lambda *_a, **_k: (_ for _ in ()).throw(OSError("secret cleanup"))
        )
    else:
        calls = 0
        original = capsule_runs.BuildLease.verify

        def verify(lease: Any) -> None:
            nonlocal calls
            calls += 1
            if calls >= 2:
                raise BuildCacheError("secret drift")
            original(lease)

        monkeypatch.setattr(capsule_runs.BuildLease, "verify", verify)

    with pytest.raises(CapsuleExecutionError) as excinfo:
        run_capsule(
            manifest,
            _manifest_bytes(manifest),
            machine_profile=machine,
            build_id=_BUILD_ID,
            home=home,
            environ={},
            hooks=hooks,
        )
    error = excinfo.value
    assert str(error) == "capsule run failed before producing a structured result"
    assert error.run_id and error.record is not None
    paths = {entry.logical_path for entry in list_portable_entries(error.record)}
    assert "capsule/result.json" not in paths
    if failure == "publication":
        assert paths == set()
    else:
        assert {"capsule/build.json", "capsule/machine.json"} <= paths


@pytest.mark.parametrize(
    ("name", "value", "environ"),
    [
        ("UNSAFE", "${AWS_SECRET_ACCESS_KEY}", {"AWS_SECRET_ACCESS_KEY": "ambient-secret"}),
        ("API_TOKEN", "canonical-child-secret-123456", {}),
    ],
)
def test_snapshot_safety_preflight_is_atomic_for_interpolation_and_child_secrets(
    tmp_path: Path, name: str, value: str, environ: dict[str, str]
) -> None:
    home, manifest, machine = _prepare(tmp_path)
    canonical = fx.canonical_record().model_copy(
        update={
            "environment": fx.default_environment() + (IdentityEntryV1(name=name, value=value),)
        }
    )
    # Replace the cached canonical through the normal fixture path so publication itself is tested.
    shutil.rmtree(home)
    home.mkdir(mode=0o700)
    fake = _FAKE.read_bytes()
    record = _record(fake, tmp_path / "state.json").model_copy(
        update={"environment": canonical.environment}
    )
    fx.make_present_build(home, record=record)
    executable = home / "builds" / "materialized" / _BUILD_ID / "bin" / "topk-capsule"
    executable.parent.mkdir(mode=0o700)
    executable.write_bytes(fake)
    executable.chmod(0o700)

    with pytest.raises(CapsuleExecutionError) as excinfo:
        run_capsule(
            manifest,
            _manifest_bytes(manifest),
            machine_profile=machine,
            build_id=_BUILD_ID,
            home=home,
            environ=environ,
            hooks=_hooks(tmp_path),
        )
    assert excinfo.value.record is not None
    assert list_portable_entries(excinfo.value.record) == ()


def test_snapshot_preflight_preserves_first_secret_across_duplicate_environment_names(
    tmp_path: Path,
) -> None:
    secret = "first-duplicate-secret-123456"
    environment = fx.default_environment() + (
        IdentityEntryV1(name="API_TOKEN", value=secret),
        IdentityEntryV1(name="API_TOKEN", value=""),
    )
    home, manifest, machine = _prepare(tmp_path, record_updates={"environment": environment})

    with pytest.raises(CapsuleExecutionError) as excinfo:
        run_capsule(
            manifest,
            _manifest_bytes(manifest),
            machine_profile=machine,
            build_id=_BUILD_ID,
            home=home,
            environ={},
            hooks=_hooks(tmp_path),
        )

    record = excinfo.value.record
    assert record is not None
    assert list_portable_entries(record) == ()
    secret_bytes = secret.encode("utf-8")
    assert all(
        secret_bytes not in path.read_bytes() for path in record.rglob("*") if path.is_file()
    )


def test_snapshot_pair_preflights_machine_payload_against_ambient_secrets(
    tmp_path: Path,
) -> None:
    home, manifest, machine = _prepare(tmp_path)
    secret = "ambient-lock-secret-123456"
    machine = machine.model_copy(
        update={"exclusive_lock": ExclusiveLockV1(path=str(tmp_path / secret / "lock"))}
    )

    with pytest.raises(CapsuleExecutionError) as excinfo:
        run_capsule(
            manifest,
            _manifest_bytes(manifest),
            machine_profile=machine,
            build_id=_BUILD_ID,
            home=home,
            environ={"MACHINE_LOCK_TOKEN": secret},
            hooks=_hooks(tmp_path),
        )
    assert excinfo.value.record is not None
    assert list_portable_entries(excinfo.value.record) == ()


def test_enclosing_models_reject_incoherent_inputs_and_status(tmp_path: Path) -> None:
    result = _run(tmp_path).result
    payload = result.model_dump(mode="python")
    payload["status"] = "failed"
    with pytest.raises(ValidationError, match="closed protocol reason"):
        CapsuleResultV1.model_validate(payload)
    payload = result.model_dump(mode="python")
    payload["inputs"] = tuple(reversed(payload["inputs"]))
    with pytest.raises(ValidationError, match="exact ordered"):
        CapsuleResultV1.model_validate(payload)
    payload = result.model_dump(mode="python")
    payload["protocol_result_sha256"] = None
    with pytest.raises(ValidationError, match="bind its protocol result"):
        CapsuleResultV1.model_validate(payload)


@pytest.mark.parametrize("field", ["executable_sha256", "protocol_result_sha256"])
def test_enclosing_digest_fields_reject_host_paths(tmp_path: Path, field: str) -> None:
    payload = _run(tmp_path).result.model_dump(mode="python")
    payload[field] = "/host/absolute/path"
    with pytest.raises(ValidationError):
        CapsuleResultV1.model_validate(payload)

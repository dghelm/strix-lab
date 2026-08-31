from __future__ import annotations

import contextlib
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any, get_args

import _suite_fixtures as fx
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import strixlab.build_cache as cache_module
from strixlab.adapters import backend_ops as _bo
from strixlab.adapters import llama_bench as _lb
from strixlab.adapters import llama_server as _ls
from strixlab.build_artifacts import ArtifactV1, BuildArtifactsV1, TargetArtifactsV1
from strixlab.build_cache import (
    BuildCacheError,
    IdentityEntryV1,
    cleanup_build,
    lease_build,
)
from strixlab.cli import app
from strixlab.config import read_manifest
from strixlab.evidence import RunOutcome, list_portable_entries, recover_runs
from strixlab.locks import LockAttempt, LockStatus
from strixlab.manifests import (
    ExclusiveLockV1,
    MachineExpectationV1,
    MachineProfileV1,
    MachineValidityV1,
    SuiteManifestV1,
    TelemetryV1,
    suite_measurement_case_id,
    suite_warmup_case_id,
    validate_manifest,
)
from strixlab.models import (
    ModelError,
    ModelReceiptEvidenceV1,
    load_model_receipt,
    receipt_evidence_digest,
)
from strixlab.suites import (
    SuiteError,
    SuiteExecutionError,
    SuiteHooks,
    _backend_category,
    _bench_category,
    _reconstruct_environment,
    _resolve_target_executable,
    _server_category,
    plan_performance,
    run_suite,
)

_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "suites" / "smoke-qwen35.yaml"


@pytest.fixture(autouse=True)
def _stub(monkeypatch: pytest.MonkeyPatch) -> None:
    fx.stub_cache_verification(monkeypatch)


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


def acquiring_lock(path: Path) -> Any:
    @contextlib.contextmanager
    def factory(lock_path: Path) -> Iterator[LockAttempt]:
        yield LockAttempt(LockStatus.ACQUIRED, lock_path)

    return factory


def refusing_lock(path: Path) -> Any:
    @contextlib.contextmanager
    def factory(lock_path: Path) -> Iterator[LockAttempt]:
        yield LockAttempt(LockStatus.CONTENDED, lock_path, "busy")

    return factory


def _hooks(tmp_path: Path, **overrides: Any) -> SuiteHooks:
    def temp_root() -> Path:
        return Path(tempfile.mkdtemp(dir=tmp_path))

    defaults: dict[str, Any] = {
        "temp_root_factory": temp_root,
        "machine_lock": acquiring_lock(tmp_path),
        "backend_ops": fx.fake_backend(),
        "llama_server": fx.fake_server(),
        "llama_bench": fx.fake_bench(),
    }
    defaults.update(overrides)
    return SuiteHooks(**defaults)


def _prepare(tmp_path: Path) -> tuple[SuiteManifestV1, MachineProfileV1, Path, str]:
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o700)
    fx.make_present_build(home)
    receipt_sha = fx.publish_receipt(home, tmp_path / "scratch")
    return _manifest(), _machine(tmp_path), home, receipt_sha


def _run(tmp_path: Path, **hook_overrides: Any) -> Any:
    manifest, machine, home, receipt_sha = _prepare(tmp_path)
    return run_suite(
        manifest,
        _CONFIG.read_bytes(),
        machine_profile=machine,
        build_id=fx.BUILD_ID,
        local_receipt_sha256=receipt_sha,
        home=home,
        environ={"PATH": "/usr/bin"},
        hooks=_hooks(tmp_path, **hook_overrides),
    )


# --- Planner ------------------------------------------------------------------


def test_plan_performance_is_windowed_interleaved() -> None:
    manifest = _manifest()
    planned = plan_performance(manifest.performance)
    ids = [case.adapter_case_id for case in planned]
    # 3 cases * (2 warmup rounds + 5 windows) = 21 planned invocations.
    assert len(planned) == 21
    assert ids[:6] == [
        suite_warmup_case_id(1, "pp512"),
        suite_warmup_case_id(1, "pp2048"),
        suite_warmup_case_id(1, "tg128"),
        suite_warmup_case_id(2, "pp512"),
        suite_warmup_case_id(2, "pp2048"),
        suite_warmup_case_id(2, "tg128"),
    ]
    assert ids[6:9] == [
        suite_measurement_case_id(1, "pp512"),
        suite_measurement_case_id(1, "pp2048"),
        suite_measurement_case_id(1, "tg128"),
    ]
    warmups = [case for case in planned if case.phase == "warmup"]
    measures = [case for case in planned if case.phase == "measurement"]
    assert all(case.repetitions == 1 for case in warmups)
    assert all(case.repetitions == 3 for case in measures)
    assert len(set(ids)) == len(ids)


# --- Happy path ---------------------------------------------------------------


def test_successful_suite_finalizes_success(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.outcome is RunOutcome.SUCCESS
    assert result.result.status == "passed"
    assert result.result.reason == "passed"
    assert result.inspection.outcome is RunOutcome.SUCCESS
    schedule = result.result.schedule
    assert schedule.planned_warmups == 6
    assert schedule.planned_measurements == 15
    assert schedule.completed_warmups == 6
    assert schedule.completed_measurements == 15
    assert len(result.result.measurements) == 15
    assert result.result.backend_ops is not None and result.result.backend_ops.passed
    assert result.result.greedy is not None and result.result.greedy.passed


def test_success_publishes_authenticated_portable_snapshots(tmp_path: Path) -> None:
    result = _run(tmp_path)
    entries = {
        entry.logical_path: entry for entry in list_portable_entries(result.inspection.record)
    }
    for logical_path in (
        "suite/build.json",
        "suite/model.json",
        "suite/machine.json",
        "suite/result.json",
    ):
        assert logical_path in entries
    # Every input ref binds a real portable blob digest.
    for ref in result.result.inputs:
        assert entries[ref.logical_path].blob_sha256 == ref.sha256
    # Every completed adapter sample reference authenticates against its portable blob,
    # including the llama-server sample (now published portably as well as locally).
    assert {sample.adapter for sample in result.result.samples} == {
        "backend-ops",
        "llama-server",
        "llama-bench",
    }
    for sample in result.result.samples:
        assert entries[sample.logical_path].blob_sha256 == sample.sample_sha256


# --- Manifest bounds and validation -------------------------------------------


def _manifest_dict() -> dict[str, Any]:
    return deepcopy(read_manifest(_CONFIG))


def test_deterministic_prompt_capture_is_exact() -> None:
    manifest = _manifest()
    text = manifest.correctness.greedy.prompts[0].text
    assert text == (
        "Continue this sequence with only the next eight integers separated by spaces:\n"
        "1 1 2 3 5 8 13 21"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda value: value["correctness"]["greedy"].__setitem__(
                "prompts",
                [{"id": f"p{i}", "text": "x"} for i in range(5)],
            ),
            id="too-many-prompts",
        ),
        pytest.param(
            lambda value: value["performance"].update(
                warmup_runs=0,
                measurement_windows=8,
                repetitions_per_window=32,
                cases=[
                    {"id": f"c{i}", "prompt_tokens": 1, "generated_tokens": 0} for i in range(8)
                ],
            ),
            id="repetition-budget",
        ),
        pytest.param(
            lambda value: value["performance"]["cases"][0].update(generated_tokens=1),
            id="two-nonzero-metrics",
        ),
        pytest.param(
            lambda value: value["performance"]["cases"][0].update(
                prompt_tokens=0, generated_tokens=0
            ),
            id="zero-metrics",
        ),
        pytest.param(
            lambda value: value["performance"]["cases"][0].update(id="x" * 60),
            id="generated-id-too-long",
        ),
        pytest.param(
            lambda value: value["correctness"]["greedy"].__setitem__(
                "prompts",
                [{"id": f"p{i}", "text": "y" * 11000} for i in range(3)],
            ),
            id="prompt-aggregate-bytes",
        ),
        pytest.param(
            lambda value: value["correctness"]["backend_ops"].update(
                operations=["MUL_MAT", "MUL_MAT"]
            ),
            id="duplicate-operations",
        ),
    ],
)
def test_suite_manifest_rejects_out_of_bounds(mutate: Any) -> None:
    value = _manifest_dict()
    mutate(value)
    with pytest.raises(ValidationError):
        validate_manifest("suite", value)


def test_suite_manifest_accepts_checked_in_config() -> None:
    assert isinstance(validate_manifest("suite", _manifest_dict()), SuiteManifestV1)


# --- Build lease --------------------------------------------------------------


def _home_with_build(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o700)
    fx.make_present_build(home)
    return home


def test_lease_build_yields_present_attested(tmp_path: Path) -> None:
    home = _home_with_build(tmp_path)
    with lease_build(fx.BUILD_ID, home=home) as lease:
        assert lease.build_id == fx.BUILD_ID
        assert lease.canonical.toolchain_mode == "rocm"
        assert lease.root.is_dir()
        lease.verify()


def test_lease_build_refuses_cleaned_build(tmp_path: Path) -> None:
    home = _home_with_build(tmp_path)
    cleanup_build(fx.BUILD_ID, home=home)
    with pytest.raises(BuildCacheError, match="not present"), lease_build(fx.BUILD_ID, home=home):
        pass


def test_lease_build_refuses_unattested_build(tmp_path: Path) -> None:
    home = _home_with_build(tmp_path)
    layout = cache_module._layout(home, create=False)
    cache_module._attestation_path(layout, fx.BUILD_ID).unlink()
    with pytest.raises(BuildCacheError, match="not attested"), lease_build(fx.BUILD_ID, home=home):
        pass


def test_lease_build_refuses_missing_build(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o700)
    cache_module._layout(home, create=True)
    with pytest.raises(BuildCacheError), lease_build(fx.BUILD_ID, home=home):
        pass


def test_lease_verify_detects_root_replacement(tmp_path: Path) -> None:
    home = _home_with_build(tmp_path)
    with lease_build(fx.BUILD_ID, home=home) as lease:
        moved = lease.root.parent / "moved"
        lease.root.rename(moved)
        lease.root.mkdir(mode=0o700)
        try:
            with pytest.raises(BuildCacheError):
                lease.verify()
        finally:
            lease.root.rmdir()
            moved.rename(lease.root)


def test_lease_release_on_exception(tmp_path: Path) -> None:
    home = _home_with_build(tmp_path)
    with pytest.raises(RuntimeError, match="boom"), lease_build(fx.BUILD_ID, home=home):
        raise RuntimeError("boom")
    # The lock is released, so a second lease acquires cleanly.
    with lease_build(fx.BUILD_ID, home=home) as lease:
        assert lease.build_id == fx.BUILD_ID


_CHILD_CLEANUP = textwrap.dedent(
    """
    import sys
    from pathlib import Path
    from strixlab.build_cache import cleanup_build, BuildCacheError
    try:
        cleanup_build(sys.argv[2], home=Path(sys.argv[1]))
    except BuildCacheError as exc:
        print("LOCKED" if "lock" in str(exc) else f"OTHER:{exc}")
        sys.exit(0)
    print("CLEANED")
    sys.exit(1)
    """
)


def test_lease_lock_excludes_cleanup_cross_process(tmp_path: Path) -> None:
    home = _home_with_build(tmp_path)
    with lease_build(fx.BUILD_ID, home=home):
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_CLEANUP, str(home), fx.BUILD_ID],
            capture_output=True,
            text=True,
            timeout=60,
        )
    assert result.stdout.strip() == "LOCKED", result.stderr
    # The build is still cleanable once the lease is released.
    cleanup_build(fx.BUILD_ID, home=home)


# --- Target resolution --------------------------------------------------------


def _artifacts_with(
    targets: tuple[TargetArtifactsV1, ...], artifacts: tuple[ArtifactV1, ...]
) -> BuildArtifactsV1:
    return BuildArtifactsV1(
        artifact_set_id="artifact-set-sha256:" + "cd" * 32,
        targets=targets,
        artifacts=artifacts,
        inspections=(),
        capture_tools=(),
        cmake_cache_sha256="cd" * 32,
        compile_commands_sha256=None,
    )


def test_resolve_target_executable_success(tmp_path: Path) -> None:
    artifacts = fx._artifacts()
    path, sha = _resolve_target_executable(artifacts, "llama-bench", tmp_path)
    assert path == str(tmp_path / "bin/llama-bench")
    assert sha == "cd" * 32


@pytest.mark.parametrize(
    ("targets", "artifacts", "match"),
    [
        ((), (), "missing or ambiguous"),
        (
            (
                TargetArtifactsV1(
                    name="llama-bench",
                    target_id="a",
                    target_type="EXECUTABLE",
                    artifacts=("bin/llama-bench",),
                ),
            ),
            (
                ArtifactV1(
                    path="bin/llama-bench",
                    kind="archive",
                    mode=0o644,
                    size_bytes=4,
                    sha256="cd" * 32,
                    targets=("llama-bench",),
                ),
            ),
            "exactly one executable",
        ),
        (
            (
                TargetArtifactsV1(
                    name="llama-bench",
                    target_id="a",
                    target_type="SHARED_LIBRARY",
                    artifacts=("bin/llama-bench",),
                ),
            ),
            (
                ArtifactV1(
                    path="bin/llama-bench",
                    kind="elf",
                    elf_type="ET_DYN",
                    mode=0o755,
                    size_bytes=4,
                    sha256="cd" * 32,
                    targets=("llama-bench",),
                ),
            ),
            "not an executable",
        ),
        (
            (
                TargetArtifactsV1(
                    name="llama-bench",
                    target_id="a",
                    target_type="EXECUTABLE",
                    artifacts=("../escape",),
                ),
            ),
            (
                ArtifactV1(
                    path="../escape",
                    kind="elf",
                    elf_type="ET_EXEC",
                    mode=0o755,
                    size_bytes=4,
                    sha256="cd" * 32,
                    targets=("llama-bench",),
                ),
            ),
            "escapes the leased root",
        ),
    ],
)
def test_resolve_target_executable_rejects(
    targets: Any, artifacts: Any, match: str, tmp_path: Path
) -> None:
    with pytest.raises(SuiteError, match=match):
        _resolve_target_executable(_artifacts_with(targets, artifacts), "llama-bench", tmp_path)


# --- Build binding ------------------------------------------------------------


def _run_with_record(tmp_path: Path, record: Any, **hook_overrides: Any) -> Any:
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o700)
    fx.make_present_build(home, record=record)
    receipt_sha = fx.publish_receipt(home, tmp_path / "scratch")
    return run_suite(
        _manifest(),
        _CONFIG.read_bytes(),
        machine_profile=_machine(tmp_path),
        build_id=fx.BUILD_ID,
        local_receipt_sha256=receipt_sha,
        home=home,
        environ={"PATH": "/usr/bin"},
        hooks=_hooks(tmp_path, **hook_overrides),
    )


def test_machine_id_mismatch_is_rejected(tmp_path: Path) -> None:
    manifest, machine, home, receipt_sha = _prepare(tmp_path)
    bad = machine.model_copy(update={"id": "other-machine"})
    with pytest.raises(SuiteError, match="machine profile id"):
        run_suite(
            manifest,
            _CONFIG.read_bytes(),
            machine_profile=bad,
            build_id=fx.BUILD_ID,
            local_receipt_sha256=receipt_sha,
            home=home,
            environ={"PATH": "/usr/bin"},
            hooks=_hooks(tmp_path),
        )


@pytest.mark.parametrize(
    ("record", "match"),
    [
        (fx.canonical_record(toolchain_mode="host"), "toolchain mode"),
        (
            fx.canonical_record(
                selections=(
                    IdentityEntryV1(name="generator", value="Ninja"),
                    IdentityEntryV1(name="gfx_targets", value="gfx1100"),
                )
            ),
            "gfx target",
        ),
        (fx.canonical_record(source_id="other-source"), "source id"),
        (fx.canonical_record(base_commit="b" * 40), "source commit"),
    ],
)
def test_build_binding_failures(tmp_path: Path, record: Any, match: str) -> None:
    with pytest.raises(SuiteError, match=match):
        _run_with_record(tmp_path, record)


# --- Hermetic runtime environment ---------------------------------------------


def _reconstruct(tmp_path: Path, environment: tuple[IdentityEntryV1, ...]) -> Any:
    record = fx.canonical_record(environment=environment)
    root = tmp_path / "leased-root"
    root.mkdir()
    scratch = tmp_path / "scratch-env"
    scratch.mkdir()
    return _reconstruct_environment(record, root, scratch)


def test_environment_rehydrates_build_root_and_replaces_home_tmp(tmp_path: Path) -> None:
    runtime = _reconstruct(tmp_path, fx.default_environment())
    root = str(tmp_path / "leased-root")
    assert runtime.environment["LD_LIBRARY_PATH"] == f"{root}/lib:/usr/lib"
    assert runtime.environment["HOME"] == str(tmp_path / "scratch-env" / "home")
    assert runtime.environment["TMPDIR"] == str(tmp_path / "scratch-env" / "tmp")
    assert (tmp_path / "scratch-env" / "home").is_dir()
    assert (tmp_path / "scratch-env" / "tmp").is_dir()
    # No ambient inheritance: only the canonical entries appear, PATH is the recorded one.
    assert set(runtime.environment) == {
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "TZ",
        "PATH",
        "ROCM_PATH",
        "LD_LIBRARY_PATH",
        "SOURCE_DATE_EPOCH",
    }
    assert runtime.environment["PATH"] == "/opt/rocm-10/bin:/usr/bin"
    assert runtime.environment["SOURCE_DATE_EPOCH"] == "0"


def _env_without(name: str) -> tuple[IdentityEntryV1, ...]:
    return tuple(entry for entry in fx.default_environment() if entry.name != name)


def _env_with(name: str, value: str) -> tuple[IdentityEntryV1, ...]:
    return _env_without(name) + (IdentityEntryV1(name=name, value=value),)


@pytest.mark.parametrize(
    ("environment", "match"),
    [
        (_env_with("EXTRA", "{SOURCE_ROOT}/x"), "placeholder"),
        (_env_with("EXTRA", "{BUILD_HOME}/x"), "placeholder"),
        (_env_with("EXTRA", "{UNKNOWN}"), "placeholder"),
        (_env_with("PATH", "/foo/{BUILD_ROOT}/bar"), "placeholder"),
        (_env_without("HOME"), "missing HOME"),
        (_env_without("TMPDIR"), "missing TMPDIR"),
        (_env_with("LANG", "en_US"), "LANG"),
        (_env_with("TZ", "PST"), "TZ"),
        (
            fx.default_environment() + (IdentityEntryV1(name="PATH", value="/dup"),),
            "duplicate",
        ),
        (_env_with("bad name", "x"), "invalid name"),
    ],
)
def test_environment_rejects_bad_grammar(
    tmp_path: Path, environment: tuple[IdentityEntryV1, ...], match: str
) -> None:
    with pytest.raises(SuiteError, match=match):
        _reconstruct(tmp_path, environment)


def test_environment_scratch_is_removed_after_run(tmp_path: Path) -> None:
    created: list[Path] = []

    def temp_root() -> Path:
        path = Path(tempfile.mkdtemp(dir=tmp_path))
        created.append(path)
        return path

    _run(tmp_path, temp_root_factory=temp_root)
    assert created and all(not path.exists() for path in created)


# --- Correctness-first protocol -----------------------------------------------


def _tripwire(name: str) -> Any:
    def runner(**_kwargs: Any) -> Any:
        raise AssertionError(f"{name} must not run after an earlier failure")

    return runner


def test_backend_failure_stops_before_performance(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        backend_ops=fx.fake_backend(fx.backend_gate_failed),
        llama_server=_tripwire("server"),
        llama_bench=_tripwire("bench"),
    )
    assert result.outcome is RunOutcome.FAILURE
    assert result.result.reason == "backend-ops-failed"
    assert result.result.greedy is None
    assert result.result.measurements == ()
    assert result.result.backend_ops is not None and not result.result.backend_ops.passed


def test_greedy_failure_stops_before_performance(tmp_path: Path) -> None:
    def unequal(inputs: Any, case: Any) -> Any:
        return fx.server_success(inputs, case, tokens_a=(1, 2, 3), tokens_b=(1, 2, 9))

    result = _run(
        tmp_path,
        llama_server=fx.fake_server(unequal),
        llama_bench=_tripwire("bench"),
    )
    assert result.outcome is RunOutcome.FAILURE
    assert result.result.reason == "greedy-parity-failed"
    assert result.result.greedy is not None and not result.result.greedy.passed
    verdict = result.result.greedy.prompts[0]
    assert not verdict.tokens_equal and not verdict.passed


@pytest.mark.parametrize(
    ("sample_fn", "reason"),
    [
        (lambda i, c: fx.server_success(i, c, tokens_a=()), "greedy-parity-failed"),
        (fx.server_capability_failed, "greedy-parity-failed"),
    ],
)
def test_greedy_variants(tmp_path: Path, sample_fn: Any, reason: str) -> None:
    result = _run(tmp_path, llama_server=fx.fake_server(sample_fn), llama_bench=_tripwire("bench"))
    assert result.outcome is RunOutcome.FAILURE
    assert result.result.reason == reason


def test_greedy_unpersisted_sample_is_integrity_failure(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        llama_server=fx.fake_server(persist="none"),
        llama_bench=_tripwire("bench"),
    )
    assert result.outcome is RunOutcome.FAILURE
    assert result.result.reason == "integrity-failed"


@pytest.mark.parametrize(
    "overrides",
    [
        {"backend_ops": fx.fake_backend(persist="local")},
        {"llama_server": fx.fake_server(persist="local"), "llama_bench": None},
        {"llama_bench": fx.fake_bench(persist="local")},
    ],
)
def test_local_only_sample_is_rejected(tmp_path: Path, overrides: dict[str, Any]) -> None:
    # A sample.json written only as local evidence (no portable entry) fails closed:
    # portable authentication is required for every adapter sample.
    hooks = {key: value for key, value in overrides.items() if value is not None}
    result = _run(tmp_path, **hooks)
    assert result.outcome is RunOutcome.FAILURE
    assert result.result.reason == "integrity-failed"


def test_backend_passed_status_with_failed_gate_is_rejected(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        backend_ops=fx.fake_backend(fx.backend_passed_status_failed_gate),
        llama_server=_tripwire("server"),
        llama_bench=_tripwire("bench"),
    )
    assert result.outcome is RunOutcome.FAILURE
    assert result.result.reason == "backend-ops-failed"
    assert result.result.backend_ops is not None
    assert not result.result.backend_ops.passed
    assert not result.result.backend_ops.gate_passed


def test_backend_thrown_integrity_is_integrity_failure(tmp_path: Path) -> None:
    from strixlab.adapters.backend_ops import BackendOpsIntegrityError

    result = _run(
        tmp_path,
        backend_ops=fx.raising_runner(BackendOpsIntegrityError("drift")),
        llama_server=_tripwire("server"),
    )
    assert result.outcome is RunOutcome.FAILURE
    assert result.result.reason == "integrity-failed"
    assert result.result.backend_ops is None


# --- Performance schedule -----------------------------------------------------


def test_measurement_failure_stops_with_partial_projection(tmp_path: Path) -> None:
    fail_id = suite_measurement_case_id(2, "pp2048")

    def bench(inputs: Any, case: Any) -> Any:
        if case.id == fail_id:
            return fx.bench_process_failed(inputs, case)
        return fx.bench_success(inputs, case)

    result = _run(tmp_path, llama_bench=fx.fake_bench(bench))
    assert result.outcome is RunOutcome.FAILURE
    assert result.result.reason == "measurement-failed"
    # Window 1 (3 cases) plus window 2 pp512 succeeded before pp2048 failed: 4 projected.
    assert len(result.result.measurements) == 4
    windows = {(m.case_id, m.window) for m in result.result.measurements}
    assert (fail_id, 2) not in {(m.adapter_case_id, m.window) for m in result.result.measurements}
    assert ("pp512", 1) in windows
    # The authenticated failure sample also counts as completed evidence: 4 + 1 = 5.
    assert result.result.schedule.completed_measurements == 5


def test_warmup_failure_stops_before_measurement(tmp_path: Path) -> None:
    fail_id = suite_warmup_case_id(1, "tg128")

    def bench(inputs: Any, case: Any) -> Any:
        if case.id == fail_id:
            return fx.bench_process_failed(inputs, case)
        return fx.bench_success(inputs, case)

    result = _run(tmp_path, llama_bench=fx.fake_bench(bench))
    assert result.outcome is RunOutcome.FAILURE
    assert result.result.reason == "warmup-failed"
    assert result.result.measurements == ()


# --- Lock refusal and integrity finalization ----------------------------------


def test_lock_unavailable_finalizes_failure(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        machine_lock=refusing_lock(tmp_path),
        backend_ops=_tripwire("backend"),
    )
    assert result.outcome is RunOutcome.FAILURE
    assert result.result.reason == "lock-unavailable"
    entries = {e.logical_path for e in list_portable_entries(result.inspection.record)}
    assert "suite/result.json" in entries


def test_snapshot_publish_failure_finalizes_without_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from strixlab.evidence import RunError

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RunError("evidence store integrity failure")

    monkeypatch.setattr("strixlab.suites._publish_input_snapshots", boom)
    manifest, machine, home, receipt_sha = _prepare(tmp_path)
    with pytest.raises(SuiteExecutionError) as excinfo:
        run_suite(
            manifest,
            _CONFIG.read_bytes(),
            machine_profile=machine,
            build_id=fx.BUILD_ID,
            local_receipt_sha256=receipt_sha,
            home=home,
            environ={"PATH": "/usr/bin"},
            hooks=_hooks(tmp_path),
        )
    error = excinfo.value
    assert error.run_id and error.record is not None
    entries = {e.logical_path for e in list_portable_entries(error.record)}
    assert "suite/result.json" not in entries
    assert "suite/build.json" not in entries


def test_scratch_cleanup_failure_finalizes_without_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    manifest, machine, home, receipt_sha = _prepare(tmp_path)

    def boom(_path: Any) -> None:
        raise OSError("synthetic scratch removal failure")

    monkeypatch.setattr(shutil, "rmtree", boom)
    with pytest.raises(SuiteExecutionError) as excinfo:
        run_suite(
            manifest,
            _CONFIG.read_bytes(),
            machine_profile=machine,
            build_id=fx.BUILD_ID,
            local_receipt_sha256=receipt_sha,
            home=home,
            environ={"PATH": "/usr/bin"},
            hooks=_hooks(tmp_path),
        )
    error = excinfo.value
    assert error.run_id and error.record is not None
    # The run finalized failure without publishing a successful structured result.
    entries = {e.logical_path for e in list_portable_entries(error.record)}
    assert "suite/result.json" not in entries


def test_bad_receipt_before_run_creates_no_run(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o700)
    fx.make_present_build(home)
    fx.publish_receipt(home, tmp_path / "scratch")
    with pytest.raises(SuiteError, match="receipt"):
        run_suite(
            _manifest(),
            _CONFIG.read_bytes(),
            machine_profile=_machine(tmp_path),
            build_id=fx.BUILD_ID,
            local_receipt_sha256="ab" * 32,
            home=home,
            environ={"PATH": "/usr/bin"},
            hooks=_hooks(tmp_path),
        )


def test_model_execution_requirements_rejected_before_run(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o700)
    fx.make_present_build(home)
    receipt_sha = fx.publish_receipt_with_execution(
        home, tmp_path / "scratch", required_sources=("cuda-graphs",)
    )
    with pytest.raises(SuiteError, match="execution requirements"):
        run_suite(
            _manifest(),
            _CONFIG.read_bytes(),
            machine_profile=_machine(tmp_path),
            build_id=fx.BUILD_ID,
            local_receipt_sha256=receipt_sha,
            home=home,
            environ={"PATH": "/usr/bin"},
            hooks=_hooks(tmp_path),
        )


# --- Versioned receipt evidence compatibility ---------------------------------


def test_legacy_v1_receipt_is_readable_and_authentic(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o700)
    receipt_sha = fx.publish_legacy_v1_receipt(home, tmp_path / "scratch")
    # load_model_receipt re-authenticates against the content address, so a successful
    # load proves the legacy v1 receipt is still readable and authentic.
    receipt = load_model_receipt("qwen35-4b-smoke", receipt_sha, home=home)
    assert isinstance(receipt.evidence, ModelReceiptEvidenceV1)
    assert receipt.evidence.schema_version == 1
    assert receipt.trust_state == "verified"
    assert len(receipt_evidence_digest(receipt.evidence)) == 64


def test_legacy_v1_receipt_is_suite_ineligible(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o700)
    fx.make_present_build(home)
    receipt_sha = fx.publish_legacy_v1_receipt(home, tmp_path / "scratch")
    with pytest.raises(SuiteError, match="legacy v1"):
        run_suite(
            _manifest(),
            _CONFIG.read_bytes(),
            machine_profile=_machine(tmp_path),
            build_id=fx.BUILD_ID,
            local_receipt_sha256=receipt_sha,
            home=home,
            environ={"PATH": "/usr/bin"},
            hooks=_hooks(tmp_path),
        )
    # The rejection happens before begin_run, so no run was ever allocated.
    assert recover_runs(home=home) == ()


def test_v2_execution_tampering_fails_receipt_authentication(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o700)
    receipt = fx.build_smoke_receipt(tmp_path / "scratch")
    digest = fx.publish_receipt_object(home, receipt)
    # Overwrite the stored envelope with a tampered execution projection at the same
    # content address: re-reading must fail the content-address check.
    fx.tamper_receipt_execution(home, receipt, digest)
    with pytest.raises(ModelError):
        load_model_receipt("qwen35-4b-smoke", digest, home=home)


# --- CLI ----------------------------------------------------------------------


def test_cli_created_run_failure_prints_run_and_exits_nonzero(tmp_path: Path) -> None:
    # Default (real) adapters hash a missing/mismatched binary and fail closed, so a run
    # is created and finalized FAILURE with a structured integrity result.
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o700)
    fx.make_present_build(home)
    receipt_sha = fx.publish_receipt(home, tmp_path / "scratch")
    machine_path = tmp_path / "machine.yaml"
    machine_path.write_text(_machine_yaml(tmp_path), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "run",
            "suite",
            str(_CONFIG),
            "--machine",
            str(machine_path),
            "--build",
            fx.BUILD_ID,
            "--model-receipt",
            receipt_sha,
            "--home",
            str(home),
        ],
    )
    assert result.exit_code == 1
    assert "run:" in result.stderr
    assert "record:" in result.stderr


def test_cli_prerun_failure_prints_no_run(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o700)
    fx.make_present_build(home)
    fx.publish_receipt(home, tmp_path / "scratch")
    machine_path = tmp_path / "machine.yaml"
    machine_path.write_text(_machine_yaml(tmp_path), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "run",
            "suite",
            str(_CONFIG),
            "--machine",
            str(machine_path),
            "--build",
            fx.BUILD_ID,
            "--model-receipt",
            "ab" * 32,
            "--home",
            str(home),
        ],
    )
    assert result.exit_code == 1
    assert "run:" not in result.stderr
    assert "suite run failed" in result.stderr


def _machine_yaml(tmp_path: Path) -> str:
    return (
        "schema_version: 1\n"
        "id: strix-halo-128g\n"
        "expect:\n"
        "  gpu_arch: gfx1151\n"
        "  integrated_gpu: true\n"
        "  memory_gib_min: 64\n"
        f"exclusive_lock:\n  path: {tmp_path / 'gpu.lock'}\n"
        "telemetry:\n  amd_smi: auto\n  sample_interval_ms: 100\n"
        "validity:\n"
        "  require_ac_power: true\n"
        "  max_background_gpu_busy_pct: 10\n"
        "  min_available_memory_gib: 8\n"
        "  temperature_warn_c: 90\n"
    )


# --- Closed adapter-status mapping --------------------------------------------

# The llama-bench reasons that refine a process/output failure into a suite category.
_BENCH_PROCESS_REASONS = frozenset(
    {
        "spawn-failed",
        "timed-out",
        "capture-failed",
        "output-oversized",
        "encoding-failed",
        "nonzero-exit",
    }
)
# The remaining llama-bench reasons, covered by their own status rather than refined.
_STATUS_COVERED_BENCH_REASONS = frozenset({"success", "parse-failed", "capability-unsupported"})


def test_every_backend_status_maps_explicitly() -> None:
    for status in get_args(_bo.SampleStatus):
        assert _backend_category(status) != "adapter-failed", status


def test_every_server_status_maps_explicitly() -> None:
    for status in get_args(_ls.SampleStatus):
        assert _server_category(status) != "adapter-failed", status


def test_every_bench_status_and_process_reason_maps_explicitly() -> None:
    process_statuses = {"process-failed", "output-truncated"}
    statuses = set(get_args(_lb.SampleStatus))
    for status in statuses - process_statuses:
        assert _bench_category(status, "success") != "adapter-failed", status
    # Closed partition of the adapter's own Reason union: process/output defects the
    # suite refines vs reasons covered by their status. A new reason drifts here rather
    # than silently falling through to adapter-failed.
    assert set(get_args(_lb.Reason)) == _BENCH_PROCESS_REASONS | _STATUS_COVERED_BENCH_REASONS
    for status in process_statuses & statuses:
        for reason in _BENCH_PROCESS_REASONS:
            assert _bench_category(status, reason) != "adapter-failed", (status, reason)

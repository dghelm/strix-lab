from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

import strixlab.capsules as capsules
from strixlab.capsules import (
    EVIDENCE_ROOT,
    CapsuleIntegrityError,
    CapsulePhaseResultV1,
    CapsuleProtocolResultV1,
    run_capsule_protocol,
)
from strixlab.evidence import RunSession, begin_run, list_portable_entries
from strixlab.manifests import CapsuleManifestV1
from strixlab.process import ProcessOutcome, ProcessResult, run_process
from strixlab.secret_policy import RedactionContext
from strixlab.serialization import canonical_json_bytes

_FAKE = Path(__file__).parents[1] / "fixtures" / "fake_capsule.py"


def _manifest(*, timeout: float = 1.0) -> CapsuleManifestV1:
    return CapsuleManifestV1.model_validate(
        {
            "schema_version": 1,
            "id": "topk-capsule",
            "candidate": "candidate-a",
            "machine": "strix-halo-128g",
            "build": {
                "source_id": "topk-source",
                "source_commit": "a" * 40,
                "toolchain_mode": "host",
                "gfx_target": "gfx1151",
                "target": "topk-capsule",
            },
            "contract": {"protocol": "native-capsule-v1", "scenario_sha256": "b" * 64},
            "timeouts": {
                "describe_seconds": timeout,
                "correctness_seconds": timeout,
                "benchmark_seconds": timeout,
            },
        }
    )


def _copy_fake(tmp_path: Path) -> tuple[Path, str]:
    executable = tmp_path / "fake-capsule"
    shutil.copyfile(_FAKE, executable)
    executable.chmod(0o700)
    return executable, hashlib.sha256(executable.read_bytes()).hexdigest()


@contextmanager
def _active_run(
    tmp_path: Path, *, environ: Mapping[str, str] | None = None
) -> Iterator[RunSession]:
    with begin_run(
        "capsule-test",
        b"schema_version: 1\nid: topk-capsule\n",
        resolved={"schema_version": 1, "id": "topk-capsule"},
        home=tmp_path / "home",
        environ={} if environ is None else environ,
    ) as run:
        yield run
        run.fail("test-complete")


def _invoke(
    tmp_path: Path,
    run: RunSession,
    *,
    mode: str = "success",
    timeout: float = 1.0,
    executable: Path | None = None,
    executable_sha256: str | None = None,
    secrets: Mapping[str, str] | None = None,
) -> tuple[CapsuleProtocolResultV1, Path, Path]:
    if executable is None:
        executable, actual_sha = _copy_fake(tmp_path)
    else:
        actual_sha = hashlib.sha256(executable.read_bytes()).hexdigest()
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700, exist_ok=True)
    state = tmp_path / "state.json"
    environment = {
        "PATH": "/usr/bin:/bin",
        "FAKE_CAPSULE_MODE": mode,
        "FAKE_CAPSULE_STATE": str(state),
        **({} if secrets is None else secrets),
    }
    manifest = _manifest(timeout=timeout)
    manifest_sha = hashlib.sha256(
        canonical_json_bytes(manifest.model_dump(mode="json"))
    ).hexdigest()
    result = run_capsule_protocol(
        run,
        manifest,
        manifest_sha256=manifest_sha,
        executable_path=executable,
        executable_sha256=actual_sha if executable_sha256 is None else executable_sha256,
        cwd=tmp_path,
        environment=environment,
        scratch_root=scratch,
        redaction_context=RedactionContext.from_environ(environment),
    )
    return result, state, executable


def _portable(run: RunSession) -> tuple[list[str], dict[str, bytes]]:
    entries = list(list_portable_entries(run.active))
    payloads = {
        entry.logical_path: (run.active / "portable" / "blobs" / entry.blob_sha256).read_bytes()
        for entry in entries
    }
    return [entry.logical_path for entry in entries], payloads


def test_success_publishes_exact_ordered_chained_evidence(tmp_path: Path) -> None:
    with _active_run(tmp_path) as run:
        result, state_path, _ = _invoke(tmp_path, run)
        paths, payloads = _portable(run)

        assert result.status == "passed"
        assert result.reason == "passed"
        assert tuple(phase.operation for phase in result.phases) == (
            "describe",
            "correctness",
            "benchmark",
        )
        assert result.scenario is not None
        assert [coordinate.coordinate_id for coordinate in result.scenario.coordinates] == [
            "train-forward",
            "eval-reverse",
        ]
        assert result.benchmark is not None
        assert result.benchmark[0].latency_seconds == (0.01, 0.011, 0.012)
        assert result.benchmark[0].workspace_bytes == 4096
        assert "opaque_payload" not in result.model_dump(mode="json")
        assert all(phase.process.category == "none" for phase in result.phases)
        assert all(phase.process.stdout_complete for phase in result.phases)
        assert all(phase.process.stderr_complete for phase in result.phases)

        assert paths == [
            f"{EVIDENCE_ROOT}/describe/request.json",
            f"{EVIDENCE_ROOT}/describe/process.json",
            f"{EVIDENCE_ROOT}/describe/stdout.json",
            f"{EVIDENCE_ROOT}/correctness/request.json",
            f"{EVIDENCE_ROOT}/correctness/process.json",
            f"{EVIDENCE_ROOT}/correctness/stdout.json",
            f"{EVIDENCE_ROOT}/benchmark/request.json",
            f"{EVIDENCE_ROOT}/benchmark/process.json",
            f"{EVIDENCE_ROOT}/benchmark/stdout.json",
            f"{EVIDENCE_ROOT}/result.json",
        ]
        assert payloads[paths[-1]] == canonical_json_bytes(result.model_dump(mode="json"))
        entries = list_portable_entries(run.active)
        assert [entry.role for entry in entries] == [
            "correctness",
            "correctness",
            "correctness",
            "correctness",
            "correctness",
            "correctness",
            "samples",
            "samples",
            "samples",
            "summary",
        ]
        assert all(entry.media_type == "application/json" for entry in entries)
        for path in paths:
            if path.endswith(".json"):
                assert payloads[path] == canonical_json_bytes(json.loads(payloads[path]))

        state = json.loads(state_path.read_bytes())
        assert [entry["operation"] for entry in state] == [
            "describe",
            "correctness",
            "benchmark",
        ]
        assert state[1]["prior_response_sha256"] == state[0]["response_sha256"]
        assert state[2]["prior_response_sha256"] == state[1]["response_sha256"]
        for index, phase in enumerate(result.phases):
            assert phase.request_sha256 == state[index]["request_sha256"]
            assert phase.response_sha256 == state[index]["response_sha256"]
        assert all(entry["request_fd_readonly_and_sealed"] for entry in state)


@pytest.mark.parametrize(
    "argv_tail",
    [
        ("describe", "/proc/self/fd/3"),
        ("describe", "--input", "/proc/self/fd/3"),
        ("describe", "--request", "request.json"),
        ("describe", "--request", "/proc/self/fd/03"),
        ("unknown", "--request", "/proc/self/fd/3"),
    ],
)
def test_fake_rejects_every_nonprotocol_argv_shape(
    tmp_path: Path, argv_tail: tuple[str, ...]
) -> None:
    executable, _ = _copy_fake(tmp_path)
    result = run_process(
        (str(executable), *argv_tail),
        cwd=tmp_path,
        inherit_env=False,
        base_env={"PATH": "/usr/bin:/bin"},
    )

    assert result.outcome is ProcessOutcome.EXITED
    assert result.returncode == 90


def test_correctness_failure_short_circuits_benchmark(tmp_path: Path) -> None:
    with _active_run(tmp_path) as run:
        result, state_path, _ = _invoke(tmp_path, run, mode="correctness-fail")
        paths, _ = _portable(run)

        assert result.status == "failed"
        assert result.reason == "correctness-failed"
        assert tuple(phase.operation for phase in result.phases) == ("describe", "correctness")
        assert result.benchmark is None
        assert [entry["operation"] for entry in json.loads(state_path.read_bytes())] == [
            "describe",
            "correctness",
        ]
        assert not any("/benchmark/" in path for path in paths)
        assert paths[-1] == f"{EVIDENCE_ROOT}/result.json"


@pytest.mark.parametrize(
    ("mode", "reason", "stdout_name"),
    [
        ("malformed-describe", "describe-response-invalid", "stdout.txt"),
        ("noncanonical-describe", "describe-response-invalid", "stdout.txt"),
        ("wrong-operation-describe", "describe-response-invalid", "stdout.json"),
        ("wrong-request-describe", "describe-response-invalid", "stdout.json"),
        ("wrong-candidate-describe", "describe-response-invalid", "stdout.json"),
        ("wrong-scenario-describe", "describe-response-invalid", "stdout.json"),
        ("wrong-manifest-describe", "describe-response-invalid", "stdout.json"),
        ("wrong-executable-describe", "describe-response-invalid", "stdout.json"),
        ("nonzero-describe", "describe-process-failed", "stdout.json"),
    ],
)
def test_describe_refusals_are_failed_results_with_truthful_evidence(
    tmp_path: Path, mode: str, reason: str, stdout_name: str
) -> None:
    with _active_run(tmp_path) as run:
        result, _, _ = _invoke(tmp_path, run, mode=mode)
        paths, _ = _portable(run)

        assert result.status == "failed"
        assert result.reason == reason
        assert len(result.phases) == 1
        assert result.phases[0].accepted is False
        assert f"{EVIDENCE_ROOT}/describe/{stdout_name}" in paths
        assert paths[-1] == f"{EVIDENCE_ROOT}/result.json"


def test_timeout_returns_failed_process_result(tmp_path: Path) -> None:
    with _active_run(tmp_path) as run:
        result, _, _ = _invoke(tmp_path, run, mode="timeout-describe", timeout=0.05)

        assert result.reason == "describe-process-failed"
        assert result.phases[0].process.category == "timed-out"


@pytest.mark.parametrize(
    "mode",
    [
        "wrong-operation-correctness",
        "wrong-request-correctness",
        "wrong-candidate-correctness",
        "wrong-scenario-correctness",
        "wrong-manifest-correctness",
        "wrong-executable-correctness",
        "wrong-prior-correctness",
        "wrong-contract-correctness",
    ],
)
def test_later_response_must_echo_chain_and_contract(tmp_path: Path, mode: str) -> None:
    with _active_run(tmp_path) as run:
        result, _, _ = _invoke(tmp_path, run, mode=mode)

        assert result.reason == "correctness-response-invalid"
        assert tuple(phase.operation for phase in result.phases) == ("describe", "correctness")
        assert result.phases[-1].accepted is False


def test_hard_output_limit_returns_failed_process_result(tmp_path: Path) -> None:
    with _active_run(tmp_path) as run:
        result, _, _ = _invoke(tmp_path, run, mode="oversize-describe")

        assert result.reason == "describe-process-failed"
        assert result.phases[0].process.category == "capture-failed"
        assert result.phases[0].process.stdout_bytes > 1024 * 1024


def test_stderr_hard_output_limit_returns_failed_process_result(tmp_path: Path) -> None:
    with _active_run(tmp_path) as run:
        result, _, _ = _invoke(tmp_path, run, mode="oversize-stderr-describe")

        assert result.reason == "describe-process-failed"
        assert result.phases[0].process.category == "capture-failed"
        assert result.phases[0].process.stderr_bytes > 256 * 1024


@pytest.mark.parametrize(
    "mode",
    ["benchmark-missing", "benchmark-duplicate", "benchmark-reordered"],
)
def test_benchmark_requires_exact_coordinate_coverage_and_order(tmp_path: Path, mode: str) -> None:
    with _active_run(tmp_path) as run:
        result, _, _ = _invoke(tmp_path, run, mode=mode)

        assert result.status == "failed"
        assert result.reason == "benchmark-incomplete"
        assert len(result.phases) == 3
        assert result.phases[-1].accepted is True


@pytest.mark.parametrize(
    "mode", ["correctness-missing", "correctness-duplicate", "correctness-reordered"]
)
def test_correctness_requires_exact_coordinate_coverage(tmp_path: Path, mode: str) -> None:
    with _active_run(tmp_path) as run:
        result, _, _ = _invoke(tmp_path, run, mode=mode)

        assert result.reason == "correctness-incomplete"
        assert len(result.phases) == 2


@pytest.mark.parametrize(
    "mode",
    [
        "benchmark-incomplete-samples",
        "benchmark-nan",
        "benchmark-inf",
        "benchmark-nonpositive",
    ],
)
def test_benchmark_rejects_invalid_sample_vectors(tmp_path: Path, mode: str) -> None:
    with _active_run(tmp_path) as run:
        result, _, _ = _invoke(tmp_path, run, mode=mode)

        assert result.status == "failed"
        assert result.reason == "benchmark-response-invalid"
        assert result.phases[-1].accepted is False
        assert result.benchmark is None


@pytest.mark.parametrize(
    "mode",
    [
        "wrong-operation-benchmark",
        "wrong-request-benchmark",
        "wrong-candidate-benchmark",
        "wrong-scenario-benchmark",
        "wrong-manifest-benchmark",
        "wrong-executable-benchmark",
        "wrong-prior-benchmark",
        "wrong-contract-benchmark",
    ],
)
def test_benchmark_response_must_echo_every_binding(tmp_path: Path, mode: str) -> None:
    with _active_run(tmp_path) as run:
        result, _, _ = _invoke(tmp_path, run, mode=mode)

        assert result.reason == "benchmark-response-invalid"
        assert len(result.phases) == 3
        assert result.phases[-1].failure == "response"


@pytest.mark.parametrize(
    "mode", ["unknown-field-describe", "coercible-describe", "oversize-opaque-describe"]
)
def test_strict_response_and_opaque_bounds_are_enforced(tmp_path: Path, mode: str) -> None:
    with _active_run(tmp_path) as run:
        result, _, _ = _invoke(tmp_path, run, mode=mode)

        assert result.reason == "describe-response-invalid"
        assert result.phases[0].failure == "response"


def test_secret_output_is_refused_before_publication(tmp_path: Path) -> None:
    secret = "capsule-secret-value"
    environment_secret = {"API_TOKEN": secret}
    with _active_run(tmp_path, environ=environment_secret) as run:
        result, _, _ = _invoke(
            tmp_path,
            run,
            mode="secret-describe",
            secrets=environment_secret,
        )
        paths, payloads = _portable(run)

        assert result.reason == "describe-unsafe-output"
        assert f"{EVIDENCE_ROOT}/describe/stdout-unavailable.txt" in paths
        assert all(secret.encode() not in payload for payload in payloads.values())


@pytest.mark.parametrize("mode", ["interpolation-describe", "interpolation-stderr-describe"])
def test_sensitive_interpolation_output_is_atomically_withheld(tmp_path: Path, mode: str) -> None:
    interpolation = b"${API_TOKEN}"
    with _active_run(tmp_path) as run:
        result, _, _ = _invoke(tmp_path, run, mode=mode)
        paths, payloads = _portable(run)

        assert result.reason == "describe-unsafe-output"
        assert result.phases[0].failure == "unsafe"
        assert paths[-1] == f"{EVIDENCE_ROOT}/result.json"
        assert all(interpolation not in payload for payload in payloads.values())


@pytest.mark.parametrize("mode", ["deep-json-describe", "lone-surrogate-describe"])
def test_pathological_json_is_an_ordinary_response_failure(tmp_path: Path, mode: str) -> None:
    with _active_run(tmp_path) as run:
        result, _, _ = _invoke(tmp_path, run, mode=mode)
        paths, _ = _portable(run)

        assert result.reason == "describe-response-invalid"
        assert result.phases[0].failure == "response"
        assert f"{EVIDENCE_ROOT}/describe/stdout.txt" in paths
        assert paths[-1] == f"{EVIDENCE_ROOT}/result.json"


def test_executable_digest_mismatch_raises_integrity_without_result(tmp_path: Path) -> None:
    executable, _ = _copy_fake(tmp_path)
    with _active_run(tmp_path) as run:
        with pytest.raises(CapsuleIntegrityError, match="digest does not match"):
            _invoke(tmp_path, run, executable=executable, executable_sha256="0" * 64)
        assert list_portable_entries(run.active) == ()


def test_executable_drift_raises_integrity_without_success(tmp_path: Path) -> None:
    with _active_run(tmp_path) as run:
        with pytest.raises(CapsuleIntegrityError, match="drifted"):
            _invoke(tmp_path, run, mode="drift-describe")
        paths = [entry.logical_path for entry in list_portable_entries(run.active)]
        assert f"{EVIDENCE_ROOT}/result.json" not in paths


def test_request_mutation_attempt_raises_integrity_without_success(tmp_path: Path) -> None:
    with _active_run(tmp_path) as run:
        with pytest.raises(CapsuleIntegrityError, match="request evidence drifted"):
            _invoke(tmp_path, run, mode="mutate-request-describe")
        paths = [entry.logical_path for entry in list_portable_entries(run.active)]
        assert f"{EVIDENCE_ROOT}/result.json" not in paths


def test_spawn_failure_returns_truthful_failed_result(tmp_path: Path) -> None:
    executable = tmp_path / "missing-interpreter-capsule"
    executable.write_text("#!/definitely/not/a/real/interpreter\n")
    executable.chmod(0o700)
    with _active_run(tmp_path) as run:
        result, _, _ = _invoke(tmp_path, run, executable=executable)

        assert result.reason == "describe-process-failed"
        assert result.phases[0].process.category == "spawn-failed"
        assert result.phases[0].process.returncode is None


def test_spool_digest_divergence_raises_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_run_process = capsules.run_process

    def corrupt_spool(*args: Any, **kwargs: Any) -> ProcessResult:
        result = real_run_process(*args, **kwargs)
        assert result.stdout_spool is not None
        result.stdout_spool.write_bytes(b"corrupt")
        return result

    monkeypatch.setattr(capsules, "run_process", corrupt_spool)
    with _active_run(tmp_path) as run:
        with pytest.raises(CapsuleIntegrityError, match="spool drifted"):
            _invoke(tmp_path, run)
        assert not any(
            entry.logical_path == f"{EVIDENCE_ROOT}/result.json"
            for entry in list_portable_entries(run.active)
        )


def test_publication_failure_raises_integrity_without_terminal_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _active_run(tmp_path) as run:

        def refuse_publication(*args: Any, **kwargs: Any) -> None:
            raise OSError("fixture publication refusal")

        monkeypatch.setattr(RunSession, "write_portable", refuse_publication)
        with pytest.raises(CapsuleIntegrityError, match="could not be published"):
            _invoke(tmp_path, run)
        assert list_portable_entries(run.active) == ()


def test_existing_protocol_subtree_is_evidence_integrity_failure(tmp_path: Path) -> None:
    executable, executable_sha = _copy_fake(tmp_path)
    with _active_run(tmp_path) as run:
        run.write_portable(
            f"{EVIDENCE_ROOT}/result.json",
            canonical_json_bytes({"existing": True}),
            media_type="application/json",
            role="summary",
        )
        with pytest.raises(CapsuleIntegrityError, match="already exists"):
            _invoke(
                tmp_path,
                run,
                executable=executable,
                executable_sha256=executable_sha,
            )


def test_child_receives_only_complete_supplied_environment(tmp_path: Path) -> None:
    sentinel = "must-not-be-inherited"
    os.environ["STRIXLAB_CAPSULE_AMBIENT_SENTINEL"] = sentinel
    try:
        with _active_run(tmp_path) as run:
            result, _, _ = _invoke(tmp_path, run, mode="check-environment")
            assert result.status == "passed"
    finally:
        os.environ.pop("STRIXLAB_CAPSULE_AMBIENT_SENTINEL", None)


def test_terminal_model_rejects_accepted_timed_out_phase(tmp_path: Path) -> None:
    with _active_run(tmp_path) as run:
        result, _, _ = _invoke(tmp_path, run)
    payload = result.model_dump(mode="json")
    payload["phases"][0]["process"].update(
        {"outcome": "timed_out", "returncode": -15, "category": "timed-out"}
    )

    with pytest.raises(ValueError, match="accepted phase"):
        CapsuleProtocolResultV1.model_validate_json(canonical_json_bytes(payload))


def test_phase_model_rejects_failure_none_without_accepted_digest(tmp_path: Path) -> None:
    with _active_run(tmp_path) as run:
        result, _, _ = _invoke(tmp_path, run)
    payload = result.phases[0].model_dump(mode="json")
    payload.update({"accepted": False, "response_sha256": None, "failure": "none"})

    with pytest.raises(ValueError, match="accepted, response digest, and phase failure"):
        CapsulePhaseResultV1.model_validate_json(canonical_json_bytes(payload))


def test_terminal_model_rejects_reason_without_required_phase_state(tmp_path: Path) -> None:
    with _active_run(tmp_path) as run:
        result, _, _ = _invoke(tmp_path, run)
    payload = result.model_dump(mode="json")
    payload.update({"status": "failed", "reason": "benchmark-incomplete", "phases": []})

    with pytest.raises(ValueError, match="one to three"):
        CapsuleProtocolResultV1.model_validate_json(canonical_json_bytes(payload))

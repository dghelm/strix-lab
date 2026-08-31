"""Shared fixtures for SUITE-001 tests: a present build, a receipt, and fake samples.

These construct valid adapter sample objects and fake adapter callables so the suite
executor can be exercised with no GPU, ROCm, model weights, network, or real binaries.
The build-cache verification entry points are stubbed by the test module (see
``_stub_cache_verification``), mirroring ``test_build_cache``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from _build_fixtures import publish_present_build, stub_cache_verification
from _model_fixtures import build_verified_receipt

import strixlab.models as models
from strixlab.adapters import backend_ops as bo
from strixlab.adapters import llama_bench as lb
from strixlab.adapters import llama_server as ls
from strixlab.build_artifacts import ArtifactV1, BuildArtifactsV1, TargetArtifactsV1
from strixlab.build_cache import (
    CanonicalBuildRecordV1,
    IdentityEntryV1,
    SourceReproducerV1,
)

__all__ = ["publish_present_build", "stub_cache_verification"]

BUILD_ID = "build-sha256:" + "aa" * 32
ATTEMPT = "attempt-" + "0" * 24 + "-" + "a" * 32
SOURCE_COMMIT = "ca94157f70a2776e8da6b6849b50b45a083d0478"
_HEX = "cd" * 32
_ZERO = "0" * 64

TARGETS = ("test-backend-ops", "llama-server", "llama-bench")


def _artifacts() -> BuildArtifactsV1:
    targets = tuple(
        TargetArtifactsV1(
            name=name, target_id=f"id-{name}", target_type="EXECUTABLE", artifacts=(f"bin/{name}",)
        )
        for name in TARGETS
    )
    artifacts = tuple(
        ArtifactV1(
            path=f"bin/{name}",
            kind="elf",
            elf_type="ET_EXEC",
            mode=0o755,
            size_bytes=4,
            sha256=_HEX,
            targets=(name,),
        )
        for name in TARGETS
    )
    return BuildArtifactsV1(
        artifact_set_id="artifact-set-sha256:" + _HEX,
        targets=targets,
        artifacts=artifacts,
        inspections=(),
        capture_tools=(),
        cmake_cache_sha256=_HEX,
        compile_commands_sha256=None,
    )


def default_environment() -> tuple[IdentityEntryV1, ...]:
    return (
        IdentityEntryV1(name="HOME", value="{BUILD_HOME}"),
        IdentityEntryV1(name="TMPDIR", value="{BUILD_TMP}"),
        IdentityEntryV1(name="LANG", value="C"),
        IdentityEntryV1(name="LC_ALL", value="C"),
        IdentityEntryV1(name="TZ", value="UTC"),
        IdentityEntryV1(name="PATH", value="/opt/rocm-10/bin:/usr/bin"),
        IdentityEntryV1(name="ROCM_PATH", value="/opt/rocm-10"),
        IdentityEntryV1(name="LD_LIBRARY_PATH", value="{BUILD_ROOT}/lib:/usr/lib"),
        IdentityEntryV1(name="SOURCE_DATE_EPOCH", value="0"),
    )


def _source(source_id: str = "strix-llama", base_commit: str = SOURCE_COMMIT) -> SourceReproducerV1:
    return SourceReproducerV1(
        candidate_id="candidate-sha256:" + _HEX,
        content_tree_id="content-tree-sha256:" + _HEX,
        snapshot_id="snapshot-sha256:" + _HEX,
        source_evidence={"source_id": source_id, "base_commit": base_commit},
        source_evidence_sha256=_HEX,
        snapshot_manifest={"schema_version": 1},
        diff=None,
        patches=(),
    )


def canonical_record(
    *,
    environment: tuple[IdentityEntryV1, ...] | None = None,
    selections: tuple[IdentityEntryV1, ...] | None = None,
    toolchain_mode: str = "rocm",
    artifacts: BuildArtifactsV1 | None = None,
    source_id: str = "strix-llama",
    base_commit: str = SOURCE_COMMIT,
) -> CanonicalBuildRecordV1:
    if selections is None:
        selections = (
            IdentityEntryV1(name="generator", value="Ninja"),
            IdentityEntryV1(name="gfx_targets", value="gfx1151"),
        )
    return CanonicalBuildRecordV1(
        build_id=BUILD_ID,
        producer_attempt_id=ATTEMPT,
        recipe_id="recipe-sha256:" + _HEX,
        profile_sha256=_HEX,
        toolchain_mode=toolchain_mode,  # type: ignore[arg-type]
        environment=environment if environment is not None else default_environment(),
        requested_targets=TARGETS,
        selections=selections,
        tools=(),
        source=_source(source_id, base_commit),
        artifacts=artifacts if artifacts is not None else _artifacts(),
    )


def make_present_build(home: Path, *, record: CanonicalBuildRecordV1 | None = None) -> str:
    """Materialize, publish, and attest one PRESENT smoke build (verification stubbed)."""

    return publish_present_build(
        home, build_id=BUILD_ID, attempt=ATTEMPT, record=record or canonical_record()
    )


def build_smoke_receipt(
    scratch: Path, *, model_id: str = "qwen35-4b-smoke"
) -> models.ModelReceiptV1:
    """Build a fresh verified receipt (v2 evidence) for a synthetic model."""

    scratch.mkdir(parents=True, exist_ok=True)
    model_path = scratch / "model.gguf"
    model_path.write_bytes(b"gguf-bytes")
    return build_verified_receipt(scratch, model_path, model_id=model_id)


def publish_receipt(home: Path, scratch: Path, *, model_id: str = "qwen35-4b-smoke") -> str:
    """Build a verified receipt for a synthetic model and publish it in ``home``.

    Returns the local receipt SHA-256 (the receipt-envelope address).
    """

    return _publish(home, build_smoke_receipt(scratch, model_id=model_id))


def publish_receipt_with_execution(
    home: Path,
    scratch: Path,
    *,
    required_sources: tuple[str, ...] = (),
    required_features: tuple[str, ...] = (),
    model_id: str = "qwen35-4b-smoke",
) -> str:
    """Publish a receipt whose authenticated v2 evidence declares execution requirements.

    Rebinds the evidence's execution projection (and republishes under the resulting
    self-consistent envelope digest) so a non-empty requirement set is authenticated.
    """

    receipt = build_smoke_receipt(scratch, model_id=model_id)
    execution = models.ModelExecutionProjectionV1(
        verification_status="unverified",
        required_sources=required_sources,
        required_features=required_features,
    )
    evidence = receipt.evidence.model_copy(update={"execution": execution})
    return _publish(home, receipt.model_copy(update={"evidence": evidence}))


def publish_legacy_v1_receipt(
    home: Path, scratch: Path, *, model_id: str = "qwen35-4b-smoke"
) -> str:
    """Publish a legacy v1 receipt (evidence without the execution projection)."""

    receipt = build_smoke_receipt(scratch, model_id=model_id)
    base_fields = {
        name: getattr(receipt.evidence, name)
        for name in models.ModelReceiptEvidenceV1.model_fields
        if name != "schema_version"
    }
    v1_evidence = models.ModelReceiptEvidenceV1(**base_fields)
    return _publish(home, receipt.model_copy(update={"evidence": v1_evidence}))


def receipt_registry_path(home: Path, digest: str, *, model_id: str = "qwen35-4b-smoke") -> Path:
    """The content-addressed on-disk path of one published receipt envelope."""

    return home.joinpath(*models._RECEIPT_REGISTRY_DIR, model_id, f"{digest}.json")


def tamper_receipt_execution(
    home: Path,
    receipt: models.ModelReceiptV1,
    digest: str,
    *,
    required_sources: tuple[str, ...] = ("cuda-graphs",),
    model_id: str = "qwen35-4b-smoke",
) -> None:
    """Overwrite a published receipt's stored bytes with a tampered execution projection.

    The content address (``digest``) is unchanged, so re-reading it must fail the
    content-address check — proving the execution projection is bound by the digest.
    """

    execution = models.ModelExecutionProjectionV1(
        verification_status="unverified", required_sources=required_sources, required_features=()
    )
    evidence = receipt.evidence.model_copy(update={"execution": execution})
    payload, _ = models._receipt_envelope(receipt.model_copy(update={"evidence": evidence}))
    path = receipt_registry_path(home, digest, model_id=model_id)
    path.unlink()
    path.write_bytes(payload)


def publish_receipt_object(home: Path, receipt: models.ModelReceiptV1) -> str:
    """Publish an already-built receipt object; return its envelope address."""

    return _publish(home, receipt)


def _publish(home: Path, receipt: models.ModelReceiptV1) -> str:
    models._prepare_home(home, create=True)
    models._publish_receipt(home, receipt)
    _, digest = models._receipt_envelope(receipt)
    return digest


# --- Valid sample factories ---------------------------------------------------


def _bench_proj() -> lb.ProcessProjectionV1:
    return lb.ProcessProjectionV1(
        outcome="exited",
        returncode=0,
        duration_seconds=0.0,
        stdout_bytes=0,
        stderr_bytes=0,
        stdout_sha256=_ZERO,
        stderr_sha256=_ZERO,
        stdout_truncated=False,
        stderr_truncated=False,
        stdout_publishable=True,
        stderr_publishable=True,
        error_category="none",
    )


def bench_success(
    inputs: lb.LlamaBenchInputsV1, case: lb.LlamaBenchCaseV1
) -> lb.LlamaBenchSampleV1:
    caps = lb.LlamaBenchCapabilitiesV1(
        binary_sha256=inputs.binary_sha256,
        advertised_output_modes=("csv", "json", "jsonl", "md", "sql"),
    )
    proj = _bench_proj()
    attempt = lb.LlamaBenchCapabilityAttemptV1(
        status="discovered", reason=None, help=proj, version=proj, capabilities=caps
    )
    invocation = lb.LlamaBenchInvocationV1(ordinal=1, argv=("bench",), process=proj)
    measurement = lb.LlamaBenchMeasurementV1(
        avg_ts=10.0, stddev_ts=0.5, samples_ts=tuple(10.0 for _ in range(case.repetitions))
    )
    return lb.LlamaBenchSampleV1(
        status="success",
        reason="success",
        case=case,
        inputs=inputs,
        capability_attempt=attempt,
        capabilities=caps,
        invocation=invocation,
        measurement=measurement,
        artifacts=(),
    )


def bench_process_failed(
    inputs: lb.LlamaBenchInputsV1, case: lb.LlamaBenchCaseV1
) -> lb.LlamaBenchSampleV1:
    caps = lb.LlamaBenchCapabilitiesV1(
        binary_sha256=inputs.binary_sha256,
        advertised_output_modes=("csv", "json", "jsonl", "md", "sql"),
    )
    proj = _bench_proj()
    failed = proj.model_copy(update={"returncode": 1, "error_category": "nonzero-exit"})
    attempt = lb.LlamaBenchCapabilityAttemptV1(
        status="discovered", reason=None, help=proj, version=proj, capabilities=caps
    )
    invocation = lb.LlamaBenchInvocationV1(ordinal=1, argv=("bench",), process=failed)
    return lb.LlamaBenchSampleV1(
        status="process-failed",
        reason="nonzero-exit",
        case=case,
        inputs=inputs,
        capability_attempt=attempt,
        capabilities=caps,
        invocation=invocation,
        measurement=None,
        artifacts=(),
    )


def _server_proj() -> ls.ProcessProjectionV1:
    return ls.ProcessProjectionV1(
        outcome="exited",
        returncode=0,
        duration_seconds=0.0,
        stdout_bytes=0,
        stderr_bytes=0,
        stdout_sha256=_ZERO,
        stderr_sha256=_ZERO,
        stdout_truncated=False,
        stderr_truncated=False,
        error_category="none",
    )


def _server_capabilities() -> ls.LlamaServerCapabilitiesV1:
    return ls.LlamaServerCapabilitiesV1(
        toolchain="cc (Ubuntu)",
        required_options=(
            "model",
            "host",
            "port",
            "offline",
            "ui",
            "cache-prompt",
            "parallel",
            "threads-http",
            "ctx-size",
            "gpu-layers",
        ),
    )


def server_success(
    inputs: ls.LlamaServerInputsV1,
    case: ls.LlamaServerCaseV1,
    *,
    tokens_a: tuple[int, ...] = (1, 2, 3, 4),
    tokens_b: tuple[int, ...] | None = None,
) -> ls.LlamaServerSampleV1:
    tokens_b = tokens_a if tokens_b is None else tokens_b
    proj = _server_proj()
    attempt = ls.CapabilityAttemptV1(
        version=proj, help=proj, capabilities=_server_capabilities(), reason=None
    )
    readiness = ls.ReadinessV1(
        attempts=1,
        elapsed_seconds=0.0,
        final_status=200,
        final_body_sha256=_ZERO,
        ready=True,
        reason=None,
    )
    requests = (
        ls.RequestAttemptV1(
            ordinal=1,
            sentinel="a" * 32,
            request_sha256=_ZERO,
            response_status=200,
            response_bytes=8,
            response_sha256=_ZERO,
            response=ls.CompletionResponseV1(prompt="p1", content="c", tokens=tokens_a),
            isolated=True,
            error_category="none",
        ),
        ls.RequestAttemptV1(
            ordinal=2,
            sentinel="b" * 32,
            request_sha256=_ZERO,
            response_status=200,
            response_bytes=8,
            response_sha256=_ZERO,
            response=ls.CompletionResponseV1(prompt="p2", content="c", tokens=tokens_b),
            isolated=True,
            error_category="none",
        ),
    )
    lifecycle = ls.ServerLifecycleV1(
        argv=("server",),
        pid=1,
        returncode=0,
        stdout_bytes=0,
        stderr_bytes=0,
        stdout_sha256=_ZERO,
        stderr_sha256=_ZERO,
        stdout_truncated=False,
        stderr_truncated=False,
        capture_error=None,
        sigterm_sent=True,
        sigkill_sent=False,
        unexpected_exit=False,
        shutdown_error=None,
        shutdown_elapsed_seconds=0.0,
    )
    return ls.LlamaServerSampleV1(
        status="success",
        reason="success",
        case=case,
        inputs=inputs,
        capability_attempt=attempt,
        readiness=readiness,
        requests=requests,
        lifecycle=lifecycle,
        artifacts=(),
    )


def server_capability_failed(
    inputs: ls.LlamaServerInputsV1, case: ls.LlamaServerCaseV1
) -> ls.LlamaServerSampleV1:
    proj = _server_proj().model_copy(update={"error_category": "nonzero-exit", "returncode": 1})
    attempt = ls.CapabilityAttemptV1(
        version=proj, help=proj, capabilities=None, reason="capability process failed"
    )
    return ls.LlamaServerSampleV1(
        status="capability-failed",
        reason="capability process failed",
        case=case,
        inputs=inputs,
        capability_attempt=attempt,
        readiness=None,
        requests=(),
        lifecycle=None,
        artifacts=(),
    )


def _backend_proj() -> bo.ProcessProjectionV1:
    return bo.ProcessProjectionV1(
        outcome="exited",
        returncode=0,
        duration_seconds=0.0,
        stdout_bytes=0,
        stderr_bytes=0,
        stdout_sha256=_ZERO,
        stderr_sha256=_ZERO,
        stdout_truncated=False,
        stderr_truncated=False,
        stdout_publishable=True,
        stderr_publishable=True,
        error_category="none",
    )


def _backend_advertised(case: bo.BackendOpsCaseV1) -> tuple[str, ...]:
    requested = list(case.operations)
    filler = [f"FILL{index}" for index in range(bo.EXPECTED_OPERATION_COUNT - len(requested))]
    return tuple(requested + filler)


def _backend_capabilities(
    inputs: bo.BackendOpsInputsV1, case: bo.BackendOpsCaseV1
) -> bo.BackendOpsCapabilitiesV1:
    return bo.BackendOpsCapabilitiesV1(
        binary_sha256=inputs.binary_sha256, operations=_backend_advertised(case)
    )


def _backend_rows(
    case: bo.BackendOpsCaseV1, *, supported: str = "1"
) -> tuple[bo.BackendOpsRowV1, ...]:
    return tuple(
        bo.BackendOpsRowV1(
            backend_name=case.backend,
            op_name=op,
            op_params="type=f32",
            test_mode="test",
            supported=supported,  # type: ignore[arg-type]
            error_message="",
            backend_reg_name="reg",
        )
        for op in case.operations
    )


def backend_passed(
    inputs: bo.BackendOpsInputsV1, case: bo.BackendOpsCaseV1
) -> bo.BackendOpsSampleV1:
    caps = _backend_capabilities(inputs, case)
    proj = _backend_proj()
    attempt = bo.BackendOpsCapabilityAttemptV1(
        status="discovered", reason=None, help=proj, list_ops=proj, capabilities=caps
    )
    rows = _backend_rows(case)
    gate = bo.BackendOpsGateSummaryV1(
        selected_count=len(case.operations),
        observed_rows=len(rows),
        observed_backends=(case.backend,),
        passed=True,
        reason="passed",
    )
    invocation = bo.BackendOpsInvocationV1(ordinal=1, argv=("backend",), process=proj)
    return bo.BackendOpsSampleV1(
        status="passed",
        reason=None,
        case=case,
        inputs=inputs,
        capability_attempt=attempt,
        capabilities=caps,
        invocation=invocation,
        rows=rows,
        gate=gate,
        artifacts=(),
    )


def backend_passed_status_failed_gate(
    inputs: bo.BackendOpsInputsV1, case: bo.BackendOpsCaseV1
) -> bo.BackendOpsSampleV1:
    """A tampered sample: status ``passed`` but a gate that did not pass.

    The adapter model forbids constructing this directly, so it is built via
    ``model_copy`` (which does not re-run the model validator) to exercise the suite's
    independent gate guard.
    """

    sample = backend_passed(inputs, case)
    assert sample.gate is not None
    failed_gate = sample.gate.model_copy(update={"passed": False, "reason": "unsupported-row"})
    return sample.model_copy(update={"gate": failed_gate})


def backend_gate_failed(
    inputs: bo.BackendOpsInputsV1, case: bo.BackendOpsCaseV1
) -> bo.BackendOpsSampleV1:
    caps = _backend_capabilities(inputs, case)
    proj = _backend_proj()
    attempt = bo.BackendOpsCapabilityAttemptV1(
        status="discovered", reason=None, help=proj, list_ops=proj, capabilities=caps
    )
    rows = _backend_rows(case, supported="0")
    gate = bo.BackendOpsGateSummaryV1(
        selected_count=len(case.operations),
        observed_rows=len(rows),
        observed_backends=(case.backend,),
        passed=False,
        reason="unsupported-row",
    )
    invocation = bo.BackendOpsInvocationV1(ordinal=1, argv=("backend",), process=proj)
    return bo.BackendOpsSampleV1(
        status="hard-gate-failed",
        reason=None,
        case=case,
        inputs=inputs,
        capability_attempt=attempt,
        capabilities=caps,
        invocation=invocation,
        rows=rows,
        gate=gate,
        artifacts=(),
    )


# --- Fake adapter runners -----------------------------------------------------
#
# Each fake persists its ``sample.json`` the way the real adapter does: as a portable
# entry (all three), plus a local copy for llama-server, which mirrors its binary-response
# tree. ``persist`` selects the mode so tests can exercise the local-only rejection path:
# "portable" (default; server also writes local), "local" (local only), or "none".


def _persist_sample(
    run: Any, root: str, case_id: str, sample: Any, *, role: str, mode: str
) -> None:
    from strixlab.serialization import canonical_json_bytes

    payload = canonical_json_bytes(sample.model_dump(mode="json"))
    path = f"{root}/{case_id}/sample.json"
    if mode == "portable":
        run.write_portable(path, payload, media_type="application/json", role=role)
    elif mode == "local":
        run.write_evidence(path, payload)


def fake_backend(sample_fn: Any = backend_passed, *, persist: str = "portable") -> Any:
    def runner(*, case: Any, inputs: Any, run: Any, **_kwargs: Any) -> Any:
        sample = sample_fn(inputs, case)
        _persist_sample(run, bo.EVIDENCE_ROOT, case.id, sample, role="correctness", mode=persist)
        return sample

    return runner


def fake_server(sample_fn: Any = server_success, *, persist: str = "portable") -> Any:
    def runner(*, case: Any, inputs: Any, run: Any, **_kwargs: Any) -> Any:
        sample = sample_fn(inputs, case)
        # The real adapter keeps both a local copy and a portable entry; only the portable
        # entry is what the suite authenticates against.
        if persist == "portable":
            _persist_sample(run, ls.EVIDENCE_ROOT, case.id, sample, role="samples", mode="local")
        _persist_sample(run, ls.EVIDENCE_ROOT, case.id, sample, role="samples", mode=persist)
        return sample

    return runner


def fake_bench(sample_fn: Any = bench_success, *, persist: str = "portable") -> Any:
    def runner(*, case: Any, inputs: Any, run: Any, **_kwargs: Any) -> Any:
        sample = sample_fn(inputs, case)
        _persist_sample(run, lb.EVIDENCE_ROOT, case.id, sample, role="samples", mode=persist)
        return sample

    return runner


def raising_runner(exc: Exception) -> Any:
    def runner(**_kwargs: Any) -> Any:
        raise exc

    return runner


def env_dict() -> dict[str, str]:
    return {name: os.environ[name] for name in ("PATH",) if name in os.environ}

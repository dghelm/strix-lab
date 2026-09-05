"""Actual compiled host fixture transport; no Python success fake or GPU claims."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

import strixlab.capsules as capsules
from strixlab.capsules import (
    BenchmarkResponseV1,
    CapsuleIntegrityError,
    CorrectnessResponseV1,
    DescribeResponseV1,
    run_capsule_protocol,
)
from strixlab.evidence import begin_run, list_portable_entries
from strixlab.manifests import CapsuleManifestV1
from strixlab.secret_policy import RedactionContext
from strixlab.serialization import canonical_json_bytes

ROOT = Path(__file__).resolve().parents[2]
IDENTITY = "strixlab-topk-host-test-v1"
SCENARIO_SHA = hashlib.sha256(IDENTITY.encode()).hexdigest()
SEALS = 0x000F  # Linux seal/write/grow/shrink flags, as in the existing runner.


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture(scope="module")
def build(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("native-transport")
    for argv in [
        [
            "cmake",
            "-S",
            str(ROOT / "native/topk"),
            "-B",
            str(path),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSTRIXLAB_TOPK_BUILD_FAULT_FIXTURES=ON",
            "-DSTRIXLAB_NATIVE_BUILD_COMMIT=" + "a" * 40,
        ],
        ["cmake", "--build", str(path), "--parallel", "2"],
        ["ctest", "--test-dir", str(path), "--output-on-failure"],
    ]:
        subprocess.run(argv, check=True, capture_output=True, text=True, timeout=120)
    cache = (path / "CMakeCache.txt").read_text()
    assert "STRIXLAB_NATIVE_BUILD_COMMIT:STRING=" + "a" * 40 in cache
    assert "STRIXLAB_NATIVE_BUILD_NUMBER:STRING=0" in cache
    assert "CMAKE_C_COMPILER:FILEPATH=" in cache
    assert "CMAKE_CXX_COMPILER:FILEPATH=" in cache
    return path


def request(exe: Path, operation: str = "describe", scenario: Any = None) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "native-capsule-v1",
        "operation": operation,
        "capsule_id": IDENTITY,
        "candidate": "host-fixture",
        "scenario_sha256": SCENARIO_SHA,
        "manifest_sha256": "a" * 64,
        "executable_sha256": digest(exe.read_bytes()),
        "prior_response_sha256": None if operation == "describe" else "b" * 64,
        "scenario_contract_sha256": None
        if scenario is None
        else digest(canonical_json_bytes(scenario)),
        "scenario": scenario,
    }


def invoke_bytes(
    exe: Path,
    payload: bytes,
    operation: str = "describe",
    *,
    readonly: bool = True,
    seals: int = SEALS,
    path_prefix: str = "/proc/self/fd/",
) -> subprocess.CompletedProcess[bytes]:
    writer = capsules._memfd_create("fixture-request")
    reader = None
    try:
        with os.fdopen(os.dup(writer), "wb") as stream:
            stream.write(payload)
        fcntl.fcntl(writer, 1033, seals)
        reader = os.open(f"/proc/self/fd/{writer}", os.O_RDONLY) if readonly else os.dup(writer)
        # Match the real runner: inherited file position is deliberately at EOF.
        os.lseek(reader, 0, os.SEEK_END)
        return subprocess.run(
            [str(exe), operation, "--request", f"{path_prefix}{reader}"],
            pass_fds=(reader,),
            capture_output=True,
            timeout=10,
        )
    finally:
        if reader is not None:
            os.close(reader)
        os.close(writer)


def described(exe: Path) -> dict[str, Any]:
    result = invoke_bytes(exe, canonical_json_bytes(request(exe)))
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_cross_language_response_bytes_and_bindings(build: Path) -> None:
    exe = build / "topk_capsule_host_test"
    scenario = None
    prior = None
    for operation, model in [
        ("describe", DescribeResponseV1),
        ("correctness", CorrectnessResponseV1),
        ("benchmark", BenchmarkResponseV1),
    ]:
        value = request(exe, operation, scenario)
        value["prior_response_sha256"] = prior
        raw = canonical_json_bytes(value)
        result = invoke_bytes(exe, raw, operation)
        assert result.returncode == 0, result.stderr
        decoded = json.loads(result.stdout)
        assert result.stdout == canonical_json_bytes(decoded)
        model.model_validate_json(result.stdout)
        for key, item in value.items():
            if key != "scenario":
                assert decoded[key] == item
        assert decoded["request_sha256"] == digest(raw)
        assert decoded["opaque_payload"]["fixture"] is True
        assert decoded["opaque_payload"]["synthetic"] is True
        if operation == "describe":
            scenario = decoded["scenario"]
        else:
            assert decoded["opaque_payload"]["readiness_checks"] == 1
            assert decoded["opaque_payload"]["operation_calls"] == 2
        prior = digest(result.stdout)


@pytest.mark.parametrize(
    "case",
    [
        "duplicate",
        "unknown",
        "missing",
        "float",
        "bool",
        "negative",
        "huge",
        "nan",
        "infinity",
        "overflow",
        "utf8",
        "surrogate",
        "unicode",
        "escape",
        "slash-escape",
        "bom",
        "trailing",
        "nul",
        "compact",
        "deep",
        "array",
        "scenario",
        "prior",
        "candidate",
        "production-id",
        "executable",
        "manifest",
        "operation",
    ],
)
def test_rejects_noncanonical_or_invalid_request(build: Path, case: str) -> None:
    exe = build / "topk_capsule_host_test"
    value = request(exe)
    raw = canonical_json_bytes(value)
    if case == "duplicate":
        raw = raw.replace(b"{\n", b'{\n  "schema_version": 1,\n', 1)
    elif case in {"float", "bool", "negative", "huge", "nan", "infinity", "overflow"}:
        token = {
            "float": b"1.0",
            "bool": b"true",
            "negative": b"-1",
            "huge": b"18446744073709551616",
            "nan": b"NaN",
            "infinity": b"Infinity",
            "overflow": b"1e400",
        }[case]
        raw = raw.replace(b'"schema_version": 1', b'"schema_version": ' + token)
    elif case in {"utf8", "surrogate", "unicode", "escape", "slash-escape"}:
        token = {
            "utf8": b"\xff",
            "surrogate": b"\\ud800",
            "unicode": "é".encode(),
            "escape": b"\\u0073",
            "slash-escape": b"\\/",
        }[case]
        raw = raw.replace(b"strixlab-topk", token + b"trixlab-topk")
    elif case == "bom":
        raw = b"\xef\xbb\xbf" + raw
    elif case == "trailing":
        raw += b" {}"
    elif case == "nul":
        raw += b"\0"
    elif case == "compact":
        raw = json.dumps(value).encode()
    elif case == "deep":
        raw = b'{"unknown":' + b"[" * 32 + b"0" + b"]" * 32 + b"}"
    elif case == "array":
        raw = b"[]\n"
    else:
        if case == "unknown":
            value["extra"] = "ignored?"
        elif case == "missing":
            del value["candidate"]
        else:
            key, item = {
                "scenario": ("scenario", {}),
                "prior": ("prior_response_sha256", "b" * 64),
                "candidate": ("candidate", "unknown-provider"),
                "production-id": ("capsule_id", "rocm10-topk-gfx1151-v1"),
                "executable": ("executable_sha256", "c" * 64),
                "manifest": ("manifest_sha256", "A" * 64),
                "operation": ("operation", "benchmark"),
            }[case]
            value[key] = item
        raw = canonical_json_bytes(value)
    result = invoke_bytes(exe, raw)
    assert result.returncode != 0
    assert result.stdout == b""
    assert result.stderr == b"native host fixture request rejected\n"


@pytest.mark.parametrize("case", ["duplicate", "unknown", "float", "bool", "order", "hash", "null"])
def test_later_scenario_is_exact_and_closed(build: Path, case: str) -> None:
    exe = build / "topk_capsule_host_test"
    value = request(exe, "correctness", described(exe)["scenario"])
    coordinate = value["scenario"]["coordinates"][0]
    if case == "unknown":
        coordinate["extra"] = None
    elif case == "float":
        coordinate["order"] = 0.0
    elif case == "bool":
        coordinate["order"] = False
    elif case == "order":
        value["scenario"]["coordinates"].reverse()
    elif case == "null":
        value["prior_response_sha256"] = None
    value["scenario_contract_sha256"] = digest(canonical_json_bytes(value["scenario"]))
    if case == "hash":
        value["scenario_contract_sha256"] = "c" * 64
    raw = canonical_json_bytes(value)
    if case == "duplicate":
        raw = raw.replace(b'"order": 0,', b'"order": 0, "order": 0,', 1)
    result = invoke_bytes(exe, raw, "correctness")
    assert result.returncode != 0 and result.stdout == b""


@pytest.mark.parametrize(
    "readonly,seals,prefix",
    [
        (False, SEALS, "/proc/self/fd/"),
        (True, 0, "/proc/self/fd/"),
        (True, SEALS & ~0x0001, "/proc/self/fd/"),
        (True, SEALS, "/proc/self/fd/0"),
    ],
)
def test_exact_runner_fd_contract(build: Path, readonly: bool, seals: int, prefix: str) -> None:
    exe = build / "topk_capsule_host_test"
    result = invoke_bytes(
        exe, canonical_json_bytes(request(exe)), readonly=readonly, seals=seals, path_prefix=prefix
    )
    assert result.returncode != 0 and result.stdout == b""


def test_request_size_bound(build: Path) -> None:
    result = invoke_bytes(build / "topk_capsule_host_test", b" " * (1024 * 1024 + 1))
    assert result.returncode != 0 and result.stdout == b""


def manifest(candidate: str) -> CapsuleManifestV1:
    # In-memory adapter fixture only: this is never sent to production run_capsule.
    return CapsuleManifestV1.model_validate(
        {
            "schema_version": 1,
            "id": IDENTITY,
            "candidate": candidate,
            "machine": "host-fixture",
            "build": {
                "source_id": "host-fixture",
                "source_commit": "a" * 40,
                "toolchain_mode": "host",
                "gfx_target": "gfx1151",
                "target": "topk_capsule_host_test",
            },
            "contract": {
                "protocol": "native-capsule-v1",
                "scenario_sha256": SCENARIO_SHA,
                "comparison": {
                    "policy": "paired-latency-log-bootstrap-v1",
                    "protected_regression_bps": None,
                    "permitted_arm_differences": ["candidate-id"],
                },
            },
            "timeouts": {
                "describe_seconds": 5.0,
                "correctness_seconds": 5.0,
                "benchmark_seconds": 5.0,
            },
        }
    )


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("none", "passed"),
        ("boundary_tie", "correctness-failed"),
        ("zero_sign", "correctness-failed"),
        ("duplicate", "correctness-failed"),
        ("index", "correctness-failed"),
        ("count", "correctness-failed"),
        ("order", "correctness-failed"),
        ("nan", "correctness-failed"),
        ("replay_order", "correctness-failed"),
        ("benchmark_unready", "benchmark-process-failed"),
    ],
)
def test_real_native_protocol_gate(
    build: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, reason: str
) -> None:
    exe = build / ("topk_capsule_host_test" if fault == "none" else f"topk_capsule_fault_{fault}")
    observed, evidence = run_native(exe, tmp_path, monkeypatch)
    assert observed.reason == reason
    expected = ["describe", "correctness"] + (
        ["benchmark"] if fault in {"none", "benchmark_unready"} else []
    )
    assert [phase.operation for phase in observed.phases] == expected
    raw = json.loads(evidence["capsule/protocol/correctness/stdout.json"])
    if fault == "nan":
        assert raw["opaque_payload"]["reason"] == "nan-input"
        assert raw["opaque_payload"]["setup_calls"] == 0
        assert raw["opaque_payload"]["operation_calls"] == 0
    if fault == "benchmark_unready":
        assert all(item["passed"] for item in raw["coordinates"])
        assert "capsule/protocol/benchmark/stdout.json" not in evidence


def run_native(
    exe: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate: str = "host-fixture",
    drift: bool = False,
) -> tuple[Any, dict[str, bytes]]:
    original = capsules.run_process
    invoked = []

    def tracked(*args: Any, **kwargs: Any) -> Any:
        invoked.append(args[0][1])
        result = original(*args, **kwargs)
        if drift and args[0][1] == "describe":
            with exe.open("ab") as stream:
                stream.write(b"drift")
        return result

    monkeypatch.setattr(capsules, "run_process", tracked)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    with begin_run(
        "host-transport-test",
        b"fixture\n",
        resolved={"fixture": True},
        home=tmp_path / "home",
        environ={},
    ) as run:
        result = run_capsule_protocol(
            run,
            manifest(candidate),
            manifest_sha256="a" * 64,
            executable_path=exe,
            executable_sha256=digest(exe.read_bytes()),
            cwd=tmp_path,
            environment={"PATH": "/usr/bin:/bin"},
            scratch_root=scratch,
            redaction_context=RedactionContext(()),
        )
        assert invoked == [phase.operation for phase in result.phases]
        evidence = {
            entry.logical_path: (run.active / "portable/blobs" / entry.blob_sha256).read_bytes()
            for entry in list_portable_entries(run.active)
        }
        run.fail("host-transport-test-complete")
    return result, evidence


@pytest.mark.parametrize("provider", ["baseline-hip", "rocprim-topk", "rocprim-segmented-topk"])
def test_real_providers_unavailable(
    build: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    result, evidence = run_native(build / "topk_capsule_host_test", tmp_path, monkeypatch, provider)
    assert result.reason == "correctness-failed"
    assert [phase.operation for phase in result.phases] == ["describe", "correctness"]
    details = json.loads(evidence["capsule/protocol/correctness/stdout.json"])["opaque_payload"]
    assert details["reason"] == "provider-unavailable" and details["setup_calls"] == 0


def test_fresh_process_readiness_is_not_a_prior_hash(build: Path) -> None:
    exe = build / "topk_capsule_fault_benchmark_unready"
    value = request(exe, "benchmark", described(exe)["scenario"])
    # Well-shaped arbitrary prior hash and accepted contract cannot bypass local readiness.
    result = invoke_bytes(exe, canonical_json_bytes(value), "benchmark")
    assert result.returncode != 0 and result.stdout == b""
    assert result.stderr == b"native host fixture benchmark readiness failed\n"


def test_native_executable_drift_rejected(
    build: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = tmp_path / "fixture"
    shutil.copy2(build / "topk_capsule_host_test", exe)
    with pytest.raises(CapsuleIntegrityError):
        run_native(exe, tmp_path, monkeypatch, drift=True)


def test_vendor_provenance_pins_content() -> None:
    root = ROOT / "native/topk/third_party/nlohmann"
    provenance = json.loads((root / "provenance.json").read_bytes())
    assert provenance["release_tag"] == "v3.12.0"
    assert provenance["commit"] == "55f93686c01528224f448c19128836e7df245f72"
    for item in provenance["files"]:
        assert digest((root / item["path"]).read_bytes()) == item["sha256"]
        assert f"/{provenance['commit']}/" in item["source_url"]

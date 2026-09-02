from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import _suite_fixtures as fx
import pytest
from test_capsule_runs import _hooks, _run

import strixlab.capsule_snapshots as snapshots
from strixlab.capsule_snapshots import CapsuleSnapshotError, load_finalized_capsule_snapshot
from strixlab.evidence import (
    PortableEvidenceV1,
    inspect_run,
    list_portable_entries,
    read_record_member,
)
from strixlab.locks import LockAttempt, LockStatus
from strixlab.serialization import canonical_json_bytes

_ERROR = "finalized capsule snapshot authentication failed"


@pytest.fixture(autouse=True)
def _stub_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    fx.stub_cache_verification(monkeypatch)


def _assert_closed(call: Callable[[], object]) -> None:
    with pytest.raises(CapsuleSnapshotError) as raised:
        call()
    assert str(raised.value) == _ERROR


class _MemoryEvidence:
    def __init__(self, run_id: str, home: Path) -> None:
        self.inspection = inspect_run(run_id, home=home)
        self.entries = list(list_portable_entries(self.inspection.record))
        self.manifest = read_record_member(self.inspection.record, "manifest.resolved.yaml")
        self.payloads = {
            entry.logical_path: read_record_member(
                self.inspection.record, f"portable/blobs/{entry.blob_sha256}"
            )
            for entry in self.entries
        }

    def _index(self, logical_path: str) -> int:
        return next(
            index for index, entry in enumerate(self.entries) if entry.logical_path == logical_path
        )

    def replace(self, logical_path: str, content: bytes) -> None:
        index = self._index(logical_path)
        digest = hashlib.sha256(content).hexdigest()
        self.entries[index] = self.entries[index].model_copy(
            update={"blob_sha256": digest, "size_bytes": len(content)}
        )
        self.payloads[logical_path] = content

    def json(self, logical_path: str, mutate: Callable[[dict[str, Any]], None]) -> None:
        value = json.loads(self.payloads[logical_path])
        mutate(value)
        self.replace(logical_path, canonical_json_bytes(value))

    def remove(self, logical_path: str) -> None:
        self.entries.pop(self._index(logical_path))
        self.payloads.pop(logical_path)

    def rename(self, logical_path: str, replacement: str) -> None:
        index = self._index(logical_path)
        self.entries[index] = self.entries[index].model_copy(update={"logical_path": replacement})
        self.payloads[replacement] = self.payloads.pop(logical_path)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(snapshots, "inspect_run", lambda *_a, **_k: self.inspection)
        monkeypatch.setattr(snapshots, "list_portable_entries", lambda _record: tuple(self.entries))

        def read(_record: Path, relative: str) -> bytes:
            if relative == "manifest.resolved.yaml":
                return self.manifest
            digest = relative.removeprefix("portable/blobs/")
            matches = [
                self.payloads[entry.logical_path]
                for entry in self.entries
                if entry.blob_sha256 == digest
            ]
            if len(matches) != 1:
                raise OSError("missing test member")
            return matches[0]

        monkeypatch.setattr(snapshots, "read_record_member", read)


@pytest.fixture
def successful(tmp_path: Path) -> tuple[Path, Any]:
    result = _run(tmp_path)
    return tmp_path / "home", result


def test_success_returns_complete_immutable_projection(successful: tuple[Path, Any]) -> None:
    home, run = successful
    value = load_finalized_capsule_snapshot(run.run_id, home=home)

    assert value.run_id == run.run_id
    assert value.record_sha256 == run.inspection.record_sha256
    assert value.resolved_manifest_sha256 == value.result.manifest_sha256
    assert value.protocol_sha256 == value.result.protocol_result_sha256
    assert tuple(coordinate.coordinate_id for coordinate in value.coordinates) == (
        "train-forward",
        "eval-reverse",
    )
    assert tuple(coordinate.coordinate_id for coordinate in value.training_coordinates) == (
        "train-forward",
    )
    assert tuple(coordinate.coordinate_id for coordinate in value.evaluation_coordinates) == (
        "eval-reverse",
    )
    assert value.latency_seconds_by_coordinate["train-forward"] == (0.01, 0.011, 0.012)
    assert value.workspace_bytes_by_coordinate == {
        "train-forward": 4096,
        "eval-reverse": 5120,
    }
    assert value.alignment.coordinate_ids == ("train-forward", "eval-reverse")
    assert "opaque_payload" not in value.__dataclass_fields__
    with pytest.raises(TypeError):
        value.workspace_bytes_by_coordinate["train-forward"] = 0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        value.run_id = "changed"  # type: ignore[misc]


def test_failed_and_lock_refused_runs_are_rejected(tmp_path: Path) -> None:
    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    failed = _run(failed_root, mode="correctness-fail")
    _assert_closed(
        lambda: load_finalized_capsule_snapshot(failed.run_id, home=failed_root / "home")
    )

    @contextlib.contextmanager
    def refusing(_path: Path) -> Iterator[LockAttempt]:
        yield LockAttempt(LockStatus.CONTENDED, _path)

    locked_root = tmp_path / "locked"
    locked_root.mkdir()
    locked = _run(locked_root, hooks=_hooks(locked_root, machine_lock=refusing))
    _assert_closed(
        lambda: load_finalized_capsule_snapshot(locked.run_id, home=locked_root / "home")
    )


def test_loader_api_is_deterministic_and_errors_are_fixed_safe(
    successful: tuple[Path, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    home, run = successful
    monkeypatch.setenv("API_TOKEN", "candidate-a")
    assert load_finalized_capsule_snapshot(run.run_id, home=home).run_id == run.run_id
    with pytest.raises(TypeError):
        load_finalized_capsule_snapshot(run.run_id, home=home, environ={})  # type: ignore[call-arg]
    _assert_closed(lambda: load_finalized_capsule_snapshot("not-a-run", home=home))


def test_intrinsic_secret_context_preserves_duplicate_sensitive_names(
    successful: tuple[Path, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    home, run = successful
    evidence = _MemoryEvidence(run.run_id, home)
    build = json.loads(evidence.payloads["capsule/build.json"])
    build["canonical"]["environment"].extend(
        (
            {"name": "API_TOKEN", "value": "intrinsic-secret-value"},
            {"name": "API_TOKEN", "value": ""},
        )
    )
    canonical_sha = (
        "record-sha256:" + hashlib.sha256(canonical_json_bytes(build["canonical"])).hexdigest()
    )
    build["canonical_record_sha256"] = canonical_sha
    evidence.replace("capsule/build.json", canonical_json_bytes(build))
    build_sha = evidence.entries[evidence._index("capsule/build.json")].blob_sha256

    def bind_build(value: dict[str, Any]) -> None:
        value["canonical_record_sha256"] = canonical_sha
        value["inputs"][0]["sha256"] = build_sha

    evidence.json("capsule/result.json", bind_build)
    parsed = snapshots._canonical_json(  # type: ignore[attr-defined]
        evidence.payloads["capsule/build.json"], snapshots._BuildSnapshotV1
    )
    assert snapshots._intrinsic_context(parsed).secrets == ("intrinsic-secret-value",)
    evidence.install(monkeypatch)
    _assert_closed(lambda: load_finalized_capsule_snapshot(run.run_id, home=home))


@pytest.mark.parametrize(
    ("path", "key", "value"),
    [
        ("capsule/result.json", "candidate", "wrong-candidate"),
        ("capsule/result.json", "manifest_sha256", "0" * 64),
        ("capsule/result.json", "scenario_sha256", "0" * 64),
        ("capsule/result.json", "target", "wrong-target"),
        ("capsule/result.json", "executable_sha256", "0" * 64),
        ("capsule/build.json", "build_id", "build-sha256:" + "0" * 64),
        ("capsule/build.json", "canonical_record_sha256", "record-sha256:" + "0" * 64),
        ("capsule/machine.json", "machine_id", "wrong-machine"),
        ("capsule/machine.json", "profile_sha256", "0" * 64),
        ("capsule/protocol/result.json", "manifest_sha256", "0" * 64),
        ("capsule/protocol/result.json", "executable_sha256", "0" * 64),
    ],
)
def test_envelope_manifest_input_protocol_and_identity_tampering_fails(
    successful: tuple[Path, Any],
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    key: str,
    value: str,
) -> None:
    home, run = successful
    evidence = _MemoryEvidence(run.run_id, home)
    evidence.json(path, lambda payload: payload.__setitem__(key, value))
    evidence.install(monkeypatch)
    _assert_closed(lambda: load_finalized_capsule_snapshot(run.run_id, home=home))


def test_manifest_build_source_toolchain_gfx_and_artifact_replay(
    successful: tuple[Path, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    home, run = successful
    mutations: tuple[Callable[[dict[str, Any]], None], ...] = (
        lambda value: value["canonical"]["source"]["source_evidence"].__setitem__(
            "source_id", "wrong-source"
        ),
        lambda value: value["canonical"]["source"]["source_evidence"].__setitem__(
            "base_commit", "0" * 40
        ),
        lambda value: value["canonical"].__setitem__("toolchain_mode", "rocm"),
        lambda value: value["canonical"]["selections"][0].__setitem__("value", "gfx900"),
        lambda value: value["canonical"]["artifacts"]["artifacts"][0].__setitem__(
            "sha256", "0" * 64
        ),
    )
    for mutate in mutations:
        evidence = _MemoryEvidence(run.run_id, home)
        evidence.json("capsule/build.json", mutate)
        with monkeypatch.context() as scoped:
            evidence.install(scoped)
            _assert_closed(lambda: load_finalized_capsule_snapshot(run.run_id, home=home))


@pytest.mark.parametrize("operation", ["describe", "correctness", "benchmark"])
@pytest.mark.parametrize("member", ["request.json", "process.json", "stdout.json"])
def test_every_phase_member_tamper_fails(
    successful: tuple[Path, Any],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    member: str,
) -> None:
    home, run = successful
    evidence = _MemoryEvidence(run.run_id, home)
    path = f"capsule/protocol/{operation}/{member}"
    evidence.replace(path, evidence.payloads[path] + b" ")
    evidence.install(monkeypatch)
    _assert_closed(lambda: load_finalized_capsule_snapshot(run.run_id, home=home))


@pytest.mark.parametrize(
    ("operation", "field"),
    [
        ("describe", "candidate"),
        ("correctness", "prior_response_sha256"),
        ("correctness", "scenario_contract_sha256"),
        ("correctness", "scenario"),
        ("benchmark", "prior_response_sha256"),
        ("benchmark", "scenario_contract_sha256"),
        ("benchmark", "scenario"),
    ],
)
def test_request_identity_prior_and_scenario_chains_fail_closed(
    successful: tuple[Path, Any],
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    field: str,
) -> None:
    home, run = successful
    evidence = _MemoryEvidence(run.run_id, home)
    path = f"capsule/protocol/{operation}/request.json"

    def mutate(value: dict[str, Any]) -> None:
        if field == "scenario":
            value[field]["coordinates"][0]["input_sha256"] = "0" * 64
        else:
            value[field] = "0" * 64 if field != "candidate" else "wrong-candidate"

    evidence.json(path, mutate)
    evidence.install(monkeypatch)
    _assert_closed(lambda: load_finalized_capsule_snapshot(run.run_id, home=home))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("request_sha256", "0" * 64),
        lambda value: value.__setitem__("candidate", "wrong-candidate"),
        lambda value: value.__setitem__("scenario_sha256", "0" * 64),
        lambda value: value.__setitem__("manifest_sha256", "0" * 64),
        lambda value: value.__setitem__("executable_sha256", "0" * 64),
        lambda value: value.__setitem__("prior_response_sha256", "0" * 64),
        lambda value: value.__setitem__("scenario_contract_sha256", "0" * 64),
    ],
)
def test_response_echo_mismatches_fail(
    successful: tuple[Path, Any],
    monkeypatch: pytest.MonkeyPatch,
    mutation: Callable[[dict[str, Any]], None],
) -> None:
    home, run = successful
    evidence = _MemoryEvidence(run.run_id, home)
    evidence.json("capsule/protocol/benchmark/stdout.json", mutation)
    evidence.install(monkeypatch)
    _assert_closed(lambda: load_finalized_capsule_snapshot(run.run_id, home=home))


@pytest.mark.parametrize("path", ["capsule/build.json", "capsule/protocol/describe/request.json"])
def test_wrong_role_or_media_fails(
    successful: tuple[Path, Any],
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    home, run = successful
    evidence = _MemoryEvidence(run.run_id, home)
    index = evidence._index(path)
    evidence.entries[index] = evidence.entries[index].model_copy(
        update={"role": "summary", "media_type": "text/plain"}
    )
    evidence.install(monkeypatch)
    _assert_closed(lambda: load_finalized_capsule_snapshot(run.run_id, home=home))


@pytest.mark.parametrize("kind", ["missing", "duplicate", "fallback", "unexpected"])
def test_missing_duplicate_fallback_and_unexpected_paths_fail(
    successful: tuple[Path, Any],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    home, run = successful
    evidence = _MemoryEvidence(run.run_id, home)
    target = "capsule/protocol/describe/stdout.json"
    if kind == "missing":
        evidence.remove(target)
    elif kind == "duplicate":
        evidence.entries.append(evidence.entries[evidence._index(target)])
    elif kind == "fallback":
        evidence.rename(target, "capsule/protocol/describe/stdout.txt")
    else:
        evidence.entries.append(
            PortableEvidenceV1(
                sequence=len(evidence.entries) + 1,
                logical_path="capsule/unexpected.json",
                role="summary",
                media_type="application/json",
                blob_sha256=hashlib.sha256(b"{}\n").hexdigest(),
                size_bytes=3,
            )
        )
        evidence.payloads["capsule/unexpected.json"] = b"{}\n"
    evidence.install(monkeypatch)
    _assert_closed(lambda: load_finalized_capsule_snapshot(run.run_id, home=home))


def test_noncanonical_strict_invalid_and_sensitive_interpolation_fail(
    successful: tuple[Path, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    home, run = successful
    mutations = (
        lambda evidence: evidence.replace(
            "capsule/result.json", evidence.payloads["capsule/result.json"] + b" "
        ),
        lambda evidence: evidence.json(
            "capsule/protocol/describe/process.json",
            lambda value: value.__setitem__("stdout_bytes", "1"),
        ),
        lambda evidence: evidence.json(
            "capsule/protocol/describe/stdout.json",
            lambda value: value.__setitem__("opaque_payload", "${API_TOKEN}"),
        ),
    )
    for mutate in mutations:
        evidence = _MemoryEvidence(run.run_id, home)
        mutate(evidence)
        with monkeypatch.context() as scoped:
            evidence.install(scoped)
            _assert_closed(lambda: load_finalized_capsule_snapshot(run.run_id, home=home))


def test_zero_stderr_must_be_absent(
    successful: tuple[Path, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    home, run = successful
    evidence = _MemoryEvidence(run.run_id, home)
    content = b"safe stderr\n"
    digest = hashlib.sha256(content).hexdigest()
    stdout_index = evidence._index("capsule/protocol/describe/stdout.json")
    evidence.entries.insert(
        stdout_index + 1,
        PortableEvidenceV1(
            sequence=4,
            logical_path="capsule/protocol/describe/stderr.txt",
            role="correctness",
            media_type="text/plain",
            blob_sha256=digest,
            size_bytes=len(content),
        ),
    )
    evidence.payloads["capsule/protocol/describe/stderr.txt"] = content
    evidence.install(monkeypatch)
    _assert_closed(lambda: load_finalized_capsule_snapshot(run.run_id, home=home))


def _with_authenticated_describe_stderr(
    evidence: _MemoryEvidence, content: bytes = b"safe diagnostic\n"
) -> None:
    digest = hashlib.sha256(content).hexdigest()
    process_path = "capsule/protocol/describe/process.json"

    def process_stderr(value: dict[str, Any]) -> None:
        value["stderr_bytes"] = len(content)
        value["stderr_sha256"] = digest

    evidence.json(process_path, process_stderr)
    process = json.loads(evidence.payloads[process_path])
    evidence.json(
        "capsule/protocol/result.json",
        lambda value: value["phases"][0].__setitem__("process", process),
    )
    protocol_sha = evidence.entries[evidence._index("capsule/protocol/result.json")].blob_sha256
    evidence.json(
        "capsule/result.json",
        lambda value: value.__setitem__("protocol_result_sha256", protocol_sha),
    )
    stdout_index = evidence._index("capsule/protocol/describe/stdout.json")
    evidence.entries.insert(
        stdout_index + 1,
        PortableEvidenceV1(
            sequence=4,
            logical_path="capsule/protocol/describe/stderr.txt",
            role="correctness",
            media_type="text/plain",
            blob_sha256=digest,
            size_bytes=len(content),
        ),
    )
    evidence.payloads["capsule/protocol/describe/stderr.txt"] = content


def test_nonempty_stderr_is_authenticated_and_required(
    successful: tuple[Path, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    home, run = successful
    evidence = _MemoryEvidence(run.run_id, home)
    _with_authenticated_describe_stderr(evidence)
    with monkeypatch.context() as scoped:
        evidence.install(scoped)
        value = load_finalized_capsule_snapshot(run.run_id, home=home)
        assert value.protocol.status == "passed"

    evidence.remove("capsule/protocol/describe/stderr.txt")
    evidence.install(monkeypatch)
    _assert_closed(lambda: load_finalized_capsule_snapshot(run.run_id, home=home))


def test_nonempty_stderr_rejects_sensitive_interpolation(
    successful: tuple[Path, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    home, run = successful
    evidence = _MemoryEvidence(run.run_id, home)
    _with_authenticated_describe_stderr(evidence, b"diagnostic ${API_TOKEN}\n")
    evidence.install(monkeypatch)
    _assert_closed(lambda: load_finalized_capsule_snapshot(run.run_id, home=home))

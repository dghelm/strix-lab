from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from strixlab import models
from strixlab.manifests import (
    MetadataPredicateV1,
    ModelArchitectureV1,
    ModelArtifactV1,
    ModelBaseV1,
    ModelExecutionV1,
    ModelFileIdentityV1,
    ModelManifestV1,
    ModelQuantizationV1,
    ModelSidecarV1,
)
from strixlab.models import (
    GgufInspectorBindingV1,
    ModelArtifactError,
    ModelCacheBusyError,
    ModelCacheError,
    ModelCompatibilityError,
    ModelInspectorError,
    ModelManifestError,
    ModelMetadataError,
    ModelSidecarError,
    inspect_local_gguf,
    lease_verified_model,
    load_model_receipt,
    manifest_digest,
    receipt_evidence_digest,
    require_current_model,
    verify_registered_model,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "models" / "gguf-dump-ca94157"
GGUF_FIXTURE = FIXTURE_DIR / "tiny-qwen35.gguf"
DUMP_FIXTURE = FIXTURE_DIR / "gguf_dump.json"

ANCHOR = models.SOURCE_ANCHOR_COMMIT
CANDIDATE_ID = "candidate-sha256:" + "1" * 64
CONTENT_TREE_ID = "content-tree-sha256:" + "2" * 64
PREPARATION_ID = "prep-strix-llama-" + "a" * 24


# --- Fixtures and harness -----------------------------------------------------


@dataclass
class FakeEvidence:
    base_commit: str = ANCHOR
    preparation_id: str = PREPARATION_ID
    candidate_id: str = CANDIDATE_ID
    content_tree_id: str = CONTENT_TREE_ID
    source_id: str = "strix-llama"
    patches: tuple[object, ...] = ()
    status: tuple[str, ...] = ()


@dataclass
class FakeLease:
    worktree: Path
    evidence: FakeEvidence = field(default_factory=FakeEvidence)
    verify_calls: int = 0
    fail_after: int | None = None

    def verify(self) -> None:
        self.verify_calls += 1
        if self.fail_after is not None and self.verify_calls > self.fail_after:
            raise RuntimeError("lease diverged")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_worktree(root: Path) -> tuple[Path, str, str, str]:
    worktree = root / "worktree"
    scripts = worktree / "gguf-py" / "gguf" / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "gguf_dump.py"
    script.write_text("# pinned inspector script placeholder\n", encoding="utf-8")
    return worktree, "gguf-py/gguf/scripts/gguf_dump.py", "gguf-py", _sha256_file(script)


def _make_python_stub(root: Path, name: str, body: str) -> tuple[Path, str]:
    stub = root / name
    stub.write_text(f"#!/usr/bin/python3\n{body}", encoding="utf-8")
    stub.chmod(0o755)
    return stub, _sha256_file(stub)


def _print_bytes_stub(root: Path, name: str, payload: bytes) -> tuple[Path, str]:
    body = f"import sys\nsys.stdout.buffer.write({payload!r})\n"
    return _make_python_stub(root, name, body)


def _make_inspector(
    *, python: Path, python_sha: str, script_rel: str, script_sha: str, gguf_py_rel: str
) -> GgufInspectorBindingV1:
    return GgufInspectorBindingV1(
        preparation_id=PREPARATION_ID,
        candidate_id=CANDIDATE_ID,
        content_tree_id=CONTENT_TREE_ID,
        base_commit=ANCHOR,
        python_executable=str(python),
        python_sha256=python_sha,
        gguf_py_relative_root=gguf_py_rel,
        script_relative_path=script_rel,
        script_sha256=script_sha,
    )


@dataclass
class Harness:
    home: Path
    lease: FakeLease
    inspector: GgufInspectorBindingV1


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    home = tmp_path / "home"
    home.mkdir()
    worktree, script_rel, gguf_py_rel, script_sha = _make_worktree(tmp_path)
    python, python_sha = _print_bytes_stub(tmp_path, "python-stub", DUMP_FIXTURE.read_bytes())
    inspector = _make_inspector(
        python=python,
        python_sha=python_sha,
        script_rel=script_rel,
        script_sha=script_sha,
        gguf_py_rel=gguf_py_rel,
    )
    return Harness(home=home, lease=FakeLease(worktree=worktree), inspector=inspector)


def _model_file(root: Path, name: str = "model.gguf") -> Path:
    target = root / name
    target.write_bytes(GGUF_FIXTURE.read_bytes())
    return target


def _registered_manifest(model_path: Path, **overrides: Any) -> ModelManifestV1:
    file = ModelFileIdentityV1(
        repository="bartowski/Qwen_Qwen3.5-2B-GGUF",
        revision="7d26695454df6de5fbcce2e58681e62dae06ce43",
        filename="model.gguf",
        local_path=str(model_path),
        size_bytes=model_path.stat().st_size,
        sha256=_sha256_file(model_path),
    )
    data: dict[str, Any] = dict(
        schema_version=1,
        id="qwen35-2b-smoke",
        registry_status="registered",
        base_model=ModelBaseV1(
            repository="Qwen/Qwen3.5-2B",
            revision="15852e8c16360a2fea060d615a32b45270f8a8fc",
            license="Apache-2.0",
        ),
        architecture=ModelArchitectureV1(
            family="qwen3_5",
            moe=False,
            gated_deltanet=True,
            full_attention=True,
            qsa=False,
            mtp=True,
            vision=True,
        ),
        artifact=ModelArtifactV1(
            format="gguf",
            file=file,
            metadata_predicates=[
                MetadataPredicateV1(
                    key="general.architecture", value_type="STRING", scalar_value="qwen35"
                )
            ],
        ),
        quantization=ModelQuantizationV1(
            format_family="Q4_K",
            storage_format="gguf",
            tensor_policy_id="unknown",
            tensor_policy_source="unknown",
            calibration_method="unknown",
            calibration_source="unknown",
            calibration_hash="unknown",
        ),
        execution=ModelExecutionV1(verification_status="unverified"),
    )
    data.update(overrides)
    return ModelManifestV1(**data)


# --- Happy paths --------------------------------------------------------------


def test_verify_registered_model_produces_deterministic_receipt(
    harness: Harness, tmp_path: Path
) -> None:
    model_path = _model_file(tmp_path)
    manifest = _registered_manifest(model_path)
    receipt = verify_registered_model(
        manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
    )
    assert receipt.trust_state == "verified"
    assert receipt.primary.sha256 == _sha256_file(model_path)
    assert receipt.metadata.tensor_count == 2
    assert dict(receipt.metadata.tensor_type_counts) == {"F16": 1, "F32": 1}
    assert receipt.compatibility == "verified"
    assert receipt.publishable is False  # quant provenance is unknown

    # The embedded portable evidence digest is stable and matches the manifest digest.
    digest = receipt_evidence_digest(receipt.evidence)
    assert len(digest) == 64 and digest == receipt_evidence_digest(receipt.evidence)
    assert receipt.evidence.manifest_sha256 == manifest_digest(manifest)

    # Content-addressed metadata was published and is byte-addressed by its digest.
    metadata_file = harness.home / receipt.metadata.metadata_registry_path
    assert metadata_file.is_file()
    assert (
        hashlib.sha256(metadata_file.read_bytes()).hexdigest() == receipt.metadata.metadata_sha256
    )

    # Verifying again is deterministic.
    again = verify_registered_model(
        manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
    )
    assert again.model_dump(mode="json") == receipt.model_dump(mode="json")


def test_receipt_round_trips_through_the_local_registry(harness: Harness, tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)
    manifest = _registered_manifest(model_path)
    receipt = verify_registered_model(
        manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
    )
    _, digest = models._receipt_envelope(receipt)
    loaded = load_model_receipt(manifest.id, digest, home=harness.home)
    assert loaded.model_dump(mode="json") == receipt.model_dump(mode="json")


def test_inspect_local_gguf_is_unregistered(harness: Harness, tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)
    observation = inspect_local_gguf(
        model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
    )
    assert observation.trust_state == "unregistered"
    assert observation.primary.sha256 == _sha256_file(model_path)
    assert observation.metadata.tensor_count == 2


def test_real_and_synthetic_metadata_agree(harness: Harness, tmp_path: Path) -> None:
    # The synthetic stub replays the exact captured tool output, so its normalized
    # projection must be byte-identical to what the golden fixture describes.
    model_path = _model_file(tmp_path)
    observation = inspect_local_gguf(
        model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
    )
    stored = (harness.home / observation.metadata.metadata_registry_path).read_bytes()
    projection = json.loads(stored)
    assert projection["domain"] == models._METADATA_DOMAIN
    assert projection["endian"] == "LITTLE"
    assert projection["metadata"]["general.architecture"] == {"type": "STRING", "value": "qwen35"}
    assert projection["metadata"]["qwen35.tiny_list"] == {"type": "ARRAY", "array_types": ["INT32"]}
    assert "filename" not in projection and "index" not in json.dumps(projection)


# --- Stub helpers for failure modes -------------------------------------------


def _dump_with(**changes: Any) -> bytes:
    data = json.loads(DUMP_FIXTURE.read_bytes())
    for key, value in changes.items():
        data[key] = value
    return json.dumps(data).encode("utf-8")


def _install_stub(harness: Harness, root: Path, payload: bytes, *, name: str = "stub") -> None:
    python, python_sha = _print_bytes_stub(root, name, payload)
    harness.inspector = _rebind(harness.inspector, python=python, python_sha=python_sha)


def _install_raw_stub(harness: Harness, root: Path, body: str, *, name: str = "stub") -> None:
    python, python_sha = _make_python_stub(root, name, body)
    harness.inspector = _rebind(harness.inspector, python=python, python_sha=python_sha)


def _rebind(
    inspector: GgufInspectorBindingV1, *, python: Path, python_sha: str
) -> GgufInspectorBindingV1:
    return inspector.model_copy(
        update={"python_executable": str(python), "python_sha256": python_sha}
    )


# --- Manifest / compatibility failures ----------------------------------------


def test_verify_refuses_a_draft(harness: Harness, tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)
    draft = ModelManifestV1(
        schema_version=1,
        id="qwen38-27b",
        registry_status="draft",
        artifact=ModelArtifactV1(format="gguf", file=ModelFileIdentityV1()),
        quantization=ModelQuantizationV1(
            format_family="unknown",
            storage_format="gguf",
            tensor_policy_id="unknown",
            tensor_policy_source="unknown",
            calibration_method="unknown",
            calibration_source="unknown",
            calibration_hash="unknown",
        ),
        execution=ModelExecutionV1(verification_status="unverified"),
        draft_reason="no reviewed conversion repository/revision or recipe is pinned yet",
    )
    assert model_path.exists()
    with pytest.raises(ModelManifestError):
        verify_registered_model(
            draft, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_size_mismatch_fails(harness: Harness, tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)
    file = ModelFileIdentityV1(
        repository="a/b",
        revision="0" * 40,
        filename="model.gguf",
        local_path=str(model_path),
        size_bytes=model_path.stat().st_size + 1,
        sha256=_sha256_file(model_path),
    )
    manifest = _registered_manifest(model_path).model_copy(
        update={
            "artifact": _registered_manifest(model_path).artifact.model_copy(update={"file": file})
        }
    )
    with pytest.raises(ModelArtifactError):
        verify_registered_model(
            manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_sha_mismatch_fails(harness: Harness, tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)
    manifest = _registered_manifest(model_path)
    bad_file = manifest.artifact.file.model_copy(update={"sha256": "0" * 64})
    manifest = manifest.model_copy(
        update={"artifact": manifest.artifact.model_copy(update={"file": bad_file})}
    )
    with pytest.raises(ModelArtifactError):
        verify_registered_model(
            manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_unmapped_family_is_compatibility_failure(harness: Harness, tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)
    manifest = _registered_manifest(model_path)
    arch = manifest.architecture.model_copy(update={"family": "unmapped_family"})
    predicates = [
        MetadataPredicateV1(key="general.architecture", value_type="STRING", scalar_value="qwen35")
    ]
    manifest = manifest.model_copy(
        update={
            "architecture": arch,
            "artifact": manifest.artifact.model_copy(update={"metadata_predicates": predicates}),
        }
    )
    with pytest.raises(ModelCompatibilityError):
        verify_registered_model(
            manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_architecture_value_mismatch_is_compatibility_failure(
    harness: Harness, tmp_path: Path
) -> None:
    model_path = _model_file(tmp_path)
    metadata = json.loads(DUMP_FIXTURE.read_bytes())["metadata"]
    metadata["general.architecture"] = {
        "index": 3,
        "type": "STRING",
        "offset": 24,
        "value": "llama",
    }
    _install_stub(harness, tmp_path, _dump_with(metadata=metadata))
    manifest = _registered_manifest(model_path)
    with pytest.raises(ModelCompatibilityError):
        verify_registered_model(
            manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_predicate_mismatch_is_compatibility_failure(harness: Harness, tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)
    manifest = _registered_manifest(model_path)
    predicates = [
        MetadataPredicateV1(key="general.architecture", value_type="STRING", scalar_value="qwen35"),
        MetadataPredicateV1(key="general.name", value_type="STRING", scalar_value="not-the-name"),
    ]
    manifest = manifest.model_copy(
        update={
            "artifact": manifest.artifact.model_copy(update={"metadata_predicates": predicates})
        }
    )
    with pytest.raises(ModelCompatibilityError):
        verify_registered_model(
            manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


# --- Inspector process / metadata defects -------------------------------------


def test_inspector_nonzero_exit_fails(harness: Harness, tmp_path: Path) -> None:
    _install_raw_stub(harness, tmp_path, "import sys\nsys.exit(3)\n")
    model_path = _model_file(tmp_path)
    with pytest.raises(ModelInspectorError):
        inspect_local_gguf(
            model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_inspector_spawn_failure_fails(harness: Harness, tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    missing.write_text("#!/nonexistent/interp\n")
    missing.chmod(0o755)
    harness.inspector = _rebind(harness.inspector, python=missing, python_sha=_sha256_file(missing))
    model_path = _model_file(tmp_path)
    with pytest.raises(ModelInspectorError):
        inspect_local_gguf(
            model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_inspector_timeout_fails(harness: Harness, tmp_path: Path) -> None:
    _install_raw_stub(harness, tmp_path, "import time\ntime.sleep(30)\n")
    model_path = _model_file(tmp_path)
    with pytest.raises(ModelInspectorError):
        inspect_local_gguf(
            model_path,
            inspector=harness.inspector,
            source_lease=harness.lease,
            home=harness.home,
            timeout=0.5,
        )


def test_inspector_hard_limit_fails(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(models, "_INSPECTOR_STDOUT_LIMIT", 64)
    _install_stub(harness, tmp_path, b"x" * 4096)
    model_path = _model_file(tmp_path)
    with pytest.raises(ModelInspectorError):
        inspect_local_gguf(
            model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_inspector_invalid_utf8_fails(harness: Harness, tmp_path: Path) -> None:
    _install_stub(harness, tmp_path, b"\xff\xfe\xfa not utf-8")
    model_path = _model_file(tmp_path)
    with pytest.raises(ModelInspectorError):
        inspect_local_gguf(
            model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_inspector_duplicate_json_key_fails(harness: Harness, tmp_path: Path) -> None:
    _install_stub(
        harness, tmp_path, b'{"endian":"LITTLE","endian":"BIG","metadata":{},"tensors":{}}'
    )
    model_path = _model_file(tmp_path)
    with pytest.raises(ModelMetadataError):
        inspect_local_gguf(
            model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_inspector_nonfinite_number_fails(harness: Harness, tmp_path: Path) -> None:
    _install_stub(
        harness,
        tmp_path,
        b'{"endian":"LITTLE","metadata":{"k":{"type":"FLOAT32","value":Infinity}},"tensors":{}}',
    )
    model_path = _model_file(tmp_path)
    with pytest.raises(ModelMetadataError):
        inspect_local_gguf(
            model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_inspector_malformed_shape_fails(harness: Harness, tmp_path: Path) -> None:
    _install_stub(harness, tmp_path, b'{"endian":"LITTLE"}')
    model_path = _model_file(tmp_path)
    with pytest.raises(ModelMetadataError):
        inspect_local_gguf(
            model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_inspector_absolute_path_string_rejected(harness: Harness, tmp_path: Path) -> None:
    _install_stub(
        harness,
        tmp_path,
        b'{"endian":"LITTLE","metadata":{"k":{"type":"STRING","value":"/etc/passwd"}},"tensors":{}}',
    )
    model_path = _model_file(tmp_path)
    with pytest.raises(ModelMetadataError):
        inspect_local_gguf(
            model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


# --- Inspector binding failures -----------------------------------------------


def test_binding_rejects_wrong_base_commit(harness: Harness, tmp_path: Path) -> None:
    harness.lease.evidence.base_commit = "1" * 40
    model_path = _model_file(tmp_path)
    with pytest.raises(ModelInspectorError):
        inspect_local_gguf(
            model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_binding_rejects_candidate_mismatch(harness: Harness, tmp_path: Path) -> None:
    harness.lease.evidence.candidate_id = "candidate-sha256:" + "9" * 64
    model_path = _model_file(tmp_path)
    with pytest.raises(ModelInspectorError):
        inspect_local_gguf(
            model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_binding_rejects_patched_or_dirty_lease(harness: Harness, tmp_path: Path) -> None:
    harness.lease.evidence.status = (" M gguf-py/gguf/scripts/gguf_dump.py",)
    model_path = _model_file(tmp_path)
    with pytest.raises(ModelInspectorError):
        inspect_local_gguf(
            model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_binding_rejects_script_digest_mismatch(harness: Harness, tmp_path: Path) -> None:
    harness.inspector = harness.inspector.model_copy(update={"script_sha256": "0" * 64})
    model_path = _model_file(tmp_path)
    with pytest.raises(ModelInspectorError):
        inspect_local_gguf(
            model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_binding_rejects_script_path_escape(harness: Harness, tmp_path: Path) -> None:
    harness.inspector = harness.inspector.model_copy(
        update={"script_relative_path": "../outside/gguf_dump.py"}
    )
    model_path = _model_file(tmp_path)
    with pytest.raises(ModelInspectorError):
        inspect_local_gguf(
            model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


# --- Model-hash cache ---------------------------------------------------------


def _model_hash_spy(monkeypatch: pytest.MonkeyPatch, model_path: Path) -> list[int]:
    calls: list[int] = []
    original = models._hash_descriptor

    def spy(descriptor: int, identity: Any, *, path: Path, error: Any) -> str:
        if path == model_path:
            calls.append(1)
        return original(descriptor, identity, path=path, error=error)

    monkeypatch.setattr(models, "_hash_descriptor", spy)
    return calls


def test_cold_hash_then_hit_hashes_once(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = _model_file(tmp_path)
    manifest = _registered_manifest(model_path)
    calls = _model_hash_spy(monkeypatch, model_path)
    verify_registered_model(
        manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
    )
    verify_registered_model(
        manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
    )
    assert len(calls) == 1


def test_two_concurrent_cold_verifiers_hash_once(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = _model_file(tmp_path)
    manifest = _registered_manifest(model_path)
    original = models._hash_descriptor

    def slow(descriptor: int, identity: Any, *, path: Path, error: Any) -> str:
        if path == model_path:
            slow.count += 1  # type: ignore[attr-defined]
            _barrier.wait(timeout=5)
        return original(descriptor, identity, path=path, error=error)

    slow.count = 0  # type: ignore[attr-defined]
    _barrier = threading.Barrier(1)
    monkeypatch.setattr(models, "_hash_descriptor", slow)

    results: list[Any] = []

    def run() -> None:
        results.append(
            verify_registered_model(
                manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
            )
        )

    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert slow.count == 1  # type: ignore[attr-defined]
    assert len(results) == 2
    assert results[0].model_dump(mode="json") == results[1].model_dump(mode="json")


def test_cache_lock_timeout_is_busy(harness: Harness, tmp_path: Path) -> None:
    from strixlab.locks import exclusive_lock

    model_path = _model_file(tmp_path)
    manifest = _registered_manifest(model_path)
    models._prepare_home(harness.home)
    identity = models._identity_of(os.lstat(model_path))
    key = models._cache_key(model_path, identity)
    lock_path = harness.home.joinpath(*models._CACHE_DIR) / f"{key}.lock"
    with exclusive_lock(lock_path) as held:
        assert held.acquired
        with pytest.raises(ModelCacheBusyError):
            verify_registered_model(
                manifest,
                inspector=harness.inspector,
                source_lease=harness.lease,
                home=harness.home,
                lock_timeout=0.3,
            )


def test_malformed_cache_entry_is_integrity_failure(harness: Harness, tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)
    manifest = _registered_manifest(model_path)
    models._prepare_home(harness.home)
    identity = models._identity_of(os.lstat(model_path))
    key = models._cache_key(model_path, identity)
    (harness.home.joinpath(*models._CACHE_DIR) / f"{key}.json").write_bytes(b"not json")
    with pytest.raises(ModelCacheError):
        verify_registered_model(
            manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_divergent_cache_identity_is_integrity_failure(harness: Harness, tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)
    manifest = _registered_manifest(model_path)
    models._prepare_home(harness.home)
    identity = models._identity_of(os.lstat(model_path))
    key = models._cache_key(model_path, identity)
    wrong = models._ModelHashCacheRecordV1(
        identity=identity.model_copy(update={"size_bytes": identity.size_bytes + 5}),
        size_bytes=identity.size_bytes + 5,
        sha256="0" * 64,
    )
    from strixlab.serialization import canonical_json_bytes

    (harness.home.joinpath(*models._CACHE_DIR) / f"{key}.json").write_bytes(
        canonical_json_bytes(wrong.model_dump(mode="json"))
    )
    with pytest.raises(ModelCacheError):
        verify_registered_model(
            manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_cache_identity_miss_rehashes(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = _model_file(tmp_path)
    manifest = _registered_manifest(model_path)
    calls = _model_hash_spy(monkeypatch, model_path)
    verify_registered_model(
        manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
    )
    # Bump mtime so the stable identity (and therefore the cache key) changes.
    os.utime(model_path, ns=(0, 0))
    manifest = _registered_manifest(model_path)  # sha unchanged, identity changed
    verify_registered_model(
        manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
    )
    assert len(calls) == 2
    cache_files = list(harness.home.joinpath(*models._CACHE_DIR).glob("*.json"))
    assert len(cache_files) == 2


def test_cache_lock_released_after_failure(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = _model_file(tmp_path)
    manifest = _registered_manifest(model_path)
    original = models._hash_descriptor
    state = {"fail": True}

    def flaky(descriptor: int, identity: Any, *, path: Path, error: Any) -> str:
        if path == model_path and state["fail"]:
            state["fail"] = False
            raise ModelArtifactError("injected hash failure")
        return original(descriptor, identity, path=path, error=error)

    monkeypatch.setattr(models, "_hash_descriptor", flaky)
    with pytest.raises(ModelArtifactError):
        verify_registered_model(
            manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )
    # The lock was released on failure, so a retry can acquire it and succeed.
    receipt = verify_registered_model(
        manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
    )
    assert receipt.trust_state == "verified"


def test_cache_record_is_read_only(harness: Harness, tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)
    manifest = _registered_manifest(model_path)
    verify_registered_model(
        manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
    )
    cache_files = list(harness.home.joinpath(*models._CACHE_DIR).glob("*.json"))
    assert cache_files
    assert stat.S_IMODE(cache_files[0].stat().st_mode) == 0o400


# --- Artifact safety ----------------------------------------------------------


def test_symlinked_model_is_rejected(harness: Harness, tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)
    link = tmp_path / "link.gguf"
    link.symlink_to(model_path)
    with pytest.raises(ModelArtifactError):
        inspect_local_gguf(
            link, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_special_file_is_rejected(harness: Harness, tmp_path: Path) -> None:
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ModelArtifactError):
        inspect_local_gguf(
            fifo, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_hash_descriptor_detects_mutation(tmp_path: Path) -> None:
    target = _model_file(tmp_path)
    descriptor, identity = models._open_stable_descriptor(target, error=ModelArtifactError)
    try:
        os.utime(target, ns=(1, 1))  # change mtime under the open descriptor
        with pytest.raises(ModelArtifactError):
            models._hash_descriptor(descriptor, identity, path=target, error=ModelArtifactError)
    finally:
        os.close(descriptor)


def test_inspection_drift_fails(
    harness: Harness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = _model_file(tmp_path)
    manifest = _registered_manifest(model_path)
    original = models._run_inspector

    def drifting(descriptor: int, path: Path, bound: Any, home: Path, *, timeout: float) -> Any:
        result = original(descriptor, path, bound, home, timeout=timeout)
        os.utime(path, ns=(2, 2))
        return result

    monkeypatch.setattr(models, "_run_inspector", drifting)
    with pytest.raises(ModelArtifactError):
        verify_registered_model(
            manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


# --- Sidecars -----------------------------------------------------------------


def _sidecar_file(root: Path, name: str, content: bytes) -> ModelFileIdentityV1:
    path = root / name
    path.write_bytes(content)
    return ModelFileIdentityV1(
        local_path=str(path), size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest()
    )


def test_gguf_sidecar_is_verified(harness: Harness, tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)
    sidecar_path = tmp_path / "mmproj.gguf"
    sidecar_path.write_bytes(GGUF_FIXTURE.read_bytes())
    sidecar = ModelSidecarV1(
        id="mmproj",
        kind="mmproj",
        format="gguf",
        inspection="gguf",
        file=ModelFileIdentityV1(
            local_path=str(sidecar_path),
            size_bytes=sidecar_path.stat().st_size,
            sha256=_sha256_file(sidecar_path),
        ),
    )
    manifest = _registered_manifest(model_path).model_copy(update={"sidecars": [sidecar]})
    receipt = verify_registered_model(
        manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
    )
    assert receipt.compatibility == "verified"
    assert len(receipt.sidecars) == 1
    assert receipt.sidecars[0].inspection == "gguf"
    assert receipt.sidecars[0].metadata is not None


def test_hash_only_sidecar_forces_asserted_and_unpublishable(
    harness: Harness, tmp_path: Path
) -> None:
    model_path = _model_file(tmp_path)
    imatrix = _sidecar_file(tmp_path, "calib.imatrix", b"imatrix-blob-bytes")
    sidecar = ModelSidecarV1(
        id="imatrix", kind="imatrix", format="opaque", inspection="hash-only", file=imatrix
    )
    manifest = _registered_manifest(model_path).model_copy(update={"sidecars": [sidecar]})
    receipt = verify_registered_model(
        manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
    )
    assert receipt.compatibility == "asserted"
    assert receipt.publishable is False
    assert receipt.sidecars[0].compatibility == "asserted"


def test_missing_sidecar_fails(harness: Harness, tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)
    sidecar = ModelSidecarV1(
        id="imatrix",
        kind="imatrix",
        format="opaque",
        inspection="hash-only",
        file=ModelFileIdentityV1(
            local_path=str(tmp_path / "absent.imatrix"), size_bytes=4, sha256="0" * 64
        ),
    )
    manifest = _registered_manifest(model_path).model_copy(update={"sidecars": [sidecar]})
    with pytest.raises(ModelSidecarError):
        verify_registered_model(
            manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_mutated_sidecar_fails(harness: Harness, tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)
    imatrix = _sidecar_file(tmp_path, "calib.imatrix", b"imatrix-blob-bytes")
    bad = imatrix.model_copy(update={"sha256": "0" * 64})
    sidecar = ModelSidecarV1(
        id="imatrix", kind="imatrix", format="opaque", inspection="hash-only", file=bad
    )
    manifest = _registered_manifest(model_path).model_copy(update={"sidecars": [sidecar]})
    with pytest.raises(ModelSidecarError):
        verify_registered_model(
            manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_publishable_when_fully_provenanced(harness: Harness, tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)
    manifest = _registered_manifest(model_path)
    quant = ModelQuantizationV1(
        format_family="Q4_K",
        storage_format="gguf",
        measured_bits_per_weight=4.5,
        tensor_policy_id="rocm-i4-mix",
        tensor_policy_source="halo-box/policies@1",
        calibration_method="imatrix",
        calibration_source="halo-box/calib@1",
        calibration_hash="a" * 64,
    )
    manifest = manifest.model_copy(update={"quantization": quant})
    receipt = verify_registered_model(
        manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
    )
    assert receipt.publishable is True


# --- Leases, current-model checks, and receipt loading ------------------------


def _receipt(harness: Harness, tmp_path: Path) -> Any:
    model_path = _model_file(tmp_path)
    manifest = _registered_manifest(model_path)
    return (
        verify_registered_model(
            manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        ),
        model_path,
    )


def test_lease_hands_the_bound_descriptor_to_a_child(harness: Harness, tmp_path: Path) -> None:
    import sys

    from strixlab.process import run_process

    snippet = "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())"
    receipt, model_path = _receipt(harness, tmp_path)
    with lease_verified_model(receipt) as lease:
        result = run_process(
            [sys.executable, "-c", snippet, lease.descriptor_path],
            cwd=tmp_path,
            timeout=30.0,
            pass_fds=(lease.descriptor,),
        )
    assert result.returncode == 0
    assert result.stdout.strip() == _sha256_file(model_path)


def test_lease_verify_detects_drift(harness: Harness, tmp_path: Path) -> None:
    receipt, model_path = _receipt(harness, tmp_path)
    with pytest.raises(ModelArtifactError), lease_verified_model(receipt) as lease:
        os.utime(model_path, ns=(3, 3))
        lease.verify()


def test_require_current_model_detects_drift(harness: Harness, tmp_path: Path) -> None:
    receipt, model_path = _receipt(harness, tmp_path)
    assert require_current_model(receipt) == receipt.primary.identity
    os.utime(model_path, ns=(4, 4))
    with pytest.raises(ModelArtifactError):
        require_current_model(receipt)


def test_load_receipt_rejects_bad_identifiers(harness: Harness, tmp_path: Path) -> None:
    receipt, _ = _receipt(harness, tmp_path)
    _, digest = models._receipt_envelope(receipt)
    with pytest.raises(ModelManifestError):
        load_model_receipt("Bad_Id", digest, home=harness.home)
    with pytest.raises(models.ModelError):
        load_model_receipt(receipt.manifest_id, "nothex", home=harness.home)
    with pytest.raises(models.ModelError):
        load_model_receipt(receipt.manifest_id, "0" * 64, home=harness.home)


def test_load_receipt_rejects_tampered_content(harness: Harness, tmp_path: Path) -> None:
    receipt, _ = _receipt(harness, tmp_path)
    _, digest = models._receipt_envelope(receipt)
    path = harness.home.joinpath(
        *models._RECEIPT_REGISTRY_DIR, receipt.manifest_id, f"{digest}.json"
    )
    path.chmod(0o600)
    path.write_bytes(path.read_bytes().replace(b"verified", b"asserted"))
    with pytest.raises(models.ModelError):
        load_model_receipt(receipt.manifest_id, digest, home=harness.home)


def test_require_receipt_inputs_match_detects_mismatch(harness: Harness, tmp_path: Path) -> None:
    receipt, _ = _receipt(harness, tmp_path)
    good = dict(
        model_id=receipt.manifest_id,
        model_path=receipt.primary.local_path,
        model_sha256=receipt.primary.sha256,
        model_receipt_sha256=receipt_evidence_digest(receipt.evidence),
        model_receipt_evidence=receipt.evidence,
    )
    models.require_receipt_inputs_match(receipt, **good)  # baseline holds
    for key, value in (
        ("model_id", "other-model"),
        ("model_path", "/models/other.gguf"),
        ("model_sha256", "0" * 64),
        ("model_receipt_sha256", "0" * 64),
    ):
        with pytest.raises(models.ModelError):
            models.require_receipt_inputs_match(receipt, **{**good, key: value})


def test_load_receipt_rejects_wrong_domain(harness: Harness, tmp_path: Path) -> None:
    receipt, _ = _receipt(harness, tmp_path)
    _, digest = models._receipt_envelope(receipt)
    path = harness.home.joinpath(
        *models._RECEIPT_REGISTRY_DIR, receipt.manifest_id, f"{digest}.json"
    )
    envelope = json.loads(path.read_bytes())
    envelope["domain"] = "strixlab.wrong.v1"
    path.chmod(0o600)
    path.write_bytes(json.dumps(envelope).encode())
    with pytest.raises(models.ModelError):
        load_model_receipt(receipt.manifest_id, digest, home=harness.home)


def test_lease_verify_compares_full_descriptor_identity(harness: Harness, tmp_path: Path) -> None:
    # A held descriptor whose full identity (here mtime_ns) no longer matches the lease
    # is rejected even though dev/ino/size still agree; require_current_model passes
    # because the receipt's own identity still matches the live path.
    receipt, model_path = _receipt(harness, tmp_path)
    fd, identity = models._open_stable_descriptor(model_path, error=ModelArtifactError)
    try:
        tampered = identity.model_copy(update={"mtime_ns": identity.mtime_ns + 1})
        lease = models.ModelLease(
            receipt=receipt,
            descriptor=fd,
            descriptor_path=f"/proc/self/fd/{fd}",
            identity=tampered,
        )
        with pytest.raises(ModelArtifactError):
            lease.verify()
    finally:
        os.close(fd)


def test_inspect_rejects_relative_path_without_side_effects(
    harness: Harness, tmp_path: Path
) -> None:
    with pytest.raises(ModelArtifactError):
        inspect_local_gguf(
            Path("relative.gguf"),
            inspector=harness.inspector,
            source_lease=harness.lease,
            home=harness.home,
        )
    # The typed rejection happened before any home preparation or scratch creation.
    assert not (harness.home / "models").exists()
    assert not (harness.home / "cache").exists()


def test_inspector_receives_constructed_environment(harness: Harness, tmp_path: Path) -> None:
    # The stub fails closed unless LANG/LC_ALL/TZ and a private HOME==TMPDIR are delivered,
    # so a successful observation proves the constructed env reached the child.
    body = (
        "import os, sys\n"
        "assert os.environ.get('LANG') == 'C', 'LANG'\n"
        "assert os.environ.get('LC_ALL') == 'C', 'LC_ALL'\n"
        "assert os.environ.get('TZ') == 'UTC', 'TZ'\n"
        "home = os.environ.get('HOME')\n"
        "tmp = os.environ.get('TMPDIR')\n"
        "assert home and tmp and home == tmp, 'scratch home/tmpdir'\n"
        f"sys.stdout.buffer.write({DUMP_FIXTURE.read_bytes()!r})\n"
    )
    _install_raw_stub(harness, tmp_path, body)
    model_path = _model_file(tmp_path)
    observation = inspect_local_gguf(
        model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
    )
    assert observation.trust_state == "unregistered"


def test_binding_rejects_wrong_source_id(harness: Harness, tmp_path: Path) -> None:
    harness.lease.evidence.source_id = "not-strix-llama"
    model_path = _model_file(tmp_path)
    with pytest.raises(ModelInspectorError):
        inspect_local_gguf(
            model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_binding_rejects_wrong_gguf_py_root(harness: Harness, tmp_path: Path) -> None:
    harness.inspector = harness.inspector.model_copy(update={"gguf_py_relative_root": "gguf-py-x"})
    model_path = _model_file(tmp_path)
    with pytest.raises(ModelInspectorError):
        inspect_local_gguf(
            model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_binding_rejects_wrong_script_relative_path(harness: Harness, tmp_path: Path) -> None:
    harness.inspector = harness.inspector.model_copy(
        update={"script_relative_path": "gguf-py/gguf/scripts/other_dump.py"}
    )
    model_path = _model_file(tmp_path)
    with pytest.raises(ModelInspectorError):
        inspect_local_gguf(
            model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_receipt_retains_exact_inspector_stdout_digest(harness: Harness, tmp_path: Path) -> None:
    model_path = _model_file(tmp_path)
    manifest = _registered_manifest(model_path)
    receipt = verify_registered_model(
        manifest, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
    )
    raw = DUMP_FIXTURE.read_bytes()
    raw_sha = hashlib.sha256(raw).hexdigest()
    # The exact raw inspector stdout digest is retained, distinct from the normalized
    # projection digest, and propagated into the portable evidence projection.
    assert receipt.metadata.inspector_stdout_sha256 == raw_sha
    assert receipt.metadata.inspector_stdout_bytes == len(raw)
    assert receipt.metadata.inspector_stdout_sha256 != receipt.metadata.metadata_sha256
    assert receipt.evidence.inspector_stdout_sha256 == raw_sha

    # It round-trips through the local receipt registry.
    _, digest = models._receipt_envelope(receipt)
    loaded = load_model_receipt(manifest.id, digest, home=harness.home)
    assert loaded.metadata.inspector_stdout_sha256 == raw_sha


def _dump_with_tensor(shape: object) -> bytes:
    return _dump_with(
        tensors={"a.weight": {"index": 0, "shape": shape, "type": "F32", "offset": 0}}
    )


@pytest.mark.parametrize("shape", [[], [True], [0], [-1], [1.5], [4, 0], "notalist"])
def test_invalid_tensor_shape_is_metadata_error(
    harness: Harness, tmp_path: Path, shape: object
) -> None:
    _install_stub(harness, tmp_path, _dump_with_tensor(shape))
    model_path = _model_file(tmp_path)
    with pytest.raises(ModelMetadataError):
        inspect_local_gguf(
            model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
        )


def test_valid_multidimensional_tensor_shape_is_accepted(harness: Harness, tmp_path: Path) -> None:
    _install_stub(harness, tmp_path, _dump_with_tensor([3, 2, 4, 1]))
    model_path = _model_file(tmp_path)
    observation = inspect_local_gguf(
        model_path, inspector=harness.inspector, source_lease=harness.lease, home=harness.home
    )
    assert observation.metadata.tensor_count == 1

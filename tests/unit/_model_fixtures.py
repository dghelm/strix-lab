"""Shared helper: build a real MODEL-001 receipt via a synthetic inspector.

The synthetic interpreter replays the pinned ca94157 inspector JSON fixture, so any
local file byte-verifies against a manifest built from its own size/SHA. Adapter and
model tests use this to exercise verified-model wiring without a real GGUF or gguf-py.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
from collections.abc import Iterator
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
)

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "models" / "gguf-dump-ca94157"
_DUMP = (_FIXTURE_DIR / "gguf_dump.json").read_bytes()
_ANCHOR = models.SOURCE_ANCHOR_COMMIT
_counter = itertools.count()


@dataclass
class FakeEvidence:
    base_commit: str = _ANCHOR
    preparation_id: str = "prep-strix-llama-" + "a" * 24
    candidate_id: str = "candidate-sha256:" + "1" * 64
    content_tree_id: str = "content-tree-sha256:" + "2" * 64
    source_id: str = "strix-llama"
    patches: tuple[object, ...] = ()
    status: tuple[str, ...] = ()


@dataclass
class FakeLease:
    worktree: Path
    evidence: FakeEvidence = field(default_factory=FakeEvidence)

    def verify(self) -> None:
        return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_lease_source(
    monkeypatch: pytest.MonkeyPatch, lease: Any, *, expected_preparation_id: str
) -> None:
    """Redirect ``verify_model_at_source``'s lazy ``lease_source`` to yield ``lease``.

    The fake context manager asserts the caller leases exactly ``expected_preparation_id``
    before yielding, so tests exercise the real orchestration without a Git worktree.
    """

    @contextlib.contextmanager
    def fake_lease_source(preparation_id: str, *, home: Path) -> Iterator[Any]:
        assert preparation_id == expected_preparation_id
        yield lease

    monkeypatch.setattr("strixlab.sources.lease_source", fake_lease_source)


def build_verified_receipt(
    root: Path, model_path: Path, *, model_id: str = "qwen35-4b-smoke"
) -> models.ModelReceiptV1:
    scratch = root / f"model-registry-{next(_counter)}"
    home = scratch / "home"
    home.mkdir(parents=True)
    worktree = scratch / "worktree"
    scripts = worktree / "gguf-py" / "gguf" / "scripts"
    scripts.mkdir(parents=True)
    script = scripts / "gguf_dump.py"
    script.write_text("# pinned inspector script placeholder\n", encoding="utf-8")
    python = scratch / "python-stub"
    python.write_text(
        f"#!/usr/bin/python3\nimport sys\nsys.stdout.buffer.write({_DUMP!r})\n",
        encoding="utf-8",
    )
    python.chmod(0o755)
    inspector = models.GgufInspectorBindingV1(
        preparation_id="prep-strix-llama-" + "a" * 24,
        candidate_id="candidate-sha256:" + "1" * 64,
        content_tree_id="content-tree-sha256:" + "2" * 64,
        base_commit=_ANCHOR,
        python_executable=str(python),
        python_sha256=_sha256(python),
        gguf_py_relative_root="gguf-py",
        script_relative_path="gguf-py/gguf/scripts/gguf_dump.py",
        script_sha256=_sha256(script),
    )
    manifest = ModelManifestV1(
        schema_version=1,
        id=model_id,
        registry_status="registered",
        base_model=ModelBaseV1(
            repository="Qwen/Qwen3.5-4B",
            revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
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
            file=ModelFileIdentityV1(
                repository="bartowski/Qwen_Qwen3.5-4B-GGUF",
                revision="4168f45a16a1290d65a4ec0fa312ae917a4c15d6",
                filename=model_path.name,
                local_path=str(model_path),
                size_bytes=model_path.stat().st_size,
                sha256=_sha256(model_path),
            ),
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
    return models.verify_registered_model(
        manifest, inspector=inspector, source_lease=FakeLease(worktree=worktree), home=home
    )

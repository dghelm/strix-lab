"""Shared construction of present, attested build-cache states for tests.

Follows ``_model_fixtures`` in style: small, explicit helpers that both the build-cache
state-machine tests and the suite tests reuse, without generalized fixture machinery.
The heavy verification entry points are stubbed (the artifact-hash comparison, producer
provenance, and attestor authentication are covered end-to-end in ``test_cmake_build``),
so a present/attested build can be materialized with no real attempt records or binaries.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import strixlab.build_cache as cache_module
from strixlab.build_cache import (
    CanonicalBuildRecordV1,
    build_cache_session,
    write_build_root_owner,
)


def stub_cache_verification(monkeypatch: Any) -> None:
    """Stub the heavy build-cache verification entry points used by cache-state tests."""

    monkeypatch.setattr(cache_module, "verify_artifact_capture", lambda *a, **k: None)
    monkeypatch.setattr(cache_module, "_verify_canonical_producer", lambda *a, **k: None)
    monkeypatch.setattr(cache_module, "_authenticate_attestor", lambda *a, **k: None)


def write_normal_attestation(
    home: Path, *, build_id: str, attempt: str, canonical_digest: str, artifact_set_id: str
) -> None:
    """Publish a schema-valid ``built`` attestation for a normal present build."""

    layout = cache_module._layout(home, create=False)
    attestation = cache_module.BuildAttestationV1(
        build_id=build_id,
        canonical_record_sha256=canonical_digest,
        attestor_attempt_id=attempt,
        execution_class="built",
        artifact_set_id=artifact_set_id,
        producer_record_sha256="record-sha256:" + "11" * 32,
        attestor_record_sha256="record-sha256:" + "11" * 32,
    )
    cache_module._publish_attestation(layout, attestation)


def publish_present_build(
    home: Path, *, build_id: str, attempt: str, record: CanonicalBuildRecordV1
) -> str:
    """Materialize, publish, and attest one PRESENT build; return its canonical digest."""

    with build_cache_session(build_id, attempt, home=home) as session:
        session.begin_materialization(rehydrate=False)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        owner = write_build_root_owner(root, attempt, build_id)
        session.bind_root(owner)
        digest = session.publish(record, rehydrate=False)
    write_normal_attestation(
        home,
        build_id=build_id,
        attempt=attempt,
        canonical_digest=digest,
        artifact_set_id=record.artifacts.artifact_set_id,
    )
    return digest

from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from strixlab.build_records import (
    BuildRecordError,
    publish_record,
    record_source_digest,
    verify_record,
)


def _source(tmp_path: Path, content: bytes = b"evidence\n") -> Path:
    source = tmp_path / "source"
    (source / "logs").mkdir(parents=True, mode=0o700)
    (source / "logs" / "configure.stdout").write_bytes(content)
    (source / "registry.json").write_text('{"state":"failed"}\n', encoding="utf-8")
    return source


def test_publish_record_copies_payloads_and_verifies_manifest(tmp_path: Path) -> None:
    source = _source(tmp_path)
    records = tmp_path / "records"
    records.mkdir(mode=0o700)
    destination = records / "attempt-1"

    source_digest = record_source_digest(source)
    published = publish_record(source, destination)
    verified = verify_record(destination)
    manifest = json.loads((destination / "record-manifest.json").read_bytes())

    assert published == verified
    assert published.record_sha256 == source_digest
    assert published.record_sha256.startswith("record-sha256:")
    assert {entry["path"] for entry in manifest["files"]} == {
        "logs/configure.stdout",
        "registry.json",
    }
    assert "record-manifest.json" not in {entry["path"] for entry in manifest["files"]}
    assert (source / "logs" / "configure.stdout").stat().st_ino != (
        destination / "logs" / "configure.stdout"
    ).stat().st_ino


def test_verify_record_rejects_payload_tampering(tmp_path: Path) -> None:
    source = _source(tmp_path)
    records = tmp_path / "records"
    records.mkdir(mode=0o700)
    destination = records / "attempt-1"
    publish_record(source, destination)
    (destination / "logs" / "configure.stdout").write_bytes(b"changed\n")

    with pytest.raises(BuildRecordError, match="integrity mismatch"):
        verify_record(destination)


def test_publish_record_rejects_symlinks(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / "escape").symlink_to("/etc/passwd")
    records = tmp_path / "records"
    records.mkdir(mode=0o700)

    with pytest.raises(BuildRecordError, match="record input"):
        publish_record(source, records / "attempt-1")

    assert not (records / "attempt-1").exists()


def test_publish_record_never_replaces_an_existing_record(tmp_path: Path) -> None:
    source = _source(tmp_path)
    records = tmp_path / "records"
    records.mkdir(mode=0o700)
    destination = records / "attempt-1"
    original = publish_record(source, destination)
    (source / "logs" / "configure.stdout").write_bytes(b"different\n")

    with pytest.raises(OSError) as raised:
        publish_record(source, destination)

    assert raised.value.errno == errno.EEXIST
    assert verify_record(destination) == original
    assert not tuple(records.glob(".*.tmp"))


def test_record_verification_rejects_unlisted_payloads(tmp_path: Path) -> None:
    source = _source(tmp_path)
    records = tmp_path / "records"
    records.mkdir(mode=0o700)
    destination = records / "attempt-1"
    publish_record(source, destination)
    descriptor = os.open(destination / "extra", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)

    with pytest.raises(BuildRecordError, match="payload set"):
        verify_record(destination)


def test_publish_record_rejects_a_caller_supplied_manifest(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / "record-manifest.json").write_text("{}\n", encoding="utf-8")
    records = tmp_path / "records"
    records.mkdir(mode=0o700)

    with pytest.raises(BuildRecordError, match="cannot supply"):
        publish_record(source, records / "attempt-1")


def test_verify_record_rejects_invalid_and_duplicate_manifest_entries(tmp_path: Path) -> None:
    source = _source(tmp_path)
    records = tmp_path / "records"
    records.mkdir(mode=0o700)
    invalid = records / "invalid"
    publish_record(source, invalid)
    manifest_path = invalid / "record-manifest.json"
    manifest_path.write_bytes(b"{}\n")
    with pytest.raises(BuildRecordError, match="manifest is invalid"):
        verify_record(invalid)

    duplicate = records / "duplicate"
    publish_record(source, duplicate)
    manifest_path = duplicate / "record-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"].append(manifest["files"][0])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BuildRecordError, match="duplicate paths"):
        verify_record(duplicate)

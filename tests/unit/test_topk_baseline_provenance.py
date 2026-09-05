"""Verify every pinned baseline file against both recorded identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "native/topk/baseline"


EXPECTED_PATHS = {"top-k.cu", "top-k.cuh", "argsort.cu", "argsort.cuh", "common.cuh", "LICENSE"}


def git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


def test_pinned_baseline_files_match_provenance() -> None:
    record = json.loads((BASELINE / "provenance.json").read_bytes())
    assert record["source_commit"] == "ca94157f70a2776e8da6b6849b50b45a083d0478"
    assert {item["path"] for item in record["files"]} == EXPECTED_PATHS
    assert len(record["files"]) == len(EXPECTED_PATHS)
    for item in record["files"]:
        path = BASELINE / "upstream" / item["path"]
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == item["sha256"]
        assert git_blob_sha1(path) == item["blob"]

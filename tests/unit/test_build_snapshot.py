from __future__ import annotations

import os
import shutil
import stat
from contextlib import contextmanager
from pathlib import Path

import pytest

import strixlab.build_snapshot as snapshot_module
from strixlab.build_snapshot import SnapshotError, materialize_snapshot, verify_snapshot
from strixlab.locks import LockAttempt, LockStatus

_CANDIDATE = "candidate-sha256:" + "12" * 32
_CONTENT = "content-tree-sha256:" + "34" * 32


def _published(snapshots: Path) -> list[Path]:
    return [entry for entry in snapshots.iterdir() if entry.name != ".locks"]


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "src").mkdir(parents=True)
    (source / "src" / "main.cc").write_text("int main() { return 0; }\n", encoding="utf-8")
    (source / "CMakeLists.txt").write_text("project(test)\n", encoding="utf-8")
    (source / "main-link.cc").symlink_to("src/main.cc")
    (source / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    return source


def test_snapshot_is_content_addressed_read_only_and_reusable(tmp_path: Path) -> None:
    source = _source(tmp_path)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()

    first = materialize_snapshot(
        source, snapshots, candidate_id=_CANDIDATE, content_tree_id=_CONTENT
    )
    second = materialize_snapshot(
        source, snapshots, candidate_id=_CANDIDATE, content_tree_id=_CONTENT
    )

    assert first == second
    assert first.snapshot_id.startswith("snapshot-sha256:")
    assert not (first.source / ".git").exists()
    assert os.readlink(first.source / "main-link.cc") == "src/main.cc"
    assert stat.S_IMODE((first.source / "src" / "main.cc").stat().st_mode) == 0o444
    assert stat.S_IMODE(first.source.stat().st_mode) == 0o500
    assert verify_snapshot(first.root) == first


def test_snapshot_rejects_escaping_links_and_detects_tampering(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / "escape").symlink_to("../outside")
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()

    with pytest.raises(SnapshotError, match="escapes"):
        materialize_snapshot(source, snapshots, candidate_id=_CANDIDATE, content_tree_id=_CONTENT)

    (source / "escape").unlink()
    snapshot = materialize_snapshot(
        source, snapshots, candidate_id=_CANDIDATE, content_tree_id=_CONTENT
    )
    payload = snapshot.source / "src" / "main.cc"
    payload.chmod(0o600)
    payload.write_text("changed\n", encoding="utf-8")
    with pytest.raises(SnapshotError, match="does not match"):
        verify_snapshot(snapshot.root)


def test_snapshot_refuses_a_source_that_changes_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    original = snapshot_module._copy_entry
    mutated = False

    def mutate_after_copy(input_path, output_path, entry):
        nonlocal mutated
        original(input_path, output_path, entry)
        if not mutated and entry.kind == "file":
            mutated = True
            (source / "src" / "main.cc").write_text("mutated\n", encoding="utf-8")

    monkeypatch.setattr(snapshot_module, "_copy_entry", mutate_after_copy)
    with pytest.raises(SnapshotError, match="changed while snapshotting"):
        materialize_snapshot(source, snapshots, candidate_id=_CANDIDATE, content_tree_id=_CONTENT)
    assert not _published(snapshots)


def test_snapshot_publication_race_reuses_the_complete_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()

    def publish_other(source_path: Path, destination: Path) -> None:
        source_path.rename(destination)
        raise FileExistsError(destination)

    monkeypatch.setattr(snapshot_module, "rename_noreplace", publish_other)
    snapshot = materialize_snapshot(
        source, snapshots, candidate_id=_CANDIDATE, content_tree_id=_CONTENT
    )

    assert verify_snapshot(snapshot.root) == snapshot
    assert len(_published(snapshots)) == 1


def test_snapshot_repairs_an_invalid_existing_destination_while_locked(tmp_path: Path) -> None:
    source = _source(tmp_path)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()

    snapshot = materialize_snapshot(
        source, snapshots, candidate_id=_CANDIDATE, content_tree_id=_CONTENT
    )

    payload = snapshot.source / "src" / "main.cc"
    payload.parent.chmod(0o700)
    payload.chmod(0o600)
    payload.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(SnapshotError, match="does not match"):
        verify_snapshot(snapshot.root)

    repaired = materialize_snapshot(
        source, snapshots, candidate_id=_CANDIDATE, content_tree_id=_CONTENT
    )

    assert repaired == snapshot
    assert verify_snapshot(repaired.root) == repaired
    assert len(_published(snapshots)) == 1


def test_snapshot_repairs_a_destination_swapped_for_a_symlink(tmp_path: Path) -> None:
    source = _source(tmp_path)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()

    snapshot = materialize_snapshot(
        source, snapshots, candidate_id=_CANDIDATE, content_tree_id=_CONTENT
    )

    for directory, _names, _files in os.walk(snapshot.root):
        os.chmod(directory, 0o700)
    shutil.rmtree(snapshot.root)
    evil = tmp_path / "evil"
    evil.mkdir()
    snapshot.root.symlink_to(evil, target_is_directory=True)

    repaired = materialize_snapshot(
        source, snapshots, candidate_id=_CANDIDATE, content_tree_id=_CONTENT
    )

    assert repaired == snapshot
    assert not repaired.root.is_symlink()
    assert verify_snapshot(repaired.root) == repaired
    assert len(_published(snapshots)) == 1


def test_snapshot_removes_a_destination_that_fails_final_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()

    def corrupt_after_publish(stage: Path, destination: Path) -> None:
        stage.rename(destination)
        payload = destination / "source" / "src" / "main.cc"
        payload.parent.chmod(0o700)
        payload.chmod(0o600)
        payload.write_text("corrupt\n", encoding="utf-8")

    monkeypatch.setattr(snapshot_module, "rename_noreplace", corrupt_after_publish)
    with pytest.raises(SnapshotError, match="does not match"):
        materialize_snapshot(source, snapshots, candidate_id=_CANDIDATE, content_tree_id=_CONTENT)

    assert not _published(snapshots)


def test_snapshot_retries_a_contended_publication_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source(tmp_path)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    original = snapshot_module.exclusive_lock
    attempts = 0

    @contextmanager
    def contend_once(path: Path):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            yield LockAttempt(LockStatus.CONTENDED, path, "held by another publisher")
        else:
            with original(path) as lock:
                yield lock

    monkeypatch.setattr(snapshot_module, "exclusive_lock", contend_once)
    monkeypatch.setattr(snapshot_module.time, "sleep", lambda _seconds: None)

    snapshot = materialize_snapshot(
        source, snapshots, candidate_id=_CANDIDATE, content_tree_id=_CONTENT
    )

    assert attempts == 2
    assert verify_snapshot(snapshot.root) == snapshot

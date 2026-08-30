from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

import strixlab.build_cache as cache_module
import strixlab.secure_fs as secure_fs
from strixlab.build_artifacts import ArtifactV1, BuildArtifactsV1
from strixlab.build_cache import (
    BuildCacheError,
    BuildIdentityProjectionV1,
    BuildRootOwnerV1,
    CacheClassification,
    CanonicalBuildRecordV1,
    IdentityEntryV1,
    MaterializationState,
    SourceReproducerV1,
    build_cache_session,
    cleanup_build,
    identity_models,
    inspect_build,
    remove_owned_build_root,
    tool_models,
    verify_build_root_owner,
    write_build_root_owner,
)
from strixlab.build_identity import IdentityEntry, ToolObservation

_BUILD = "build-sha256:" + "aa" * 32
_ATTEMPT_A = "attempt-" + "0" * 24 + "-" + "a" * 32
_ATTEMPT_B = "attempt-" + "0" * 24 + "-" + "b" * 32
_HEX = "cd" * 32


@pytest.fixture(autouse=True)
def _stub_artifact_verification(monkeypatch: pytest.MonkeyPatch) -> None:
    # These tests exercise the journal/record/recovery state machine, not the
    # artifact-hash comparison or producer provenance (both covered end-to-end in
    # test_cmake_build with real build attempts).
    monkeypatch.setattr(cache_module, "verify_artifact_capture", lambda *a, **k: None)
    # Stub the record-verification and attestor authentication entry points; these
    # tests never publish real attempt records, so the producer record and the
    # attestor result/digest bindings are covered end-to-end in test_cmake_build.
    # _load_attestation stays real so the dangling/missing-attestation paths remain
    # exercised; _make_present writes a schema-valid attestation for a normal build.
    monkeypatch.setattr(cache_module, "_verify_canonical_producer", lambda *a, **k: None)
    monkeypatch.setattr(cache_module, "_authenticate_attestor", lambda *a, **k: None)


def _write_attestation(home: Path, digest: str) -> None:
    layout = cache_module._layout(home, create=False)
    attestation = cache_module.BuildAttestationV1(
        build_id=_BUILD,
        canonical_record_sha256=digest,
        attestor_attempt_id=_ATTEMPT_A,
        execution_class="built",
        artifact_set_id=_artifacts().artifact_set_id,
        producer_record_sha256="record-sha256:" + "11" * 32,
        attestor_record_sha256="record-sha256:" + "11" * 32,
    )
    cache_module._publish_attestation(layout, attestation)


def _artifacts() -> BuildArtifactsV1:
    return BuildArtifactsV1(
        artifact_set_id="artifact-set-sha256:" + _HEX,
        targets=(),
        artifacts=(
            ArtifactV1(
                path="bin/llama-bench",
                kind="elf",
                mode=0o755,
                size_bytes=4,
                sha256=_HEX,
                targets=("llama-bench",),
            ),
        ),
        inspections=(),
        capture_tools=(),
        cmake_cache_sha256=_HEX,
    )


def _source() -> SourceReproducerV1:
    return SourceReproducerV1(
        candidate_id="candidate-sha256:" + _HEX,
        content_tree_id="content-tree-sha256:" + _HEX,
        snapshot_id="snapshot-sha256:" + _HEX,
        source_evidence={"preparation_id": "prep-test"},
        source_evidence_sha256=_HEX,
        snapshot_manifest={"schema_version": 1},
        diff=None,
        patches=(),
    )


def _identity(
    *,
    requested_targets: tuple[str, ...] = ("llama-bench",),
    selections: tuple[IdentityEntryV1, ...] = (IdentityEntryV1(name="generator", value="Ninja"),),
) -> BuildIdentityProjectionV1:
    return BuildIdentityProjectionV1(
        recipe_id="recipe-sha256:" + _HEX,
        profile_sha256=_HEX,
        toolchain_mode="rocm",
        environment=(IdentityEntryV1(name="TZ", value="UTC"),),
        requested_targets=requested_targets,
        selections=selections,
        tools=(),
        source=_source(),
    )


def _record(
    *,
    build_id: str = _BUILD,
    producer: str = _ATTEMPT_A,
    artifacts: BuildArtifactsV1 | None = None,
    requested_targets: tuple[str, ...] = ("llama-bench",),
    selections: tuple[IdentityEntryV1, ...] = (IdentityEntryV1(name="generator", value="Ninja"),),
) -> CanonicalBuildRecordV1:
    identity = _identity(requested_targets=requested_targets, selections=selections)
    return CanonicalBuildRecordV1(
        build_id=build_id,
        producer_attempt_id=producer,
        recipe_id=identity.recipe_id,
        profile_sha256=identity.profile_sha256,
        toolchain_mode=identity.toolchain_mode,
        environment=identity.environment,
        requested_targets=identity.requested_targets,
        selections=identity.selections,
        tools=identity.tools,
        source=identity.source,
        artifacts=artifacts or _artifacts(),
    )


def _make_present(home: Path, *, attempt: str = _ATTEMPT_A, rehydrate: bool = False) -> str:
    with build_cache_session(_BUILD, attempt, home=home) as session:
        session.begin_materialization(rehydrate=rehydrate)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        owner = write_build_root_owner(root, attempt, _BUILD)
        session.bind_root(owner)
        digest = session.publish(_record(producer=attempt), rehydrate=rehydrate)
    layout = cache_module._layout(home, create=False)
    if not os.path.lexists(cache_module._attestation_path(layout, _BUILD)):
        _write_attestation(home, digest)  # a normal present build is attested
    return digest


def test_miss_then_present_then_hit(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with build_cache_session(_BUILD, _ATTEMPT_A, home=home) as session:
        assert session.lookup(_identity(), home=home).classification is (CacheClassification.MISS)
    digest = _make_present(home)
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        hit = session.lookup(_identity(), home=home)
    assert hit.classification is CacheClassification.HIT
    assert hit.canonical_record_sha256 == digest


def test_cleanup_then_rehydrate_and_mismatch(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    cleaned = cleanup_build(_BUILD, home=home)
    assert cleaned.state is MaterializationState.CLEANED
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        assert session.lookup(_identity(), home=home).classification is (
            CacheClassification.REHYDRATE
        )
        session.begin_materialization(rehydrate=True)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        owner = write_build_root_owner(root, _ATTEMPT_B, _BUILD)
        session.bind_root(owner)
        with pytest.raises(BuildCacheError, match="diverged"):
            session.publish(
                _record(producer=_ATTEMPT_A, requested_targets=("other",)), rehydrate=True
            )
        session.publish(_record(producer=_ATTEMPT_A), rehydrate=True)
    assert inspect_build(_BUILD, home=home).state is MaterializationState.PRESENT


def test_interrupted_building_is_discarded_to_vacant(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with build_cache_session(_BUILD, _ATTEMPT_A, home=home) as session:
        session.begin_materialization(rehydrate=False)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        owner = write_build_root_owner(root, _ATTEMPT_A, _BUILD)
        session.bind_root(owner)  # never published
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        assert session.lookup(_identity(), home=home).classification is (CacheClassification.MISS)
    assert not (home / "builds" / "materialized" / _BUILD).exists()


def test_vacant_dangling_symlink_root_is_integrity_failure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with build_cache_session(_BUILD, _ATTEMPT_A, home=home) as session:
        session.begin_materialization(rehydrate=False)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        session.bind_root(write_build_root_owner(root, _ATTEMPT_A, _BUILD))  # never published
    # First lookup recovers the interrupted build to VACANT and removes the root.
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        assert session.lookup(_identity(), home=home).classification is CacheClassification.MISS
    root = home / "builds" / "materialized" / _BUILD
    root.symlink_to(tmp_path / "does-not-exist")
    # A dangling symlink at a VACANT root is corrupt state, never a MISS.
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="vacant build cache has retained"),
    ):
        session.lookup(_identity(), home=home)


def test_cleaned_dangling_symlink_root_is_integrity_failure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    assert cleanup_build(_BUILD, home=home).state is MaterializationState.CLEANED
    root = home / "builds" / "materialized" / _BUILD
    assert not os.path.lexists(root)
    root.symlink_to(tmp_path / "does-not-exist")
    # A dangling symlink at a CLEANED root must fail closed, not read as REHYDRATE.
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="cleaned build unexpectedly has a materialized"),
    ):
        session.lookup(_identity(), home=home)


def _record_parent_fsyncs(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Record every containing-parent fsync performed by ensure_directory_fsynced."""

    fsynced: list[Path] = []
    real = secure_fs.fsync_directory
    monkeypatch.setattr(
        secure_fs,
        "fsync_directory",
        lambda path: (fsynced.append(Path(path)), real(path))[1],
    )
    return fsynced


def test_journal_and_staging_directory_creation_fsyncs_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Finding 5: the per-build journal directory and the publication staging
    # directory must fsync their containing parent after exclusive creation so a
    # crash cannot lose the directory entry itself.
    home = tmp_path / "home"
    layout = cache_module._layout(home, create=True)
    fsynced = _record_parent_fsyncs(monkeypatch)
    journal = cache_module._journal(layout, _BUILD)
    events = cache_module._ensure_journal_events(journal)
    assert events.is_dir()
    # The journal's parent and the journal itself were fsynced on creation.
    assert journal.parent in fsynced
    assert journal in fsynced

    fsynced.clear()
    stage = cache_module._stage_path(layout, _BUILD, _ATTEMPT_A)
    assert stage.parent.is_dir()
    assert layout.staging in fsynced


def test_first_use_top_level_creation_fsyncs_builds_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Defect 5: first-use creation of materializations and publication-staging under
    # builds/ must fsync the builds/ parent, not only the later per-build dirs.
    fsynced = _record_parent_fsyncs(monkeypatch)
    home = tmp_path / "home"
    layout = cache_module._layout(home, create=True)
    # builds/ is the parent of both materializations and publication-staging.
    assert layout.root in fsynced
    assert layout.staging.is_dir() and layout.journals.is_dir()


def test_publish_immutable_file_is_atomic_idempotent_and_no_replace(tmp_path: Path) -> None:
    directory = tmp_path / "d"
    directory.mkdir()
    path = directory / "x.json"
    cache_module._publish_immutable_file(path, b"data", directory, "thing")
    assert path.read_bytes() == b"data"
    assert not path.is_symlink()
    assert path.stat().st_mode & 0o777 == 0o400
    # A byte-identical republish is accepted (idempotent); a divergent one fails.
    cache_module._publish_immutable_file(path, b"data", directory, "thing")
    with pytest.raises(BuildCacheError, match="divergent thing collision"):
        cache_module._publish_immutable_file(path, b"other", directory, "thing")


def _no_tmp(directory: Path) -> bool:
    return not any(name.name.endswith(".tmp") for name in directory.iterdir())


def test_publish_immutable_file_temp_write_failure_leaves_no_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "d"
    directory.mkdir()
    path = directory / "x.json"

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("temp write/fsync failed")

    monkeypatch.setattr(cache_module, "write_exclusive", boom)
    with pytest.raises(OSError, match="temp write"):
        cache_module._publish_immutable_file(path, b"data", directory, "thing")
    assert not os.path.lexists(path)  # no partial final becomes authoritative
    assert _no_tmp(directory)
    monkeypatch.undo()
    cache_module._publish_immutable_file(path, b"data", directory, "thing")  # retry succeeds
    assert path.read_bytes() == b"data"


def test_publish_immutable_file_rename_failure_cleans_temp_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import errno

    directory = tmp_path / "d"
    directory.mkdir()
    path = directory / "x.json"

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "no space for rename")

    monkeypatch.setattr(cache_module, "rename_noreplace", boom)
    with pytest.raises(OSError, match="no space"):
        cache_module._publish_immutable_file(path, b"data", directory, "thing")
    assert not os.path.lexists(path)
    assert _no_tmp(directory)
    monkeypatch.undo()
    cache_module._publish_immutable_file(path, b"data", directory, "thing")
    assert path.read_bytes() == b"data"


def test_publish_immutable_file_parent_fsync_failure_then_idempotent_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "d"
    directory.mkdir()
    path = directory / "x.json"
    real = cache_module.fsync_directory
    fsynced: list[Path] = []

    def flaky(target: Path) -> None:
        fsynced.append(Path(target))
        if len(fsynced) == 1:
            raise OSError("parent fsync failed")
        real(target)

    monkeypatch.setattr(cache_module, "fsync_directory", flaky)
    # The rename already published the final durably; only the parent fsync failed.
    with pytest.raises(OSError, match="parent fsync"):
        cache_module._publish_immutable_file(path, b"data", directory, "thing")
    assert path.read_bytes() == b"data"
    assert fsynced == [directory]  # the first (failed) durability barrier
    # Retry sees the byte-identical final and MUST repeat the durability barrier
    # rather than returning without fsyncing the parent.
    cache_module._publish_immutable_file(path, b"data", directory, "thing")
    assert fsynced == [directory, directory]  # barrier repeated on accept-identical path


def test_publishing_recovery_completes_index_from_published_record(tmp_path: Path) -> None:
    import hashlib

    # Crash after the canonical record was atomically published but before its index:
    # PUBLISHING recovery completes publication forward from the journal-bound record.
    home = tmp_path / "home"
    with build_cache_session(_BUILD, _ATTEMPT_A, home=home) as session:
        session.begin_materialization(rehydrate=False)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        owner = write_build_root_owner(root, _ATTEMPT_A, _BUILD)
        session.bind_root(owner)
        layout = session.layout
        record = _record(producer=_ATTEMPT_A)
        payload = cache_module.canonical_json_bytes(record.model_dump(mode="json"))
        cache_module._publish_immutable_file(
            cache_module._record_path(layout, _BUILD), payload, layout.success, "canonical record"
        )
        cache_module._transition(
            layout,
            _BUILD,
            _ATTEMPT_A,
            MaterializationState.PUBLISHING,
            staging_sha256=hashlib.sha256(payload).hexdigest(),
            root_device=owner.root_device,
            root_inode=owner.root_inode,
        )
    assert not cache_module._index_path(cache_module._layout(home, create=False), _BUILD).exists()
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        assert session.lookup(_identity(), home=home).classification is CacheClassification.HIT
    assert cache_module._index_path(cache_module._layout(home, create=False), _BUILD).is_file()


def test_ensure_journal_events_rejects_symlinked_per_build_dir(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = cache_module._layout(home, create=True)
    journal = cache_module._journal(layout, _BUILD)
    journal.symlink_to(tmp_path / "elsewhere")  # pre-existing symlink, never followed
    with pytest.raises(BuildCacheError, match="unsafe"):
        cache_module._ensure_journal_events(journal)


def test_stage_path_rejects_symlinked_per_build_dir(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = cache_module._layout(home, create=True)
    root = layout.staging / _BUILD.removeprefix("build-sha256:")
    root.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(BuildCacheError, match="unsafe"):
        cache_module._stage_path(layout, _BUILD, _ATTEMPT_A)


def _swap_dir_for_symlink(directory: Path, aside: Path) -> None:
    """Move a real directory aside and replace it with a symlink to that target."""

    directory.rename(aside)
    directory.symlink_to(aside, target_is_directory=True)


def test_lookup_rejects_symlinked_per_build_journal_dir(tmp_path: Path) -> None:
    # Read side: a symlinked per-build journal directory must never be followed,
    # and must not read as an absent (MISS) registry.
    home = tmp_path / "home"
    _make_present(home)
    layout = cache_module._layout(home, create=False)
    _swap_dir_for_symlink(cache_module._journal(layout, _BUILD), tmp_path / "real-journal")
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="unsafe"),
    ):
        session.lookup(_identity(), home=home)


def test_lookup_rejects_symlinked_events_dir(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    layout = cache_module._layout(home, create=False)
    events = cache_module._journal(layout, _BUILD) / "events"
    _swap_dir_for_symlink(events, tmp_path / "real-events")
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="unsafe"),
    ):
        session.lookup(_identity(), home=home)


def test_lookup_rejects_dangling_current_json_symlink(tmp_path: Path) -> None:
    # A dangling current.json must be corruption (nofollow read fails closed), never
    # an absent registry read as a fresh MISS.
    home = tmp_path / "home"
    _make_present(home)
    layout = cache_module._layout(home, create=False)
    current = cache_module._journal(layout, _BUILD) / "current.json"
    current.unlink()
    current.symlink_to(tmp_path / "does-not-exist")
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError),
    ):
        session.lookup(_identity(), home=home)


def test_missing_events_dir_for_existing_journal_fails_closed(tmp_path: Path) -> None:
    # An existing journal/registry whose events directory has vanished is corruption,
    # not a silently empty chain: verification fails closed cleanly.
    home = tmp_path / "home"
    _make_present(home)
    layout = cache_module._layout(home, create=False)
    shutil.rmtree(cache_module._journal(layout, _BUILD) / "events")
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="events directory is missing"),
    ):
        session.lookup(_identity(), home=home)


def test_publishing_recovery_rejects_symlinked_staging_dir(tmp_path: Path) -> None:
    # _load_staged_record must validate the per-build staging directory descriptor-
    # anchored: a symlinked staging directory is never followed.
    home = tmp_path / "home"
    _stall_at_publishing(home)  # PUBLISHING with a staging digest but no staged file
    layout = cache_module._layout(home, create=False)
    stage_dir = layout.staging / _BUILD.removeprefix("build-sha256:")
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.symlink_to(tmp_path / "evil", target_is_directory=True)
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="unsafe"),
    ):
        session.lookup(_identity(), home=home)


def test_orphan_event_reconciliation_fsyncs_events_before_current_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Crash after an orphan event was renamed into place but before current.json was
    # promoted. Recovery must fsync the events directory BEFORE writing current.json,
    # so a failure there leaves current.json unpromoted rather than pointing at a
    # not-yet-durable event.
    home = tmp_path / "home"
    digest = _make_present(home)
    layout = cache_module._layout(home, create=False)
    root = home / "builds" / "materialized" / _BUILD
    owner = verify_build_root_owner(root)

    # Create the orphan: publish the next (CLEANING) event, then fail current.json.
    # Use scoped contexts so the autouse stubs stay in place.
    def failing_atomic(path: Path, payload: bytes) -> None:
        raise OSError("crash before current.json")

    with monkeypatch.context() as patched:
        patched.setattr(cache_module, "_atomic_write", failing_atomic)
        with pytest.raises(OSError, match="crash before current.json"):
            cache_module._transition(
                layout,
                _BUILD,
                owner.attempt_id,
                MaterializationState.CLEANING,
                canonical_record_sha256=digest,
                root_device=owner.root_device,
                root_inode=owner.root_inode,
            )

    journal = cache_module._journal(layout, _BUILD)
    events = journal / "events"
    events_stat = os.stat(events)
    real_os_fsync = os.fsync

    def fail_events_fsync(fd: int) -> None:
        # Reconciliation fsyncs the events directory descriptor before promoting
        # current.json; fail exactly that fsync (matched by device+inode).
        info = os.fstat(fd)
        if info.st_dev == events_stat.st_dev and info.st_ino == events_stat.st_ino:
            raise OSError("events fsync failed")
        real_os_fsync(fd)

    with monkeypatch.context() as patched:
        patched.setattr(os, "fsync", fail_events_fsync)
        with (
            build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
            pytest.raises(OSError, match="events fsync failed"),
        ):
            session.lookup(_identity(), home=home)
    # current.json was NOT promoted to the orphan: it is still the pre-orphan state.
    reg = cache_module._read_model(journal / "current.json", cache_module.MaterializationRegistryV1)
    assert reg.state is MaterializationState.PRESENT
    # A normal recovery now adopts the orphan and completes cleanup.
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        assert session.lookup(_identity(), home=home).classification is (
            CacheClassification.REHYDRATE
        )


def _dangle(path: Path, tmp_path: Path) -> None:
    path.unlink()
    path.symlink_to(tmp_path / "does-not-exist")


def test_dangling_canonical_record_symlink_is_integrity_failure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    layout = cache_module._layout(home, create=False)
    _dangle(cache_module._record_path(layout, _BUILD), tmp_path)
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError),  # nofollow read fails closed, never "absent"
    ):
        session.lookup(_identity(), home=home)


def test_dangling_build_index_symlink_is_integrity_failure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    layout = cache_module._layout(home, create=False)
    _dangle(cache_module._index_path(layout, _BUILD), tmp_path)
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError),
    ):
        session.lookup(_identity(), home=home)


def test_dangling_attestation_symlink_is_integrity_failure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    layout = cache_module._layout(home, create=False)
    attestation = cache_module._attestation_path(layout, _BUILD)
    attestation.unlink()  # replace the real attestation with a dangling symlink
    attestation.symlink_to(tmp_path / "does-not-exist")  # dangling, must not read as absent
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="unavailable"),
    ):
        session.lookup(_identity(), home=home)


def _events_dir(home: Path) -> Path:
    layout = cache_module._layout(home, create=False)
    return cache_module._journal(layout, _BUILD) / "events"


def _corrupt_event_chain_raises(home: Path) -> None:
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError),
    ):
        session.lookup(_identity(), home=home)


def test_event_chain_rejects_higher_sequence_extra(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    events = _events_dir(home)
    terminal = sorted(events.glob("*.json"))[-1]
    (events / "00000099.json").write_bytes(terminal.read_bytes())
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="unexpected entries"),
    ):
        session.lookup(_identity(), home=home)


def test_event_chain_rejects_interior_gap(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    events = _events_dir(home)
    sorted(events.glob("*.json"))[1].unlink()  # remove event 2 of the committed chain
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="unexpected entries"),
    ):
        session.lookup(_identity(), home=home)


def test_event_chain_rejects_unexpected_filename(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    (_events_dir(home) / "stray.json").write_bytes(b"{}")
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="unexpected entries"),
    ):
        session.lookup(_identity(), home=home)


def test_event_chain_rejects_multiple_orphans(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    events = _events_dir(home)
    terminal = sorted(events.glob("*.json"))[-1]
    sequence = int(terminal.stem)
    # Two non-adoptable events beyond the terminal: reconciliation adopts at most one
    # legal orphan; these copies fail to link, so recovery fails closed.
    (events / f"{sequence + 1:08d}.json").write_bytes(terminal.read_bytes())
    (events / f"{sequence + 2:08d}.json").write_bytes(terminal.read_bytes())
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="divergent orphan|unexpected entries"),
    ):
        session.lookup(_identity(), home=home)


def _write_temp_event(events: Path, name: str, content: bytes, *, mode: int = 0o400) -> Path:
    temp = events / name
    temp.write_bytes(content)
    temp.chmod(mode)
    return temp


def test_leftover_writer_temp_is_recovered_not_bricking(tmp_path: Path) -> None:
    # Crash boundary: _publish_event wrote its temp but crashed before the rename.
    # The uncommitted temp at the next sequence is authenticated and removed; the
    # journal recovers rather than failing the strict cardinality check.
    home = tmp_path / "home"
    _make_present(home)
    events = _events_dir(home)
    sequence = int(sorted(events.glob("*.json"))[-1].stem)
    temp = _write_temp_event(events, f".{sequence + 1:08d}.json.{'ab' * 8}.tmp", b"partial")
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        assert session.lookup(_identity(), home=home).classification is CacheClassification.HIT
    assert not temp.exists()


def test_redundant_writer_temp_copy_is_recovered(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    events = _events_dir(home)
    terminal = sorted(events.glob("*.json"))[-1]
    sequence = int(terminal.stem)
    # A byte-identical copy of an already-committed event, left beside it.
    temp = _write_temp_event(events, f".{sequence:08d}.json.{'cd' * 8}.tmp", terminal.read_bytes())
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        assert session.lookup(_identity(), home=home).classification is CacheClassification.HIT
    assert not temp.exists()


def test_writer_temp_with_unsafe_mode_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    events = _events_dir(home)
    sequence = int(sorted(events.glob("*.json"))[-1].stem)
    _write_temp_event(events, f".{sequence + 1:08d}.json.{'ef' * 8}.tmp", b"x", mode=0o600)
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="unsafe writer temp"),
    ):
        session.lookup(_identity(), home=home)


def test_multiple_writer_temps_fail_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    events = _events_dir(home)
    sequence = int(sorted(events.glob("*.json"))[-1].stem)
    _write_temp_event(events, f".{sequence + 1:08d}.json.{'00' * 8}.tmp", b"x")
    _write_temp_event(events, f".{sequence + 1:08d}.json.{'11' * 8}.tmp", b"y")
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="multiple writer temp"),
    ):
        session.lookup(_identity(), home=home)


def test_divergent_writer_temp_sequence_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    events = _events_dir(home)
    sequence = int(sorted(events.glob("*.json"))[-1].stem)
    # A temp far beyond the next committable sequence, with no committed event.
    _write_temp_event(events, f".{sequence + 5:08d}.json.{'22' * 8}.tmp", b"x")
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="divergent writer temp"),
    ):
        session.lookup(_identity(), home=home)


def test_foreign_temp_name_is_not_silently_removed(tmp_path: Path) -> None:
    # An entry that does not match the exact writer-temp shape is never treated as a
    # writer temp; strict cardinality rejects it as an unexpected entry.
    home = tmp_path / "home"
    _make_present(home)
    _write_temp_event(_events_dir(home), ".stray.json.deadbeef.tmp", b"x")
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="unexpected entries"),
    ):
        session.lookup(_identity(), home=home)


def _stall_at_publishing(home: Path) -> BuildRootOwnerV1:
    """Drive a build into a PUBLISHING journal with a staged digest but no canonical
    record and no recoverable staged bytes, returning the bound owner."""

    with build_cache_session(_BUILD, _ATTEMPT_A, home=home) as session:
        session.begin_materialization(rehydrate=False)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        owner = write_build_root_owner(root, _ATTEMPT_A, _BUILD)
        session.bind_root(owner)
        cache_module._transition(
            session.layout,
            _BUILD,
            _ATTEMPT_A,
            MaterializationState.PUBLISHING,
            staging_sha256="ab" * 32,  # no staged file was written for this digest
            root_device=owner.root_device,
            root_inode=owner.root_inode,
        )
    return owner


def test_publishing_missing_root_is_integrity_failure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _stall_at_publishing(home)
    shutil.rmtree(home / "builds" / "materialized" / _BUILD)
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="lost its root before recovery"),
    ):
        session.lookup(_identity(), home=home)


def test_publishing_dangling_root_is_integrity_failure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _stall_at_publishing(home)
    root = home / "builds" / "materialized" / _BUILD
    shutil.rmtree(root)
    root.symlink_to(tmp_path / "nowhere")
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="ownership changed"),
    ):
        session.lookup(_identity(), home=home)


def test_publishing_missing_owner_marker_is_integrity_failure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _stall_at_publishing(home)
    (home / "builds" / "materialized" / _BUILD / ".strixlab-owner.json").unlink()
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="ownership marker is unavailable"),
    ):
        session.lookup(_identity(), home=home)


def test_publishing_mismatched_owner_marker_is_integrity_failure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    owner = _stall_at_publishing(home)
    marker = home / "builds" / "materialized" / _BUILD / ".strixlab-owner.json"
    tampered = owner.model_copy(update={"root_inode": owner.root_inode + 1})
    marker.unlink()
    marker.write_bytes(cache_module.canonical_json_bytes(tampered.model_dump(mode="json")))
    marker.chmod(0o400)  # _read_owner requires the marker to be an owned 0o400 file
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="ownership changed"),
    ):
        session.lookup(_identity(), home=home)


def test_interrupted_rehydrating_returns_to_cleaned(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    cleanup_build(_BUILD, home=home)
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        session.begin_materialization(rehydrate=True)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        owner = write_build_root_owner(root, _ATTEMPT_B, _BUILD)
        session.bind_root(owner)  # interrupted before republish
    with build_cache_session(_BUILD, _ATTEMPT_A, home=home) as session:
        assert session.lookup(_identity(), home=home).classification is (
            CacheClassification.REHYDRATE
        )
    assert not (home / "builds" / "materialized" / _BUILD).exists()


def test_cleaning_recovery_completes_removal(tmp_path: Path) -> None:
    home = tmp_path / "home"
    digest = _make_present(home)
    layout = cache_module._layout(home, create=False)
    root = home / "builds" / "materialized" / _BUILD
    owner = verify_build_root_owner(root)
    cache_module._transition(
        layout,
        _BUILD,
        owner.attempt_id,
        MaterializationState.CLEANING,
        canonical_record_sha256=digest,
        root_device=owner.root_device,
        root_inode=owner.root_inode,
    )
    # root still present: recovery must finish removal and reach CLEANED
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        assert session.lookup(_identity(), home=home).classification is (
            CacheClassification.REHYDRATE
        )
    assert not root.exists()


def _publish_present_unattested(home: Path) -> None:
    """A PRESENT build that never reached the attestation boundary (crash-forward)."""

    with build_cache_session(_BUILD, _ATTEMPT_A, home=home) as session:
        session.begin_materialization(rehydrate=False)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        session.bind_root(write_build_root_owner(root, _ATTEMPT_A, _BUILD))
        session.publish(_record(producer=_ATTEMPT_A), rehydrate=False)


def test_unattested_present_inspect_reports_not_attested(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _publish_present_unattested(home)
    inspected = inspect_build(_BUILD, home=home)
    assert inspected.state is MaterializationState.PRESENT
    assert inspected.attested is False  # observable, but not reported as fully verified


def test_unattested_present_cleanup_refuses_destructive_removal(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _publish_present_unattested(home)
    with pytest.raises(BuildCacheError, match="refusing to clean a recovery-pending"):
        cleanup_build(_BUILD, home=home)
    assert (home / "builds" / "materialized" / _BUILD).is_dir()  # root/evidence preserved


def test_cleaned_missing_attestation_is_integrity_failure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    cleanup_build(_BUILD, home=home)
    layout = cache_module._layout(home, create=False)
    cache_module._attestation_path(layout, _BUILD).unlink()  # attestation lost after cleanup
    with pytest.raises(BuildCacheError, match="cleaned build is missing its attestation"):
        inspect_build(_BUILD, home=home)


def _replace_immutable(path: Path, payload: bytes) -> None:
    path.unlink()
    cache_module.write_exclusive(path, payload, 0o400)


def test_inspect_rejects_journal_canonical_digest_mismatch(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)  # registry bound to the original canonical digest
    layout = cache_module._layout(home, create=False)
    other = _record(
        producer=_ATTEMPT_A,
        artifacts=_artifacts().model_copy(update={"cmake_cache_sha256": "ab" * 32}),
    )
    payload = cache_module.canonical_json_bytes(other.model_dump(mode="json"))
    new_digest = cache_module._record_digest(payload)
    _replace_immutable(cache_module._record_path(layout, _BUILD), payload)
    index_payload = cache_module.canonical_json_bytes(
        cache_module.BuildIndexV1(
            build_id=_BUILD, canonical_record_sha256=new_digest, producer_attempt_id=_ATTEMPT_A
        ).model_dump(mode="json")
    )
    _replace_immutable(cache_module._index_path(layout, _BUILD), index_payload)
    # The record/index pair is internally consistent but no longer matches the
    # journal's bound canonical digest.
    with pytest.raises(BuildCacheError, match="journal changed canonical record binding"):
        inspect_build(_BUILD, home=home)


def test_cleaning_recovery_missing_canonical_is_integrity_failure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    digest = _make_present(home)
    layout = cache_module._layout(home, create=False)
    root = home / "builds" / "materialized" / _BUILD
    owner = verify_build_root_owner(root)
    cache_module._transition(
        layout,
        _BUILD,
        owner.attempt_id,
        MaterializationState.CLEANING,
        canonical_record_sha256=digest,
        root_device=owner.root_device,
        root_inode=owner.root_inode,
    )
    cache_module._record_path(layout, _BUILD).unlink()  # canonical lost before cleanup finishes
    cache_module._index_path(layout, _BUILD).unlink()
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="post-canonical recovery lost"),
    ):
        session.lookup(_identity(), home=home)
    assert root.is_dir()  # root preserved: cleanup never destructively deleted it


def _bind_root_in_state(home: Path, state: MaterializationState) -> Path:
    """Drive a cleaned build into REHYDRATING (or onward) with a fresh bound root."""

    _make_present(home)
    cleanup_build(_BUILD, home=home)
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        session.begin_materialization(rehydrate=True)  # REHYDRATING, canonical bound
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        owner = write_build_root_owner(root, _ATTEMPT_B, _BUILD)
        session.bind_root(owner)
        if state is MaterializationState.DISCARDING:
            cache_module._transition(
                session.layout,
                _BUILD,
                _ATTEMPT_B,
                MaterializationState.DISCARDING,
                root_device=owner.root_device,
                root_inode=owner.root_inode,
            )
    return root


def test_rehydrating_recovery_missing_canonical_is_integrity_failure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    root = _bind_root_in_state(home, MaterializationState.REHYDRATING)
    layout = cache_module._layout(home, create=False)
    cache_module._record_path(layout, _BUILD).unlink()  # post-canonical record lost
    with (
        build_cache_session(_BUILD, _ATTEMPT_A, home=home) as session,
        pytest.raises(BuildCacheError, match="post-canonical recovery lost|cardinality diverged"),
    ):
        session.lookup(_identity(), home=home)
    assert root.is_dir()  # rehydrate root preserved, never regressed to VACANT


def test_discarding_recovery_missing_canonical_is_integrity_failure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    root = _bind_root_in_state(home, MaterializationState.DISCARDING)
    layout = cache_module._layout(home, create=False)
    cache_module._record_path(layout, _BUILD).unlink()
    cache_module._index_path(layout, _BUILD).unlink()
    with (
        build_cache_session(_BUILD, _ATTEMPT_A, home=home) as session,
        pytest.raises(BuildCacheError, match="post-canonical recovery lost"),
    ):
        session.lookup(_identity(), home=home)
    assert root.is_dir()  # post-canonical DISCARDING never regresses to VACANT on loss


def test_post_canonical_discarding_with_present_root_resumes_to_cleaned(tmp_path: Path) -> None:
    # Two-pass crash: a DISCARDING event was recorded but the root was not removed
    # before the crash. Recovery must resume removal from its own DISCARDING state
    # (never a DISCARDING->DISCARDING transition) and reach CLEANED.
    home = tmp_path / "home"
    root = _bind_root_in_state(home, MaterializationState.DISCARDING)
    assert root.is_dir()  # DISCARDING recorded, removal not yet done
    with build_cache_session(_BUILD, _ATTEMPT_A, home=home) as session:
        assert session.lookup(_identity(), home=home).classification is (
            CacheClassification.REHYDRATE
        )
    assert not root.exists()  # removal resumed and completed
    assert inspect_build(_BUILD, home=home).state is MaterializationState.CLEANED


def test_pre_canonical_discarding_with_present_root_resumes_to_vacant(tmp_path: Path) -> None:
    # The pre-canonical counterpart: an interrupted BUILDING discard resumes to VACANT.
    home = tmp_path / "home"
    with build_cache_session(_BUILD, _ATTEMPT_A, home=home) as session:
        session.begin_materialization(rehydrate=False)  # BUILDING
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        owner = write_build_root_owner(root, _ATTEMPT_A, _BUILD)
        session.bind_root(owner)
        cache_module._transition(
            session.layout,
            _BUILD,
            _ATTEMPT_A,
            MaterializationState.DISCARDING,  # BUILDING->DISCARDING, no canonical bound
            root_device=owner.root_device,
            root_inode=owner.root_inode,
        )
    root = home / "builds" / "materialized" / _BUILD
    assert root.is_dir()
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        assert session.lookup(_identity(), home=home).classification is CacheClassification.MISS
    assert not root.exists()


def test_present_owner_marker_tampering_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    marker = home / "builds" / "materialized" / _BUILD / ".strixlab-owner.json"
    marker.chmod(0o600)
    marker.write_text("{}\n", encoding="utf-8")
    marker.chmod(0o400)
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError),
    ):
        session.lookup(_identity(), home=home)


def test_inspect_rejects_missing_and_invalid_ids(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    with pytest.raises(ValueError, match="invalid machine-local build ID"):
        inspect_build("nope", home=home)
    other = "build-sha256:" + "ff" * 32
    with pytest.raises(BuildCacheError):
        inspect_build(other, home=home)


def test_cleanup_missing_build_root_is_integrity_failure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    import shutil

    shutil.rmtree(home / "builds" / "materialized" / _BUILD)
    with pytest.raises(BuildCacheError):
        cleanup_build(_BUILD, home=home)


def test_owner_helpers_round_trip_and_remove(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    owner = write_build_root_owner(root, _ATTEMPT_A, _BUILD)
    assert verify_build_root_owner(root, owner) == owner
    with pytest.raises(BuildCacheError, match="ownership changed"):
        verify_build_root_owner(
            root,
            BuildRootOwnerV1(
                attempt_id=_ATTEMPT_B,
                build_id=_BUILD,
                root_device=owner.root_device,
                root_inode=owner.root_inode,
            ),
        )
    remove_owned_build_root(root, owner)
    assert not root.exists()
    remove_owned_build_root(None, None)  # no-op


def test_identity_and_tool_model_projections() -> None:
    entries = identity_models((IdentityEntry("generator", "Ninja"),))
    assert entries[0].name == "generator" and entries[0].value == "Ninja"
    tools = tool_models(
        (
            ToolObservation(
                role="cmake",
                path="/usr/bin/cmake",
                realpath="/usr/bin/cmake",
                mode=0o755,
                size_bytes=10,
                sha256=_HEX,
                version_sha256=_HEX,
                search_sha256=_HEX,
            ),
        )
    )
    assert tools[0].role == "cmake"


def test_write_owner_rejects_unsafe_root(tmp_path: Path) -> None:
    link = tmp_path / "link"
    link.symlink_to(tmp_path)
    with pytest.raises(BuildCacheError, match="unsafe"):
        write_build_root_owner(link, _ATTEMPT_A, _BUILD)


def test_publish_and_begin_state_guards(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with build_cache_session(_BUILD, _ATTEMPT_A, home=home) as session:
        with pytest.raises(BuildCacheError, match="cleaned materialization"):
            session.begin_materialization(rehydrate=True)
        session.begin_materialization(rehydrate=False)
        other = "build-sha256:" + "bb" * 32
        with pytest.raises(BuildCacheError, match="bound to another build"):
            session.publish(_record(build_id=other), rehydrate=False)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        owner = write_build_root_owner(root, _ATTEMPT_A, _BUILD)
        with pytest.raises(BuildCacheError, match="inconsistent"):
            session.bind_root(
                BuildRootOwnerV1(
                    attempt_id=_ATTEMPT_B,
                    build_id=_BUILD,
                    root_device=owner.root_device,
                    root_inode=owner.root_inode,
                )
            )


def test_new_build_requires_vacant_after_present(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="vacant materialization"),
    ):
        session.begin_materialization(rehydrate=False)


def test_corrupt_staging_before_canonical_is_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    original = cache_module._publish_canonical

    def failing(*args: object, **kwargs: object) -> None:
        raise BuildCacheError("injected")

    monkeypatch.setattr(cache_module, "_publish_canonical", failing)
    with (
        pytest.raises(BuildCacheError, match="injected"),
        build_cache_session(_BUILD, _ATTEMPT_A, home=home) as session,
    ):
        session.begin_materialization(rehydrate=False)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        owner = write_build_root_owner(root, _ATTEMPT_A, _BUILD)
        session.bind_root(owner)
        session.publish(_record(), rehydrate=False)
    monkeypatch.setattr(cache_module, "_publish_canonical", original)
    # corrupt the staged payload so recovery cannot complete forward
    hexid = _BUILD.removeprefix("build-sha256:")
    stage = home / "builds" / "publication-staging" / hexid / f"{_ATTEMPT_A}.json"
    stage.chmod(0o600)
    stage.write_bytes(b"corrupt")
    stage.chmod(0o400)
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        assert session.lookup(_identity(), home=home).classification is (CacheClassification.MISS)
    assert not (home / "builds" / "materialized" / _BUILD).exists()


def test_inspect_building_has_no_canonical_record(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = cache_module._layout(home, create=True)
    cache_module._transition(layout, _BUILD, _ATTEMPT_A, MaterializationState.BUILDING)
    with pytest.raises(BuildCacheError, match="canonical build record is missing"):
        inspect_build(_BUILD, home=home)


def test_inspect_cleaning_state_is_not_inspectable(tmp_path: Path) -> None:
    home = tmp_path / "home"
    digest = _make_present(home)
    layout = cache_module._layout(home, create=False)
    root = home / "builds" / "materialized" / _BUILD
    owner = verify_build_root_owner(root)
    cache_module._transition(
        layout,
        _BUILD,
        owner.attempt_id,
        MaterializationState.CLEANING,
        canonical_record_sha256=digest,
        root_device=owner.root_device,
        root_inode=owner.root_inode,
    )
    with pytest.raises(BuildCacheError, match="not inspectable"):
        inspect_build(_BUILD, home=home)


def test_lookup_state_without_canonical_record(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = cache_module._layout(home, create=True)
    # a journal transitioned to PRESENT without any canonical record is corrupt
    cache_module._transition(layout, _BUILD, _ATTEMPT_A, MaterializationState.BUILDING)
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        # BUILDING with no root -> recovery discards to VACANT -> MISS
        assert session.lookup(_identity(), home=home).classification is (CacheClassification.MISS)


def test_cleaning_recovery_when_root_already_gone(tmp_path: Path) -> None:
    import shutil

    home = tmp_path / "home"
    digest = _make_present(home)
    layout = cache_module._layout(home, create=False)
    root = home / "builds" / "materialized" / _BUILD
    owner = verify_build_root_owner(root)
    cache_module._transition(
        layout,
        _BUILD,
        owner.attempt_id,
        MaterializationState.CLEANING,
        canonical_record_sha256=digest,
        root_device=owner.root_device,
        root_inode=owner.root_inode,
    )
    shutil.rmtree(root)  # crash after removal, before CLEANED
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        # lookup runs recovery: CLEANING with the root gone finishes to CLEANED
        assert session.lookup(_identity(), home=home).classification is (
            CacheClassification.REHYDRATE
        )
    assert inspect_build(_BUILD, home=home).state is MaterializationState.CLEANED


def test_lookup_deleted_canonical_record_diverges(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    hexid = _BUILD.removeprefix("build-sha256:")
    (home / "builds" / "records" / "success" / f"{hexid}.json").unlink()
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="cardinality diverged"),
    ):
        session.lookup(_identity(), home=home)


def test_lookup_cleaned_state_with_unexpected_root(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    cleanup_build(_BUILD, home=home)
    root = home / "builds" / "materialized" / _BUILD
    root.mkdir(parents=True, exist_ok=True)  # a cleaned build must not have a root
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="materialized root"),
    ):
        session.lookup(_identity(), home=home)


def test_publish_canonical_divergent_collision(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = cache_module._layout(home, create=True)
    record = _record()
    payload_a = cache_module.canonical_json_bytes(record.model_dump(mode="json"))
    cache_module._publish_canonical(
        layout, record, payload_a, cache_module._record_digest(payload_a)
    )
    divergent = _record(producer=_ATTEMPT_B)
    payload_b = cache_module.canonical_json_bytes(divergent.model_dump(mode="json"))
    with pytest.raises(BuildCacheError, match="divergent canonical build record"):
        cache_module._publish_canonical(
            layout, divergent, payload_b, cache_module._record_digest(payload_b)
        )


def test_bind_root_rejected_in_present_state(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with build_cache_session(_BUILD, _ATTEMPT_A, home=home) as session:
        session.begin_materialization(rehydrate=False)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        owner = write_build_root_owner(root, _ATTEMPT_A, _BUILD)
        session.bind_root(owner)
        session.publish(_record(), rehydrate=False)
        with pytest.raises(BuildCacheError, match="cannot be bound"):
            session.bind_root(owner)


def test_verify_owner_missing_root(tmp_path: Path) -> None:
    with pytest.raises(BuildCacheError, match="unavailable"):
        verify_build_root_owner(tmp_path / "nope")


def test_rehydrate_publish_without_canonical(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    cleanup_build(_BUILD, home=home)
    hexid = _BUILD.removeprefix("build-sha256:")
    (home / "builds" / "records" / "success" / f"{hexid}.json").unlink()
    (home / "builds" / "indexes" / "builds" / f"{hexid}.json").unlink()
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        session.begin_materialization(rehydrate=True)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        owner = write_build_root_owner(root, _ATTEMPT_B, _BUILD)
        session.bind_root(owner)
        with pytest.raises(BuildCacheError, match="lost its canonical"):
            session.publish(_record(), rehydrate=True)


def test_tampered_registry_diverges_from_event_chain(tmp_path: Path) -> None:
    import json

    home = tmp_path / "home"
    _make_present(home)
    hexid = _BUILD.removeprefix("build-sha256:")
    current = home / "builds" / "materializations" / hexid / "current.json"
    data = json.loads(current.read_bytes())
    data["last_event_sha256"] = "f" * 64
    current.chmod(0o600)
    current.write_bytes(json.dumps(data).encode())
    current.chmod(0o400)
    with pytest.raises(BuildCacheError, match="diverged from its event chain"):
        inspect_build(_BUILD, home=home)


def test_layout_requires_absolute_home(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        cache_module._layout(Path("relative"), create=True)


def test_cache_does_not_exist_without_create(tmp_path: Path) -> None:
    with pytest.raises(BuildCacheError, match="does not exist"):
        inspect_build(_BUILD, home=tmp_path / "empty")


def test_owner_marker_wrong_mode_is_unsafe(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    write_build_root_owner(root, _ATTEMPT_A, _BUILD)
    marker = root / ".strixlab-owner.json"
    marker.chmod(0o600)  # marker must remain read-only
    with pytest.raises(BuildCacheError, match="marker is unsafe"):
        verify_build_root_owner(root)


def test_read_bytes_rejects_oversize_and_missing(tmp_path: Path) -> None:
    target = tmp_path / "f.json"
    target.write_bytes(b"0123456789")
    with pytest.raises(BuildCacheError, match="unsafe"):
        cache_module._read_bytes(target, 3)
    with pytest.raises(BuildCacheError, match="unavailable"):
        cache_module._read_bytes(tmp_path / "missing", 100)


def test_load_staged_record_without_staging_digest(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = cache_module._layout(home, create=True)
    registry = cache_module._transition(layout, _BUILD, _ATTEMPT_A, MaterializationState.BUILDING)
    assert cache_module._load_staged_record(layout, registry) is None


def test_remove_owned_build_root_rejects_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    owner = write_build_root_owner(root, _ATTEMPT_A, _BUILD)
    mismatched = BuildRootOwnerV1(
        attempt_id=_ATTEMPT_B,
        build_id=_BUILD,
        root_device=owner.root_device,
        root_inode=owner.root_inode,
    )
    with pytest.raises(BuildCacheError, match="ownership changed"):
        remove_owned_build_root(root, mismatched)
    assert root.exists()


def _rewrite_owner(root: Path, owner: BuildRootOwnerV1) -> None:
    marker = root / ".strixlab-owner.json"
    marker.chmod(0o600)
    marker.write_bytes(cache_module.canonical_json_bytes(owner.model_dump(mode="json")))
    marker.chmod(0o400)


def test_cache_hit_identity_divergence_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="cache identity diverged"),
    ):
        session.lookup(_identity(requested_targets=("divergent",)), home=home)


def test_rehydrate_identity_divergence_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    cleanup_build(_BUILD, home=home)
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="cache identity diverged"),
    ):
        session.lookup(
            _identity(selections=(IdentityEntryV1(name="generator", value="Other"),)), home=home
        )


def test_owner_journal_binding_tamper_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    root = home / "builds" / "materialized" / _BUILD
    owner = verify_build_root_owner(root)
    # A well-formed marker whose device no longer matches the journal binding.
    _rewrite_owner(
        root,
        BuildRootOwnerV1(
            attempt_id=owner.attempt_id,
            build_id=owner.build_id,
            root_device=owner.root_device + 1,
            root_inode=owner.root_inode,
        ),
    )
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="diverged from its journal binding|ownership changed"),
    ):
        session.lookup(_identity(), home=home)


def test_publish_rejects_foreign_producer(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with build_cache_session(_BUILD, _ATTEMPT_A, home=home) as session:
        session.begin_materialization(rehydrate=False)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        session.bind_root(write_build_root_owner(root, _ATTEMPT_A, _BUILD))
        with pytest.raises(BuildCacheError, match="producer is not the publishing attempt"):
            session.publish(_record(producer=_ATTEMPT_B), rehydrate=False)


def test_publishing_recovery_repairs_missing_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    original = cache_module._publish_canonical

    def record_then_fail(layout: object, record: object, payload: bytes, digest: str) -> None:
        cache_module.write_exclusive(
            cache_module._record_path(layout, record.build_id), payload, 0o400
        )
        raise BuildCacheError("crash between record and index")

    monkeypatch.setattr(cache_module, "_publish_canonical", record_then_fail)
    with (
        pytest.raises(BuildCacheError, match="between record and index"),
        build_cache_session(_BUILD, _ATTEMPT_A, home=home) as session,
    ):
        session.begin_materialization(rehydrate=False)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        session.bind_root(write_build_root_owner(root, _ATTEMPT_A, _BUILD))
        session.publish(_record(producer=_ATTEMPT_A), rehydrate=False)

    hexid = _BUILD.removeprefix("build-sha256:")
    assert (home / "builds" / "records" / "success" / f"{hexid}.json").exists()
    assert not (home / "builds" / "indexes" / "builds" / f"{hexid}.json").exists()

    monkeypatch.setattr(cache_module, "_publish_canonical", original)
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        assert session.lookup(_identity(), home=home).classification is CacheClassification.HIT
    assert (home / "builds" / "indexes" / "builds" / f"{hexid}.json").exists()


def test_orphan_index_without_record_is_integrity_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"

    def index_only(layout: object, record: object, payload: bytes, digest: str) -> None:
        index = cache_module.BuildIndexV1(
            build_id=record.build_id,
            canonical_record_sha256=digest,
            producer_attempt_id=record.producer_attempt_id,
        )
        cache_module.write_exclusive(
            cache_module._index_path(layout, record.build_id),
            cache_module.canonical_json_bytes(index.model_dump(mode="json")),
            0o400,
        )
        raise BuildCacheError("crash after index before record")

    monkeypatch.setattr(cache_module, "_publish_canonical", index_only)
    with (
        pytest.raises(BuildCacheError, match="after index before record"),
        build_cache_session(_BUILD, _ATTEMPT_A, home=home) as session,
    ):
        session.begin_materialization(rehydrate=False)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        session.bind_root(write_build_root_owner(root, _ATTEMPT_A, _BUILD))
        session.publish(_record(producer=_ATTEMPT_A), rehydrate=False)

    monkeypatch.undo()
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="index published without its canonical record"),
    ):
        session.lookup(_identity(), home=home)


def test_index_producer_mismatch_is_binding_failure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    hexid = _BUILD.removeprefix("build-sha256:")
    index_path = home / "builds" / "indexes" / "builds" / f"{hexid}.json"
    index = cache_module._read_model(index_path, cache_module.BuildIndexV1)
    tampered = index.model_copy(update={"producer_attempt_id": _ATTEMPT_B})
    index_path.chmod(0o600)
    index_path.write_bytes(cache_module.canonical_json_bytes(tampered.model_dump(mode="json")))
    index_path.chmod(0o400)
    with pytest.raises(BuildCacheError, match="binding diverged"):
        inspect_build(_BUILD, home=home)


def test_rehydrate_verifies_rebuilt_root_against_canonical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A rehydrate must re-verify the rebuilt root against the *retained canonical*
    # artifacts (CMake cache / compile-commands / files / capture tools), exactly
    # as a later HIT will, so it can never publish PRESENT over a root that would
    # immediately fail the next HIT. The full artifact evidence cannot be compared
    # byte-for-byte because inspection digests embed ASLR-varying ldd output.
    from strixlab.build_artifacts import BuildArtifactError

    home = tmp_path / "home"
    _make_present(home)
    cleanup_build(_BUILD, home=home)

    verified: dict[str, object] = {}

    def divergent_root(root: Path, expected: object, **_kwargs: object) -> None:
        verified["expected"] = expected
        raise BuildArtifactError("CMakeCache.txt changed")

    monkeypatch.setattr(cache_module, "verify_artifact_capture", divergent_root)
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        assert (
            session.lookup(_identity(), home=home).classification is CacheClassification.REHYDRATE
        )
        session.begin_materialization(rehydrate=True)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        session.bind_root(write_build_root_owner(root, _ATTEMPT_B, _BUILD))
        with pytest.raises(BuildArtifactError, match="CMakeCache.txt changed"):
            session.publish(_record(producer=_ATTEMPT_A), rehydrate=True)
    # The rebuilt root was verified against the canonical (retained) artifacts.
    assert verified["expected"] == _artifacts()


def test_identity_derivation_covers_all_projection_fields(tmp_path: Path) -> None:
    identity = _identity()
    record = _record()  # built from _identity() via the shared model machinery
    assert record.identity() == identity
    # Every projection field is derived, not a hand-listed subset.
    for name in BuildIdentityProjectionV1.model_fields:
        assert getattr(record.identity(), name) == getattr(record, name)
    # A representative projection field participates in round-trip equality.
    host = record.model_copy(update={"toolchain_mode": "host"})
    assert host.identity() != identity
    assert host.identity().toolchain_mode == "host"


def test_atomic_write_cleans_up_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os as _os

    target = tmp_path / "current.json"
    target.write_bytes(b"original")

    def failing_replace(src: object, dst: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(_os, "replace", failing_replace)
    with pytest.raises(OSError, match="replace failed"):
        cache_module._atomic_write(target, b"new payload")
    assert list(tmp_path.glob(".current.json.*.tmp")) == []  # no orphan temp
    assert target.read_bytes() == b"original"  # original untouched


def test_layout_create_false_never_creates_missing(tmp_path: Path) -> None:
    import shutil as _shutil

    home = tmp_path / "home"
    with pytest.raises(BuildCacheError, match="does not exist"):
        cache_module._layout(home, create=False)
    assert not (home / "builds").exists()  # validation must not materialize the tree

    cache_module._layout(home, create=True)
    materialized = home / "builds" / "materialized"
    _shutil.rmtree(materialized)  # a path removed out from under a create=False validate
    with pytest.raises(BuildCacheError, match="does not exist"):
        cache_module._layout(home, create=False)
    assert not materialized.exists()  # fails closed; never re-created


def test_carry_forward_prefers_new_then_previous_then_none() -> None:
    from strixlab.build_cache import MaterializationRegistryV1, _carry_forward

    previous = MaterializationRegistryV1(
        build_id=_BUILD,
        attempt_id=_ATTEMPT_A,
        state=MaterializationState.BUILDING,
        sequence=1,
        last_event_sha256=_HEX,
        canonical_record_sha256="record-sha256:" + _HEX,
        staging_sha256=_HEX,
        root_device=5,
        root_inode=7,
    )
    # str field: explicit new wins, else carry previous, else None
    assert _carry_forward("record-sha256:" + "ab" * 32, previous, "canonical_record_sha256") == (
        "record-sha256:" + "ab" * 32
    )
    assert _carry_forward(None, previous, "canonical_record_sha256") == "record-sha256:" + _HEX
    # int field: explicit new wins, else carry previous, else None
    assert _carry_forward(9, previous, "root_device") == 9
    assert _carry_forward(None, previous, "root_device") == 5
    # no previous registry -> None
    assert _carry_forward(None, None, "root_device") is None
    # previous present but the carried attribute is itself None -> None
    empty = previous.model_copy(update={"root_device": None})
    assert _carry_forward(None, empty, "root_device") is None


def _journal_dir(home: Path) -> Path:
    return home / "builds" / "materializations" / _BUILD.removeprefix("build-sha256:")


def test_transition_reconciles_after_first_event_current_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    layout = cache_module._layout(home, create=True)
    real_atomic = cache_module._atomic_write
    failed = {"done": False}

    def failing_atomic(path: Path, payload: bytes) -> None:
        if path.name == "current.json" and not failed["done"]:
            failed["done"] = True
            raise OSError("current write failed")
        real_atomic(path, payload)

    monkeypatch.setattr(cache_module, "_atomic_write", failing_atomic)
    with pytest.raises(OSError, match="current write failed"):
        cache_module._transition(layout, _BUILD, _ATTEMPT_A, MaterializationState.BUILDING)

    journal = _journal_dir(home)
    assert (journal / "events" / "00000001.json").exists()  # event durable
    assert not (journal / "current.json").exists()  # registry never advanced

    # Retry: loading reconciles the single orphan event (no FileExistsError).
    reconciled = cache_module._load_registry(layout, _BUILD)
    assert reconciled is not None
    assert reconciled.state is MaterializationState.BUILDING
    assert reconciled.sequence == 1
    # A subsequent transition proceeds from the reconciled state.
    nxt = cache_module._transition(layout, _BUILD, _ATTEMPT_A, MaterializationState.BUILDING)
    assert nxt.sequence == 2
    assert cache_module._load_verified_registry(layout, _BUILD) == nxt


def test_transition_reconciles_after_later_event_current_write_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    layout = cache_module._layout(home, create=True)
    cache_module._transition(layout, _BUILD, _ATTEMPT_A, MaterializationState.BUILDING)  # seq 1 ok
    real_atomic = cache_module._atomic_write
    failed = {"done": False}

    def failing_atomic(path: Path, payload: bytes) -> None:
        if path.name == "current.json" and not failed["done"]:
            failed["done"] = True
            raise OSError("current write failed")
        real_atomic(path, payload)

    monkeypatch.setattr(cache_module, "_atomic_write", failing_atomic)
    with pytest.raises(OSError, match="current write failed"):
        cache_module._transition(layout, _BUILD, _ATTEMPT_A, MaterializationState.PUBLISHING)

    journal = _journal_dir(home)
    assert (journal / "events" / "00000002.json").exists()
    current = cache_module._read_model(
        journal / "current.json", cache_module.MaterializationRegistryV1
    )
    assert current.sequence == 1  # registry still at the pre-crash event

    reconciled = cache_module._load_registry(layout, _BUILD)
    assert reconciled is not None
    assert reconciled.state is MaterializationState.PUBLISHING
    assert reconciled.sequence == 2
    assert cache_module._load_verified_registry(layout, _BUILD) == reconciled


def test_divergent_orphan_event_is_rejected(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = cache_module._layout(home, create=True)
    cache_module._transition(layout, _BUILD, _ATTEMPT_A, MaterializationState.BUILDING)  # seq 1
    events = _journal_dir(home) / "events"
    forged = cache_module.MaterializationEventV1(
        build_id=_BUILD,
        attempt_id=_ATTEMPT_A,
        sequence=2,
        previous_sha256="f" * 64,  # does not chain to event 1
        from_state=MaterializationState.BUILDING,
        to_state=MaterializationState.PUBLISHING,
    )
    payload = cache_module.canonical_json_bytes(forged.model_dump(mode="json", by_alias=True))
    (events / "00000002.json").write_bytes(payload)
    with pytest.raises(BuildCacheError, match="divergent orphan"):
        cache_module._load_registry(layout, _BUILD)


def test_cleaning_recovery_rejects_dangling_symlink_root(tmp_path: Path) -> None:
    import shutil as _shutil

    home = tmp_path / "home"
    digest = _make_present(home)
    layout = cache_module._layout(home, create=False)
    root = home / "builds" / "materialized" / _BUILD
    owner = verify_build_root_owner(root)
    cache_module._transition(
        layout,
        _BUILD,
        owner.attempt_id,
        MaterializationState.CLEANING,
        canonical_record_sha256=digest,
        root_device=owner.root_device,
        root_inode=owner.root_inode,
    )
    _shutil.rmtree(root)
    root.symlink_to(tmp_path / "nowhere")  # dangling symlink where the owned root was
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError),
    ):
        session.lookup(_identity(), home=home)
    assert root.is_symlink()  # not treated as absent; preserved fail-closed


def _artifacts_with_inspection(ldd_sha256: str) -> BuildArtifactsV1:
    from strixlab.build_artifacts import DynamicInspectionV1

    inspection = DynamicInspectionV1(
        artifact="bin/llama-bench",
        elf_type="ET_DYN",
        dynamic=True,
        static=False,
        needed=("libc.so.6",),
        dependencies=(),
        readelf_sha256=_HEX,
        ldd_sha256=ldd_sha256,
    )
    return _artifacts().model_copy(update={"inspections": (inspection,)})


def _publish_present(home: Path, artifacts: BuildArtifactsV1, *, attempt: str = _ATTEMPT_A) -> None:
    with build_cache_session(_BUILD, attempt, home=home) as session:
        session.begin_materialization(rehydrate=False)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        session.bind_root(write_build_root_owner(root, attempt, _BUILD))
        digest = session.publish(_record(producer=attempt, artifacts=artifacts), rehydrate=False)
    _write_attestation(home, digest)  # a normal present build is attested


def test_rehydrate_accepts_divergent_non_identity_observations(tmp_path: Path) -> None:
    # The approved rehydrate equivalence is artifact-set ID + requested-target/
    # selection identity only. A fresh capture that differs in a non-identity
    # observation (here the CMake cache digest) but keeps the same artifact-set ID
    # and identity must NOT be rejected; the retained-canonical root verification
    # (stubbed here) is what guards the materialized root.
    home = tmp_path / "home"
    _make_present(home)
    cleanup_build(_BUILD, home=home)
    non_identity = _artifacts().model_copy(update={"cmake_cache_sha256": "ab" * 32})
    assert non_identity.artifact_set_id == _artifacts().artifact_set_id
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        assert (
            session.lookup(_identity(), home=home).classification is CacheClassification.REHYDRATE
        )
        session.begin_materialization(rehydrate=True)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        session.bind_root(write_build_root_owner(root, _ATTEMPT_B, _BUILD))
        session.publish(_record(producer=_ATTEMPT_A, artifacts=non_identity), rehydrate=True)
    assert inspect_build(_BUILD, home=home).state is MaterializationState.PRESENT
    # Defect 1: the PRESENT root is bound to the REHYDRATING attempt's own fresh
    # evidence (non-identity observations included), not the original producer's.
    layout = cache_module._layout(home, create=False)
    registry = cache_module._required_registry(layout, _BUILD)
    evidence = cache_module._load_materialization_evidence(layout, registry)
    assert evidence.attempt_id == _ATTEMPT_B  # the rehydrating attempt, not _ATTEMPT_A
    assert evidence.artifacts.cmake_cache_sha256 == "ab" * 32
    assert evidence.artifacts.artifact_set_id == _artifacts().artifact_set_id
    # A subsequent HIT authenticates the root against that fresh materialization.
    with build_cache_session(_BUILD, _ATTEMPT_A, home=home) as session:
        assert session.lookup(_identity(), home=home).classification is CacheClassification.HIT


def test_materialization_evidence_digest_tamper_fails_hit(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _make_present(home)
    layout = cache_module._layout(home, create=False)
    mat = cache_module._materialization_path(layout, _BUILD)
    mat.chmod(0o600)
    # Same schema, different bytes: the journal-bound digest no longer matches.
    mat.write_bytes(mat.read_bytes() + b"\n")
    with (
        build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session,
        pytest.raises(BuildCacheError, match="materialization evidence diverged"),
    ):
        session.lookup(_identity(), home=home)


def test_materialization_evidence_binding_tamper_fails_hit(tmp_path: Path) -> None:
    import hashlib

    home = tmp_path / "home"
    _make_present(home)
    layout = cache_module._layout(home, create=False)
    registry = cache_module._required_registry(layout, _BUILD)
    # Re-point the evidence at a foreign attempt while keeping the journal digest in
    # sync, so only the build/attempt/canonical binding check (not the digest) is
    # left to catch it.
    evidence = cache_module._load_materialization_evidence(layout, registry)
    forged = evidence.model_copy(update={"attempt_id": _ATTEMPT_B})
    payload = cache_module.canonical_json_bytes(forged.model_dump(mode="json"))
    mat = cache_module._materialization_path(layout, _BUILD)
    mat.chmod(0o600)
    mat.write_bytes(payload)
    mat.chmod(0o400)
    tampered = registry.model_copy(
        update={"materialization_sha256": hashlib.sha256(payload).hexdigest()}
    )
    with pytest.raises(BuildCacheError, match="materialization evidence is not bound"):
        cache_module._load_materialization_evidence(layout, tampered)


def _complete_inventory() -> dict[str, tuple[str, int]]:
    digest = "0" * 64
    paths = [
        "build/artifacts.json",
        "build/profile.resolved.json",
        "build/environment.json",
        "source/evidence.json",
        "source/snapshot.json",
        "cmake/file-api/index.json",
        "cmake/probe-configure-cache.txt",
        "cmake/compile_commands.json",
        "tools/observations.json",
        "process/build.json",
    ]
    return {path: (digest, 1) for path in paths}


def test_require_complete_inventory_accepts_full_set() -> None:
    cache_module._require_complete_inventory(_complete_inventory())  # no raise


def test_require_complete_inventory_rejects_each_missing_requirement() -> None:
    for dropped in ("build/artifacts.json", "source/snapshot.json"):
        inventory = _complete_inventory()
        del inventory[dropped]
        with pytest.raises(BuildCacheError, match="missing required evidence"):
            cache_module._require_complete_inventory(inventory)
    for prefix_item in (
        "cmake/file-api/index.json",
        "tools/observations.json",
        "process/build.json",
    ):
        inventory = _complete_inventory()
        del inventory[prefix_item]
        with pytest.raises(BuildCacheError, match="missing required evidence"):
            cache_module._require_complete_inventory(inventory)


def test_require_complete_inventory_rejects_missing_configure_cache() -> None:
    inventory = _complete_inventory()
    del inventory["cmake/probe-configure-cache.txt"]
    with pytest.raises(BuildCacheError, match="missing a configure cache"):
        cache_module._require_complete_inventory(inventory)


def test_require_complete_inventory_requires_exactly_one_compile_database() -> None:
    absent = _complete_inventory()
    del absent["cmake/compile_commands.json"]
    with pytest.raises(BuildCacheError, match="compile-database"):
        cache_module._require_complete_inventory(absent)
    both = _complete_inventory()
    both["cmake/compile_commands.absent.json"] = ("0" * 64, 1)
    with pytest.raises(BuildCacheError, match="compile-database"):
        cache_module._require_complete_inventory(both)


def test_rehydrate_rejects_divergent_artifact_set_id(tmp_path: Path) -> None:
    # A divergent artifact-set ID is an identity difference and must fail closed.
    home = tmp_path / "home"
    _make_present(home)
    cleanup_build(_BUILD, home=home)
    divergent = _artifacts().model_copy(
        update={"artifact_set_id": "artifact-set-sha256:" + "ab" * 32}
    )
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        assert (
            session.lookup(_identity(), home=home).classification is CacheClassification.REHYDRATE
        )
        session.begin_materialization(rehydrate=True)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        session.bind_root(write_build_root_owner(root, _ATTEMPT_B, _BUILD))
        with pytest.raises(BuildCacheError, match="diverged from canonical evidence"):
            session.publish(_record(producer=_ATTEMPT_A, artifacts=divergent), rehydrate=True)


def test_rehydrate_ignores_ldd_process_digest_difference(tmp_path: Path) -> None:
    home = tmp_path / "home"
    _publish_present(home, _artifacts_with_inspection("aa" * 32))
    cleanup_build(_BUILD, home=home)
    # A fresh capture that differs only in the ASLR-varying ldd digest must rehydrate.
    with build_cache_session(_BUILD, _ATTEMPT_B, home=home) as session:
        assert (
            session.lookup(_identity(), home=home).classification is CacheClassification.REHYDRATE
        )
        session.begin_materialization(rehydrate=True)
        root = session.root
        root.mkdir(parents=True, exist_ok=True)
        session.bind_root(write_build_root_owner(root, _ATTEMPT_B, _BUILD))
        session.publish(
            _record(producer=_ATTEMPT_A, artifacts=_artifacts_with_inspection("bb" * 32)),
            rehydrate=True,
        )
    assert inspect_build(_BUILD, home=home).state is MaterializationState.PRESENT

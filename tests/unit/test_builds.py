from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import strixlab.builds as build_module
from strixlab.build_records import publish_record
from strixlab.builds import (
    AttemptOutcome,
    AttemptState,
    BuildBusyError,
    BuildStateError,
    ProcessOwnerV1,
    begin_build_attempt,
    inspect_recipe,
)

_RECIPE = "recipe-sha256:" + "ab" * 32
_OTHER_RECIPE = "recipe-sha256:" + "ef" * 32
_BUILD = "build-sha256:" + "cd" * 32


def _nonce(value: int):
    return lambda: bytes((value,)) * 16


def _stale_owner(owner: ProcessOwnerV1) -> ProcessOwnerV1:
    return owner.model_copy(update={"boot_id": "stale-boot"})


def test_attempt_finalization_publishes_one_immutable_record_and_index(tmp_path: Path) -> None:
    home = tmp_path / "home"

    with begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(1)) as attempt:
        identifier = attempt.registry.attempt_id
        attempt.mark_active()
        attempt.write_evidence("logs/configure.stdout", b"configure failed\n")
        result = attempt.finalize(AttemptOutcome.FAILED)

    index = inspect_recipe(_RECIPE, home=home)
    assert result.attempt_id == identifier
    assert result.record.is_dir()
    assert not (home / "builds" / "attempts" / identifier).exists()
    assert index.attempts[0].outcome is AttemptOutcome.FAILED
    assert index.attempts[0].record_sha256 == result.record_sha256


def test_attempt_context_finalizes_uncaught_failures_without_persisting_error_text(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"

    with (
        pytest.raises(RuntimeError, match="secret-bearing-message"),
        begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(2)) as attempt,
    ):
        identifier = attempt.registry.attempt_id
        raise RuntimeError("secret-bearing-message")

    index = inspect_recipe(_RECIPE, home=home)
    record = home / "builds" / "records" / "attempts" / identifier
    assert index.attempts[0].outcome is AttemptOutcome.FAILED
    assert b"secret-bearing-message" not in (record / "terminal.json").read_bytes()


def test_successful_outcomes_require_a_machine_build_id(tmp_path: Path) -> None:
    home = tmp_path / "home"

    with begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(3)) as attempt:
        with pytest.raises(BuildStateError, match="require a build ID"):
            attempt.finalize(AttemptOutcome.SUCCESS)
        result = attempt.finalize(AttemptOutcome.SUCCESS, build_id=_BUILD)

    assert result.build_id == _BUILD
    assert inspect_recipe(_RECIPE, home=home).attempts[0].build_id == _BUILD


def test_live_attempt_owner_is_not_recovered_as_stale(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    active = build_module._allocate_attempt(layout, _RECIPE, _nonce(4))

    with (
        pytest.raises(BuildBusyError, match="still owned"),
        begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(5)),
    ):
        pytest.fail("a live attempt must not be replaced")

    assert active.root.is_dir()


def test_stale_attempt_is_finalized_as_interrupted_before_retry(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    stale = build_module._allocate_attempt(layout, _RECIPE, _nonce(6))
    stale.mark_active()
    stale.registry = stale.registry.model_copy(update={"owner": _stale_owner(stale.registry.owner)})
    build_module._atomic_model(stale.root / "current.json", stale.registry)

    with begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(7)) as fresh:
        fresh.finalize(AttemptOutcome.FAILED)

    index = inspect_recipe(_RECIPE, home=home)
    assert [entry.outcome for entry in index.attempts] == [
        AttemptOutcome.INTERRUPTED,
        AttemptOutcome.FAILED,
    ]


def test_recovery_finishes_publication_interrupted_after_record_rename(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    stale = build_module._allocate_attempt(layout, _RECIPE, _nonce(8))
    stale.registry = stale.registry.model_copy(update={"owner": _stale_owner(stale.registry.owner)})
    build_module._atomic_model(stale.root / "current.json", stale.registry)
    terminal = build_module._transition(
        stale.root,
        stale.registry,
        AttemptState.FINALIZING,
        changes={"outcome": AttemptOutcome.FAILED},
    )
    destination = layout.attempt_records / terminal.attempt_id
    publish_record(stale.root, destination)

    with begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(9)) as fresh:
        fresh.finalize(AttemptOutcome.FAILED)

    index = inspect_recipe(_RECIPE, home=home)
    assert len(index.attempts) == 2
    assert index.attempts[0].outcome is AttemptOutcome.FAILED
    assert not stale.root.exists()


def test_recovery_uses_the_journaled_digest_after_record_publication(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    stale = build_module._allocate_attempt(layout, _RECIPE, _nonce(31))
    stale.registry = stale.registry.model_copy(update={"owner": _stale_owner(stale.registry.owner)})
    build_module._atomic_model(stale.root / "current.json", stale.registry)
    terminal = build_module._transition(
        stale.root,
        stale.registry,
        AttemptState.FINALIZING,
        changes={"outcome": AttemptOutcome.FAILED},
    )
    verification = publish_record(stale.root, layout.attempt_records / terminal.attempt_id)
    build_module._transition(
        stale.root,
        terminal,
        AttemptState.RECORD_PUBLISHED,
        changes={"record_sha256": verification.record_sha256},
    )

    with begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(32)) as fresh:
        fresh.finalize(AttemptOutcome.FAILED)

    index = inspect_recipe(_RECIPE, home=home)
    assert index.attempts[0].record_sha256 == verification.record_sha256
    assert index.attempts[0].state is AttemptState.TORN_DOWN


def test_recovery_rejects_a_changed_journaled_record_digest(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    stale = build_module._allocate_attempt(layout, _RECIPE, _nonce(33))
    stale.registry = stale.registry.model_copy(update={"owner": _stale_owner(stale.registry.owner)})
    build_module._atomic_model(stale.root / "current.json", stale.registry)
    terminal = build_module._transition(
        stale.root,
        stale.registry,
        AttemptState.FINALIZING,
        changes={"outcome": AttemptOutcome.FAILED},
    )
    verification = publish_record(stale.root, layout.attempt_records / terminal.attempt_id)
    published = build_module._transition(
        stale.root,
        terminal,
        AttemptState.RECORD_PUBLISHED,
        changes={"record_sha256": verification.record_sha256},
    )
    corrupted = published.model_copy(update={"record_sha256": "record-sha256:" + "00" * 32})
    build_module._atomic_model(stale.root / "current.json", corrupted)

    with (
        pytest.raises(BuildStateError, match="digest changed"),
        begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(34)),
    ):
        pytest.fail("recovery must trust the journaled digest")


def test_recovery_journals_teardown_interrupted_after_root_removal(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    stale = build_module._allocate_attempt(layout, _RECIPE, _nonce(15))
    stale.registry = stale.registry.model_copy(update={"owner": _stale_owner(stale.registry.owner)})
    build_module._atomic_model(stale.root / "current.json", stale.registry)
    terminal = build_module._transition(
        stale.root,
        stale.registry,
        AttemptState.FINALIZING,
        changes={"outcome": AttemptOutcome.FAILED},
    )
    verification = publish_record(stale.root, layout.attempt_records / terminal.attempt_id)
    published = build_module._transition(
        stale.root,
        terminal,
        AttemptState.RECORD_PUBLISHED,
        changes={"record_sha256": verification.record_sha256},
    )
    build_module._store_recipe_entry(
        layout,
        _RECIPE,
        build_module.AttemptIndexEntryV1(
            attempt_id=published.attempt_id,
            state=AttemptState.RECORD_PUBLISHED,
            outcome=AttemptOutcome.FAILED,
            record_sha256=verification.record_sha256,
        ),
    )
    shutil.rmtree(stale.root)

    with begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(16)) as fresh:
        fresh.finalize(AttemptOutcome.FAILED)

    index = inspect_recipe(_RECIPE, home=home)
    assert index.attempts[0].state is AttemptState.TORN_DOWN


def test_recipe_inspection_rejects_record_tampering(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(10)) as attempt:
        result = attempt.finalize(AttemptOutcome.FAILED)
    (result.record / "current.json").write_bytes(b"{}\n")

    with pytest.raises(BuildStateError, match="digest"):
        inspect_recipe(_RECIPE, home=home)


def test_recipe_inspection_rejects_active_event_chain_corruption(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    active = build_module._allocate_attempt(layout, _RECIPE, _nonce(11))
    event = active.root / "events" / "0001.json"
    event.write_text(
        event.read_text(encoding="utf-8").replace('"sequence": 1', '"sequence": 2'),
        encoding="utf-8",
    )

    with pytest.raises(BuildStateError, match="event chain"):
        inspect_recipe(_RECIPE, home=home)


def test_event_publication_never_exposes_the_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "attempt"
    (root / "events").mkdir(parents=True)

    def fail_publication(_source: Path, _destination: Path) -> None:
        raise OSError("injected rename failure")

    monkeypatch.setattr(build_module, "rename_noreplace", fail_publication)
    with pytest.raises(OSError, match="injected"):
        build_module._write_event(
            root,
            "attempt-" + "00" * 12 + "-" + "11" * 16,
            1,
            "now",
            None,
            AttemptState.ALLOCATED,
        )

    assert not (root / "events" / "0001.json").exists()
    assert not tuple((root / "events").iterdir())


def test_control_plane_rejects_invalid_identifiers_nonces_and_paths(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with (
        pytest.raises(ValueError, match="recipe ID"),
        begin_build_attempt("recipe-sha256:00", home=home),
    ):
        pytest.fail("invalid recipe must fail before allocation")
    with pytest.raises(BuildStateError, match="does not exist"):
        inspect_recipe(_RECIPE, home=tmp_path / "missing-home")

    layout = build_module._layout(home, create=True)
    with pytest.raises(ValueError, match="nonce factory"):
        build_module._allocate_attempt(layout, _RECIPE, lambda: b"short")
    active = build_module._allocate_attempt(layout, _RECIPE, _nonce(12))
    with pytest.raises(BuildStateError, match="identifier collision"):
        build_module._allocate_attempt(layout, _RECIPE, _nonce(12))

    for path in (
        "",
        "/absolute",
        "../escape",
        "events/0002.json",
        "current.json",
        "terminal.json",
        "logs/control\nname",
    ):
        with pytest.raises(BuildStateError, match="unsafe attempt-relative"):
            active.write_evidence(path, b"unsafe")


def test_attempt_finalization_rejects_invalid_reuse_and_build_ids(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(13)) as attempt:
        with pytest.raises(ValueError, match="machine-local build ID"):
            attempt.finalize(AttemptOutcome.FAILED, build_id="build-sha256:00")
        result = attempt.finalize(AttemptOutcome.FAILED)
        with pytest.raises(BuildStateError, match="already finalized"):
            attempt.finalize(AttemptOutcome.FAILED)

    assert result.outcome is AttemptOutcome.FAILED


def test_escaping_invalid_finalization_still_records_a_failed_attempt(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with (
        pytest.raises(ValueError, match="machine-local build ID"),
        begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(29)) as attempt,
    ):
        identifier = attempt.registry.attempt_id
        attempt.finalize(AttemptOutcome.FAILED, build_id="build-sha256:00")

    index = inspect_recipe(_RECIPE, home=home)
    assert index.attempts[0].attempt_id == identifier
    assert index.attempts[0].outcome is AttemptOutcome.FAILED


def test_attempt_rejects_repeated_state_transitions(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(30)) as attempt:
        attempt.mark_active()
        with pytest.raises(BuildStateError, match="illegal build attempt transition"):
            attempt.mark_active()
        attempt.finalize(AttemptOutcome.FAILED)


def test_recipe_inspection_rejects_missing_active_root(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    active = build_module._allocate_attempt(layout, _RECIPE, _nonce(14))
    shutil.rmtree(active.root)

    with pytest.raises(BuildStateError, match="no mutable attempt root"):
        inspect_recipe(_RECIPE, home=home)


def test_layout_rejects_relative_and_symlinked_homes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        build_module._layout(Path("relative"), create=True)
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "home-link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(BuildStateError, match="symbolic link"):
        build_module._layout(link, create=True)


def test_owner_liveness_handles_a_reused_or_missing_process() -> None:
    owner = build_module._current_owner()
    missing = owner.model_copy(update={"pid": 2_000_000_000})

    assert not build_module._owner_alive(missing)


def test_recipe_inspection_rejects_index_digest_and_binding_corruption(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(17)) as attempt:
        attempt.finalize(AttemptOutcome.FAILED)
    layout = build_module._layout(home, create=False)
    index = build_module._load_recipe_index(layout, _RECIPE)
    entry = index.attempts[0]

    build_module._atomic_model(
        build_module._recipe_index_path(layout, _RECIPE),
        index.model_copy(
            update={
                "attempts": (
                    entry.model_copy(update={"record_sha256": "record-sha256:" + "00" * 32}),
                )
            }
        ),
    )
    with pytest.raises(BuildStateError, match="does not match"):
        inspect_recipe(_RECIPE, home=home)

    build_module._atomic_model(
        build_module._recipe_index_path(layout, _RECIPE),
        index.model_copy(
            update={"attempts": (entry.model_copy(update={"outcome": AttemptOutcome.INTERRUPTED}),)}
        ),
    )
    with pytest.raises(BuildStateError, match="not bound"):
        inspect_recipe(_RECIPE, home=home)


def test_recovery_rejects_unsafe_attempt_directory_entries(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    (layout.attempts / "hostile").symlink_to(tmp_path)

    with (
        pytest.raises(BuildStateError, match="unsafe build attempt entry"),
        begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(18)),
    ):
        pytest.fail("unsafe attempt entries must block recovery")


def test_recovery_tolerates_an_orphan_removed_by_another_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    vanished = layout.attempts / build_module.attempt_id(_OTHER_RECIPE, b"\xfb" * 16)
    vanished.mkdir()
    original_lstat = Path.lstat

    def remove_before_lstat(path: Path):
        if path == vanished and path.exists():
            path.rmdir()
            raise FileNotFoundError(path)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", remove_before_lstat)

    assert build_module._recover_stale_attempts(layout, _RECIPE) == ()


def test_layout_rejects_unsafe_state_and_reconciles_a_missing_event(tmp_path: Path) -> None:
    unsafe_home = tmp_path / "unsafe-home"
    unsafe_home.mkdir()
    (unsafe_home / "builds").write_text("not a directory", encoding="utf-8")
    with pytest.raises(BuildStateError, match="path is unsafe"):
        build_module._layout(unsafe_home, create=True)

    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    active = build_module._allocate_attempt(layout, _RECIPE, _nonce(19))
    (active.root / "events" / "0001.json").unlink()
    assert inspect_recipe(_RECIPE, home=home).attempts[0].attempt_id == active.registry.attempt_id
    assert (active.root / "events" / "0001.json").is_file()

    active.mark_active()
    (active.root / "events" / "0002.json").unlink()
    inspected = inspect_recipe(_RECIPE, home=home)
    assert inspected.attempts[0].state is AttemptState.ACTIVE
    assert (active.root / "events" / "0002.json").is_file()


def test_recovery_removes_an_orphaned_preallocation_directory(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    orphan = layout.attempts / build_module.attempt_id(_RECIPE, b"\xfe" * 16)
    (orphan / "events").mkdir(parents=True)

    with begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(26)) as attempt:
        attempt.finalize(AttemptOutcome.FAILED)

    assert not orphan.exists()


def test_preallocation_cleanup_preserves_live_and_unrecognized_roots(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    paused = layout.attempts / build_module.attempt_id(_OTHER_RECIPE, b"\xfd" * 16)
    (paused / "events").mkdir(parents=True)
    unrelated = layout.attempts / build_module.attempt_id(_OTHER_RECIPE, b"\xfc" * 16)
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("not allocator state", encoding="utf-8")
    allocation_lock = layout.locks / "build-attempt-allocation.lock"

    with build_module.exclusive_lock(allocation_lock) as lock:
        assert lock.acquired
        with (
            pytest.raises(BuildBusyError, match="held by another process"),
            begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(28)),
        ):
            pytest.fail("a concurrent recipe allocator must retain the global lock")

    assert paused.is_dir()
    with begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(28)) as attempt:
        attempt.finalize(AttemptOutcome.FAILED)
    assert not paused.exists()
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "not allocator state"


def test_recipe_inspection_observes_the_recipe_lock(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    build_module._allocate_attempt(layout, _RECIPE, _nonce(27))
    lock_path = layout.locks / f"build-recipe-{_RECIPE.removeprefix('recipe-sha256:')}.lock"

    with build_module.exclusive_lock(lock_path) as lock:
        assert lock.acquired
        with pytest.raises(BuildBusyError, match="held by another process"):
            inspect_recipe(_RECIPE, home=home)


def test_attempt_evidence_cannot_escape_through_a_symlinked_parent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    active = build_module._allocate_attempt(layout, _RECIPE, _nonce(20))
    outside = tmp_path / "outside"
    outside.mkdir()
    (active.root / "logs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OSError):
        active.write_evidence("logs/stdout", b"must stay inside")
    assert not (outside / "stdout").exists()


def test_recovery_rejects_a_divergent_existing_record_collision(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    stale = build_module._allocate_attempt(layout, _RECIPE, _nonce(21))
    stale.registry = stale.registry.model_copy(update={"owner": _stale_owner(stale.registry.owner)})
    build_module._atomic_model(stale.root / "current.json", stale.registry)
    terminal = build_module._transition(
        stale.root,
        stale.registry,
        AttemptState.FINALIZING,
        changes={"outcome": AttemptOutcome.FAILED},
    )
    publish_record(stale.root, layout.attempt_records / terminal.attempt_id)
    stale.write_evidence("late-evidence", b"divergent")

    with (
        pytest.raises(BuildStateError, match="divergent immutable"),
        begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(22)),
    ):
        pytest.fail("divergent collision must block recovery")


def test_recipe_index_rejects_duplicate_attempts_and_nonce_drift(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    active = build_module._allocate_attempt(layout, _RECIPE, _nonce(23))
    index = build_module._load_recipe_index(layout, _RECIPE)
    build_module._atomic_model(
        build_module._recipe_index_path(layout, _RECIPE),
        index.model_copy(update={"attempts": (index.attempts[0], index.attempts[0])}),
    )
    with pytest.raises(BuildStateError, match="duplicate attempts"):
        inspect_recipe(_RECIPE, home=home)

    build_module._atomic_model(build_module._recipe_index_path(layout, _RECIPE), index)
    active.registry = active.registry.model_copy(update={"nonce": "00" * 16})
    build_module._atomic_model(active.root / "current.json", active.registry)
    with pytest.raises(BuildStateError, match="not derived"):
        inspect_recipe(_RECIPE, home=home)


def test_active_registry_rejects_terminal_fields(tmp_path: Path) -> None:
    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    active = build_module._allocate_attempt(layout, _RECIPE, _nonce(35))

    with_outcome = active.registry.model_copy(update={"outcome": AttemptOutcome.FAILED})
    with pytest.raises(BuildStateError, match="state/outcome"):
        build_module._validate_attempt_root(active.root, with_outcome)

    with_digest = active.registry.model_copy(update={"record_sha256": "record-sha256:" + "00" * 32})
    with pytest.raises(BuildStateError, match="record cardinality"):
        build_module._validate_attempt_root(active.root, with_digest)


def test_partial_finalization_does_not_recreate_a_removed_attempt_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    original_store = build_module._store_recipe_entry

    def fail_torn_down(layout, recipe_id, entry):
        if entry.state is AttemptState.TORN_DOWN:
            raise OSError("injected index failure")
        original_store(layout, recipe_id, entry)

    with begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(24)) as attempt:
        identifier = attempt.registry.attempt_id
        with monkeypatch.context() as patch:
            patch.setattr(build_module, "_store_recipe_entry", fail_torn_down)
            with pytest.raises(OSError, match="injected"):
                attempt.finalize(AttemptOutcome.FAILED)

    assert not (home / "builds" / "attempts" / identifier).exists()
    with begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(25)) as retry:
        retry.finalize(AttemptOutcome.FAILED)
    assert inspect_recipe(_RECIPE, home=home).attempts[0].state is AttemptState.TORN_DOWN


@pytest.mark.parametrize(
    "failed_state",
    [
        AttemptState.FINALIZING,
        AttemptState.RECORD_PUBLISHED,
        AttemptState.TORN_DOWN,
    ],
)
def test_context_recovers_each_durable_finalization_boundary_in_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_state: AttemptState,
) -> None:
    home = tmp_path / "home"
    original_store = build_module._store_recipe_entry
    injected = False

    def fail_once(layout, recipe_id, entry):
        nonlocal injected
        if entry.state is failed_state and not injected:
            injected = True
            raise OSError(f"injected {failed_state} failure")
        original_store(layout, recipe_id, entry)

    monkeypatch.setattr(build_module, "_store_recipe_entry", fail_once)
    with (
        pytest.raises(OSError, match="injected"),
        begin_build_attempt(_RECIPE, home=home, nonce_factory=_nonce(36)) as attempt,
    ):
        identifier = attempt.registry.attempt_id
        attempt.finalize(AttemptOutcome.FAILED)

    entry = inspect_recipe(_RECIPE, home=home).attempts[0]
    assert injected
    assert entry.attempt_id == identifier
    assert entry.state is AttemptState.TORN_DOWN


def test_event_chain_rejects_coherent_illegal_edges_in_mutable_and_recorded_state(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    layout = build_module._layout(home, create=True)
    active = build_module._allocate_attempt(layout, _RECIPE, _nonce(37))
    active.mark_active()
    first_event = active.root / "events" / "0001.json"
    event = json.loads(first_event.read_bytes())
    event["to"] = AttemptState.ACTIVE
    first_event.write_bytes(build_module.canonical_json_bytes(event))
    (active.root / "events" / "0002.json").unlink()

    with pytest.raises(BuildStateError, match="event chain"):
        inspect_recipe(_RECIPE, home=home)

    record = layout.attempt_records / active.registry.attempt_id
    publish_record(active.root, record)
    recorded = build_module._read_model(record / "current.json", build_module.AttemptRegistryV1)
    with pytest.raises(BuildStateError, match="event chain"):
        build_module._validate_event_chain(record, recorded)

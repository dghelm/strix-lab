from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

import strixlab.evidence as ev
from strixlab.secure_fs import write_exclusive

_ENV = {"PATH": "/usr/bin", "API_TOKEN": "supersecret-value"}
_FIXED = datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)


def _begin(home: Path, *, experiment: str = "exp-demo", resolved: dict | None = None, **kw):
    return ev.begin_run(
        experiment,
        kw.pop("manifest_input", b"suite: demo\n"),
        resolved=resolved if resolved is not None else {"a": 1, "b": [1, 2]},
        home=home,
        environ=kw.pop("environ", _ENV),
        **kw,
    )


def _tokens(*values: bytes):
    iterator = iter(values)
    return lambda: next(iterator)


# --------------------------------------------------------------------- allocation


def test_allocation_produces_sortable_unique_ids_and_captures_manifests(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home, clock=lambda: _FIXED) as run:
        run_id = run.run_id
        assert run_id.startswith("run-20260830T120000Z-exp-demo-")
        record_input = (run.active / "manifest.input.yaml").read_bytes()
        record_resolved = (run.active / "manifest.resolved.yaml").read_bytes()
        assert record_input == b"suite: demo\n"
        # deterministic sorted-key YAML with one trailing newline
        assert record_resolved == b"a: 1\nb:\n- 1\n- 2\n"
        descriptor = ev._read_model(run.active / "run.json", ev.RunDescriptorV1)
        assert descriptor.input_manifest_sha256 == hashlib.sha256(record_input).hexdigest()
        run.succeed()


def test_two_runs_get_distinct_ids(tmp_path: Path) -> None:
    home = tmp_path / "home"
    ids = set()
    for _ in range(3):
        with _begin(home) as run:
            ids.add(run.run_id)
            run.succeed()
    assert len(ids) == 3


def test_id_collision_retries_a_new_random_suffix(tmp_path: Path) -> None:
    home = tmp_path / "home"
    first = _tokens(b"\x01" * 16)
    with _begin(home, clock=lambda: _FIXED, token_factory=first) as run:
        taken = run.run_id
        run.succeed()
    # A colliding first token forces a retry to the second, distinct suffix.
    collide = _tokens(b"\x01" * 16, b"\x02" * 16)
    with _begin(home, clock=lambda: _FIXED, token_factory=collide) as run:
        assert run.run_id != taken
        assert run.run_id.endswith("02" * 16)
        run.succeed()


def test_stage_rename_collision_retries_then_exhausts(tmp_path: Path, monkeypatch) -> None:
    # If the staged tree's no-replace publish keeps losing the race, allocation
    # discards the stage and retries a fresh ID until it exhausts its budget.
    def collide(_source, _destination):
        raise FileExistsError()

    monkeypatch.setattr(ev, "rename_noreplace", collide)
    with pytest.raises(ev.RunError, match="exhausted"):
        _begin(tmp_path / "home", clock=lambda: _FIXED)


def test_allocation_exhausts_retries_when_every_id_collides(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home, clock=lambda: _FIXED, token_factory=lambda: b"\x03" * 16) as run:
        run.succeed()
    with pytest.raises(ev.RunError, match="exhausted"):
        _begin(home, clock=lambda: _FIXED, token_factory=lambda: b"\x03" * 16)


def test_invalid_experiment_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ev.RunError, match="experiment id"):
        _begin(tmp_path / "home", experiment="Not_A_DashId")


def test_resolved_manifest_sensitive_interpolation_is_rejected(tmp_path: Path) -> None:
    # A literal ${API_TOKEN} reference passes secret-value scanning but must be
    # rejected as a live interpolation of a sensitive environment name.
    with pytest.raises(ev.RunError, match="sensitive environment"):
        _begin(tmp_path / "home", resolved={"cmd": "${API_TOKEN}"})


# ---------------------------------------------------------------------- lifecycle


def test_success_and_failure_records_both_verify_and_are_immutable(tmp_path: Path) -> None:
    for outcome in ("success", "failure"):
        home = tmp_path / outcome
        with _begin(home) as run:
            run_id = run.run_id
            run.write_evidence("logs/out.txt", b"payload\n")
            inspection = run.succeed() if outcome == "success" else run.fail("declared failure")
        assert str(inspection.outcome) == outcome
        again = ev.inspect_run(run_id, home=home)
        assert again.record_sha256 == inspection.record_sha256
        # record is immutable: republication over it fails no-replace
        from strixlab.records import RecordError, publish_record

        with pytest.raises((RecordError, OSError)):
            publish_record(again.record, again.record)


def test_context_exit_without_outcome_finalizes_failure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home) as run:
        run_id = run.run_id
    inspection = ev.inspect_run(run_id, home=home)
    assert inspection.outcome is ev.RunOutcome.FAILURE
    status = ev._read_model(inspection.record / "status.json", ev.RunStatusV1)
    assert status.reason == "run-context-exited-without-outcome"


def test_escaping_exception_finalizes_failure_and_is_reraised(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with pytest.raises(ValueError, match="boom"), _begin(home) as run:
        run_id = run.run_id
        raise ValueError("boom")
    status = ev._read_model(
        ev.inspect_run(run_id, home=home).record / "status.json", ev.RunStatusV1
    )
    assert status.outcome is ev.RunOutcome.FAILURE
    assert status.reason == "boom"


def test_sensitive_exception_text_is_replaced_with_fixed_reason(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with pytest.raises(RuntimeError), _begin(home) as run:
        run_id = run.run_id
        raise RuntimeError("leak supersecret-value now")
    record = ev.inspect_run(run_id, home=home).record
    status = ev._read_model(record / "status.json", ev.RunStatusV1)
    assert status.reason == "run-failed-with-sensitive-error"
    assert b"supersecret-value" not in (record / "status.json").read_bytes()


def test_double_finalization_is_rejected(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home) as run:
        run.succeed()
        with pytest.raises(ev.RunError, match="already finalized"):
            run.fail("late")


# --------------------------------------------------------------- evidence writing


def test_write_evidence_rejects_unsafe_and_reserved_paths(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home) as run:
        for bad in [
            "../escape",
            "/abs",
            "a\\b",
            "run.json",
            ".notes.tmp",
            "events/x",
            "portable/blobs/x",
            "a//b",
        ]:
            with pytest.raises(ev.RunError):
                run.write_evidence(bad, b"x")
        run.write_evidence("logs/a.txt", b"ok\n")
        with pytest.raises(ev.RunError, match="already exists"):
            run.write_evidence("logs/a.txt", b"again\n")
        run.succeed()


def test_write_evidence_rejects_symlinked_intermediate_parent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    outside = tmp_path / "outside"
    outside.mkdir()
    with _begin(home) as run:
        # An attacker swaps an intermediate parent directory for a symlink; the
        # descriptor-anchored, no-follow descent must refuse to traverse it.
        (run.active / "logs").symlink_to(outside)
        with pytest.raises(ev.RunError, match="run subdirectory is unavailable"):
            run.write_evidence("logs/x.txt", b"data\n")
        (run.active / "logs").unlink()
        run.fail("cleanup")


def test_write_portable_rejects_symlinked_portable_parent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    outside = tmp_path / "outside"
    outside.mkdir()
    with _begin(home) as run:
        (run.active / "portable").symlink_to(outside)
        with pytest.raises(ev.RunError, match="unavailable|unsafe run subdirectory"):
            run.write_portable(
                "a.json", b'{"k":1}\n', media_type="application/json", role="summary"
            )
        (run.active / "portable").unlink()
        run.fail("cleanup")


def test_write_evidence_rejects_non_directory_intermediate(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home) as run:
        # A regular file where a parent directory is expected: the no-follow open of
        # the intermediate as a directory fails closed rather than clobbering it.
        write_exclusive(run.active / "logs", b"not a dir\n", 0o600)
        with pytest.raises(ev.RunError, match="run subdirectory is unavailable"):
            run.write_evidence("logs/x.txt", b"data\n")
        (run.active / "logs").unlink()
        run.fail("cleanup")


def test_open_owned_directory_missing_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ev.RunError, match="run directory is unavailable"):
        ev._open_owned_directory_fd(tmp_path / "does-not-exist")


def test_committed_event_with_wrong_mode_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    (run.active / "events" / "00000001.json").chmod(0o644)
    with pytest.raises(ev.RunError, match="unsafe run file"):
        ev._required_status(run.active, run.run_id)
    run._stack.close()


def test_recover_rejects_cross_run_active_descriptor(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    run_id = run.run_id
    # Forge run.json to name a different run (a cross-run replacement).
    descriptor = ev._read_model(run.active / "run.json", ev.RunDescriptorV1)
    other = "run-20260830T120000Z-exp-x-" + "0" * 32
    forged = descriptor.model_copy(update={"run_id": other})
    ev._atomic_write(
        run.active / "run.json", ev.canonical_json_bytes(forged.model_dump(mode="json"))
    )
    run._stack.close()
    with pytest.raises(ev.RunError, match="active run root diverged from its descriptor"):
        ev.recover_run(run_id, home=home)


def test_recover_rejects_active_root_inode_mismatch(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    run_id = run.run_id
    descriptor = ev._read_model(run.active / "run.json", ev.RunDescriptorV1)
    forged = descriptor.model_copy(update={"stage_inode": descriptor.stage_inode + 1})
    ev._atomic_write(
        run.active / "run.json", ev.canonical_json_bytes(forged.model_dump(mode="json"))
    )
    run._stack.close()
    with pytest.raises(ev.RunError, match="active run root diverged from its descriptor"):
        ev.recover_run(run_id, home=home)


def test_recover_rejects_status_bound_to_another_run(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    run_id = run.run_id
    status = ev._read_model(run.active / "status.json", ev.RunStatusV1)
    other = "run-20260830T120000Z-exp-x-" + "0" * 32
    forged = status.model_copy(update={"run_id": other})
    ev._atomic_write(
        run.active / "status.json", ev.canonical_json_bytes(forged.model_dump(mode="json"))
    )
    run._stack.close()
    with pytest.raises(ev.RunError, match="run status is bound to another run"):
        ev.recover_run(run_id, home=home)


def test_read_owned_regular_at_missing_fails_closed(tmp_path: Path) -> None:
    fd = os.open(tmp_path, ev._DIR_OPEN_FLAGS)
    try:
        with pytest.raises(ev.RunError, match="run file is unavailable"):
            ev._read_owned_regular_at(fd, "nope")
    finally:
        os.close(fd)


def test_try_open_owned_child_dir_on_regular_file_fails_closed(tmp_path: Path) -> None:
    write_exclusive(tmp_path / "afile", b"x\n", 0o600)
    fd = os.open(tmp_path, ev._DIR_OPEN_FLAGS)
    try:
        with pytest.raises(ev.RunError, match="run subdirectory is unavailable"):
            ev._try_open_owned_child_dir(fd, "afile")
    finally:
        os.close(fd)


def test_multiple_blob_writer_temps_fail_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    blobs = _blobs_dir(run)
    write_exclusive(blobs / f".{'a' * 64}.{'0' * 16}.tmp", b"x\n", 0o600)
    write_exclusive(blobs / f".{'b' * 64}.{'1' * 16}.tmp", b"y\n", 0o600)
    with pytest.raises(ev.RunError, match="multiple writer temp portable blobs"):
        ev._recover_portable(run.active)
    run._stack.close()


def test_write_portable_rejects_reserved_control_logical_path(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home) as run:
        for reserved in ["status.json", "events/x", "portable/blobs/y", "checksums.sha256"]:
            with pytest.raises(ev.RunError, match="reserved control path"):
                run.write_portable(
                    reserved, b'{"k":1}\n', media_type="application/json", role="summary"
                )
        run.succeed()


def test_write_portable_enforces_aggregate_limit(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(ev, "_MAX_AGGREGATE_BYTES", 8)
    with _begin(home) as run:
        with pytest.raises(ev.RunError, match="aggregate payload limit"):
            run.write_portable(
                "a.json", b'{"k":123456}\n', media_type="application/json", role="summary"
            )
        run.fail("cleanup")


def test_write_portable_enforces_total_file_limit(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(ev, "_MAX_TOTAL_FILES", 1)
    with _begin(home) as run:
        with pytest.raises(ev.RunError, match="total-file limit"):
            run.write_portable(
                "a.json", b'{"k":1}\n', media_type="application/json", role="summary"
            )
        run.fail("cleanup")


def test_write_evidence_rejects_secret_payload_and_after_terminal(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home) as run:
        with pytest.raises(ev.RunError):
            run.write_evidence("logs/leak.txt", b"supersecret-value")
        run.succeed()
        with pytest.raises(ev.RunError, match="after the run terminates"):
            run.write_evidence("logs/late.txt", b"x")


def test_overlong_and_control_paths_rejected() -> None:
    with pytest.raises(ev.RunError):
        ev.run_relative("a/" + "x" * 300)
    with pytest.raises(ev.RunError):
        ev.run_relative("a\nb")


# ----------------------------------------------------------------------- portable


def test_portable_classification_dedup_and_unique_paths(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home) as run:
        one = run.write_portable(
            "a.json", b'{"k":1}\n', media_type="application/json", role="summary"
        )
        two = run.write_portable(
            "b.json", b'{"k":1}\n', media_type="application/json", role="build"
        )
        assert one.blob_sha256 == two.blob_sha256  # deduplicated blob
        assert (one.sequence, two.sequence) == (1, 2)
        with pytest.raises(ev.RunError, match="duplicate portable logical path"):
            run.write_portable("a.json", b"other\n", media_type="text/plain", role="summary")
        run.succeed()
    blobs = list((ev.inspect_run(run.run_id, home=home).record / "portable" / "blobs").iterdir())
    assert len(blobs) == 1


def test_portable_accepts_every_policy_media_type(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cases = [
        ("a.json", b'{"k":1}\n', "application/json", "environment"),
        ("a.ndjson", b'{"k":1}\n{"k":2}\n\n', "application/x-ndjson", "samples"),
        ("a.yaml", b"k: 1\n", "application/yaml", "source"),
        ("a.txt", b"plain\n", "text/plain", "summary"),
        ("a.csv", b"h,v\n1,2\n", "text/csv", "comparison"),
        ("a.md", b"# t\n", "text/markdown", "profiler-summary"),
        ("a.diff", b"--- a\n+++ b\n", "text/x-diff", "correctness"),
    ]
    with _begin(home) as run:
        for path, payload, media, role in cases:
            run.write_portable(path, payload, media_type=media, role=role)
        run.succeed()
    inspection = ev.inspect_run(run.run_id, home=home)
    entries = list((inspection.record / "portable" / "entries").iterdir())
    assert len(entries) == len(cases)


def test_portable_rejects_malformed_ndjson_and_yaml(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home) as run:
        with pytest.raises(ev.RunError, match="application/x-ndjson"):
            run.write_portable(
                "x.ndjson", b"{not json}\n", media_type="application/x-ndjson", role="samples"
            )
        with pytest.raises(ev.RunError, match="application/yaml"):
            run.write_portable(
                "x.yaml", b"k: [unterminated\n", media_type="application/yaml", role="source"
            )
        run.succeed()


def test_portable_shared_blob_conflicting_media_rejected(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home) as run:
        # Identical bytes deduplicate to one blob; a second reference under a
        # different media type is rejected at write time.
        run.write_portable("a.txt", b"same bytes\n", media_type="text/plain", role="summary")
        with pytest.raises(ev.RunError, match="shared under conflicting media types"):
            run.write_portable("b.md", b"same bytes\n", media_type="text/markdown", role="summary")
        run.succeed()


def test_portable_rejects_bad_role_media_and_structure(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home) as run:
        with pytest.raises(ev.RunError, match="role"):
            run.write_portable("x", b"{}", media_type="application/json", role="model")
        with pytest.raises(ev.RunError, match="media type"):
            run.write_portable("x", b"{}", media_type="application/octet-stream", role="summary")
        with pytest.raises(ev.RunError, match="valid application/json"):
            run.write_portable("x.json", b"not json", media_type="application/json", role="summary")
        with pytest.raises(ev.RunError, match="UTF-8"):
            run.write_portable("x", b"\xff\xfe", media_type="text/plain", role="summary")
        with pytest.raises(ev.RunError, match="control bytes"):
            run.write_portable("x", b"a\x00b", media_type="text/plain", role="summary")
        run.succeed()


def test_orphan_portable_blob_is_cleaned_and_not_exported(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home) as run:
        run.write_portable("a.json", b'{"k":1}\n', media_type="application/json", role="summary")
        blobs = run.active / "portable" / "blobs"
        orphan = hashlib.sha256(b"orphan").hexdigest()
        (blobs / orphan).write_bytes(b"orphan")
        (blobs / orphan).chmod(0o600)
        run.succeed()
    record = ev.inspect_run(run.run_id, home=home).record
    assert not (record / "portable" / "blobs" / orphan).exists()


# ---------------------------------------------------------------------- checksums


def test_checksums_cover_all_files_except_itself_in_canonical_order(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home) as run:
        run.write_evidence("logs/z.txt", b"z\n")
        run.write_evidence("logs/a.txt", b"a\n")
        run.succeed()
    record = ev.inspect_run(run.run_id, home=home).record
    lines = (record / "checksums.sha256").read_text().splitlines()
    paths = [line[66:] for line in lines]
    assert "checksums.sha256" not in paths
    assert paths == sorted(paths)
    parsed = ev.parse_checksums((record / "checksums.sha256").read_bytes())
    assert parsed["logs/a.txt"] == hashlib.sha256(b"a\n").hexdigest()


def test_parse_checksums_rejects_noncanonical_input() -> None:
    with pytest.raises(ev.RunError):
        ev.parse_checksums(b"deadbeef  x\n")  # short digest
    with pytest.raises(ev.RunError, match="trailing newline"):
        ev.parse_checksums(b"%s  a\n%s  b" % (b"0" * 64, b"1" * 64))
    with pytest.raises(ev.RunError, match="byte order"):
        ev.parse_checksums(b"%s  b\n%s  a\n" % (b"0" * 64, b"1" * 64))


# ----------------------------------------------------------------------- recovery


def test_live_owner_is_reported_busy(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home) as run:
        outcome = ev.recover_run(run.run_id, home=home)
        assert isinstance(outcome, ev.RunBusy)
        run.succeed()


def test_dead_owner_is_finalized_as_interrupted(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    run_id = run.run_id
    # Simulate a crash: forge a dead owner in the descriptor and drop the lock.
    descriptor = ev._read_model(run.active / "run.json", ev.RunDescriptorV1)
    dead = descriptor.model_copy(
        update={"owner": descriptor.owner.model_copy(update={"boot_id": "dead"})}
    )
    ev._atomic_write(run.active / "run.json", ev.canonical_json_bytes(dead.model_dump(mode="json")))
    run._stack.close()  # release lock without finalizing
    inspection = ev.recover_run(run_id, home=home)
    assert isinstance(inspection, ev.RunInspection)
    assert inspection.outcome is ev.RunOutcome.INTERRUPTED


def test_recover_runs_recovers_dead_and_skips_finalized(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home, experiment="exp-one") as run:
        run.succeed()
    dead = _begin(home, experiment="exp-two")
    descriptor = ev._read_model(dead.active / "run.json", ev.RunDescriptorV1)
    forged = descriptor.model_copy(
        update={"owner": descriptor.owner.model_copy(update={"boot_id": "x"})}
    )
    ev._atomic_write(
        dead.active / "run.json", ev.canonical_json_bytes(forged.model_dump(mode="json"))
    )
    dead._stack.close()
    recovered = ev.recover_runs(home=home)
    assert any(item.outcome is ev.RunOutcome.INTERRUPTED for item in recovered)


def test_inspect_run_on_busy_run_raises(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home) as run:
        with pytest.raises(ev.RunError, match="busy"):
            ev.inspect_run(run.run_id, home=home)
        run.succeed()


def test_inspect_unknown_run_raises(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(parents=True)
    with pytest.raises(ev.RunError):
        ev.inspect_run("run-20260830T120000Z-exp-x-" + "0" * 32, home=home)


# ------------------------------------------------------------- crash injection


def test_recovery_completes_after_crash_before_record(tmp_path: Path) -> None:
    # Terminal event recorded but no record/index yet: recovery is crash-forward.
    home = tmp_path / "home"
    run = _begin(home)
    run_id = run.run_id
    status = ev._required_status(run.active, run_id)
    ev._transition(
        run.active,
        status,
        ev.RunState.TERMINAL,
        run_id=run_id,
        clock=lambda: _FIXED,
        outcome=ev.RunOutcome.SUCCESS,
    )
    run._stack.close()
    inspection = ev.recover_run(run_id, home=home)
    assert isinstance(inspection, ev.RunInspection)
    assert inspection.outcome is ev.RunOutcome.SUCCESS
    assert not run.active.exists()


def test_recovery_completes_after_crash_before_index(tmp_path: Path, monkeypatch) -> None:
    # Crash after record publication but before the index: recovery finishes it.
    home = tmp_path / "home"
    run = _begin(home)
    run_id = run.run_id
    real_publish = ev._publish_index

    def boom(*a, **k):
        raise OSError("crash before index")

    monkeypatch.setattr(ev, "_publish_index", boom)
    with pytest.raises(OSError, match="crash before index"):
        run.succeed()
    monkeypatch.setattr(ev, "_publish_index", real_publish)
    inspection = ev.recover_run(run_id, home=home)
    assert isinstance(inspection, ev.RunInspection)
    assert ev._index_path(ev._layout(home, create=False), run_id).is_file()


def test_recovery_finishes_teardown_after_index(tmp_path: Path, monkeypatch) -> None:
    # Crash after index publication but before active teardown: recovery finishes it.
    home = tmp_path / "home"
    run = _begin(home)
    run_id = run.run_id

    def boom(*a, **k):
        raise OSError("crash before teardown")

    real = ev._teardown_active
    monkeypatch.setattr(ev, "_teardown_active", boom)
    with pytest.raises(OSError, match="crash before teardown"):
        run.succeed()
    monkeypatch.setattr(ev, "_teardown_active", real)
    inspection = ev.recover_run(run_id, home=home)
    assert isinstance(inspection, ev.RunInspection)
    assert not run.active.exists()


def test_event_model_rejects_nonterminal_outcome_or_reason() -> None:
    import pydantic

    base = {
        "run_id": _STAGE_RUN_ID,
        "sequence": 2,
        "previous_sha256": "a" * 64,
        "from_state": "allocated",
        "to_state": "active",
        "timestamp": "2026-08-30T12:00:00+00:00",
    }
    with pytest.raises(pydantic.ValidationError):
        ev.RunEventV1(**base, outcome=ev.RunOutcome.SUCCESS)
    with pytest.raises(pydantic.ValidationError):
        ev.RunEventV1(**base, reason="should not be here")


def test_event_model_requires_outcome_on_terminal() -> None:
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        ev.RunEventV1(
            run_id=_STAGE_RUN_ID,
            sequence=3,
            previous_sha256="a" * 64,
            from_state=ev.RunState.ACTIVE,
            to_state=ev.RunState.TERMINAL,
            timestamp="2026-08-30T12:00:00+00:00",
            outcome=None,
        )


def test_status_model_rejects_nonterminal_outcome() -> None:
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        ev.RunStatusV1(
            run_id=_STAGE_RUN_ID,
            state=ev.RunState.ACTIVE,
            sequence=2,
            last_event_sha256="a" * 64,
            outcome=ev.RunOutcome.SUCCESS,
        )


def test_recovery_rejects_nonterminal_event_with_forged_outcome(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    # Forge the committed ACTIVE event to carry a terminal-only outcome field.
    active_event = run.active / "events" / "00000002.json"
    raw = json.loads(active_event.read_bytes())
    raw["outcome"] = "success"
    active_event.chmod(0o600)
    ev._atomic_write(active_event, ev.canonical_json_bytes(raw))
    with pytest.raises(ev.RunError, match="stored run model is invalid"):
        ev._required_status(run.active, run.run_id)
    run._stack.close()


def test_event_chain_with_unexpected_entry_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    # A committed event file far beyond the projected sequence: the chain length no
    # longer matches status.json, so verification fails closed.
    (run.active / "events" / "00000099.json").write_bytes(b"{}\n")
    with pytest.raises(ev.RunError, match="unexpected entries"):
        ev._required_status(run.active, run.run_id)
    run._stack.close()


def test_orphan_event_beyond_status_is_adopted(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    run_id = run.run_id
    status = ev._required_status(run.active, run_id)
    # Manually publish the next event (TERMINAL) but do not advance status.json.
    event = ev.RunEventV1(
        run_id=run_id,
        sequence=status.sequence + 1,
        previous_sha256=status.last_event_sha256,
        from_state=status.state,
        to_state=ev.RunState.TERMINAL,
        timestamp=ev._iso(_FIXED),
        outcome=ev.RunOutcome.SUCCESS,
    )
    payload, _digest = ev._event_bytes(event)
    ev._write_no_replace(run.active / "events" / f"{event.sequence:08d}.json", payload, 0o600)
    run._stack.close()
    adopted = ev._required_status(run.active, run_id)
    assert adopted.state is ev.RunState.TERMINAL and adopted.outcome is ev.RunOutcome.SUCCESS


def test_divergent_orphan_event_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    run_id = run.run_id
    status = ev._required_status(run.active, run_id)
    forged = ev.RunEventV1(
        run_id=run_id,
        sequence=status.sequence + 1,
        previous_sha256="f" * 64,  # does not chain
        from_state=status.state,
        to_state=ev.RunState.TERMINAL,
        timestamp=ev._iso(_FIXED),
        outcome=ev.RunOutcome.SUCCESS,
    )
    payload, _digest = ev._event_bytes(forged)
    ev._write_no_replace(run.active / "events" / f"{forged.sequence:08d}.json", payload, 0o600)
    with pytest.raises(ev.RunError, match="divergent orphan"):
        ev._required_status(run.active, run_id)
    run._stack.close()


def test_tampered_checksums_are_rejected_on_recovery(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    run_id = run.run_id
    status = ev._required_status(run.active, run_id)
    ev._transition(
        run.active,
        status,
        ev.RunState.TERMINAL,
        run_id=run_id,
        clock=lambda: _FIXED,
        outcome=ev.RunOutcome.SUCCESS,
    )
    checksums = run.active / "checksums.sha256"
    checksums.write_bytes(b"%s  logs/x\n" % (b"0" * 64))
    checksums.chmod(0o600)
    run._stack.close()
    with pytest.raises(ev.RunError, match="checksums.sha256 diverged"):
        ev.recover_run(run_id, home=home)


# ------------------------------------------------------------- allocation staging


def test_recover_runs_removes_empty_stage_and_rejects_foreign_entry(tmp_path: Path) -> None:
    home = tmp_path / "home"
    roots = ev._layout(home, create=True)
    name = f".run-20260830T120000Z-exp-x-{'0' * 32}.abcdef0123456789.tmp"
    empty = roots.allocation_staging / name
    empty.mkdir(mode=0o700)
    ev.recover_runs(home=home)
    assert not empty.exists()
    (roots.allocation_staging / "unexpected").mkdir()
    with pytest.raises(ev.RunError, match="unexpected"):
        ev.recover_runs(home=home)


_STAGE_RUN_ID = "run-20260830T120000Z-exp-x-" + "0" * 32


def _dead_owner() -> ev.RunOwnerV1:
    # Our own uid, but a boot id that cannot match the live host: a genuine dead
    # owner of ours (a foreign uid is a different case that must fail closed).
    import os

    return ev.RunOwnerV1(pid=1, boot_id="dead-boot", process_start_ticks=1, uid=os.geteuid())


def _stage_with_descriptor(roots, *, stage_inode: int | None = None, owner=None):
    name = f".{_STAGE_RUN_ID}.{'0' * 16}.tmp"
    stage = roots.allocation_staging / name
    stage.mkdir(mode=0o700)
    meta = stage.stat()
    descriptor = ev.RunDescriptorV1(
        run_id=_STAGE_RUN_ID,
        experiment_id="exp-x",
        created_at=ev._iso(_FIXED),
        input_manifest_sha256="0" * 64,
        resolved_manifest_sha256="0" * 64,
        owner=owner if owner is not None else _dead_owner(),
        stage_device=meta.st_dev,
        stage_inode=stage_inode if stage_inode is not None else meta.st_ino,
    )
    ev._atomic_write(
        stage / ev._DESCRIPTOR_NAME,
        ev.canonical_json_bytes(descriptor.model_dump(mode="json")),
    )
    return stage


def test_dead_owner_allocation_stage_is_reclaimed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    roots = ev._layout(home, create=True)
    stage = _stage_with_descriptor(roots)
    ev.recover_runs(home=home)
    assert not stage.exists()


def test_descriptorless_allocation_stage_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    roots = ev._layout(home, create=True)
    name = f".{_STAGE_RUN_ID}.{'0' * 16}.tmp"
    stage = roots.allocation_staging / name
    stage.mkdir(mode=0o700)
    (stage / "stray").write_bytes(b"x\n")
    with pytest.raises(ev.RunError, match="descriptorless"):
        ev.recover_runs(home=home)


def test_allocation_stage_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    roots = ev._layout(home, create=True)
    _stage_with_descriptor(roots, stage_inode=1)
    with pytest.raises(ev.RunError, match="identity mismatch"):
        ev.recover_runs(home=home)


def test_live_owner_allocation_stage_is_preserved(tmp_path: Path) -> None:
    home = tmp_path / "home"
    roots = ev._layout(home, create=True)
    stage = _stage_with_descriptor(roots, owner=ev._current_owner())
    ev.recover_runs(home=home)
    assert stage.exists()


def test_foreign_owner_allocation_stage_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    roots = ev._layout(home, create=True)
    foreign = ev.RunOwnerV1(pid=1, boot_id="dead-boot", process_start_ticks=1, uid=999999)
    _stage_with_descriptor(roots, owner=foreign)
    # A stage descriptor owned by another uid is never reclaimed/deleted as "dead".
    with pytest.raises(ev.RunError, match="foreign-owned run allocation-staging"):
        ev.recover_runs(home=home)


# ----------------------------------------------------- index/record binding


def _finalize(home: Path) -> str:
    with _begin(home) as run:
        run_id = run.run_id
        run.succeed()
    return run_id


def _retamper_index(home: Path, run_id: str, mutate) -> None:
    roots = ev._layout(home, create=False)
    path = ev._index_path(roots, run_id)
    index = json.loads(path.read_bytes())
    mutate(index)
    path.chmod(0o600)
    ev._atomic_write(path, ev.canonical_json_bytes(index))


def test_index_bound_to_another_run_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run_id = _finalize(home)
    _retamper_index(home, run_id, lambda i: i.__setitem__("run_id", _STAGE_RUN_ID))
    with pytest.raises(ev.RunError, match="bound to another run"):
        ev.inspect_run(run_id, home=home)


def test_record_digest_diverged_from_index_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run_id = _finalize(home)
    _retamper_index(
        home, run_id, lambda i: i.__setitem__("record_sha256", "record-sha256:" + "0" * 64)
    )
    with pytest.raises(ev.RunError, match="run record diverged from its index"):
        ev.inspect_run(run_id, home=home)


def test_checksum_digest_diverged_from_index_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run_id = _finalize(home)
    _retamper_index(home, run_id, lambda i: i.__setitem__("checksums_sha256", "0" * 64))
    with pytest.raises(ev.RunError, match="run record diverged from its index"):
        ev.inspect_run(run_id, home=home)


def test_record_without_index_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run_id = _finalize(home)
    roots = ev._layout(home, create=False)
    index = ev._index_path(roots, run_id)
    index.chmod(0o600)
    index.unlink()
    with pytest.raises(ev.RunError, match="without an authenticated index"):
        ev.inspect_run(run_id, home=home)


# --------------------------------------------------- writer-temp reconciliation


def _event_temp(run, sequence: int) -> Path:
    return run.active / "events" / f".{sequence:08d}.json.{'0' * 16}.tmp"


def _next_event_bytes(run, status, *, previous=None, sequence=None) -> bytes:
    event = ev.RunEventV1(
        run_id=run.run_id,
        sequence=status.sequence + 1 if sequence is None else sequence,
        previous_sha256=status.last_event_sha256 if previous is None else previous,
        from_state=status.state,
        to_state=ev.RunState.TERMINAL,
        timestamp=ev._iso(_FIXED),
        outcome=ev.RunOutcome.SUCCESS,
    )
    payload, _digest = ev._event_bytes(event)
    return payload


def test_valid_next_event_temp_is_removed_on_reconcile(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    status = ev._required_status(run.active, run.run_id)
    # A crash left a fsynced-but-unrenamed writer temp for the exact next event.
    write_exclusive(_event_temp(run, status.sequence + 1), _next_event_bytes(run, status), 0o600)
    ev._reconcile_event_temp(run.active, status)
    assert not _event_temp(run, status.sequence + 1).exists()
    run._stack.close()


def test_multiple_event_writer_temps_fail_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    status = ev._required_status(run.active, run.run_id)
    write_exclusive(_event_temp(run, 2), b"a\n", 0o600)
    write_exclusive(_event_temp(run, 3), b"b\n", 0o600)
    with pytest.raises(ev.RunError, match="multiple writer temp"):
        ev._reconcile_event_temp(run.active, status)
    run._stack.close()


def test_unsafe_event_writer_temp_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    status = ev._required_status(run.active, run.run_id)
    temp = _event_temp(run, 2)
    write_exclusive(temp, b"a\n", 0o600)
    temp.chmod(0o644)
    with pytest.raises(ev.RunError, match="unsafe run file"):
        ev._reconcile_event_temp(run.active, status)
    run._stack.close()


def test_uncommitted_event_temp_wrong_sequence_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    status = ev._required_status(run.active, run.run_id)
    # A temp for a sequence far beyond the next, with no committed counterpart.
    write_exclusive(_event_temp(run, 9), _next_event_bytes(run, status, sequence=9), 0o600)
    with pytest.raises(ev.RunError, match="divergent writer temp run event"):
        ev._reconcile_event_temp(run.active, status)
    run._stack.close()


def test_unlinkable_event_temp_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    status = ev._required_status(run.active, run.run_id)
    # Exact next sequence, but its previous digest does not chain onto the status.
    payload = _next_event_bytes(run, status, previous="f" * 64)
    write_exclusive(_event_temp(run, status.sequence + 1), payload, 0o600)
    with pytest.raises(ev.RunError, match="divergent writer temp run event"):
        ev._reconcile_event_temp(run.active, status)
    run._stack.close()


def test_event_temp_conflicting_with_committed_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    status = ev._required_status(run.active, run.run_id)
    # A temp for an already-committed sequence whose bytes differ is a divergence.
    write_exclusive(_event_temp(run, 1), b"different-from-committed\n", 0o600)
    with pytest.raises(ev.RunError, match="divergent writer temp run event"):
        ev._reconcile_event_temp(run.active, status)
    run._stack.close()


def test_event_directory_unexpected_member_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    status = ev._required_status(run.active, run.run_id)
    write_exclusive(run.active / "events" / "stray", b"x\n", 0o600)
    with pytest.raises(ev.RunError, match="unexpected run event directory member"):
        ev._reconcile_event_temp(run.active, status)
    run._stack.close()


def _entries_dir(run) -> Path:
    return run.active / "portable" / "entries"


def _seed_entry(run) -> Path:
    run.write_portable("seed.json", b'{"k":1}\n', media_type="application/json", role="summary")
    return _entries_dir(run) / "00000001.json"


def _next_entry_temp_bytes(run, **overrides) -> bytes:
    seed = ev._read_model(_seed_entry(run), ev.PortableEvidenceV1)
    update = {"sequence": 2, "logical_path": "second.json"}
    update.update(overrides)
    entry = seed.model_copy(update=update)
    return ev.canonical_json_bytes(entry.model_dump(mode="json"))


def test_valid_next_portable_entry_temp_is_removed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home) as run:
        payload = _next_entry_temp_bytes(run)
        temp = _entries_dir(run) / f".00000002.json.{'0' * 16}.tmp"
        write_exclusive(temp, payload, 0o600)
        ev._load_portable_entries(run.active)
        assert not temp.exists()
        run.succeed()


def test_multiple_portable_entry_temps_fail_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    entries = _entries_dir(run)
    _seed_entry(run)
    write_exclusive(entries / f".00000002.json.{'0' * 16}.tmp", b"a\n", 0o600)
    write_exclusive(entries / f".00000003.json.{'0' * 16}.tmp", b"b\n", 0o600)
    with pytest.raises(ev.RunError, match="multiple writer temp portable"):
        ev._load_portable_entries(run.active)
    run._stack.close()


def test_divergent_portable_entry_temp_committed_conflict(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    committed = _seed_entry(run)
    temp = _entries_dir(run) / f".00000001.json.{'0' * 16}.tmp"
    write_exclusive(temp, committed.read_bytes() + b"tamper", 0o600)
    with pytest.raises(ev.RunError, match="divergent writer temp portable"):
        ev._load_portable_entries(run.active)
    run._stack.close()


def test_uncommitted_portable_entry_temp_bad_policy_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    payload = _next_entry_temp_bytes(run, role="model")
    write_exclusive(_entries_dir(run) / f".00000002.json.{'0' * 16}.tmp", payload, 0o600)
    with pytest.raises(ev.RunError, match="out-of-policy"):
        ev._load_portable_entries(run.active)
    run._stack.close()


def test_portable_entry_directory_unexpected_member_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    _seed_entry(run)
    write_exclusive(_entries_dir(run) / "stray", b"x\n", 0o600)
    with pytest.raises(ev.RunError, match="unexpected portable entry directory member"):
        ev._load_portable_entries(run.active)
    run._stack.close()


# ------------------------------------------------------ orphan-blob reconciliation


def test_load_portable_entries_rejects_noncontiguous(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    entry = _seed_entry(run)
    entry.rename(entry.parent / "00000002.json")
    with pytest.raises(ev.RunError, match="noncontiguous"):
        ev._load_portable_entries(run.active)
    run._stack.close()


def test_load_entries_rejects_shared_blob_conflicting_media(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    run.write_portable("a.txt", b"dup\n", media_type="text/plain", role="summary")
    run.write_portable("b.txt", b"dup\n", media_type="text/plain", role="build")
    entry2 = run.active / "portable" / "entries" / "00000002.json"
    forged = ev._read_model(entry2, ev.PortableEvidenceV1).model_copy(
        update={"media_type": "text/markdown"}
    )
    entry2.chmod(0o600)
    ev._atomic_write(entry2, ev.canonical_json_bytes(forged.model_dump(mode="json")))
    with pytest.raises(ev.RunError, match="shared under conflicting media"):
        ev._load_portable_entries(run.active)
    run._stack.close()


def test_load_portable_entries_rejects_out_of_policy_role(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    entry_path = _seed_entry(run)
    entry = ev._read_model(entry_path, ev.PortableEvidenceV1)
    forged = entry.model_copy(update={"role": "model"})
    entry_path.chmod(0o600)
    ev._atomic_write(entry_path, ev.canonical_json_bytes(forged.model_dump(mode="json")))
    with pytest.raises(ev.RunError, match="out-of-policy"):
        ev._load_portable_entries(run.active)
    run._stack.close()


def test_load_portable_entries_rejects_blob_divergence(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    entry = ev._read_model(_seed_entry(run), ev.PortableEvidenceV1)
    blob = run.active / "portable" / "blobs" / entry.blob_sha256
    blob.chmod(0o600)
    blob.write_bytes(b"tampered\n")
    with pytest.raises(ev.RunError, match="diverged from its blob"):
        ev._load_portable_entries(run.active)
    run._stack.close()


def _blobs_dir(run) -> Path:
    run.write_portable("seed.json", b'{"k":1}\n', media_type="application/json", role="summary")
    return run.active / "portable" / "blobs"


def test_valid_blob_temp_is_removed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with _begin(home) as run:
        blobs = _blobs_dir(run)
        content = b"orphan blob bytes\n"
        digest = hashlib.sha256(content).hexdigest()
        temp = blobs / f".{digest}.{'0' * 16}.tmp"
        write_exclusive(temp, content, 0o600)
        ev._recover_portable(run.active)
        assert not temp.exists()
        run.succeed()


def test_blob_temp_content_mismatch_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    blobs = _blobs_dir(run)
    # The name-derived digest is not what the bytes hash to: fail closed.
    temp = blobs / f".{'a' * 64}.{'0' * 16}.tmp"
    write_exclusive(temp, b"does not hash to a's\n", 0o600)
    with pytest.raises(ev.RunError, match="divergent writer temp portable blob"):
        ev._recover_portable(run.active)
    run._stack.close()


def test_unsafe_blob_temp_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    blobs = _blobs_dir(run)
    content = b"orphan blob bytes\n"
    digest = hashlib.sha256(content).hexdigest()
    temp = blobs / f".{digest}.{'0' * 16}.tmp"
    write_exclusive(temp, content, 0o600)
    temp.chmod(0o644)
    with pytest.raises(ev.RunError, match="unsafe writer temp portable blob"):
        ev._recover_portable(run.active)
    run._stack.close()


def test_unexpected_blob_directory_member_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    blobs = _blobs_dir(run)
    write_exclusive(blobs / "not-a-digest", b"x\n", 0o600)
    with pytest.raises(ev.RunError, match="unexpected portable blob directory member"):
        ev._recover_portable(run.active)
    run._stack.close()


def test_orphan_blob_content_address_mismatch_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    blobs = _blobs_dir(run)
    # A valid-looking but unreferenced blob whose bytes do not hash to its name.
    write_exclusive(blobs / ("a" * 64), b"unreferenced\n", 0o600)
    with pytest.raises(ev.RunError, match="content-address"):
        ev._recover_portable(run.active)
    run._stack.close()


def test_unsafe_orphan_blob_fails_closed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    blobs = _blobs_dir(run)
    orphan = blobs / ("b" * 64)
    write_exclusive(orphan, b"unreferenced\n", 0o600)
    orphan.chmod(0o644)
    with pytest.raises(ev.RunError, match="unsafe orphan portable blob"):
        ev._recover_portable(run.active)
    run._stack.close()


# ---------------------------------------------------- storage preparation safety


def test_layout_rejects_symlinked_home_without_mutation(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    home = tmp_path / "home"
    home.symlink_to(target)
    with pytest.raises(ev.RunError, match="unsafe run storage directory"):
        ev._layout(home, create=True)
    # Nothing was created through the symlinked home.
    assert list(target.iterdir()) == []


def test_layout_rejects_symlinked_storage_root_without_mutation(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o700)
    (home / "runs").symlink_to(outside)  # attacker plants a symlinked storage root
    with pytest.raises(ev.RunError, match="unsafe run storage directory"):
        ev._layout(home, create=True)
    # No child directories were created inside the symlink target.
    assert list(outside.iterdir()) == []


def test_layout_rejects_permissive_run_storage_directory(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    (home / "runs").mkdir(mode=0o755)  # documented storage mode is 0700
    with pytest.raises(ev.RunError, match="unsafe run storage directory"):
        ev._layout(home, create=True)


# ------------------------------------------- complete finalized-record verifier


def _reseal_record(record: Path, work: Path, mutate) -> Path:
    """Copy a finalized record, mutate it, and re-publish so its manifest is
    self-consistent again (generic verify_record passes) though its semantics may
    be corrupt."""
    import shutil

    from strixlab.records import publish_record

    src = work / "src"
    shutil.copytree(record, src)
    manifest = src / "record-manifest.json"
    manifest.chmod(0o600)
    manifest.unlink()
    mutate(src)
    dest = work / "resealed"
    publish_record(src, dest)
    return dest


def test_verify_finalized_record_rejects_resealed_broken_event_chain(tmp_path: Path) -> None:
    from strixlab.records import verify_record

    home = tmp_path / "home"
    run_id = _finalize(home)
    record = ev._record_root(ev._layout(home, create=False), run_id)

    def mutate(src: Path) -> None:
        event_path = src / "events" / "00000002.json"
        event = ev.RunEventV1.model_validate_json(event_path.read_bytes(), strict=True)
        forged = event.model_copy(update={"previous_sha256": "f" * 64})
        event_path.chmod(0o600)
        event_path.write_bytes(ev.canonical_json_bytes(forged.model_dump(mode="json")))

    resealed = _reseal_record(record, tmp_path / "work", mutate)
    verify_record(resealed)  # generic record verification still passes
    with pytest.raises(ev.RunError, match="chain is inconsistent|checksum"):
        ev._verify_finalized_record(resealed, run_id)


def test_verify_finalized_record_rejects_resealed_incomplete_checksums(tmp_path: Path) -> None:
    from strixlab.records import verify_record

    home = tmp_path / "home"
    run_id = _finalize(home)
    record = ev._record_root(ev._layout(home, create=False), run_id)

    def mutate(src: Path) -> None:
        path = src / "checksums.sha256"
        kept = [
            line
            for line in path.read_bytes().decode().splitlines()
            if not line.endswith("  run.json")
        ]
        path.chmod(0o600)
        path.write_bytes(("\n".join(kept) + "\n").encode())

    resealed = _reseal_record(record, tmp_path / "work", mutate)
    verify_record(resealed)  # generic record verification still passes
    with pytest.raises(ev.RunError, match="do not cover the exact payload set"):
        ev._verify_finalized_record(resealed, run_id)


def test_verify_finalized_record_rejects_foreign_descriptor_run_id(tmp_path: Path) -> None:
    from strixlab.records import verify_record

    home = tmp_path / "home"
    run_id = _finalize(home)
    record = ev._record_root(ev._layout(home, create=False), run_id)

    def mutate(src: Path) -> None:
        path = src / "run.json"
        descriptor = ev.RunDescriptorV1.model_validate_json(path.read_bytes(), strict=True)
        forged = descriptor.model_copy(update={"run_id": "run-20260830T120000Z-exp-x-" + "0" * 32})
        path.chmod(0o600)
        path.write_bytes(ev.canonical_json_bytes(forged.model_dump(mode="json")))

    resealed = _reseal_record(record, tmp_path / "work", mutate)
    verify_record(resealed)  # generic record verification still passes
    with pytest.raises(ev.RunError, match="descriptor is bound to another run"):
        ev._verify_finalized_record(resealed, run_id)


# ------------------------------------------- descriptor-relative safe deletion


def test_remove_authenticated_subdir_deletes_matching_identity(tmp_path: Path) -> None:
    parent = tmp_path / "p"
    parent.mkdir()
    child = parent / "child"
    (child / "sub").mkdir(parents=True)
    (child / "sub" / "f.txt").write_bytes(b"x\n")
    meta = child.stat()
    fd = os.open(parent, ev._DIR_OPEN_FLAGS)
    try:
        ev._remove_authenticated_subdir(fd, "child", expect_dev=meta.st_dev, expect_ino=meta.st_ino)
    finally:
        os.close(fd)
    assert not child.exists()


def test_remove_authenticated_subdir_rejects_inode_mismatch(tmp_path: Path) -> None:
    parent = tmp_path / "p"
    parent.mkdir()
    (parent / "child").mkdir()
    fd = os.open(parent, ev._DIR_OPEN_FLAGS)
    try:
        with pytest.raises(ev.RunError, match="diverged from its authenticated identity"):
            ev._remove_authenticated_subdir(fd, "child", expect_dev=1, expect_ino=1)
    finally:
        os.close(fd)
    assert (parent / "child").exists()  # never deleted


def test_remove_authenticated_subdir_rejects_swap_before_rmdir(tmp_path: Path, monkeypatch) -> None:
    parent = tmp_path / "p"
    parent.mkdir()
    child = parent / "child"
    child.mkdir()
    other = parent / "other"
    other.mkdir()
    meta = child.stat()
    real = ev._rmtree_fd

    def hook(dir_fd: int) -> None:
        real(dir_fd)  # empty the authenticated child via its held descriptor
        # Swap the name to a different-inode directory before the empty-dir rmdir.
        os.rename(parent / "child", parent / "stash")
        os.rename(parent / "other", parent / "child")

    monkeypatch.setattr(ev, "_rmtree_fd", hook)
    fd = os.open(parent, ev._DIR_OPEN_FLAGS)
    try:
        with pytest.raises(ev.RunError, match="was replaced before removal"):
            ev._remove_authenticated_subdir(
                fd, "child", expect_dev=meta.st_dev, expect_ino=meta.st_ino
            )
    finally:
        os.close(fd)


def test_recover_runs_leaves_a_locked_empty_stage(tmp_path: Path) -> None:
    from strixlab.locks import exclusive_lock

    home = tmp_path / "home"
    roots = ev._layout(home, create=True)
    run_id = "run-20260830T120000Z-exp-x-" + "0" * 32
    stage = roots.allocation_staging / f".{run_id}.{'0' * 16}.tmp"
    stage.mkdir(mode=0o700)
    # Hold the run lock, as a live allocator does throughout staging.
    with exclusive_lock(ev._lock_path(roots, run_id)) as held:
        assert held.acquired
        ev.recover_runs(home=home)
        assert stage.exists()  # a locked (live-allocator) empty stage is never reclaimed


def test_recover_rejects_symlinked_active_root(tmp_path: Path) -> None:
    import shutil

    home = tmp_path / "home"
    run = _begin(home)
    run_id = run.run_id
    active = run.active
    run._stack.close()
    decoy = tmp_path / "decoy"
    decoy.mkdir(mode=0o700)
    # Swap the whole active/<run-id> tree for a symlink to a decoy directory.
    shutil.rmtree(active)
    active.symlink_to(decoy)
    with pytest.raises(ev.RunError, match="run directory is unavailable"):
        ev.recover_run(run_id, home=home)


def test_load_status_rejects_symlinked_active_root(tmp_path: Path) -> None:
    import shutil

    home = tmp_path / "home"
    run = _begin(home)
    active = run.active
    run._stack.close()
    decoy = tmp_path / "decoy"
    decoy.mkdir(mode=0o700)
    (decoy / "status.json").write_bytes(b"{}\n")  # attacker-controlled decoy content
    shutil.rmtree(active)
    active.symlink_to(decoy)
    # _load_status must not follow the intermediate symlink to read decoy content.
    with pytest.raises(ev.RunError, match="run directory is unavailable"):
        ev._load_status(active)


def test_remove_authenticated_subdir_rmdir_window_refuses_nonempty_swap(
    tmp_path: Path, monkeypatch
) -> None:
    # A same-UID swap in the inherent lstat->rmdir window can at worst target an empty
    # directory; a swap to a NON-EMPTY directory fails ENOTEMPTY and removes no data.
    parent = tmp_path / "p"
    parent.mkdir()
    child = parent / "child"
    child.mkdir()
    other = parent / "other"
    other.mkdir()
    (other / "keep.txt").write_bytes(b"data\n")
    meta = child.stat()
    real_rmdir = os.rmdir

    def hook(name, *, dir_fd=None):
        if name == "child" and dir_fd is not None:
            os.rename("child", "stash", src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            os.rename("other", "child", src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
        return real_rmdir(name, dir_fd=dir_fd)

    monkeypatch.setattr(os, "rmdir", hook)
    fd = os.open(parent, ev._DIR_OPEN_FLAGS)
    try:
        with pytest.raises(OSError):  # ENOTEMPTY: the swapped non-empty dir is not removed
            ev._remove_authenticated_subdir(
                fd, "child", expect_dev=meta.st_dev, expect_ino=meta.st_ino
            )
    finally:
        os.close(fd)
    assert (parent / "child" / "keep.txt").read_bytes() == b"data\n"  # data survived


# ---------------------------------------- durable-before-teardown (finding 1)


def _crash_before(run, monkeypatch, target: str, message: str) -> None:
    real = getattr(ev, target)

    def boom(*_a, **_k):
        raise OSError(message)

    monkeypatch.setattr(ev, target, boom)
    with pytest.raises(OSError, match=message):
        run.succeed()
    monkeypatch.setattr(ev, target, real)


def _order_spy(monkeypatch) -> list[tuple[str, Path | None]]:
    order: list[tuple[str, Path | None]] = []
    real_fsync = ev.fsync_directory
    real_teardown = ev._teardown_active

    def spy_fsync(path):
        order.append(("fsync", Path(path)))
        return real_fsync(path)

    def spy_teardown(*a, **k):
        order.append(("teardown", None))
        return real_teardown(*a, **k)

    monkeypatch.setattr(ev, "fsync_directory", spy_fsync)
    monkeypatch.setattr(ev, "_teardown_active", spy_teardown)
    return order


def test_recovery_fsyncs_record_parent_before_teardown_on_existing_record(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    run_id = run.run_id
    # Crash after publish_record, before the index: record present, index absent.
    _crash_before(run, monkeypatch, "_publish_index", "crash before index")
    roots = ev._layout(home, create=False)
    order = _order_spy(monkeypatch)
    ev.recover_run(run_id, home=home)
    fsync_records = [i for i, e in enumerate(order) if e == ("fsync", roots.records)]
    teardown = [i for i, e in enumerate(order) if e[0] == "teardown"]
    assert fsync_records and teardown and min(fsync_records) < min(teardown)


def test_finish_from_index_fsyncs_record_and_index_parents_before_teardown(
    tmp_path: Path, monkeypatch
) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    run_id = run.run_id
    # Crash after index publication, before teardown: record + index present, active present.
    _crash_before(run, monkeypatch, "_teardown_active", "crash before teardown")
    roots = ev._layout(home, create=False)
    order = _order_spy(monkeypatch)
    ev.recover_run(run_id, home=home)
    teardown = min(i for i, e in enumerate(order) if e[0] == "teardown")
    frecords = [i for i, e in enumerate(order) if e == ("fsync", roots.records)]
    findexes = [i for i, e in enumerate(order) if e == ("fsync", roots.indexes)]
    assert frecords and min(frecords) < teardown
    assert findexes and min(findexes) < teardown


# ------------------------------------------ control writer-temp reconcile (finding 2)


def _finalize_after_crash(run, run_id: str, home: Path):
    status = ev._required_status(run.active, run_id)
    ev._transition(
        run.active,
        status,
        ev.RunState.TERMINAL,
        run_id=run_id,
        clock=lambda: _FIXED,
        outcome=ev.RunOutcome.SUCCESS,
    )
    run._stack.close()
    return ev.recover_run(run_id, home=home)


def test_finalization_removes_stray_status_writer_temp(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    run_id = run.run_id
    write_exclusive(run.active / f".status.json.{'0' * 16}.tmp", b"garbage\n", 0o600)
    inspection = _finalize_after_crash(run, run_id, home)
    checks = (inspection.record / "checksums.sha256").read_bytes().decode()
    assert ".status.json" not in checks
    assert not (inspection.record / f".status.json.{'0' * 16}.tmp").exists()


def test_finalization_removes_stray_checksums_writer_temp(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    run_id = run.run_id
    write_exclusive(run.active / f".checksums.sha256.{'0' * 16}.tmp", b"garbage\n", 0o600)
    inspection = _finalize_after_crash(run, run_id, home)
    checks = (inspection.record / "checksums.sha256").read_bytes().decode()
    assert ".checksums.sha256" not in checks


def test_reconcile_control_temps_rejects_unexpected_root_temp(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    write_exclusive(run.active / ".weird.tmp", b"x\n", 0o600)
    with pytest.raises(ev.RunError, match="unexpected run root writer temp"):
        ev._reconcile_control_temps(run.active)
    run._stack.close()


def test_reconcile_control_temps_rejects_multiple_status_temps(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    write_exclusive(run.active / f".status.json.{'0' * 16}.tmp", b"a\n", 0o600)
    write_exclusive(run.active / f".status.json.{'1' * 16}.tmp", b"b\n", 0o600)
    with pytest.raises(ev.RunError, match="multiple status.json writer temps"):
        ev._reconcile_control_temps(run.active)
    run._stack.close()


# ------------------------------------------------ lock status handling (finding 3)


def test_recover_runs_raises_on_unavailable_active_lock(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run = _begin(home)
    run_id = run.run_id
    run._stack.close()
    roots = ev._layout(home, create=False)
    ev._lock_path(roots, run_id).chmod(0o644)  # permissive lock -> UNAVAILABLE, not CONTENDED
    with pytest.raises(ev.RunError, match="lock|0o600"):
        ev.recover_runs(home=home)


def test_reconcile_allocation_staging_raises_on_unavailable_lock(tmp_path: Path) -> None:
    home = tmp_path / "home"
    roots = ev._layout(home, create=True)
    run_id = "run-20260830T120000Z-exp-x-" + "0" * 32
    (roots.allocation_staging / f".{run_id}.{'0' * 16}.tmp").mkdir(mode=0o700)
    lock = ev._lock_path(roots, run_id)
    lock.write_bytes(b"")
    lock.chmod(0o644)
    with pytest.raises(ev.RunError, match="lock|0o600"):
        ev.recover_runs(home=home)

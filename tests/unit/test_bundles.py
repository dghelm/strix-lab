from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

import strixlab.bundles as bd
import strixlab.evidence as ev
from strixlab.serialization import canonical_json_bytes

_ENV = {"PATH": "/usr/bin"}


def _finalized_run(home: Path) -> str:
    with ev.begin_run(
        "exp-bundle", b"suite: b\n", resolved={"a": 1}, home=home, environ=_ENV
    ) as run:
        run.write_evidence("logs/local.txt", b"local, not exported\n")
        run.write_portable(
            "env.json", b'{"k":1}\n', media_type="application/json", role="environment"
        )
        run.write_portable("notes.md", b"# notes\n", media_type="text/markdown", role="summary")
        run.succeed()
    return run.run_id


def _export(home: Path, tmp_path: Path, name: str = "bundle") -> Path:
    run_id = _finalized_run(home)
    destination = tmp_path / name
    bd.export_bundle(run_id, destination, home=home, environ=_ENV)
    return destination


def test_export_is_verifiable_and_deterministic(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run_id = _finalized_run(home)
    one = tmp_path / "one"
    two = tmp_path / "two"
    bd.export_bundle(run_id, one, home=home, environ=_ENV)
    bd.export_bundle(run_id, two, home=home, environ=_ENV)

    def tree(root: Path) -> dict[str, bytes]:
        return {
            str(path.relative_to(root)): path.read_bytes()
            for path in sorted(root.rglob("*"))
            if path.is_file()
        }

    assert tree(one) == tree(two)
    inspection = bd.verify_bundle(one)
    assert inspection.run_id == run_id
    assert inspection.outcome == "success"


def test_bundle_excludes_local_evidence_and_includes_only_referenced_blobs(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    members = {str(path.relative_to(bundle)) for path in bundle.rglob("*") if path.is_file()}
    assert not any("local.txt" in member for member in members)
    blobs = [member for member in members if member.startswith("run/portable/blobs/")]
    entries = [member for member in members if member.startswith("run/portable/entries/")]
    assert len(blobs) == 2 and len(entries) == 2


def test_export_enforces_total_file_limit(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    run_id = _finalized_run(home)
    monkeypatch.setattr(bd, "MAX_TOTAL_FILES", 1)
    with pytest.raises(bd.BundleError, match="total file limit"):
        bd.export_bundle(run_id, tmp_path / "bundle", home=home, environ=_ENV)


def test_export_enforces_aggregate_limit(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    run_id = _finalized_run(home)
    monkeypatch.setattr(bd, "MAX_AGGREGATE_BYTES", 4)
    with pytest.raises(bd.BundleError, match="aggregate payload limit"):
        bd.export_bundle(run_id, tmp_path / "bundle", home=home, environ=_ENV)


def test_export_refuses_existing_destination(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run_id = _finalized_run(home)
    destination = tmp_path / "bundle"
    destination.mkdir()
    with pytest.raises(bd.BundleError, match="already exists"):
        bd.export_bundle(run_id, destination, home=home, environ=_ENV)


def test_export_fails_on_secret_in_environment(tmp_path: Path) -> None:
    home = tmp_path / "home"
    with ev.begin_run(
        "exp-s", b"suite: s\n", resolved={"a": 1}, home=home, environ={"PATH": "/usr/bin"}
    ) as run:
        run_id = run.run_id
        run.write_portable(
            "env.json",
            b'{"marker":"leakme123"}\n',
            media_type="application/json",
            role="environment",
        )
        run.succeed()
    with pytest.raises(bd.BundleError, match="sensitive value"):
        bd.export_bundle(
            run_id,
            tmp_path / "bundle",
            home=home,
            environ={"API_TOKEN": "leakme123"},
        )


@pytest.mark.parametrize("secret_name", [None, "API_TOKEN", "SESSION", "XDG_SESSION_CLASS_TOKEN"])
def test_export_session_class_with_user_containing_evidence(
    tmp_path: Path, secret_name: str | None
) -> None:
    home = tmp_path / "home"
    payload = b'{"path":"examples/llama.android/app/src/main/res/drawable/bg_user_message.xml"}\n'
    with ev.begin_run(
        "exp-class", b"suite: s\n", resolved={"a": 1}, home=home, environ=_ENV
    ) as run:
        run.write_portable("suite/build.json", payload, media_type="application/json", role="build")
        run.succeed()

    environ = {**_ENV, "XDG_SESSION_CLASS": "user"}
    if secret_name is not None:
        environ[secret_name] = "user"
    destination = tmp_path / "bundle"
    if secret_name is not None:
        with pytest.raises(bd.BundleError, match="sensitive value"):
            bd.export_bundle(run.run_id, destination, home=home, environ=environ)
        assert not destination.exists()
    else:
        bd.export_bundle(run.run_id, destination, home=home, environ=environ)
        assert bd.verify_bundle(destination).run_id == run.run_id
        blob = destination / "run" / "portable" / "blobs" / hashlib.sha256(payload).hexdigest()
        assert blob.read_bytes() == payload


def _tamper(bundle: Path, relative: str, content: bytes) -> None:
    target = bundle / relative
    target.chmod(0o600)
    target.write_bytes(content)


def test_verify_rejects_payload_and_size_and_manifest_tamper(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    _tamper(bundle, "run/status.json", b"{}")
    with pytest.raises(bd.BundleError):
        bd.verify_bundle(bundle)


def test_verify_rejects_undeclared_extra_member(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    (bundle / "run" / "extra.txt").write_bytes(b"undeclared\n")
    with pytest.raises(bd.BundleError, match="membership"):
        bd.verify_bundle(bundle)


def test_verify_rejects_symlinked_member(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    # Replace a declared member with a symlink: the descriptor-anchored walk catches
    # the unsafe entry before any read.
    target = bundle / "run" / "status.json"
    target.chmod(0o600)
    target.unlink()
    target.symlink_to(bundle / "bundle.json")
    with pytest.raises(bd.BundleError, match="unsafe directory entry"):
        bd.verify_bundle(bundle)


def test_verify_rejects_executable_member(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    (bundle / "run" / "status.json").chmod(0o700)
    with pytest.raises(bd.BundleError, match="executable"):
        bd.verify_bundle(bundle)


def test_verify_rejects_noncanonical_or_tampered_manifest(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    manifest = json.loads((bundle / "bundle.json").read_bytes())
    manifest["members"][0]["sha256"] = "0" * 64
    _tamper(bundle, "bundle.json", canonical_json_bytes(manifest))
    with pytest.raises(bd.BundleError):
        bd.verify_bundle(bundle)


def test_verify_rejects_media_type_tamper(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    manifest = json.loads((bundle / "bundle.json").read_bytes())
    for member in manifest["members"]:
        if member["path"] == "run/status.json":
            member["media_type"] = "text/plain"
    _tamper(bundle, "bundle.json", canonical_json_bytes(manifest))
    with pytest.raises(bd.BundleError, match="wrong media type"):
        bd.verify_bundle(bundle)


def test_verify_rejects_non_directory_path(tmp_path: Path) -> None:
    target = tmp_path / "afile"
    target.write_bytes(b"not a bundle\n")
    with pytest.raises(bd.BundleError, match="unavailable|not an owned directory"):
        bd.verify_bundle(target)


def test_verify_rejects_non_json_manifest(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    _tamper(bundle, "bundle.json", b"not json at all")
    with pytest.raises(bd.BundleError, match="invalid"):
        bd.verify_bundle(bundle)


def test_verify_rejects_noncanonical_manifest_bytes(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    manifest = json.loads((bundle / "bundle.json").read_bytes())
    # Valid JSON for the same model, but pretty-printed (noncanonical) bytes.
    _tamper(bundle, "bundle.json", json.dumps(manifest, indent=2).encode("utf-8"))
    with pytest.raises(bd.BundleError, match="not canonical"):
        bd.verify_bundle(bundle)


def test_verify_rejects_manifest_omitting_record_manifest(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    record_member = bundle / "run" / "record-manifest.json"
    record_member.chmod(0o600)
    record_member.unlink()

    def mutate(manifest: dict) -> None:
        manifest["members"] = [
            m for m in manifest["members"] if m["path"] != "run/record-manifest.json"
        ]

    _rewrite_manifest(bundle, mutate)
    with pytest.raises(bd.BundleError, match="omits the embedded record manifest"):
        bd.verify_bundle(bundle)


def test_verify_rejects_member_absent_from_immutable_record(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    # A self-consistent, correctly-declared member that the immutable record never
    # listed at all: membership matches its manifest, but the record does not.
    extra = b"undeclared by the record\n"
    (bundle / "run" / "extra.txt").write_bytes(extra)

    def mutate(manifest: dict) -> None:
        manifest["members"].append(
            {
                "path": "run/extra.txt",
                "sha256": hashlib.sha256(extra).hexdigest(),
                "size_bytes": len(extra),
                "media_type": "text/plain",
            }
        )

    _rewrite_manifest(bundle, mutate)
    with pytest.raises(bd.BundleError, match="not in the immutable record"):
        bd.verify_bundle(bundle)


def test_verify_rejects_run_record_digest_binding(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    # Manifest self-consistent, but its declared run-record digest no longer matches
    # the embedded record manifest it is supposed to bind.
    _rewrite_manifest(
        bundle,
        lambda m: m.__setitem__("run_record_sha256", "record-sha256:" + "0" * 64),
    )
    with pytest.raises(bd.BundleError, match="run-record digest diverged"):
        bd.verify_bundle(bundle)


def test_verify_rejects_missing_manifest(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    (bundle / "bundle.json").chmod(0o600)
    (bundle / "bundle.json").unlink()
    with pytest.raises(bd.BundleError, match="missing bundle.json"):
        bd.verify_bundle(bundle)


def test_verify_rejects_symlinked_subdirectory(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    (bundle / "run" / "evil").symlink_to(bundle / "run" / "portable")
    with pytest.raises(bd.BundleError, match="unsafe directory entry"):
        bd.verify_bundle(bundle)


def _rewrite_manifest(bundle: Path, mutate) -> None:
    manifest = json.loads((bundle / "bundle.json").read_bytes())
    mutate(manifest)
    _tamper(bundle, "bundle.json", canonical_json_bytes(manifest))


def test_verify_rejects_manifest_self_declaration(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)

    def mutate(manifest: dict) -> None:
        clone = dict(manifest["members"][0])
        clone["path"] = "bundle.json"
        manifest["members"].append(clone)

    _rewrite_manifest(bundle, mutate)
    with pytest.raises(bd.BundleError, match="must not declare itself"):
        bd.verify_bundle(bundle)


def test_verify_rejects_manifest_duplicate_member(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    _rewrite_manifest(bundle, lambda m: m["members"].append(dict(m["members"][0])))
    with pytest.raises(bd.BundleError, match="duplicate member"):
        bd.verify_bundle(bundle)


def test_verify_rejects_portable_entry_wrong_media_type(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)

    def mutate(manifest: dict) -> None:
        for member in manifest["members"]:
            if member["path"].startswith("run/portable/entries/"):
                member["media_type"] = "text/plain"

    _rewrite_manifest(bundle, mutate)
    with pytest.raises(bd.BundleError, match="portable entry has the wrong media type"):
        bd.verify_bundle(bundle)


def test_verify_rejects_forged_run_id(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    other = "run-20260830T120000Z-exp-x-" + "0" * 32
    _rewrite_manifest(bundle, lambda m: m.__setitem__("run_id", other))
    with pytest.raises(bd.BundleError, match="run identity does not match"):
        bd.verify_bundle(bundle)


def test_verify_rejects_forged_outcome(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    _rewrite_manifest(bundle, lambda m: m.__setitem__("outcome", "failure"))
    with pytest.raises(bd.BundleError, match="outcome does not match"):
        bd.verify_bundle(bundle)


def test_verify_rejects_bad_blob_media_binding(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)

    def mutate(manifest: dict) -> None:
        for member in manifest["members"]:
            if member["path"].startswith("run/portable/blobs/"):
                member["media_type"] = "text/markdown"

    _rewrite_manifest(bundle, mutate)
    with pytest.raises(bd.BundleError, match="portable blob metadata diverged"):
        bd.verify_bundle(bundle)


def test_verify_rejects_incomplete_checksum_coverage(tmp_path: Path) -> None:
    from strixlab.records import record_manifest_digest

    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    # Add a phantom file to the embedded record manifest (rebinding its digest) so
    # the checksum declarations no longer cover the exact record payload set.
    record = json.loads((bundle / "run" / "record-manifest.json").read_bytes())
    record["files"].append(
        {"path": "phantom.txt", "mode": 384, "size_bytes": 1, "sha256": "0" * 64}
    )
    new_record = canonical_json_bytes(record)
    _tamper(bundle, "run/record-manifest.json", new_record)

    def mutate(manifest: dict) -> None:
        manifest["run_record_sha256"] = record_manifest_digest(new_record)
        for member in manifest["members"]:
            if member["path"] == "run/record-manifest.json":
                member["sha256"] = hashlib.sha256(new_record).hexdigest()
                member["size_bytes"] = len(new_record)

    _rewrite_manifest(bundle, mutate)
    with pytest.raises(bd.BundleError, match="checksums do not cover"):
        bd.verify_bundle(bundle)


def test_export_rejects_missing_destination_parent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run_id = _finalized_run(home)
    destination = tmp_path / "no-such-parent" / "bundle"
    with pytest.raises(bd.BundleError, match="bundle directory is unavailable"):
        bd.export_bundle(run_id, destination, home=home, environ=_ENV)


def test_export_rejects_symlinked_destination_parent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run_id = _finalized_run(home)
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    # The no-follow open of the symlinked parent fails closed.
    with pytest.raises(bd.BundleError, match="bundle directory is unavailable"):
        bd.export_bundle(run_id, link / "bundle", home=home, environ=_ENV)


def test_export_publishes_under_destination_parent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run_id = _finalized_run(home)
    # A destination whose parent is a wholly separate tree from home: staging is a
    # sibling of the destination, so no bundle-staging directory under home is used.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    destination = elsewhere / "bundle"
    bd.export_bundle(run_id, destination, home=home, environ=_ENV)
    assert bd.verify_bundle(destination).run_id == run_id
    staging = home / "runs" / "bundle-staging"
    assert not staging.exists() or not any(staging.iterdir())


# ------------------------------------------------------- adversarial hardening


def test_verify_rejects_symlinked_intermediate_directory(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    # Replace the whole run/ subtree with a symlink: the descriptor-anchored walk
    # never re-resolves it from the root path, so it fails closed.
    shutil.rmtree(bundle / "run")
    (bundle / "run").symlink_to(outside)
    with pytest.raises(bd.BundleError, match="unsafe directory entry"):
        bd.verify_bundle(bundle)


def _chain_from_bundle(bundle: Path) -> tuple[dict[str, bytes], ev.RunStatusV1]:
    contents = {
        f"run/events/{path.name}": path.read_bytes()
        for path in sorted((bundle / "run" / "events").iterdir())
    }
    status = ev.RunStatusV1.model_validate_json(
        (bundle / "run" / "status.json").read_bytes(), strict=True
    )
    return contents, status


def test_event_chain_rejects_gapped_sequence(tmp_path: Path) -> None:
    home = tmp_path / "home"
    contents, status = _chain_from_bundle(_export(home, tmp_path))
    contents.pop("run/events/00000002.json")
    with pytest.raises(bd.BundleError, match="exact contiguous sequence"):
        bd._verify_event_chain(contents, status)


def test_event_chain_rejects_unexpected_member_name(tmp_path: Path) -> None:
    home = tmp_path / "home"
    contents, status = _chain_from_bundle(_export(home, tmp_path))
    contents["run/events/weird"] = b"{}"
    with pytest.raises(bd.BundleError, match="exact contiguous sequence"):
        bd._verify_event_chain(contents, status)


def test_event_chain_rejects_broken_link(tmp_path: Path) -> None:
    home = tmp_path / "home"
    contents, status = _chain_from_bundle(_export(home, tmp_path))
    # Re-serialize event 2 with a previous digest that does not chain.
    event = ev.RunEventV1.model_validate_json(contents["run/events/00000002.json"], strict=True)
    forged = event.model_copy(update={"previous_sha256": "f" * 64})
    contents["run/events/00000002.json"] = ev.canonical_json_bytes(forged.model_dump(mode="json"))
    with pytest.raises(bd.BundleError, match="chain is inconsistent"):
        bd._verify_event_chain(contents, status)


def test_event_chain_rejects_nonterminal_event_with_outcome(tmp_path: Path) -> None:
    home = tmp_path / "home"
    contents, status = _chain_from_bundle(_export(home, tmp_path))
    # Forge the ACTIVE (nonterminal) event to carry a terminal-only outcome.
    raw = json.loads(contents["run/events/00000002.json"])
    raw["outcome"] = "success"
    contents["run/events/00000002.json"] = canonical_json_bytes(raw)
    with pytest.raises(bd.BundleError, match="event chain is invalid"):
        bd._verify_event_chain(contents, status)


def test_event_chain_rejects_status_divergence(tmp_path: Path) -> None:
    home = tmp_path / "home"
    contents, status = _chain_from_bundle(_export(home, tmp_path))
    forged = status.model_copy(update={"sequence": status.sequence + 1})
    with pytest.raises(bd.BundleError, match="event chain"):
        bd._verify_event_chain(contents, forged)


def test_verify_rejects_diverged_input_manifest(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    # The descriptor's captured input-manifest digest no longer matches the bundled
    # manifest bytes. Fixing the member digest still fails the run-descriptor binding.
    new = b"suite: tampered\n"
    _tamper(bundle, "run/manifest.input.yaml", new)

    def mutate(manifest: dict) -> None:
        for member in manifest["members"]:
            if member["path"] == "run/manifest.input.yaml":
                member["sha256"] = hashlib.sha256(new).hexdigest()
                member["size_bytes"] = len(new)

    _rewrite_manifest(bundle, mutate)
    with pytest.raises(bd.BundleError, match="input manifest diverged"):
        bd.verify_bundle(bundle)


def _rebind_record(bundle: Path, mutate_record) -> None:
    from strixlab.records import record_manifest_digest

    record = json.loads((bundle / "run" / "record-manifest.json").read_bytes())
    mutate_record(record)
    new_record = canonical_json_bytes(record)
    _tamper(bundle, "run/record-manifest.json", new_record)

    def mutate(manifest: dict) -> None:
        manifest["run_record_sha256"] = record_manifest_digest(new_record)
        for member in manifest["members"]:
            if member["path"] == "run/record-manifest.json":
                member["sha256"] = hashlib.sha256(new_record).hexdigest()
                member["size_bytes"] = len(new_record)

    _rewrite_manifest(bundle, mutate)


def test_verify_rejects_duplicate_record_paths(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    _rebind_record(bundle, lambda r: r["files"].append(dict(r["files"][0])))
    with pytest.raises(bd.BundleError, match="duplicate paths"):
        bd.verify_bundle(bundle)


def test_verify_rejects_record_member_size_mismatch(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)

    def mutate(record: dict) -> None:
        for entry in record["files"]:
            if entry["path"] == "status.json":
                entry["size_bytes"] = entry["size_bytes"] + 1

    _rebind_record(bundle, mutate)
    with pytest.raises(bd.BundleError, match="size diverged from the record"):
        bd.verify_bundle(bundle)


def _reseal_member(bundle: Path, member: str, new: bytes) -> None:
    """Rewrite a member's bytes and re-declare its digest/size in bundle.json."""

    _tamper(bundle, member, new)

    def mutate(manifest: dict) -> None:
        for entry in manifest["members"]:
            if entry["path"] == member:
                entry["sha256"] = hashlib.sha256(new).hexdigest()
                entry["size_bytes"] = len(new)

    _rewrite_manifest(bundle, mutate)


def test_verify_rejects_portable_entry_filename_sequence_mismatch(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    # Rewrite entry 1's content so its sequence field disagrees with its filename.
    member = "run/portable/entries/00000001.json"
    entry = ev.PortableEvidenceV1.model_validate_json((bundle / member).read_bytes(), strict=True)
    forged = entry.model_copy(update={"sequence": 9})
    _reseal_member(bundle, member, ev.canonical_json_bytes(forged.model_dump(mode="json")))
    with pytest.raises(bd.BundleError, match="filename does not match its sequence"):
        bd.verify_bundle(bundle)


def _shared_blob_run(home: Path) -> str:
    with ev.begin_run(
        "exp-shared", b"suite: s\n", resolved={"a": 1}, home=home, environ=_ENV
    ) as run:
        run.write_portable("a.txt", b"dup\n", media_type="text/plain", role="summary")
        run.write_portable("b.txt", b"dup\n", media_type="text/plain", role="build")
        run.succeed()
    return run.run_id


def test_verify_rejects_shared_blob_conflicting_media(tmp_path: Path) -> None:
    home = tmp_path / "home"
    run_id = _shared_blob_run(home)
    bundle = tmp_path / "bundle"
    bd.export_bundle(run_id, bundle, home=home, environ=_ENV)
    # Two entries share one blob; forge one entry's media type so they disagree.
    member = "run/portable/entries/00000002.json"
    entry = ev.PortableEvidenceV1.model_validate_json((bundle / member).read_bytes(), strict=True)
    forged = entry.model_copy(update={"media_type": "text/markdown"})
    _reseal_member(bundle, member, ev.canonical_json_bytes(forged.model_dump(mode="json")))
    with pytest.raises(bd.BundleError, match="conflicting media types"):
        bd.verify_bundle(bundle)


def test_verify_rejects_invalid_run_json(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    _reseal_member(bundle, "run/run.json", b"not json at all")
    with pytest.raises(bd.BundleError, match="run/run.json is invalid"):
        bd.verify_bundle(bundle)


def test_verify_rejects_diverged_resolved_manifest(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    _reseal_member(bundle, "run/manifest.resolved.yaml", b"a: 999\n")
    with pytest.raises(bd.BundleError, match="resolved manifest diverged"):
        bd.verify_bundle(bundle)


def test_verify_rejects_special_file_member(tmp_path: Path) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    os.mkfifo(bundle / "run" / "pipe")
    with pytest.raises(bd.BundleError, match="unsafe directory entry"):
        bd.verify_bundle(bundle)


def _record_dir(tmp_path: Path) -> tuple[Path, str]:
    home = tmp_path / "home"
    run_id = _finalized_run(home)
    return home / "runs" / "records" / run_id, run_id


def _collect(record: Path):
    return bd._collect_members(record, bd.RedactionContext.from_environ(_ENV))


def test_export_refuses_snapshot_mutated_between_inspect_and_collect(
    tmp_path: Path, monkeypatch
) -> None:
    # Simulate a concurrent record mutation after inspect_run authenticated the record
    # but before the export snapshot is collected. The pre-publish snapshot binding
    # must fail closed and publish nothing.
    home = tmp_path / "home"
    run_id = _finalized_run(home)
    destination = tmp_path / "bundle"
    real_collect = bd._collect_members

    def collect_hook(record: Path, context):
        manifest_path = record / "record-manifest.json"
        manifest_path.chmod(0o600)
        data = json.loads(manifest_path.read_bytes())
        data["files"][0]["mode"] = 0o644  # benign change → different record digest
        manifest_path.write_bytes(canonical_json_bytes(data))
        return real_collect(record, context)

    monkeypatch.setattr(bd, "_collect_members", collect_hook)
    with pytest.raises(bd.BundleError, match="run-record digest diverged"):
        bd.export_bundle(run_id, destination, home=home, environ=_ENV)
    assert not destination.exists()


def test_verify_fails_closed_on_directory_swap_between_list_and_open(
    tmp_path: Path, monkeypatch
) -> None:
    # Swap the run/ subdirectory for a different-inode directory between the walk's
    # lstat and its open; the held-descriptor identity check catches the redirect.
    bundle = _export(tmp_path / "home", tmp_path)
    real_open = os.open
    swapped = {"done": False}

    def hook(path, flags, mode=0o777, *, dir_fd=None):
        if not swapped["done"] and path == "run" and dir_fd is not None:
            swapped["done"] = True
            os.rename("run", "run.stash", src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            os.mkdir("run", 0o700, dir_fd=dir_fd)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", hook)
    with pytest.raises(bd.BundleError, match="directory changed during walk"):
        bd.verify_bundle(bundle)


def test_export_fails_closed_on_member_swap_between_list_and_read(
    tmp_path: Path, monkeypatch
) -> None:
    # Swap a record member for a different inode between enumeration and read; the
    # descriptor-held identity check in _read_child_regular fails closed and nothing
    # is published.
    home = tmp_path / "home"
    run_id = _finalized_run(home)
    destination = tmp_path / "bundle"
    real_open = os.open
    swapped = {"done": False}

    def hook(path, flags, mode=0o777, *, dir_fd=None):
        if not swapped["done"] and path == "status.json" and dir_fd is not None:
            swapped["done"] = True
            os.unlink("status.json", dir_fd=dir_fd)
            new_fd = real_open(
                "status.json", os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600, dir_fd=dir_fd
            )
            os.write(new_fd, b"{}\n")
            os.close(new_fd)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", hook)
    with pytest.raises(bd.BundleError, match="changed during verification"):
        bd.export_bundle(run_id, destination, home=home, environ=_ENV)
    assert not destination.exists()


def test_export_revalidation_rejects_unexpected_event_member(tmp_path: Path) -> None:
    record, _run_id = _record_dir(tmp_path)
    (record / "events" / "weird").write_bytes(b"{}\n")
    with pytest.raises(bd.BundleError, match="unexpected run record event member"):
        _collect(record)


def test_export_revalidation_rejects_blob_content_address(tmp_path: Path) -> None:
    record, _run_id = _record_dir(tmp_path)
    blob = next(iter((record / "portable" / "blobs").iterdir()))
    blob.chmod(0o600)
    blob.write_bytes(b"corrupted; wrong hash\n")
    with pytest.raises(bd.BundleError, match="not content-addressed"):
        _collect(record)


def test_export_revalidation_rejects_out_of_policy_role(tmp_path: Path) -> None:
    record, _run_id = _record_dir(tmp_path)
    entry_path = record / "portable" / "entries" / "00000001.json"
    entry = ev.PortableEvidenceV1.model_validate_json(entry_path.read_bytes(), strict=True)
    forged = entry.model_copy(update={"role": "model"})
    entry_path.chmod(0o600)
    entry_path.write_bytes(ev.canonical_json_bytes(forged.model_dump(mode="json")))
    with pytest.raises(bd.BundleError, match="out-of-policy role or media type"):
        _collect(record)


def test_export_revalidation_rejects_entry_sequence_mismatch(tmp_path: Path) -> None:
    record, _run_id = _record_dir(tmp_path)
    first = record / "portable" / "entries" / "00000001.json"
    forged = ev.PortableEvidenceV1.model_validate_json(first.read_bytes(), strict=True).model_copy(
        update={"sequence": 5}
    )
    first.chmod(0o600)
    first.write_bytes(ev.canonical_json_bytes(forged.model_dump(mode="json")))
    with pytest.raises(bd.BundleError, match="filename does not match its sequence"):
        _collect(record)


def test_export_revalidation_enforces_entry_limit(tmp_path: Path, monkeypatch) -> None:
    record, _run_id = _record_dir(tmp_path)
    monkeypatch.setattr(bd, "MAX_PORTABLE_ENTRIES", 1)
    with pytest.raises(bd.BundleError, match="exceeds the portable entry limit"):
        _collect(record)


def test_export_revalidation_rejects_conflicting_logical_paths(tmp_path: Path) -> None:
    record, _run_id = _record_dir(tmp_path)
    entries = record / "portable" / "entries"
    first = ev.PortableEvidenceV1.model_validate_json(
        (entries / "00000001.json").read_bytes(), strict=True
    )
    second_path = entries / "00000002.json"
    forged = ev.PortableEvidenceV1.model_validate_json(
        second_path.read_bytes(), strict=True
    ).model_copy(update={"logical_path": first.logical_path})
    second_path.chmod(0o600)
    second_path.write_bytes(ev.canonical_json_bytes(forged.model_dump(mode="json")))
    with pytest.raises(bd.BundleError, match="conflicting portable logical paths"):
        _collect(record)


def test_export_revalidation_rejects_bad_portable_entry_name(tmp_path: Path) -> None:
    record, _run_id = _record_dir(tmp_path)
    (record / "portable" / "entries" / "weird.json").write_bytes(b"{}\n")
    with pytest.raises(bd.BundleError, match="unexpected run record portable entry"):
        _collect(record)


def test_read_child_regular_missing_fails_closed(tmp_path: Path) -> None:
    fd = os.open(tmp_path, bd._DIR_OPEN_FLAGS)
    try:
        with pytest.raises(bd.BundleError, match="member is unavailable"):
            bd._read_child_regular(fd, "nope", "nope")
    finally:
        os.close(fd)


def test_try_open_owned_child_dir_on_regular_file(tmp_path: Path) -> None:
    (tmp_path / "afile").write_bytes(b"x\n")
    fd = os.open(tmp_path, bd._DIR_OPEN_FLAGS)
    try:
        with pytest.raises(bd.BundleError, match="subdirectory is unavailable"):
            bd._try_open_owned_child_dir(fd, "afile")
    finally:
        os.close(fd)


def test_collect_events_requires_events(tmp_path: Path) -> None:
    empty = tmp_path / "rec"
    empty.mkdir(mode=0o700)
    fd = os.open(empty, bd._DIR_OPEN_FLAGS)
    try:
        with pytest.raises(bd.BundleError, match="omits its event chain"):
            bd._collect_events(fd, lambda *_a: None)
    finally:
        os.close(fd)


def test_bundle_relative_rejects_traversal_and_backslash() -> None:
    for bad in ["../x", "/abs", "a\\b", "a//b", "a\nb", "", ".", "a/" + "x" * 300, "a\x7fb"]:
        with pytest.raises(bd.BundleError):
            bd._bundle_relative(bad)


def test_bundle_policy_is_single_sourced_from_evidence() -> None:
    # The bundle limits, numbered-file grammar, and role/media policy are the same
    # objects evidence exports — bundles.py declares none of these literals itself.
    assert bd.MAX_MEMBER_BYTES is ev.MAX_MEMBER_BYTES
    assert bd.MAX_AGGREGATE_BYTES is ev.MAX_AGGREGATE_BYTES
    assert bd.MAX_TOTAL_FILES is ev.MAX_TOTAL_FILES
    assert bd.MAX_PORTABLE_ENTRIES is ev.MAX_PORTABLE_ENTRIES
    assert bd.BLOB_NAME_RE is ev.BLOB_NAME_RE
    assert bd.NUMBERED_JSON_NAME_RE is ev.NUMBERED_JSON_NAME_RE
    assert bd.PORTABLE_ROLES is ev.PORTABLE_ROLES
    assert bd.PORTABLE_MEDIA_TYPES is ev.PORTABLE_MEDIA_TYPES


def test_evidence_event_and_entry_name_grammar_is_unified() -> None:
    # The committed event and portable-entry filename grammars are one compiled regex.
    assert ev._EVENT_NAME_RE is ev._ENTRY_NAME_RE is ev._NUMBERED_JSON_RE
    assert ev._EVENT_TEMP_RE is ev._ENTRY_TEMP_RE is ev._NUMBERED_JSON_TEMP_RE


def test_bundle_relative_reuses_run_relative_syntax(tmp_path: Path) -> None:
    # A valid path resolves; an unsafe one raises BundleError (translated RunError).
    assert bd._bundle_relative("run/status.json").as_posix() == "run/status.json"
    with pytest.raises(bd.BundleError, match="unsafe bundle path"):
        bd._bundle_relative("../escape")


def test_publish_fails_closed_on_stage_swap_before_rename(tmp_path: Path, monkeypatch) -> None:
    # Swap the verified stage directory for a different inode between verification and
    # the publishing rename; the post-rename inode binding must fail closed.
    home = tmp_path / "home"
    run_id = _finalized_run(home)
    destination = tmp_path / "bundle"
    real_rename = bd.rename_noreplace_at

    def hook(old_dir_fd, old_name, new_dir_fd, new_name):
        evil = f"{old_name}.evil"
        os.mkdir(evil, 0o700, dir_fd=old_dir_fd)
        os.rename(old_name, f"{old_name}.stash", src_dir_fd=old_dir_fd, dst_dir_fd=old_dir_fd)
        os.rename(evil, old_name, src_dir_fd=old_dir_fd, dst_dir_fd=old_dir_fd)
        return real_rename(old_dir_fd, old_name, new_dir_fd, new_name)

    monkeypatch.setattr(bd, "rename_noreplace_at", hook)
    with pytest.raises(bd.BundleError, match="diverged from its verified stage"):
        bd.export_bundle(run_id, destination, home=home, environ=_ENV)
    # The divergent (unverified) publication must not remain at the destination.
    assert not destination.exists()


# ------------------------------------ streaming aggregate bound (finding 4)


def test_verify_aborts_before_reading_after_aggregate_crossing(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "home"
    bundle = _export(home, tmp_path)
    reads: list[str] = []
    real_read = bd._read_child_regular

    def spy(dir_fd, name, describe):
        reads.append(describe)
        return real_read(dir_fd, name, describe)

    monkeypatch.setattr(bd, "_read_child_regular", spy)
    # A limit smaller than any member forces the first regular file to cross it, so no
    # member is read into memory at all.
    monkeypatch.setattr(bd, "MAX_AGGREGATE_BYTES", 3)
    with pytest.raises(bd.BundleError, match="aggregate payload limit"):
        bd.verify_bundle(bundle)
    assert reads == []  # the crossing member (and all later members) were never read


# ------------------------------------ staging durability (finding 5)


def test_export_fsyncs_every_intermediate_bundle_directory(tmp_path: Path, monkeypatch) -> None:
    import stat as stat_mod

    fsynced: set[tuple[int, int]] = set()
    real_fsync = os.fsync

    def spy(fd):
        try:
            meta = os.fstat(fd)
            if stat_mod.S_ISDIR(meta.st_mode):
                fsynced.add((meta.st_dev, meta.st_ino))
        except OSError:
            pass
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", spy)
    home = tmp_path / "home"
    run_id = _finalized_run(home)
    destination = tmp_path / "bundle"
    bd.export_bundle(run_id, destination, home=home, environ=_ENV)
    # The stage was renamed to the destination, so its subtree inodes are the destination
    # inodes: assert every intermediate directory was fsynced during staging.
    for sub in ("run", "run/portable", "run/portable/blobs", "run/portable/entries", "run/events"):
        meta = (destination / sub).stat()
        assert (meta.st_dev, meta.st_ino) in fsynced

"""Deterministic, read-only run-evidence bundle-directory skeleton.

A bundle is a deterministic immutable directory (not an archive) holding only the
control files and entry-authenticated portable evidence admitted by a closed v1
role/media policy. Export first verifies the finalized run and publishes a staged
directory no-replace. Verification is bounded and read-only: no extraction, links,
special files, duplicate paths, unsafe paths, undeclared members, or unchecked
bytes. BUNDLE-001 will add a portable archive container and hostile-archive parser.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import secrets
import shutil
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from strixlab.evidence import (
    BLOB_NAME_RE,
    MAX_AGGREGATE_BYTES,
    MAX_MEMBER_BYTES,
    MAX_PORTABLE_ENTRIES,
    MAX_TOTAL_FILES,
    NUMBERED_JSON_NAME_RE,
    PORTABLE_MEDIA_TYPES,
    PORTABLE_ROLES,
    PortableEvidenceV1,
    RunDescriptorV1,
    RunError,
    RunState,
    RunStatusV1,
    parse_checksums,
    run_relative,
    validate_event_chain,
    validate_portable_payload,
)
from strixlab.records import RecordManifestV1, record_manifest_digest
from strixlab.secret_policy import RedactionContext, UnsafeOutputError
from strixlab.secure_fs import (
    UnownedDirectoryError,
    directory_open_flags,
    exclusive_create_flags,
    open_owned_directory,
    rename_noreplace_at,
    try_open_owned_directory,
    write_all,
)
from strixlab.serialization import canonical_json_bytes

# The bundle limits, blob/numbered-file grammar, and portable role/media policy are
# the run-evidence policy (imported from evidence); this module must not redeclare them.
_BUNDLE_MANIFEST = "bundle.json"
_RUN_PREFIX = "run/"
_DIR_OPEN_FLAGS = directory_open_flags()

# Control members and their declared media types. Portable entries/blobs are added
# from authenticated portable metadata.
_CONTROL_MEDIA: dict[str, str] = {
    "record-manifest.json": "application/json",
    "run.json": "application/json",
    "status.json": "application/json",
    "manifest.input.yaml": "application/yaml",
    "manifest.resolved.yaml": "application/yaml",
    "checksums.sha256": "text/plain",
}
_EVENT_MEDIA = "application/json"


class BundleError(RuntimeError):
    """A run-evidence bundle is unsafe, incomplete, or divergent."""


class BundleMemberV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    media_type: str


class BundleManifestV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    kind: Literal["run-evidence-bundle"] = "run-evidence-bundle"
    run_id: str
    run_record_sha256: str = Field(pattern=r"^record-sha256:[0-9a-f]{64}$")
    outcome: str
    members: tuple[BundleMemberV1, ...]


@dataclass(frozen=True, slots=True)
class BundleInspection:
    path: Path
    run_id: str
    outcome: str
    run_record_sha256: str
    member_count: int


# ----------------------------------------------------------------------------- read


def _bundle_relative(value: str) -> PurePosixPath:
    """Validate one bundle-relative path via the shared run-path syntax predicate.

    Reuses :func:`run_relative` (the single run-path syntax validator) and translates
    its ``RunError`` into a bundle-facing ``BundleError``.
    """

    try:
        return run_relative(value)
    except RunError as exc:
        raise BundleError(f"unsafe bundle path: {value!r}") from exc


def _open_owned_dir(path: Path) -> int:
    """Open ``path`` no-follow as an owned directory and return its descriptor."""

    try:
        return open_owned_directory(path)
    except UnownedDirectoryError as exc:
        raise BundleError(f"bundle path is not an owned directory: {path}") from exc
    except OSError as exc:
        raise BundleError(f"bundle directory is unavailable: {path}") from exc


def _try_open_owned_child_dir(parent_fd: int, name: str) -> int | None:
    """Open child directory ``name`` no-follow, or ``None`` when it does not exist."""

    try:
        return try_open_owned_directory(name, dir_fd=parent_fd)
    except UnownedDirectoryError as exc:
        raise BundleError(f"unsafe bundle subdirectory: {name}") from exc
    except OSError as exc:
        raise BundleError(f"bundle subdirectory is unavailable: {name}") from exc


def _read_child_regular(dir_fd: int, name: str, describe: str) -> bytes:
    """Read one owned, non-executable regular child ``name`` held under ``dir_fd``.

    A single-component ``openat`` relative to the held directory descriptor, with an
    ``lstat``/``fstat`` device+inode identity check across the open, so a same-uid
    rename or replacement of the child between enumeration and read cannot redirect
    the read. No pathname is ever re-resolved from an ancestor.
    """

    euid = os.geteuid()
    try:
        pre = os.lstat(name, dir_fd=dir_fd)
    except OSError as exc:
        raise BundleError(f"bundle member is unavailable: {describe}") from exc
    if not stat.S_ISREG(pre.st_mode) or pre.st_uid != euid:
        raise BundleError(f"bundle member is not an owned regular file: {describe}")
    if pre.st_mode & 0o111:
        raise BundleError(f"bundle member is executable: {describe}")
    if pre.st_size > MAX_MEMBER_BYTES:
        raise BundleError(f"bundle member exceeds the size limit: {describe}")
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        raise BundleError(f"bundle member is unavailable: {describe}") from exc
    try:
        post = os.fstat(descriptor)
        if (
            not stat.S_ISREG(post.st_mode)
            or post.st_uid != euid
            or post.st_mode & 0o111
            or post.st_dev != pre.st_dev
            or post.st_ino != pre.st_ino
        ):
            raise BundleError(f"bundle member changed during verification: {describe}")
        content = os.read(descriptor, MAX_MEMBER_BYTES + 1)
        if len(content) > MAX_MEMBER_BYTES:
            raise BundleError(f"bundle member exceeds the size limit: {describe}")
        return content
    finally:
        os.close(descriptor)


def _walk_and_read(root_fd: int) -> dict[str, bytes]:
    """Enumerate and read every regular member in one descriptor-held traversal.

    Each subtree is walked with its directory descriptor held open (identity-checked
    across the open), and every regular child is read while that descriptor is still
    held. Enumeration and read never reopen an intermediate component by pathname, so
    a same-uid rename of a directory between listing and reading cannot redirect a
    read. Symlinks, special files, and foreign-owned entries fail closed.
    """

    members: dict[str, bytes] = {}
    euid = os.geteuid()
    aggregate = 0

    def walk(dir_fd: int, prefix: str) -> None:
        nonlocal aggregate
        for name in os.listdir(dir_fd):
            relative = f"{prefix}{name}"
            _bundle_relative(relative)
            metadata = os.lstat(name, dir_fd=dir_fd)
            if stat.S_ISDIR(metadata.st_mode):
                if metadata.st_uid != euid:
                    raise BundleError(f"bundle contains an unsafe directory: {relative}")
                child = os.open(name, _DIR_OPEN_FLAGS, dir_fd=dir_fd)
                try:
                    inner = os.fstat(child)
                    if inner.st_dev != metadata.st_dev or inner.st_ino != metadata.st_ino:
                        raise BundleError(f"bundle directory changed during walk: {relative}")
                    walk(child, f"{relative}/")
                finally:
                    os.close(child)
            elif stat.S_ISREG(metadata.st_mode) and metadata.st_uid == euid:
                if len(members) + 1 > MAX_TOTAL_FILES:
                    raise BundleError("bundle exceeds the total file limit")
                # Enforce the aggregate bound during traversal so a member that would
                # cross it — and every later member — is never read into memory.
                aggregate += metadata.st_size
                if aggregate > MAX_AGGREGATE_BYTES:
                    raise BundleError("bundle exceeds the aggregate payload limit")
                members[relative] = _read_child_regular(dir_fd, name, relative)
            else:
                raise BundleError(f"bundle contains an unsafe directory entry: {relative}")

    walk(root_fd, "")
    return members


def verify_bundle(path: Path) -> BundleInspection:
    """Open and fully verify a bundle directory, bounded and read-only.

    The whole directory is enumerated and read in one descriptor-held traversal into
    an in-memory snapshot; all subsequent binding checks run against that snapshot, so
    there is no window between enumeration and read for a swap to redirect anything.
    """

    root_fd = _open_owned_dir(path)
    try:
        snapshot = _walk_and_read(root_fd)
    finally:
        os.close(root_fd)
    if _BUNDLE_MANIFEST not in snapshot:
        raise BundleError("bundle is missing bundle.json")
    manifest_bytes = snapshot[_BUNDLE_MANIFEST]
    try:
        manifest = BundleManifestV1.model_validate_json(manifest_bytes, strict=True)
    except ValidationError as exc:
        raise BundleError("bundle.json is invalid") from exc
    if canonical_json_bytes(manifest.model_dump(mode="json")) != manifest_bytes:
        raise BundleError("bundle.json is not canonical")
    contents = {name: data for name, data in snapshot.items() if name != _BUNDLE_MANIFEST}
    _verify_snapshot(manifest, contents, membership=set(snapshot))
    return BundleInspection(
        path=path,
        run_id=manifest.run_id,
        outcome=manifest.outcome,
        run_record_sha256=manifest.run_record_sha256,
        member_count=len(manifest.members),
    )


def _verify_snapshot(
    manifest: BundleManifestV1,
    contents: Mapping[str, bytes],
    *,
    membership: set[str],
    digests: Mapping[str, str] | None = None,
) -> None:
    """Validate a fully-read in-memory bundle snapshot against its manifest.

    Shared by ``verify_bundle`` (reading from disk) and ``export_bundle`` (validating
    the just-collected bytes before publishing), so both apply byte-for-byte identical
    membership, size, and record/checksum/event/manifest/portable binding rules to the
    exact bytes that will exist on disk. ``digests`` lets the export path supply the
    digests it already computed while collecting the members (which are of the very
    same bytes), avoiding a redundant rehash of every member; standalone verification
    passes ``None`` and re-hashes the bytes it read back from disk. The record-manifest
    digest, checksum, and event-chain bindings always recompute from ``contents``, so
    a mutation between inspection and collection is still caught regardless.
    """

    declared = {member.path: member for member in manifest.members}
    if len(declared) != len(manifest.members):
        raise BundleError("bundle.json declares a duplicate member")
    if _BUNDLE_MANIFEST in declared:
        raise BundleError("bundle.json must not declare itself")
    if membership != {_BUNDLE_MANIFEST} | set(declared):
        raise BundleError("bundle membership does not match its manifest")
    aggregate = 0
    record_manifest: RecordManifestV1 | None = None
    checksums: dict[str, str] = {}
    for member_path, member in sorted(declared.items()):
        content = contents[member_path]
        actual = (
            digests[member_path] if digests is not None else hashlib.sha256(content).hexdigest()
        )
        if actual != member.sha256:
            raise BundleError(f"bundle member digest mismatch: {member_path}")
        if len(content) != member.size_bytes:
            raise BundleError(f"bundle member size mismatch: {member_path}")
        aggregate += member.size_bytes
        if member_path == "run/record-manifest.json":
            record_manifest = _parse_record_manifest(content)
        elif member_path == "run/checksums.sha256":
            checksums = parse_checksums(content)
    if aggregate > MAX_AGGREGATE_BYTES:
        raise BundleError("bundle exceeds the aggregate payload limit")
    if record_manifest is None:
        raise BundleError("bundle omits the embedded record manifest")
    _verify_bindings(manifest, declared, contents, record_manifest, checksums)


def _parse_record_manifest(content: bytes) -> RecordManifestV1:
    try:
        manifest = RecordManifestV1.model_validate_json(content, strict=True)
    except ValidationError as exc:
        raise BundleError("embedded record manifest is invalid") from exc
    paths = [entry.path for entry in manifest.files]
    if len(set(paths)) != len(paths):
        raise BundleError("embedded record manifest declares duplicate paths")
    return manifest


def _verify_bindings(
    manifest: BundleManifestV1,
    declared: Mapping[str, BundleMemberV1],
    contents: Mapping[str, bytes],
    record_manifest: RecordManifestV1,
    checksums: Mapping[str, str],
) -> None:
    record_bytes = contents["run/record-manifest.json"]
    if record_manifest_digest(record_bytes) != manifest.run_record_sha256:
        raise BundleError("bundle run-record digest diverged from its manifest")
    if canonical_json_bytes(record_manifest.model_dump(mode="json")) != record_bytes:
        raise BundleError("embedded record manifest is not canonical")
    record_files = {entry.path: entry for entry in record_manifest.files}
    status = _verify_run_binding(manifest, contents)
    _verify_event_chain(contents, status)
    _verify_checksum_binding(record_files, checksums)
    portable_entries = _authenticated_portable_entries(declared, contents)
    referenced_blobs = {entry.blob_sha256 for entry in portable_entries.values()}
    declared_blobs: set[str] = set()
    for member_path, member in declared.items():
        if member_path == "run/record-manifest.json":
            continue
        if not member_path.startswith(_RUN_PREFIX):
            raise BundleError(f"bundle member is outside run/: {member_path}")
        run_rel = member_path.removeprefix(_RUN_PREFIX)
        record_entry = record_files.get(run_rel)
        if record_entry is None or record_entry.sha256 != member.sha256:
            raise BundleError(f"bundle member is not in the immutable record: {member_path}")
        if record_entry.size_bytes != member.size_bytes:
            raise BundleError(f"bundle member size diverged from the record: {member_path}")
        if record_entry.mode & 0o111:
            raise BundleError(f"immutable record marks member executable: {member_path}")
        if run_rel != "checksums.sha256":
            declared_digest = checksums.get(run_rel)
            if declared_digest is None or declared_digest != member.sha256:
                raise BundleError(f"bundle member diverged from checksums: {member_path}")
        _classify_member(
            member_path,
            run_rel,
            member,
            referenced_blobs,
            portable_entries,
            contents,
            declared_blobs,
        )
    # Every referenced blob must be present exactly once; no extra, no missing.
    if declared_blobs != referenced_blobs:
        raise BundleError("bundle portable blob set diverges from its entries")


def _verify_run_binding(manifest: BundleManifestV1, contents: Mapping[str, bytes]) -> RunStatusV1:
    """Bind the manifest run identity, outcome, and manifest digests to control files."""

    descriptor = _parse_run_model(contents.get("run/run.json"), RunDescriptorV1, "run.json")
    status = _parse_run_model(contents.get("run/status.json"), RunStatusV1, "status.json")
    if descriptor.run_id != manifest.run_id or status.run_id != manifest.run_id:
        raise BundleError("bundle run identity does not match its control files")
    if status.state is not RunState.TERMINAL:
        raise BundleError("bundle run is not in a terminal state")
    # status.outcome is always a valid RunOutcome, so binding the manifest outcome
    # to it also rejects any unknown or forged outcome string.
    if status.outcome is None or str(status.outcome) != manifest.outcome:
        raise BundleError("bundle outcome does not match its status")
    # Bind the descriptor's captured manifest digests to the bundled manifest bytes.
    input_bytes = contents.get("run/manifest.input.yaml")
    resolved_bytes = contents.get("run/manifest.resolved.yaml")
    if input_bytes is None or resolved_bytes is None:
        raise BundleError("bundle omits a scoping manifest")
    if descriptor.input_manifest_sha256 != hashlib.sha256(input_bytes).hexdigest():
        raise BundleError("bundle input manifest diverged from the run descriptor")
    if descriptor.resolved_manifest_sha256 != hashlib.sha256(resolved_bytes).hexdigest():
        raise BundleError("bundle resolved manifest diverged from the run descriptor")
    return status


def _verify_event_chain(contents: Mapping[str, bytes], status: RunStatusV1) -> None:
    """Authenticate the embedded event chain via the shared run-chain validator."""

    event_keys = sorted(key for key in contents if key.startswith("run/events/"))
    names = [key.removeprefix("run/events/") for key in event_keys]
    expected = [f"{index:08d}.json" for index in range(1, len(names) + 1)]
    if names != expected:
        raise BundleError("bundle events are not an exact contiguous sequence")
    try:
        validate_event_chain(
            [contents[key] for key in event_keys], run_id=status.run_id, status=status
        )
    except RunError as exc:
        raise BundleError(f"bundle event chain is invalid: {exc}") from exc


def _parse_run_model[ModelT: BaseModel](
    data: bytes | None, model: type[ModelT], describe: str
) -> ModelT:
    if data is None:
        raise BundleError(f"bundle omits run/{describe}")
    try:
        return model.model_validate_json(data, strict=True)
    except ValidationError as exc:
        raise BundleError(f"bundle run/{describe} is invalid") from exc


def _verify_checksum_binding(
    record_files: Mapping[str, object], checksums: Mapping[str, str]
) -> None:
    """Checksums must cover exactly the record payload set, minus checksums itself."""

    expected = set(record_files) - {"checksums.sha256"}
    if set(checksums) != expected:
        raise BundleError("bundle checksums do not cover the exact record payload set")
    for path, digest in checksums.items():
        entry = record_files[path]
        if getattr(entry, "sha256", None) != digest:
            raise BundleError(f"bundle checksum diverged from the immutable record: {path}")


def _classify_member(
    member_path: str,
    run_rel: str,
    member: BundleMemberV1,
    referenced_blobs: set[str],
    portable_entries: Mapping[int, PortableEvidenceV1],
    contents: Mapping[str, bytes],
    declared_blobs: set[str],
) -> None:
    if run_rel.startswith("portable/blobs/"):
        blob = run_rel.removeprefix("portable/blobs/")
        if BLOB_NAME_RE.match(blob) is None or blob != member.sha256:
            raise BundleError(f"portable blob filename is not its content digest: {member_path}")
        if blob not in referenced_blobs:
            raise BundleError(f"bundle includes an unreferenced portable blob: {member_path}")
        referencing = [e for e in portable_entries.values() if e.blob_sha256 == blob]
        medias = {e.media_type for e in referencing}
        sizes = {e.size_bytes for e in referencing}
        if len(medias) > 1:
            raise BundleError(f"portable blob shared under conflicting media types: {member_path}")
        if member.media_type not in medias or member.size_bytes not in sizes:
            raise BundleError(f"portable blob metadata diverged from its entry: {member_path}")
        _validate_blob_payload(contents[member_path], member.media_type, member_path)
        declared_blobs.add(blob)
    elif run_rel.startswith("portable/entries/"):
        if member.media_type != _EVENT_MEDIA:
            raise BundleError(f"portable entry has the wrong media type: {member_path}")
    elif run_rel.startswith("events/"):
        if member.media_type != _EVENT_MEDIA:
            raise BundleError(f"event has the wrong media type: {member_path}")
    else:
        expected = _CONTROL_MEDIA.get(run_rel)
        if expected is None or member.media_type != expected:
            raise BundleError(f"control member has the wrong media type: {member_path}")


def _validate_blob_payload(data: bytes, media_type: str, member_path: str) -> None:
    try:
        validate_portable_payload(data, media_type)
    except RunError as exc:
        raise BundleError(f"portable blob violates the payload policy: {member_path}") from exc


def _authenticated_portable_entries(
    declared: Mapping[str, BundleMemberV1], contents: Mapping[str, bytes]
) -> dict[int, PortableEvidenceV1]:
    entries: dict[int, PortableEvidenceV1] = {}
    logical_paths: set[str] = set()
    media_by_blob: dict[str, str] = {}
    prefix = "run/portable/entries/"
    for member_path in sorted(declared):
        if not member_path.startswith(prefix):
            continue
        name = member_path.removeprefix(prefix)
        if NUMBERED_JSON_NAME_RE.match(name) is None:
            raise BundleError(f"malformed portable entry filename: {member_path}")
        try:
            entry = PortableEvidenceV1.model_validate_json(contents[member_path], strict=True)
        except ValidationError as exc:
            raise BundleError(f"portable entry is invalid: {member_path}") from exc
        if name != f"{entry.sequence:08d}.json":
            raise BundleError(f"portable entry filename does not match its sequence: {member_path}")
        if entry.role not in PORTABLE_ROLES or entry.media_type not in PORTABLE_MEDIA_TYPES:
            raise BundleError(
                f"portable entry has an out-of-policy role or media type: {member_path}"
            )
        run_relative(entry.logical_path)
        if entry.sequence in entries:
            raise BundleError("bundle has duplicate portable entry sequences")
        if entry.logical_path in logical_paths:
            raise BundleError("bundle has conflicting portable logical paths")
        media = media_by_blob.setdefault(entry.blob_sha256, entry.media_type)
        if media != entry.media_type:
            raise BundleError("bundle shares a portable blob under conflicting media types")
        logical_paths.add(entry.logical_path)
        entries[entry.sequence] = entry
    if len(entries) > MAX_PORTABLE_ENTRIES:
        raise BundleError("bundle exceeds the portable entry limit")
    ordered = sorted(entries)
    if ordered != list(range(1, len(ordered) + 1)):
        raise BundleError("bundle portable entries are noncontiguous")
    return entries


# --------------------------------------------------------------------------- export


def export_bundle(
    run_id: str, destination: Path, *, home: Path, environ: Mapping[str, str]
) -> Path:
    """Verify a finalized run, then publish its deterministic bundle directory.

    The collected in-memory snapshot is validated with the exact same binding logic
    ``verify_bundle`` uses **before** anything is published, so a concurrent mutation
    of the record between ``inspect_run`` and collection fails closed with no
    destination created.
    """

    from strixlab.evidence import inspect_run  # local import avoids an import cycle

    inspection = inspect_run(run_id, home=home)
    context = RedactionContext.from_environ(environ)
    members = _collect_members(inspection.record, context)
    manifest = BundleManifestV1(
        run_id=run_id,
        run_record_sha256=inspection.record_sha256,
        outcome=str(inspection.outcome),
        members=tuple(
            BundleMemberV1(path=path, sha256=digest, size_bytes=len(content), media_type=media)
            for path, (content, digest, media) in sorted(members.items())
        ),
    )
    contents = {path: content for path, (content, _digest, _media) in members.items()}
    digests = {path: digest for path, (_content, digest, _media) in members.items()}
    _verify_snapshot(
        manifest, contents, membership={_BUNDLE_MANIFEST} | set(contents), digests=digests
    )
    _publish_bundle(destination, manifest, members)
    return destination


def _collect_members(record: Path, context: RedactionContext) -> dict[str, tuple[bytes, str, str]]:
    """Return ``bundle path -> (bytes, sha256, media_type)`` for every allowlisted file.

    Every child is read while its own directory descriptor is held (single-component
    ``openat`` with an identity check), so a same-uid rename of an intermediate
    directory between enumeration and read cannot redirect any read. The portable
    surface is revalidated here independently of earlier finalization.
    """

    record_fd = _open_owned_dir(record)
    try:
        members: dict[str, tuple[bytes, str, str]] = {}

        def add(run_rel: str, content: bytes, media_type: str) -> None:
            _fail_on_secret(context, f"{_RUN_PREFIX}{run_rel}", content)
            members[f"{_RUN_PREFIX}{run_rel}"] = (
                content,
                hashlib.sha256(content).hexdigest(),
                media_type,
            )

        for control, media in _CONTROL_MEDIA.items():
            add(control, _read_child_regular(record_fd, control, control), media)
        _collect_events(record_fd, add)
        _collect_portable(record_fd, add)
        if len(members) + 1 > MAX_TOTAL_FILES:
            raise BundleError("bundle exceeds the total file limit")
        if sum(len(content) for content, _d, _m in members.values()) > MAX_AGGREGATE_BYTES:
            raise BundleError("bundle exceeds the aggregate payload limit")
        return members
    finally:
        os.close(record_fd)


def _collect_events(record_fd: int, add: Callable[[str, bytes, str], None]) -> None:
    events_fd = _try_open_owned_child_dir(record_fd, "events")
    if events_fd is None:
        raise BundleError("run record omits its event chain")
    try:
        names = sorted(os.listdir(events_fd))
        for name in names:
            if NUMBERED_JSON_NAME_RE.match(name) is None:
                raise BundleError(f"unexpected run record event member: {name}")
        if names != [f"{index:08d}.json" for index in range(1, len(names) + 1)]:
            raise BundleError("run record events are not an exact contiguous sequence")
        for name in names:
            rel = f"events/{name}"
            add(rel, _read_child_regular(events_fd, name, rel), _EVENT_MEDIA)
    finally:
        os.close(events_fd)


def _collect_portable(record_fd: int, add: Callable[[str, bytes, str], None]) -> None:
    portable_fd = _try_open_owned_child_dir(record_fd, "portable")
    if portable_fd is None:
        return
    try:
        entries_fd = _try_open_owned_child_dir(portable_fd, "entries")
        blobs_fd = _try_open_owned_child_dir(portable_fd, "blobs")
        try:
            if entries_fd is None:
                return
            entries = _collect_entries(entries_fd, add)
            _collect_blobs(blobs_fd, entries, add)
        finally:
            if blobs_fd is not None:
                os.close(blobs_fd)
            if entries_fd is not None:
                os.close(entries_fd)
    finally:
        os.close(portable_fd)


def _collect_entries(
    entries_fd: int, add: Callable[[str, bytes, str], None]
) -> tuple[PortableEvidenceV1, ...]:
    names = sorted(os.listdir(entries_fd))
    result: list[PortableEvidenceV1] = []
    seen: set[str] = set()
    media_by_blob: dict[str, str] = {}
    for index, name in enumerate(names, start=1):
        if NUMBERED_JSON_NAME_RE.match(name) is None:
            raise BundleError(f"unexpected run record portable entry: {name}")
        content = _read_child_regular(entries_fd, name, f"portable/entries/{name}")
        try:
            entry = PortableEvidenceV1.model_validate_json(content, strict=True)
        except ValidationError as exc:
            raise BundleError(f"run record portable entry is invalid: {name}") from exc
        if name != f"{entry.sequence:08d}.json" or entry.sequence != index:
            raise BundleError(f"portable entry filename does not match its sequence: {name}")
        if entry.role not in PORTABLE_ROLES or entry.media_type not in PORTABLE_MEDIA_TYPES:
            raise BundleError(f"portable entry has an out-of-policy role or media type: {name}")
        run_relative(entry.logical_path)
        if entry.logical_path in seen:
            raise BundleError("run record has conflicting portable logical paths")
        media = media_by_blob.setdefault(entry.blob_sha256, entry.media_type)
        if media != entry.media_type:
            raise BundleError("portable blob is shared under conflicting media types")
        seen.add(entry.logical_path)
        add(f"portable/entries/{name}", content, _EVENT_MEDIA)
        result.append(entry)
    if len(result) > MAX_PORTABLE_ENTRIES:
        raise BundleError("run record exceeds the portable entry limit")
    return tuple(result)


def _collect_blobs(
    blobs_fd: int | None,
    entries: tuple[PortableEvidenceV1, ...],
    add: Callable[[str, bytes, str], None],
) -> None:
    referenced: dict[str, tuple[str, int]] = {}
    for entry in entries:
        referenced[entry.blob_sha256] = (entry.media_type, entry.size_bytes)
    if referenced and blobs_fd is None:
        raise BundleError("run record omits referenced portable blobs")
    for blob in sorted(referenced):
        media, size = referenced[blob]
        assert blobs_fd is not None
        content = _read_child_regular(blobs_fd, blob, f"portable/blobs/{blob}")
        if hashlib.sha256(content).hexdigest() != blob:
            raise BundleError(f"referenced portable blob is not content-addressed: {blob}")
        if len(content) != size:
            raise BundleError(f"referenced portable blob size diverged from its entry: {blob}")
        add(f"portable/blobs/{blob}", content, media)


def _fail_on_secret(context: RedactionContext, path: str, content: bytes) -> None:
    try:
        context.assert_text_safe(path)
        context.assert_payload_safe(content)
    except UnsafeOutputError as exc:
        raise BundleError(f"bundle member discloses a sensitive value: {path}") from exc


def _child_exists(dir_fd: int, name: str) -> bool:
    try:
        os.lstat(name, dir_fd=dir_fd)
    except FileNotFoundError:
        return False
    return True


def _publish_bundle(
    destination: Path,
    manifest: BundleManifestV1,
    members: Mapping[str, tuple[bytes, str, str]],
) -> None:
    """Publish the bundle through one held, authenticated parent directory fd.

    The sibling stage is created, written, and renamed no-replace entirely relative
    to the parent descriptor, so a parent swapped for a symlink after validation
    cannot redirect creation or publication.
    """

    parent = destination.parent
    parent_fd = _open_owned_dir(parent)
    try:
        if _child_exists(parent_fd, destination.name):
            raise BundleError(f"bundle destination already exists: {destination}")
        stage_name = f".{destination.name}.{secrets.token_hex(16)}.tmp"
        os.mkdir(stage_name, 0o700, dir_fd=parent_fd)
        os.fsync(parent_fd)
        try:
            # Hold the verified stage descriptor across the rename and bind the
            # published destination to that exact inode: a same-UID swap of the stage
            # name between verification and rename cannot publish an unverified tree.
            stage_fd = os.open(stage_name, _DIR_OPEN_FLAGS, dir_fd=parent_fd)
            try:
                _write_bundle_tree(stage_fd, manifest, members)
                os.fsync(stage_fd)
                staged = os.fstat(stage_fd)
                rename_noreplace_at(parent_fd, stage_name, parent_fd, destination.name)
                published = os.open(destination.name, _DIR_OPEN_FLAGS, dir_fd=parent_fd)
                try:
                    metadata = os.fstat(published)
                finally:
                    os.close(published)
                if metadata.st_dev != staged.st_dev or metadata.st_ino != staged.st_ino:
                    # The published name is not our verified stage inode. Remove the
                    # detected divergent publication, then fail the export. The owning
                    # UID is the documented local-storage trust boundary.
                    with contextlib.suppress(OSError):
                        shutil.rmtree(destination.name, dir_fd=parent_fd)
                    raise BundleError("published bundle diverged from its verified stage")
                os.fsync(parent_fd)
            finally:
                os.close(stage_fd)
        except BaseException:
            with contextlib.suppress(OSError):
                shutil.rmtree(stage_name, dir_fd=parent_fd)
            raise
    finally:
        os.close(parent_fd)


def _write_bundle_tree(
    stage_fd: int,
    manifest: BundleManifestV1,
    members: Mapping[str, tuple[bytes, str, str]],
) -> None:
    for path, (content, _digest, _media) in members.items():
        _write_anchored(stage_fd, _bundle_relative(path), content)
    _write_anchored(
        stage_fd,
        PurePosixPath(_BUNDLE_MANIFEST),
        canonical_json_bytes(manifest.model_dump(mode="json")),
    )


def _write_anchored(root_fd: int, relative: PurePosixPath, content: bytes) -> None:
    *parents, filename = relative.parts
    opened: list[int] = []
    dir_fd = root_fd
    try:
        for name in parents:
            created = True
            try:
                os.mkdir(name, 0o700, dir_fd=dir_fd)
            except FileExistsError:
                created = False
            child = os.open(name, _DIR_OPEN_FLAGS, dir_fd=dir_fd)
            # Flush the parent's new directory entry so a crash cannot lose an
            # intermediate directory whose file we later fsynced.
            if created:
                os.fsync(dir_fd)
            opened.append(child)
            dir_fd = child
        descriptor = os.open(filename, exclusive_create_flags(), 0o600, dir_fd=dir_fd)
        try:
            write_all(descriptor, content)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(dir_fd)
    finally:
        for fd in reversed(opened):
            os.close(fd)

"""Verified local model registry: inspection, verification, receipts, and leases.

MODEL-001's first trust boundary above the adapters. It has three explicit trust
states:

* ``unregistered`` -- a local GGUF was safely inspected but no registered manifest
  claims it (:class:`ModelObservationV1`, practice evidence only);
* ``registered`` -- a manifest pins upstream/local identity, exact size/SHA-256, and
  compatibility predicates (:class:`strixlab.manifests.ModelManifestV1`); local
  presence is not implied;
* ``verified`` -- the local primary artifact and every required sidecar matched the
  registered manifest and stable-file checks, and the pinned inspector output passed
  (:class:`ModelReceiptV1`).

This module is not a downloader, converter, quantizer, suite, CLI, or general artifact
database, and it never fetches weights, keeps model bytes in memory beyond a bounded
hash, or copies model artifacts into StrixLab's home.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from strixlab.build_paths import is_unsafe_directory, prepare_directory_tree
from strixlab.executable_identity import hash_executable
from strixlab.locks import LockStatus, wait_for_exclusive_lock
from strixlab.manifests import (
    DASH_ID_PATTERN,
    AbsolutePathString,
    MetadataPredicateV1,
    ModelManifestV1,
    Sha256Lower,
)
from strixlab.secure_fs import (
    ensure_directory_fsynced,
    fsync_directory,
    readonly_open_flags,
    rename_noreplace,
    write_exclusive,
)
from strixlab.serialization import canonical_json_bytes

if TYPE_CHECKING:
    from strixlab.sources import SourceLease

__all__ = [
    "GgufInspectorBindingV1",
    "GgufMetadataV1",
    "ModelArtifactError",
    "ModelCacheBusyError",
    "ModelCacheError",
    "ModelCompatibilityError",
    "ModelError",
    "ModelFileReceiptV1",
    "ModelInspectorError",
    "ModelLease",
    "ModelManifestError",
    "ModelMetadataError",
    "ModelObservationV1",
    "ModelReceiptEvidenceV1",
    "ModelReceiptV1",
    "ModelSidecarError",
    "SidecarReceiptV1",
    "StableFileIdentityV1",
    "inspect_local_gguf",
    "lease_verified_model",
    "load_model_receipt",
    "manifest_digest",
    "receipt_evidence_digest",
    "require_current_model",
    "require_receipt_inputs_match",
    "verify_registered_model",
]

# --- Pinned constants ---------------------------------------------------------

SOURCE_ANCHOR_COMMIT = "ca94157f70a2776e8da6b6849b50b45a083d0478"

# The pinned reviewed inspector lives at an exact relative path beneath the leased
# ``strix-llama`` worktree; verification rejects any other source id, root, or script.
SOURCE_ID = "strix-llama"
GGUF_PY_RELATIVE_ROOT = "gguf-py"
GGUF_DUMP_RELATIVE_PATH = "gguf-py/gguf/scripts/gguf_dump.py"

# The v1 family map is closed: a manifest family maps to the exact GGUF
# ``general.architecture`` STRING value it requires. An unmapped family is a typed
# compatibility failure rather than a silent pass.
FAMILY_ARCHITECTURE_MAP: Mapping[str, str] = {"qwen3_5": "qwen35"}

_READ_CHUNK_BYTES = 64 * 1024
_INSPECTOR_STDOUT_LIMIT = 64 * 1024 * 1024
_INSPECTOR_STDERR_LIMIT = 256 * 1024
_STDERR_PREFIX_BYTES = 64 * 1024
_RECEIPT_FILE_LIMIT = 8 * 1024 * 1024
_CACHE_FILE_LIMIT = 64 * 1024
_INSPECTOR_TIMEOUT_DEFAULT = 120.0
_LOCK_TIMEOUT_DEFAULT = 300.0

_CACHE_KEY_DOMAIN = "strixlab.model-hash-cache-key.v1"
_METADATA_DOMAIN = "strixlab.gguf-metadata.v1"
_EVIDENCE_DOMAIN = "strixlab.model-receipt-evidence.v1"
_RECEIPT_DOMAIN = "strixlab.model-receipt.v1"
_MANIFEST_DOMAIN = "strixlab.model-manifest.v1"

# The fixed inspector bootstrap. Its shape is validated by the child before it inserts
# the verified leased ``gguf-py`` root and executes the verified script under
# ``runpy``, so isolated mode loads dependencies only from the bound source candidate.
INSPECTOR_BOOTSTRAP = (
    "import runpy, sys\n"
    "argv = sys.argv[1:]\n"
    "if len(argv) != 4 or argv[3] != '--json':\n"
    "    raise SystemExit('strixlab-inspector-bootstrap: unexpected argv shape')\n"
    "gguf_py_root, script, model_path, json_flag = argv\n"
    "sys.path.insert(0, gguf_py_root)\n"
    "sys.argv = [script, model_path, json_flag]\n"
    "runpy.run_path(script, run_name='__main__')\n"
)

_METADATA_REGISTRY_DIR = ("models", "metadata", "v1")
_RECEIPT_REGISTRY_DIR = ("models", "receipts", "v1")
_CACHE_DIR = ("cache", "model-hashes", "v1")
_INSPECT_SCRATCH_DIR = ("models", "inspect")

_GGML_TYPE_PATTERN = r"^[A-Z0-9_]+$"


# --- Typed errors -------------------------------------------------------------


class ModelError(RuntimeError):
    """Base class for every typed model-registry failure."""


class ModelManifestError(ModelError):
    """A manifest is not registered, refuses verification, or is a draft."""


class ModelArtifactError(ModelError):
    """A local artifact is unavailable, unsafe, or diverged from expected bytes."""


class ModelCacheError(ModelError):
    """The model-hash cache holds a malformed, unsafe, or divergent entry."""


class ModelCacheBusyError(ModelCacheError):
    """A cold hash could not acquire its per-key lock before ``lock_timeout``."""


class ModelInspectorError(ModelError):
    """The GGUF inspector could not be bound or produced a defective process result."""


class ModelMetadataError(ModelError):
    """Inspector JSON was malformed, unnormalizable, or violated a predicate shape."""


class ModelSidecarError(ModelError):
    """A declared sidecar was missing, mutated, or violated its inspection contract."""


class ModelCompatibilityError(ModelError):
    """GGUF metadata did not satisfy the manifest's family or predicate requirements."""


# --- Runtime models -----------------------------------------------------------


class _Model(BaseModel):
    """Strict, frozen, finite base for every runtime model."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


Sha256Hex = Sha256Lower
GgmlTypeName = Annotated[str, Field(pattern=_GGML_TYPE_PATTERN)]
_RelativePath = Annotated[str, Field(min_length=1, max_length=4096)]
_BoundedText = Annotated[str, Field(min_length=1, max_length=256)]


class StableFileIdentityV1(_Model):
    """A no-follow regular file's device, inode, size, and change/modify times."""

    schema_version: Literal[1] = 1
    dev: int
    ino: Annotated[int, Field(gt=0)]
    size_bytes: Annotated[int, Field(ge=0)]
    mtime_ns: int
    ctime_ns: int


class GgufInspectorBindingV1(_Model):
    """The pinned inspector: leased source identities plus interpreter/script digests.

    Binds the source-compatible ``gguf_dump.py`` inspector to an already-authenticated
    Git candidate rather than inventing another source-tree hash. Every field is an
    expectation the caller records; verification requires the live lease, interpreter,
    and script to match it exactly.
    """

    schema_version: Literal[1] = 1
    preparation_id: _BoundedText
    candidate_id: Annotated[str, Field(pattern=r"^candidate-sha256:[0-9a-f]{64}$")]
    content_tree_id: Annotated[str, Field(pattern=r"^content-tree-sha256:[0-9a-f]{64}$")]
    base_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
    # The caller resolves the interpreter to an absolute realpath (``Path.resolve``) so
    # verification can no-follow a regular file rather than chase a system symlink.
    python_executable: AbsolutePathString
    python_sha256: Sha256Hex
    gguf_py_relative_root: _RelativePath
    script_relative_path: _RelativePath
    script_sha256: Sha256Hex


class GgufMetadataV1(_Model):
    """Compact projection digests and counts of one GGUF's normalized metadata.

    The full normalized projection is a content-addressed local registry artifact; this
    model records its digest, registry path, and derived counts, never the tensor map.
    """

    schema_version: Literal[1] = 1
    endian: Literal["LITTLE", "BIG"]
    # The exact bounded inspector stdout digest and byte count, verified against the
    # process result. Distinct from ``metadata_sha256`` (the normalized projection).
    inspector_stdout_sha256: Sha256Hex
    inspector_stdout_bytes: Annotated[int, Field(ge=0)]
    metadata_sha256: Sha256Hex
    metadata_registry_path: _RelativePath
    metadata_key_count: Annotated[int, Field(ge=0)]
    tensor_count: Annotated[int, Field(ge=0)]
    tensor_type_counts: tuple[tuple[GgmlTypeName, Annotated[int, Field(gt=0)]], ...]


class ModelFileReceiptV1(_Model):
    """One verified local file: its public path, stable identity, and full SHA-256."""

    schema_version: Literal[1] = 1
    local_path: AbsolutePathString
    identity: StableFileIdentityV1
    sha256: Sha256Hex


class SidecarReceiptV1(_Model):
    """One verified sidecar: identity, inspection mode, metadata, and compatibility."""

    schema_version: Literal[1] = 1
    id: str
    kind: Literal["mmproj", "imatrix", "opaque"]
    inspection: Literal["gguf", "hash-only"]
    file: ModelFileReceiptV1
    metadata: GgufMetadataV1 | None
    compatibility: Literal["verified", "asserted"]


class _SidecarEvidenceV1(_Model):
    id: str
    kind: Literal["mmproj", "imatrix", "opaque"]
    inspection: Literal["gguf", "hash-only"]
    local_path: AbsolutePathString
    sha256: Sha256Hex
    size_bytes: Annotated[int, Field(ge=0)]
    compatibility: Literal["verified", "asserted"]
    metadata_sha256: Sha256Hex | None
    tensor_count: Annotated[int, Field(ge=0)] | None


class ModelReceiptEvidenceV1(_Model):
    """The compact, portable receipt projection adapters embed in their samples.

    Its canonical digest equals ``model_receipt_sha256``, so exported sample evidence
    independently substantiates what ``verified`` means after the local registry (its
    metadata/receipt files) disappears. It contains no local registry paths.
    """

    schema_version: Literal[1] = 1
    manifest_id: str
    manifest_sha256: Sha256Hex
    primary_local_path: AbsolutePathString
    primary_sha256: Sha256Hex
    primary_size_bytes: Annotated[int, Field(ge=0)]
    endian: Literal["LITTLE", "BIG"]
    inspector_stdout_sha256: Sha256Hex
    metadata_sha256: Sha256Hex
    metadata_key_count: Annotated[int, Field(ge=0)]
    tensor_count: Annotated[int, Field(ge=0)]
    tensor_type_counts: tuple[tuple[GgmlTypeName, Annotated[int, Field(gt=0)]], ...]
    sidecars: tuple[_SidecarEvidenceV1, ...]
    compatibility: Literal["verified", "asserted"]
    publishable: bool
    inspector_python_sha256: Sha256Hex
    inspector_script_sha256: Sha256Hex
    source_preparation_id: _BoundedText
    source_candidate_id: str
    source_content_tree_id: str
    source_base_commit: str


class ModelObservationV1(_Model):
    """Practice-only evidence: a safely inspected but explicitly unregistered GGUF."""

    schema_version: Literal[1] = 1
    trust_state: Literal["unregistered"] = "unregistered"
    primary: ModelFileReceiptV1
    metadata: GgufMetadataV1
    inspector: GgufInspectorBindingV1


class ModelReceiptV1(_Model):
    """The deterministic receipt of one registered model's local verification."""

    schema_version: Literal[1] = 1
    trust_state: Literal["verified"] = "verified"
    manifest_id: str
    manifest_sha256: Sha256Hex
    primary: ModelFileReceiptV1
    metadata: GgufMetadataV1
    sidecars: tuple[SidecarReceiptV1, ...]
    inspector: GgufInspectorBindingV1
    compatibility: Literal["verified", "asserted"]
    publishable: bool
    evidence: ModelReceiptEvidenceV1


# --- Digests ------------------------------------------------------------------


def _domain_digest(domain: str, payload: Mapping[str, Any]) -> tuple[bytes, str]:
    """Return the canonical domain-tagged bytes and their SHA-256 hex digest."""

    body: dict[str, Any] = {"domain": domain}
    body.update(payload)
    if len(body) != len(payload) + 1:
        raise ModelError(f"digest payload collides with the reserved domain key: {domain}")
    encoded = canonical_json_bytes(body)
    return encoded, hashlib.sha256(encoded).hexdigest()


def manifest_digest(manifest: ModelManifestV1) -> str:
    """SHA-256 of the fully resolved, validated manifest under its digest domain."""

    _, digest = _domain_digest(_MANIFEST_DOMAIN, manifest.model_dump(mode="json"))
    return digest


def receipt_evidence_digest(evidence: ModelReceiptEvidenceV1) -> str:
    """SHA-256 of the exact canonical portable evidence projection."""

    _, digest = _domain_digest(_EVIDENCE_DOMAIN, evidence.model_dump(mode="json"))
    return digest


def require_receipt_inputs_match(
    receipt: ModelReceiptV1,
    *,
    model_id: str,
    model_path: str,
    model_sha256: str,
    model_receipt_sha256: str,
    model_receipt_evidence: ModelReceiptEvidenceV1,
) -> None:
    """Bind an adapter's verified-receipt inputs to a live :class:`ModelReceiptV1`.

    Requires the embedded portable projection to be self-consistent (its canonical
    digest equals ``model_receipt_sha256``), to equal the receipt's own evidence, and to
    agree on model id, public path, and primary SHA-256. Any mismatch is a
    :class:`ModelError` the caller translates into its own integrity failure.
    """

    if receipt_evidence_digest(model_receipt_evidence) != model_receipt_sha256:
        raise ModelError("embedded receipt evidence digest does not match model_receipt_sha256")
    if receipt.evidence != model_receipt_evidence:
        raise ModelError("receipt evidence does not match the embedded projection")
    if receipt.evidence.manifest_id != model_id:
        raise ModelError("receipt manifest id does not match model_id")
    if receipt.primary.local_path != model_path:
        raise ModelError("receipt primary path does not match model_path")
    if receipt.primary.sha256 != model_sha256:
        raise ModelError("receipt primary sha256 does not match model_sha256")


def _receipt_envelope(receipt: ModelReceiptV1) -> tuple[bytes, str]:
    return _domain_digest(_RECEIPT_DOMAIN, receipt.model_dump(mode="json"))


# --- Storage layout -----------------------------------------------------------


def _prepare_home(home: Path, *, create: bool = True) -> None:
    if not home.is_absolute():
        raise ModelError("home must be an absolute path")
    dirs = [
        home / "cache",
        home.joinpath(*_CACHE_DIR[:1], _CACHE_DIR[1]),
        home.joinpath(*_CACHE_DIR),
        home / "models",
        home.joinpath("models", "metadata"),
        home.joinpath(*_METADATA_REGISTRY_DIR),
        home.joinpath("models", "receipts"),
        home.joinpath(*_RECEIPT_REGISTRY_DIR),
        home.joinpath(*_INSPECT_SCRATCH_DIR),
    ]
    prepare_directory_tree(home, dirs, create=create, validate=_validate_directory)


def _validate_directory(path: Path) -> None:
    if is_unsafe_directory(path.lstat()):
        raise ModelError(f"model storage directory is unsafe: {path}")


# --- No-follow file identity and hashing --------------------------------------


def _lstat_regular(path: Path, *, error: type[ModelError]) -> os.stat_result:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise error(f"model file is unavailable: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise error(f"model path is not a regular file: {path}")
    if metadata.st_uid != os.geteuid():
        raise error(f"model file is owned by another user: {path}")
    return metadata


def _identity_of(metadata: os.stat_result) -> StableFileIdentityV1:
    return StableFileIdentityV1(
        dev=metadata.st_dev,
        ino=metadata.st_ino,
        size_bytes=metadata.st_size,
        mtime_ns=metadata.st_mtime_ns,
        ctime_ns=metadata.st_ctime_ns,
    )


def _current_identity(path: Path, *, error: type[ModelError]) -> StableFileIdentityV1:
    return _identity_of(_lstat_regular(path, error=error))


def _open_stable_descriptor(
    path: Path, *, error: type[ModelError]
) -> tuple[int, StableFileIdentityV1]:
    """Open ``path`` no-follow and bind the descriptor's fstat to the path's lstat."""

    link = _lstat_regular(path, error=error)
    descriptor = os.open(path, readonly_open_flags())
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise error(f"model path is not a regular file: {path}")
        if metadata.st_uid != os.geteuid():
            raise error(f"model file is owned by another user: {path}")
        if metadata.st_dev != link.st_dev or metadata.st_ino != link.st_ino:
            raise error(f"model descriptor does not match the current path: {path}")
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, _identity_of(metadata)


def _hash_descriptor(
    descriptor: int, identity: StableFileIdentityV1, *, path: Path, error: type[ModelError]
) -> str:
    """Stream-hash an open descriptor and require its identity to stay stable."""

    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while chunk := os.read(descriptor, _READ_CHUNK_BYTES):
        digest.update(chunk)
        size += len(chunk)
    after = os.fstat(descriptor)
    if (
        size != identity.size_bytes
        or after.st_dev != identity.dev
        or after.st_ino != identity.ino
        or after.st_size != identity.size_bytes
        or after.st_mtime_ns != identity.mtime_ns
        or after.st_ctime_ns != identity.ctime_ns
    ):
        raise error(f"model file changed while hashing: {path}")
    return digest.hexdigest()


def _hash_regular_file(
    path: Path, *, executable: bool, error: type[ModelError]
) -> tuple[StableFileIdentityV1, str]:
    descriptor, identity = _open_stable_descriptor(path, error=error)
    try:
        if executable and not os.fstat(descriptor).st_mode & 0o111:
            raise error(f"file is not executable: {path}")
        sha256 = _hash_descriptor(descriptor, identity, path=path, error=error)
    finally:
        os.close(descriptor)
    return identity, sha256


# --- Model-hash cache ---------------------------------------------------------


class _ModelHashCacheRecordV1(_Model):
    schema_version: Literal[1] = 1
    identity: StableFileIdentityV1
    size_bytes: Annotated[int, Field(ge=0)]
    sha256: Sha256Hex


def _cache_key(path: Path, identity: StableFileIdentityV1) -> str:
    if not path.is_absolute():
        raise ModelArtifactError(f"model-hash cache key requires an absolute path: {path}")
    _, digest = _domain_digest(
        _CACHE_KEY_DOMAIN,
        {"absolute_path": str(path), "identity": identity.model_dump(mode="json")},
    )
    return digest


def _read_bounded(path: Path, limit: int, *, error: type[ModelError]) -> bytes | None:
    try:
        descriptor = os.open(path, readonly_open_flags())
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise error(f"model registry file is unsafe: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise error(f"model registry file is unsafe: {path}")
        if metadata.st_size > limit:
            raise error(f"model registry file is oversized: {path}")
        content = os.read(descriptor, limit + 1)
    finally:
        os.close(descriptor)
    if len(content) > limit:
        raise error(f"model registry file is oversized: {path}")
    return content


def _publish_content_addressed(
    directory: Path, name: str, payload: bytes, *, error: type[ModelError], description: str
) -> None:
    """Publish immutable ``payload`` at ``directory/name``: no-replace, idempotent."""

    destination = directory / name
    if os.path.lexists(destination):
        existing = _read_bounded(destination, len(payload) + 1, error=error)
        if existing == payload:
            fsync_directory(directory)
            return
        raise error(f"divergent {description} at {destination}")
    temporary = directory / f".{name}.{os.getpid()}.{_token()}.tmp"
    try:
        write_exclusive(temporary, payload, 0o400)
        try:
            rename_noreplace(temporary, destination)
        except FileExistsError:
            existing = _read_bounded(destination, len(payload) + 1, error=error)
            if existing != payload:
                raise error(f"divergent {description} at {destination}") from None
    finally:
        if temporary.exists():
            temporary.unlink()
    fsync_directory(directory)


def _token() -> str:
    return os.urandom(8).hex()


def _resolve_file_sha256(
    descriptor: int,
    identity: StableFileIdentityV1,
    path: Path,
    home: Path,
    *,
    lock_timeout: float,
    error: type[ModelError],
) -> str:
    """Return the file's SHA-256, hashing at most once even under concurrency.

    A hit is honored only after the fresh identity matches the stored record; a cold
    miss serializes cold verifiers through a per-key advisory lock, rechecks the cache,
    hashes once, and publishes a crash-safe record.
    """

    cache_dir = home.joinpath(*_CACHE_DIR)
    key = _cache_key(path, identity)
    record = _load_cache_record(cache_dir, key, identity)
    if record is not None:
        return record.sha256
    lock_path = cache_dir / f"{key}.lock"
    with wait_for_exclusive_lock(lock_path, timeout=lock_timeout) as held:
        if not held.acquired:
            if held.status is LockStatus.CONTENDED:
                raise ModelCacheBusyError(
                    f"model-hash cache lock was not acquired within {lock_timeout}s"
                )
            raise ModelCacheError(held.reason or "model-hash cache lock is unavailable")
        record = _load_cache_record(cache_dir, key, identity)
        if record is not None:
            return record.sha256
        sha256 = _hash_descriptor(descriptor, identity, path=path, error=error)
        published = _ModelHashCacheRecordV1(
            identity=identity, size_bytes=identity.size_bytes, sha256=sha256
        )
        _publish_content_addressed(
            cache_dir,
            f"{key}.json",
            canonical_json_bytes(published.model_dump(mode="json")),
            error=ModelCacheError,
            description="model-hash cache record",
        )
        return sha256


def _load_cache_record(
    cache_dir: Path, key: str, identity: StableFileIdentityV1
) -> _ModelHashCacheRecordV1 | None:
    content = _read_bounded(cache_dir / f"{key}.json", _CACHE_FILE_LIMIT, error=ModelCacheError)
    if content is None:
        return None
    try:
        record = _ModelHashCacheRecordV1.model_validate_json(content, strict=True)
    except ValidationError as exc:
        raise ModelCacheError("stored model-hash cache record is invalid") from exc
    if record.identity != identity or record.size_bytes != identity.size_bytes:
        raise ModelCacheError("model-hash cache record diverged from the current file identity")
    return record


# --- Inspector binding and execution ------------------------------------------


@dataclass(frozen=True, slots=True)
class _BoundInspector:
    binding: GgufInspectorBindingV1
    lease: SourceLease
    python_executable: Path
    script_path: Path
    gguf_py_root: Path

    def verify(self) -> None:
        self.lease.verify()
        interpreter = hash_executable(
            self.python_executable, error=ModelInspectorError, subject="inspector interpreter"
        )
        if interpreter.sha256 != self.binding.python_sha256:
            raise ModelInspectorError(f"digest mismatch for {self.python_executable}")
        _require_digest(
            self.script_path,
            self.binding.script_sha256,
            executable=False,
            error=ModelInspectorError,
        )


def _require_digest(
    path: Path, expected: str, *, executable: bool, error: type[ModelError]
) -> StableFileIdentityV1:
    identity, sha256 = _hash_regular_file(path, executable=executable, error=error)
    if sha256 != expected:
        raise error(f"digest mismatch for {path}")
    return identity


def _resolve_within(root: Path, relative: str, *, error: type[ModelError]) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part == ".." for part in pure.parts):
        raise error(f"inspector path escapes the leased worktree: {relative}")
    return root.joinpath(*pure.parts)


def _bind_inspector(inspector: GgufInspectorBindingV1, lease: SourceLease) -> _BoundInspector:
    evidence = lease.evidence
    if (
        inspector.base_commit != SOURCE_ANCHOR_COMMIT
        or evidence.base_commit != SOURCE_ANCHOR_COMMIT
    ):
        raise ModelInspectorError("inspector is not bound to the pinned source commit")
    if (
        inspector.preparation_id != evidence.preparation_id
        or inspector.candidate_id != evidence.candidate_id
        or inspector.content_tree_id != evidence.content_tree_id
    ):
        raise ModelInspectorError("inspector binding does not match the source lease")
    if evidence.source_id != SOURCE_ID:
        raise ModelInspectorError("inspector requires the pinned strix-llama source lease")
    if evidence.patches:
        raise ModelInspectorError("inspector requires an unpatched source preparation")
    if evidence.status:
        raise ModelInspectorError("inspector requires a clean source preparation")
    if inspector.gguf_py_relative_root != GGUF_PY_RELATIVE_ROOT:
        raise ModelInspectorError("inspector gguf-py root is not the pinned relative path")
    if inspector.script_relative_path != GGUF_DUMP_RELATIVE_PATH:
        raise ModelInspectorError("inspector script is not the pinned gguf_dump.py path")

    worktree = lease.worktree
    script_path = _resolve_within(
        worktree, inspector.script_relative_path, error=ModelInspectorError
    )
    gguf_py_root = _resolve_within(
        worktree, inspector.gguf_py_relative_root, error=ModelInspectorError
    )
    if not gguf_py_root.is_dir():
        raise ModelInspectorError(f"leased gguf-py root is not a directory: {gguf_py_root}")
    python_executable = Path(inspector.python_executable)
    bound = _BoundInspector(
        binding=inspector,
        lease=lease,
        python_executable=python_executable,
        script_path=script_path,
        gguf_py_root=gguf_py_root,
    )
    bound.verify()
    return bound


@dataclass(frozen=True, slots=True)
class _InspectorOutput:
    dump: dict[str, Any]
    stdout_sha256: str
    stdout_bytes: int


def _run_inspector(
    descriptor: int, path: Path, bound: _BoundInspector, home: Path, *, timeout: float
) -> _InspectorOutput:
    """Run the bound inspector against ``/proc/self/fd/<fd>`` and return parsed JSON.

    The returned output carries the exact bounded stdout digest and byte count already
    verified against the process result, so the raw inspector-output identity survives
    into the receipt independently of the normalized projection.
    """

    from strixlab.process import ProcessOutcome, run_process

    scratch = home.joinpath(*_INSPECT_SCRATCH_DIR) / _token()
    ensure_directory_fsynced(scratch)
    fd_path = f"/proc/self/fd/{descriptor}"
    argv = [
        str(bound.python_executable),
        "-I",
        "-c",
        INSPECTOR_BOOTSTRAP,
        str(bound.gguf_py_root),
        str(bound.script_path),
        fd_path,
        "--json",
    ]
    stdout_spool = scratch / "stdout.bin"
    bound.lease.verify()
    try:
        result = run_process(
            argv,
            cwd=scratch,
            timeout=timeout,
            inherit_env=False,
            base_env={
                "LANG": "C",
                "LC_ALL": "C",
                "TZ": "UTC",
                "HOME": str(scratch),
                "TMPDIR": str(scratch),
            },
            output_limit_bytes=_STDERR_PREFIX_BYTES,
            stdout_total_limit_bytes=_INSPECTOR_STDOUT_LIMIT,
            stderr_total_limit_bytes=_INSPECTOR_STDERR_LIMIT,
            stdout_spool=stdout_spool,
            spool_root=scratch,
            pass_fds=(descriptor,),
        )
        if result.outcome is ProcessOutcome.SPAWN_FAILED:
            raise ModelInspectorError("inspector process could not be spawned")
        if result.outcome is ProcessOutcome.TIMED_OUT:
            raise ModelInspectorError("inspector process timed out")
        if result.outcome is ProcessOutcome.CAPTURE_FAILED:
            raise ModelInspectorError(f"inspector output exceeded bounds: {result.capture_error}")
        if result.returncode != 0:
            raise ModelInspectorError(f"inspector exited nonzero: {result.returncode}")
        raw = _read_inspector_stdout(result.stdout_spool, result.stdout_bytes, result.stdout_sha256)
        return _InspectorOutput(
            dump=_parse_inspector_json(raw),
            stdout_sha256=result.stdout_sha256,
            stdout_bytes=result.stdout_bytes,
        )
    finally:
        _remove_scratch(scratch)


def _read_inspector_stdout(spool: Path | None, byte_count: int, sha256: str) -> bytes:
    if spool is None:
        raise ModelInspectorError("inspector stdout was not captured")
    content = _read_bounded(spool, _INSPECTOR_STDOUT_LIMIT, error=ModelInspectorError)
    if content is None:
        raise ModelInspectorError("inspector stdout spool disappeared")
    if len(content) != byte_count or hashlib.sha256(content).hexdigest() != sha256:
        raise ModelInspectorError("inspector stdout spool does not match the captured digest")
    return content


def _remove_scratch(scratch: Path) -> None:
    for child in sorted(scratch.glob("*")):
        if child.is_file() and not child.is_symlink():
            child.unlink()
    with suppress(OSError):
        scratch.rmdir()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ModelMetadataError("inspector JSON contains a duplicate key")
        seen[key] = value
    return seen


def _reject_nonfinite(_value: str) -> float:
    raise ModelMetadataError("inspector JSON contains a non-finite number")


def _parse_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ModelMetadataError("inspector JSON contains a non-finite number")
    return parsed


def _parse_inspector_json(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ModelInspectorError("inspector stdout is not valid UTF-8") from exc
    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
            parse_float=_parse_float,
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ModelMetadataError("inspector stdout is not a single JSON object") from exc
    if not isinstance(parsed, dict):
        raise ModelMetadataError("inspector stdout is not a JSON object")
    return parsed


# --- Metadata normalization ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class _NormalizedMetadata:
    projection: dict[str, Any]
    metadata_map: dict[str, dict[str, Any]]
    endian: str
    tensor_count: int
    tensor_type_counts: tuple[tuple[str, int], ...]
    key_count: int


def _normalize_metadata(dump: Mapping[str, Any]) -> _NormalizedMetadata:
    endian = dump.get("endian")
    if endian not in ("LITTLE", "BIG"):
        raise ModelMetadataError("inspector JSON has an invalid endianness")
    metadata_in = dump.get("metadata")
    tensors_in = dump.get("tensors")
    if not isinstance(metadata_in, dict) or not isinstance(tensors_in, dict):
        raise ModelMetadataError("inspector JSON is missing metadata or tensors")

    metadata_map: dict[str, dict[str, Any]] = {}
    for key, entry in metadata_in.items():
        if not isinstance(entry, dict) or "type" not in entry:
            raise ModelMetadataError(f"inspector metadata entry is malformed: {key}")
        value_type = entry["type"]
        if not isinstance(value_type, str):
            raise ModelMetadataError(f"inspector metadata type is not a string: {key}")
        if "array_types" in entry:
            array_types = entry["array_types"]
            if not isinstance(array_types, list) or not all(
                isinstance(name, str) for name in array_types
            ):
                raise ModelMetadataError(f"inspector array_types is malformed: {key}")
            metadata_map[key] = {"type": value_type, "array_types": list(array_types)}
        elif "value" in entry:
            metadata_map[key] = {"type": value_type, "value": entry["value"]}
        else:
            raise ModelMetadataError(f"inspector metadata entry has neither value nor array: {key}")

    tensor_counts: dict[str, int] = {}
    tensors_out: dict[str, dict[str, Any]] = {}
    for name, entry in tensors_in.items():
        if not isinstance(entry, dict) or "shape" not in entry or "type" not in entry:
            raise ModelMetadataError(f"inspector tensor entry is malformed: {name}")
        shape = entry["shape"]
        tensor_type = entry["type"]
        if not isinstance(tensor_type, str):
            raise ModelMetadataError(f"inspector tensor entry is malformed: {name}")
        # A tensor shape must be a non-empty list of positive integers: no empty rank,
        # boolean, zero, negative, or non-integer dimension. The handoff pins no maximum
        # rank, so none is invented here.
        if (
            not isinstance(shape, list)
            or not shape
            or not all(
                isinstance(dim, int) and not isinstance(dim, bool) and dim > 0 for dim in shape
            )
        ):
            raise ModelMetadataError(f"inspector tensor shape is invalid: {name}")
        tensors_out[name] = {"shape": list(shape), "type": tensor_type}
        tensor_counts[tensor_type] = tensor_counts.get(tensor_type, 0) + 1

    projection = {"endian": endian, "metadata": metadata_map, "tensors": tensors_out}
    _reject_absolute_path_strings(projection)
    tensor_type_counts = tuple(sorted(tensor_counts.items()))
    return _NormalizedMetadata(
        projection=projection,
        metadata_map=metadata_map,
        endian=endian,
        tensor_count=len(tensors_out),
        tensor_type_counts=tensor_type_counts,
        key_count=len(metadata_map),
    )


def _reject_absolute_path_strings(value: Any) -> None:
    if isinstance(value, str):
        if value.startswith("/"):
            raise ModelMetadataError("normalized metadata contains an absolute path string")
    elif isinstance(value, dict):
        for child in value.values():
            _reject_absolute_path_strings(child)
    elif isinstance(value, list):
        for child in value:
            _reject_absolute_path_strings(child)


def _store_metadata(
    home: Path, normalized: _NormalizedMetadata, output: _InspectorOutput
) -> GgufMetadataV1:
    payload, digest = _domain_digest(_METADATA_DOMAIN, normalized.projection)
    directory = home.joinpath(*_METADATA_REGISTRY_DIR)
    _publish_content_addressed(
        directory,
        f"{digest}.json",
        payload,
        error=ModelMetadataError,
        description="normalized metadata projection",
    )
    return GgufMetadataV1(
        endian=normalized.endian,  # type: ignore[arg-type]
        inspector_stdout_sha256=output.stdout_sha256,
        inspector_stdout_bytes=output.stdout_bytes,
        metadata_sha256=digest,
        metadata_registry_path="/".join((*_METADATA_REGISTRY_DIR, f"{digest}.json")),
        metadata_key_count=normalized.key_count,
        tensor_count=normalized.tensor_count,
        tensor_type_counts=normalized.tensor_type_counts,
    )


# --- Predicate evaluation and compatibility -----------------------------------


def _strict_equal(observed: Any, expected: Any) -> bool:
    return type(observed) is type(expected) and observed == expected


def _predicate_holds(
    predicate: MetadataPredicateV1, metadata: Mapping[str, dict[str, Any]]
) -> bool:
    entry = metadata.get(predicate.key)
    if entry is None or entry.get("type") != predicate.value_type:
        return False
    if predicate.array_types is not None:
        return entry.get("array_types") == list(predicate.array_types)
    if "value" not in entry:
        return False
    return _strict_equal(entry["value"], predicate.scalar_value)


def _require_predicates(
    predicates: list[MetadataPredicateV1],
    metadata: Mapping[str, dict[str, Any]],
    *,
    error: type[ModelError],
    subject: str,
) -> None:
    for predicate in predicates:
        if not _predicate_holds(predicate, metadata):
            raise error(f"{subject} metadata predicate did not hold: {predicate.key}")


def _require_family(manifest: ModelManifestV1, metadata: Mapping[str, dict[str, Any]]) -> None:
    architecture = manifest.architecture
    assert architecture is not None
    expected = FAMILY_ARCHITECTURE_MAP.get(architecture.family)
    if expected is None:
        raise ModelCompatibilityError(f"unmapped model family: {architecture.family}")
    entry = metadata.get("general.architecture")
    if entry is None or entry.get("type") != "STRING" or entry.get("value") != expected:
        raise ModelCompatibilityError(
            "GGUF general.architecture does not match the manifest family mapping"
        )


# --- Inspection core ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _InspectedFile:
    receipt: ModelFileReceiptV1
    metadata: GgufMetadataV1
    metadata_map: dict[str, dict[str, Any]]


def _inspect_file(
    path: Path,
    *,
    expected_size: int | None,
    expected_sha256: str | None,
    bound: _BoundInspector,
    home: Path,
    timeout: float,
    lock_timeout: float,
    error: type[ModelError],
) -> _InspectedFile:
    descriptor, identity = _open_stable_descriptor(path, error=error)
    try:
        if expected_size is not None and identity.size_bytes != expected_size:
            raise error(f"model file size does not match the manifest: {path}")
        sha256 = _resolve_file_sha256(
            descriptor, identity, path, home, lock_timeout=lock_timeout, error=error
        )
        if expected_sha256 is not None and sha256 != expected_sha256:
            raise error(f"model file SHA-256 does not match the manifest: {path}")
        output = _run_inspector(descriptor, path, bound, home, timeout=timeout)
        bound.lease.verify()
        current = _current_identity(path, error=error)
        if current != identity:
            raise error(f"model file drifted across inspection: {path}")
        normalized = _normalize_metadata(output.dump)
        metadata = _store_metadata(home, normalized, output)
    finally:
        os.close(descriptor)
    receipt = ModelFileReceiptV1(local_path=str(path), identity=identity, sha256=sha256)
    return _InspectedFile(receipt=receipt, metadata=metadata, metadata_map=normalized.metadata_map)


# --- Public API ---------------------------------------------------------------


def inspect_local_gguf(
    path: Path,
    *,
    inspector: GgufInspectorBindingV1,
    source_lease: SourceLease,
    home: Path,
    timeout: float = _INSPECTOR_TIMEOUT_DEFAULT,
    lock_timeout: float = _LOCK_TIMEOUT_DEFAULT,
) -> ModelObservationV1:
    """Safely inspect one local GGUF into an explicitly unregistered observation.

    This is practice evidence only: no registered manifest claims the file, and an
    observation can never become a verified receipt.
    """

    # Cache keys must derive from an absolute path; reject a relative one before any
    # side effect (no home preparation, scratch, or inspector spawn).
    if not path.is_absolute():
        raise ModelArtifactError(f"model path must be absolute: {path}")
    _prepare_home(home)
    bound = _bind_inspector(inspector, source_lease)
    inspected = _inspect_file(
        path,
        expected_size=None,
        expected_sha256=None,
        bound=bound,
        home=home,
        timeout=timeout,
        lock_timeout=lock_timeout,
        error=ModelArtifactError,
    )
    bound.verify()
    return ModelObservationV1(
        primary=inspected.receipt, metadata=inspected.metadata, inspector=inspector
    )


def verify_registered_model(
    manifest: ModelManifestV1,
    *,
    inspector: GgufInspectorBindingV1,
    source_lease: SourceLease,
    home: Path,
    timeout: float = _INSPECTOR_TIMEOUT_DEFAULT,
    lock_timeout: float = _LOCK_TIMEOUT_DEFAULT,
) -> ModelReceiptV1:
    """Verify a registered model's local artifact and sidecars into a receipt.

    Refuses drafts. Requires the primary GGUF to match the manifest family mapping,
    have nonzero tensors, satisfy every declared predicate, and byte-match the manifest
    size/SHA-256; every required sidecar is byte-verified and, when declared ``gguf``,
    metadata-checked. Hash-only sidecars remain ``asserted`` and force ``publishable``
    false. The full receipt and content-addressed metadata are published locally.
    """

    if manifest.registry_status != "registered":
        raise ModelManifestError("verify_registered_model refuses a draft manifest")
    if manifest.architecture is None:
        raise ModelManifestError("a registered manifest requires an architecture")
    _prepare_home(home)
    bound = _bind_inspector(inspector, source_lease)

    primary_file = manifest.artifact.file
    assert primary_file.local_path is not None
    primary = _inspect_file(
        Path(primary_file.local_path),
        expected_size=primary_file.size_bytes,
        expected_sha256=primary_file.sha256,
        bound=bound,
        home=home,
        timeout=timeout,
        lock_timeout=lock_timeout,
        error=ModelArtifactError,
    )
    if primary.metadata.tensor_count == 0:
        raise ModelCompatibilityError("primary GGUF declares zero tensors")
    _require_family(manifest, primary.metadata_map)
    _require_predicates(
        manifest.artifact.metadata_predicates,
        primary.metadata_map,
        error=ModelCompatibilityError,
        subject="primary",
    )

    sidecars, compatibility = _verify_sidecars(
        manifest, bound, home, timeout=timeout, lock_timeout=lock_timeout
    )
    publishable = (
        compatibility == "verified"
        and manifest.quantization.is_fully_provenanced()
        and manifest.quantization.measured_bits_per_weight is not None
    )

    digest = manifest_digest(manifest)
    evidence = _build_evidence(
        manifest,
        digest,
        primary,
        sidecars,
        inspector,
        compatibility=compatibility,
        publishable=publishable,
    )
    receipt = ModelReceiptV1(
        manifest_id=manifest.id,
        manifest_sha256=digest,
        primary=primary.receipt,
        metadata=primary.metadata,
        sidecars=sidecars,
        inspector=inspector,
        compatibility=compatibility,
        publishable=publishable,
        evidence=evidence,
    )
    bound.verify()
    _publish_receipt(home, receipt)
    return receipt


def _verify_sidecars(
    manifest: ModelManifestV1,
    bound: _BoundInspector,
    home: Path,
    *,
    timeout: float,
    lock_timeout: float,
) -> tuple[tuple[SidecarReceiptV1, ...], Literal["verified", "asserted"]]:
    receipts: list[SidecarReceiptV1] = []
    compatibility: Literal["verified", "asserted"] = "verified"
    for sidecar in manifest.sidecars:
        file = sidecar.file
        assert file.local_path is not None
        path = Path(file.local_path)
        if sidecar.inspection == "gguf":
            inspected = _inspect_file(
                path,
                expected_size=file.size_bytes,
                expected_sha256=file.sha256,
                bound=bound,
                home=home,
                timeout=timeout,
                lock_timeout=lock_timeout,
                error=ModelSidecarError,
            )
            if inspected.metadata.tensor_count == 0:
                raise ModelSidecarError(f"gguf sidecar declares zero tensors: {sidecar.id}")
            _require_predicates(
                sidecar.metadata_predicates,
                inspected.metadata_map,
                error=ModelSidecarError,
                subject=f"sidecar {sidecar.id}",
            )
            receipts.append(
                SidecarReceiptV1(
                    id=sidecar.id,
                    kind=sidecar.kind,
                    inspection="gguf",
                    file=inspected.receipt,
                    metadata=inspected.metadata,
                    compatibility="verified",
                )
            )
        else:
            file_receipt = _byte_verify_sidecar(path, file, home, lock_timeout=lock_timeout)
            compatibility = "asserted"
            receipts.append(
                SidecarReceiptV1(
                    id=sidecar.id,
                    kind=sidecar.kind,
                    inspection="hash-only",
                    file=file_receipt,
                    metadata=None,
                    compatibility="asserted",
                )
            )
    return tuple(receipts), compatibility


def _byte_verify_sidecar(
    path: Path,
    file: Any,
    home: Path,
    *,
    lock_timeout: float,
) -> ModelFileReceiptV1:
    descriptor, identity = _open_stable_descriptor(path, error=ModelSidecarError)
    try:
        if identity.size_bytes != file.size_bytes:
            raise ModelSidecarError(f"sidecar size does not match the manifest: {path}")
        sha256 = _resolve_file_sha256(
            descriptor, identity, path, home, lock_timeout=lock_timeout, error=ModelSidecarError
        )
        if sha256 != file.sha256:
            raise ModelSidecarError(f"sidecar SHA-256 does not match the manifest: {path}")
    finally:
        os.close(descriptor)
    return ModelFileReceiptV1(local_path=str(path), identity=identity, sha256=sha256)


def _build_evidence(
    manifest: ModelManifestV1,
    digest: str,
    primary: _InspectedFile,
    sidecars: tuple[SidecarReceiptV1, ...],
    inspector: GgufInspectorBindingV1,
    *,
    compatibility: Literal["verified", "asserted"],
    publishable: bool,
) -> ModelReceiptEvidenceV1:
    sidecar_evidence = tuple(
        _SidecarEvidenceV1(
            id=sidecar.id,
            kind=sidecar.kind,
            inspection=sidecar.inspection,
            local_path=sidecar.file.local_path,
            sha256=sidecar.file.sha256,
            size_bytes=sidecar.file.identity.size_bytes,
            compatibility=sidecar.compatibility,
            metadata_sha256=None if sidecar.metadata is None else sidecar.metadata.metadata_sha256,
            tensor_count=None if sidecar.metadata is None else sidecar.metadata.tensor_count,
        )
        for sidecar in sidecars
    )
    return ModelReceiptEvidenceV1(
        manifest_id=manifest.id,
        manifest_sha256=digest,
        primary_local_path=primary.receipt.local_path,
        primary_sha256=primary.receipt.sha256,
        primary_size_bytes=primary.receipt.identity.size_bytes,
        endian=primary.metadata.endian,
        inspector_stdout_sha256=primary.metadata.inspector_stdout_sha256,
        metadata_sha256=primary.metadata.metadata_sha256,
        metadata_key_count=primary.metadata.metadata_key_count,
        tensor_count=primary.metadata.tensor_count,
        tensor_type_counts=primary.metadata.tensor_type_counts,
        sidecars=sidecar_evidence,
        compatibility=compatibility,
        publishable=publishable,
        inspector_python_sha256=inspector.python_sha256,
        inspector_script_sha256=inspector.script_sha256,
        source_preparation_id=inspector.preparation_id,
        source_candidate_id=inspector.candidate_id,
        source_content_tree_id=inspector.content_tree_id,
        source_base_commit=inspector.base_commit,
    )


def _publish_receipt(home: Path, receipt: ModelReceiptV1) -> None:
    payload, digest = _receipt_envelope(receipt)
    directory = home.joinpath(*_RECEIPT_REGISTRY_DIR, receipt.manifest_id)
    ensure_directory_fsynced(directory)
    _validate_directory(directory)
    _publish_content_addressed(
        directory,
        f"{digest}.json",
        payload,
        error=ModelError,
        description="model receipt envelope",
    )


def load_model_receipt(
    manifest_id: str, local_receipt_sha256: str, *, home: Path
) -> ModelReceiptV1:
    """Load and re-authenticate a published receipt by its two validated identifiers."""

    if not _is_dash_id(manifest_id):
        raise ModelManifestError(f"invalid manifest id: {manifest_id!r}")
    if not _is_sha256(local_receipt_sha256):
        raise ModelError(f"invalid local receipt digest: {local_receipt_sha256!r}")
    path = home.joinpath(*_RECEIPT_REGISTRY_DIR, manifest_id, f"{local_receipt_sha256}.json")
    content = _read_bounded(path, _RECEIPT_FILE_LIMIT, error=ModelError)
    if content is None:
        raise ModelError(f"model receipt does not exist: {manifest_id}/{local_receipt_sha256}")
    if hashlib.sha256(content).hexdigest() != local_receipt_sha256:
        raise ModelError("model receipt content does not match its address")
    try:
        envelope = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ModelError("model receipt envelope is not JSON") from exc
    if not isinstance(envelope, dict) or envelope.get("domain") != _RECEIPT_DOMAIN:
        raise ModelError("model receipt envelope has the wrong digest domain")
    body = {key: value for key, value in envelope.items() if key != "domain"}
    try:
        receipt = ModelReceiptV1.model_validate_json(json.dumps(body), strict=True)
    except ValidationError as exc:
        raise ModelError("stored model receipt is invalid") from exc
    _, recomputed = _receipt_envelope(receipt)
    if recomputed != local_receipt_sha256:
        raise ModelError("re-hashed model receipt diverged from its address")
    if receipt.manifest_id != manifest_id:
        raise ModelError("model receipt names a different manifest id")
    return receipt


def require_current_model(receipt: ModelReceiptV1) -> StableFileIdentityV1:
    """Recheck the primary and every sidecar against the receipt's stable identities.

    Performs no full content rehash: it reopens each file no-follow and compares device,
    inode, size, and change/modify times to the receipt, failing closed on any drift.
    """

    primary_identity = _current_identity(Path(receipt.primary.local_path), error=ModelArtifactError)
    if primary_identity != receipt.primary.identity:
        raise ModelArtifactError(f"primary model drifted: {receipt.primary.local_path}")
    for sidecar in receipt.sidecars:
        identity = _current_identity(Path(sidecar.file.local_path), error=ModelSidecarError)
        if identity != sidecar.file.identity:
            raise ModelSidecarError(f"sidecar drifted: {sidecar.file.local_path}")
    return primary_identity


@dataclass(slots=True)
class ModelLease:
    """A process-local, non-serializable handle over a verified primary descriptor.

    It owns the no-follow descriptor and its ``/proc/self/fd`` path, and exposes an
    explicit :meth:`verify` that a caller invokes before every terminal publication.
    """

    receipt: ModelReceiptV1
    descriptor: int
    descriptor_path: str
    identity: StableFileIdentityV1

    def verify(self) -> None:
        require_current_model(self.receipt)
        if _identity_of(os.fstat(self.descriptor)) != self.identity:
            raise ModelArtifactError("leased model descriptor diverged from the receipt")


@contextmanager
def lease_verified_model(receipt: ModelReceiptV1) -> Iterator[ModelLease]:
    """Hold the receipt-bound primary descriptor and yield it for descriptor handoff.

    On entry it rechecks the primary and sidecars and binds the held descriptor to the
    receipt identity; on normal exit it rechecks defensively. The inherited descriptor
    and its ``/proc/self/fd/<fd>`` path let a child open the receipt-bound inode even if
    the pathname is swapped during execution.
    """

    require_current_model(receipt)
    descriptor, identity = _open_stable_descriptor(
        Path(receipt.primary.local_path), error=ModelArtifactError
    )
    try:
        if identity != receipt.primary.identity:
            raise ModelArtifactError("primary model drifted before leasing")
        lease = ModelLease(
            receipt=receipt,
            descriptor=descriptor,
            descriptor_path=f"/proc/self/fd/{descriptor}",
            identity=identity,
        )
        yield lease
        lease.verify()
    finally:
        os.close(descriptor)


# --- Small validators ---------------------------------------------------------

_DASH_ID = re.compile(DASH_ID_PATTERN)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _is_dash_id(value: str) -> bool:
    return bool(_DASH_ID.fullmatch(value))


def _is_sha256(value: str) -> bool:
    return bool(_SHA256.fullmatch(value))

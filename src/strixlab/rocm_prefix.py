"""Bounded prefix observations and comparisons; never metadata admission.

Guarded opens do not cross mounts. The reused observer's named probes are only
bracketed, not an atomic transaction. Same-UID writers and post-scan mutation
remain outside the quiescent-owner guarantee. No archive or SDK is executed.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import re
import stat
from collections import deque
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict

from strixlab.rocm_metadata import (
    InodeIdentityV1,
    InodeMetadataObservationV1,
    MetadataError,
    _identity,
    observe_inode_metadata,
)
from strixlab.serialization import canonical_json_bytes

_CHUNK_BYTES = 64 * 1024
_MAX_MEMBERS = 1_000_000
_MAX_FILE_BYTES = 8 * 1024**3
_MAX_PAYLOAD_BYTES = 32 * 1024**3
_MAX_DEPTH = 256
_MAX_FDS = 256
_MAX_EVIDENCE_BYTES = 256 * 1024**2
_MAX_LINK_EVIDENCE_BYTES = 64 * 1024**2
_MAX_PATH_BYTES = 4096
_MAX_DIFFERENCES = 128
_MAX_LINK_EXPANSIONS = 40
_MAX_LINK_STEPS = 1_000_000
_MAX_PENDING_BYTES = 64 * 1024
_SYS_OPENAT2 = 437
_RESOLVE = 0x08 | 0x04 | 0x01  # BENEATH | NO_SYMLINKS | NO_XDEV
_LIBC = ctypes.CDLL(None, use_errno=True)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ESCAPED = re.compile(r"(?:\\x[0-9a-f]{2})*")
type _Kind = Literal["file", "directory", "symlink"]
type _LinkStatus = Literal[
    "resolved", "absolute", "escape", "dangling", "not-directory", "cycle", "expansion-limit"
]


class PrefixError(ValueError):
    """Bounded invalid-input/resource/drift/IO reason; no completed result."""


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class PrefixEntryV1(_Report):
    path: str
    path_bytes_escaped: str
    kind: _Kind
    mode: int
    uid: int
    gid: int
    byte_length: int | None
    sha256: str | None
    link_target: str | None
    link_target_bytes_escaped: str | None
    nlink: int | None
    identity: InodeIdentityV1
    metadata: InodeMetadataObservationV1


class PrefixInventoryV1(_Report):
    schema_version: Literal[1] = 1
    inventory_id: Literal["rocm-prefix-observation-v1"] = "rocm-prefix-observation-v1"
    validation: Literal["complete"] = "complete"
    metadata_coverage: Literal["unknown"] = "unknown"
    link_closure: Literal["not-checked"] = "not-checked"
    resolution_policy: Literal["openat2-beneath-no-symlinks-no-xdev"] = (
        "openat2-beneath-no-symlinks-no-xdev"
    )
    root: PrefixEntryV1
    entries: tuple[PrefixEntryV1, ...]
    member_count: int
    regular_payload_bytes: int
    evidence_bytes_charged: int
    peak_owned_fds: int

    def canonical_bytes(self) -> bytes:
        _validated_map(self)
        data = canonical_json_bytes(self.model_dump(mode="json"))
        if len(data) > _MAX_EVIDENCE_BYTES:
            raise PrefixError("prefix-evidence-limit")
        return data


class PrefixComparisonV1(_Report):
    validation: Literal["complete"] = "complete"
    scope: Literal["recorded-semantic-fields-only"] = "recorded-semantic-fields-only"
    metadata_coverage: Literal["unknown"] = "unknown"
    semantic_equal: bool
    differing_path_count: int
    differing_paths: tuple[str, ...]
    sample_truncated: bool


class PrefixLinkV1(_Report):
    path: str
    status: _LinkStatus
    resolved_path: str | None
    resolved_kind: _Kind | None
    expansions: int


class PrefixLinkMapV1(_Report):
    validation: Literal["complete"] = "complete"
    scope: Literal["completed-map-only"] = "completed-map-only"
    metadata_coverage: Literal["unknown"] = "unknown"
    links: tuple[PrefixLinkV1, ...]
    component_steps: int


class _OpenHow(ctypes.Structure):
    _fields_ = [("flags", ctypes.c_uint64), ("mode", ctypes.c_uint64), ("resolve", ctypes.c_uint64)]


def _supported_abi() -> bool:
    return (
        platform.system() == "Linux"
        and platform.machine() == "x86_64"
        and ctypes.sizeof(ctypes.c_void_p) == 8
        and ctypes.sizeof(ctypes.c_long) == 8
        and ctypes.sizeof(ctypes.c_size_t) == 8
        and ctypes.sizeof(_OpenHow) == 24
        and hasattr(os, "O_PATH")
        and hasattr(os, "O_NOFOLLOW")
        and hasattr(_LIBC, "syscall")
    )


def _openat2(parent_fd: int, name: str, flags: int) -> int:
    """Return a caller-owned read-only FD under the fixed single-leaf policy."""

    _leaf(name)
    if type(parent_fd) is not int or not 0 <= parent_fd < 2**31:
        raise PrefixError("prefix-parent-fd")
    if not _supported_abi():
        raise PrefixError("prefix-abi-unsupported")
    allowed = os.O_PATH | os.O_DIRECTORY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
    if type(flags) is not int or flags < 0 or flags & ~allowed:
        raise PrefixError("prefix-open-flags")
    how = _OpenHow(flags | os.O_NOFOLLOW | os.O_CLOEXEC, 0, _RESOLVE)
    syscall = _LIBC.syscall
    syscall.restype = ctypes.c_long
    result = int(
        syscall(
            ctypes.c_long(_SYS_OPENAT2),
            ctypes.c_int(parent_fd),
            ctypes.c_char_p(name.encode("utf-8")),
            ctypes.byref(how),
            ctypes.c_size_t(ctypes.sizeof(how)),
        )
    )
    if result == -1:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return result


def _encoded(value: str, maximum: int | None = None) -> bytes:
    if maximum is None:
        maximum = _MAX_PATH_BYTES
    if not isinstance(value, str) or len(value) > maximum:
        raise PrefixError("prefix-text-limit")
    try:
        data = value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise PrefixError("prefix-text-utf8") from exc
    if len(data) > maximum or b"\0" in data:
        raise PrefixError("prefix-text-limit")
    return data


def _leaf(name: str) -> None:
    _encoded(name, 255)
    if not name or name in {".", ".."} or "/" in name:
        raise PrefixError("prefix-leaf")


def _escaped(value: bytes) -> str:
    return "".join(f"\\x{byte:02x}" for byte in value)


def _kind(identity: InodeIdentityV1) -> _Kind:
    if stat.S_ISREG(identity.mode):
        return "file"
    if stat.S_ISDIR(identity.mode):
        return "directory"
    if stat.S_ISLNK(identity.mode):
        return "symlink"
    raise PrefixError("prefix-entry-type")


@dataclass(slots=True)
class _Budget:
    evidence: int = 4096  # Reserve fixed report envelope/counters before collecting entries.
    payload: int = 0
    live_fds: int = 0
    peak_fds: int = 0

    def charge(self, amount: int) -> None:
        self.evidence += amount
        if self.evidence > _MAX_EVIDENCE_BYTES:
            raise PrefixError("prefix-evidence-limit")

    @contextmanager
    def descriptors(self, amount: int) -> Iterator[None]:
        if self.live_fds + amount > _MAX_FDS:
            raise PrefixError("prefix-fd-limit")
        self.live_fds += amount
        self.peak_fds = max(self.peak_fds, self.live_fds)
        try:
            yield
        finally:
            self.live_fds -= amount


@contextmanager
def _opened(budget: _Budget, parent_fd: int, name: str, flags: int) -> Iterator[int]:
    with budget.descriptors(1):
        fd = _openat2(parent_fd, name, flags | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            yield fd
        finally:
            os.close(fd)


def _same(actual: InodeIdentityV1, expected: InodeIdentityV1) -> None:
    if actual != expected:
        raise PrefixError("prefix-identity-changed")


def _binding(budget: _Budget, parent_fd: int, name: str, expected: InodeIdentityV1) -> None:
    with _opened(budget, parent_fd, name, os.O_PATH) as fd:
        _same(_identity(os.fstat(fd)), expected)


def _bounded_identity(identity: InodeIdentityV1) -> None:
    # Linux stat's fixed-width seconds plus nanoseconds fit comfortably here.
    for value in identity.model_dump().values():
        if type(value) is not int or value.bit_length() > 128:
            raise PrefixError("prefix-identity-range")


def _validate_metadata(value: InodeMetadataObservationV1, identity: InodeIdentityV1) -> None:
    if (
        type(value.schema_version) is not int
        or value.schema_version != 1
        or value.validation != "complete"
        or value.coverage != "unknown"
        or value.observer_id != "linux-x86_64-listxattrat-v1"
        or value.scope != "stored-xattr-names-only"
        or value.list_status
        not in {"observed", "error", "unsupported", "resource-limit", "malformed"}
    ):
        raise PrefixError("prefix-metadata-record")
    for captured in (
        value.leaf_before,
        value.leaf_opened,
        value.leaf_after,
        value.leaf_named_after,
    ):
        _bounded_identity(captured)
        _same(captured, identity)
    _bounded_identity(value.parent_before)
    _bounded_identity(value.parent_after)
    _same(value.parent_before, value.parent_after)
    if (
        not value.name_bytes_escaped
        or len(value.name_bytes_escaped) > 4 * 255
        or not _ESCAPED.fullmatch(value.name_bytes_escaped)
    ):
        raise PrefixError("prefix-metadata-record")
    try:
        leaf = bytes.fromhex(value.name_bytes_escaped.replace("\\x", "")).decode("utf-8")
    except UnicodeError as exc:
        raise PrefixError("prefix-metadata-record") from exc
    _leaf(leaf)
    if value.list_errno is not None and (
        type(value.list_errno) is not int or not 0 <= value.list_errno <= 4095
    ):
        raise PrefixError("prefix-metadata-record")
    size = value.name_list_size_bytes
    if size is not None and (type(size) is not int or not 0 <= size < 2**63):
        raise PrefixError("prefix-metadata-record")
    names = value.names_bytes_escaped
    if value.list_status != "observed":
        if names is not None:
            raise PrefixError("prefix-metadata-record")
        return
    if (
        value.list_errno is not None
        or size is None
        or size > 65536
        or names is None
        or len(names) > 65536
    ):
        raise PrefixError("prefix-metadata-record")
    total = 0
    previous = ""
    for name in names:
        if (
            not name
            or len(name) > 4 * 65536
            or not _ESCAPED.fullmatch(name)
            or "\\x00" in name
            or name <= previous
        ):
            raise PrefixError("prefix-metadata-record")
        total += len(name) // 4 + 1
        if total > 65536:
            raise PrefixError("prefix-metadata-record")
        previous = name
    if total != size:
        raise PrefixError("prefix-metadata-record")


def _validate_entry(entry: PrefixEntryV1, budget: _Budget) -> None:
    path_bytes = _encoded(entry.path)
    for scalar in (entry.mode, entry.uid, entry.gid, entry.byte_length, entry.nlink):
        if scalar is not None and (type(scalar) is not int or scalar.bit_length() > 128):
            raise PrefixError("prefix-entry-record")
    if entry.path:
        for part in entry.path.split("/"):
            _leaf(part)
        if len(entry.path.split("/")) > _MAX_DEPTH:
            raise PrefixError("prefix-depth-limit")
    if entry.path_bytes_escaped != _escaped(path_bytes):
        raise PrefixError("prefix-path-record")
    _bounded_identity(entry.identity)
    if (entry.kind, entry.mode, entry.uid, entry.gid) != (
        _kind(entry.identity),
        stat.S_IMODE(entry.identity.mode),
        entry.identity.uid,
        entry.identity.gid,
    ):
        raise PrefixError("prefix-entry-record")
    _validate_metadata(entry.metadata, entry.identity)
    if entry.metadata.kind != entry.kind:
        raise PrefixError("prefix-metadata-record")
    if entry.path and entry.metadata.name_bytes_escaped != _escaped(
        entry.path.rsplit("/", 1)[-1].encode("utf-8")
    ):
        raise PrefixError("prefix-metadata-record")
    if entry.kind == "directory":
        if any(
            value is not None
            for value in (
                entry.byte_length,
                entry.sha256,
                entry.link_target,
                entry.link_target_bytes_escaped,
                entry.nlink,
            )
        ):
            raise PrefixError("prefix-entry-record")
    else:
        if entry.nlink != 1 or entry.identity.nlink != 1:
            raise PrefixError("prefix-non-directory-links")
        if (
            entry.byte_length != entry.identity.size
            or entry.byte_length is None
            or entry.byte_length < 0
        ):
            raise PrefixError("prefix-entry-record")
        if entry.kind == "file":
            if entry.byte_length > _MAX_FILE_BYTES:
                raise PrefixError("prefix-file-limit")
            if (
                entry.sha256 is None
                or not _SHA256.fullmatch(entry.sha256)
                or entry.link_target is not None
                or entry.link_target_bytes_escaped is not None
            ):
                raise PrefixError("prefix-entry-record")
        else:
            if not entry.link_target or entry.sha256 is not None:
                raise PrefixError("prefix-entry-record")
            target = _encoded(entry.link_target)
            if len(target) != entry.byte_length or entry.link_target_bytes_escaped != _escaped(
                target
            ):
                raise PrefixError("prefix-entry-record")
    data = canonical_json_bytes(entry.model_dump(mode="json"))
    # Include the maximum extra indentation of a nested inventory entry.
    budget.charge(len(data) + 4 * data.count(b"\n"))


def _metadata_fields(value: InodeMetadataObservationV1) -> tuple[object, ...]:
    return (
        value.observer_id,
        value.scope,
        value.coverage,
        value.list_status,
        value.list_errno,
        value.name_list_size_bytes,
        value.names_bytes_escaped,
    )


def _semantic_fields(entry: PrefixEntryV1) -> tuple[object, ...]:
    return (
        entry.kind,
        entry.mode,
        entry.uid,
        entry.gid,
        entry.byte_length,
        entry.sha256,
        entry.link_target,
        entry.nlink,
        _metadata_fields(entry.metadata),
    )


def _hash_file(fd: int, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        count = min(remaining, _CHUNK_BYTES)
        chunk = os.read(fd, count)
        if not chunk or len(chunk) > count:
            raise PrefixError("prefix-content-changed")
        digest.update(chunk)
        remaining -= len(chunk)
    if os.read(fd, 1):
        raise PrefixError("prefix-content-changed")
    return digest.hexdigest()


def _capture(
    budget: _Budget, parent_fd: int, name: str, path: str, expected: PrefixEntryV1 | None
) -> PrefixEntryV1:
    _leaf(name)
    _encoded(path)
    parent = _identity(os.fstat(parent_fd))
    with _opened(budget, parent_fd, name, os.O_PATH) as held:
        identity = _identity(os.fstat(held))
        kind = _kind(identity)
        if not path and kind != "directory":
            raise PrefixError("prefix-root-type")
        if kind != "directory" and identity.nlink != 1:
            raise PrefixError("prefix-non-directory-links")
        if expected is not None:
            _same(identity, expected.identity)
        target: str | None = None
        if kind == "file":
            if not 0 <= identity.size <= _MAX_FILE_BYTES:
                raise PrefixError("prefix-file-limit")
            if expected is None:
                budget.payload += identity.size
                if budget.payload > _MAX_PAYLOAD_BYTES:
                    raise PrefixError("prefix-payload-limit")
        elif kind == "symlink":
            if not 0 < identity.size <= _MAX_PATH_BYTES:
                raise PrefixError("prefix-text-limit")
            raw_target = os.readlink(b"", dir_fd=held)
            try:
                target = raw_target.decode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise PrefixError("prefix-text-utf8") from exc
            if len(_encoded(target)) != identity.size:
                raise PrefixError("prefix-link-changed")
        with budget.descriptors(2):  # The reused observer duplicates parent and holds one leaf.
            observed = observe_inode_metadata(parent_fd, name)
        _same(observed.parent_before, parent)
        _same(observed.parent_after, parent)
        entry = PrefixEntryV1(
            path=path,
            path_bytes_escaped=_escaped(path.encode("utf-8")),
            kind=kind,
            mode=stat.S_IMODE(identity.mode),
            uid=identity.uid,
            gid=identity.gid,
            byte_length=None if kind == "directory" else identity.size,
            sha256=(expected.sha256 if expected else "0" * 64) if kind == "file" else None,
            link_target=target,
            link_target_bytes_escaped=_escaped(target.encode("utf-8"))
            if target is not None
            else None,
            nlink=None if kind == "directory" else identity.nlink,
            identity=identity,
            metadata=observed,
        )
        # Validate and charge the fixed-width digest placeholder BEFORE reading payload.
        _validate_entry(entry, budget)
        if kind == "file" and expected is None:
            with _opened(budget, parent_fd, name, os.O_RDONLY | os.O_NONBLOCK) as data_fd:
                _same(_identity(os.fstat(data_fd)), identity)
                digest = _hash_file(data_fd, identity.size)
                _same(_identity(os.fstat(data_fd)), identity)
            entry = entry.model_copy(update={"sha256": digest})
        _same(_identity(os.fstat(held)), identity)
        _binding(budget, parent_fd, name, identity)
        _same(_identity(os.fstat(parent_fd)), parent)
        if expected is not None and entry != expected:
            raise PrefixError("prefix-entry-changed")
        return entry


def _directory_names(budget: _Budget, fd: int) -> tuple[str, ...]:
    names: list[str] = []
    seen: set[str] = set()
    with budget.descriptors(1), os.scandir(fd) as iterator:
        for item in iterator:
            name = item.name
            _leaf(name)
            if name in seen:
                raise PrefixError("prefix-directory-changed")
            if len(names) >= _MAX_MEMBERS:
                raise PrefixError("prefix-member-limit")
            budget.charge(len(canonical_json_bytes(name)))
            names.append(name)
            seen.add(name)
    return tuple(sorted(names, key=lambda value: value.encode("utf-8")))


@dataclass(slots=True)
class _Frame:
    entry: PrefixEntryV1
    parent_fd: int
    name: str
    fd: int
    names: tuple[str, ...]
    resources: ExitStack
    index: int = 0


def _frame(budget: _Budget, parent_fd: int, name: str, entry: PrefixEntryV1) -> _Frame:
    resources = ExitStack()
    try:
        fd = resources.enter_context(_opened(budget, parent_fd, name, os.O_RDONLY | os.O_DIRECTORY))
        _same(_identity(os.fstat(fd)), entry.identity)
        names = _directory_names(budget, fd)
        return _Frame(entry, parent_fd, name, fd, names, resources)
    except BaseException:
        resources.close()
        raise


def _walk(
    budget: _Budget, parent_fd: int, name: str, baseline: dict[str, PrefixEntryV1] | None = None
) -> dict[str, PrefixEntryV1]:
    root = _capture(budget, parent_fd, name, "", baseline[""] if baseline is not None else None)
    records = {"": root} if baseline is None else {}
    frames: list[_Frame] = []
    count = 0
    try:
        frames.append(_frame(budget, parent_fd, name, root))
        while frames:
            current = frames[-1]
            if current.index == len(current.names):
                if _directory_names(budget, current.fd) != current.names:
                    raise PrefixError("prefix-directory-changed")
                _same(_identity(os.fstat(current.fd)), current.entry.identity)
                _binding(budget, current.parent_fd, current.name, current.entry.identity)
                frames.pop().resources.close()
                continue
            leaf = current.names[current.index]
            current.index += 1
            path = current.entry.path + "/" + leaf if current.entry.path else leaf
            _encoded(path)
            if len(frames) > _MAX_DEPTH:
                raise PrefixError("prefix-depth-limit")
            count += 1
            if count > _MAX_MEMBERS:
                raise PrefixError("prefix-member-limit")
            expected = baseline.get(path) if baseline is not None else None
            if baseline is not None and expected is None:
                raise PrefixError("prefix-directory-changed")
            entry = _capture(budget, current.fd, leaf, path, expected)
            if baseline is None:
                records[path] = entry
            if entry.kind == "directory":
                frames.append(_frame(budget, current.fd, leaf, entry))
        if baseline is not None and count != len(baseline) - 1:
            raise PrefixError("prefix-directory-changed")
        return records
    finally:
        for frame in reversed(frames):
            frame.resources.close()


def inspect_prefix(parent_fd: int, name: str) -> PrefixInventoryV1:
    """Return completed observations only after a guarded walk and final rewalk."""

    _leaf(name)
    if not _supported_abi():
        raise PrefixError("prefix-abi-unsupported")
    budget = _Budget()
    try:
        with budget.descriptors(1):
            parent = os.dup(parent_fd)
            try:
                parent_identity = _identity(os.fstat(parent))
                if not stat.S_ISDIR(parent_identity.mode):
                    raise PrefixError("prefix-parent-type")
                records = _walk(budget, parent, name)
                _walk(budget, parent, name, records)
                _binding(budget, parent, name, records[""].identity)
                _same(_identity(os.fstat(parent)), parent_identity)
            finally:
                os.close(parent)
    except (OSError, MetadataError) as exc:
        raise PrefixError("prefix-observation-failed") from exc
    root = records.pop("")
    return PrefixInventoryV1(
        root=root,
        entries=tuple(sorted(records.values(), key=lambda entry: entry.path.encode("utf-8"))),
        member_count=len(records),
        regular_payload_bytes=budget.payload,
        evidence_bytes_charged=budget.evidence,
        peak_owned_fds=budget.peak_fds,
    )


def _validated_map(inventory: PrefixInventoryV1) -> dict[str, PrefixEntryV1]:
    if (
        inventory.validation != "complete"
        or inventory.metadata_coverage != "unknown"
        or inventory.link_closure != "not-checked"
    ):
        raise PrefixError("prefix-inventory-record")
    if (
        type(inventory.schema_version) is not int
        or inventory.schema_version != 1
        or inventory.inventory_id != "rocm-prefix-observation-v1"
        or inventory.resolution_policy != "openat2-beneath-no-symlinks-no-xdev"
    ):
        raise PrefixError("prefix-inventory-record")
    if len(inventory.entries) > _MAX_MEMBERS or inventory.member_count != len(inventory.entries):
        raise PrefixError("prefix-member-limit")
    if inventory.root.path or inventory.root.kind != "directory":
        raise PrefixError("prefix-root-type")
    if (
        not 0 <= inventory.evidence_bytes_charged <= _MAX_EVIDENCE_BYTES
        or not 0 <= inventory.peak_owned_fds <= _MAX_FDS
    ):
        raise PrefixError("prefix-inventory-record")
    budget = _Budget()
    _validate_entry(inventory.root, budget)
    records = {"": inventory.root}
    previous = b""
    for entry in inventory.entries:
        encoded = _encoded(entry.path)
        if not encoded or encoded <= previous:
            raise PrefixError("prefix-map-order")
        _validate_entry(entry, budget)
        parent_path = entry.path.rpartition("/")[0]
        parent = records.get(parent_path)
        if parent is None or parent.kind != "directory":
            raise PrefixError("prefix-map-parent")
        _same(entry.metadata.parent_before, parent.identity)
        if entry.kind == "file":
            assert entry.byte_length is not None
            budget.payload += entry.byte_length
            if budget.payload > _MAX_PAYLOAD_BYTES:
                raise PrefixError("prefix-payload-limit")
        records[entry.path] = entry
        previous = encoded
    if budget.payload != inventory.regular_payload_bytes:
        raise PrefixError("prefix-payload-record")
    return records


def compare_prefixes(
    expected: PrefixInventoryV1, observed: PrefixInventoryV1
) -> PrefixComparisonV1:
    """Compare recorded semantic fields only; equal unknown evidence is not admission."""

    left, right = _validated_map(expected), _validated_map(observed)
    different = 0
    sample: list[str] = []
    for path in sorted(left.keys() | right.keys(), key=lambda value: value.encode("utf-8")):
        a, b = left.get(path), right.get(path)
        if a is None or b is None or _semantic_fields(a) != _semantic_fields(b):
            different += 1
            if len(sample) < _MAX_DIFFERENCES:
                sample.append(path)
    return PrefixComparisonV1(
        semantic_equal=different == 0,
        differing_path_count=different,
        differing_paths=tuple(sample),
        sample_truncated=different > len(sample),
    )


def _link_work(steps: list[int], amount: int) -> None:
    steps[0] += amount
    if steps[0] > _MAX_LINK_STEPS:
        raise PrefixError("prefix-link-work-limit")


def _resolve_link(
    entry: PrefixEntryV1, records: dict[str, PrefixEntryV1], steps: list[int]
) -> PrefixLinkV1:
    assert entry.link_target is not None
    resolved = entry.path.split("/")[:-1]
    initial = entry.link_target.split("/")
    _link_work(steps, len(resolved) + len(initial))
    pending = deque(initial)
    pending_bytes = sum(len(part.encode("utf-8")) + 1 for part in initial)
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = {
        (tuple(resolved), (entry.path.rsplit("/", 1)[-1],))
    }
    expansions = 1  # Reading this link's target is already one expansion.
    status: _LinkStatus = "absolute" if entry.link_target.startswith("/") else "resolved"
    if expansions > _MAX_LINK_EXPANSIONS:
        status = "expansion-limit"
    while pending and status == "resolved":
        if pending_bytes > _MAX_PENDING_BYTES:
            raise PrefixError("prefix-link-pending-limit")
        _link_work(steps, 1)
        part = pending.popleft()
        pending_bytes -= len(part.encode("utf-8")) + 1
        current = records["/".join(resolved)]
        if current.kind != "directory":
            status = "not-directory"
        elif part in {"", "."}:
            continue
        elif part == "..":
            if not resolved:
                status = "escape"
            else:
                resolved.pop()
        else:
            candidate = "/".join([*resolved, part])
            target = records.get(candidate)
            if target is None:
                status = "dangling"
            elif target.kind != "symlink":
                resolved.append(part)
            else:
                # Account for component copies used by cycle detection, not just pops.
                _link_work(steps, len(resolved) + len(pending) + 1)
                state = (tuple(resolved), (part, *pending))
                if state in seen:
                    status = "cycle"
                elif expansions >= _MAX_LINK_EXPANSIONS:
                    status = "expansion-limit"
                else:
                    seen.add(state)
                    expansions += 1
                    assert target.link_target is not None
                    if target.link_target.startswith("/"):
                        status = "absolute"
                    else:
                        parts = target.link_target.split("/")
                        _link_work(steps, len(parts))
                        pending_bytes += sum(len(item.encode("utf-8")) + 1 for item in parts)
                        if pending_bytes > _MAX_PENDING_BYTES:
                            raise PrefixError("prefix-link-pending-limit")
                        pending.extendleft(reversed(parts))
    path = "/".join(resolved) if status == "resolved" else None
    return PrefixLinkV1(
        path=entry.path,
        status=status,
        resolved_path=path,
        resolved_kind=records[path].kind if path is not None else None,
        expansions=expansions,
    )


def resolve_inventory_links(inventory: PrefixInventoryV1) -> PrefixLinkMapV1:
    """Resolve only the completed map, expanding links before subsequent dot-dot."""

    records = _validated_map(inventory)
    evidence = 4096
    steps = [0]
    links: list[PrefixLinkV1] = []
    for entry in inventory.entries:
        if entry.kind == "symlink":
            result = _resolve_link(entry, records, steps)
            evidence += len(canonical_json_bytes(result.model_dump(mode="json")))
            if evidence > _MAX_LINK_EVIDENCE_BYTES:
                raise PrefixError("prefix-evidence-limit")
            links.append(result)
    return PrefixLinkMapV1(links=tuple(links), component_steps=steps[0])

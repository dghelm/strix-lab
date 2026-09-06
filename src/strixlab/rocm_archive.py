"""Bounded gzip archive inspection with strict POSIX and explicit GNU profiles.

No extraction or provenance acceptance. The grammar is frozen in
docs/rocm10-bringup.md. Internal member events are provisional until the entire
iterator, compressed stream and input-identity checks have completed.
"""

from __future__ import annotations

import hashlib
import os
import posixpath
import re
import stat
import unicodedata
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from strixlab.secure_fs import readonly_open_flags
from strixlab.serialization import canonical_json_bytes

_BLOCK_BYTES = 512
_CHUNK_BYTES = 64 * 1024
_MAX_MEMBERS = 1_000_000
_MAX_FILE_BYTES = 8 * 1024**3
_MAX_PAYLOAD_BYTES = 32 * 1024**3
_MAX_EXPANSION_RATIO = 64
_MAX_METADATA_BYTES = 64 * 1024**2
_MAX_TEXT_BYTES = 4096
_OCTAL = re.compile(rb" *[0-7]+[\x00 ]+")
_CHECKSUM = re.compile(rb"[0-7]{6}\x00 ")

type _Kind = Literal["file", "directory", "symlink"]


class ArchiveError(ValueError):
    """Structural inspection failed; the message is a bounded reason code."""


class ArchiveEntryV1(BaseModel):
    """Archive observations, not an installed filesystem entry or authorization."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    ordinal: int = Field(ge=1)
    header_offset: int = Field(ge=0)
    path: str
    path_bytes_escaped: str
    kind: _Kind
    mode: int = Field(ge=0, le=0o777)
    payload_size_bytes: int = Field(ge=0, le=_MAX_FILE_BYTES)
    sha256: str | None
    link_target: str | None
    link_target_bytes_escaped: str | None


class ArchiveManifestV1(BaseModel):
    """A completed structural report. Local digests do not authenticate AMD bytes."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    schema_version: Literal[1] = 1
    parser_id: Literal["rocm-archive-ustar-v1"] = "rocm-archive-ustar-v1"
    validation: Literal["complete"] = "complete"
    admission: Literal["structural-only"] = "structural-only"
    symlink_validation: Literal["lexical-only"] = "lexical-only"
    observed_sha256: str
    compressed_size_bytes: int = Field(ge=0)
    expanded_size_bytes: int = Field(ge=0)
    regular_payload_bytes: int = Field(ge=0, le=_MAX_PAYLOAD_BYTES)
    member_count: int = Field(ge=0, le=_MAX_MEMBERS)
    entries: tuple[ArchiveEntryV1, ...]

    def canonical_bytes(self) -> bytes:
        """Reuse the repository's canonical JSON encoding; no tree digest is added."""

        return canonical_json_bytes(self.model_dump(mode="json"))


def _text(raw: bytes) -> str:
    if len(raw) > _MAX_TEXT_BYTES:
        raise ArchiveError("text-limit")
    try:
        value = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ArchiveError("text-utf8") from exc
    if unicodedata.normalize("NFC", value) != value or any(
        unicodedata.category(char).startswith("C") for char in value
    ):
        raise ArchiveError("text-unicode")
    if "\\" in value:
        raise ArchiveError("text-backslash")
    return value


def _field_text(raw: bytes) -> str:
    value, separator, suffix = raw.partition(b"\0")
    if separator and any(suffix):
        raise ArchiveError("text-padding")
    return _text(value)


def _octal(raw: bytes) -> int:
    if _OCTAL.fullmatch(raw) is None:
        raise ArchiveError("header-octal")
    return int(raw.strip(b" \0"), 8)


def _escaped(value: str) -> str:
    return "".join(f"\\x{byte:02x}" for byte in value.encode("utf-8"))


def _member_path(value: str, kind: _Kind) -> str:
    # Directory syntax permits removing exactly one trailing slash, not normalizing.
    if kind == "directory" and value.endswith("/"):
        value = value[:-1]
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ArchiveError("member-path")
    return value


def _link_target(value: str, member_path: str) -> str:
    if not value or value.startswith("/") or posixpath.normpath(value) != value:
        raise ArchiveError("link-target")
    joined = posixpath.normpath(posixpath.join(posixpath.dirname(member_path), value))
    if joined == ".." or joined.startswith("../"):
        raise ArchiveError("link-escape")
    return value


@dataclass(frozen=True, slots=True)
class _Header:
    ordinal: int
    offset: int
    path: str
    kind: _Kind
    mode: int
    size: int
    target: str | None

    def entry(self, digest: str | None) -> ArchiveEntryV1:
        return ArchiveEntryV1(
            ordinal=self.ordinal,
            header_offset=self.offset,
            path=self.path,
            path_bytes_escaped=_escaped(self.path),
            kind=self.kind,
            mode=self.mode,
            payload_size_bytes=self.size,
            sha256=digest,
            link_target=self.target,
            link_target_bytes_escaped=_escaped(self.target) if self.target is not None else None,
        )


@dataclass(frozen=True, slots=True)
class _ProvisionalStart:
    header: _Header
    validation: Literal["provisional"] = field(default="provisional", init=False)


@dataclass(frozen=True, slots=True)
class _ProvisionalChunk:
    data: bytes
    validation: Literal["provisional"] = field(default="provisional", init=False)


@dataclass(frozen=True, slots=True)
class _ProvisionalEnd:
    entry: ArchiveEntryV1
    validation: Literal["provisional"] = field(default="provisional", init=False)


type _ProvisionalEvent = _ProvisionalStart | _ProvisionalChunk | _ProvisionalEnd


def _check_header_checksum(block: bytes) -> None:
    checksum = block[148:156]
    if _CHECKSUM.fullmatch(checksum) is None or int(checksum[:6], 8) != (
        sum(block[:148]) + 8 * 32 + sum(block[156:])
    ):
        raise ArchiveError("header-checksum")


def _parse_header(block: bytes, ordinal: int, offset: int) -> _Header:
    if block[257:265] != b"ustar\x0000":
        raise ArchiveError("header-ustar")
    if any(block[500:512]):
        raise ArchiveError("header-reserved")
    _check_header_checksum(block)
    kind: _Kind
    match block[156:157]:
        case b"0" | b"\0":
            kind = "file"
        case b"5":
            kind = "directory"
        case b"2":
            kind = "symlink"
        case _:
            raise ArchiveError("header-type")
    mode = _octal(block[100:108])
    _octal(block[108:116])  # UID/GID/mtime are inert, but still have a strict wire grammar.
    _octal(block[116:124])
    size = _octal(block[124:136])
    _octal(block[136:148])
    if mode & ~0o777 or (kind == "symlink" and mode != 0o777):
        raise ArchiveError("header-mode")
    if kind != "file" and size != 0:
        raise ArchiveError("nonregular-payload")
    for device_field in (block[329:337], block[337:345]):
        if any(device_field) and _octal(device_field) != 0:
            raise ArchiveError("header-device")
    name = _field_text(block[:100])
    prefix = _field_text(block[345:500])
    if not name:
        raise ArchiveError("member-path")
    path = _member_path(_text((prefix + "/" + name if prefix else name).encode()), kind)
    target = _field_text(block[157:257])
    if kind != "symlink" and target:
        raise ArchiveError("header-linkname")
    _field_text(block[265:297])
    _field_text(block[297:329])
    return _Header(
        ordinal,
        offset,
        path,
        kind,
        mode,
        size,
        _link_target(target, path) if kind == "symlink" else None,
    )


@dataclass(slots=True)
class _Budget:
    members: int = 0
    payload_bytes: int = 0
    metadata_bytes: int = 0

    def add(self, header: _Header) -> None:
        if self.members >= _MAX_MEMBERS:
            raise ArchiveError("member-count-limit")
        if header.size > _MAX_FILE_BYTES:
            raise ArchiveError("file-size-limit")
        if self.payload_bytes + header.size > _MAX_PAYLOAD_BYTES:
            raise ArchiveError("payload-size-limit")
        # The digest has fixed length; charge the final entry before retaining it or its path.
        entry = header.entry("0" * 64 if header.kind == "file" else None)
        charge = len(canonical_json_bytes(entry.model_dump(mode="json")))
        if self.metadata_bytes + charge > _MAX_METADATA_BYTES:
            raise ArchiveError("metadata-size-limit")
        self.members += 1
        self.payload_bytes += header.size
        self.metadata_bytes += charge


def _check_expansion(expanded: int, compressed: int) -> None:
    if expanded > compressed * _MAX_EXPANSION_RATIO:
        raise ArchiveError("gzip-expansion-limit")


class _GzipStream:
    def __init__(self, fd: int, compressed_size: int) -> None:
        self.fd = fd
        self.compressed_size = compressed_size
        self.digest = hashlib.sha256()
        self.expanded_size = 0

    def chunks(self) -> Iterator[bytes]:
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        consumed = 0
        pending = b""
        while True:
            if pending:
                data = pending
            elif consumed < self.compressed_size:
                data = os.read(self.fd, min(_CHUNK_BYTES, self.compressed_size - consumed))
                if not data:
                    raise ArchiveError("archive-premature-eof")
                consumed += len(data)
                self.digest.update(data)
            else:
                # Boundedly drain any decoder state; never use an unbounded flush().
                data = b""
            try:
                output = decoder.decompress(data, _CHUNK_BYTES)
            except zlib.error as exc:
                raise ArchiveError("gzip-invalid") from exc
            self.expanded_size += len(output)
            _check_expansion(self.expanded_size, self.compressed_size)
            pending = decoder.unconsumed_tail
            if decoder.eof:
                if decoder.unused_data or consumed != self.compressed_size:
                    raise ArchiveError("gzip-trailing")
                if output:
                    yield output
                if os.read(self.fd, 1):
                    raise ArchiveError("archive-input-changed")
                return
            if output:
                yield output
            elif not data:
                raise ArchiveError("gzip-truncated")


class _DecodedReader:
    """Small lookahead over bounded decoded chunks; no payload-size allocation."""

    def __init__(self, chunks: Iterator[bytes]) -> None:
        self.chunks = chunks
        self.pending = b""
        self.offset = 0

    def read(self, size: int) -> bytes:
        if not 0 <= size <= _CHUNK_BYTES:
            raise ArchiveError("read-size-limit")
        output = bytearray()
        while len(output) < size:
            if not self.pending:
                self.pending = next(self.chunks, b"")
                if not self.pending:
                    break
            count = min(size - len(output), len(self.pending))
            output.extend(self.pending[:count])
            self.pending = self.pending[count:]
        self.offset += len(output)
        return bytes(output)

    def exact(self, size: int) -> bytes:
        data = self.read(size)
        if len(data) != size:
            raise ArchiveError("tar-truncated")
        return data


def _validate_topology(kinds: dict[str, _Kind]) -> None:
    for path in kinds:
        parent = posixpath.dirname(path)
        while parent:
            if kinds.get(parent) != "directory":
                raise ArchiveError("member-parent")
            parent = posixpath.dirname(parent)


def _iter_provisional_members(reader: _DecodedReader) -> Iterator[_ProvisionalEvent]:
    """Emit explicitly provisional starts, payload chunks and member observations.

    Exhaustion validates the stream/topology, but does NOT validate descriptor
    stability or return an admitted manifest. Only _consume_archive completes
    that boundary for inspect_archive and future consumers. Never use a
    separate permissive tar parser or individual events as admission evidence.
    """

    kinds: dict[str, _Kind] = {}
    budget = _Budget()
    while True:
        offset = reader.offset
        block = reader.exact(_BLOCK_BYTES)
        if not any(block):
            if reader.read(_BLOCK_BYTES) != bytes(_BLOCK_BYTES):
                raise ArchiveError("tar-termination")
            while tail := reader.read(_CHUNK_BYTES):
                if any(tail):
                    raise ArchiveError("tar-trailing")
            if reader.offset % _BLOCK_BYTES:
                raise ArchiveError("tar-alignment")
            _validate_topology(kinds)
            return
        header = _parse_header(block, budget.members + 1, offset)
        if header.path in kinds:
            raise ArchiveError("member-duplicate")
        budget.add(header)
        kinds[header.path] = header.kind
        yield _ProvisionalStart(header)
        digest = hashlib.sha256()
        remaining = header.size
        while remaining:
            payload = reader.exact(min(remaining, _CHUNK_BYTES))
            digest.update(payload)
            remaining -= len(payload)
            yield _ProvisionalChunk(payload)
        if any(reader.exact(-header.size % _BLOCK_BYTES)):
            raise ArchiveError("tar-payload-padding")
        yield _ProvisionalEnd(header.entry(digest.hexdigest() if header.kind == "file" else None))


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def inspect_archive(directory_fd: int, name: str) -> ArchiveManifestV1:
    """Inspect a regular archive leaf beneath a caller-held directory descriptor.

    Returns only after full structural validation and observed-drift checks.
    A returned manifest is not proof of provenance, safe link resolution or an
    approved installed prefix. No filesystem entries are written.
    """

    return _consume_archive(directory_fd, name)


def _consume_archive(
    directory_fd: int,
    name: str,
    consumer: Callable[[_ProvisionalEvent], None] | None = None,
) -> ArchiveManifestV1:
    """Run the single strict lifecycle, optionally delivering provisional events.

    The synchronous callback cannot accept an entry or finish validation. Its
    return value is ignored; exceptions abort the operation and close the input.
    Only the returned manifest establishes complete structural inspection after
    all stream and input-identity checks. Callback side effects are not rolled
    back: any future writer must quarantine them until this function returns.
    """

    def collect(reader: _DecodedReader) -> list[ArchiveEntryV1]:
        entries: list[ArchiveEntryV1] = []
        for event in _iter_provisional_members(reader):
            if consumer is not None:
                consumer(event)
            if isinstance(event, _ProvisionalEnd):
                entries.append(event.entry)
        return entries

    entries, evidence = _run_archive(directory_fd, name, collect)
    return ArchiveManifestV1(
        observed_sha256=evidence.sha256,
        compressed_size_bytes=evidence.compressed_size,
        expanded_size_bytes=evidence.expanded_size,
        regular_payload_bytes=sum(entry.payload_size_bytes for entry in entries),
        member_count=len(entries),
        entries=tuple(sorted(entries, key=lambda entry: entry.path.encode("utf-8"))),
    )


@dataclass(frozen=True, slots=True)
class _StreamEvidence:
    sha256: str
    compressed_size: int
    expanded_size: int


def _run_archive[T](
    directory_fd: int, name: str, parse: Callable[[_DecodedReader], T]
) -> tuple[T, _StreamEvidence]:
    """Share descriptor and gzip completion, without selecting a tar grammar."""

    if not name or name in {".", ".."} or "/" in name or "\0" in name:
        raise ArchiveError("archive-name")
    if not hasattr(os, "O_NOFOLLOW"):
        raise ArchiveError("archive-nofollow-unavailable")
    descriptor: int | None = None
    try:
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            raise ArchiveError("archive-directory")
        before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ArchiveError("archive-not-regular")
        descriptor = os.open(name, readonly_open_flags() | os.O_NONBLOCK, dir_fd=directory_fd)
        opened = os.fstat(descriptor)
        if _identity(opened) != _identity(before):
            raise ArchiveError("archive-input-changed")
        compressed = _GzipStream(descriptor, opened.st_size)
        reader = _DecodedReader(compressed.chunks())
        result = parse(reader)
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _identity(opened) != _identity(after) or _identity(after) != _identity(named):
            raise ArchiveError("archive-input-changed")
        return result, _StreamEvidence(
            compressed.digest.hexdigest(), opened.st_size, compressed.expanded_size
        )
    except OSError as exc:
        raise ArchiveError("archive-io") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


type _GnuKind = Literal["file", "directory", "symlink", "hardlink"]


class GnuArchiveEntryV1(BaseModel):
    """GNU wire evidence; a hardlink has no payload or payload digest."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    ordinal: int = Field(ge=2)
    header_offset: int = Field(ge=512)
    path: str
    path_bytes_escaped: str
    kind: _GnuKind
    mode: int = Field(ge=0, le=0o777)
    payload_size_bytes: int = Field(ge=0, le=_MAX_FILE_BYTES)
    sha256: str | None
    wire_name: str
    wire_name_bytes_escaped: str
    wire_linkname: str | None
    wire_linkname_bytes_escaped: str | None
    link_target: str | None
    link_target_bytes_escaped: str | None
    hardlink_target: str | None
    longname_header_ordinal: int | None
    longname_header_offset: int | None
    longname_payload_size_bytes: int | None


class GnuRootMarkerV1(BaseModel):
    """The inert first header is excluded from the material tree."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    ordinal: Literal[1] = 1
    header_offset: Literal[0] = 0
    wire_name: Literal["./"] = "./"
    kind: Literal["directory"] = "directory"
    mode: Literal[493] = 0o755
    payload_size_bytes: Literal[0] = 0


class GnuHardlinkCopyV1(BaseModel):
    """Independent-file deployment projection, separate from wire payloads."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    path: str
    source_path: str
    mode: int = Field(ge=0, le=0o777)
    materialized_size_bytes: int = Field(ge=0, le=_MAX_FILE_BYTES)
    materialized_sha256: str


class GnuArchiveManifestV1(BaseModel):
    """Completed inspection under the explicit narrow GNU profile."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    schema_version: Literal[1] = 1
    parser_id: Literal["rocm-archive-gnu-direct-hardlink-v1"] = (
        "rocm-archive-gnu-direct-hardlink-v1"
    )
    validation: Literal["complete"] = "complete"
    admission: Literal["structural-only"] = "structural-only"
    symlink_validation: Literal["lexical-only"] = "lexical-only"
    observed_sha256: str
    compressed_size_bytes: int = Field(ge=0)
    expanded_size_bytes: int = Field(ge=0)
    regular_payload_bytes: int = Field(ge=0, le=_MAX_PAYLOAD_BYTES)
    independent_copy_bytes: int = Field(ge=0, le=_MAX_PAYLOAD_BYTES)
    materialized_regular_bytes: int = Field(ge=0, le=_MAX_PAYLOAD_BYTES)
    member_count: int = Field(ge=0, le=_MAX_MEMBERS)
    raw_header_count: int = Field(ge=1, le=2 * _MAX_MEMBERS + 1)
    longname_control_count: int = Field(ge=0, le=_MAX_MEMBERS)
    root_marker: GnuRootMarkerV1
    entries: tuple[GnuArchiveEntryV1, ...]
    hardlink_copies: tuple[GnuHardlinkCopyV1, ...]

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


@dataclass(frozen=True, slots=True)
class _LongName:
    name: str
    raw: bytes
    ordinal: int
    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class _GnuHeader:
    ordinal: int
    offset: int
    path: str
    kind: _GnuKind
    mode: int
    size: int
    wire_name: str
    wire_linkname: str | None
    target: str | None
    hardlink_target: str | None
    longname: _LongName | None

    def entry(self, digest: str | None) -> GnuArchiveEntryV1:
        return GnuArchiveEntryV1(
            ordinal=self.ordinal,
            header_offset=self.offset,
            path=self.path,
            path_bytes_escaped=_escaped(self.path),
            kind=self.kind,
            mode=self.mode,
            payload_size_bytes=self.size,
            sha256=digest,
            wire_name=self.wire_name,
            wire_name_bytes_escaped=_escaped(self.wire_name),
            wire_linkname=self.wire_linkname,
            wire_linkname_bytes_escaped=(
                _escaped(self.wire_linkname) if self.wire_linkname is not None else None
            ),
            link_target=self.target,
            link_target_bytes_escaped=_escaped(self.target) if self.target is not None else None,
            hardlink_target=self.hardlink_target,
            longname_header_ordinal=self.longname.ordinal if self.longname else None,
            longname_header_offset=self.longname.offset if self.longname else None,
            longname_payload_size_bytes=self.longname.size if self.longname else None,
        )


@dataclass(frozen=True, slots=True)
class _GnuProvisionalStart:
    header: _GnuHeader
    validation: Literal["provisional"] = field(default="provisional", init=False)


@dataclass(frozen=True, slots=True)
class _GnuProvisionalEnd:
    entry: GnuArchiveEntryV1
    validation: Literal["provisional"] = field(default="provisional", init=False)


type _GnuProvisionalEvent = _GnuProvisionalStart | _ProvisionalChunk | _GnuProvisionalEnd


def _gnu_fields(block: bytes) -> tuple[bytes, int, int, str]:
    if block[257:265] != b"ustar  \0":
        raise ArchiveError("header-gnu")
    if any(block[345:512]):
        raise ArchiveError("header-gnu-tail")
    _check_header_checksum(block)
    kind = block[156:157]
    if kind not in (b"0", b"5", b"2", b"1", b"L"):
        raise ArchiveError("header-type")
    mode = _octal(block[100:108])
    _octal(block[108:116])
    _octal(block[116:124])
    size = _octal(block[124:136])
    _octal(block[136:148])
    allowed_modes = {
        b"0": (0o644, 0o755),
        b"5": (0o755,),
        b"2": (0o777,),
        b"1": (0o644, 0o755),
        b"L": (0o644,),
    }
    if mode not in allowed_modes[kind]:
        raise ArchiveError("header-mode")
    if kind not in (b"0", b"L") and size:
        raise ArchiveError("nonregular-payload")
    for device_field in (block[329:337], block[337:345]):
        if any(device_field) and _octal(device_field) != 0:
            raise ArchiveError("header-device")
    _field_text(block[265:297])
    _field_text(block[297:329])
    linkname = _field_text(block[157:257])
    if kind not in (b"1", b"2") and linkname:
        raise ArchiveError("header-linkname")
    return kind, mode, size, linkname


def _gnu_path(name: str, kind: _Kind) -> str:
    if not name.startswith("./"):
        raise ArchiveError("gnu-member-prefix")
    return _member_path(name[2:], kind)


def _gnu_header(
    block: bytes,
    ordinal: int,
    offset: int,
    fields: tuple[bytes, int, int, str],
    longname: _LongName | None,
) -> _GnuHeader:
    wire_kind, mode, size, linkname = fields
    if longname is not None:
        if wire_kind != b"0" or block[:100] != longname.raw[:100]:
            raise ArchiveError("gnu-longname-next")
        name = longname.name
    else:
        name = _field_text(block[:100])
    kind: _GnuKind
    match wire_kind:
        case b"0":
            kind = "file"
        case b"5":
            kind = "directory"
        case b"2":
            kind = "symlink"
        case b"1":
            kind = "hardlink"
        case _:
            raise ArchiveError("gnu-longname-next")
    path = _gnu_path(name, "file" if kind == "hardlink" else kind)
    target = _link_target(linkname, path) if kind == "symlink" else None
    hardlink_target = _gnu_path(linkname, "file") if kind == "hardlink" else None
    return _GnuHeader(
        ordinal,
        offset,
        path,
        kind,
        mode,
        size,
        name,
        linkname or None,
        target,
        hardlink_target,
        longname,
    )


@dataclass(slots=True)
class _GnuBudget:
    members: int = 0
    regular_bytes: int = 0
    copy_bytes: int = 0
    metadata_bytes: int = 0
    raw_headers: int = 0
    controls: int = 0

    def charge(self, size: int) -> None:
        if self.metadata_bytes + size > _MAX_METADATA_BYTES:
            raise ArchiveError("metadata-size-limit")
        self.metadata_bytes += size

    def header(self) -> None:
        if self.raw_headers >= 2 * _MAX_MEMBERS + 1:
            raise ArchiveError("raw-header-count-limit")
        self.raw_headers += 1

    def member(self, header: _GnuHeader, copy: GnuHardlinkCopyV1 | None) -> None:
        if self.members >= _MAX_MEMBERS:
            raise ArchiveError("member-count-limit")
        if header.size > _MAX_FILE_BYTES:
            raise ArchiveError("file-size-limit")
        if self.regular_bytes + header.size > _MAX_PAYLOAD_BYTES:
            raise ArchiveError("payload-size-limit")
        additional = copy.materialized_size_bytes if copy else 0
        if additional > _MAX_FILE_BYTES:
            raise ArchiveError("file-size-limit")
        if self.regular_bytes + self.copy_bytes + header.size + additional > _MAX_PAYLOAD_BYTES:
            raise ArchiveError("materialization-size-limit")
        entry = header.entry("0" * 64 if header.kind == "file" else None)
        charge = len(canonical_json_bytes(entry.model_dump(mode="json")))
        if copy is not None:
            charge += len(canonical_json_bytes(copy.model_dump(mode="json")))
        self.charge(charge)
        self.members += 1
        self.regular_bytes += header.size
        self.copy_bytes += additional


@dataclass(slots=True)
class _GnuParsed:
    budget: _GnuBudget
    root: GnuRootMarkerV1
    entries: dict[str, GnuArchiveEntryV1]
    copies: list[GnuHardlinkCopyV1]


def _parse_gnu(
    reader: _DecodedReader, consumer: Callable[[_GnuProvisionalEvent], None] | None
) -> _GnuParsed:
    budget = _GnuBudget()
    root = GnuRootMarkerV1()
    budget.charge(len(canonical_json_bytes(root.model_dump(mode="json"))))
    entries: dict[str, GnuArchiveEntryV1] = {}
    kinds: dict[str, _Kind] = {}
    copies: list[GnuHardlinkCopyV1] = []
    pending: _LongName | None = None
    while True:
        offset = reader.offset
        block = reader.exact(_BLOCK_BYTES)
        if not any(block):
            if pending is not None:
                raise ArchiveError("gnu-longname-dangling")
            if budget.raw_headers == 0:
                raise ArchiveError("gnu-root-marker")
            if reader.read(_BLOCK_BYTES) != bytes(_BLOCK_BYTES):
                raise ArchiveError("tar-termination")
            while tail := reader.read(_CHUNK_BYTES):
                if any(tail):
                    raise ArchiveError("tar-trailing")
            if reader.offset % _BLOCK_BYTES:
                raise ArchiveError("tar-alignment")
            _validate_topology(kinds)
            return _GnuParsed(budget, root, entries, copies)
        budget.header()
        fields = _gnu_fields(block)
        wire_kind, mode, size, linkname = fields
        if budget.raw_headers == 1:
            if (wire_kind, mode, size, linkname, _field_text(block[:100])) != (
                b"5",
                0o755,
                0,
                "",
                "./",
            ):
                raise ArchiveError("gnu-root-marker")
            continue
        if wire_kind == b"L":
            if pending is not None:
                raise ArchiveError("gnu-longname-next")
            if _field_text(block[:100]) != "././@LongLink":
                raise ArchiveError("gnu-longname-control")
            if not 102 <= size <= _MAX_TEXT_BYTES + 1:
                raise ArchiveError("gnu-longname-size")
            # Bound and charge before reading or retaining the one pending name.
            # Effective name/escapes are also charged with its material entry.
            budget.charge(size + 256)
            raw = reader.exact(size)
            if raw[-1:] != b"\0" or b"\0" in raw[:-1]:
                raise ArchiveError("gnu-longname-nul")
            if any(reader.exact(-size % _BLOCK_BYTES)):
                raise ArchiveError("tar-payload-padding")
            name = _text(raw[:-1])
            _gnu_path(name, "file")
            pending = _LongName(name, raw[:-1], budget.raw_headers, offset, size)
            budget.controls += 1
            continue
        header = _gnu_header(block, budget.raw_headers, offset, fields, pending)
        pending = None
        if header.path in entries:
            raise ArchiveError("member-duplicate")
        copy = None
        if header.kind == "hardlink":
            source = entries.get(header.hardlink_target or "")
            if source is None or source.kind != "file" or source.sha256 is None:
                raise ArchiveError("gnu-hardlink-source")
            if source.mode != header.mode:
                raise ArchiveError("gnu-hardlink-mode")
            copy = GnuHardlinkCopyV1(
                path=header.path,
                source_path=source.path,
                mode=header.mode,
                materialized_size_bytes=source.payload_size_bytes,
                materialized_sha256=source.sha256,
            )
        budget.member(header, copy)
        kinds[header.path] = "file" if header.kind == "hardlink" else header.kind
        if consumer is not None:
            consumer(_GnuProvisionalStart(header))
        digest = hashlib.sha256()
        remaining = header.size
        while remaining:
            payload = reader.exact(min(remaining, _CHUNK_BYTES))
            digest.update(payload)
            remaining -= len(payload)
            if consumer is not None:
                consumer(_ProvisionalChunk(payload))
        if any(reader.exact(-header.size % _BLOCK_BYTES)):
            raise ArchiveError("tar-payload-padding")
        entry = header.entry(digest.hexdigest() if header.kind == "file" else None)
        if consumer is not None:
            consumer(_GnuProvisionalEnd(entry))
        entries[entry.path] = entry
        if copy is not None:
            copies.append(copy)


def inspect_gnu_archive(directory_fd: int, name: str) -> GnuArchiveManifestV1:
    """Explicit GNU subset opt-in; observe bytes without extraction or admission."""

    return _consume_gnu_archive(directory_fd, name)


def _consume_gnu_archive(
    directory_fd: int,
    name: str,
    consumer: Callable[[_GnuProvisionalEvent], None] | None = None,
) -> GnuArchiveManifestV1:
    """GNU events remain provisional through CRC, topology and input-drift checks.

    Root and long-name controls produce no member events. Callback side effects
    are not rolled back on failure. This distinct contract never emits V1 entries.
    """

    parsed, evidence = _run_archive(directory_fd, name, lambda reader: _parse_gnu(reader, consumer))
    budget = parsed.budget
    return GnuArchiveManifestV1(
        observed_sha256=evidence.sha256,
        compressed_size_bytes=evidence.compressed_size,
        expanded_size_bytes=evidence.expanded_size,
        regular_payload_bytes=budget.regular_bytes,
        independent_copy_bytes=budget.copy_bytes,
        materialized_regular_bytes=budget.regular_bytes + budget.copy_bytes,
        member_count=budget.members,
        raw_header_count=budget.raw_headers,
        longname_control_count=budget.controls,
        root_marker=parsed.root,
        entries=tuple(
            sorted(parsed.entries.values(), key=lambda entry: entry.path.encode("utf-8"))
        ),
        hardlink_copies=tuple(sorted(parsed.copies, key=lambda copy: copy.path.encode("utf-8"))),
    )

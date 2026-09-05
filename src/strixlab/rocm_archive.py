"""Bounded structural inspection of one gzip/POSIX-ustar archive.

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
from collections.abc import Iterator
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


def _parse_header(block: bytes, ordinal: int, offset: int) -> _Header:
    if block[257:265] != b"ustar\x0000":
        raise ArchiveError("header-ustar")
    if any(block[500:512]):
        raise ArchiveError("header-reserved")
    checksum = block[148:156]
    if _CHECKSUM.fullmatch(checksum) is None or int(checksum[:6], 8) != (
        sum(block[:148]) + 8 * 32 + sum(block[156:])
    ):
        raise ArchiveError("header-checksum")
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
    stability or return an admitted manifest. Only inspect_archive completes
    that boundary. A future extractor must use this interpretation, never a
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
        entries = tuple(
            event.entry
            for event in _iter_provisional_members(reader)
            if isinstance(event, _ProvisionalEnd)
        )
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if _identity(opened) != _identity(after) or _identity(after) != _identity(named):
            raise ArchiveError("archive-input-changed")
        return ArchiveManifestV1(
            observed_sha256=compressed.digest.hexdigest(),
            compressed_size_bytes=opened.st_size,
            expanded_size_bytes=compressed.expanded_size,
            regular_payload_bytes=sum(entry.payload_size_bytes for entry in entries),
            member_count=len(entries),
            entries=tuple(sorted(entries, key=lambda entry: entry.path.encode("utf-8"))),
        )
    except OSError as exc:
        raise ArchiveError("archive-io") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)

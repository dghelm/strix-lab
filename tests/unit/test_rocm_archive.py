"""Wire fixtures are built byte by byte, independently of tarfile/the reader."""

from __future__ import annotations

import gzip
import hashlib
import os
import random
import struct
import zlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from strixlab import rocm_archive as archive


def _octal(value: int, width: int) -> bytes:
    return f"{value:0{width - 1}o}".encode() + b"\0"


def _checksum(header: bytearray) -> bytes:
    header[148:156] = b" " * 8
    header[148:156] = f"{sum(header):06o}".encode() + b"\0 "
    return bytes(header)


def _header(
    name: bytes = b"file",
    *,
    kind: bytes = b"0",
    size: int = 0,
    mode: int = 0o644,
    target: bytes = b"",
    prefix: bytes = b"",
) -> bytes:
    assert len(name) <= 100 and len(target) <= 100 and len(prefix) <= 155
    header = bytearray(512)
    header[: len(name)] = name
    header[100:108] = _octal(mode, 8)
    header[108:116] = _octal(123, 8)  # inert ownership, not the test user's UID
    header[116:124] = _octal(456, 8)
    header[124:136] = _octal(size, 12)
    header[136:148] = _octal(789, 12)
    header[156:157] = kind
    header[157 : 157 + len(target)] = target
    header[257:265] = b"ustar\00000"
    header[265:271] = b"author"
    header[297:302] = b"group"
    header[345 : 345 + len(prefix)] = prefix
    return _checksum(header)


def _replace(header: bytes, offset: int, value: bytes) -> bytes:
    changed = bytearray(header)
    changed[offset : offset + len(value)] = value
    return _checksum(changed)


def _file(name: bytes = b"file", payload: bytes = b"data", **kwargs: object) -> bytes:
    # kwargs deliberately limited to mode in this independent fixture helper.
    mode = kwargs.get("mode", 0o644)
    assert isinstance(mode, int)
    return _header(name, size=len(payload), mode=mode) + payload + b"\0" * (-len(payload) % 512)


def _gzip(tar: bytes) -> bytes:
    # Stored DEFLATE blocks keep ordinary structural fixtures below the ratio cap.
    return gzip.compress(tar, compresslevel=0, mtime=0)


def _inspect(tmp_path: Path, compressed: bytes) -> archive.ArchiveManifestV1:
    (tmp_path / "input.gz").write_bytes(compressed)
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        return archive.inspect_archive(fd, "input.gz")
    finally:
        os.close(fd)


def _reject(tmp_path: Path, tar: bytes, reason: str) -> None:
    with pytest.raises(archive.ArchiveError, match=reason):
        _inspect(tmp_path, _gzip(tar))


def test_manifest_has_exact_independent_values_and_no_authority(tmp_path: Path) -> None:
    # Child before parent and a forward link are explicitly permitted.
    tar = (
        _file(b"dir/z", b"hello", mode=0o755)
        + _header(b"dir/link", kind=b"2", mode=0o777, target=b"../other")
        + _header(b"dir/", kind=b"5", mode=0o750)
        + _file(b"other", b"")
        + bytes(1024)
    )
    compressed = _gzip(tar)
    result = _inspect(tmp_path, compressed)
    assert result.model_dump(mode="json") == {
        "schema_version": 1,
        "parser_id": "rocm-archive-ustar-v1",
        "validation": "complete",
        "admission": "structural-only",
        "symlink_validation": "lexical-only",
        "observed_sha256": hashlib.sha256(compressed).hexdigest(),
        "compressed_size_bytes": len(compressed),
        "expanded_size_bytes": 3584,
        "regular_payload_bytes": 5,
        "member_count": 4,
        "entries": [
            {
                "ordinal": 3,
                "header_offset": 1536,
                "path": "dir",
                "path_bytes_escaped": "\\x64\\x69\\x72",
                "kind": "directory",
                "mode": 0o750,
                "payload_size_bytes": 0,
                "sha256": None,
                "link_target": None,
                "link_target_bytes_escaped": None,
            },
            {
                "ordinal": 2,
                "header_offset": 1024,
                "path": "dir/link",
                "path_bytes_escaped": "\\x64\\x69\\x72\\x2f\\x6c\\x69\\x6e\\x6b",
                "kind": "symlink",
                "mode": 0o777,
                "payload_size_bytes": 0,
                "sha256": None,
                "link_target": "../other",
                "link_target_bytes_escaped": "\\x2e\\x2e\\x2f\\x6f\\x74\\x68\\x65\\x72",
            },
            {
                "ordinal": 1,
                "header_offset": 0,
                "path": "dir/z",
                "path_bytes_escaped": "\\x64\\x69\\x72\\x2f\\x7a",
                "kind": "file",
                "mode": 0o755,
                "payload_size_bytes": 5,
                "sha256": "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
                "link_target": None,
                "link_target_bytes_escaped": None,
            },
            {
                "ordinal": 4,
                "header_offset": 2048,
                "path": "other",
                "path_bytes_escaped": "\\x6f\\x74\\x68\\x65\\x72",
                "kind": "file",
                "mode": 0o644,
                "payload_size_bytes": 0,
                "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                "link_target": None,
                "link_target_bytes_escaped": None,
            },
        ],
    }
    assert result.canonical_bytes() == _inspect(tmp_path, compressed).canonical_bytes()
    assert result.canonical_bytes().endswith(b"\n")
    assert set(tmp_path.iterdir()) == {tmp_path / "input.gz"}  # nothing extracted


@pytest.mark.parametrize("kind", [b"0", b"\0"])
def test_regular_type_flags_and_empty_archive(tmp_path: Path, kind: bytes) -> None:
    assert _inspect(tmp_path, _gzip(_header(kind=kind) + bytes(1024))).member_count == 1
    assert _inspect(tmp_path, _gzip(bytes(1536))).member_count == 0


@pytest.mark.parametrize("number", [b"0000644\0", b"    644 ", b"644\0 \0  "])
def test_explicit_octal_padding(tmp_path: Path, number: bytes) -> None:
    header = _replace(_header(), 100, number)
    assert _inspect(tmp_path, _gzip(header + bytes(1024))).entries[0].mode == 0o644


@pytest.mark.parametrize(
    "number",
    [
        b"\0" * 8,
        b" " * 8,
        b"00000644",
        b"0000648\0",
        b"+000644\0",
        b"\t000644\0",
        b"\x80" + bytes(7),
    ],
)
def test_reject_ambiguous_numeric_fields(tmp_path: Path, number: bytes) -> None:
    _reject(tmp_path, _replace(_header(), 100, number) + bytes(1024), "octal")


@pytest.mark.parametrize(
    ("offset", "value", "reason"),
    [
        (257, b"ustar ", "ustar"),
        (263, b" \0", "ustar"),
        (500, b"x", "reserved"),
        (329, _octal(1, 8), "device"),
        (337, _octal(1, 8), "device"),
        (4, b"\0x", "text-padding"),
        (157, b"target", "linkname"),
        (265, b"\xff", "utf8"),
    ],
)
def test_reject_header_ambiguities(tmp_path: Path, offset: int, value: bytes, reason: str) -> None:
    _reject(tmp_path, _replace(_header(), offset, value) + bytes(1024), reason)


@pytest.mark.parametrize("kind", [b"1", b"3", b"4", b"6", b"7", b"x", b"g", b"L", b"K", b"S", b"X"])
def test_reject_every_unapproved_type(tmp_path: Path, kind: bytes) -> None:
    _reject(tmp_path, _header(kind=kind) + bytes(1024), "type")


@pytest.mark.parametrize(
    "name",
    [b"", b"/file", b"../file", b"a/../b", b"./file", b"a//b", b"a/./b", b"file/", b"a\\b"],
)
def test_reject_unsafe_member_names(tmp_path: Path, name: bytes) -> None:
    _reject(tmp_path, _header(name) + bytes(1024), "path|text")


@pytest.mark.parametrize(
    "name", [b"\xff", "e\u0301".encode(), b"a\n", "a\u202e".encode(), "\ue000".encode()]
)
def test_reject_noncanonical_unicode(tmp_path: Path, name: bytes) -> None:
    _reject(tmp_path, _header(name) + bytes(1024), "utf8|unicode")


@pytest.mark.parametrize(
    "tar",
    [
        _header(b"d", kind=b"5") + _header(b"d/", kind=b"5"),
        _header(b"a") + _header(b"a"),
        _header(b"missing/child"),
        _header(b"a/child") + _header(b"a"),
        _header(b"a", kind=b"2", mode=0o777, target=b"b") + _header(b"a/child"),
    ],
)
def test_final_topology_rejects_aliases_and_implicit_parents(tmp_path: Path, tar: bytes) -> None:
    _reject(tmp_path, tar + bytes(1024), "duplicate|parent")


@pytest.mark.parametrize(
    "target", [b"", b"/x", b"../x", b"a/../x", b"./x", b"a//x", b"x/", b"x\\y"]
)
def test_unsafe_symlink_targets(tmp_path: Path, target: bytes) -> None:
    _reject(tmp_path, _header(kind=b"2", mode=0o777, target=target) + bytes(1024), "link|text")


def test_lexical_links_do_not_claim_resolved_prefix_safety(tmp_path: Path) -> None:
    tar = (
        _header(b"a", kind=b"2", mode=0o777, target=b"b")
        + _header(b"b", kind=b"2", mode=0o777, target=b"a")
        + _header(b"dangling", kind=b"2", mode=0o777, target=b"missing")
        + _header(b"parent", kind=b"2", mode=0o777, target=b".")
        + bytes(1024)
    )
    result = _inspect(tmp_path, _gzip(tar))
    assert result.symlink_validation == "lexical-only"
    assert result.admission == "structural-only"


@pytest.mark.parametrize("mode", [0o1644, 0o2644, 0o4644, 0o100644])
def test_reject_special_mode_bits(tmp_path: Path, mode: int) -> None:
    _reject(tmp_path, _header(mode=mode) + bytes(1024), "mode")


@pytest.mark.parametrize("kind", [b"2", b"5"])
def test_nonregular_payload_is_forbidden(tmp_path: Path, kind: bytes) -> None:
    _reject(
        tmp_path,
        _header(kind=kind, target=b"x" if kind == b"2" else b"", mode=0o777, size=1) + bytes(1536),
        "payload",
    )


def test_symlink_mode_must_be_777(tmp_path: Path) -> None:
    _reject(tmp_path, _header(kind=b"2", target=b"x") + bytes(1024), "mode")


@pytest.mark.parametrize(
    ("tar", "reason"),
    [
        (bytes(512), "termination"),
        (bytes(512) + _header(), "termination"),
        (bytes(1024) + _header(), "trailing"),
        (bytes(1025), "alignment"),
        (_header()[:511], "truncated"),
        (_header(size=1024) + b"x", "truncated"),
        (_header(size=1) + b"x" + b"y" + bytes(510) + bytes(1024), "payload-padding"),
    ],
)
def test_tar_framing_and_zero_padding(tmp_path: Path, tar: bytes, reason: str) -> None:
    _reject(tmp_path, tar, reason)


def test_checksum_spelling_and_value(tmp_path: Path) -> None:
    good = _header()
    for damaged in (good[:154] + b" \0" + good[156:], b"X" + good[1:]):
        _reject(tmp_path, damaged + bytes(1024), "checksum")


@pytest.mark.parametrize("suffix", [b"\0", b"junk", _gzip(bytes(1024))])
def test_gzip_rejects_every_suffix_and_second_member(tmp_path: Path, suffix: bytes) -> None:
    with pytest.raises(archive.ArchiveError, match="gzip-trailing"):
        _inspect(tmp_path, _gzip(bytes(1024)) + suffix)


def test_gzip_requires_valid_framing_and_trailer(tmp_path: Path) -> None:
    good = _gzip(bytes(1024))
    bad_crc = good[:-8] + bytes([good[-8] ^ 1]) + good[-7:]
    bad_size = good[:-4] + struct.pack("<I", 1)
    bad_flags = good[:3] + b"\xe0" + good[4:]
    for compressed in (b"", good[:-1], bad_crc, bad_size, bad_flags, zlib.compress(bytes(1024))):
        with pytest.raises(archive.ArchiveError, match="gzip"):
            _inspect(tmp_path, compressed)


def test_full_width_text_prefix_unicode_and_zero_device_fields(tmp_path: Path) -> None:
    parent = b"p" * 100
    child = "\u00e9".encode() * 50  # exactly 100 UTF-8 bytes, no terminator
    header = _header(child, prefix=parent)
    header = _replace(header, 329, b"0000000\0")
    header = _replace(header, 337, b"      0 ")
    tar = header + _header(parent, kind=b"5") + _header(b"z") + bytes(1024)
    result = _inspect(tmp_path, _gzip(tar))
    assert [entry.path for entry in result.entries] == [
        parent.decode(),
        parent.decode() + "/" + child.decode(),
        "z",
    ]
    assert result.entries[1].path_bytes_escaped.endswith("\\xc3\\xa9" * 50)
    assert result.entries[1].ordinal == 1


def test_utf8_sorting_is_independent_of_archive_order(tmp_path: Path) -> None:
    tar = _header("\u00e9".encode()) + _header(b"z") + _header(b"a") + bytes(1024)
    assert [entry.path for entry in _inspect(tmp_path, _gzip(tar)).entries] == ["a", "z", "\u00e9"]


@pytest.mark.parametrize("offset,width", [(108, 8), (116, 8), (124, 12), (136, 12)])
def test_every_numeric_field_has_the_same_grammar(tmp_path: Path, offset: int, width: int) -> None:
    _reject(tmp_path, _replace(_header(), offset, b" " * width) + bytes(1024), "octal")


def test_text_bound_and_directory_extra_slash(tmp_path: Path) -> None:
    with pytest.raises(archive.ArchiveError, match="text-limit"):
        archive._text(b"x" * 4097)
    assert archive._text(b"x" * 4096) == "x" * 4096
    _reject(tmp_path, _header(b"d//", kind=b"5") + bytes(1024), "path")


def _optional_gzip(tar: bytes) -> bytes:
    # Exercise every optional header, crossing several compressed-read boundaries.
    header = (
        b"\x1f\x8b\x08\x1f"
        + bytes(4)
        + b"\x00\xff"
        + struct.pack("<H", 3)
        + b"abc"
        + b"inert-name" * 20_000
        + b"\0"
        + b"inert-comment\xff" * 10_000
        + b"\0"
    )
    header += struct.pack("<H", zlib.crc32(header) & 0xFFFF)
    compressor = zlib.compressobj(level=0, wbits=-15)
    body = compressor.compress(tar) + compressor.flush()
    return header + body + struct.pack("<II", zlib.crc32(tar), len(tar))


def test_gzip_optional_data_is_validated_but_not_retained(tmp_path: Path) -> None:
    data = _optional_gzip(_header() + bytes(1024))
    result = _inspect(tmp_path, data)
    assert result.member_count == 1
    assert result.observed_sha256 == hashlib.sha256(data).hexdigest()
    assert b"inert-name" not in result.canonical_bytes()
    assert b"inert-comment" not in result.canonical_bytes()
    # Flip MTIME: harmless to DEFLATE but the optional header CRC must catch it.
    bad_header_crc = data[:4] + b"\x01" + data[5:]
    with pytest.raises(archive.ArchiveError, match="gzip-invalid"):
        _inspect(tmp_path, bad_header_crc)
    # Unterminated FNAME and an incomplete FEXTRA, no fallback interpretation.
    for bad in (
        b"\x1f\x8b\x08\x08" + bytes(6) + b"name",
        b"\x1f\x8b\x08\x04" + bytes(6) + b"\xff\xffx",
    ):
        with pytest.raises(archive.ArchiveError, match="gzip"):
            _inspect(tmp_path, bad)


def test_bounded_reads_and_decoding_with_large_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An early compressible run exercises unconsumed_tail. Later entropy ensures
    # the complete archive is below 64:1; a running denominator would reject it.
    payload = b"x" * (4 * 65536) + random.Random(17).randbytes(2 * 65536)
    tar = _file(payload=payload) + bytes(1024)
    compressed = gzip.compress(tar, compresslevel=6, mtime=0)
    path = tmp_path / "large.gz"
    path.write_bytes(compressed)
    real_read = os.read
    requests: list[int] = []

    def tracked_read(fd: int, count: int) -> bytes:
        requests.append(count)
        return real_read(fd, count)

    monkeypatch.setattr(archive.os, "read", tracked_read)
    fd = os.open(path, os.O_RDONLY)
    try:
        stream = archive._GzipStream(fd, len(compressed))
        chunks = list(stream.chunks())
    finally:
        os.close(fd)
    assert max(requests) <= 65536
    assert max(map(len, chunks)) <= 65536
    assert len(chunks) > 4 and b"".join(chunks) == tar
    assert stream.digest.hexdigest() == hashlib.sha256(compressed).hexdigest()
    result = _inspect(tmp_path, compressed)
    assert result.entries[0].sha256 == hashlib.sha256(payload).hexdigest()
    assert result.regular_payload_bytes == len(payload)


def test_short_nonempty_reads_are_not_premature_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_read = os.read

    def short_read(fd: int, count: int) -> bytes:
        return real_read(fd, min(count, 7))

    monkeypatch.setattr(archive.os, "read", short_read)
    assert _inspect(tmp_path, _gzip(_file() + bytes(1024))).member_count == 1


def test_all_expanded_bytes_count_in_ratio_including_terminal_padding(tmp_path: Path) -> None:
    with pytest.raises(archive.ArchiveError, match="gzip-expansion-limit"):
        _inspect(tmp_path, gzip.compress(bytes(128 * 1024), mtime=0))
    archive._check_expansion(123 * 64, 123)
    with pytest.raises(archive.ArchiveError, match="gzip-expansion-limit"):
        archive._check_expansion(123 * 64 + 1, 123)


def test_fixed_resource_limits_without_large_allocations() -> None:
    # Exercise the accounting seam at actual production limits, not a reduced
    # user-controlled policy. Ustar cannot encode 8 GiB with this size grammar.
    small = archive._Header(1, 0, "f", "file", 0o600, 1, None)
    big = archive._Header(1, 0, "f", "file", 0o600, 8 * 1024**3, None)
    members = archive._Budget(members=999_999)
    members.add(small)
    assert members.members == 1_000_000
    with pytest.raises(archive.ArchiveError, match="member-count-limit"):
        members.add(small)
    assert members.members == 1_000_000
    payload = archive._Budget(payload_bytes=24 * 1024**3)
    payload.add(big)
    assert payload.payload_bytes == 32 * 1024**3
    with pytest.raises(archive.ArchiveError, match="payload-size-limit"):
        payload.add(small)
    with pytest.raises(archive.ArchiveError, match="file-size-limit"):
        archive._Budget().add(archive._Header(1, 0, "f", "file", 0o600, 8 * 1024**3 + 1, None))
    empty = archive._Budget()
    empty.add(small)
    charge = empty.metadata_bytes
    metadata = archive._Budget(metadata_bytes=64 * 1024**2 - charge)
    metadata.add(small)
    assert metadata.metadata_bytes == 64 * 1024**2
    with pytest.raises(archive.ArchiveError, match="metadata-size-limit"):
        metadata.add(small)
    assert metadata.members == 1
    with pytest.raises(archive.ArchiveError, match="read-size-limit"):
        archive._DecodedReader(iter(())).read(65537)


def test_event_stream_preserves_member_boundaries_but_requires_full_consumption() -> None:
    tar = _file(b"child/f", b"hello") + _header(b"child", kind=b"5") + bytes(1024)
    reader = archive._DecodedReader(iter([tar[:777], tar[777:]]))
    events = list(archive._iter_provisional_members(reader))
    assert all(event.validation == "provisional" for event in events)
    assert isinstance(events[0], archive._ProvisionalStart)
    assert events[0].header.path == "child/f"
    assert isinstance(events[1], archive._ProvisionalChunk)
    assert events[1].data == b"hello"
    assert isinstance(events[2], archive._ProvisionalEnd)
    assert isinstance(events[3], archive._ProvisionalStart)
    assert events[3].header.offset == 1024
    # Earlier complete entries are still provisional: missing final parent fails.
    invalid = archive._DecodedReader(iter([_file(b"missing/f") + bytes(1024)]))
    iterator = archive._iter_provisional_members(invalid)
    assert isinstance(next(iterator), archive._ProvisionalStart)
    with pytest.raises(archive.ArchiveError, match="member-parent"):
        list(iterator)


@pytest.mark.parametrize("failure", ["crc", "termination", "topology"])
def test_provisional_end_is_not_admission_when_later_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    tar = _file(b"missing/f" if failure == "topology" else b"f")
    tar += bytes(512 if failure == "termination" else 1024)
    compressed = _gzip(tar)
    if failure == "crc":
        compressed = compressed[:-8] + bytes([compressed[-8] ^ 1]) + compressed[-7:]
    path = tmp_path / "late-error.gz"
    path.write_bytes(compressed)
    real_read = os.read

    def fragment_read(fd: int, count: int) -> bytes:
        return real_read(fd, min(count, 64))

    monkeypatch.setattr(archive.os, "read", fragment_read)
    fd = os.open(path, os.O_RDONLY)
    try:
        stream = archive._GzipStream(fd, len(compressed))
        events = archive._iter_provisional_members(archive._DecodedReader(stream.chunks()))
        assert isinstance(next(events), archive._ProvisionalStart)
        assert isinstance(next(events), archive._ProvisionalChunk)
        end = next(events)
        assert isinstance(end, archive._ProvisionalEnd)
        assert end.validation == "provisional"
        assert not isinstance(end, archive.ArchiveManifestV1)
        with pytest.raises(
            archive.ArchiveError, match="gzip-invalid|tar-termination|member-parent"
        ):
            list(events)
    finally:
        os.close(fd)
    with pytest.raises(archive.ArchiveError, match="gzip-invalid|tar-termination|member-parent"):
        _inspect(tmp_path, compressed)


def test_complete_iterator_still_needs_descriptor_stability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_iterator = archive._iter_provisional_members

    def changed_after_exhaustion(reader: archive._DecodedReader):
        for event in real_iterator(reader):
            assert event.validation == "provisional"
            yield event
        os.chmod(tmp_path / "input.gz", 0o400)

    monkeypatch.setattr(archive, "_iter_provisional_members", changed_after_exhaustion)
    with pytest.raises(archive.ArchiveError, match="archive-input-changed"):
        _inspect(tmp_path, _gzip(_file() + bytes(1024)))


def test_manifest_cannot_supply_approval_or_change_admission(tmp_path: Path) -> None:
    result = _inspect(tmp_path, _gzip(bytes(1024)))
    assert result.validation == "complete"
    for extra in (
        {"vendor_verified": True},
        {"installation_allowed": True},
        {"admission": "trusted"},
        {"validation": "provisional"},
    ):
        with pytest.raises(ValidationError):
            archive.ArchiveManifestV1.model_validate({**result.model_dump(), **extra})


@pytest.mark.parametrize("name", ["", ".", "..", "../input.gz", "a/b", "x\0y"])
def test_archive_leaf_name_is_single_component(tmp_path: Path, name: str) -> None:
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(archive.ArchiveError, match="archive-name"):
            archive.inspect_archive(fd, name)
    finally:
        os.close(fd)


def test_input_must_be_regular_under_a_directory_fd(tmp_path: Path) -> None:
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        (tmp_path / "real").write_bytes(_gzip(bytes(1024)))
        (tmp_path / "link").symlink_to("real")
        (tmp_path / "dir").mkdir()
        os.mkfifo(tmp_path / "fifo")
        for name in ("link", "dir", "fifo"):
            with pytest.raises(archive.ArchiveError, match="archive-not-regular"):
                archive.inspect_archive(fd, name)
        with pytest.raises(archive.ArchiveError, match="archive-io"):
            archive.inspect_archive(fd, "absent")
    finally:
        os.close(fd)
    regular_fd = os.open(tmp_path / "real", os.O_RDONLY)
    try:
        with pytest.raises(archive.ArchiveError, match="archive-directory"):
            archive.inspect_archive(regular_fd, "real")
    finally:
        os.close(regular_fd)


def test_missing_nofollow_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(archive.os, "O_NOFOLLOW")
    with pytest.raises(archive.ArchiveError, match="nofollow-unavailable"):
        _inspect(tmp_path, _gzip(bytes(1024)))


@pytest.mark.parametrize("change", ["same-size", "rename", "append"])
def test_observed_input_drift_fails_without_partial_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    compressed = _gzip(bytes(1024))
    path = tmp_path / "input.gz"
    real_read = os.read
    changed = False

    def changing_read(fd: int, count: int) -> bytes:
        nonlocal changed
        data = real_read(fd, count)
        if not changed:
            changed = True
            if change == "same-size":
                path.write_bytes(compressed[:4] + b"\x01" + compressed[5:])
            elif change == "rename":
                path.rename(tmp_path / "original.gz")
                path.write_bytes(_gzip(_header() + bytes(1024)))
            else:
                with path.open("ab") as output:
                    output.write(b"x")
        return data

    monkeypatch.setattr(archive.os, "read", changing_read)
    with pytest.raises(archive.ArchiveError, match="archive-input-changed"):
        _inspect(tmp_path, compressed)


@pytest.mark.parametrize("change", ["regular", "symlink", "fifo"])
def test_lstat_open_swap_cannot_redirect_or_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    real_open = os.open
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside bytes must not be read as the archive")

    def swapped_open(
        path: str | Path, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> int:
        if path == "input.gz":
            assert flags & os.O_NOFOLLOW and flags & os.O_NONBLOCK
            leaf = tmp_path / "input.gz"
            leaf.rename(tmp_path / "original.gz")
            if change == "regular":
                leaf.write_bytes(_gzip(_header() + bytes(1024)))
            elif change == "symlink":
                leaf.symlink_to(outside)
            else:
                os.mkfifo(leaf)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(archive.os, "open", swapped_open)
    with pytest.raises(archive.ArchiveError, match="archive-input-changed|archive-io"):
        _inspect(tmp_path, _gzip(bytes(1024)))
    assert outside.read_bytes() == b"outside bytes must not be read as the archive"


@pytest.mark.parametrize("failure", ["eof", "error"])
def test_read_failures_are_closed_and_descriptors_are_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    descriptors: list[int] = []

    def failing_read(fd: int, count: int) -> bytes:
        descriptors.append(fd)
        if failure == "error":
            raise OSError("injected")
        return b""

    monkeypatch.setattr(archive.os, "read", failing_read)
    with pytest.raises(archive.ArchiveError, match="archive-premature-eof|archive-io"):
        _inspect(tmp_path, _gzip(bytes(1024)))
    assert descriptors
    with pytest.raises(OSError):
        os.fstat(descriptors[0])

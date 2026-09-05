"""Independent byte fixtures for the deliberately narrow opt-in GNU grammar."""

from __future__ import annotations

import gzip
import hashlib
import os
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from strixlab import rocm_archive as archive


def _checksum(header: bytearray) -> bytes:
    header[148:156] = b" " * 8
    header[148:156] = f"{sum(header):06o}".encode() + b"\0 "
    return bytes(header)


def _header(
    name: bytes = b"./f",
    *,
    kind: bytes = b"0",
    mode: int = 0o644,
    size: int = 0,
    target: bytes = b"",
) -> bytes:
    assert len(name) <= 100 and len(target) <= 100
    block = bytearray(512)
    block[: len(name)] = name
    for offset, width, value in (
        (100, 8, mode),
        (108, 8, 123),
        (116, 8, 456),
        (124, 12, size),
        (136, 12, 789),
    ):
        block[offset : offset + width] = f"{value:0{width - 1}o}".encode() + b"\0"
    block[156:157] = kind
    block[157 : 157 + len(target)] = target
    block[257:265] = b"ustar  \0"
    block[265:269] = b"user"
    block[297:302] = b"group"
    return _checksum(block)


def _replace(header: bytes, offset: int, raw: bytes) -> bytes:
    block = bytearray(header)
    block[offset : offset + len(raw)] = raw
    return _checksum(block)


def _root() -> bytes:
    return _header(b"./", kind=b"5", mode=0o755)


def _payload(raw: bytes) -> bytes:
    return raw + bytes(-len(raw) % 512)


def _file(name: bytes = b"./f", data: bytes = b"abc", mode: int = 0o644) -> bytes:
    return _header(name, size=len(data), mode=mode) + _payload(data)


def _long(name: bytes, data: bytes = b"abc") -> bytes:
    return (
        _header(b"././@LongLink", kind=b"L", size=len(name) + 1)
        + _payload(name + b"\0")
        + _file(name[:100], data)
    )


def _compressed(tar: bytes) -> bytes:
    return gzip.compress(tar, compresslevel=0, mtime=0)


def _consume(
    tmp_path: Path,
    raw: bytes,
    consumer: Callable[[archive._GnuProvisionalEvent], None] | None = None,
    *,
    compressed: bool = False,
) -> archive.GnuArchiveManifestV1:
    (tmp_path / "a.gz").write_bytes(raw if compressed else _compressed(raw))
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        return archive._consume_gnu_archive(parent, "a.gz", consumer)
    finally:
        os.close(parent)


def _inspect(tmp_path: Path, material: bytes) -> archive.GnuArchiveManifestV1:
    return _consume(tmp_path, _root() + material + bytes(1024))


def _reject(tmp_path: Path, material: bytes, reason: str) -> None:
    with pytest.raises(archive.ArchiveError, match=f"^{reason}$"):
        _inspect(tmp_path, material)


def test_exact_wire_and_copy_evidence(tmp_path: Path) -> None:
    material = (
        _file(b"./z", b"hello", 0o755)
        + _header(b"./a", kind=b"1", mode=0o755, target=b"./z")
        + _header(b"./d/", kind=b"5", mode=0o755)
        + _header(b"./d/l", kind=b"2", mode=0o777, target=b"../z")
    )
    tar = _root() + material + bytes(1024)
    result = _consume(tmp_path, tar)
    assert result.parser_id == "rocm-archive-gnu-direct-hardlink-v1"
    assert (result.validation, result.admission, result.symlink_validation) == (
        "complete",
        "structural-only",
        "lexical-only",
    )
    assert result.observed_sha256 == hashlib.sha256(_compressed(tar)).hexdigest()
    assert result.compressed_size_bytes == len(_compressed(tar))
    assert result.expanded_size_bytes == len(tar)
    assert (result.member_count, result.raw_header_count, result.longname_control_count) == (
        4,
        5,
        0,
    )
    assert (
        result.regular_payload_bytes,
        result.independent_copy_bytes,
        result.materialized_regular_bytes,
    ) == (5, 5, 10)
    assert result.root_marker.model_dump() == dict(
        ordinal=1,
        header_offset=0,
        wire_name="./",
        kind="directory",
        mode=0o755,
        payload_size_bytes=0,
    )
    assert [entry.path for entry in result.entries] == ["a", "d", "d/l", "z"]
    hardlink = result.entries[0]
    assert hardlink.model_dump() == dict(
        ordinal=3,
        header_offset=1536,
        path="a",
        path_bytes_escaped="\\x61",
        kind="hardlink",
        mode=0o755,
        payload_size_bytes=0,
        sha256=None,
        wire_name="./a",
        wire_name_bytes_escaped="\\x2e\\x2f\\x61",
        wire_linkname="./z",
        wire_linkname_bytes_escaped="\\x2e\\x2f\\x7a",
        link_target=None,
        link_target_bytes_escaped=None,
        hardlink_target="z",
        longname_header_ordinal=None,
        longname_header_offset=None,
        longname_payload_size_bytes=None,
    )
    assert result.hardlink_copies[0].model_dump() == dict(
        path="a",
        source_path="z",
        mode=0o755,
        materialized_size_bytes=5,
        materialized_sha256=hashlib.sha256(b"hello").hexdigest(),
    )
    assert result.entries[2].link_target == "../z"
    assert result.entries[2].wire_linkname == "../z"
    assert result.canonical_bytes().endswith(b"\n")
    with pytest.raises(ValidationError):
        result.member_count = 3
    with pytest.raises(ValidationError):
        archive.GnuArchiveManifestV1.model_validate(
            {**result.model_dump(), "admission": "approved"}
        )


def test_empty_root_and_public_opt_in(tmp_path: Path) -> None:
    expected = _inspect(tmp_path, b"")
    assert expected.entries == expected.hardlink_copies == ()
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        assert archive.inspect_gnu_archive(fd, "a.gz") == expected
        with pytest.raises(archive.ArchiveError, match="header-ustar"):
            archive.inspect_archive(fd, "a.gz")
    finally:
        os.close(fd)


@pytest.mark.parametrize("name", [b"./" + b"x" * 99, b"./" + b"x" * 97 + "é".encode() + b"z"])
def test_long_name_offsets_and_split_utf8(tmp_path: Path, name: bytes) -> None:
    events: list[archive._GnuProvisionalEvent] = []
    result = _consume(tmp_path, _root() + _long(name) + bytes(1024), events.append)
    entry = result.entries[0]
    assert entry.path == name[2:].decode()
    assert entry.wire_name == name.decode()
    assert (
        entry.ordinal,
        entry.header_offset,
        entry.longname_header_ordinal,
        entry.longname_header_offset,
        entry.longname_payload_size_bytes,
    ) == (3, 1536, 2, 512, len(name) + 1)
    assert result.raw_header_count == 3 and result.longname_control_count == 1
    assert [type(event) for event in events] == [
        archive._GnuProvisionalStart,
        archive._ProvisionalChunk,
        archive._GnuProvisionalEnd,
    ]
    assert all(event.validation == "provisional" for event in events)
    assert isinstance(events[0], archive._GnuProvisionalStart)
    assert events[0].header.entry(entry.sha256) == entry
    assert (
        _consume(tmp_path, _root() + _long(name) + bytes(1024)).canonical_bytes()
        == result.canonical_bytes()
    )


@pytest.mark.parametrize(
    "root",
    [
        b"",
        _header(),
        _header(b".", kind=b"5", mode=0o755),
        _header(b"./", kind=b"0", mode=0o755),
        _header(b"././@LongLink", kind=b"L", size=102),
    ],
)
def test_required_initial_marker(tmp_path: Path, root: bytes) -> None:
    with pytest.raises(archive.ArchiveError, match="gnu-root-marker"):
        _consume(tmp_path, root + bytes(1024))


def test_repeated_root_rejected(tmp_path: Path) -> None:
    _reject(tmp_path, _root(), "member-path")


@pytest.mark.parametrize(
    ("offset", "value", "reason"),
    [
        (257, b"ustar\00000", "header-gnu"),
        (345, b"x", "header-gnu-tail"),
        (500, b"x", "header-gnu-tail"),
        (511, b"x", "header-gnu-tail"),
        (100, b"0000600\0", "header-mode"),
        (100, b"0004644\0", "header-mode"),
        (108, b"\x80", "header-octal"),
        (116, b"+", "header-octal"),
        (124, b"\x80", "header-octal"),
        (136, b"9", "header-octal"),
        (329, b"0000001\0", "header-device"),
        (337, b"0000001\0", "header-device"),
        (265, b"bad\0x", "text-padding"),
        (297, b"\xff", "text-utf8"),
        (157, b"target", "header-linkname"),
        (0, b"./f\0x", "text-padding"),
        (0, b"./\xff", "text-utf8"),
        (0, b"./e\xcc\x81", "text-unicode"),
        (0, b"./\\", "text-backslash"),
        (0, b"./\n", "text-unicode"),
    ],
)
def test_header_rejections(tmp_path: Path, offset: int, value: bytes, reason: str) -> None:
    _reject(tmp_path, _replace(_header(), offset, value), reason)


@pytest.mark.parametrize("kind", [b"\0", b"3", b"4", b"6", b"7", b"K", b"x", b"g", b"S", b"Z"])
def test_other_types_rejected(tmp_path: Path, kind: bytes) -> None:
    _reject(tmp_path, _header(kind=kind), "header-type")


@pytest.mark.parametrize("name", [b"f", b"/f", b"././f", b"./../f", b"./a//f", b"./f/", b"./"])
def test_paths_rejected(tmp_path: Path, name: bytes) -> None:
    _reject(
        tmp_path,
        _header(name),
        "gnu-member-prefix" if not name.startswith(b"./") else "member-path",
    )


@pytest.mark.parametrize("target", [b"", b"/f", b"../f", b"a/../f", b"a//f", b"./f"])
def test_symlink_policy(tmp_path: Path, target: bytes) -> None:
    _reject(
        tmp_path,
        _header(b"./l", kind=b"2", mode=0o777, target=target),
        "link-escape" if target == b"../f" else "link-target",
    )


@pytest.mark.parametrize("kind,mode", [(b"5", 0o755), (b"2", 0o777), (b"1", 0o644)])
def test_nonregular_payload(tmp_path: Path, kind: bytes, mode: int) -> None:
    _reject(tmp_path, _header(kind=kind, mode=mode, size=1), "nonregular-payload")


@pytest.mark.parametrize(
    "material",
    [
        _header(b"./h", kind=b"1", target=b"./f") + _file(),
        _header(b"./h", kind=b"1", target=b"./h"),
        _header(b"./h", kind=b"1", target=b"./missing"),
        _header(b"./d", kind=b"5", mode=0o755) + _header(b"./h", kind=b"1", target=b"./d"),
        _header(b"./l", kind=b"2", mode=0o777, target=b"f")
        + _header(b"./h", kind=b"1", target=b"./l"),
        _file()
        + _header(b"./h", kind=b"1", target=b"./f")
        + _header(b"./j", kind=b"1", target=b"./h"),
        _header(b"./h", kind=b"1", target=b"./j") + _header(b"./j", kind=b"1", target=b"./h"),
    ],
)
def test_hardlink_relationships_rejected(tmp_path: Path, material: bytes) -> None:
    _reject(tmp_path, material, "gnu-hardlink-source")


@pytest.mark.parametrize(
    "target,reason",
    [
        (b"f", "gnu-member-prefix"),
        (b"/f", "gnu-member-prefix"),
        (b"././f", "member-path"),
        (b"./../f", "member-path"),
        (b"./f/", "member-path"),
    ],
)
def test_hardlink_target_grammar(tmp_path: Path, target: bytes, reason: str) -> None:
    _reject(tmp_path, _file() + _header(b"./h", kind=b"1", target=target), reason)


def test_hardlink_mode_and_duplicate(tmp_path: Path) -> None:
    _reject(
        tmp_path,
        _file() + _header(b"./h", kind=b"1", mode=0o755, target=b"./f"),
        "gnu-hardlink-mode",
    )
    _reject(tmp_path, _file() + _header(b"./f", kind=b"1", target=b"./f"), "member-duplicate")
    _reject(
        tmp_path,
        _header(b"./d/", kind=b"5", mode=0o755) + _header(b"./d", kind=b"5", mode=0o755),
        "member-duplicate",
    )


def test_topology_and_late_directories(tmp_path: Path) -> None:
    _reject(tmp_path, _file(b"./d/f"), "member-parent")
    _reject(tmp_path, _file() + _file(b"./f/child"), "member-parent")
    result = _inspect(tmp_path, _file(b"./d/f") + _header(b"./d", kind=b"5", mode=0o755))
    assert result.member_count == 2


@pytest.mark.parametrize("size", [0, 1, 100, 101, 4098, 1024**3])
def test_longname_size_before_read(tmp_path: Path, size: int) -> None:
    _reject(tmp_path, _header(b"././@LongLink", kind=b"L", size=size), "gnu-longname-size")


@pytest.mark.parametrize("suffix", [b"x", b"\0\0", b"\0z"])
def test_longname_nul(tmp_path: Path, suffix: bytes) -> None:
    payload = b"./" + b"x" * 100 + suffix
    _reject(
        tmp_path,
        _header(b"././@LongLink", kind=b"L", size=len(payload)) + _payload(payload),
        "gnu-longname-nul",
    )


def test_longname_control_and_padding(tmp_path: Path) -> None:
    _reject(tmp_path, _header(b"./wrong", kind=b"L", size=102), "gnu-longname-control")
    control = _header(b"././@LongLink", kind=b"L", size=102)
    payload = b"./" + b"x" * 99 + b"\0"
    _reject(tmp_path, control + payload + b"x" + bytes(409), "tar-payload-padding")
    _reject(tmp_path, control + _payload(payload), "gnu-longname-dangling")
    _reject(tmp_path, control + _payload(payload) + control, "gnu-longname-next")
    _reject(tmp_path, control + _payload(payload) + _header(b"./wrong"), "gnu-longname-next")
    _reject(
        tmp_path,
        control + _payload(payload) + _header(payload[:100], kind=b"5", mode=0o755),
        "gnu-longname-next",
    )


@pytest.mark.parametrize(
    "limit,material,reason",
    [
        ("_MAX_MEMBERS", _file() + _file(b"./g"), "member-count-limit"),
        ("_MAX_FILE_BYTES", _header(size=2), "file-size-limit"),
        ("_MAX_PAYLOAD_BYTES", _file(data=b"a") + _header(b"./g", size=1), "payload-size-limit"),
        (
            "_MAX_PAYLOAD_BYTES",
            _file(data=b"a") + _header(b"./h", kind=b"1", target=b"./f"),
            "materialization-size-limit",
        ),
        ("_MAX_METADATA_BYTES", _file(), "metadata-size-limit"),
    ],
)
def test_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: str, material: bytes, reason: str
) -> None:
    monkeypatch.setattr(archive, limit, 1)
    _reject(tmp_path, material, reason)


def test_copy_budget_also_counts_later_regulars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(archive, "_MAX_PAYLOAD_BYTES", 3)
    _reject(
        tmp_path,
        _file(data=b"a") + _header(b"./h", kind=b"1", target=b"./f") + _header(b"./g", size=2),
        "materialization-size-limit",
    )


def test_long_control_evidence_before_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(archive, "_MAX_METADATA_BYTES", 200)
    _reject(tmp_path, _header(b"././@LongLink", kind=b"L", size=102), "metadata-size-limit")


def test_raw_header_limit_before_parse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(archive, "_MAX_MEMBERS", 0)
    _reject(tmp_path, _header(), "raw-header-count-limit")


@pytest.mark.parametrize(
    "tail,reason",
    [
        (bytes(512), "tar-termination"),
        (bytes(512) + b"x" * 512, "tar-termination"),
        (bytes(1024) + b"x" * 512, "tar-trailing"),
        (bytes(1025), "tar-alignment"),
    ],
)
def test_termination(tmp_path: Path, tail: bytes, reason: str) -> None:
    with pytest.raises(archive.ArchiveError, match=reason):
        _consume(tmp_path, _root() + _file() + tail)


def test_checksum_padding_and_truncation(tmp_path: Path) -> None:
    bad = bytearray(_header())
    bad[148] ^= 1
    _reject(tmp_path, bytes(bad), "header-checksum")
    _reject(tmp_path, _header(size=1) + b"ax" + bytes(510), "tar-payload-padding")
    with pytest.raises(archive.ArchiveError, match="tar-truncated"):
        _consume(tmp_path, _root() + _header(size=10) + b"x")


def test_late_crc_failure_and_bounded_provisional_payload(tmp_path: Path) -> None:
    raw = bytearray(_compressed(_root() + _file(data=b"a" * 200_000) + bytes(1024)))
    raw[-8] ^= 1
    events: list[archive._GnuProvisionalEvent] = []
    with pytest.raises(archive.ArchiveError, match="gzip-invalid"):
        _consume(tmp_path, bytes(raw), events.append, compressed=True)
    chunks = [event.data for event in events if isinstance(event, archive._ProvisionalChunk)]
    assert chunks and max(map(len, chunks)) <= 65536
    assert sum(map(len, chunks)) < 200_000


def test_gzip_completion_rules(tmp_path: Path) -> None:
    raw = _compressed(_root() + bytes(1024))
    for changed, reason in (
        (raw + raw, "gzip-trailing"),
        (raw[:-4], "gzip-truncated"),
        (gzip.compress(_root() + bytes(100_000)), "gzip-expansion-limit"),
    ):
        with pytest.raises(archive.ArchiveError, match=reason):
            _consume(tmp_path, changed, compressed=True)


def test_input_drift_after_events(tmp_path: Path) -> None:
    def mutate(event: archive._GnuProvisionalEvent) -> None:
        if isinstance(event, archive._GnuProvisionalEnd):
            os.chmod(tmp_path / "a.gz", 0o600)

    with pytest.raises(archive.ArchiveError, match="archive-input-changed"):
        _consume(tmp_path, _root() + _file() + bytes(1024), mutate)


def test_callback_failure_closes_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[int] = []
    original = os.open

    def record(*args: object, **kwargs: object) -> int:
        fd = original(*args, **kwargs)  # type: ignore[arg-type,call-overload]
        opened.append(fd)
        return fd

    def abort(event: archive._GnuProvisionalEvent) -> None:
        raise RuntimeError("consumer-failure")

    monkeypatch.setattr(os, "open", record)
    with pytest.raises(RuntimeError, match="consumer-failure"):
        _consume(tmp_path, _root() + _file() + bytes(1024), abort)
    assert len(opened) == 2
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_maximum_longname_and_zero_size_copies(tmp_path: Path) -> None:
    name = b"./" + b"x" * 4094
    result = _inspect(tmp_path, _long(name, b""))
    assert result.entries[0].longname_payload_size_bytes == 4097
    result = _inspect(
        tmp_path,
        _file(data=b"")
        + _header(b"./a", kind=b"1", target=b"./f")
        + _header(b"./b", kind=b"1", target=b"./f"),
    )
    assert len(result.hardlink_copies) == 2
    assert result.materialized_regular_bytes == result.independent_copy_bytes == 0
    assert all(
        copy.materialized_sha256 == hashlib.sha256(b"").hexdigest()
        for copy in result.hardlink_copies
    )


def test_copy_and_entry_evidence_budget_before_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material = _file() + _header(b"./h", kind=b"1", target=b"./f")
    result = _inspect(tmp_path, material)
    root_charge = len(archive.canonical_json_bytes(result.root_marker.model_dump(mode="json")))
    file_charge = len(archive.canonical_json_bytes(result.entries[0].model_dump(mode="json")))
    hard_charge = len(archive.canonical_json_bytes(result.entries[1].model_dump(mode="json")))
    copy_charge = len(
        archive.canonical_json_bytes(result.hardlink_copies[0].model_dump(mode="json"))
    )
    monkeypatch.setattr(
        archive, "_MAX_METADATA_BYTES", root_charge + file_charge + hard_charge + copy_charge - 1
    )
    events: list[archive._GnuProvisionalEvent] = []
    with pytest.raises(archive.ArchiveError, match="metadata-size-limit"):
        _consume(tmp_path, _root() + material + bytes(1024), events.append)
    assert [
        event.header.path for event in events if isinstance(event, archive._GnuProvisionalStart)
    ] == ["f"]


def test_longname_invalid_full_text(tmp_path: Path) -> None:
    for name, reason in (
        (b"./" + b"x" * 100 + b"\xff", "text-utf8"),
        (b"../" + b"x" * 100, "gnu-member-prefix"),
        (b"./../" + b"x" * 100, "member-path"),
    ):
        _reject(tmp_path, _long(name), reason)


def test_gnu_rejects_posix_and_strict_rejects_hardlink(tmp_path: Path) -> None:
    _reject(tmp_path, _replace(_header(), 257, b"ustar\00000"), "header-gnu")
    (tmp_path / "strict.gz").write_bytes(
        _compressed(
            _replace(_header(b"h", kind=b"1", target=b"f"), 257, b"ustar\00000") + bytes(1024)
        )
    )
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(archive.ArchiveError, match="header-type"):
            archive.inspect_archive(fd, "strict.gz")
    finally:
        os.close(fd)


def test_late_topology_failure_after_member_end(tmp_path: Path) -> None:
    events: list[archive._GnuProvisionalEvent] = []
    with pytest.raises(archive.ArchiveError, match="member-parent"):
        _consume(tmp_path, _root() + _file(b"./missing/f") + bytes(1024), events.append)
    assert isinstance(events[-1], archive._GnuProvisionalEnd)

"""Provisional callbacks share the inspector's complete validation boundary."""

from __future__ import annotations

import gzip
import io
import os
import tarfile
from pathlib import Path

import pytest

from strixlab import rocm_archive as archive


def _wire(payload: bytes = b"data", name: str = "file") -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as writer:
        entry = tarfile.TarInfo(name)
        entry.size = len(payload)
        entry.mode = 0o644
        writer.addfile(entry, io.BytesIO(payload))
    return gzip.compress(output.getvalue(), compresslevel=0, mtime=0)


def test_consumer_gets_bounded_ordered_provisional_events_and_identical_manifest(
    tmp_path: Path,
) -> None:
    payload = bytes(range(256)) * 700
    (tmp_path / "input.gz").write_bytes(_wire(payload))
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    events: list[archive._ProvisionalEvent] = []
    try:
        expected = archive.inspect_archive(directory, "input.gz")
        observed = archive._consume_archive(directory, "input.gz", events.append)
        os.fstat(directory)
    finally:
        os.close(directory)
    assert observed.canonical_bytes() == expected.canonical_bytes()
    assert isinstance(events[0], archive._ProvisionalStart)
    assert isinstance(events[-1], archive._ProvisionalEnd)
    assert all(event.validation == "provisional" for event in events)
    chunks = events[1:-1]
    assert all(isinstance(event, archive._ProvisionalChunk) for event in chunks)
    assert (
        b"".join(event.data for event in chunks if isinstance(event, archive._ProvisionalChunk))
        == payload
    )
    assert (
        max(len(event.data) for event in chunks if isinstance(event, archive._ProvisionalChunk))
        <= 65536
    )
    assert set(tmp_path.iterdir()) == {tmp_path / "input.gz"}


@pytest.mark.parametrize("failure", ["crc", "topology", "identity"])
def test_late_failure_after_callbacks_never_returns_completed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    path = tmp_path / "input.gz"
    compressed = _wire(name="missing/file" if failure == "topology" else "file")
    if failure == "crc":
        compressed = compressed[:-8] + bytes([compressed[-8] ^ 1]) + compressed[-7:]
    path.write_bytes(compressed)
    # Keep the trailer out of early decoder chunks so callbacks precede failure.
    real_read = os.read
    monkeypatch.setattr(archive.os, "read", lambda fd, count: real_read(fd, min(count, 64)))
    events: list[archive._ProvisionalEvent] = []

    def consume(event: archive._ProvisionalEvent) -> None:
        events.append(event)
        if failure == "identity" and isinstance(event, archive._ProvisionalEnd):
            path.chmod(0o400)

    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(
            archive.ArchiveError, match="gzip-invalid|member-parent|archive-input-changed"
        ):
            archive._consume_archive(directory, "input.gz", consume)
        os.fstat(directory)
    finally:
        os.close(directory)
    assert any(isinstance(event, archive._ProvisionalEnd) for event in events)
    assert all(event.validation == "provisional" for event in events)


@pytest.mark.parametrize("boundary", ["start", "chunk", "end"])
@pytest.mark.parametrize("io_error", [False, True])
def test_callback_exception_aborts_and_closes_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str, io_error: bool
) -> None:
    (tmp_path / "input.gz").write_bytes(_wire())
    directory = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    opened: list[int] = []
    real_open = os.open

    def tracked_open(*args: object, **kwargs: object) -> int:
        descriptor = real_open(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(descriptor)
        return descriptor

    monkeypatch.setattr(archive.os, "open", tracked_open)
    target = {
        "start": archive._ProvisionalStart,
        "chunk": archive._ProvisionalChunk,
        "end": archive._ProvisionalEnd,
    }[boundary]
    failure = OSError("consumer I/O") if io_error else RuntimeError("consumer aborted")

    def consume(event: archive._ProvisionalEvent) -> None:
        if isinstance(event, target):
            raise failure

    try:
        with pytest.raises(archive.ArchiveError if io_error else RuntimeError) as caught:
            archive._consume_archive(directory, "input.gz", consume)
        assert (caught.value.__cause__ if io_error else caught.value) is failure
        os.fstat(directory)
        assert len(opened) == 1
        with pytest.raises(OSError):
            os.fstat(opened[0])
    finally:
        os.close(directory)

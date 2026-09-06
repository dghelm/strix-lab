"""Synthetic GNU archives only; no SDK bytes, installation or execution."""

from __future__ import annotations

import errno
import gzip
import os
import stat

import pytest

from strixlab import rocm_archive as a
from strixlab import rocm_extract as q
from strixlab import rocm_prefix as p
from strixlab.rocm_metadata import InodeMetadataObservationV1, _identity, observe_inode_metadata


def _header(name, kind, mode, size=0, target=b""):
    assert len(name) <= 100 and len(target) <= 100
    block = bytearray(512)
    block[: len(name)] = name
    for offset, width, value in (
        (100, 8, mode),
        (108, 8, 0),
        (116, 8, 0),
        (124, 12, size),
        (136, 12, 0),
    ):
        block[offset : offset + width] = f"{value:0{width - 1}o}".encode() + b"\0"
    block[156:157] = kind
    block[157 : 157 + len(target)] = target
    block[257:265] = b"ustar  \0"
    block[148:156] = b" " * 8
    block[148:156] = f"{sum(block):06o}".encode() + b"\0 "
    return bytes(block)


def wire(entries=None):
    entries = (
        entries
        if entries is not None
        else [
            ("src", b"hello", 0o755),
            ("d", None, 0o755),
            ("d/copy", ("hardlink", "src"), 0o755),
            ("link", ("symlink", "d/copy"), 0o777),
        ]
    )
    parts = [_header(b"./", b"5", 0o755)]
    for path, value, mode in entries:
        name = ("./" + path).encode()
        payload = value if isinstance(value, bytes) else b""
        kind = (
            b"0"
            if isinstance(value, bytes)
            else b"5"
            if value is None
            else b"1"
            if value[0] == "hardlink"
            else b"2"
        )
        target = (
            (("./" if value[0] == "hardlink" else "") + value[1]).encode()
            if isinstance(value, tuple)
            else b""
        )
        if len(name) > 100:
            long_payload = name + b"\0"
            parts.extend(
                (
                    _header(b"././@LongLink", b"L", 0o644, len(long_payload)),
                    long_payload,
                    bytes(-len(long_payload) % 512),
                )
            )
        parts.extend(
            (
                _header(name[:100], kind, mode, len(payload), target),
                payload,
                bytes(-len(payload) % 512),
            )
        )
    parts.append(bytes(1024))
    return gzip.compress(b"".join(parts), compresslevel=0, mtime=0)


def observed_empty(parent_fd, name):
    parent = _identity(os.fstat(parent_fd))
    leaf = _identity(os.stat(name, dir_fd=parent_fd, follow_symlinks=False))
    kind = (
        "directory" if stat.S_ISDIR(leaf.mode) else "symlink" if stat.S_ISLNK(leaf.mode) else "file"
    )
    return InodeMetadataObservationV1(
        name_bytes_escaped=p._escaped(name.encode()),
        kind=kind,
        parent_before=parent,
        parent_after=parent,
        leaf_before=leaf,
        leaf_opened=leaf,
        leaf_after=leaf,
        leaf_named_after=leaf,
        list_status="observed",
        list_errno=None,
        name_list_size_bytes=0,
        names_bytes_escaped=(),
    )


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    (tmp_path / "input.gz").write_bytes(wire())
    (tmp_path / "stage").mkdir(mode=0o700)
    if not p._supported_abi():
        pytest.skip("native Linux x86_64 openat2 required")
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            probe = p._openat2(parent, "stage", os.O_PATH)
        except OSError as exc:
            if exc.errno in {errno.ENOSYS, errno.EINVAL}:
                pytest.skip("native guarded opens unavailable")
            raise
        os.close(probe)
        monkeypatch.setattr(q, "observe_inode_metadata", observed_empty)
        monkeypatch.setattr(p, "observe_inode_metadata", observed_empty)
        yield tmp_path, parent
    finally:
        os.close(parent)


def extract(sandbox):
    _, fd = sandbox
    return q.extract_gnu_quarantine(fd, "input.gz", fd, "stage", "quarantine")


def fdset():
    return set(os.listdir("/proc/self/fd"))


def retained_failure(sandbox, reason):
    base, fd = sandbox
    before = fdset()
    with pytest.raises(q.QuarantineError, match=reason) as caught:
        extract(sandbox)
    root = base / "stage/quarantine"
    assert caught.value.quarantine_name == "quarantine"
    assert caught.value.root_identity == (root.stat().st_dev, root.stat().st_ino)
    assert root.is_dir() and stat.S_IMODE(root.stat().st_mode) == 0o700
    assert not (root / "link").is_symlink()
    if (root / "src").exists():
        assert stat.S_IMODE((root / "src").stat().st_mode) != 0o755
    assert before == fdset()
    os.fstat(fd)
    return root


def test_exact_independent_copies_and_wire_result(sandbox, monkeypatch):
    base, parent = sandbox
    monkeypatch.setattr(os, "link", lambda *args, **kwargs: pytest.fail("filesystem hardlink"))
    before = fdset()
    result = extract(sandbox)
    root = base / "stage/quarantine"
    assert isinstance(result, q.GnuQuarantineResultV1)
    assert result.scope == "structural-quarantine-only"
    assert result.materialization_policy == "independent-hardlink-copies-v1"
    assert result.metadata_coverage == "unknown" and result.link_closure == "not-checked"
    assert (
        result.archive.canonical_bytes()
        == a.inspect_gnu_archive(parent, "input.gz").canonical_bytes()
    )
    assert result.inventory.root.mode == 0o700
    assert (root / "src").read_bytes() == (root / "d/copy").read_bytes() == b"hello"
    assert (root / "src").stat().st_ino != (root / "d/copy").stat().st_ino
    assert (root / "src").stat().st_nlink == (root / "d/copy").stat().st_nlink == 1
    assert stat.S_IMODE((root / "d/copy").stat().st_mode) == 0o755
    assert stat.S_IMODE((root / "d").stat().st_mode) == 0o755
    assert os.readlink(root / "link") == "d/copy"
    source, copy = (
        {e.path: e for e in result.inventory.entries}[name] for name in ("src", "d/copy")
    )
    assert source.sha256 == copy.sha256 and copy.byte_length == 5
    wire_copy = next(e for e in result.archive.entries if e.path == "d/copy")
    assert (
        wire_copy.kind == "hardlink"
        and wire_copy.payload_size_bytes == 0
        and wire_copy.sha256 is None
    )
    assert before == fdset()
    os.fstat(parent)


def test_empty_tree_and_long_unicode_source(sandbox):
    base, _ = sandbox
    long_name = "x" * 97 + "é"
    # Full name crosses the 100-byte GNU short-name boundary.
    (base / "input.gz").write_bytes(wire([(long_name, b"", 0o644)]))
    result = extract(sandbox)
    assert result.archive.longname_control_count == 1
    assert result.inventory.entries[0].byte_length == 0
    (base / "input.gz").write_bytes(wire([]))
    fd = sandbox[1]
    result = q.extract_gnu_quarantine(fd, "input.gz", fd, "stage", "empty")
    assert result.inventory.entries == () and result.archive.root_marker.mode == 0o755
    assert result.inventory.root.mode == 0o700


def test_multiple_copies_zero_size_and_child_before_directory(sandbox):
    base, _ = sandbox
    (base / "input.gz").write_bytes(
        wire(
            [
                ("src", b"", 0o644),
                ("d/z", ("hardlink", "src"), 0o644),
                ("a", ("hardlink", "src"), 0o644),
                ("d", None, 0o755),
            ]
        )
    )
    result = extract(sandbox)
    root = base / "stage/quarantine"
    stats = [(root / name).stat() for name in ("src", "d/z", "a")]
    assert len({(item.st_dev, item.st_ino) for item in stats}) == 3
    assert all(item.st_nlink == 1 and item.st_size == 0 for item in stats)
    assert result.archive.materialized_regular_bytes == 0


def test_strict_default_rejects_gnu_before_writes(sandbox):
    base, fd = sandbox
    with pytest.raises(q.QuarantineError, match="quarantine-archive-failed") as caught:
        q.extract_quarantine(fd, "input.gz", fd, "stage", "quarantine")
    assert caught.value.quarantine_name is None and not list((base / "stage").iterdir())


def test_invalid_hardlink_and_copy_budget_before_mkdir(sandbox, monkeypatch):
    base, _ = sandbox
    monkeypatch.setattr(
        os, "mkdir", lambda *args, **kwargs: pytest.fail("mkdir before full inspection")
    )
    monkeypatch.setattr(a, "_MAX_PAYLOAD_BYTES", 9)
    with pytest.raises(q.QuarantineError, match="quarantine-archive-failed") as caught:
        extract(sandbox)
    assert caught.value.quarantine_name is None
    assert not list((base / "stage").iterdir())
    (base / "input.gz").write_bytes(wire([("h", ("hardlink", "missing"), 0o644)]))
    with pytest.raises(q.QuarantineError, match="quarantine-archive-failed"):
        extract(sandbox)


def test_long_leaf_preflight_before_mkdir(sandbox, monkeypatch):
    base, _ = sandbox
    (base / "input.gz").write_bytes(wire([("x" * 256, b"", 0o644)]))
    monkeypatch.setattr(os, "mkdir", lambda *args, **kwargs: pytest.fail("mkdir before preflight"))
    with pytest.raises(q.QuarantineError, match="quarantine-text-limit"):
        extract(sandbox)


@pytest.mark.parametrize("failure", ["manifest", "header", "entry", "parser", "interrupt"])
def test_second_pass_failure_prevents_all_copies_links_and_modes(sandbox, monkeypatch, failure):
    original = q._consume_gnu_archive

    def consume(fd, name, consumer):
        def event(value):
            if failure == "header" and isinstance(value, a._GnuProvisionalStart):
                from dataclasses import replace

                value = a._GnuProvisionalStart(replace(value.header, mode=0o644))
            if failure == "entry" and isinstance(value, a._GnuProvisionalEnd):
                value = a._GnuProvisionalEnd(value.entry.model_copy(update={"sha256": "0" * 64}))
            consumer(value)

        result = original(fd, name, event)
        if failure == "manifest":
            return result.model_copy(update={"observed_sha256": "0" * 64})
        if failure == "parser":
            raise a.ArchiveError("late-failure")
        if failure == "interrupt":
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(q, "_consume_gnu_archive", consume)
    monkeypatch.setattr(q, "_copy_regular", lambda *args: pytest.fail("copy before wire equality"))
    root = retained_failure(sandbox, "quarantine-(second|archive-failed|interrupted)")
    assert not (root / "d/copy").exists()


@pytest.mark.parametrize("change", ["replace", "symlink", "hardlink", "mode", "size", "content"])
def test_copy_source_identity_and_content_changes(sandbox, monkeypatch, change):
    base, _ = sandbox
    original = q._copy_regular

    def copy(tree, entry):
        source = base / "stage/quarantine/src"
        if change in {"replace", "symlink"}:
            source.rename(source.with_name("old"))
            if change == "replace":
                source.write_bytes(b"hello")
                source.chmod(0o600)
            else:
                source.symlink_to("old")
        elif change == "hardlink":
            os.link(source, source.with_name("alias"))
        elif change == "mode":
            source.chmod(0o400)
        elif change == "size":
            source.write_bytes(b"longer")
        else:
            source.write_bytes(b"wrong")
        original(tree, entry)

    monkeypatch.setattr(q, "_copy_regular", copy)
    retained_failure(
        sandbox, "quarantine-(inode-mismatch|copy-source-mismatch|copy-content-mismatch)"
    )


@pytest.mark.parametrize(
    "phase", ["source-before", "source-after", "target-before", "target-after"]
)
@pytest.mark.parametrize("status", ["names", "unsupported", "denied"])
def test_copy_metadata_failure(sandbox, monkeypatch, phase, status):
    original = q._copy_regular
    copying = False
    counts = {"src": 0, "copy": 0}

    def copy(tree, entry):
        nonlocal copying
        copying = True
        original(tree, entry)

    def observe(parent, name):
        result = observed_empty(parent, name)
        if copying and name in counts:
            counts[name] += 1
            bad = (
                (phase == "source-before" and name == "src" and counts[name] == 1)
                or (phase == "source-after" and name == "src" and counts[name] == 2)
                or (phase == "target-before" and name == "copy" and counts[name] == 1)
                or (phase == "target-after" and name == "copy" and counts[name] == 3)
            )
            if bad:
                if status == "names":
                    return result.model_copy(
                        update={
                            "name_list_size_bytes": 7,
                            "names_bytes_escaped": ("\\x75\\x73\\x65\\x72\\x2e\\x61",),
                        }
                    )
                return result.model_copy(
                    update={
                        "list_status": status,
                        "list_errno": errno.EPERM,
                        "names_bytes_escaped": None,
                        "name_list_size_bytes": None,
                    }
                )
        return result

    monkeypatch.setattr(q, "_copy_regular", copy)
    monkeypatch.setattr(q, "observe_inode_metadata", observe)
    retained_failure(sandbox, "quarantine-metadata-not-observed-empty")


@pytest.mark.parametrize("name,flags_kind", [("src", "path"), ("src", "read"), ("d", "directory")])
def test_copy_mount_guard_failures(sandbox, monkeypatch, name, flags_kind):
    original_copy, original_open = q._copy_regular, q._openat2
    copying = False

    def copy(tree, entry):
        nonlocal copying
        copying = True
        original_copy(tree, entry)

    def openat(parent, leaf, flags):
        match = (
            (flags_kind == "path" and flags & os.O_PATH)
            or (flags_kind == "read" and not flags & (os.O_PATH | os.O_DIRECTORY))
            or (flags_kind == "directory" and flags & os.O_DIRECTORY)
        )
        if copying and leaf == name and match:
            raise OSError(errno.EXDEV, "synthetic mount boundary")
        return original_open(parent, leaf, flags)

    monkeypatch.setattr(q, "_copy_regular", copy)
    monkeypatch.setattr(q, "_openat2", openat)
    retained_failure(sandbox, "quarantine-io")


def test_source_swap_between_path_and_read_open(sandbox, monkeypatch):
    base, _ = sandbox
    original_copy, original_open = q._copy_regular, q._openat2
    copying = False
    swapped = False

    def copy(tree, entry):
        nonlocal copying
        copying = True
        original_copy(tree, entry)

    def openat(parent, name, flags):
        nonlocal swapped
        if copying and name == "src" and not flags & os.O_PATH and not swapped:
            swapped = True
            source = base / "stage/quarantine/src"
            source.rename(source.with_name("old"))
            source.write_bytes(b"hello")
            source.chmod(0o600)
        return original_open(parent, name, flags)

    monkeypatch.setattr(q, "_copy_regular", copy)
    monkeypatch.setattr(q, "_openat2", openat)
    retained_failure(sandbox, "quarantine-copy-source-drift")


@pytest.mark.parametrize("failure", ["short", "zero", "oversize", "enospc", "interrupt"])
def test_copy_checked_writes_and_cleanup(sandbox, monkeypatch, failure):
    base, _ = sandbox
    payload = bytes(range(256)) * 600
    (base / "input.gz").write_bytes(
        wire(
            [
                ("src", payload, 0o755),
                ("copy", ("hardlink", "src"), 0o755),
                ("link", ("symlink", "copy"), 0o777),
            ]
        )
    )
    original_copy, original_write = q._copy_regular, os.write
    copying = False
    sizes = []

    def copy(tree, entry):
        nonlocal copying
        copying = True
        original_copy(tree, entry)

    def write(fd, data):
        if copying:
            sizes.append(len(data))
            if failure == "zero":
                return 0
            if failure == "oversize":
                return len(data) + 1
            if failure == "enospc":
                raise OSError(errno.ENOSPC, "synthetic disk full")
            if failure == "interrupt":
                raise KeyboardInterrupt
            return original_write(fd, data[:137])
        return original_write(fd, data)

    monkeypatch.setattr(q, "_copy_regular", copy)
    monkeypatch.setattr(os, "write", write)
    if failure == "short":
        before = fdset()
        extract(sandbox)
        assert (base / "stage/quarantine/copy").read_bytes() == payload
        assert before == fdset()
    else:
        retained_failure(sandbox, "quarantine-(write-progress|io|interrupted)")
    assert sizes and max(sizes) <= 65536


@pytest.mark.parametrize("failure", ["eof", "extra", "replace-after", "mode-after", "read-error"])
def test_copy_source_read_and_final_binding(sandbox, monkeypatch, failure):
    base, _ = sandbox
    original_copy, original_read = q._copy_regular, os.read
    copying = False
    mutated = False

    def copy(tree, entry):
        nonlocal copying
        copying = True
        original_copy(tree, entry)

    def read(fd, size):
        nonlocal mutated
        if copying:
            if failure == "eof" and size > 1:
                return b""
            if failure == "extra" and size == 1:
                return b"x"
            if failure == "read-error":
                raise OSError(errno.EIO, "synthetic read failure")
            data = original_read(fd, size)
            if not mutated and size > 1 and failure in {"replace-after", "mode-after"}:
                mutated = True
                source = base / "stage/quarantine/src"
                if failure == "replace-after":
                    source.rename(source.with_name("old"))
                    source.write_bytes(b"hello")
                    source.chmod(0o600)
                else:
                    source.chmod(0o400)
            return data
        return original_read(fd, size)

    monkeypatch.setattr(q, "_copy_regular", copy)
    monkeypatch.setattr(os, "read", read)
    retained_failure(
        sandbox,
        "quarantine-(copy-size-mismatch|copy-content-mismatch|copy-source-drift|metadata-not-observed-empty|io)",
    )


def test_copy_existing_destination_not_adopted(sandbox, monkeypatch):
    base, _ = sandbox
    original = q._copy_regular

    def copy(tree, entry):
        target = base / "stage/quarantine/d/copy"
        target.write_bytes(b"sentinel")
        original(tree, entry)

    monkeypatch.setattr(q, "_copy_regular", copy)
    root = retained_failure(sandbox, "quarantine-io")
    assert (root / "d/copy").read_bytes() == b"sentinel"


@pytest.mark.parametrize("change", ["content", "mode", "link", "nlink", "extra", "parent"])
def test_final_projection_catches_copy_or_destination_drift(sandbox, monkeypatch, change):
    base, _ = sandbox
    original = q.inspect_prefix

    def inspect(parent, name):
        target = base / "stage/quarantine/d/copy"
        if change == "content":
            target.write_bytes(b"wrong")
        elif change == "mode":
            target.chmod(0o644)
        elif change == "link":
            target.unlink()
            target.symlink_to("../src")
        elif change == "nlink":
            os.link(target, target.with_name("alias"))
        elif change == "extra":
            target.with_name("extra").write_bytes(b"")
        elif change == "parent":
            (base / "stage").chmod(0o755)
        return original(parent, name)

    monkeypatch.setattr(q, "inspect_prefix", inspect)
    before = fdset()
    with pytest.raises(
        q.QuarantineError, match="quarantine-(final|observation-failed|inode-mismatch|parent-drift)"
    ) as caught:
        extract(sandbox)
    assert caught.value.quarantine_name == "quarantine"
    assert before == fdset()


def test_native_metadata_copy_smoke(sandbox, monkeypatch):
    monkeypatch.setattr(q, "observe_inode_metadata", observe_inode_metadata)
    monkeypatch.setattr(p, "observe_inode_metadata", observe_inode_metadata)
    result = extract(sandbox)
    assert result.metadata_coverage == result.inventory.metadata_coverage == "unknown"


def test_native_copy_source_xattr_rejected(sandbox, monkeypatch):
    base, _ = sandbox
    monkeypatch.setattr(q, "observe_inode_metadata", observe_inode_metadata)
    monkeypatch.setattr(p, "observe_inode_metadata", observe_inode_metadata)
    original = q._copy_regular

    def copy(tree, entry):
        os.setxattr(base / "stage/quarantine/src", b"user.test", b"present")
        original(tree, entry)

    monkeypatch.setattr(q, "_copy_regular", copy)
    retained_failure(sandbox, "quarantine-metadata-not-observed-empty")


@pytest.mark.parametrize("failure", ["replace", "truncate", "fchmod", "owner-mask"])
def test_copy_destination_failures_retain_and_close(sandbox, monkeypatch, failure):
    base, _ = sandbox
    original_copy, original_write, original_chmod = q._copy_regular, os.write, os.fchmod
    copying = False
    changed = False

    def copy(tree, entry):
        nonlocal copying
        copying = True
        if failure == "owner-mask":
            old_umask = os.umask(0o600)
            try:
                original_copy(tree, entry)
            finally:
                os.umask(old_umask)
        else:
            original_copy(tree, entry)

    def write(fd, data):
        nonlocal changed
        result = original_write(fd, data)
        if copying and not changed:
            changed = True
            target = base / "stage/quarantine/d/copy"
            if failure == "replace":
                target.rename(target.with_name("old-copy"))
                target.write_bytes(b"hello")
                target.chmod(0o600)
            elif failure == "truncate":
                os.ftruncate(fd, 1)
        return result

    def chmod(fd, mode):
        if copying and failure == "fchmod":
            raise OSError(errno.EPERM, "synthetic chmod failure")
        return original_chmod(fd, mode)

    monkeypatch.setattr(q, "_copy_regular", copy)
    monkeypatch.setattr(os, "write", write)
    monkeypatch.setattr(os, "fchmod", chmod)
    retained_failure(
        sandbox,
        "quarantine-(metadata-not-observed-empty|copy-size-mismatch|io|creation-owner-permissions)",
    )


def test_copy_two_deep_directory_chains_keep_fds_bounded(sandbox, monkeypatch):
    base, _ = sandbox
    left = ["/".join(["left"] + ["d"] * n) for n in range(25)]
    right = ["/".join(["right"] + ["d"] * n) for n in range(25)]
    source, target = left[-1] + "/src", right[-1] + "/copy"
    # Keep the hardlink wire target below 100 bytes, while exercising two chains.
    (base / "input.gz").write_bytes(
        wire(
            [(path, None, 0o755) for path in left + right]
            + [(source, b"hello", 0o644), (target, ("hardlink", source), 0o644)]
        )
    )
    original_copy, original_open = q._copy_regular, q._openat2
    copying = False
    baseline = peak = 0

    def copy(tree, entry):
        nonlocal copying, baseline, peak
        baseline = peak = len(fdset())
        copying = True
        try:
            original_copy(tree, entry)
        finally:
            copying = False

    def openat(parent, name, flags):
        nonlocal peak
        fd = original_open(parent, name, flags)
        if copying:
            peak = max(peak, len(fdset()))
        return fd

    monkeypatch.setattr(q, "_copy_regular", copy)
    monkeypatch.setattr(q, "_openat2", openat)
    before = fdset()
    result = extract(sandbox)
    assert result.inventory.member_count == 52
    assert (base / "stage/quarantine" / target).read_bytes() == b"hello"
    assert 0 < peak - baseline <= 8
    assert before == fdset()


def test_copy_source_short_reads_are_valid(sandbox, monkeypatch):
    original_copy, original_read = q._copy_regular, os.read
    copying = False

    def copy(tree, entry):
        nonlocal copying
        copying = True
        try:
            original_copy(tree, entry)
        finally:
            copying = False

    def read(fd, size):
        return original_read(fd, min(size, 2) if copying else size)

    monkeypatch.setattr(q, "_copy_regular", copy)
    monkeypatch.setattr(os, "read", read)
    extract(sandbox)
    assert (sandbox[0] / "stage/quarantine/d/copy").read_bytes() == b"hello"

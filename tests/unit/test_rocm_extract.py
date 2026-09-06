"""Only generated synthetic archives and temporary operator-owned directories."""

from __future__ import annotations

import errno
import gzip
import io
import os
import stat
import tarfile

import pytest

from strixlab import rocm_archive as a
from strixlab import rocm_extract as q
from strixlab import rocm_prefix as p
from strixlab.rocm_metadata import InodeMetadataObservationV1, _identity


def wire(entries=None):
    entries = (
        entries
        if entries is not None
        else [("d", None, 0o555), ("d/file", b"hello", 0o444), ("link", ("link", "d/file"), 0o777)]
    )
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as writer:
        for path, value, mode in entries:
            entry = tarfile.TarInfo(path)
            entry.mode = mode
            if value is None:
                entry.type = tarfile.DIRTYPE
            elif isinstance(value, tuple):
                entry.type = tarfile.SYMTYPE
                entry.linkname = value[1]
            else:
                entry.size = len(value)
            writer.addfile(entry, io.BytesIO(value) if isinstance(value, bytes) else None)
    return gzip.compress(output.getvalue(), compresslevel=0, mtime=0)


def observed_empty(parent_fd, name):
    """Synthetic metadata seam; production never replaces failed/unknown probes."""
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
        pytest.skip("Linux x86_64 openat2 required for native synthetic extraction")
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
    return q.extract_quarantine(fd, "input.gz", fd, "stage", "quarantine")


def fdset():
    return set(os.listdir("/proc/self/fd"))


def test_valid_quarantine_exact_manifest_readonly_modes_and_unknown(sandbox):
    base, parent = sandbox
    before = fdset()
    result = extract(sandbox)
    root = base / "stage/quarantine"
    assert result.scope == "structural-quarantine-only"
    assert result.metadata_coverage == "unknown" and result.link_closure == "not-checked"
    assert result.inventory.root.mode == 0o700
    assert (
        result.archive.canonical_bytes() == a.inspect_archive(parent, "input.gz").canonical_bytes()
    )
    assert (root / "d/file").read_bytes() == b"hello"
    assert stat.S_IMODE((root / "d/file").stat().st_mode) == 0o444
    assert stat.S_IMODE((root / "d").stat().st_mode) == 0o555
    entries = {v.path: v for v in result.inventory.entries}
    assert set(entries) == {"d", "d/file", "link"}
    assert entries["d"].byte_length is None
    assert entries["link"].byte_length == len(b"d/file")
    assert entries["link"].link_target == "d/file" and entries["link"].nlink == 1
    assert all((v.uid, v.gid) == (os.geteuid(), os.getegid()) for v in entries.values())
    assert fdset() == before
    os.fstat(parent)


def test_child_before_parent_unicode_link_length_and_empty_file(sandbox):
    base, _ = sandbox
    (base / "input.gz").write_bytes(
        wire([("dir/é", b"", 0o400), ("link", ("link", "dir/é"), 0o777), ("dir", None, 0o500)])
    )
    result = extract(sandbox)
    assert result.inventory.entries[1].byte_length == 0
    assert result.inventory.entries[2].byte_length == len("dir/é".encode())


@pytest.mark.parametrize(
    "entries", [[("f", b"x", 0o200)], [("d", None, 0o400)], [("d", None, 0o100)]]
)
def test_permission_preflight_no_writes(sandbox, entries, monkeypatch):
    base, _ = sandbox
    (base / "input.gz").write_bytes(wire(entries))
    monkeypatch.setattr(q.os, "mkdir", lambda *a, **kw: pytest.fail("mkdir before preflight"))
    with pytest.raises(q.QuarantineError, match="final-owner-permissions") as caught:
        extract(sandbox)
    assert caught.value.quarantine_name is None and caught.value.root_identity is None
    assert not list((base / "stage").iterdir())


@pytest.mark.parametrize("limit,value", [("_MAX_PATH_BYTES", 2), ("_MAX_DEPTH", 1)])
def test_path_depth_preflight(sandbox, monkeypatch, limit, value):
    base, _ = sandbox
    monkeypatch.setattr(q, limit, value)
    with pytest.raises(q.QuarantineError) as caught:
        extract(sandbox)
    assert caught.value.quarantine_name is None and not list((base / "stage").iterdir())


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "\udcff", "x" * 256, "a\0b"])
def test_leaf_preflight(name):
    with pytest.raises(q.QuarantineError) as caught:
        q.extract_quarantine(-1, name, -1, "stage", "q")
    assert caught.value.quarantine_name is None


def test_full_initial_parse_before_writes(sandbox, monkeypatch):
    base, _ = sandbox
    data = wire()
    (base / "input.gz").write_bytes(data[:-8] + bytes([data[-8] ^ 1]) + data[-7:])
    monkeypatch.setattr(q.os, "mkdir", lambda *a, **kw: pytest.fail("write before full parse"))
    with pytest.raises(q.QuarantineError, match="archive-failed") as caught:
        extract(sandbox)
    assert caught.value.quarantine_name is None


def test_existing_quarantine_never_adopted(sandbox):
    base, _ = sandbox
    root = base / "stage/quarantine"
    root.mkdir()
    (root / "sentinel").write_bytes(b"untouched")
    with pytest.raises(q.QuarantineError) as caught:
        extract(sandbox)
    assert caught.value.quarantine_name is None and caught.value.root_identity is None
    assert list(root.iterdir()) == [root / "sentinel"]
    assert (root / "sentinel").read_bytes() == b"untouched"


@pytest.mark.parametrize("mode", [0o755, 0o500, 0o1700])
def test_destination_parent_exact_mode(sandbox, mode):
    base, _ = sandbox
    (base / "stage").chmod(mode)
    with pytest.raises(q.QuarantineError, match="parent-mode") as caught:
        extract(sandbox)
    assert caught.value.quarantine_name is None


@pytest.mark.parametrize("field", ["uid", "gid"])
def test_destination_parent_exact_owner_even_when_privileged(sandbox, monkeypatch, field):
    function = "geteuid" if field == "uid" else "getegid"
    real = getattr(os, function)()
    monkeypatch.setattr(q.os, function, lambda: real + 1)
    with pytest.raises(q.QuarantineError, match="inode-mismatch") as caught:
        extract(sandbox)
    assert caught.value.quarantine_name is None


@pytest.mark.parametrize(
    "updates",
    [
        {"list_status": "error", "list_errno": errno.EPERM, "names_bytes_escaped": None},
        {"list_status": "unsupported", "list_errno": errno.ENOSYS, "names_bytes_escaped": None},
        {"list_status": "malformed", "names_bytes_escaped": None},
        {"list_status": "resource-limit", "names_bytes_escaped": None},
        {"names_bytes_escaped": ("\\x61",), "name_list_size_bytes": 2},
        {"list_errno": errno.EPERM},
    ],
)
@pytest.mark.parametrize("where", ["stage", "file", "final"])
def test_metadata_failures_never_complete(sandbox, monkeypatch, updates, where):
    base, _ = sandbox

    def observe(fd, name):
        value = observed_empty(fd, name)
        if name == ("file" if where == "final" else where):
            value = value.model_copy(update=updates)
        return value

    monkeypatch.setattr(p if where == "final" else q, "observe_inode_metadata", observe)
    before = fdset()
    with pytest.raises(q.QuarantineError) as caught:
        extract(sandbox)
    assert (caught.value.quarantine_name is not None) == (where != "stage")
    assert (base / "stage/quarantine").exists() == (where != "stage")
    assert fdset() == before


@pytest.mark.parametrize("fail_read", [False, True])
def test_post_mkdir_root_capture_or_read_failure_retains_context(sandbox, monkeypatch, fail_read):
    base, _ = sandbox
    original = q._openat2

    def opened(parent, name, flags):
        if name == "quarantine" and bool(flags & os.O_DIRECTORY) == fail_read:
            raise OSError(errno.EXDEV, "synthetic mount rejection")
        return original(parent, name, flags)

    monkeypatch.setattr(q, "_openat2", opened)
    before = fdset()
    with pytest.raises(q.QuarantineError) as caught:
        extract(sandbox)
    assert caught.value.quarantine_name == "quarantine"
    assert (caught.value.root_identity is not None) == fail_read
    assert (base / "stage/quarantine").is_dir()
    if fail_read:
        metadata = (base / "stage/quarantine").stat()
        assert caught.value.root_identity == (metadata.st_dev, metadata.st_ino)
    assert fdset() == before


@pytest.mark.parametrize("mask", [0o100, 0o200, 0o400, 0o777])
def test_restrictive_umask_owner_loss_fails_even_with_privileged_access(sandbox, mask):
    previous = os.umask(mask)
    try:
        with pytest.raises(q.QuarantineError, match="creation-owner-permissions") as caught:
            extract(sandbox)
    finally:
        os.umask(previous)
    try:
        assert (
            caught.value.quarantine_name == "quarantine" and caught.value.root_identity is not None
        )
    finally:
        # Test teardown only: leave the failed leaf in place, but let pytest remove its tmpdir.
        (sandbox[0] / "stage/quarantine").chmod(0o700)


def test_group_other_umask_does_not_change_final_modes(sandbox):
    previous = os.umask(0o077)
    try:
        result = extract(sandbox)
    finally:
        os.umask(previous)
    assert result.inventory.entries[0].mode == 0o555
    assert result.inventory.entries[1].mode == 0o444


def test_short_writes_make_bounded_progress(sandbox, monkeypatch):
    base, _ = sandbox
    payload = bytes(range(256)) * 600
    (base / "input.gz").write_bytes(wire([("file", payload, 0o644)]))
    original = os.write
    sizes = []

    def write(fd, data):
        sizes.append(len(data))
        return original(fd, data[:101])

    monkeypatch.setattr(q.os, "write", write)
    extract(sandbox)
    assert (base / "stage/quarantine/file").read_bytes() == payload
    assert max(sizes) <= 65536 and len(sizes) > 3


@pytest.mark.parametrize("failure", ["zero", "enospc", "fchmod", "observer"])
def test_active_file_failure_closes_all_owned_fds(sandbox, monkeypatch, failure):
    base, parent = sandbox
    before = fdset()
    if failure == "zero":
        monkeypatch.setattr(q.os, "write", lambda *args: 0)
    elif failure == "enospc":

        def write(*args):
            raise OSError(errno.ENOSPC, "synthetic no space")

        monkeypatch.setattr(q.os, "write", write)
    elif failure == "fchmod":
        original = os.fchmod

        def fchmod(fd, mode):
            if stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError(errno.EPERM, "synthetic chmod failure")
            original(fd, mode)

        monkeypatch.setattr(q.os, "fchmod", fchmod)
    else:

        def observe(fd, name):
            if name == "file":
                raise RuntimeError("synthetic unexpected observer failure")
            return observed_empty(fd, name)

        monkeypatch.setattr(q, "observe_inode_metadata", observe)
    with pytest.raises(q.QuarantineError) as caught:
        extract(sandbox)
    assert caught.value.quarantine_name == "quarantine" and caught.value.root_identity is not None
    assert not (base / "stage/quarantine/link").is_symlink()
    assert fdset() == before
    os.fstat(parent)


@pytest.mark.parametrize("failure", ["late-parser", "archive-digest", "entry", "header"])
def test_second_pass_mismatch_prevents_symlinks_and_final_modes(sandbox, monkeypatch, failure):
    base, _ = sandbox
    original = q._consume_archive

    def consume(fd, name, callback):
        def events(event):
            if (
                failure == "entry"
                and isinstance(event, a._ProvisionalEnd)
                and event.entry.kind == "file"
            ):
                event = a._ProvisionalEnd(event.entry.model_copy(update={"sha256": "0" * 64}))
            if (
                failure == "header"
                and isinstance(event, a._ProvisionalStart)
                and event.header.kind == "file"
            ):
                from dataclasses import replace

                event = a._ProvisionalStart(replace(event.header, mode=0o600))
            callback(event)
            if (
                failure == "late-parser"
                and isinstance(event, a._ProvisionalEnd)
                and event.entry.kind == "file"
            ):
                raise a.ArchiveError("synthetic-late-parser")

        result = original(fd, name, events)
        return (
            result.model_copy(update={"observed_sha256": "0" * 64})
            if failure == "archive-digest"
            else result
        )

    monkeypatch.setattr(q, "_consume_archive", consume)
    before = fdset()
    with pytest.raises(q.QuarantineError) as caught:
        extract(sandbox)
    assert caught.value.quarantine_name == "quarantine"
    root = base / "stage/quarantine"
    assert not (root / "link").is_symlink()
    assert stat.S_IMODE((root / "d").stat().st_mode) == 0o700
    if failure != "header":
        assert stat.S_IMODE((root / "d/file").stat().st_mode) == 0o600
    assert fdset() == before


@pytest.mark.parametrize("boundary", ["stage", "quarantine", "d", "file"])
def test_guarded_mount_boundary_failure(sandbox, monkeypatch, boundary):
    original = q._openat2

    def opened(fd, name, flags):
        if name == boundary:
            raise OSError(errno.EXDEV, "synthetic same-device mount")
        return original(fd, name, flags)

    monkeypatch.setattr(q, "_openat2", opened)
    with pytest.raises(q.QuarantineError) as caught:
        extract(sandbox)
    assert (caught.value.quarantine_name is not None) == (boundary != "stage")


@pytest.mark.parametrize("path", ["d", "d/file", ""])
def test_replaced_created_inodes_fail(sandbox, monkeypatch, path):
    base, _ = sandbox
    original = q._consume_archive

    def consume(fd, name, callback):
        result = original(fd, name, callback)
        root = base / "stage/quarantine"
        target = root / path if path else root
        target.rename(target.with_name(target.name + ".old"))
        if path == "d/file":
            target.write_bytes(b"hello")
        else:
            target.mkdir(mode=0o700)
        return result

    monkeypatch.setattr(q, "_consume_archive", consume)
    with pytest.raises(q.QuarantineError, match="inode-mismatch"):
        extract(sandbox)


@pytest.mark.parametrize("change", ["extra", "payload", "mode", "link", "nlink", "parent"])
def test_final_disk_projection_and_destination_recheck(sandbox, monkeypatch, change):
    base, _ = sandbox
    original = q.inspect_prefix

    def inspect(parent, name):
        root = base / "stage/quarantine"
        if change == "extra":
            (root / "extra").touch()
        elif change == "payload":
            (root / "d/file").chmod(0o644)
            (root / "d/file").write_bytes(b"wrong")
            (root / "d/file").chmod(0o444)
        elif change == "mode":
            (root / "d/file").chmod(0o644)
        elif change == "link":
            (root / "link").unlink()
            (root / "link").symlink_to("other")
        elif change == "nlink":
            os.link(root / "d/file", base / "external")
        elif change == "parent":
            (base / "stage").chmod(0o755)
        return original(parent, name)

    monkeypatch.setattr(q, "inspect_prefix", inspect)
    with pytest.raises(q.QuarantineError) as caught:
        extract(sandbox)
    assert caught.value.quarantine_name == "quarantine"


def test_empty_archive_creates_only_root(sandbox):
    base, _ = sandbox
    (base / "input.gz").write_bytes(wire([]))
    result = extract(sandbox)
    assert result.inventory.entries == ()
    assert result.inventory.root.mode == 0o700


def test_file_creation_owner_permission_loss_rejected(sandbox, monkeypatch):
    original = os.open

    def opened(path, flags, *args, **kwargs):
        fd = original(path, flags, *args, **kwargs)
        if flags & os.O_CREAT:
            os.fchmod(fd, 0o400)
        return fd

    monkeypatch.setattr(q.os, "open", opened)
    before = fdset()
    with pytest.raises(q.QuarantineError, match="creation-owner-permissions"):
        extract(sandbox)
    assert fdset() == before


@pytest.mark.parametrize("field", ["leaf_opened", "parent_after"])
def test_metadata_identity_brackets_must_match_held_inode(sandbox, monkeypatch, field):
    def observe(fd, name):
        value = observed_empty(fd, name)
        if name == "file":
            identity = getattr(value, field).model_copy(update={"ino": 0})
            value = value.model_copy(update={field: identity})
        return value

    monkeypatch.setattr(q, "observe_inode_metadata", observe)
    monkeypatch.setattr(q.os, "write", lambda *args: pytest.fail("write before identity check"))
    before = fdset()
    with pytest.raises(q.QuarantineError):
        extract(sandbox)
    assert fdset() == before


def test_new_directory_guarded_read_identity_mismatch(sandbox, monkeypatch):
    base, _ = sandbox
    (base / "other").mkdir(mode=0o700)
    other = os.open(base / "other", os.O_RDONLY | os.O_DIRECTORY)
    original = q._openat2

    def opened(parent, name, flags):
        if name == "quarantine" and flags & os.O_DIRECTORY:
            return os.dup(other)
        return original(parent, name, flags)

    monkeypatch.setattr(q, "_openat2", opened)
    before = fdset()
    try:
        with pytest.raises(q.QuarantineError, match="binding-drift") as caught:
            extract(sandbox)
        assert caught.value.root_identity is not None
        assert fdset() == before
    finally:
        os.close(other)


def test_root_final_binding_rechecked_after_inventory(sandbox, monkeypatch):
    base, _ = sandbox
    original = q.inspect_prefix

    def inspect(fd, name):
        result = original(fd, name)
        (base / "stage/quarantine").chmod(0o755)
        return result

    monkeypatch.setattr(q, "inspect_prefix", inspect)
    with pytest.raises(q.QuarantineError, match="binding-drift"):
        extract(sandbox)


def test_deep_tree_reopens_only_one_bounded_chain(sandbox, monkeypatch):
    base, _ = sandbox
    entries = [("/".join(["d"] * depth), None, 0o500) for depth in range(1, 41)]
    entries.insert(0, ("/".join(["d"] * 40) + "/f", b"x", 0o400))
    (base / "input.gz").write_bytes(wire(entries))
    original = q._openat2
    initial = len(fdset())
    peak = initial

    def opened(parent, name, flags):
        nonlocal peak
        fd = original(parent, name, flags)
        peak = max(peak, len(fdset()))
        return fd

    monkeypatch.setattr(q, "_openat2", opened)
    result = extract(sandbox)
    assert result.inventory.member_count == 41
    assert peak - initial <= 12
    assert len(fdset()) == initial


def test_native_metadata_observer_smoke_when_supported(sandbox, monkeypatch):
    from strixlab import rocm_metadata as metadata

    _, fd = sandbox
    probe = metadata.observe_inode_metadata(fd, "stage")
    if probe.list_status != "observed":
        pytest.skip(
            "native listxattrat probe unavailable; synthetic fail-closed cases run separately"
        )
    monkeypatch.setattr(q, "observe_inode_metadata", metadata.observe_inode_metadata)
    monkeypatch.setattr(p, "observe_inode_metadata", metadata.observe_inode_metadata)
    assert extract(sandbox).metadata_coverage == "unknown"


def test_actual_inherited_metadata_never_stripped(sandbox, monkeypatch):
    from strixlab import rocm_metadata as metadata

    base, fd = sandbox
    probe = metadata.observe_inode_metadata(fd, "stage")
    if probe.list_status != "observed":
        pytest.skip("native listxattrat probe unavailable")
    try:
        os.setxattr(base / "stage", "user.synthetic", b"keep")
    except OSError as exc:
        if exc.errno in {errno.ENOTSUP, errno.EPERM}:
            pytest.skip("temporary filesystem xattrs unavailable")
        raise
    monkeypatch.setattr(q, "observe_inode_metadata", metadata.observe_inode_metadata)
    with pytest.raises(q.QuarantineError, match="metadata-not-observed-empty") as caught:
        extract(sandbox)
    assert caught.value.quarantine_name is None
    assert os.getxattr(base / "stage", "user.synthetic") == b"keep"


@pytest.mark.parametrize("after_mkdir", [False, True])
def test_cancellation_reports_retained_context_and_closes_fds(sandbox, monkeypatch, after_mkdir):
    base, parent = sandbox
    original = q._consume_archive
    if after_mkdir:

        def consume(fd, name, callback):
            def events(event):
                callback(event)
                if isinstance(event, a._ProvisionalChunk):
                    raise KeyboardInterrupt

            return original(fd, name, events)

        monkeypatch.setattr(q, "_consume_archive", consume)
    else:

        def inspect(fd, name):
            raise KeyboardInterrupt

        monkeypatch.setattr(q, "inspect_archive", inspect)
    before = fdset()
    with pytest.raises(q.QuarantineError, match="quarantine-interrupted") as caught:
        extract(sandbox)
    assert isinstance(caught.value.__cause__, KeyboardInterrupt)
    assert (caught.value.quarantine_name is not None) == after_mkdir
    assert (caught.value.root_identity is not None) == after_mkdir
    assert (base / "stage/quarantine").exists() == after_mkdir
    assert fdset() == before
    os.fstat(parent)

"""Synthetic observations exercise bounds and semantics without an SDK payload."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import stat

import pytest

from strixlab import rocm_prefix as p
from strixlab.rocm_metadata import InodeIdentityV1, InodeMetadataObservationV1


def inventory(spec=None, *, offset=0, root_name="prefix"):
    spec = spec or {}
    records = {}
    outside = InodeIdentityV1(
        dev=1,
        ino=1 + offset,
        mode=stat.S_IFDIR | 0o755,
        uid=1000,
        gid=1000,
        nlink=2,
        size=0,
        mtime_ns=offset,
        ctime_ns=offset,
    )
    for index, (path, value) in enumerate([("", None), *sorted(spec.items())]):
        kind = "directory" if value is None else "file" if isinstance(value, bytes) else "symlink"
        target = value[1] if kind == "symlink" else None
        size = len(value) if kind == "file" else len(target.encode()) if target else 0
        mode = {"directory": stat.S_IFDIR, "file": stat.S_IFREG, "symlink": stat.S_IFLNK}[
            kind
        ] | 0o755
        identity = outside.model_copy(
            update={
                "ino": index + 10 + offset,
                "mode": mode,
                "size": size,
                "nlink": 2 if kind == "directory" else 1,
            }
        )
        parent = records[path.rpartition("/")[0]].identity if path else outside
        metadata = InodeMetadataObservationV1(
            name_bytes_escaped=p._escaped(
                (path.rsplit("/", 1)[-1] if path else root_name).encode()
            ),
            kind=kind,
            parent_before=parent,
            parent_after=parent,
            leaf_before=identity,
            leaf_opened=identity,
            leaf_after=identity,
            leaf_named_after=identity,
            list_status="observed",
            list_errno=None,
            name_list_size_bytes=0,
            names_bytes_escaped=(),
        )
        records[path] = p.PrefixEntryV1(
            path=path,
            path_bytes_escaped=p._escaped(path.encode()),
            kind=kind,
            mode=0o755,
            uid=1000,
            gid=1000,
            byte_length=None if kind == "directory" else size,
            sha256=hashlib.sha256(value).hexdigest() if kind == "file" else None,
            link_target=target,
            link_target_bytes_escaped=p._escaped(target.encode()) if target else None,
            nlink=None if kind == "directory" else 1,
            identity=identity,
            metadata=metadata,
        )
    return p.PrefixInventoryV1(
        root=records.pop(""),
        entries=tuple(records.values()),
        member_count=len(records),
        regular_payload_bytes=sum(e.byte_length for e in records.values() if e.kind == "file"),
        evidence_bytes_charged=4096,
        peak_owned_fds=1,
    )


def replace_entry(report, index=0, **updates):
    entries = list(report.entries)
    entries[index] = entries[index].model_copy(update=updates)
    return report.model_copy(update={"entries": tuple(entries)})


def test_semantics_ignore_inode_times_and_outer_name(monkeypatch):
    a = inventory({"d": None, "d/f": b"abc", "l": ("link", "d/f")})
    b = inventory({"d": None, "d/f": b"abc", "l": ("link", "d/f")}, offset=100, root_name="other")
    monkeypatch.setattr(
        p.os, "open", lambda *a, **kw: pytest.fail("pure comparison opened filesystem")
    )
    assert p.compare_prefixes(a, b).semantic_equal
    assert a.canonical_bytes() == a.canonical_bytes()
    assert p.resolve_inventory_links(a).links[0].resolved_path == "d/f"
    assert a.metadata_coverage == "unknown" and a.link_closure == "not-checked"


def test_compare_changes_and_bounded_samples(monkeypatch):
    monkeypatch.setattr(p, "_MAX_DIFFERENCES", 2)
    a = inventory({"a": b"a", "b": b"b", "c": b"c"})
    b = inventory({"a": b"x", "b": b"y", "d": b"d"})
    result = p.compare_prefixes(a, b)
    assert not result.semantic_equal
    assert result.differing_path_count == 4
    assert result.differing_paths == ("a", "b") and result.sample_truncated


def test_unknown_error_semantics_are_preserved():
    a = inventory({"f": b"x"})
    metadata = a.entries[0].metadata.model_copy(
        update={
            "list_status": "unsupported",
            "list_errno": errno.ENOSYS,
            "name_list_size_bytes": None,
            "names_bytes_escaped": None,
        }
    )
    b = replace_entry(a, metadata=metadata)
    assert not p.compare_prefixes(a, b).semantic_equal
    assert p.compare_prefixes(b, b).semantic_equal
    assert p.compare_prefixes(b, b).metadata_coverage == "unknown"


@pytest.mark.parametrize(
    ("target", "status", "resolved"),
    [
        (".", "resolved", ""),
        ("dir/..", "resolved", ""),
        ("dir//./", "resolved", "dir"),
        ("f", "resolved", "f"),
        ("f/..", "not-directory", None),
        ("f/.", "not-directory", None),
        ("f/", "not-directory", None),
        ("missing", "dangling", None),
        ("../f", "escape", None),
        ("/f", "absolute", None),
        ("l", "cycle", None),
        ("other", "cycle", None),
    ],
)
def test_link_resolution(target, status, resolved):
    report = inventory({"dir": None, "f": b"x", "l": ("link", target), "other": ("link", "l")})
    link = next(v for v in p.resolve_inventory_links(report).links if v.path == "l")
    assert (link.status, link.resolved_path) == (status, resolved)


def test_links_expand_before_later_parent_components():
    report = inventory(
        {
            "a": ("link", "b/../x"),
            "b": ("link", "dir/sub"),
            "dir": None,
            "dir/sub": None,
            "dir/x": b"right",
            "x": b"wrong",
            "root": ("link", "."),
            "repeat": ("link", "root/root/dir/x"),
        }
    )
    links = {v.path: v for v in p.resolve_inventory_links(report).links}
    assert links["a"].resolved_path == "dir/x"
    assert links["repeat"].resolved_path == "dir/x"


def test_link_expansion_limit():
    report = inventory({**{f"l{i:02}": ("link", f"l{i + 1:02}") for i in range(41)}, "l41": b"x"})
    links = p.resolve_inventory_links(report).links
    assert links[0].status == "expansion-limit" and links[0].expansions == 40
    assert links[1].status == "resolved" and links[1].expansions == 40


@pytest.mark.parametrize(
    ("limit", "value", "reason"),
    [
        ("_MAX_LINK_STEPS", 1, "link-work-limit"),
        ("_MAX_PENDING_BYTES", 1, "link-pending-limit"),
        ("_MAX_EVIDENCE_BYTES", 4096, "evidence-limit"),
    ],
)
def test_link_resource_limits(monkeypatch, limit, value, reason):
    report = inventory({"f": b"x", "l": ("link", "././f")})
    monkeypatch.setattr(p, limit, value)
    with pytest.raises(p.PrefixError, match=reason):
        p.resolve_inventory_links(report)


@pytest.mark.parametrize(
    "updates",
    [
        {"path_bytes_escaped": "bad"},
        {"mode": 0},
        {"uid": 2**129},
        {"sha256": "bad"},
        {"nlink": 2},
        {"byte_length": -1},
        {"link_target": "wrong"},
    ],
)
def test_invalid_entry_rejected(updates):
    report = replace_entry(inventory({"f": b"x"}), **updates)
    with pytest.raises(p.PrefixError):
        p.compare_prefixes(report, report)


@pytest.mark.parametrize(
    "updates",
    [
        {"validation": "partial"},
        {"metadata_coverage": "complete"},
        {"link_closure": "closed"},
        {"schema_version": True},
        {"inventory_id": "wrong"},
        {"resolution_policy": "wrong"},
        {"member_count": 2},
        {"regular_payload_bytes": 4},
        {"peak_owned_fds": 9999},
        {"evidence_bytes_charged": -1},
    ],
)
def test_invalid_inventory_rejected(updates):
    report = inventory({"f": b"x"}).model_copy(update=updates)
    with pytest.raises(p.PrefixError):
        report.canonical_bytes()


def test_map_topology_order_and_parent_identity():
    report = inventory({"a": None, "a/f": b"x", "z": b"y"})
    bads = [
        report.model_copy(update={"entries": tuple(reversed(report.entries))}),
        report.model_copy(update={"entries": report.entries[1:], "member_count": 2}),
        replace_entry(
            report,
            1,
            metadata=report.entries[1].metadata.model_copy(
                update={"parent_before": report.root.identity, "parent_after": report.root.identity}
            ),
        ),
    ]
    for bad in bads:
        with pytest.raises(p.PrefixError):
            p.resolve_inventory_links(bad)


@pytest.mark.parametrize(
    "updates",
    [
        {"coverage": "complete"},
        {"list_status": "wrong"},
        {"name_bytes_escaped": "bad"},
        {"list_errno": 9999},
        {"names_bytes_escaped": ("\\x00",)},
        {"names_bytes_escaped": ("\\x62", "\\x61"), "name_list_size_bytes": 4},
        {"names_bytes_escaped": ("\\x61",), "name_list_size_bytes": 0},
        {"list_status": "unsupported"},
        {"kind": "directory"},
        {"name_list_size_bytes": -1},
    ],
)
def test_invalid_metadata_rejected(updates):
    report = inventory({"f": b"x"})
    report = replace_entry(report, metadata=report.entries[0].metadata.model_copy(update=updates))
    with pytest.raises(p.PrefixError):
        report.canonical_bytes()


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "prefix"
    root.mkdir()
    (root / "d").mkdir()
    (root / "d" / "f").write_bytes(b"abc")
    (root / "l").symlink_to("d/f")
    if not p._supported_abi():
        pytest.skip("native Linux x86_64 openat2 required")
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            probe = p._openat2(fd, "prefix", os.O_PATH)
        except OSError as exc:
            if exc.errno in {errno.ENOSYS, errno.EINVAL}:
                pytest.skip("openat2 unsupported by kernel")
            raise
        os.close(probe)
        yield fd, root
    finally:
        os.close(fd)


def test_native_inventory_and_all_opens_guarded(tree, monkeypatch):
    parent, root = tree
    calls = []
    original = p._openat2

    def opened(fd, name, flags):
        calls.append((name, flags))
        return original(fd, name, flags)

    monkeypatch.setattr(p, "_openat2", opened)
    result = p.inspect_prefix(parent, root.name)
    assert result.member_count == 3 and result.regular_payload_bytes == 3
    assert result.root.path == "" and result.metadata_coverage == "unknown"
    assert result.entries[1].sha256 == hashlib.sha256(b"abc").hexdigest()
    assert p.compare_prefixes(result, result).semantic_equal
    assert len(result.canonical_bytes()) <= result.evidence_bytes_charged
    assert all(flags & os.O_NOFOLLOW and flags & os.O_CLOEXEC for _, flags in calls)
    assert any(name == "f" and flags & os.O_NONBLOCK for name, flags in calls)
    assert sum(name == "f" and flags & os.O_NONBLOCK != 0 for name, flags in calls) == 1
    os.fstat(parent)  # Caller still owns the original FD.


@pytest.mark.parametrize(
    ("limit", "value", "reason"),
    [
        ("_MAX_MEMBERS", 1, "member-limit"),
        ("_MAX_FILE_BYTES", 2, "file-limit"),
        ("_MAX_PAYLOAD_BYTES", 2, "payload-limit"),
        ("_MAX_DEPTH", 1, "depth-limit"),
        ("_MAX_FDS", 3, "fd-limit"),
        ("_MAX_EVIDENCE_BYTES", 5000, "evidence-limit"),
    ],
)
def test_scan_limits_release_owned_fds(tree, monkeypatch, limit, value, reason):
    parent, root = tree
    before = set(os.listdir("/proc/self/fd"))
    monkeypatch.setattr(p, limit, value)
    monkeypatch.setattr(
        p, "_hash_file", lambda *a: pytest.fail("payload read before budget rejection")
    )
    with pytest.raises(p.PrefixError, match=reason):
        p.inspect_prefix(parent, root.name)
    assert set(os.listdir("/proc/self/fd")) == before


def test_hardlink_special_and_root_types(tree):
    parent, root = tree
    os.link(root / "d/f", root / "hard")
    with pytest.raises(p.PrefixError, match="non-directory-links"):
        p.inspect_prefix(parent, root.name)
    (root / "hard").unlink()
    os.mkfifo(root / "fifo")
    with pytest.raises(p.PrefixError, match="entry-type"):
        p.inspect_prefix(parent, root.name)
    fd = os.open(root / "d", os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(p.PrefixError, match="root-type"):
            p.inspect_prefix(fd, "f")
    finally:
        os.close(fd)


@pytest.mark.parametrize("boundary", ["prefix", "d", "f"])
def test_mount_boundary_errors_fail_closed(tree, monkeypatch, boundary):
    parent, root = tree
    original = p._openat2

    def opened(fd, name, flags):
        if name == boundary:
            raise OSError(errno.EXDEV, "synthetic same-device bind mount")
        return original(fd, name, flags)

    monkeypatch.setattr(p, "_openat2", opened)
    with pytest.raises(p.PrefixError, match="observation-failed") as caught:
        p.inspect_prefix(parent, root.name)
    assert caught.value.__cause__.errno == errno.EXDEV


def test_metadata_identity_mismatch_precedes_hash(tree, monkeypatch):
    parent, root = tree
    original = p.observe_inode_metadata

    def observe(fd, name):
        result = original(fd, name)
        if name == "f":
            result = result.model_copy(
                update={"leaf_opened": result.leaf_opened.model_copy(update={"ino": 0})}
            )
        return result

    monkeypatch.setattr(p, "observe_inode_metadata", observe)
    monkeypatch.setattr(p, "_hash_file", lambda *a: pytest.fail("hash after inconsistent metadata"))
    with pytest.raises(p.PrefixError, match="identity-changed"):
        p.inspect_prefix(parent, root.name)


def test_final_rewalk_detects_payload_drift(tree, monkeypatch):
    parent, root = tree
    original = p._walk

    def walk(budget, fd, name, baseline=None):
        if baseline is not None:
            (root / "d/f").write_bytes(b"different")
        return original(budget, fd, name, baseline)

    monkeypatch.setattr(p, "_walk", walk)
    with pytest.raises(p.PrefixError, match="identity-changed"):
        p.inspect_prefix(parent, root.name)


def test_directory_after_set_detects_change(tree, monkeypatch):
    parent, root = tree
    original = p._directory_names
    calls = 0

    def names(budget, fd):
        nonlocal calls
        calls += 1
        result = original(budget, fd)
        return (*result, "injected") if calls == 3 else result

    monkeypatch.setattr(p, "_directory_names", names)
    with pytest.raises(p.PrefixError, match="directory-changed"):
        p.inspect_prefix(parent, root.name)


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\0b", "\udcff", "x" * 256])
def test_invalid_leaf(name):
    with pytest.raises(p.PrefixError):
        p._openat2(0, name, os.O_PATH)


def test_openat2_abi_flags_and_resolution(monkeypatch):
    monkeypatch.setattr(p, "_supported_abi", lambda: False)
    with pytest.raises(p.PrefixError, match="abi-unsupported"):
        p._openat2(0, "p", os.O_PATH)
    monkeypatch.setattr(p, "_supported_abi", lambda: True)
    for flags in [os.O_WRONLY, os.O_CREAT, -1, True]:
        with pytest.raises(p.PrefixError, match="open-flags"):
            p._openat2(0, "p", flags)
    for parent in [-1, 2**31, True]:
        with pytest.raises(p.PrefixError, match="parent-fd"):
            p._openat2(parent, "p", os.O_PATH)

    def syscall(number, parent, name, how_pointer, size):
        how = ctypes.cast(how_pointer, ctypes.POINTER(p._OpenHow)).contents
        assert number.value == 437 and parent.value == 7 and name.value == b"p"
        assert size.value == 24 and how.mode == 0 and how.resolve == 0x0D
        assert how.flags == os.O_PATH | os.O_NOFOLLOW | os.O_CLOEXEC
        return 123

    class Libc:
        pass

    libc = Libc()
    libc.syscall = syscall
    monkeypatch.setattr(p, "_LIBC", libc)
    assert p._openat2(7, "p", os.O_PATH) == 123


@pytest.mark.parametrize("content,size", [(b"short", 6), (b"long", 3)])
def test_hash_exact_size(tmp_path, content, size):
    file = tmp_path / "file"
    file.write_bytes(content)
    fd = os.open(file, os.O_RDONLY)
    try:
        with pytest.raises(p.PrefixError, match="content-changed"):
            p._hash_file(fd, size)
    finally:
        os.close(fd)


def test_hash_reads_fixed_chunks(tmp_path, monkeypatch):
    content = b"x" * (p._CHUNK_BYTES * 2 + 1)
    file = tmp_path / "file"
    file.write_bytes(content)
    original = os.read
    counts = []

    def read(fd, size):
        counts.append(size)
        return original(fd, size)

    monkeypatch.setattr(p.os, "read", read)
    fd = os.open(file, os.O_RDONLY)
    try:
        assert p._hash_file(fd, len(content)) == hashlib.sha256(content).hexdigest()
    finally:
        os.close(fd)
    assert counts == [p._CHUNK_BYTES, p._CHUNK_BYTES, 1, 1]


def test_enumeration_bounds_before_collecting_all_names(monkeypatch):
    consumed = []

    class Item:
        def __init__(self, name):
            self.name = name

    class Scan:
        def __enter__(self):
            def items():
                for i in range(100):
                    consumed.append(i)
                    yield Item(str(i))

            return items()

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(p.os, "scandir", lambda fd: Scan())
    monkeypatch.setattr(p, "_MAX_MEMBERS", 2)
    budget = p._Budget()
    with pytest.raises(p.PrefixError, match="member-limit"):
        p._directory_names(budget, 0)
    assert consumed == [0, 1, 2] and budget.live_fds == 0


def test_final_metadata_drift_and_unsupported_probes(tree, monkeypatch):
    parent, root = tree
    original = p.observe_inode_metadata
    counts = {}

    def observe(fd, name):
        result = original(fd, name)
        counts[name] = counts.get(name, 0) + 1
        return result.model_copy(
            update={
                "list_status": "unsupported",
                "list_errno": errno.ENOSYS,
                "name_list_size_bytes": None,
                "names_bytes_escaped": None,
            }
        )

    monkeypatch.setattr(p, "observe_inode_metadata", observe)
    result = p.inspect_prefix(parent, root.name)
    assert result.entries[1].metadata.list_status == "unsupported"
    assert result.metadata_coverage == "unknown"
    counts.clear()

    def drifting(fd, name):
        result = observe(fd, name)
        if name == "f" and counts[name] == 2:
            result = result.model_copy(update={"list_errno": errno.EPERM})
        return result

    monkeypatch.setattr(p, "observe_inode_metadata", drifting)
    with pytest.raises(p.PrefixError, match="entry-changed"):
        p.inspect_prefix(parent, root.name)


def test_openat2_syscall_error(monkeypatch):
    class Syscall:
        def __call__(self, *args):
            ctypes.set_errno(errno.ENOSYS)
            return -1

    class Libc:
        syscall = Syscall()

    monkeypatch.setattr(p, "_supported_abi", lambda: True)
    monkeypatch.setattr(p, "_LIBC", Libc())
    with pytest.raises(OSError) as caught:
        p._openat2(0, "prefix", os.O_PATH)
    assert caught.value.errno == errno.ENOSYS


def test_final_binding_replacement_detected(tree, monkeypatch):
    parent, root = tree
    original = p._binding
    changed = False

    def binding(budget, fd, name, expected):
        nonlocal changed
        if name == "f" and not changed:
            changed = True
            (root / "d/f").unlink()
            (root / "d/f").write_bytes(b"abc")
        return original(budget, fd, name, expected)

    monkeypatch.setattr(p, "_binding", binding)
    with pytest.raises(p.PrefixError, match="identity-changed"):
        p.inspect_prefix(parent, root.name)


def test_fixed_inventory_capacity_and_exact_boundary():
    assert p._MAX_EVIDENCE_BYTES == 256 * 1024**2
    budget = p._Budget()
    # Account for a measured-size workload without allocating a large fixture.
    budget.charge(177_169_168 - budget.evidence)
    assert budget.evidence > 128 * 1024**2
    budget.charge(p._MAX_EVIDENCE_BYTES - budget.evidence)
    assert budget.evidence == p._MAX_EVIDENCE_BYTES
    with pytest.raises(p.PrefixError, match="prefix-evidence-limit"):
        budget.charge(1)


def test_inventory_cap_rejects_before_payload_hash(tree, monkeypatch):
    parent, root = tree
    baseline = p.inspect_prefix(parent, root.name)
    entry = next(entry for entry in baseline.entries if entry.path == "d/f")
    data = p.canonical_json_bytes(entry.model_dump(mode="json"))
    charge = len(data) + 4 * data.count(b"\n")
    budget = p._Budget(evidence=p._MAX_EVIDENCE_BYTES - charge + 1)
    monkeypatch.setattr(p, "_hash_file", lambda *args: pytest.fail("hash after evidence overflow"))
    with (
        p._opened(budget, parent, root.name, os.O_RDONLY | os.O_DIRECTORY) as held,
        p._opened(budget, held, "d", os.O_RDONLY | os.O_DIRECTORY) as directory,
    ):
        before = set(os.listdir("/proc/self/fd"))
        with pytest.raises(p.PrefixError, match="prefix-evidence-limit"):
            p._capture(budget, directory, "f", "d/f", None)
        assert set(os.listdir("/proc/self/fd")) == before
    assert budget.live_fds == 0


def test_native_charge_includes_both_walks_and_four_name_scans(tree):
    parent, root = tree
    result = p.inspect_prefix(parent, root.name)
    serialized = [
        p.canonical_json_bytes(entry.model_dump(mode="json"))
        for entry in (result.root, *result.entries)
    ]
    entry_charge = sum(len(data) + 4 * data.count(b"\n") for data in serialized)
    names = sum(
        len(p.canonical_json_bytes(entry.path.rsplit("/", 1)[-1])) for entry in result.entries
    )
    assert result.evidence_bytes_charged == 4096 + 2 * entry_charge + 4 * names


@pytest.mark.parametrize("overflow", [False, True])
def test_link_output_keeps_independent_fixed_cap(monkeypatch, overflow):
    assert p._MAX_LINK_EVIDENCE_BYTES == 64 * 1024**2
    report = inventory(
        {"a": ("link", "."), **({"b": ("link", "."), "c": ("link", ".")} if overflow else {})}
    )
    original = p.canonical_json_bytes
    link_serializations = []

    class ChargedSize:
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

    def serialize(value):
        if isinstance(value, dict) and "status" in value:
            link_serializations.append(value["path"])
            # Only substitute length for link-output accounting; input inventory
            # validation and its serializer remain real. No large byte allocation.
            return ChargedSize(p._MAX_LINK_EVIDENCE_BYTES - 4096 if value["path"] == "a" else 1)
        return original(value)

    monkeypatch.setattr(p, "canonical_json_bytes", serialize)
    if overflow:
        with pytest.raises(p.PrefixError, match="prefix-evidence-limit"):
            p.resolve_inventory_links(report)
        assert link_serializations == ["a", "b"]
    else:
        result = p.resolve_inventory_links(report)
        assert [link.path for link in result.links] == ["a"]

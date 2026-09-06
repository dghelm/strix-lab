from __future__ import annotations

import ctypes
import errno
import json
import os
import socket
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from strixlab import rocm_metadata as metadata


@pytest.fixture
def tree(tmp_path: Path) -> Iterator[tuple[Path, int]]:
    root = tmp_path / "tree"
    root.mkdir(mode=0o700)
    (root / "file").write_bytes(b"synthetic")
    (root / "directory").mkdir()
    (root / "link").symlink_to("file")
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield root, descriptor
    finally:
        os.close(descriptor)


@pytest.fixture
def empty_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "_supported_abi", lambda: True)
    monkeypatch.setattr(metadata, "_listxattrat", lambda parent_fd, name, size: (0, b""))


def live_report(parent_fd: int, name: str) -> metadata.InodeMetadataObservationV1:
    if not metadata._supported_abi():
        pytest.skip("live support check only: native ABI unavailable; no qualification")
    result = metadata.observe_inode_metadata(parent_fd, name)
    assert result.coverage == "unknown"
    if result.list_status == "unsupported":
        assert result.list_errno in {errno.ENOSYS, errno.ENOTSUP}
        assert result.names_bytes_escaped is None
        pytest.skip(
            f"live support check only: listxattrat unsupported errno={result.list_errno}; "
            "no qualification"
        )
    return result


@pytest.mark.parametrize(
    "name,kind", [("file", "file"), ("directory", "directory"), ("link", "symlink")]
)
def test_live_empty_observation(tree: tuple[Path, int], name: str, kind: str) -> None:
    _, fd = tree
    result = live_report(fd, name)
    assert result.kind == kind
    assert result.list_status == "observed"
    assert result.list_errno is None
    assert result.names_bytes_escaped == ()
    assert result.name_list_size_bytes == 0
    assert result.coverage == "unknown"
    assert result.leaf_before == result.leaf_opened == result.leaf_after == result.leaf_named_after
    assert result.parent_before == result.parent_after
    os.fstat(fd)  # Caller still owns a usable descriptor.


def set_user_attribute(path: Path, name: bytes, value: bytes) -> None:
    try:
        os.setxattr(path, name, value)
    except OSError as exc:
        if exc.errno == errno.ENOTSUP:
            pytest.skip("fixture filesystem has no user xattrs")
        raise


@pytest.mark.parametrize("value", [b"", b"not read by the observer"])
def test_live_attributes_are_names_only(tree: tuple[Path, int], value: bytes) -> None:
    root, fd = tree
    set_user_attribute(root / "file", b"user.z", value)
    set_user_attribute(root / "file", b"user.a", b"other")
    result = live_report(fd, "file")
    assert result.names_bytes_escaped == (
        metadata._escaped(b"user.a"),
        metadata._escaped(b"user.z"),
    )
    assert result.name_list_size_bytes == len(b"user.a\0user.z\0")
    assert result.coverage == "unknown"
    assert value == b"" or value not in result.canonical_bytes()


def test_live_symlink_does_not_probe_referent(tree: tuple[Path, int]) -> None:
    root, fd = tree
    set_user_attribute(root / "file", b"user.referent-only", b"x")
    result = live_report(fd, "link")
    assert result.kind == "symlink"
    assert result.names_bytes_escaped == ()
    assert result.leaf_opened.ino == os.lstat(root / "link").st_ino
    assert result.leaf_opened.ino != os.stat(root / "link").st_ino


@pytest.mark.parametrize("target", ["missing", ".", "loop", "/definitely-not-a-fixture-target"])
def test_live_dangling_directory_cyclic_and_absolute_links_are_only_observed(
    tree: tuple[Path, int], target: str
) -> None:
    root, fd = tree
    (root / "loop").symlink_to(target)
    result = live_report(fd, "loop")
    assert result.kind == "symlink"
    assert result.names_bytes_escaped == ()
    assert result.coverage == "unknown"  # No link-closure admission.


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "/file", "file\0", "\udcff", "x" * 256])
def test_invalid_leaf_rejected(tree: tuple[Path, int], name: str) -> None:
    with pytest.raises(metadata.MetadataError, match="metadata-name"):
        metadata.observe_inode_metadata(tree[1], name)


def test_nonstring_leaf_rejected(tree: tuple[Path, int]) -> None:
    with pytest.raises(metadata.MetadataError, match="metadata-name"):
        metadata.observe_inode_metadata(tree[1], b"file")  # type: ignore[arg-type]


def test_utf8_leaf_is_escaped(tree: tuple[Path, int], empty_probe: None) -> None:
    root, fd = tree
    name = "é\n"
    (root / name).touch()
    result = metadata.observe_inode_metadata(fd, name)
    assert result.name_bytes_escaped == "\\xc3\\xa9\\x0a"
    assert json.loads(result.canonical_bytes())["coverage"] == "unknown"


def test_no_qualification_or_admission_api(tree: tuple[Path, int], empty_probe: None) -> None:
    result = metadata.observe_inode_metadata(tree[1], "file")
    assert not hasattr(metadata, "require_empty_metadata")
    assert not hasattr(metadata, "MetadataQualificationV1")
    assert not hasattr(result, "approved")
    with pytest.raises(ValidationError):
        metadata.InodeMetadataObservationV1(**(result.model_dump() | {"coverage": "qualified"}))
    with pytest.raises(ValidationError):
        result.coverage = "unknown"
    with pytest.raises(ValidationError):
        metadata.InodeMetadataObservationV1(**(result.model_dump() | {"qualification": True}))


@pytest.mark.parametrize(
    "code,status",
    [
        (errno.ENODATA, "error"),
        (errno.EACCES, "error"),
        (errno.EPERM, "error"),
        (errno.EIO, "error"),
        (errno.ERANGE, "error"),
        (errno.EINVAL, "error"),
        (errno.ENOSYS, "unsupported"),
        (errno.ENOTSUP, "unsupported"),
        (errno.E2BIG, "resource-limit"),
    ],
)
def test_errno_never_becomes_empty(
    tree: tuple[Path, int],
    monkeypatch: pytest.MonkeyPatch,
    empty_probe: None,
    code: int,
    status: str,
) -> None:
    calls = 0

    def fail(parent_fd: int, name: bytes, size: int) -> tuple[int, bytes]:
        nonlocal calls
        calls += 1
        raise OSError(code, "synthetic")

    monkeypatch.setattr(metadata, "_listxattrat", fail)
    result = metadata.observe_inode_metadata(tree[1], "file")
    assert result.list_status == status
    assert result.list_errno == code
    assert result.names_bytes_escaped is None
    assert result.name_list_size_bytes is None
    assert result.coverage == "unknown"
    assert calls == (2 if code == errno.ERANGE else 1)


def sequence_probe(
    monkeypatch: pytest.MonkeyPatch, responses: list[tuple[int, bytes] | OSError]
) -> list[int]:
    iterator = iter(responses)
    sizes: list[int] = []

    def probe(parent_fd: int, name: bytes, size: int) -> tuple[int, bytes]:
        sizes.append(size)
        result = next(iterator)
        if isinstance(result, OSError):
            raise result
        return result

    monkeypatch.setattr(metadata, "_listxattrat", probe)
    return sizes


def test_one_size_race_retry(
    tree: tuple[Path, int], monkeypatch: pytest.MonkeyPatch, empty_probe: None
) -> None:
    sizes = sequence_probe(
        monkeypatch, [(2, b""), OSError(errno.ERANGE, "grow"), (4, b""), (4, b"a\0b\0")]
    )
    result = metadata.observe_inode_metadata(tree[1], "file")
    assert result.names_bytes_escaped == ("\\x61", "\\x62")
    assert sizes == [0, 2, 0, 4]


def test_retry_is_bounded(
    tree: tuple[Path, int], monkeypatch: pytest.MonkeyPatch, empty_probe: None
) -> None:
    sizes = sequence_probe(
        monkeypatch,
        [(2, b""), OSError(errno.ERANGE, "grow"), (4, b""), OSError(errno.ERANGE, "grow")],
    )
    result = metadata.observe_inode_metadata(tree[1], "file")
    assert result.list_errno == errno.ERANGE
    assert result.list_status == "error"
    assert len(sizes) == 4


@pytest.mark.parametrize("data", [b"a\0a\0", b"\0", b"a\0\0", b"no-terminator"])
def test_malformed_name_lists(
    tree: tuple[Path, int], monkeypatch: pytest.MonkeyPatch, empty_probe: None, data: bytes
) -> None:
    sequence_probe(monkeypatch, [(len(data), b""), (len(data), data)])
    result = metadata.observe_inode_metadata(tree[1], "file")
    assert result.list_status == "malformed"
    assert result.names_bytes_escaped is None
    assert result.coverage == "unknown"


@pytest.mark.parametrize(
    "responses",
    [[(-2, b"")], [(4, b""), (-2, b"")], [(4, b""), (5, b"abcde")], [(4, b""), (4, b"a\0")]],
)
def test_invalid_syscall_lengths(
    tree: tuple[Path, int],
    monkeypatch: pytest.MonkeyPatch,
    empty_probe: None,
    responses: list[tuple[int, bytes]],
) -> None:
    sequence_probe(monkeypatch, list(responses))
    result = metadata.observe_inode_metadata(tree[1], "file")
    assert result.list_status == "malformed"


@pytest.mark.parametrize("used,data", [(0, b""), (2, b"a\0")])
def test_shrinking_list_is_drift(
    tree: tuple[Path, int],
    monkeypatch: pytest.MonkeyPatch,
    empty_probe: None,
    used: int,
    data: bytes,
) -> None:
    sequence_probe(monkeypatch, [(4, b""), (used, data)])
    with pytest.raises(metadata.MetadataError, match="metadata-names-changed"):
        metadata.observe_inode_metadata(tree[1], "file")


def test_arbitrary_name_bytes_sorted_and_escaped(
    tree: tuple[Path, int], monkeypatch: pytest.MonkeyPatch, empty_probe: None
) -> None:
    data = b"user.\xff\0user.\n\0"
    sequence_probe(monkeypatch, [(len(data), b""), (len(data), data)])
    result = metadata.observe_inode_metadata(tree[1], "file")
    assert result.names_bytes_escaped == (
        metadata._escaped(b"user.\n"),
        metadata._escaped(b"user.\xff"),
    )


def test_exact_metadata_bound(
    tree: tuple[Path, int], monkeypatch: pytest.MonkeyPatch, empty_probe: None
) -> None:
    data = b"".join(f"{i:04x}.".encode() + b"x" * 250 + b"\0" for i in range(256))
    assert len(data) == 65536
    sizes = sequence_probe(monkeypatch, [(len(data), b""), (len(data), data)])
    result = metadata.observe_inode_metadata(tree[1], "file")
    assert result.list_status == "observed"
    assert result.name_list_size_bytes == 65536
    assert len(result.names_bytes_escaped or ()) == 256
    assert max(sizes) == 65536


def test_over_limit_does_not_allocate_or_read(
    tree: tuple[Path, int], monkeypatch: pytest.MonkeyPatch, empty_probe: None
) -> None:
    sizes = sequence_probe(monkeypatch, [(65537, b"")])
    result = metadata.observe_inode_metadata(tree[1], "file")
    assert result.list_status == "resource-limit"
    assert result.name_list_size_bytes == 65537
    assert result.names_bytes_escaped is None
    assert sizes == [0]


@pytest.mark.parametrize(
    "change", ["bytes", "mode", "attribute", "replace", "hardlink", "parent-entry", "parent-mode"]
)
def test_observed_drift_never_returns_report(
    tree: tuple[Path, int], monkeypatch: pytest.MonkeyPatch, empty_probe: None, change: str
) -> None:
    root, fd = tree

    def probe(parent_fd: int, name: bytes, size: int) -> tuple[int, bytes]:
        if change == "bytes":
            (root / "file").write_bytes(b"changed length")
        elif change == "mode":
            (root / "file").chmod(0o400)
        elif change == "attribute":
            set_user_attribute(root / "file", b"user.changed", b"x")
        elif change == "replace":
            (root / "file").unlink()
            (root / "file").symlink_to("directory")
        elif change == "hardlink":
            os.link(root / "file", root / "alias")
        elif change == "parent-entry":
            (root / "other").touch()
        else:
            root.chmod(0o500)
        return 0, b""

    monkeypatch.setattr(metadata, "_listxattrat", probe)
    with pytest.raises(metadata.MetadataError, match="metadata-(leaf|parent)-changed"):
        metadata.observe_inode_metadata(fd, "file")


def test_symlink_replacement_during_probe(
    tree: tuple[Path, int], monkeypatch: pytest.MonkeyPatch, empty_probe: None
) -> None:
    root, fd = tree

    def probe(parent_fd: int, name: bytes, size: int) -> tuple[int, bytes]:
        (root / "link").unlink()
        (root / "link").symlink_to("directory")
        return 0, b""

    monkeypatch.setattr(metadata, "_listxattrat", probe)
    with pytest.raises(metadata.MetadataError, match="metadata-leaf-changed"):
        metadata.observe_inode_metadata(fd, "link")


def test_preopen_replacement(
    tree: tuple[Path, int], monkeypatch: pytest.MonkeyPatch, empty_probe: None
) -> None:
    root, fd = tree
    original = os.open

    def replace(path: object, flags: int, *args: object, **kwargs: object) -> int:
        if path == b"file":
            (root / "file").unlink()
            (root / "file").symlink_to("directory")
        return original(path, flags, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", replace)
    with pytest.raises(metadata.MetadataError, match="metadata-leaf-changed"):
        metadata.observe_inode_metadata(fd, "file")


def test_disappearance_is_io_failure(
    tree: tuple[Path, int], monkeypatch: pytest.MonkeyPatch, empty_probe: None
) -> None:
    root, fd = tree

    def probe(parent_fd: int, name: bytes, size: int) -> tuple[int, bytes]:
        (root / "file").unlink()
        raise OSError(errno.ENOENT, "disappeared")

    monkeypatch.setattr(metadata, "_listxattrat", probe)
    with pytest.raises(metadata.MetadataError, match="metadata-io"):
        metadata.observe_inode_metadata(fd, "file")


def test_special_inodes_are_not_opened(
    tree: tuple[Path, int], monkeypatch: pytest.MonkeyPatch, empty_probe: None
) -> None:
    root, fd = tree
    os.mkfifo(root / "fifo")
    with socket.socket(socket.AF_UNIX) as sock:
        sock.bind(str(root / "socket"))
        monkeypatch.setattr(os, "open", lambda *args, **kwargs: pytest.fail("special inode opened"))
        for name in ["fifo", "socket"]:
            with pytest.raises(metadata.MetadataError, match="metadata-leaf-type"):
                metadata.observe_inode_metadata(fd, name)


def test_parent_must_be_directory(tree: tuple[Path, int], empty_probe: None) -> None:
    fd = os.open(tree[0] / "file", os.O_RDONLY)
    try:
        with pytest.raises(metadata.MetadataError, match="metadata-parent-not-directory"):
            metadata.observe_inode_metadata(fd, "file")
    finally:
        os.close(fd)


def test_invalid_descriptor_and_missing_leaf(tree: tuple[Path, int], empty_probe: None) -> None:
    for fd, name in [(-1, "file"), (tree[1], "absent")]:
        with pytest.raises(metadata.MetadataError, match="metadata-io"):
            metadata.observe_inode_metadata(fd, name)


def test_descriptors_closed_on_success_and_failure(
    tree: tuple[Path, int], monkeypatch: pytest.MonkeyPatch, empty_probe: None
) -> None:
    original_dup, original_open = os.dup, os.open
    opened: list[int] = []

    def duplicate(fd: int) -> int:
        result = original_dup(fd)
        opened.append(result)
        return result

    def open_leaf(*args: object, **kwargs: object) -> int:
        result = original_open(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(result)
        return result

    monkeypatch.setattr(os, "dup", duplicate)
    monkeypatch.setattr(os, "open", open_leaf)
    metadata.observe_inode_metadata(tree[1], "file")
    sequence_probe(monkeypatch, [(2, b""), (0, b"")])
    with pytest.raises(metadata.MetadataError):
        metadata.observe_inode_metadata(tree[1], "file")
    assert len(opened) == 4
    for fd in opened:
        with pytest.raises(OSError):
            os.fstat(fd)
    os.fstat(tree[1])


@pytest.mark.parametrize(
    "system,machine,width",
    [("Darwin", "x86_64", 8), ("Linux", "aarch64", 8), ("Linux", "x86_64", 4)],
)
def test_abi_guards(monkeypatch: pytest.MonkeyPatch, system: str, machine: str, width: int) -> None:
    monkeypatch.setattr(metadata.platform, "system", lambda: system)
    monkeypatch.setattr(metadata.platform, "machine", lambda: machine)
    monkeypatch.setattr(metadata.ctypes, "sizeof", lambda value: width)
    assert not metadata._supported_abi()
    with pytest.raises(metadata.MetadataError, match="metadata-abi-unsupported"):
        metadata.observe_inode_metadata(-1, "file")


@pytest.mark.parametrize("attribute", ["O_PATH", "O_NOFOLLOW"])
def test_missing_open_flags_fail_closed(monkeypatch: pytest.MonkeyPatch, attribute: str) -> None:
    monkeypatch.delattr(os, attribute)
    assert not metadata._supported_abi()


def test_missing_syscall_symbol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, "_LIBC", SimpleNamespace())
    assert not metadata._supported_abi()


def test_native_syscall_argument_types_and_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSyscall:
        restype: object = None

        def __call__(self, *args: object) -> int:
            assert [type(arg) for arg in args] == [
                ctypes.c_long,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
                ctypes.c_void_p,
                ctypes.c_size_t,
            ]
            number, fd, name, flags, buffer, size = args
            assert number.value == 465  # type: ignore[attr-defined]
            assert fd.value == 17  # type: ignore[attr-defined]
            assert name.value == b"link"  # type: ignore[attr-defined]
            assert flags.value == 0x100  # type: ignore[attr-defined]
            if size.value:  # type: ignore[attr-defined]
                ctypes.memmove(buffer, b"a\0", 2)
            else:
                assert buffer.value is None  # type: ignore[attr-defined]
            return 2

    syscall = FakeSyscall()
    monkeypatch.setattr(metadata, "_LIBC", SimpleNamespace(syscall=syscall))
    assert metadata._listxattrat(17, b"link", 0) == (2, b"")
    assert metadata._listxattrat(17, b"link", 2) == (2, b"a\0")
    assert syscall.restype is ctypes.c_long


def test_syscall_errno_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    class Fail:
        restype: object = None

        def __call__(self, *args: object) -> int:
            ctypes.set_errno(errno.EPERM)
            return -1

    monkeypatch.setattr(metadata, "_LIBC", SimpleNamespace(syscall=Fail()))
    with pytest.raises(OSError) as caught:
        metadata._listxattrat(17, b"file", 0)
    assert caught.value.errno == errno.EPERM

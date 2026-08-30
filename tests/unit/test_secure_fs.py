from __future__ import annotations

import stat
from pathlib import Path

import pytest

from strixlab.secure_fs import fsync_directory, write_exclusive


def test_write_exclusive_refuses_overwrite_and_symlink_following(tmp_path: Path) -> None:
    destination = tmp_path / "record"
    write_exclusive(destination, b"first")

    with pytest.raises(FileExistsError):
        write_exclusive(destination, b"second")
    assert destination.read_bytes() == b"first"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600

    target = tmp_path / "target"
    target.write_bytes(b"target")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        write_exclusive(link, b"replacement")
    assert target.read_bytes() == b"target"

    fsync_directory(tmp_path)


def test_fsync_tree_flushes_files_and_rejects_symlink(tmp_path: Path) -> None:
    from strixlab.secure_fs import fsync_tree

    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_bytes(b"a")
    (root / "sub" / "b.txt").write_bytes(b"b")
    fsync_tree(root)  # succeeds over a clean owned tree

    (root / "sub" / "link").symlink_to(root / "a.txt")
    with pytest.raises(OSError, match="unsafe file in fsync tree"):
        fsync_tree(root)


def test_fsync_tree_rejects_special_file(tmp_path: Path) -> None:
    import os

    from strixlab.secure_fs import fsync_tree

    root = tmp_path / "tree"
    root.mkdir()
    os.mkfifo(root / "pipe")
    with pytest.raises(OSError, match="unsafe file in fsync tree"):
        fsync_tree(root)


def test_open_owned_directory_opens_and_rejects(tmp_path: Path) -> None:
    import os

    from strixlab.secure_fs import (
        UnownedDirectoryError,
        directory_open_flags,
        open_owned_directory,
        try_open_owned_directory,
    )

    flags = directory_open_flags()
    assert flags & os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        assert flags & os.O_NOFOLLOW

    fd = open_owned_directory(tmp_path)
    try:
        assert stat.S_ISDIR(os.fstat(fd).st_mode)
    finally:
        os.close(fd)

    # A regular file fails to open as a directory (ENOTDIR), never as unowned.
    (tmp_path / "afile").write_bytes(b"x")
    with pytest.raises(OSError) as caught:
        open_owned_directory(tmp_path / "afile")
    assert not isinstance(caught.value, UnownedDirectoryError)

    # Absent entry: hard error from open, None from the try- variant.
    with pytest.raises(FileNotFoundError):
        open_owned_directory(tmp_path / "missing")
    assert try_open_owned_directory(tmp_path / "missing") is None


def test_rename_noreplace_preserves_file_exists_behavior(tmp_path: Path) -> None:
    from strixlab.secure_fs import rename_noreplace

    source = tmp_path / "src"
    source.write_bytes(b"payload")
    destination = tmp_path / "dst"
    rename_noreplace(source, destination)
    assert destination.read_bytes() == b"payload"

    other = tmp_path / "src2"
    other.write_bytes(b"other")
    with pytest.raises(FileExistsError):
        rename_noreplace(other, destination)
    assert destination.read_bytes() == b"payload"

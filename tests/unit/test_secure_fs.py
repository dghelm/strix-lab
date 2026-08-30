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

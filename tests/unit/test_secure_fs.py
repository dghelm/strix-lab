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

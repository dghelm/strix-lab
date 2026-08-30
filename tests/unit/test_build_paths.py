from __future__ import annotations

from pathlib import Path

import pytest

from strixlab.build_paths import build_storage_roots, is_unsafe_directory


def test_is_unsafe_directory_rejects_symlink_and_non_directory(tmp_path: Path) -> None:
    directory = tmp_path / "dir"
    directory.mkdir()
    assert is_unsafe_directory(directory.lstat()) is False

    regular = tmp_path / "file"
    regular.write_text("x", encoding="utf-8")
    assert is_unsafe_directory(regular.lstat()) is True

    link = tmp_path / "link"
    link.symlink_to(directory, target_is_directory=True)
    # lstat reports the symlink itself, which is never a safe owned directory.
    assert is_unsafe_directory(link.lstat()) is True


def test_build_storage_roots_requires_absolute_home() -> None:
    with pytest.raises(ValueError, match="absolute"):
        build_storage_roots(Path("relative"))
    roots = build_storage_roots(Path("/srv/strixlab"))
    assert roots.root == Path("/srv/strixlab/builds")
    assert roots.locks == Path("/srv/strixlab/locks")

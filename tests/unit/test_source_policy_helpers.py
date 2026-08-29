from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from strixlab.sources import (
    SourcePolicyError,
    _fsync_tree,
    _reject_symlinked_ancestors,
    _resolve_relative_locator,
    _safe_repo_path,
    _status,
    _submodule_config,
    _tree_bytes,
    _validate_candidate_paths,
    _validate_index_state,
    _validated_submodule_path,
)


class BytesGit:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def bytes(self, *_args: object, **_kwargs: object) -> bytes:
        return self.content


def gitmodules_bytes(*entries: tuple[str, str]) -> bytes:
    return b"".join(
        key.encode("utf-8") + b"\n" + value.encode("utf-8") + b"\0" for key, value in entries
    )


@pytest.mark.parametrize("value", ["", "/absolute", "../escape", ".git/config", "bad\npath"])
def test_repository_paths_reject_unsafe_spellings(value: str) -> None:
    with pytest.raises(SourcePolicyError, match="unsafe repository-relative path"):
        _safe_repo_path(value)


def test_relative_submodule_locators_follow_parent_transport_semantics() -> None:
    assert _resolve_relative_locator("/srv/parent", "../child") == "/srv/child"
    assert _resolve_relative_locator("file:///srv/parent", "../child") == "file:///srv/child"
    assert (
        _resolve_relative_locator("https://example.test/org/parent", "../child")
        == "https://example.test/org/child"
    )
    assert (
        _resolve_relative_locator("git@example.test:org/parent", "../child")
        == "git@example.test:org/child"
    )
    assert (
        _resolve_relative_locator("https://example.test/parent", "ssh://git@example.test/child")
        == "ssh://git@example.test/child"
    )


def test_submodule_config_is_parsed_as_restricted_data(tmp_path: Path) -> None:
    (tmp_path / ".gitmodules").write_text("fixture", encoding="utf-8")
    commit = "1" * 40
    content = gitmodules_bytes(
        ("submodule.child.path", "deps/child"),
        ("submodule.child.url", "../child"),
        ("submodule.child.branch", "ignored-for-pinning"),
        ("submodule.child.update", "checkout"),
    )

    parsed = _submodule_config(tmp_path, {"deps/child": commit}, "/srv/parent", BytesGit(content))  # type: ignore[arg-type]

    assert parsed == {"deps/child": ("child", "/srv/child")}
    assert _submodule_config(tmp_path, {}, "/srv/parent", BytesGit(b"")) == {}  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        (("unexpected.key", "value"), "unsupported"),
        (("submodule.child.path", "bad\npath"), "unsafe submodule metadata"),
        (("submodule.child.path", "deps/child"), "lacks path or URL"),
        (
            (
                "submodule.child.path",
                "deps/child",
                "submodule.child.url",
                "../child",
                "submodule.child.update",
                "merge",
            ),
            "unsupported submodule update",
        ),
    ],
)
def test_submodule_config_rejects_unsupported_data(
    tmp_path: Path, entries: tuple[str, ...], message: str
) -> None:
    (tmp_path / ".gitmodules").write_text("fixture", encoding="utf-8")
    pairs = tuple(zip(entries[::2], entries[1::2], strict=True))

    with pytest.raises(SourcePolicyError, match=message):
        _submodule_config(  # type: ignore[arg-type]
            tmp_path,
            {"deps/child": "1" * 40},
            "/srv/parent",
            BytesGit(gitmodules_bytes(*pairs)),
        )


def test_submodule_config_rejects_duplicates_and_path_mismatches(tmp_path: Path) -> None:
    (tmp_path / ".gitmodules").write_text("fixture", encoding="utf-8")
    duplicate = gitmodules_bytes(
        ("submodule.child.path", "deps/child"),
        ("submodule.child.path", "deps/other"),
        ("submodule.child.url", "../child"),
    )
    mismatch = gitmodules_bytes(
        ("submodule.child.path", "deps/other"),
        ("submodule.child.url", "../child"),
    )

    with pytest.raises(SourcePolicyError, match="duplicate .gitmodules key"):
        _submodule_config(tmp_path, {"deps/child": "1" * 40}, "/srv/parent", BytesGit(duplicate))  # type: ignore[arg-type]
    with pytest.raises(SourcePolicyError, match="do not match"):
        _submodule_config(tmp_path, {"deps/child": "1" * 40}, "/srv/parent", BytesGit(mismatch))  # type: ignore[arg-type]


def test_submodule_paths_reject_symlinked_ancestors(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / "deps").symlink_to(target, target_is_directory=True)

    with pytest.raises(SourcePolicyError, match="symlinked ancestor"):
        _reject_symlinked_ancestors(tmp_path, PurePosixPath("deps/child"))


def test_cleanup_submodule_path_walk_is_nofollow(tmp_path: Path) -> None:
    child = tmp_path / "deps" / "child"
    child.mkdir(parents=True)
    resolved, exists = _validated_submodule_path(tmp_path, "deps/child")
    assert resolved == child
    assert exists is True

    child.rmdir()
    child.write_text("not a directory", encoding="utf-8")
    assert _validated_submodule_path(tmp_path, "deps/child") == (child, False)

    child.unlink()
    assert _validated_submodule_path(tmp_path, "deps/missing") == (
        tmp_path / "deps" / "missing",
        False,
    )


def test_tree_byte_budget_ignores_git_and_symlinks(tmp_path: Path) -> None:
    (tmp_path / "file").write_bytes(b"123")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "file").write_bytes(b"45")
    (tmp_path / "linked").symlink_to(nested, target_is_directory=True)
    metadata = tmp_path / ".git"
    metadata.mkdir()
    (metadata / "ignored").write_bytes(b"ignored")

    assert _tree_bytes(tmp_path) == 5


def test_tree_fsync_skips_symlink_entries(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"durable")
    (tmp_path / "linked").symlink_to(target)

    _fsync_tree(tmp_path)


def test_status_and_index_validation_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    invalid_git = BytesGit(b"?? \xff\0")
    with pytest.raises(SourcePolicyError, match="non-UTF-8"):
        _status(Path("/tmp/worktree"), invalid_git)  # type: ignore[arg-type]

    monkeypatch.setattr("strixlab.sources._status", lambda *_args: ("?? untracked",))
    with pytest.raises(SourcePolicyError, match="unstaged or untracked"):
        _validate_index_state(Path("/tmp/worktree"), invalid_git)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"invalid\0", "invalid staged raw diff"),
        (
            b":160000 100644 " + b"0" * 40 + b" " + b"1" * 40 + b" M\0deps/child\0",
            "change gitlinks",
        ),
    ],
)
def test_raw_candidate_path_validation_rejects_invalid_records(
    tmp_path: Path, raw: bytes, message: str
) -> None:
    with pytest.raises(SourcePolicyError, match=message):
        _validate_candidate_paths(tmp_path, "0" * 40, BytesGit(raw))  # type: ignore[arg-type]

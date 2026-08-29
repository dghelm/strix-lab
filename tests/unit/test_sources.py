from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

import strixlab.sources as source_module
from strixlab.git_boundary import GitBoundary, GitBoundaryError
from strixlab.locks import exclusive_lock
from strixlab.manifests import SourceLockV1
from strixlab.process import ProcessOutcome, ProcessResult
from strixlab.serialization import canonical_json_bytes
from strixlab.sources import (
    RegistryState,
    SourceCommandError,
    SourceDivergedError,
    SourcePolicyError,
    SourceTransitionInterrupt,
    cleanup_source,
    inspect_source,
    lease_source,
    prepare_source,
)


@pytest.fixture(autouse=True)
def allow_root_source_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.geteuid() == 0:
        monkeypatch.setattr("strixlab.sources._refuse_root", lambda: None)


@dataclass(frozen=True)
class GitRepository:
    path: Path
    commit: str


def process_result(*, stdout: str = "", returncode: int = 0) -> ProcessResult:
    return ProcessResult(
        ProcessOutcome.EXITED,
        ("git",),
        returncode,
        stdout,
        "",
        0.0,
        0.0,
        0.0,
        None,
    )


class FakeFetchBoundary:
    def __init__(self, commit: str, network_errors: list[str]) -> None:
        self.commit = commit
        self.network_errors = network_errors
        self.network_calls: list[tuple[str, ...]] = []

    def run_network(self, arguments: list[str], **_kwargs: object) -> ProcessResult:
        self.network_calls.append(tuple(arguments))
        if self.network_errors:
            raise GitBoundaryError(self.network_errors.pop(0))
        return process_result()

    def run(self, arguments: list[str], **_kwargs: object) -> ProcessResult:
        if "cat-file" in arguments:
            return process_result(stdout="commit\n")
        if "rev-parse" in arguments:
            return process_result(stdout=f"{self.commit}\n")
        if "show-ref" in arguments:
            return process_result(returncode=1)
        return process_result()


def git(cwd: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        env={
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": str(cwd),
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": os.environ["PATH"],
        },
    )
    return result.stdout.strip()


def create_repository(root: Path, name: str, content: str = "baseline\n") -> GitRepository:
    repository = root / name
    git(root, "init", "--initial-branch=main", str(repository))
    git(repository, "config", "user.name", "StrixLab Test")
    git(repository, "config", "user.email", "strixlab@example.invalid")
    (repository / "hello.txt").write_text(content, encoding="utf-8")
    git(repository, "add", "hello.txt")
    git(repository, "commit", "-m", "fixture")
    return GitRepository(repository, git(repository, "rev-parse", "HEAD"))


@pytest.fixture
def upstream(tmp_path: Path) -> GitRepository:
    return create_repository(tmp_path, "upstream")


def source_lock(
    repository: GitRepository, *, source_id: str = "fixture", submodules: bool = False
) -> SourceLockV1:
    return SourceLockV1.model_validate(
        {
            "schema_version": 1,
            "id": source_id,
            "kind": "git",
            "url": str(repository.path),
            "commit": repository.commit,
            "branch_hint": "main",
            "submodules": submodules,
            "adapter": "llama_cpp",
            "allowed_dirty_state": False,
        }
    )


def fixed_nonce() -> str:
    return "0123456789abcdef0123456789abcdef"


def test_prepare_patch_inspect_and_cleanup_are_evidence_first(
    tmp_path: Path, upstream: GitRepository
) -> None:
    home = tmp_path / "home"
    patch = tmp_path / "candidate.patch"
    patch.write_text(
        """diff --git a/hello.txt b/hello.txt
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-baseline
+candidate
""",
        encoding="utf-8",
    )

    prepared = prepare_source(
        source_lock(upstream),
        home=home,
        patches=[patch],
        nonce_factory=fixed_nonce,
    )

    assert prepared.worktree != upstream.path
    assert (prepared.worktree / "hello.txt").read_text(encoding="utf-8") == "candidate\n"
    assert prepared.evidence.preparation_id.startswith("prep-fixture-")
    assert prepared.evidence.request_digest not in {
        prepared.evidence.content_tree_id,
        prepared.evidence.candidate_id,
    }
    assert prepared.evidence.content_tree_id.startswith("content-tree-sha256:")
    assert prepared.evidence.candidate_id.startswith("candidate-sha256:")
    assert prepared.evidence.status == ("M  hello.txt",)
    assert prepared.evidence.source_locator is None
    assert (prepared.record / "patch-001.patch").read_bytes() == patch.read_bytes()
    candidate_diff = (prepared.record / "candidate.diff").read_bytes()
    assert b"-baseline\n+candidate\n" in candidate_diff
    assert len(candidate_diff) == prepared.evidence.diff_size_bytes

    portable = (prepared.record / "evidence.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in portable
    registry_directory = next((home / "sources" / "registry").iterdir())
    local_registry = (registry_directory / "current.json").read_text(encoding="utf-8")
    assert str(tmp_path) in local_registry

    inspected = inspect_source(prepared.evidence.preparation_id, home=home)
    assert inspected.registry.state is RegistryState.PUBLISHED
    assert inspected.evidence == prepared.evidence
    assert inspected.worktree_exists is inspected.record_exists is True

    cleaned = cleanup_source(prepared.evidence.preparation_id, home=home)
    assert cleaned.state is RegistryState.CLEANED
    assert not prepared.worktree.exists()
    assert prepared.record.is_dir()
    assert cleanup_source(prepared.evidence.preparation_id, home=home) == cleaned
    assert (upstream.path / "hello.txt").read_text(encoding="utf-8") == "baseline\n"
    assert git(upstream.path, "status", "--porcelain") == ""


def test_patch_content_and_identity_are_fixed_before_slow_git_work(
    tmp_path: Path, upstream: GitRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    patch = tmp_path / "candidate.patch"
    original = b"""diff --git a/hello.txt b/hello.txt
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-baseline
+candidate
"""
    patch.write_bytes(original)
    original_prepare_mirror = source_module._prepare_mirror

    def mutate_input_after_staging(*args: object, **kwargs: object) -> Path:
        patch.write_bytes(b"changed after authenticated read")
        return original_prepare_mirror(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(source_module, "_prepare_mirror", mutate_input_after_staging)
    prepared = prepare_source(
        source_lock(upstream),
        home=tmp_path / "home",
        patches=[patch],
        nonce_factory=fixed_nonce,
    )

    assert (prepared.record / "patch-001.patch").read_bytes() == original
    assert prepared.evidence.patches[0].sha256 == hashlib.sha256(original).hexdigest()
    cleanup_source(prepared.evidence.preparation_id, home=tmp_path / "home")


def test_cleanup_refuses_worktree_divergence(tmp_path: Path, upstream: GitRepository) -> None:
    prepared = prepare_source(
        source_lock(upstream),
        home=tmp_path / "home",
        nonce_factory=fixed_nonce,
    )
    (prepared.worktree / "hello.txt").write_text("unpreserved\n", encoding="utf-8")

    with pytest.raises(SourceDivergedError, match="candidate state diverged"):
        cleanup_source(prepared.evidence.preparation_id, home=tmp_path / "home")

    inspected = inspect_source(prepared.evidence.preparation_id, home=tmp_path / "home")
    assert inspected.registry.state is RegistryState.PUBLISHED
    assert prepared.worktree.exists()
    (prepared.worktree / "hello.txt").write_text("baseline\n", encoding="utf-8")
    cleanup_source(prepared.evidence.preparation_id, home=tmp_path / "home")


def test_patch_cannot_modify_submodule_metadata(tmp_path: Path, upstream: GitRepository) -> None:
    patch = tmp_path / "bad.patch"
    patch.write_text(
        """diff --git a/.gitmodules b/.gitmodules
new file mode 100644
--- /dev/null
+++ b/.gitmodules
@@ -0,0 +1 @@
+blocked
""",
        encoding="utf-8",
    )

    with pytest.raises(SourcePolicyError, match="gitmodules"):
        prepare_source(
            source_lock(upstream),
            home=tmp_path / "home",
            patches=[patch],
            nonce_factory=fixed_nonce,
        )
    assert (tmp_path / "home" / "sources" / "mirrors" / "fixture.git").is_dir()


def test_existing_mirror_refuses_a_different_remote(tmp_path: Path) -> None:
    first = create_repository(tmp_path, "first")
    second = create_repository(tmp_path, "second")
    home = tmp_path / "home"
    prepared = prepare_source(source_lock(first), home=home, nonce_factory=fixed_nonce)
    cleanup_source(prepared.evidence.preparation_id, home=home)

    with pytest.raises(SourcePolicyError, match="mirror-origin-mismatch"):
        prepare_source(
            source_lock(second),
            home=home,
            nonce_factory=lambda: "f" * 32,
        )
    assert git(first.path, "status", "--porcelain") == ""
    assert git(second.path, "status", "--porcelain") == ""


def test_submodules_are_materialized_and_verified_one_level_at_a_time(tmp_path: Path) -> None:
    child = create_repository(tmp_path, "child", "child\n")
    parent = create_repository(tmp_path, "parent", "parent\n")
    git(
        parent.path,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child.path),
        "deps/child",
    )
    git(parent.path, "commit", "-am", "add pinned child")
    parent = GitRepository(parent.path, git(parent.path, "rev-parse", "HEAD"))

    prepared = prepare_source(
        source_lock(parent, submodules=True),
        home=tmp_path / "home",
        nonce_factory=fixed_nonce,
    )

    assert (prepared.worktree / "deps" / "child" / "hello.txt").read_text() == "child\n"
    assert len(prepared.evidence.submodules) == 1
    submodule = prepared.evidence.submodules[0]
    assert submodule.path == "deps/child"
    assert submodule.commit == child.commit
    assert submodule.locator is None
    assert submodule.locator_sha256 == hashlib.sha256(str(child.path).encode()).hexdigest()
    portable = json.loads((prepared.record / "evidence.json").read_bytes())
    assert str(tmp_path) not in json.dumps(portable)
    cleanup_source(prepared.evidence.preparation_id, home=tmp_path / "home")


def test_inspect_missing_state_is_read_only(tmp_path: Path) -> None:
    home = tmp_path / "missing-home"

    with pytest.raises(SourcePolicyError, match="does not exist"):
        inspect_source("prep-fixture-0123456789abcdef01234567", home=home)

    assert not home.exists()


def test_failed_fetch_is_recorded_and_cleanup_is_recoverable(
    tmp_path: Path, upstream: GitRepository
) -> None:
    lock = source_lock(upstream).model_copy(update={"commit": "f" * 40})
    home = tmp_path / "home"

    with pytest.raises((SourceCommandError, SourcePolicyError)):
        prepare_source(lock, home=home, nonce_factory=fixed_nonce)

    preparation_id = next((home / "sources" / "registry").iterdir()).name
    inspected = inspect_source(preparation_id, home=home)
    assert inspected.registry.state is RegistryState.FAILED
    assert inspected.registry.failure_code in {"SourceCommandError", "SourcePolicyError"}
    assert inspected.evidence is None
    assert inspected.worktree_exists is False
    assert (home / "sources" / "registry" / preparation_id / "current.json").is_file()
    assert not (home / "sources" / "mirrors" / "fixture.git").exists()
    assert cleanup_source(preparation_id, home=home).state is RegistryState.CLEANED


def test_exact_object_fetch_failure_uses_the_validated_branch_fallback(
    tmp_path: Path, upstream: GitRepository
) -> None:
    lock = source_lock(upstream)
    fake = FakeFetchBoundary(lock.commit, ["exact fetch failed"])

    source_module._fetch_and_verify(
        cast(GitBoundary, fake),
        tmp_path / "mirror.git",
        lock,
        str(upstream.path),
        "prep-fixture-0123456789abcdef01234567",
        tmp_path,
    )

    assert len(fake.network_calls) == 2
    assert fake.network_calls[0][-1].endswith(
        ":refs/strixlab/quarantine/prep-fixture-0123456789abcdef01234567/raw"
    )
    assert "refs/heads/main" in fake.network_calls[1][-1]


def test_exact_object_fetch_failure_without_a_branch_hint_is_terminal(
    tmp_path: Path, upstream: GitRepository
) -> None:
    lock = source_lock(upstream).model_copy(update={"branch_hint": None})
    fake = FakeFetchBoundary(lock.commit, ["exact fetch failed"])

    with pytest.raises(SourceCommandError, match="exact fetch failed"):
        source_module._fetch_and_verify(
            cast(GitBoundary, fake),
            tmp_path / "mirror.git",
            lock,
            str(upstream.path),
            "prep-fixture-0123456789abcdef01234567",
            tmp_path,
        )

    assert len(fake.network_calls) == 1


def test_exact_object_fetch_reports_a_failed_branch_fallback(
    tmp_path: Path, upstream: GitRepository
) -> None:
    lock = source_lock(upstream)
    fake = FakeFetchBoundary(lock.commit, ["exact fetch failed", "branch fetch failed"])

    with pytest.raises(SourceCommandError, match="exact fetch failed; branch fallback failed"):
        source_module._fetch_and_verify(
            cast(GitBoundary, fake),
            tmp_path / "mirror.git",
            lock,
            str(upstream.path),
            "prep-fixture-0123456789abcdef01234567",
            tmp_path,
        )

    assert len(fake.network_calls) == 2


def test_patch_symlinks_are_rejected(tmp_path: Path, upstream: GitRepository) -> None:
    target = tmp_path / "target.patch"
    target.write_text("patch", encoding="utf-8")
    patch = tmp_path / "candidate.patch"
    patch.symlink_to(target)

    with pytest.raises(SourcePolicyError, match="regular file"):
        prepare_source(
            source_lock(upstream),
            home=tmp_path / "home",
            patches=[patch],
            nonce_factory=fixed_nonce,
        )


def test_cleanup_requires_the_nonce_worktree_lock(tmp_path: Path, upstream: GitRepository) -> None:
    home = tmp_path / "home"
    prepared = prepare_source(source_lock(upstream), home=home, nonce_factory=fixed_nonce)
    mirror = home / "sources" / "mirrors" / "fixture.git"
    git(home, "--git-dir", str(mirror), "worktree", "unlock", str(prepared.worktree))

    with pytest.raises(SourceDivergedError, match="lock reason"):
        cleanup_source(prepared.evidence.preparation_id, home=home)

    git(
        home,
        "--git-dir",
        str(mirror),
        "worktree",
        "lock",
        "--reason",
        f"strixlab-source-v1:{prepared.evidence.preparation_id}:{fixed_nonce()}",
        str(prepared.worktree),
    )
    cleanup_source(prepared.evidence.preparation_id, home=home)


def test_cleanup_refuses_a_changed_staged_diff(tmp_path: Path, upstream: GitRepository) -> None:
    home = tmp_path / "home"
    patch = tmp_path / "candidate.patch"
    patch.write_text(
        """diff --git a/hello.txt b/hello.txt
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-baseline
+candidate
""",
        encoding="utf-8",
    )
    prepared = prepare_source(
        source_lock(upstream), home=home, patches=[patch], nonce_factory=fixed_nonce
    )
    (prepared.worktree / "hello.txt").write_text("different\n", encoding="utf-8")
    git(prepared.worktree, "add", "hello.txt")

    with pytest.raises(SourceDivergedError, match="candidate state diverged"):
        cleanup_source(prepared.evidence.preparation_id, home=home)

    (prepared.worktree / "hello.txt").write_text("candidate\n", encoding="utf-8")
    git(prepared.worktree, "add", "hello.txt")
    cleanup_source(prepared.evidence.preparation_id, home=home)


def test_preparation_policy_guards_fail_before_git_side_effects(
    tmp_path: Path, upstream: GitRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock = source_lock(upstream)
    with pytest.raises(ValueError, match="absolute"):
        prepare_source(lock, home=Path("relative"), nonce_factory=fixed_nonce)

    with monkeypatch.context() as root_context:
        root_context.setattr("strixlab.sources.os.geteuid", lambda: 0)
        with pytest.raises(SourcePolicyError, match="root"):
            prepare_source(lock, home=tmp_path / "root-home", nonce_factory=fixed_nonce)

    with pytest.raises(ValueError, match="nonce factory"):
        prepare_source(lock, home=tmp_path / "nonce-home", nonce_factory=lambda: "invalid")

    monkeypatch.setattr("strixlab.git_boundary.shutil.which", lambda _name, *, path: None)
    with pytest.raises(SourceCommandError, match="required executable is unavailable: git"):
        prepare_source(lock, home=tmp_path / "git-home", nonce_factory=fixed_nonce)


def test_home_and_patch_files_must_be_safe(tmp_path: Path, upstream: GitRepository) -> None:
    real_home = tmp_path / "real-home"
    real_home.mkdir()
    linked_home = tmp_path / "linked-home"
    linked_home.symlink_to(real_home, target_is_directory=True)
    with pytest.raises(SourcePolicyError, match="symbolic link"):
        prepare_source(source_lock(upstream), home=linked_home, nonce_factory=fixed_nonce)

    oversized = tmp_path / "oversized.patch"
    with oversized.open("wb") as stream:
        stream.truncate(64 * 1024 * 1024 + 1)
    with pytest.raises(SourcePolicyError, match="exceeds"):
        prepare_source(
            source_lock(upstream),
            home=tmp_path / "patch-home",
            patches=[oversized],
            nonce_factory=fixed_nonce,
        )


def test_registry_and_evidence_corruption_are_rejected(
    tmp_path: Path, upstream: GitRepository
) -> None:
    home = tmp_path / "home"
    prepared = prepare_source(source_lock(upstream), home=home, nonce_factory=fixed_nonce)
    registry_path = (
        home / "sources" / "registry" / prepared.evidence.preparation_id / "current.json"
    )
    registry = json.loads(registry_path.read_bytes())
    registry["mirror_path"] = str(tmp_path / "unowned")
    registry_path.write_bytes(canonical_json_bytes(registry))
    with pytest.raises(SourcePolicyError, match="owned layout"):
        inspect_source(prepared.evidence.preparation_id, home=home)

    registry["mirror_path"] = str(home / "sources" / "mirrors" / "fixture.git")
    registry_path.write_bytes(canonical_json_bytes(registry))
    (prepared.record / "evidence.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SourcePolicyError, match="invalid source record"):
        inspect_source(prepared.evidence.preparation_id, home=home)


def test_cleanup_recovers_an_admin_entry_with_a_missing_worktree(
    tmp_path: Path, upstream: GitRepository
) -> None:
    home = tmp_path / "home"
    prepared = prepare_source(source_lock(upstream), home=home, nonce_factory=fixed_nonce)
    shutil.rmtree(prepared.worktree)

    assert (
        cleanup_source(prepared.evidence.preparation_id, home=home).state is RegistryState.CLEANED
    )


def test_cleanup_recovers_a_worktree_with_a_missing_admin_entry(
    tmp_path: Path, upstream: GitRepository
) -> None:
    home = tmp_path / "home"
    prepared = prepare_source(source_lock(upstream), home=home, nonce_factory=fixed_nonce)
    owner = inspect_source(prepared.evidence.preparation_id, home=home).registry.ownership
    assert owner is not None
    shutil.rmtree(owner.admin_path)

    assert (
        cleanup_source(prepared.evidence.preparation_id, home=home).state is RegistryState.CLEANED
    )
    assert not prepared.worktree.exists()


@pytest.mark.parametrize("replacement_side", ["worktree", "admin"])
def test_cleanup_rejects_dangling_replacement_symlinks(
    tmp_path: Path, upstream: GitRepository, replacement_side: str
) -> None:
    home = tmp_path / "home"
    prepared = prepare_source(source_lock(upstream), home=home, nonce_factory=fixed_nonce)
    owner = inspect_source(prepared.evidence.preparation_id, home=home).registry.ownership
    assert owner is not None
    admin = Path(owner.admin_path)
    if replacement_side == "worktree":
        shutil.rmtree(prepared.worktree)
        prepared.worktree.symlink_to(tmp_path / "missing-worktree", target_is_directory=True)
        survivor = admin
        replacement = prepared.worktree
    else:
        shutil.rmtree(admin)
        admin.symlink_to(tmp_path / "missing-admin", target_is_directory=True)
        survivor = prepared.worktree
        replacement = admin

    with pytest.raises(SourceDivergedError, match="unsafe"):
        cleanup_source(prepared.evidence.preparation_id, home=home)

    assert replacement.is_symlink()
    assert survivor.is_dir()


@pytest.mark.parametrize(
    "interrupted_state",
    [
        RegistryState.MIRROR_READY,
        RegistryState.WORKTREE_CREATED,
        RegistryState.CANDIDATE_READY,
        RegistryState.PUBLISHED,
    ],
)
def test_transition_interruptions_remain_inspectable_and_cleanable(
    tmp_path: Path, upstream: GitRepository, interrupted_state: RegistryState
) -> None:
    home = tmp_path / "home"

    def interrupt(state: RegistryState) -> None:
        if state is interrupted_state:
            raise SourceTransitionInterrupt(state)

    with pytest.raises(BaseException, match=interrupted_state):
        prepare_source(
            source_lock(upstream),
            home=home,
            nonce_factory=fixed_nonce,
            transition_hook=interrupt,
        )

    preparation_id = next((home / "sources" / "registry").iterdir()).name
    assert inspect_source(preparation_id, home=home).registry.state is interrupted_state
    assert cleanup_source(preparation_id, home=home).state is RegistryState.CLEANED


def test_force_changed_captures_divergence_before_removal(
    tmp_path: Path, upstream: GitRepository
) -> None:
    home = tmp_path / "home"
    prepared = prepare_source(source_lock(upstream), home=home, nonce_factory=fixed_nonce)
    (prepared.worktree / "hello.txt").write_text("changed\n", encoding="utf-8")
    git(prepared.worktree, "add", "hello.txt")

    cleaned = cleanup_source(prepared.evidence.preparation_id, home=home, force_changed=True)

    assert cleaned.state is RegistryState.CLEANED
    events = home / "sources" / "registry" / prepared.evidence.preparation_id / "events"
    final_event = json.loads(sorted(events.iterdir())[-1].read_bytes())
    assert final_event["details"]["matches_evidence"] is False
    assert len(final_event["details"]["diff_sha256"]) == 64
    assert final_event["details"]["diff_preview"] == "[CONTENT OMITTED]"
    assert len(final_event["details"]["status_sha256"]) == 64
    assert "status" not in final_event["details"]


def test_cleanup_resumes_after_a_crash_following_worktree_unlock(
    tmp_path: Path, upstream: GitRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    prepared = prepare_source(source_lock(upstream), home=home, nonce_factory=fixed_nonce)
    original_git_run = source_module._git_run

    def crash_after_unlock(*args: object, **kwargs: object) -> str:
        result = original_git_run(*args, **kwargs)  # type: ignore[arg-type]
        arguments = args[1]
        if "unlock" in arguments:  # type: ignore[operator]
            raise SourceTransitionInterrupt("after-unlock")
        return result

    monkeypatch.setattr(source_module, "_git_run", crash_after_unlock)
    with pytest.raises(SourceTransitionInterrupt, match="after-unlock"):
        cleanup_source(prepared.evidence.preparation_id, home=home)
    monkeypatch.setattr(source_module, "_git_run", original_git_run)

    inspection = inspect_source(prepared.evidence.preparation_id, home=home)
    assert inspection.registry.state is RegistryState.CLEANUP_STARTED
    assert (
        cleanup_source(prepared.evidence.preparation_id, home=home).state is RegistryState.CLEANED
    )


def test_force_changed_allows_initialized_submodule_divergence(tmp_path: Path) -> None:
    child = create_repository(tmp_path, "child", "child\n")
    parent = create_repository(tmp_path, "parent", "parent\n")
    git(
        parent.path,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child.path),
        "deps/child",
    )
    git(parent.path, "commit", "-am", "add pinned child")
    parent = GitRepository(parent.path, git(parent.path, "rev-parse", "HEAD"))
    home = tmp_path / "home"
    prepared = prepare_source(
        source_lock(parent, submodules=True),
        home=home,
        nonce_factory=fixed_nonce,
    )
    (prepared.worktree / "deps" / "child" / "hello.txt").write_text("changed\n", encoding="utf-8")

    cleaned = cleanup_source(prepared.evidence.preparation_id, home=home, force_changed=True)

    assert cleaned.state is RegistryState.CLEANED
    events = home / "sources" / "registry" / prepared.evidence.preparation_id / "events"
    final_event = json.loads(sorted(events.iterdir())[-1].read_bytes())
    observed = final_event["details"]["submodules"][0]
    assert observed["matches_evidence"] is False
    assert observed["status_preview"] == "[CONTENT OMITTED]"
    assert len(observed["status_sha256"]) == 64


def test_force_changed_rejects_a_submodule_replaced_by_an_external_symlink(
    tmp_path: Path,
) -> None:
    child = create_repository(tmp_path, "child", "child\n")
    external = create_repository(tmp_path, "external", "external\n")
    parent = create_repository(tmp_path, "parent", "parent\n")
    git(
        parent.path,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child.path),
        "deps/child",
    )
    git(parent.path, "commit", "-am", "add pinned child")
    parent = GitRepository(parent.path, git(parent.path, "rev-parse", "HEAD"))
    home = tmp_path / "home"
    prepared = prepare_source(
        source_lock(parent, submodules=True),
        home=home,
        nonce_factory=fixed_nonce,
    )
    prepared_child = prepared.worktree / "deps" / "child"
    shutil.rmtree(prepared_child)
    prepared_child.symlink_to(external.path, target_is_directory=True)

    with pytest.raises(SourceDivergedError, match="symbolic link"):
        cleanup_source(prepared.evidence.preparation_id, home=home, force_changed=True)

    assert (external.path / "hello.txt").read_text(encoding="utf-8") == "external\n"


def test_submodules_remain_uninitialized_when_lock_disables_them(tmp_path: Path) -> None:
    child = create_repository(tmp_path, "child")
    parent = create_repository(tmp_path, "parent")
    git(
        parent.path,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child.path),
        "deps/child",
    )
    git(parent.path, "commit", "-am", "add child")
    parent = GitRepository(parent.path, git(parent.path, "rev-parse", "HEAD"))

    home = tmp_path / "home"
    prepared = prepare_source(
        source_lock(parent, submodules=False),
        home=home,
        nonce_factory=fixed_nonce,
    )
    assert not (prepared.worktree / "deps" / "child" / "hello.txt").exists()
    assert prepared.evidence.submodules[0].path == "deps/child"
    assert prepared.evidence.submodules[0].locator_sha256 is None
    cleanup_source(prepared.evidence.preparation_id, home=home)


def test_remote_submodule_evidence_retains_both_portable_locator_and_digest() -> None:
    locator = "https://example.test/organization/child.git"
    evidence = source_module.SubmoduleEvidenceV2(
        path="deps/child",
        commit="a" * 40,
        locator=locator,
        locator_sha256=hashlib.sha256(locator.encode()).hexdigest(),
    )

    assert evidence.locator == locator
    assert evidence.locator_sha256 is not None


def test_v1_uninitialized_submodule_evidence_remains_inspectable_and_cleanable(
    tmp_path: Path,
) -> None:
    child = create_repository(tmp_path, "legacy-child")
    parent = create_repository(tmp_path, "legacy-parent")
    git(
        parent.path,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child.path),
        "deps/child",
    )
    git(parent.path, "commit", "-am", "add child")
    parent = GitRepository(parent.path, git(parent.path, "rev-parse", "HEAD"))
    home = tmp_path / "home"
    prepared = prepare_source(
        source_lock(parent, submodules=False),
        home=home,
        nonce_factory=fixed_nonce,
    )
    evidence_path = prepared.record / "evidence.json"
    evidence = json.loads(evidence_path.read_bytes())
    evidence["schema_version"] = 1
    evidence["submodules"][0]["locator_sha256"] = hashlib.sha256(b"uninitialized").hexdigest()
    legacy_bytes = canonical_json_bytes(evidence)
    evidence_path.write_bytes(legacy_bytes)
    current_path = home / "sources" / "registry" / prepared.evidence.preparation_id / "current.json"
    current = json.loads(current_path.read_bytes())
    current["evidence_sha256"] = hashlib.sha256(legacy_bytes).hexdigest()
    current_path.write_bytes(canonical_json_bytes(current))

    inspected = inspect_source(prepared.evidence.preparation_id, home=home)
    assert inspected.evidence is not None
    assert inspected.evidence.schema_version == 1
    assert (
        cleanup_source(prepared.evidence.preparation_id, home=home).state is RegistryState.CLEANED
    )


def test_v1_initialized_submodule_evidence_rejects_the_legacy_sentinel(
    tmp_path: Path,
) -> None:
    child = create_repository(tmp_path, "initialized-child")
    parent = create_repository(tmp_path, "initialized-parent")
    git(
        parent.path,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child.path),
        "deps/child",
    )
    git(parent.path, "commit", "-am", "add child")
    parent = GitRepository(parent.path, git(parent.path, "rev-parse", "HEAD"))
    home = tmp_path / "home"
    prepared = prepare_source(
        source_lock(parent, submodules=True),
        home=home,
        nonce_factory=fixed_nonce,
    )
    evidence_path = prepared.record / "evidence.json"
    current_path = home / "sources" / "registry" / prepared.evidence.preparation_id / "current.json"
    original_evidence = evidence_path.read_bytes()
    original_current = current_path.read_bytes()
    evidence = json.loads(original_evidence)
    evidence["schema_version"] = 1
    evidence["submodules"][0]["locator_sha256"] = hashlib.sha256(b"uninitialized").hexdigest()
    legacy_bytes = canonical_json_bytes(evidence)
    evidence_path.write_bytes(legacy_bytes)
    current = json.loads(original_current)
    current["evidence_sha256"] = hashlib.sha256(legacy_bytes).hexdigest()
    current_path.write_bytes(canonical_json_bytes(current))

    with pytest.raises(SourcePolicyError, match="initialization evidence is inconsistent"):
        inspect_source(prepared.evidence.preparation_id, home=home)

    evidence_path.write_bytes(original_evidence)
    current_path.write_bytes(original_current)
    cleanup_source(prepared.evidence.preparation_id, home=home)


def test_existing_mirror_rejects_hostile_local_configuration(
    tmp_path: Path, upstream: GitRepository
) -> None:
    home = tmp_path / "home"
    prepared = prepare_source(source_lock(upstream), home=home, nonce_factory=fixed_nonce)
    cleanup_source(prepared.evidence.preparation_id, home=home)
    mirror = home / "sources" / "mirrors" / "fixture.git"
    git(home, "--git-dir", str(mirror), "config", "core.sshCommand", "malicious")

    with pytest.raises(SourcePolicyError, match="mirror-config-forbidden"):
        prepare_source(
            source_lock(upstream),
            home=home,
            nonce_factory=lambda: "f" * 32,
        )


def test_ssh_locator_requires_explicit_trust_before_network_access(
    tmp_path: Path, upstream: GitRepository
) -> None:
    lock = source_lock(upstream).model_copy(update={"url": "ssh://git@example.test/repository.git"})

    with pytest.raises(SourceCommandError, match="ssh-auth-unconfigured"):
        prepare_source(lock, home=tmp_path / "home", nonce_factory=fixed_nonce)


def test_source_lock_serializes_concurrent_cleanup(tmp_path: Path, upstream: GitRepository) -> None:
    home = tmp_path / "home"
    prepared = prepare_source(source_lock(upstream), home=home, nonce_factory=fixed_nonce)
    lock_path = home / "locks" / "source-fixture.lock"

    with exclusive_lock(lock_path) as held:
        assert held.acquired
        from strixlab.sources import SourceBusyError

        with pytest.raises(SourceBusyError):
            cleanup_source(prepared.evidence.preparation_id, home=home)

    cleanup_source(prepared.evidence.preparation_id, home=home)


def test_source_lease_authenticates_candidate_and_blocks_cleanup(
    tmp_path: Path, upstream: GitRepository
) -> None:
    home = tmp_path / "home"
    prepared = prepare_source(source_lock(upstream), home=home, nonce_factory=fixed_nonce)
    records = home / "sources" / "records"

    with lease_source(prepared.evidence.preparation_id, home=home) as lease:
        assert lease.evidence == prepared.evidence
        assert lease.worktree == prepared.worktree
        lease_scratch = tuple(records.glob(".lease-*"))
        assert len(lease_scratch) == 1
        assert stat.S_IMODE(lease_scratch[0].stat().st_mode) == 0o700
        with pytest.raises(source_module.SourceBusyError):
            cleanup_source(prepared.evidence.preparation_id, home=home)

    assert not tuple(records.glob(".lease-*"))
    cleanup_source(prepared.evidence.preparation_id, home=home)


def test_source_lease_removes_private_scratch_when_consumer_fails(
    tmp_path: Path, upstream: GitRepository
) -> None:
    home = tmp_path / "home"
    prepared = prepare_source(source_lock(upstream), home=home, nonce_factory=fixed_nonce)
    records = home / "sources" / "records"

    with (
        pytest.raises(RuntimeError, match="consumer failed"),
        lease_source(prepared.evidence.preparation_id, home=home),
    ):
        assert tuple(records.glob(".lease-*"))
        raise RuntimeError("consumer failed")

    assert not tuple(records.glob(".lease-*"))


def test_source_lease_refuses_repeated_private_scratch_collisions(
    tmp_path: Path, upstream: GitRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    prepared = prepare_source(source_lock(upstream), home=home, nonce_factory=fixed_nonce)
    monkeypatch.setattr(source_module.secrets, "token_hex", lambda _size: "c" * 32)
    collision = (
        home / "sources" / "records" / f".lease-{prepared.evidence.preparation_id}-{'c' * 32}"
    )
    collision.mkdir(mode=0o700)

    with (
        pytest.raises(source_module.SourcePolicyError, match="unable to allocate unique"),
        lease_source(prepared.evidence.preparation_id, home=home),
    ):
        pytest.fail("colliding lease scratch must not be reused")

    assert collision.is_dir()


def test_transition_log_corruption_is_rejected(tmp_path: Path, upstream: GitRepository) -> None:
    home = tmp_path / "home"
    prepared = prepare_source(source_lock(upstream), home=home, nonce_factory=fixed_nonce)
    first_event = (
        home / "sources" / "registry" / prepared.evidence.preparation_id / "events" / "0001.json"
    )
    event = json.loads(first_event.read_bytes())
    event["sequence"] = 2
    first_event.write_bytes(canonical_json_bytes(event))

    with pytest.raises(SourcePolicyError, match="event chain"):
        inspect_source(prepared.evidence.preparation_id, home=home)


def test_patch_count_is_bounded_before_files_are_opened(
    tmp_path: Path, upstream: GitRepository
) -> None:
    patch = tmp_path / "candidate.patch"
    patch.write_bytes(b"")

    with pytest.raises(SourcePolicyError, match="patch count exceeds 64"):
        prepare_source(
            source_lock(upstream),
            home=tmp_path / "home",
            patches=[patch] * 65,
            nonce_factory=fixed_nonce,
        )


def test_patch_paths_with_spaces_are_validated_from_raw_nul_records(
    tmp_path: Path, upstream: GitRepository
) -> None:
    path = upstream.path / "space name.txt"
    path.write_text("before\n", encoding="utf-8")
    git(upstream.path, "add", "space name.txt")
    git(upstream.path, "commit", "-m", "add spaced path")
    upstream = GitRepository(upstream.path, git(upstream.path, "rev-parse", "HEAD"))
    path.write_text("after\n", encoding="utf-8")
    patch = tmp_path / "candidate.patch"
    patch.write_text(git(upstream.path, "diff", "--binary") + "\n", encoding="utf-8")
    git(upstream.path, "restore", "space name.txt")

    home = tmp_path / "home"
    prepared = prepare_source(
        source_lock(upstream),
        home=home,
        patches=[patch],
        nonce_factory=fixed_nonce,
    )

    assert (prepared.worktree / "space name.txt").read_text(encoding="utf-8") == "after\n"
    cleanup_source(prepared.evidence.preparation_id, home=home)


@pytest.mark.parametrize(
    "corruption",
    ["request", "evidence-digest", "patch", "diff", "identity"],
)
def test_inspection_rejects_each_portable_integrity_boundary(
    tmp_path: Path, upstream: GitRepository, corruption: str
) -> None:
    patch = tmp_path / "candidate.patch"
    patch.write_text(
        """diff --git a/hello.txt b/hello.txt
--- a/hello.txt
+++ b/hello.txt
@@ -1 +1 @@
-baseline
+candidate
""",
        encoding="utf-8",
    )
    home = tmp_path / "home"
    prepared = prepare_source(
        source_lock(upstream),
        home=home,
        patches=[patch],
        nonce_factory=fixed_nonce,
    )
    registry_directory = home / "sources" / "registry" / prepared.evidence.preparation_id

    if corruption == "request":
        request_path = registry_directory / "request.json"
        request = json.loads(request_path.read_bytes())
        request["request_digest"] = "0" * 64
        request_path.write_bytes(canonical_json_bytes(request))
        message = "request identity"
    elif corruption == "patch":
        (prepared.record / "patch-001.patch").write_bytes(b"changed")
        message = "patch integrity"
    elif corruption == "diff":
        (prepared.record / "candidate.diff").write_bytes(b"changed")
        message = "diff integrity"
    else:
        evidence_path = prepared.record / "evidence.json"
        evidence = json.loads(evidence_path.read_bytes())
        if corruption == "identity":
            evidence["candidate_id"] = "candidate-sha256:" + "0" * 64
            message = "source identity"
        else:
            evidence["created_at"] = "changed"
            message = "evidence digest"
        content = canonical_json_bytes(evidence)
        evidence_path.write_bytes(content)
        if corruption == "identity":
            current_path = registry_directory / "current.json"
            current = json.loads(current_path.read_bytes())
            current["evidence_sha256"] = hashlib.sha256(content).hexdigest()
            current_path.write_bytes(canonical_json_bytes(current))

    with pytest.raises(SourcePolicyError, match=message):
        inspect_source(prepared.evidence.preparation_id, home=home)


@pytest.mark.parametrize("admin_field", ["locked", "gitdir", "HEAD"])
def test_cleanup_rejects_changed_admin_ownership_metadata(
    tmp_path: Path, upstream: GitRepository, admin_field: str
) -> None:
    home = tmp_path / "home"
    prepared = prepare_source(source_lock(upstream), home=home, nonce_factory=fixed_nonce)
    owner = inspect_source(prepared.evidence.preparation_id, home=home).registry.ownership
    assert owner is not None
    (Path(owner.admin_path) / admin_field).write_text("changed\n", encoding="utf-8")

    with pytest.raises(SourceDivergedError):
        cleanup_source(prepared.evidence.preparation_id, home=home)


def test_cleanup_rebinds_forged_ownership_to_the_derived_layout(
    tmp_path: Path, upstream: GitRepository
) -> None:
    home = tmp_path / "home"
    prepared = prepare_source(source_lock(upstream), home=home, nonce_factory=fixed_nonce)
    registry_directory = home / "sources" / "registry" / prepared.evidence.preparation_id
    current_path = registry_directory / "current.json"
    current = json.loads(current_path.read_bytes())
    unrelated = tmp_path / "unrelated-admin"
    unrelated.mkdir()
    current["ownership"]["admin_path"] = str(unrelated)
    current_path.write_bytes(canonical_json_bytes(current))
    (registry_directory / "owner.json").write_bytes(canonical_json_bytes(current["ownership"]))

    with pytest.raises(SourceDivergedError, match="not bound"):
        cleanup_source(prepared.evidence.preparation_id, home=home)

    assert unrelated.is_dir()


def test_cleanup_rejects_unowned_and_unsafe_recovery_artifacts(
    tmp_path: Path, upstream: GitRepository
) -> None:
    home = tmp_path / "home"

    def interrupt(state: RegistryState) -> None:
        if state is RegistryState.MIRROR_READY:
            raise SourceTransitionInterrupt(state)

    with pytest.raises(SourceTransitionInterrupt):
        prepare_source(
            source_lock(upstream),
            home=home,
            nonce_factory=fixed_nonce,
            transition_hook=interrupt,
        )
    preparation_id = next((home / "sources" / "registry").iterdir()).name
    worktree = home / "sources" / "worktrees" / preparation_id
    worktree.mkdir()

    with pytest.raises(SourceDivergedError, match="unowned"):
        cleanup_source(preparation_id, home=home)


def test_invalid_cleanup_id_is_rejected_without_state_access(tmp_path: Path) -> None:
    with pytest.raises(SourcePolicyError, match="invalid preparation ID"):
        cleanup_source("not-a-preparation", home=tmp_path)

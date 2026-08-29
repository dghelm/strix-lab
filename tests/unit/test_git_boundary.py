from __future__ import annotations

import socket
import stat
from pathlib import Path

import pytest

from strixlab.git_boundary import GitBoundary, GitBoundaryError, SshTrust


def boundary(tmp_path: Path, locator: str, trust: SshTrust | None = None) -> GitBoundary:
    home = tmp_path / "git-home"
    scratch = tmp_path / "scratch"
    home.mkdir(parents=True)
    scratch.mkdir()
    return GitBoundary.create(
        git_home=home,
        scratch=scratch,
        locator=locator,
        ssh_trust=trust,
    )


def test_boundary_runs_with_an_explicit_hermetic_environment(tmp_path: Path) -> None:
    git = boundary(tmp_path, "/srv/git/repository")

    result = git.run(["--version"], cwd=tmp_path)

    assert result.returncode == 0
    assert result.stdout.startswith("git version ")
    assert git.environment["GIT_ALLOW_PROTOCOL"] == "file"
    assert git.environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert git.for_locator("https://example.test/repository")["GIT_ALLOW_PROTOCOL"] == "https"
    with pytest.raises(GitBoundaryError, match="ssh-auth-unconfigured"):
        git.for_locator("git@example.test:organization/repository")


def test_network_operations_require_a_locator_and_local_operations_use_file_gate(
    tmp_path: Path,
) -> None:
    git = boundary(tmp_path, "https://example.test/repository")

    assert git.environment["GIT_ALLOW_PROTOCOL"] == "file"
    assert git.for_locator("https://example.test/repository")["GIT_ALLOW_PROTOCOL"] == "https"
    with pytest.raises(TypeError):
        git.run_network(["--version"], cwd=tmp_path)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        git.bytes_network(["--version"], cwd=tmp_path)  # type: ignore[call-arg]


def test_private_key_ssh_trust_is_copied_and_hardened(tmp_path: Path) -> None:
    known_hosts = tmp_path / "known-hosts"
    known_hosts.write_text("example.test ssh-ed25519 fixture\n", encoding="utf-8")
    private_key = tmp_path / "identity"
    private_key.write_text("private fixture\n", encoding="utf-8")
    private_key.chmod(0o600)

    git = boundary(
        tmp_path,
        "ssh://git@example.test/repository",
        SshTrust(known_hosts=known_hosts, private_key=private_key),
    )

    command = git.environment["GIT_SSH_COMMAND"]
    assert "StrictHostKeyChecking=yes" in command
    assert "BatchMode=yes" in command
    assert "ProxyCommand=none" in command
    copied_key = git.scratch / "ssh-identity"
    assert copied_key.read_bytes() == private_key.read_bytes()
    assert stat.S_IMODE(copied_key.stat().st_mode) == 0o600
    assert "SSH_AUTH_SOCK" not in git.environment


def test_agent_ssh_trust_requires_an_owned_socket(tmp_path: Path) -> None:
    known_hosts = tmp_path / "known-hosts"
    known_hosts.write_text("host key\n", encoding="utf-8")
    selector = tmp_path / "selector.pub"
    selector.write_text("public key\n", encoding="utf-8")
    agent_path = tmp_path / "agent.sock"
    agent = socket.socket(socket.AF_UNIX)
    agent.bind(str(agent_path))
    try:
        git = boundary(
            tmp_path,
            "https://example.test/repository",
            SshTrust(
                known_hosts=known_hosts,
                public_key_selector=selector,
                auth_sock=agent_path,
            ),
        )
        assert git.environment["SSH_AUTH_SOCK"] == str(agent_path)
        assert str(git.scratch / "ssh-agent-selector.pub") in git.environment["GIT_SSH_COMMAND"]
    finally:
        agent.close()


def test_ssh_trust_rejects_missing_or_ambiguous_identity(tmp_path: Path) -> None:
    known_hosts = tmp_path / "known-hosts"
    known_hosts.write_text("host key\n", encoding="utf-8")
    private_key = tmp_path / "identity"
    private_key.write_text("private\n", encoding="utf-8")
    private_key.chmod(0o600)

    with pytest.raises(GitBoundaryError, match="ssh-auth-unconfigured"):
        boundary(tmp_path / "missing", "ssh://git@example.test/repository")
    with pytest.raises(GitBoundaryError, match="exactly one identity mode"):
        boundary(
            tmp_path / "ambiguous",
            "ssh://git@example.test/repository",
            SshTrust(known_hosts=known_hosts),
        )


def test_ssh_trust_rejects_symlinks_and_permissive_private_keys(tmp_path: Path) -> None:
    known_hosts = tmp_path / "known-hosts"
    known_hosts.write_text("host key\n", encoding="utf-8")
    linked_hosts = tmp_path / "linked-hosts"
    linked_hosts.symlink_to(known_hosts)
    private_key = tmp_path / "identity"
    private_key.write_text("private\n", encoding="utf-8")
    private_key.chmod(0o644)

    with pytest.raises(GitBoundaryError, match="unavailable"):
        boundary(
            tmp_path / "linked",
            "ssh://git@example.test/repository",
            SshTrust(known_hosts=linked_hosts, private_key=private_key),
        )
    with pytest.raises(GitBoundaryError, match="permissions"):
        boundary(
            tmp_path / "permissive",
            "ssh://git@example.test/repository",
            SshTrust(known_hosts=known_hosts, private_key=private_key),
        )

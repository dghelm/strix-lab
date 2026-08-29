"""Hermetic Git process and transport boundary for source preparation."""

from __future__ import annotations

import builtins
import os
import secrets
import shlex
import shutil
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from strixlab.process import ProcessOutcome, ProcessResult, run_process
from strixlab.secure_fs import write_exclusive
from strixlab.source_identity import locator_class

_OUTPUT_PREFIX_BYTES = 256 * 1024
_GIT_TIMEOUT_SECONDS = 300.0
_SAFE_CONFIG = {
    "core.bare",
    "core.filemode",
    "core.logallrefupdates",
    "core.repositoryformatversion",
    "remote.origin.url",
}


class GitBoundaryError(RuntimeError):
    """A Git command or trust-boundary check failed."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr


@dataclass(frozen=True, slots=True)
class SshTrust:
    """Explicit SSH trust input; exactly one identity mode is required."""

    known_hosts: Path
    private_key: Path | None = None
    public_key_selector: Path | None = None
    auth_sock: Path | None = None


def _read_owned(path: Path, *, private: bool = False) -> bytes:
    if not path.is_absolute():
        raise GitBoundaryError("SSH trust inputs must be absolute paths")
    flags = os.O_CLOEXEC | os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise GitBoundaryError("SSH trust input is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid():
            raise GitBoundaryError("SSH trust input is not an owned regular file")
        if private and stat.S_IMODE(metadata.st_mode) & 0o077:
            raise GitBoundaryError("SSH private key permissions are broader than 0600")
        chunks = []
        while chunk := os.read(descriptor, 64 * 1024):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _trusted_path() -> str:
    raw = os.environ.get("PATH", os.defpath)
    directories = []
    for value in raw.split(os.pathsep):
        if not value or not Path(value).is_absolute():
            raise GitBoundaryError("PATH contains an empty or relative component")
        try:
            resolved = Path(value).resolve(strict=True)
        except FileNotFoundError:
            continue
        metadata = resolved.stat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise GitBoundaryError("PATH component is not a directory")
        if metadata.st_uid not in {0, os.geteuid()} or stat.S_IMODE(metadata.st_mode) & 0o022:
            raise GitBoundaryError("PATH component is not trusted")
        text = str(resolved)
        if text not in directories:
            directories.append(text)
    return os.pathsep.join(directories)


def _trusted_executable(name: str, path: str) -> str:
    executable = shutil.which(name, path=path)
    if executable is None:
        raise GitBoundaryError(f"required executable is unavailable: {name}")
    resolved = Path(executable).resolve(strict=True)
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise GitBoundaryError(f"required executable is unsafe: {name}")
    if metadata.st_uid not in {0, os.geteuid()} or stat.S_IMODE(metadata.st_mode) & 0o022:
        raise GitBoundaryError(f"required executable is not trusted: {name}")
    return str(resolved)


def _protocol(locator: str) -> str:
    kind = locator_class(locator)
    if kind in {"local-path", "file"}:
        return "file"
    if kind == "https":
        return "https"
    return "ssh"


@dataclass(slots=True)
class GitBoundary:
    executable: str
    environment: dict[str, str]
    scratch: Path

    @classmethod
    def create(
        cls,
        *,
        git_home: Path,
        scratch: Path,
        locator: str,
        ssh_trust: SshTrust | None = None,
    ) -> GitBoundary:
        path = _trusted_path()
        git = _trusted_executable("git", path)
        false_program = _trusted_executable("false", path)
        config = git_home / "global.config"
        if not config.exists():
            write_exclusive(config, b"")
        else:
            config_metadata = config.lstat()
            if (
                stat.S_ISLNK(config_metadata.st_mode)
                or not stat.S_ISREG(config_metadata.st_mode)
                or config_metadata.st_uid != os.geteuid()
                or stat.S_IMODE(config_metadata.st_mode) & 0o022
            ):
                raise GitBoundaryError("owned Git config is unsafe")

        preflight_environment = {
            "GIT_CONFIG_GLOBAL": str(config),
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": str(git_home),
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": path,
            "TZ": "UTC",
        }
        preflight = run_process(
            [git, "--exec-path"],
            cwd=scratch,
            timeout=30,
            inherit_env=False,
            env_overrides=preflight_environment,
        )
        if preflight.outcome is not ProcessOutcome.EXITED or preflight.returncode != 0:
            raise GitBoundaryError("Git exec-path preflight failed")
        exec_path = Path(preflight.stdout.strip()).resolve(strict=True)
        if not exec_path.is_absolute() or not exec_path.is_dir():
            raise GitBoundaryError("Git exec-path is invalid")
        exec_metadata = exec_path.stat()
        if (
            exec_metadata.st_uid not in {0, os.geteuid()}
            or stat.S_IMODE(exec_metadata.st_mode) & 0o022
        ):
            raise GitBoundaryError("Git exec-path is not trusted")

        protocol = _protocol(locator)
        environment = {
            "GCM_INTERACTIVE": "Never",
            "GIT_ALLOW_PROTOCOL": "file",
            "GIT_ASKPASS": false_program,
            "GIT_CONFIG_GLOBAL": str(config),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_EXEC_PATH": str(exec_path),
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "HOME": str(git_home),
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": path,
            "SSH_ASKPASS": false_program,
            "TMPDIR": str(scratch),
            "TZ": "UTC",
        }
        if protocol == "ssh" or ssh_trust is not None:
            environment.update(_prepare_ssh(ssh_trust, scratch, path))
        return cls(git, environment, scratch)

    def for_locator(self, locator: str) -> Mapping[str, str]:
        protocol = _protocol(locator)
        environment = dict(self.environment)
        environment["GIT_ALLOW_PROTOCOL"] = protocol
        if protocol == "ssh" and "GIT_SSH_COMMAND" not in environment:
            raise GitBoundaryError("ssh-auth-unconfigured")
        return environment

    def run(
        self,
        arguments: Sequence[str],
        *,
        cwd: Path,
        stdout_spool: Path | None = None,
        output_limit_bytes: int | None = _OUTPUT_PREFIX_BYTES,
        allowed_returncodes: frozenset[int] = frozenset({0}),
    ) -> ProcessResult:
        return self._run(
            arguments,
            cwd=cwd,
            environment=self.environment,
            stdout_spool=stdout_spool,
            output_limit_bytes=output_limit_bytes,
            allowed_returncodes=allowed_returncodes,
        )

    def run_network(
        self,
        arguments: Sequence[str],
        *,
        cwd: Path,
        locator: str,
        stdout_spool: Path | None = None,
        output_limit_bytes: int | None = _OUTPUT_PREFIX_BYTES,
        allowed_returncodes: frozenset[int] = frozenset({0}),
    ) -> ProcessResult:
        """Run a transport-bearing Git command with an explicit locator gate."""

        return self._run(
            arguments,
            cwd=cwd,
            environment=self.for_locator(locator),
            stdout_spool=stdout_spool,
            output_limit_bytes=output_limit_bytes,
            allowed_returncodes=allowed_returncodes,
        )

    def _run(
        self,
        arguments: Sequence[str],
        *,
        cwd: Path,
        environment: Mapping[str, str],
        stdout_spool: Path | None,
        output_limit_bytes: int | None,
        allowed_returncodes: frozenset[int],
    ) -> ProcessResult:
        result = run_process(
            [
                self.executable,
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "credential.helper=",
                "-c",
                "core.fsmonitor=false",
                *arguments,
            ],
            cwd=cwd,
            timeout=_GIT_TIMEOUT_SECONDS,
            inherit_env=False,
            env_overrides=environment,
            output_limit_bytes=output_limit_bytes,
            stdout_spool=stdout_spool,
            spool_root=self.scratch if stdout_spool is not None else None,
        )
        if result.capture_error is not None:
            raise GitBoundaryError(f"git-capture-failed:{result.capture_error}")
        if (
            result.outcome is not ProcessOutcome.EXITED
            or result.returncode not in allowed_returncodes
        ):
            raise GitBoundaryError(
                f"git-command-failed:{result.returncode}",
                returncode=result.returncode,
                stderr=result.stderr,
            )
        return result

    def bytes(self, arguments: Sequence[str], *, cwd: Path) -> bytes:
        return self._bytes(arguments, cwd=cwd, locator=None)

    def bytes_network(self, arguments: Sequence[str], *, cwd: Path, locator: str) -> builtins.bytes:
        """Capture a transport-bearing Git command with an explicit locator gate."""

        return self._bytes(arguments, cwd=cwd, locator=locator)

    def _bytes(self, arguments: Sequence[str], *, cwd: Path, locator: str | None) -> builtins.bytes:
        spool = self.scratch / f"git-output-{secrets.token_hex(8)}"
        if locator is None:
            result = self.run(
                arguments,
                cwd=cwd,
                stdout_spool=spool,
                output_limit_bytes=0,
            )
        else:
            result = self.run_network(
                arguments,
                cwd=cwd,
                locator=locator,
                stdout_spool=spool,
                output_limit_bytes=0,
            )
        try:
            content = spool.read_bytes()
        finally:
            spool.unlink(missing_ok=True)
        if len(content) != result.stdout_bytes:
            raise GitBoundaryError("git-spool-size-mismatch")
        return content

    def validate_mirror_config(self, mirror: Path, expected_locator: str) -> None:
        content = self.bytes(
            ["--git-dir", str(mirror), "config", "--local", "--null", "--list"],
            cwd=self.scratch,
        )
        seen_origin = False
        for entry in content.rstrip(b"\0").split(b"\0") if content else ():
            try:
                key_bytes, value_bytes = entry.split(b"\n", 1)
                key = key_bytes.decode("utf-8").lower()
                value = value_bytes.decode("utf-8")
            except (UnicodeDecodeError, ValueError) as exc:
                raise GitBoundaryError("mirror-config-invalid") from exc
            if key not in _SAFE_CONFIG:
                raise GitBoundaryError(f"mirror-config-forbidden:{key}")
            if key == "remote.origin.url":
                if seen_origin or value != expected_locator:
                    raise GitBoundaryError("mirror-origin-mismatch")
                seen_origin = True
        if not seen_origin:
            raise GitBoundaryError("mirror-origin-missing")


def _prepare_ssh(trust: SshTrust | None, scratch: Path, path: str) -> dict[str, str]:
    if trust is None:
        raise GitBoundaryError("ssh-auth-unconfigured")
    ssh = _trusted_executable("ssh", path)
    known_hosts = scratch / "ssh-known-hosts"
    write_exclusive(known_hosts, _read_owned(trust.known_hosts))
    private_mode = trust.private_key is not None
    agent_mode = trust.public_key_selector is not None and trust.auth_sock is not None
    if private_mode == agent_mode:
        raise GitBoundaryError("SSH requires exactly one identity mode")
    environment: dict[str, str] = {}
    if private_mode:
        assert trust.private_key is not None
        identity = scratch / "ssh-identity"
        write_exclusive(identity, _read_owned(trust.private_key, private=True))
    else:
        assert trust.public_key_selector is not None
        assert trust.auth_sock is not None
        identity = scratch / "ssh-agent-selector.pub"
        write_exclusive(identity, _read_owned(trust.public_key_selector))
        socket_metadata = trust.auth_sock.lstat()
        if (
            not trust.auth_sock.is_absolute()
            or stat.S_ISLNK(socket_metadata.st_mode)
            or not stat.S_ISSOCK(socket_metadata.st_mode)
            or socket_metadata.st_uid != os.geteuid()
        ):
            raise GitBoundaryError("SSH_AUTH_SOCK is unsafe")
        environment["SSH_AUTH_SOCK"] = str(trust.auth_sock)
    command = [
        ssh,
        "-F",
        "/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "ProxyCommand=none",
        "-o",
        "PermitLocalCommand=no",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "RequestTTY=no",
        "-o",
        "NumberOfPasswordPrompts=0",
        "-o",
        "PreferredAuthentications=publickey",
        "-i",
        str(identity),
    ]
    environment["GIT_SSH_COMMAND"] = shlex.join(command)
    return environment

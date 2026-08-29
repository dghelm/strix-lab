from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path

import pytest

from strixlab.process import ProcessOutcome, run_process


def command(source: str) -> list[str]:
    return [sys.executable, "-c", source]


def test_spooling_preserves_full_exact_streams_while_memory_is_bounded(tmp_path: Path) -> None:
    stdout = b"alpha\x00\xff" * 100_000
    stderr = b"omega\x00\xfe" * 80_000
    stdout_spool = tmp_path / "stdout.bin"
    stderr_spool = tmp_path / "stderr.bin"
    result = run_process(
        command(
            "import os; "
            "os.write(1, b'alpha\\x00\\xff' * 100_000); "
            "os.write(2, b'omega\\x00\\xfe' * 80_000)"
        ),
        cwd=tmp_path,
        output_limit_bytes=31,
        stdout_spool=stdout_spool,
        stderr_spool=stderr_spool,
        spool_root=tmp_path,
    )

    assert result.outcome is ProcessOutcome.EXITED
    assert result.stdout_bytes == len(stdout)
    assert result.stderr_bytes == len(stderr)
    assert result.stdout_sha256 == hashlib.sha256(stdout).hexdigest()
    assert result.stderr_sha256 == hashlib.sha256(stderr).hexdigest()
    assert result.stdout_spool == stdout_spool
    assert result.stderr_spool == stderr_spool
    assert stdout_spool.read_bytes() == stdout
    assert stderr_spool.read_bytes() == stderr
    assert result.stdout_truncated is result.stderr_truncated is True
    assert result.capture_error is None
    assert stat.S_IMODE(stdout_spool.stat().st_mode) == 0o600
    assert stat.S_IMODE(stderr_spool.stat().st_mode) == 0o600


def test_hashes_cover_unbounded_streams_without_spooling(tmp_path: Path) -> None:
    result = run_process(command("print('exact')"), cwd=tmp_path)

    assert result.stdout_bytes == len(b"exact\n")
    assert result.stderr_bytes == 0
    assert result.stdout_sha256 == hashlib.sha256(b"exact\n").hexdigest()
    assert result.stderr_sha256 == hashlib.sha256(b"").hexdigest()
    assert result.stdout_spool is result.stderr_spool is None


def test_spool_paths_must_be_distinct_and_new(tmp_path: Path) -> None:
    path = tmp_path / "capture"
    with pytest.raises(ValueError, match="different"):
        run_process(
            command("pass"),
            cwd=tmp_path,
            stdout_spool=path,
            stderr_spool=path,
            spool_root=tmp_path,
        )

    path.write_text("owned by caller", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_process(command("pass"), cwd=tmp_path, stdout_spool=path, spool_root=tmp_path)
    assert path.read_text(encoding="utf-8") == "owned by caller"


def test_second_spool_open_failure_removes_the_first_created_file(tmp_path: Path) -> None:
    stdout = tmp_path / "stdout"
    stderr = tmp_path / "stderr"
    stderr.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        run_process(
            command("pass"),
            cwd=tmp_path,
            stdout_spool=stdout,
            stderr_spool=stderr,
            spool_root=tmp_path,
        )

    assert not stdout.exists()
    assert stderr.read_text(encoding="utf-8") == "existing"


def test_spool_parent_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_process(
            command("pass"),
            cwd=tmp_path,
            stdout_spool=tmp_path / "missing" / "stdout",
            spool_root=tmp_path,
        )


def test_spawn_failure_still_produces_verifiable_empty_spools(tmp_path: Path) -> None:
    stdout_spool = tmp_path / "stdout"
    stderr_spool = tmp_path / "stderr"
    result = run_process(
        ["/definitely/not/a/command"],
        cwd=tmp_path,
        stdout_spool=stdout_spool,
        stderr_spool=stderr_spool,
        spool_root=tmp_path,
    )

    assert result.outcome is ProcessOutcome.SPAWN_FAILED
    assert stdout_spool.read_bytes() == stderr_spool.read_bytes() == b""
    assert result.stdout_sha256 == result.stderr_sha256 == hashlib.sha256(b"").hexdigest()
    assert result.capture_error is None


def test_spooling_requires_an_explicit_owned_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="spool_root is required"):
        run_process(command("pass"), cwd=tmp_path, stdout_spool=tmp_path / "stdout")

    outside = tmp_path.parent / "outside-spool"
    with pytest.raises(ValueError, match="beneath spool_root"):
        run_process(
            command("pass"),
            cwd=tmp_path,
            stdout_spool=outside,
            spool_root=tmp_path,
        )


def test_spooling_rejects_a_symlinked_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="unsafe"):
        run_process(
            command("pass"),
            cwd=tmp_path,
            stdout_spool=linked / "stdout",
            spool_root=tmp_path,
        )


def test_publication_failure_never_exposes_a_partial_final_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "stdout"

    def fail_link(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected link failure")

    monkeypatch.setattr("strixlab.process.os.link", fail_link)
    result = run_process(
        command("print('captured')"),
        cwd=tmp_path,
        stdout_spool=final,
        spool_root=tmp_path,
    )

    assert result.outcome is ProcessOutcome.EXITED
    assert result.capture_error == "stdout:spool-publish-failed"
    assert result.stdout_spool is None
    assert not final.exists()
    assert not tuple(tmp_path.glob(".stdout.tmp-*"))


def test_spool_open_failure_terminates_the_process_and_reports_capture_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "stdout"
    real_open = os.open

    def fail_temporary_open(path: object, *args: object, **kwargs: object) -> int:
        if ".stdout.tmp-" in str(path):
            raise OSError("injected open failure")
        return real_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("strixlab.process.os.open", fail_temporary_open)
    result = run_process(
        command("import time; time.sleep(30)"),
        cwd=tmp_path,
        stdout_spool=final,
        spool_root=tmp_path,
    )

    assert result.outcome is ProcessOutcome.CAPTURE_FAILED
    assert result.capture_error == "stdout:spool-open-failed"
    assert result.stdout_spool is None
    assert not final.exists()


def test_spool_fsync_failure_discards_the_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "stdout"
    real_fsync = os.fsync

    def fail_regular_file_fsync(descriptor: int) -> None:
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("injected fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr("strixlab.process.os.fsync", fail_regular_file_fsync)
    result = run_process(
        command("print('captured')"),
        cwd=tmp_path,
        stdout_spool=final,
        spool_root=tmp_path,
    )

    assert result.capture_error == "stdout:spool-fsync-failed"
    assert result.stdout_spool is None
    assert not final.exists()

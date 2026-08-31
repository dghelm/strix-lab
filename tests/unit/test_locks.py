from __future__ import annotations

import threading
import time
from pathlib import Path

from strixlab.locks import LockStatus, exclusive_lock, wait_for_exclusive_lock


def test_wait_acquires_immediately_when_uncontended(tmp_path: Path) -> None:
    lock = tmp_path / "a.lock"
    with wait_for_exclusive_lock(lock, timeout=1.0) as attempt:
        assert attempt.acquired


def test_wait_times_out_on_sustained_contention(tmp_path: Path) -> None:
    lock = tmp_path / "b.lock"
    with exclusive_lock(lock) as held:
        assert held.acquired
        start = time.monotonic()
        with wait_for_exclusive_lock(lock, timeout=0.3, poll_interval=0.02) as attempt:
            assert not attempt.acquired
            assert attempt.status is LockStatus.CONTENDED
        assert time.monotonic() - start >= 0.3


def test_wait_succeeds_once_the_holder_releases(tmp_path: Path) -> None:
    lock = tmp_path / "c.lock"
    released = threading.Event()

    def hold_briefly() -> None:
        with exclusive_lock(lock) as held:
            assert held.acquired
            time.sleep(0.15)
        released.set()

    holder = threading.Thread(target=hold_briefly)
    holder.start()
    time.sleep(0.02)
    with wait_for_exclusive_lock(lock, timeout=2.0, poll_interval=0.02) as attempt:
        assert attempt.acquired
        assert released.is_set()
    holder.join()


def test_wait_reports_unavailable_immediately(tmp_path: Path) -> None:
    # A lock whose parent directory does not exist is unavailable, not contended.
    missing_parent = tmp_path / "absent" / "d.lock"
    start = time.monotonic()
    with wait_for_exclusive_lock(missing_parent, timeout=5.0) as attempt:
        assert not attempt.acquired
        assert attempt.status is LockStatus.UNAVAILABLE
    assert time.monotonic() - start < 1.0

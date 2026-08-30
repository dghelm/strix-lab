"""Shared, cycle-free layout of the on-disk build storage tree under a home.

Both the build-attempt control plane (:mod:`strixlab.builds`) and the durable
build cache (:mod:`strixlab.build_cache`) key their state off the same physical
directories. Spelling those roots once here keeps the two subsystems from
drifting apart; each still applies its own ownership and existence policy.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from strixlab.secure_fs import ensure_directory_fsynced


def is_unsafe_directory(metadata: os.stat_result) -> bool:
    """Return True when a stat result is not a real, current-user-owned directory.

    Rejects symlinks, non-directories, and directories owned by another user —
    the exact predicate both the build control plane and the build cache apply
    to every StrixLab-owned storage directory.
    """

    return (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
    )


@dataclass(frozen=True, slots=True)
class BuildStorageRoots:
    home: Path
    root: Path
    records: Path
    attempts: Path
    attempt_records: Path
    success_records: Path
    materialized: Path
    snapshots: Path
    indexes: Path
    recipe_indexes: Path
    build_indexes: Path
    materializations: Path
    staging: Path
    locks: Path


def build_storage_roots(home: Path) -> BuildStorageRoots:
    """Compute every build storage root beneath an absolute StrixLab home."""

    if not home.is_absolute():
        raise ValueError("StrixLab home must be absolute")
    root = home / "builds"
    records = root / "records"
    indexes = root / "indexes"
    return BuildStorageRoots(
        home=home,
        root=root,
        records=records,
        attempts=root / "attempts",
        attempt_records=records / "attempts",
        success_records=records / "success",
        materialized=root / "materialized",
        snapshots=root / "snapshots",
        indexes=indexes,
        recipe_indexes=indexes / "recipes",
        build_indexes=indexes / "builds",
        materializations=root / "materializations",
        staging=root / "publication-staging",
        locks=home / "locks",
    )


def storage_root_dirs(roots: BuildStorageRoots) -> tuple[Path, ...]:
    """Every storage directory in parent-before-child order.

    The shared, cycle-free creation order used by both subsystems: each entry's
    parent appears earlier, so a single non-recursive `mkdir` per entry (no
    `parents=True`) always has an existing parent. `home` is excluded — it is the
    externally supplied root each subsystem validates on its own terms.
    """

    return (
        roots.root,
        roots.records,
        roots.attempts,
        roots.attempt_records,
        roots.success_records,
        roots.materialized,
        roots.snapshots,
        roots.indexes,
        roots.recipe_indexes,
        roots.build_indexes,
        roots.materializations,
        roots.staging,
        roots.locks,
    )


def prepare_storage_tree(
    roots: BuildStorageRoots, *, create: bool, validate: Callable[[Path], None]
) -> None:
    """Create (when ``create``) and validate the shared storage tree.

    The single procedure both subsystems share: when creating, ``home`` is made if
    absent, then every storage root is exclusively created in parent-before-child
    order with its parent fsynced once on creation; a pre-existing entry is validated
    in place so an unsafe parent fails closed before a child is created beneath it.
    Every entry (``home`` included) is validated afterwards. ``validate(path)`` is the
    caller's domain adapter, raising its own exception type/message on a missing or
    unsafe directory — this function itself raises nothing.
    """

    if create:
        if not roots.home.exists():
            roots.home.mkdir(mode=0o700, parents=True)
        for path in storage_root_dirs(roots):
            if not ensure_directory_fsynced(path):
                validate(path)
    for path in (roots.home, *storage_root_dirs(roots)):
        validate(path)

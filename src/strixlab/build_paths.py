"""Shared, cycle-free layout of the on-disk build storage tree under a home.

Both the build-attempt control plane (:mod:`strixlab.builds`) and the durable
build cache (:mod:`strixlab.build_cache`) key their state off the same physical
directories. Spelling those roots once here keeps the two subsystems from
drifting apart; each still applies its own ownership and existence policy.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from strixlab import secure_fs


def prepare_directory_tree(
    home: Path,
    dirs: Sequence[Path],
    *,
    create: bool,
    validate: Callable[[Path], None],
) -> None:
    """Create (when ``create``) and validate a parent-before-child directory sequence.

    The single storage-preparation procedure shared by every StrixLab storage tree.
    When creating, ``home`` is made if absent and validated **before** any child is
    created, then each directory in ``dirs`` (in parent-before-child order) is created
    descriptor-relative under its already-validated parent's held, no-follow directory
    descriptor and validated in turn — so a symlinked or foreign home/parent is caught
    before anything is created through or beneath it, and a child is never created by
    re-resolving a mutable ancestor by pathname. Every entry (``home`` included) is
    validated again afterwards. ``validate(path)`` is the caller's domain adapter,
    raising its own exception type/message on a missing or unsafe directory.
    """

    if create:
        _create_directory_tree(home, dirs, validate)
    for path in (home, *dirs):
        validate(path)


def _create_directory_tree(
    home: Path, dirs: Sequence[Path], validate: Callable[[Path], None]
) -> None:
    if not home.exists():
        home.mkdir(mode=0o700, parents=True)
    validate(home)
    held: dict[Path, int] = {}
    try:
        held[home] = secure_fs.open_owned_directory(home)
        for path in dirs:
            parent_fd = held[path.parent]
            name = path.name
            created = True
            try:
                os.mkdir(name, 0o700, dir_fd=parent_fd)
            except FileExistsError:
                created = False
            # Domain-validate the child (symlink/owner/mode) before descending into it,
            # then hold its no-follow descriptor so its own children are created under a
            # verified, non-mutable directory. The parent flush (durability only, not the
            # security boundary) makes the new entry durable.
            validate(path)
            held[path] = secure_fs.open_owned_directory(name, dir_fd=parent_fd)
            if created:
                secure_fs.fsync_directory(path.parent)
    finally:
        for fd in held.values():
            os.close(fd)


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
    """Create (when ``create``) and validate the build storage tree.

    A thin wrapper binding the build storage root sequence to the shared
    :func:`prepare_directory_tree` procedure.
    """

    prepare_directory_tree(roots.home, storage_root_dirs(roots), create=create, validate=validate)

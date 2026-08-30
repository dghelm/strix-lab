"""Shared, cycle-free layout of the on-disk run-evidence tree under a home.

The run-evidence subsystem (:mod:`strixlab.evidence`) and the bundle skeleton
(:mod:`strixlab.bundles`) key their state off the same physical directories.
Spelling those roots once here keeps them from drifting apart; each caller still
applies its own ownership and existence policy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from strixlab.build_paths import prepare_directory_tree


@dataclass(frozen=True, slots=True)
class RunStorageRoots:
    home: Path
    root: Path
    allocation_staging: Path
    active: Path
    records: Path
    indexes: Path
    locks: Path
    bundle_staging: Path


def run_storage_roots(home: Path) -> RunStorageRoots:
    """Compute every run storage root beneath an absolute StrixLab home."""

    if not home.is_absolute():
        raise ValueError("StrixLab home must be absolute")
    root = home / "runs"
    return RunStorageRoots(
        home=home,
        root=root,
        allocation_staging=root / "allocation-staging",
        active=root / "active",
        records=root / "records",
        indexes=root / "indexes",
        locks=root / "locks",
        bundle_staging=root / "bundle-staging",
    )


def run_storage_dirs(roots: RunStorageRoots) -> tuple[Path, ...]:
    """Every run storage directory in parent-before-child order.

    ``home`` is excluded; it is the externally supplied root each caller validates
    on its own terms. Each entry's parent appears earlier so a single non-recursive
    ``mkdir`` per entry always has an existing parent.
    """

    return (
        roots.root,
        roots.allocation_staging,
        roots.active,
        roots.records,
        roots.indexes,
        roots.locks,
        roots.bundle_staging,
    )


def prepare_run_storage(
    roots: RunStorageRoots, *, create: bool, validate: Callable[[Path], None]
) -> None:
    """Create (when ``create``) and validate the run storage tree.

    A thin wrapper binding the run storage root sequence to the shared
    :func:`strixlab.build_paths.prepare_directory_tree` procedure.
    """

    prepare_directory_tree(roots.home, run_storage_dirs(roots), create=create, validate=validate)

"""Side-effect-free path resolution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from os import PathLike
from pathlib import Path

from platformdirs import user_data_path

STRIXLAB_HOME_ENV = "STRIXLAB_HOME"


def resolve_home(
    explicit: str | PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve StrixLab's state root without creating it.

    Explicit and environment-provided paths must be absolute after Linux user
    expansion. An explicitly empty value is an error so a future state-writing
    command can never fall back to the current directory accidentally.
    """

    environment = os.environ if environ is None else environ
    if explicit is not None:
        raw = os.fspath(explicit)
    elif STRIXLAB_HOME_ENV in environment:
        raw = environment[STRIXLAB_HOME_ENV]
    else:
        return user_data_path("strixlab", appauthor=False).resolve(strict=False)

    if not raw.strip():
        raise ValueError("StrixLab home cannot be empty or whitespace")
    if "\x00" in raw:
        raise ValueError("StrixLab home cannot contain NUL bytes")

    try:
        path = Path(raw).expanduser()
    except RuntimeError as exc:
        raise ValueError(f"cannot expand StrixLab home: {raw!r}") from exc

    if not path.is_absolute():
        raise ValueError("StrixLab home must be absolute")
    return path.resolve(strict=False)

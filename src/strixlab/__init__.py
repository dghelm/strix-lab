"""StrixLab foundation package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("strixlab")
except PackageNotFoundError:  # pragma: no cover - exercised by a source-only subprocess test
    __version__ = "0+unknown"

__all__ = ["__version__"]

"""StrixLab command-line interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError
from rich.logging import RichHandler

from strixlab import __version__
from strixlab.config import read_manifest
from strixlab.manifests import ManifestRegistry, validate_manifest
from strixlab.schema_registry import schema_resource_bytes

app = typer.Typer(
    help="Evidence-first optimization research tooling for AMD Strix Halo.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
schema_app = typer.Typer(help="Inspect versioned manifest schemas.")
manifest_app = typer.Typer(help="Validate versioned manifests.")
app.add_typer(schema_app, name="schema")
app.add_typer(manifest_app, name="manifest")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


def _manifest_kind_callback(value: str) -> str:
    if value not in ManifestRegistry.kinds():
        choices = ", ".join(ManifestRegistry.kinds())
        raise typer.BadParameter(f"unknown manifest kind {value!r}; choose one of: {choices}")
    return value


@app.callback()
def root(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = None,
) -> None:
    """Run StrixLab."""


@schema_app.command("show")
def schema_show(
    kind: Annotated[
        str,
        typer.Argument(help="Registered manifest kind.", callback=_manifest_kind_callback),
    ],
) -> None:
    """Print the canonical JSON Schema for KIND."""

    try:
        content = schema_resource_bytes(kind).decode("utf-8")
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(content, nl=False)


@manifest_app.command("validate")
def manifest_validate(
    kind: Annotated[
        str,
        typer.Argument(help="Registered manifest kind.", callback=_manifest_kind_callback),
    ],
    path: Annotated[Path, typer.Argument(help="YAML manifest path.")],
) -> None:
    """Validate raw manifest structure without resolving environment values."""

    try:
        value = read_manifest(path)
        validate_manifest(kind, value)
    except (OSError, ValueError, yaml.YAMLError, ValidationError) as exc:
        typer.echo(f"invalid raw {kind} manifest: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"valid raw {kind} manifest: {path}")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure interactive logging at the application boundary."""

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
        force=True,
    )


def main() -> None:
    """Installed console-script entry point."""

    configure_logging()
    app(prog_name="strixlab")

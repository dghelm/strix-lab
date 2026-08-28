"""StrixLab command-line interface."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import ValidationError
from rich.logging import RichHandler

from strixlab import __version__
from strixlab.config import read_manifest
from strixlab.doctor import (
    ReportWriteError,
    SensitiveInterpolationError,
    UnsafeDiagnosticError,
    assert_terminal_text_safe,
    run_doctor,
)
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


def _doctor_echo(message: str, environ: dict[str, str], *, err: bool = False) -> None:
    try:
        assert_terminal_text_safe(message, environ)
    except UnsafeDiagnosticError:
        typer.echo("doctor failed: unable to safely render terminal output", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(message, err=err)


@app.command("doctor")
def doctor(
    machine: Annotated[
        Path,
        typer.Option(
            "--machine",
            help="Resolved machine-profile YAML path.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    home: Annotated[
        Path | None,
        typer.Option("--home", help="Override the StrixLab data home."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Override the complete doctor report path."),
    ] = None,
) -> None:
    """Observe machine readiness without changing machine settings."""

    frozen_environ = dict(os.environ)
    try:
        result = run_doctor(
            machine,
            home=home,
            output=output,
            environ=frozen_environ,
        )
    except ValidationError as exc:
        _doctor_echo("invalid machine profile:", frozen_environ, err=True)
        for error in exc.errors(include_input=False, include_url=False):
            location = ".".join(str(part) for part in error["loc"]) or "manifest"
            _doctor_echo(f"  {location}: {error['msg']}", frozen_environ, err=True)
        raise typer.Exit(code=1) from None
    except SensitiveInterpolationError:
        typer.echo(
            "invalid machine profile: sensitive environment interpolation is forbidden",
            err=True,
        )
        raise typer.Exit(code=1) from None
    except (OSError, yaml.YAMLError):
        typer.echo("doctor failed: unable to read the machine profile", err=True)
        raise typer.Exit(code=1) from None
    except (ReportWriteError, UnsafeDiagnosticError):
        typer.echo("doctor failed: unable to safely publish the report", err=True)
        raise typer.Exit(code=1) from None

    if result.ready:
        _doctor_echo(f"ready: {result.path}", frozen_environ)
        return
    for check in result.report.checks:
        if check.status == "blocker":
            _doctor_echo(
                f"blocker [{check.id}]: {check.message}",
                frozen_environ,
                err=True,
            )
    _doctor_echo(f"report: {result.path}", frozen_environ, err=True)
    raise typer.Exit(code=1)


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

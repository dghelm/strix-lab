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
    RedactionContext,
    ReportWriteError,
    SensitiveInterpolationError,
    UnsafeDiagnosticError,
    run_doctor,
)
from strixlab.git_boundary import SshTrust
from strixlab.manifests import ManifestRegistry, SourceLockV1, validate_manifest
from strixlab.paths import resolve_home
from strixlab.schema_registry import schema_resource_bytes
from strixlab.serialization import canonical_json_bytes
from strixlab.sources import SourceError, cleanup_source, inspect_source, prepare_source

app = typer.Typer(
    help="Evidence-first optimization research tooling for AMD Strix Halo.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
schema_app = typer.Typer(help="Inspect versioned manifest schemas.")
manifest_app = typer.Typer(help="Validate versioned manifests.")
source_app = typer.Typer(help="Prepare and manage isolated Git source worktrees.")
app.add_typer(schema_app, name="schema")
app.add_typer(manifest_app, name="manifest")
app.add_typer(source_app, name="source")


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


@source_app.command("prepare")
def source_prepare(
    manifest: Annotated[
        Path,
        typer.Argument(
            help="Versioned source-lock YAML path.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    patch: Annotated[
        list[Path] | None,
        typer.Option(
            "--patch",
            help="Reviewed patch to stage in order; repeat for multiple patches.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    home: Annotated[
        Path | None,
        typer.Option("--home", help="Override the StrixLab data home."),
    ] = None,
    ssh_known_hosts: Annotated[
        Path | None,
        typer.Option("--ssh-known-hosts", help="Owned SSH known-hosts file."),
    ] = None,
    ssh_private_key: Annotated[
        Path | None,
        typer.Option("--ssh-private-key", help="Owned mode-0600 SSH private key."),
    ] = None,
    ssh_public_key: Annotated[
        Path | None,
        typer.Option("--ssh-public-key", help="Public-key selector for agent authentication."),
    ] = None,
    ssh_auth_sock: Annotated[
        Path | None,
        typer.Option("--ssh-auth-sock", help="Owned SSH agent socket."),
    ] = None,
) -> None:
    """Prepare a pinned detached worktree and immutable source evidence."""

    try:
        value = read_manifest(manifest)
        validated = validate_manifest("source-lock", value)
        if not isinstance(validated, SourceLockV1):
            raise TypeError("source-lock registry returned the wrong model")
        ssh_values = (ssh_known_hosts, ssh_private_key, ssh_public_key, ssh_auth_sock)
        ssh_trust = None
        if any(value is not None for value in ssh_values):
            if ssh_known_hosts is None:
                raise ValueError("--ssh-known-hosts is required for SSH authentication")
            ssh_trust = SshTrust(
                known_hosts=ssh_known_hosts,
                private_key=ssh_private_key,
                public_key_selector=ssh_public_key,
                auth_sock=ssh_auth_sock,
            )
        result = prepare_source(
            validated,
            home=resolve_home(home),
            patches=patch or (),
            ssh_trust=ssh_trust,
        )
    except ValidationError as exc:
        typer.echo("invalid source lock:", err=True)
        for error in exc.errors(include_input=False, include_url=False):
            location = ".".join(str(part) for part in error["loc"]) or "manifest"
            typer.echo(f"  {location}: {error['msg']}", err=True)
        raise typer.Exit(code=1) from None
    except (OSError, SourceError, TypeError, ValueError, yaml.YAMLError) as exc:
        typer.echo(f"source prepare failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(result.evidence.preparation_id)
    typer.echo(f"worktree: {result.worktree}")
    typer.echo(f"record: {result.record}")


@source_app.command("inspect")
def source_inspect(
    preparation_id: Annotated[str, typer.Argument(help="Preparation ID.")],
    home: Annotated[
        Path | None,
        typer.Option("--home", help="Override the StrixLab data home."),
    ] = None,
) -> None:
    """Print local registry state and portable evidence for a preparation."""

    try:
        result = inspect_source(preparation_id, home=resolve_home(home))
    except (OSError, SourceError, ValueError) as exc:
        typer.echo(f"source inspect failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    payload = {
        "evidence": None if result.evidence is None else result.evidence.model_dump(mode="json"),
        "record_exists": result.record_exists,
        "registry": result.registry.model_dump(mode="json"),
        "worktree_exists": result.worktree_exists,
    }
    typer.echo(canonical_json_bytes(payload).decode(), nl=False)


@source_app.command("cleanup")
def source_cleanup(
    preparation_id: Annotated[str, typer.Argument(help="Preparation ID.")],
    home: Annotated[
        Path | None,
        typer.Option("--home", help="Override the StrixLab data home."),
    ] = None,
    force_changed: Annotated[
        bool,
        typer.Option(
            "--force-changed",
            help="Remove a changed candidate after ownership checks and evidence capture.",
        ),
    ] = False,
) -> None:
    """Remove one exact owned worktree after verifying preserved evidence."""

    try:
        result = cleanup_source(
            preparation_id,
            home=resolve_home(home),
            force_changed=force_changed,
        )
    except (OSError, SourceError, ValueError) as exc:
        typer.echo(f"source cleanup failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"{result.preparation_id}: {result.state}")
    typer.echo(f"record retained: {result.record}")


def _doctor_echo(message: str, context: RedactionContext, *, err: bool = False) -> None:
    try:
        context.assert_text_safe(message)
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
    context = RedactionContext.from_environ(frozen_environ)
    try:
        result = run_doctor(
            machine,
            home=home,
            output=output,
            environ=frozen_environ,
        )
    except ValidationError as exc:
        _doctor_echo("invalid machine profile:", context, err=True)
        for error in exc.errors(include_input=False, include_url=False):
            location = ".".join(str(part) for part in error["loc"]) or "manifest"
            _doctor_echo(f"  {location}: {error['msg']}", context, err=True)
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
        _doctor_echo(f"ready: {result.path}", context)
        return
    for check in result.report.checks:
        if check.status == "blocker":
            _doctor_echo(
                f"blocker [{check.id}]: {check.message}",
                context,
                err=True,
            )
    _doctor_echo(f"report: {result.path}", context, err=True)
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

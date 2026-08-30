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
from strixlab.build_artifacts import BuildArtifactError
from strixlab.build_cache import BuildCacheError, cleanup_build, inspect_build
from strixlab.builds import (
    BuildBusyError,
    BuildStateError,
    inspect_attempt,
    inspect_recipe,
)
from strixlab.bundles import BundleError, export_bundle, verify_bundle
from strixlab.cmake_build import CMakeBuildError, execute_cmake_build
from strixlab.config import read_manifest
from strixlab.doctor import (
    ReportWriteError,
    SensitiveInterpolationError,
    run_doctor,
)
from strixlab.evidence import RunError, inspect_run
from strixlab.git_boundary import SshTrust
from strixlab.manifests import (
    BuildProfileV1,
    ManifestRegistry,
    SourceLockV1,
    resolve_and_validate_manifest,
    validate_manifest,
)
from strixlab.paths import resolve_home
from strixlab.records import RecordError
from strixlab.schema_registry import schema_resource_bytes
from strixlab.secret_policy import RedactionContext
from strixlab.secret_policy import UnsafeOutputError as UnsafeDiagnosticError
from strixlab.serialization import canonical_json_bytes
from strixlab.sources import SourceError, cleanup_source, inspect_source, prepare_source

_BUILD_DOMAIN_ERRORS = (
    OSError,
    ValueError,
    yaml.YAMLError,
    SourceError,
    CMakeBuildError,
    BuildCacheError,
    BuildArtifactError,
    BuildStateError,
    BuildBusyError,
)

app = typer.Typer(
    help="Evidence-first optimization research tooling for AMD Strix Halo.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
schema_app = typer.Typer(help="Inspect versioned manifest schemas.")
manifest_app = typer.Typer(help="Validate versioned manifests.")
source_app = typer.Typer(help="Prepare and manage isolated Git source worktrees.")
build_app = typer.Typer(help="Reproducibly build, inspect, and clean pinned source trees.")
run_app = typer.Typer(help="Inspect finalized run-evidence records.")
bundle_app = typer.Typer(help="Export and verify deterministic run-evidence bundles.")
app.add_typer(schema_app, name="schema")
app.add_typer(manifest_app, name="manifest")
app.add_typer(source_app, name="source")
app.add_typer(build_app, name="build")
app.add_typer(run_app, name="run")
app.add_typer(bundle_app, name="bundle")

_EVIDENCE_DOMAIN_ERRORS = (OSError, ValueError, RunError, RecordError, BundleError)


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


@build_app.command("prepare")
def build_prepare(
    preparation_id: Annotated[str, typer.Argument(help="Source preparation ID.")],
    manifest: Annotated[
        Path,
        typer.Argument(
            help="Versioned build-profile YAML path.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    home: Annotated[
        Path | None,
        typer.Option("--home", help="Override the StrixLab data home."),
    ] = None,
) -> None:
    """Configure and build a pinned source tree, reusing the cache when possible."""

    try:
        value = read_manifest(manifest)
        profile = resolve_and_validate_manifest("build", value, dict(os.environ))
        if not isinstance(profile, BuildProfileV1):
            raise TypeError("build registry returned the wrong model")
        result = execute_cmake_build(preparation_id, profile, home=resolve_home(home))
    except ValidationError as exc:
        typer.echo("invalid build profile:", err=True)
        for error in exc.errors(include_input=False, include_url=False):
            location = ".".join(str(part) for part in error["loc"]) or "manifest"
            typer.echo(f"  {location}: {error['msg']}", err=True)
        raise typer.Exit(code=1) from None
    except SensitiveInterpolationError:
        typer.echo(
            "invalid build profile: sensitive environment interpolation is forbidden", err=True
        )
        raise typer.Exit(code=1) from None
    except _BUILD_DOMAIN_ERRORS as exc:
        typer.echo(f"build prepare failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(result.build_id)
    typer.echo(f"execution: {result.execution_class}")
    typer.echo(f"record: {result.attempt.record}")


@build_app.command("inspect")
def build_inspect(
    identifier: Annotated[str, typer.Argument(help="Recipe, build, or attempt ID.")],
    home: Annotated[
        Path | None,
        typer.Option("--home", help="Override the StrixLab data home."),
    ] = None,
) -> None:
    """Print verified immutable state for one recipe, build, or attempt ID."""

    try:
        resolved = resolve_home(home)
        if identifier.startswith("recipe-sha256:"):
            payload: dict[str, object] = inspect_recipe(identifier, home=resolved).model_dump(
                mode="json"
            )
        elif identifier.startswith("build-sha256:"):
            inspection = inspect_build(identifier, home=resolved)
            payload = {
                "attested": inspection.attested,
                "build_id": inspection.build_id,
                "canonical": inspection.canonical.model_dump(mode="json"),
                "canonical_record_sha256": inspection.canonical_record_sha256,
                "root": None if inspection.root is None else str(inspection.root),
                "state": str(inspection.state),
            }
        elif identifier.startswith("attempt-"):
            payload = inspect_attempt(identifier, home=resolved).model_dump(mode="json")
        else:
            raise ValueError("unrecognized build inspection ID")
    except _BUILD_DOMAIN_ERRORS as exc:
        typer.echo(f"build inspect failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(canonical_json_bytes(payload).decode(), nl=False)


@build_app.command("cleanup")
def build_cleanup(
    build_id: Annotated[str, typer.Argument(help="Machine-local build ID.")],
    home: Annotated[
        Path | None,
        typer.Option("--home", help="Override the StrixLab data home."),
    ] = None,
) -> None:
    """Remove one exact verified build root while retaining immutable evidence."""

    try:
        result = cleanup_build(build_id, home=resolve_home(home))
    except _BUILD_DOMAIN_ERRORS as exc:
        typer.echo(f"build cleanup failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"{result.build_id}: {result.state}")
    typer.echo(f"record retained: {result.record}")


def _doctor_echo(message: str, context: RedactionContext, *, err: bool = False) -> None:
    try:
        context.assert_text_safe(message)
    except UnsafeDiagnosticError:
        typer.echo("doctor failed: unable to safely render terminal output", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(message, err=err)


def _evidence_terminal_context() -> tuple[dict[str, str], RedactionContext]:
    environ = dict(os.environ)
    return environ, RedactionContext.from_environ(environ)


def _evidence_echo(
    message: str,
    context: RedactionContext,
    *,
    err: bool = False,
    nl: bool = True,
) -> None:
    try:
        context.assert_text_safe(message)
    except UnsafeDiagnosticError:
        typer.echo("evidence command failed: unable to safely render terminal output", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(message, err=err, nl=nl)


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


@run_app.command("inspect")
def run_inspect(
    run_id: Annotated[str, typer.Argument(help="Run ID.")],
    home: Annotated[
        Path | None,
        typer.Option("--home", help="Override the StrixLab data home."),
    ] = None,
) -> None:
    """Verify a finalized run's index, record, checksums, and terminal status."""

    _environ, context = _evidence_terminal_context()
    try:
        inspection = inspect_run(run_id, home=resolve_home(home))
    except _EVIDENCE_DOMAIN_ERRORS as exc:
        _evidence_echo(f"run inspect failed: {exc}", context, err=True)
        raise typer.Exit(code=1) from None
    payload = {
        "checksums_sha256": inspection.checksums_sha256,
        "outcome": str(inspection.outcome),
        "record": str(inspection.record),
        "record_sha256": inspection.record_sha256,
        "run_id": inspection.run_id,
        "state": str(inspection.state),
    }
    _evidence_echo(canonical_json_bytes(payload).decode(), context, nl=False)


@bundle_app.command("export")
def bundle_export(
    run_id: Annotated[str, typer.Argument(help="Run ID.")],
    destination: Annotated[Path, typer.Argument(help="New bundle directory to create.")],
    home: Annotated[
        Path | None,
        typer.Option("--home", help="Override the StrixLab data home."),
    ] = None,
) -> None:
    """Export a finalized run to a deterministic, verified bundle directory."""

    environ, context = _evidence_terminal_context()
    try:
        export_bundle(run_id, destination, home=resolve_home(home), environ=environ)
    except _EVIDENCE_DOMAIN_ERRORS as exc:
        _evidence_echo(f"bundle export failed: {exc}", context, err=True)
        raise typer.Exit(code=1) from None
    _evidence_echo(str(destination), context)


@bundle_app.command("verify")
def bundle_verify(
    bundle_directory: Annotated[Path, typer.Argument(help="Bundle directory to verify.")],
) -> None:
    """Verify a bundle directory and print its canonical verified summary."""

    _environ, context = _evidence_terminal_context()
    try:
        inspection = verify_bundle(bundle_directory)
    except _EVIDENCE_DOMAIN_ERRORS as exc:
        _evidence_echo(f"bundle verify failed: {exc}", context, err=True)
        raise typer.Exit(code=1) from None
    payload = {
        "member_count": inspection.member_count,
        "outcome": inspection.outcome,
        "path": str(inspection.path),
        "run_id": inspection.run_id,
        "run_record_sha256": inspection.run_record_sha256,
    }
    _evidence_echo(canonical_json_bytes(payload).decode(), context, nl=False)


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

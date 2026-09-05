"""Campaign command-line interface."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from strixlab.campaigns import (
    CampaignState,
    create_campaign,
    inspect_campaign,
    render_campaign_report,
    resume_campaign,
)
from strixlab.paths import resolve_home
from strixlab.secret_policy import RedactionContext
from strixlab.secret_policy import UnsafeOutputError as UnsafeDiagnosticError
from strixlab.serialization import canonical_json_bytes

_SUCCESS_STATUSES = frozenset({"completed", "ready"})
_DOMAIN_ERRORS = (OSError, TypeError, ValueError)

campaign_app = typer.Typer(
    help="Create, resume, inspect, and report bounded profile-guided campaigns.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def _terminal_context() -> tuple[dict[str, str], RedactionContext]:
    environ = dict(os.environ)
    return environ, RedactionContext.from_environ(environ)


def _campaign_echo(
    message: str,
    context: RedactionContext,
    *,
    err: bool = False,
    nl: bool = True,
) -> None:
    try:
        context.assert_text_safe(message)
    except UnsafeDiagnosticError:
        typer.echo("campaign command failed: unable to safely render terminal output", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(message, err=err, nl=nl)


def _fail(prefix: str, exc: BaseException, context: RedactionContext) -> NoReturn:
    _campaign_echo(f"{prefix}: {exc}", context, err=True)
    raise typer.Exit(code=1) from None


def _exit_for_status(status: str) -> None:
    if status not in _SUCCESS_STATUSES:
        raise typer.Exit(code=1)


def _print_state(state: CampaignState, context: RedactionContext) -> None:
    payload = state.model_dump(mode="json")
    _campaign_echo(canonical_json_bytes(payload).decode(), context, nl=False)
    _exit_for_status(state.status)


@campaign_app.command("create")
def campaign_create(
    plan: Annotated[
        Path,
        # No parser-level filesystem validation: a missing, unreadable, or directory path
        # is surfaced by the campaign engine inside the RedactionContext-protected body,
        # so a secret-bearing path can never be echoed by Typer before the safety check.
        typer.Argument(help="Reviewed campaign plan YAML path."),
    ],
    home: Annotated[
        Path | None,
        typer.Option("--home", help="Override the StrixLab data home."),
    ] = None,
) -> None:
    """Freeze a reviewed campaign plan without running suites."""

    environ, context = _terminal_context()
    try:
        state = create_campaign(plan, home=resolve_home(home), environ=environ)
    except UnsafeDiagnosticError:
        typer.echo("campaign command failed: unable to safely render terminal output", err=True)
        raise typer.Exit(code=1) from None
    except _DOMAIN_ERRORS as exc:
        _fail("campaign create failed", exc, context)
    _print_state(state, context)


@campaign_app.command("resume")
def campaign_resume(
    campaign_id: Annotated[str, typer.Argument(help="Campaign ID.")],
    home: Annotated[
        Path | None,
        typer.Option("--home", help="Override the StrixLab data home."),
    ] = None,
) -> None:
    """Evaluate remaining campaign phases against the frozen evaluator."""

    environ, context = _terminal_context()
    try:
        state = resume_campaign(campaign_id, home=resolve_home(home), environ=environ)
    except UnsafeDiagnosticError:
        typer.echo("campaign command failed: unable to safely render terminal output", err=True)
        raise typer.Exit(code=1) from None
    except _DOMAIN_ERRORS as exc:
        _fail("campaign resume failed", exc, context)
    _print_state(state, context)


@campaign_app.command("inspect")
def campaign_inspect(
    campaign_id: Annotated[str, typer.Argument(help="Campaign ID.")],
    home: Annotated[
        Path | None,
        typer.Option("--home", help="Override the StrixLab data home."),
    ] = None,
) -> None:
    """Print canonical JSON state for one campaign."""

    _environ, context = _terminal_context()
    try:
        state = inspect_campaign(campaign_id, home=resolve_home(home))
    except UnsafeDiagnosticError:
        typer.echo("campaign command failed: unable to safely render terminal output", err=True)
        raise typer.Exit(code=1) from None
    except _DOMAIN_ERRORS as exc:
        _fail("campaign inspect failed", exc, context)
    _print_state(state, context)


@campaign_app.command("report")
def campaign_report(
    campaign_id: Annotated[str, typer.Argument(help="Campaign ID.")],
    home: Annotated[
        Path | None,
        typer.Option("--home", help="Override the StrixLab data home."),
    ] = None,
) -> None:
    """Print a human-readable, actionable campaign report."""

    _environ, context = _terminal_context()
    try:
        state = inspect_campaign(campaign_id, home=resolve_home(home))
        report = render_campaign_report(state)
    except UnsafeDiagnosticError:
        typer.echo("campaign command failed: unable to safely render terminal output", err=True)
        raise typer.Exit(code=1) from None
    except _DOMAIN_ERRORS as exc:
        _fail("campaign report failed", exc, context)
    text = report if report.endswith("\n") else f"{report}\n"
    _campaign_echo(text, context, nl=False)
    _exit_for_status(state.status)

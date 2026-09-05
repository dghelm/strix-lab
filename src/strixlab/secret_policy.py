"""Shared policy for secret-like environment names and interpolation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from strixlab.config import iter_environment_references

_SENSITIVE_NAME_RE = re.compile(
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|KEY|CREDENTIAL|AUTH|COOKIE|SESSION)",
    re.IGNORECASE,
)
_KNOWN_NONSECRET_SESSION_NAMES = frozenset(
    {
        "CLAUDE_CODE_CHILD_SESSION",
        "DBUS_SESSION_BUS_ADDRESS",
        "SESSION_MANAGER",
        "XDG_SESSION_CLASS",
        "XDG_SESSION_ID",
    }
)


class SensitiveInterpolationError(ValueError):
    """Raised before resolution when a manifest references a secret-like name."""


def is_sensitive_name(name: str) -> bool:
    return (
        name not in _KNOWN_NONSECRET_SESSION_NAMES and _SENSITIVE_NAME_RE.search(name) is not None
    )


def reject_sensitive_interpolations(value: Any) -> None:
    """Reject live interpolation references to secret-like environment names."""

    if isinstance(value, str):
        if any(is_sensitive_name(name) for name in iter_environment_references(value)):
            raise SensitiveInterpolationError(
                "manifest cannot interpolate a sensitive environment variable"
            )
    elif isinstance(value, list):
        for child in value:
            reject_sensitive_interpolations(child)
    elif isinstance(value, dict):
        for child in value.values():
            reject_sensitive_interpolations(child)


class UnsafeOutputError(RuntimeError):
    """Raised when an outgoing artifact could disclose a sensitive value."""


def secret_values(environ: Mapping[str, str]) -> tuple[str, ...]:
    """Discover the fixed set of secret-like environment values, longest first."""

    return tuple(
        sorted(
            {value for name, value in environ.items() if value and is_sensitive_name(name)},
            key=len,
            reverse=True,
        )
    )


@dataclass(frozen=True, slots=True)
class RedactionContext:
    """Precomputed secret values shared by every redaction and safety check.

    The secret set is fixed for a run, so it is discovered once and reused for
    designated free-text redaction, whole-payload verification, and terminal-sink
    verification instead of rescanning the environment at each site.
    """

    secrets: tuple[str, ...] = field(repr=False)

    @classmethod
    def from_environ(cls, environ: Mapping[str, str]) -> RedactionContext:
        return cls(secret_values(environ))

    def redact(self, value: str | None) -> str | None:
        if value is None:
            return None
        for secret in self.secrets:
            value = value.replace(secret, "[REDACTED]")
        return value

    def assert_payload_safe(self, payload: bytes) -> None:
        """Fail closed if a serialized artifact still discloses a secret."""

        for secret in self.secrets:
            if secret.encode() in payload:
                raise UnsafeOutputError("output failed secret-safety validation")

    def assert_text_safe(self, value: str) -> None:
        """Fail closed if a terminal line would disclose a sensitive value."""

        self.assert_payload_safe(value.encode())

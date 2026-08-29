"""Shared policy for secret-like environment names and interpolation."""

from __future__ import annotations

import re
from typing import Any

from strixlab.config import iter_environment_references

_SENSITIVE_NAME_RE = re.compile(
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|KEY|CREDENTIAL|AUTH|COOKIE|SESSION)",
    re.IGNORECASE,
)
_KNOWN_NONSECRET_SESSION_NAMES = frozenset(
    {"DBUS_SESSION_BUS_ADDRESS", "SESSION_MANAGER", "XDG_SESSION_ID"}
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

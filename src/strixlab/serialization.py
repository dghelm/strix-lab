"""Stable serialization helpers shared across persisted artifacts."""

from __future__ import annotations

import json
from typing import Any

import yaml


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible value to canonical, stable bytes."""

    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def canonical_yaml_bytes(value: Any) -> bytes:
    """Serialize a mapping/scalar tree to canonical, stable YAML bytes.

    Keys are sorted, output is block style, Unicode is preserved, and exactly one
    trailing newline is emitted so a resolved manifest capture is byte-deterministic.
    """

    text = yaml.safe_dump(
        value,
        sort_keys=True,
        allow_unicode=True,
        default_flow_style=False,
        width=1 << 30,
    )
    return (text.rstrip("\n") + "\n").encode("utf-8")

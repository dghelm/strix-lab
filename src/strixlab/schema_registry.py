"""Canonical JSON Schema generation and packaged-resource access."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

from strixlab.manifests import ManifestRegistry
from strixlab.serialization import canonical_json_bytes

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


def _normalize_pattern(pattern: str) -> str:
    return pattern[:-1] + r"(?![\s\S])" if pattern.endswith("$") else pattern


def _normalize_schema(value: Any) -> None:
    """Apply structural fixes JSON Schema needs but Pydantic omits.

    Every node is normalized in place: generated regexes gain a true end guard
    under JSON Schema semantics, and constrained-key mappings (``patternProperties``
    with no ``additionalProperties``) are closed so the packaged schema rejects
    the same keys the runtime model does.
    """

    if isinstance(value, dict):
        pattern = value.get("pattern")
        if isinstance(pattern, str):
            value["pattern"] = _normalize_pattern(pattern)
        pattern_properties = value.get("patternProperties")
        if isinstance(pattern_properties, dict):
            value["patternProperties"] = {
                _normalize_pattern(key): child for key, child in pattern_properties.items()
            }
            if "additionalProperties" not in value:
                value["additionalProperties"] = False
        for child in value.values():
            _normalize_schema(child)
    elif isinstance(value, list):
        for child in value:
            _normalize_schema(child)


def schema_filename(kind: str, version: int = 1) -> str:
    ManifestRegistry.model_for(kind, version)
    return f"{kind}.schema.json"


def generate_schema(kind: str, version: int = 1) -> dict[str, Any]:
    model = ManifestRegistry.model_for(kind, version)
    schema = model.model_json_schema(mode="validation")
    schema["$schema"] = JSON_SCHEMA_DIALECT
    schema["$id"] = f"urn:strixlab:schema:{kind}:{version}"
    _normalize_schema(schema)
    return schema


def canonical_schema_bytes(kind: str, version: int = 1) -> bytes:
    return canonical_json_bytes(generate_schema(kind, version))


def schema_resource_bytes(kind: str, version: int = 1) -> bytes:
    resource = files("strixlab").joinpath("schemas", f"v{version}", schema_filename(kind, version))
    return resource.read_bytes()


def write_schemas(root: Path | None = None) -> None:
    target = Path(__file__).parent / "schemas" / "v1" if root is None else root
    target.mkdir(parents=True, exist_ok=True)
    for kind in ManifestRegistry.kinds():
        (target / schema_filename(kind)).write_bytes(canonical_schema_bytes(kind))

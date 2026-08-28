from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from pydantic import ValidationError as PydanticValidationError

from strixlab.manifests import ManifestRegistry, validate_manifest
from strixlab.schema_registry import (
    JSON_SCHEMA_DIALECT,
    canonical_schema_bytes,
    schema_filename,
    schema_resource_bytes,
    write_schemas,
)


def test_checked_in_schemas_match_canonical_bytes() -> None:
    for kind in ManifestRegistry.kinds():
        resource = schema_resource_bytes(kind)
        assert resource == canonical_schema_bytes(kind)
        assert resource.endswith(b"\n")


def test_schema_identity_and_dialect() -> None:
    for kind in ManifestRegistry.kinds():
        schema = json.loads(schema_resource_bytes(kind))
        assert schema["$schema"] == JSON_SCHEMA_DIALECT
        assert schema["$id"] == f"urn:strixlab:schema:{kind}:1"
        assert schema["additionalProperties"] is False


def test_write_schemas_uses_canonical_generator(tmp_path: Path) -> None:
    write_schemas(tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        schema_filename(kind) for kind in ManifestRegistry.kinds()
    ]
    for kind in ManifestRegistry.kinds():
        assert (tmp_path / schema_filename(kind)).read_bytes() == canonical_schema_bytes(kind)


def _assert_runtime_and_schema_reject(kind: str, value: dict[str, Any]) -> None:
    with pytest.raises(PydanticValidationError):
        validate_manifest(kind, value)
    validator = Draft202012Validator(json.loads(schema_resource_bytes(kind)))
    with pytest.raises(JsonSchemaValidationError):
        validator.validate(value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(url=""),
        lambda value: value.update(branch_hint=""),
    ],
)
def test_source_schema_matches_runtime_string_constraints(
    mutate: Any, source_value: dict[str, Any]
) -> None:
    mutate(source_value)
    _assert_runtime_and_schema_reject("source-lock", source_value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(id="source\n"),
        lambda value: value.update(commit="0123456789abcdef0123456789abcdef01234567\r"),
        lambda value: value.update(adapter="adapter\n"),
    ],
)
def test_source_schema_rejects_trailing_line_terminators(
    mutate: Any, source_value: dict[str, Any]
) -> None:
    mutate(source_value)
    _assert_runtime_and_schema_reject("source-lock", source_value)


def test_machine_schema_matches_runtime_absolute_path_constraint() -> None:
    value = {
        "schema_version": 1,
        "id": "machine",
        "expect": {"gpu_arch": "gfx1151", "integrated_gpu": True, "memory_gib_min": 1},
        "exclusive_lock": {"path": "relative.lock"},
        "telemetry": {"amd_smi": "auto", "sample_interval_ms": 1},
        "validity": {
            "require_ac_power": True,
            "max_background_gpu_busy_pct": 0,
            "min_available_memory_gib": 0,
            "temperature_warn_c": 90,
        },
    }
    _assert_runtime_and_schema_reject("machine", value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["environment"].update({"BAD.KEY": "value"}),
        lambda value: value["cmake"].update({"BAD.KEY": True}),
        lambda value: value.update(targets=["llama-bench", "llama-bench"]),
        lambda value: value.update(post_build_capture=["ldd", "ldd"]),
    ],
)
def test_build_schema_matches_runtime_mapping_and_uniqueness_constraints(
    mutate: Any, build_value: dict[str, Any]
) -> None:
    mutate(build_value)
    _assert_runtime_and_schema_reject("build", build_value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(targets=["llama-bench\n"]),
        lambda value: value["environment"].update({"BAD_KEY\n": "value"}),
        lambda value: value["cmake"].update({"BAD_KEY\r": True}),
    ],
)
def test_build_schema_rejects_trailing_line_terminators(
    mutate: Any, build_value: dict[str, Any]
) -> None:
    mutate(build_value)
    _assert_runtime_and_schema_reject("build", build_value)

from __future__ import annotations

import json
from copy import deepcopy
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
    ("url", "accepted"),
    [
        ("/srv/git/repository", True),
        ("file:///srv/git/repository", True),
        ("https://example.test/repository.git", True),
        ("ssh://git@example.test/repository.git", True),
        ("git@example.test:organization/repository.git", True),
        ("relative/repository", False),
        ("http://example.test/repository.git", False),
        ("https://user@example.test/repository.git", False),
        ("https://example.test/repository.git?token=secret", False),
        ("file://remote-host/srv/repository", False),
        ("ssh:///missing-host", False),
    ],
)
def test_source_locator_schema_and_runtime_agree(
    url: str, accepted: bool, source_value: dict[str, Any]
) -> None:
    source_value["url"] = url
    schema = Draft202012Validator(json.loads(schema_resource_bytes("source-lock")))

    runtime_accepts = True
    try:
        validate_manifest("source-lock", source_value)
    except PydanticValidationError:
        runtime_accepts = False
    schema_accepts = not tuple(schema.iter_errors(source_value))

    assert runtime_accepts is schema_accepts is accepted


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
        lambda value: value["environment"]["path_lists"].update({"BAD.KEY": "value"}),
        lambda value: value["cmake"].update({"BAD.KEY": True}),
        lambda value: value.update(targets=["llama-bench", "llama-bench"]),
        lambda value: value["toolchain"].update(path=["/usr/bin", "/usr/bin"]),
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
        lambda value: value["environment"]["path_lists"].update({"BAD_KEY\n": "value"}),
        lambda value: value["cmake"].update({"BAD_KEY\r": True}),
    ],
)
def test_build_schema_rejects_trailing_line_terminators(
    mutate: Any, build_value: dict[str, Any]
) -> None:
    mutate(build_value)
    _assert_runtime_and_schema_reject("build", build_value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.pop("machine"),
        lambda value: value["build"].pop("source_commit"),
        lambda value: value["build"].update(profile="untrusted-label"),
        lambda value: value["build"].update(toolchain_mode="cuda"),
        lambda value: value["build"].update(gfx_target="sm_90"),
        lambda value: value["timeouts"].update(describe_seconds=3600.1),
        lambda value: value["timeouts"].update(correctness_seconds=0),
        lambda value: value["timeouts"].update(benchmark_seconds=3601),
        lambda value: value["contract"].pop("comparison"),
        lambda value: value["contract"]["comparison"].update(policy="topk-paired-log-bootstrap-v1"),
        lambda value: value["contract"]["comparison"].update(protected_regression_bps=10_001),
        lambda value: value["contract"]["comparison"].update(
            permitted_arm_differences=["candidate-id", "build-output"]
        ),
    ],
)
def test_capsule_schema_matches_authoritative_runtime_contract(
    mutate: Any, capsule_value: dict[str, Any]
) -> None:
    mutate(capsule_value)
    _assert_runtime_and_schema_reject("capsule", capsule_value)


def _resolved_model(value: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(value)
    resolved["artifact"]["file"]["local_path"] = "/data/models/qwen35-2b/model.gguf"
    return resolved


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["artifact"]["file"].update(sha256="zz"),
        lambda value: value["artifact"]["file"].update(filename="bad\x00name"),
        lambda value: value["artifact"]["file"].update(repository="not a repository"),
        lambda value: value.update(id="model\n"),
        lambda value: value["artifact"]["file"].update(revision="0123"),
    ],
)
def test_model_schema_matches_runtime_string_constraints(
    mutate: Any, model_value: dict[str, Any]
) -> None:
    value = _resolved_model(model_value)
    mutate(value)
    _assert_runtime_and_schema_reject("model", value)

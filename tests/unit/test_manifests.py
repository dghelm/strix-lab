from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from strixlab.manifests import (
    BuildProfileV1,
    MachineProfileV1,
    ManifestRegistry,
    SourceLockV1,
    UnknownManifestKind,
    resolve_and_validate_manifest,
    validate_manifest,
)


@pytest.fixture
def machine_value() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": "strix-halo-128g",
        "expect": {
            "gpu_arch": "gfx1151",
            "integrated_gpu": True,
            "memory_gib_min": 120,
        },
        "exclusive_lock": {"path": "/tmp/strixlab-gfx1151.lock"},
        "telemetry": {"amd_smi": "auto", "sample_interval_ms": 500},
        "validity": {
            "require_ac_power": True,
            "max_background_gpu_busy_pct": 2,
            "min_available_memory_gib": 24,
            "temperature_warn_c": 90,
        },
    }


def test_all_v1_manifests_validate(
    source_value: dict[str, Any],
    machine_value: dict[str, Any],
    build_value: dict[str, Any],
) -> None:
    assert isinstance(validate_manifest("source-lock", source_value), SourceLockV1)
    assert isinstance(validate_manifest("machine", machine_value), MachineProfileV1)
    assert isinstance(validate_manifest("build", build_value), BuildProfileV1)


def test_registry_is_extensible_and_sorted() -> None:
    assert ManifestRegistry.kinds() == ("build", "machine", "source-lock")


def test_registry_rejects_duplicate_kind_and_version() -> None:
    with pytest.raises(ValueError, match="already registered"):
        ManifestRegistry.register("source-lock", 1, SourceLockV1)


def test_trusted_resolution_is_followed_by_validation(source_value: dict[str, Any]) -> None:
    source_value["url"] = "${SOURCE_URL}"

    model = resolve_and_validate_manifest(
        "source-lock", source_value, {"SOURCE_URL": "https://example.test/source.git"}
    )

    assert model.url == "https://example.test/source.git"


def test_trusted_resolution_rejects_newly_invalid_values(source_value: dict[str, Any]) -> None:
    source_value["url"] = "${SOURCE_URL}"

    with pytest.raises(ValidationError, match="Git URL must be a nonempty"):
        resolve_and_validate_manifest("source-lock", source_value, {"SOURCE_URL": ""})


@pytest.mark.parametrize("kind", ["candidate", "challenge", "unknown"])
def test_unknown_manifest_kind_is_rejected(kind: str, source_value: dict[str, Any]) -> None:
    with pytest.raises(UnknownManifestKind, match="unknown manifest kind"):
        validate_manifest(kind, source_value)


@pytest.mark.parametrize("version", [None, True, "1", 2])
def test_unsupported_schema_versions_are_rejected(
    version: Any, source_value: dict[str, Any]
) -> None:
    source_value["schema_version"] = version
    with pytest.raises(UnknownManifestKind, match="unsupported schema version"):
        validate_manifest("source-lock", source_value)


def test_source_defaults_only_branch_hint(source_value: dict[str, Any]) -> None:
    source_value.pop("branch_hint")
    model = SourceLockV1.model_validate(source_value)

    assert model.branch_hint is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", "Strix-Llama"),
        ("id", "strix--llama"),
        ("adapter", "llama-cpp"),
        ("commit", "ABC"),
        ("url", " https://example.test/repo "),
        ("allowed_dirty_state", True),
        ("submodules", 0),
    ],
)
def test_source_contract_is_strict(field: str, value: Any, source_value: dict[str, Any]) -> None:
    source_value[field] = value
    with pytest.raises(ValidationError):
        SourceLockV1.model_validate(source_value)


@pytest.mark.parametrize(
    "url",
    [
        "/srv/git/repository",
        "file:///srv/git/repository",
        "https://example.test/repository.git",
        "ssh://git@example.test/repository.git",
        "git@example.test:organization/repository.git",
    ],
)
def test_source_git_url_accepts_explicit_supported_locators(
    url: str, source_value: dict[str, Any]
) -> None:
    source_value["url"] = url
    assert SourceLockV1.model_validate(source_value).url == url


@pytest.mark.parametrize(
    "url",
    [
        "relative/repository",
        "../repository",
        "--upload-pack=malicious",
        "http://example.test/repository.git",
        "https://user@example.test/repository.git",
        "https://user:secret@example.test/repository.git",
        "https://example.test/repository.git?token=secret",
        "https://example.test/repository.git#mutable",
        "file://remote-host/srv/repository",
        "ssh:///missing-host",
        "https://example.test/repository.git\n",
    ],
)
def test_source_git_url_rejects_ambiguous_or_credential_bearing_locators(
    url: str, source_value: dict[str, Any]
) -> None:
    source_value["url"] = url
    with pytest.raises(ValidationError, match="Git URL|unsupported"):
        SourceLockV1.model_validate(source_value)


def test_unknown_fields_are_rejected(source_value: dict[str, Any]) -> None:
    source_value["surprise"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SourceLockV1.model_validate(source_value)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("expect", "memory_gib_min"), 0),
        (("expect", "memory_gib_min"), float("nan")),
        (("telemetry", "sample_interval_ms"), 0),
        (("validity", "max_background_gpu_busy_pct"), 101),
        (("validity", "min_available_memory_gib"), -1),
        (("validity", "temperature_warn_c"), float("inf")),
        (("exclusive_lock", "path"), "relative.lock"),
    ],
)
def test_machine_structural_constraints(
    path: tuple[str, str], value: Any, machine_value: dict[str, Any]
) -> None:
    machine_value[path[0]][path[1]] = value
    with pytest.raises(ValidationError):
        MachineProfileV1.model_validate(machine_value)


@pytest.mark.parametrize("field", ["targets", "post_build_capture"])
def test_build_lists_are_nonempty(field: str, build_value: dict[str, Any]) -> None:
    build_value[field] = []
    with pytest.raises(ValidationError):
        BuildProfileV1.model_validate(build_value)


@pytest.mark.parametrize("field", ["targets", "post_build_capture"])
def test_build_lists_are_ordered_unique(field: str, build_value: dict[str, Any]) -> None:
    original = build_value[field]
    build_value[field] = [*original, original[0]]
    with pytest.raises(ValidationError, match="must be unique"):
        BuildProfileV1.model_validate(build_value)


def test_build_list_order_is_preserved(build_value: dict[str, Any]) -> None:
    model = BuildProfileV1.model_validate(build_value)
    assert model.targets == build_value["targets"]
    assert model.post_build_capture == build_value["post_build_capture"]


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("environment", "BAD.KEY", "value"),
        ("environment", "GOOD_KEY", "bad\x00value"),
        ("cmake", "BAD.KEY", True),
        ("cmake", "GOOD_KEY", float("inf")),
    ],
)
def test_build_mappings_are_strict(
    section: str,
    key: str,
    value: Any,
    build_value: dict[str, Any],
) -> None:
    build_value[section] = {key: value}
    with pytest.raises(ValidationError):
        BuildProfileV1.model_validate(build_value)


def test_build_target_grammar(build_value: dict[str, Any]) -> None:
    build_value["targets"] = [" bad target "]
    with pytest.raises(ValidationError):
        BuildProfileV1.model_validate(build_value)

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from strixlab.manifests import (
    BuildProfileV1,
    MachineProfileV1,
    ManifestRegistry,
    ModelManifestV1,
    SourceLockV1,
    UnknownManifestKind,
    resolve_and_validate_manifest,
    validate_manifest,
)
from strixlab.secret_policy import SensitiveInterpolationError


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
    assert validate_manifest("build", build_value).model_dump(mode="python") == build_value


def test_registry_is_extensible_and_sorted() -> None:
    assert ManifestRegistry.kinds() == ("build", "machine", "model", "source-lock")


def test_registered_model_resolves_and_keeps_raw_template(model_value: dict[str, Any]) -> None:
    raw = validate_manifest("model", model_value)
    assert raw.artifact.file.local_path == "${MODELS}/qwen35-2b/Qwen_Qwen3.5-2B-Q4_K_M.gguf"
    resolved = resolve_and_validate_manifest("model", model_value, {"MODELS": "/data/models"})
    assert isinstance(resolved, ModelManifestV1)
    assert resolved.artifact.file.local_path == "/data/models/qwen35-2b/Qwen_Qwen3.5-2B-Q4_K_M.gguf"
    assert resolved.registry_status == "registered"
    assert resolved.quantization.is_fully_provenanced() is False


def test_draft_model_forbids_local_identity(draft_model_value: dict[str, Any]) -> None:
    draft = resolve_and_validate_manifest("model", draft_model_value, {})
    assert draft.registry_status == "draft"
    assert draft.artifact.file.local_path is None
    with_path = {
        **draft_model_value,
        "artifact": {"format": "gguf", "file": {"local_path": "/models/x.gguf"}},
    }
    with pytest.raises(ValidationError):
        resolve_and_validate_manifest("model", with_path, {})


def test_registered_model_requires_full_identity(model_value: dict[str, Any]) -> None:
    broken = {
        **model_value,
        "artifact": {
            "format": "gguf",
            "file": {k: v for k, v in model_value["artifact"]["file"].items() if k != "sha256"},
            "metadata_predicates": model_value["artifact"]["metadata_predicates"],
        },
    }
    with pytest.raises(ValidationError):
        resolve_and_validate_manifest("model", broken, {"MODELS": "/data/models"})


def test_model_rejects_sensitive_interpolation(model_value: dict[str, Any]) -> None:
    poisoned = {
        **model_value,
        "artifact": {
            **model_value["artifact"],
            "file": {**model_value["artifact"]["file"], "local_path": "${API_TOKEN}/x.gguf"},
        },
    }
    with pytest.raises(SensitiveInterpolationError):
        resolve_and_validate_manifest("model", poisoned, {"API_TOKEN": "secret"})


def test_model_predicate_shape_is_strict(model_value: dict[str, Any]) -> None:
    bad = {
        **model_value,
        "artifact": {
            **model_value["artifact"],
            "metadata_predicates": [
                {"key": "k", "value_type": "STRING", "scalar_value": "s", "array_types": ["INT32"]}
            ],
        },
    }
    with pytest.raises(ValidationError):
        validate_manifest("model", bad)
    bool_as_int = {
        **model_value,
        "artifact": {
            **model_value["artifact"],
            "metadata_predicates": [{"key": "k", "value_type": "UINT32", "scalar_value": True}],
        },
    }
    with pytest.raises(ValidationError):
        validate_manifest("model", bool_as_int)


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


def test_build_targets_are_nonempty(build_value: dict[str, Any]) -> None:
    build_value["targets"] = []
    with pytest.raises(ValidationError):
        BuildProfileV1.model_validate(build_value)


def test_build_targets_are_ordered_unique(build_value: dict[str, Any]) -> None:
    original = build_value["targets"]
    build_value["targets"] = [*original, original[0]]
    with pytest.raises(ValidationError, match="must be unique"):
        BuildProfileV1.model_validate(build_value)


def test_build_list_order_is_preserved(build_value: dict[str, Any]) -> None:
    model = BuildProfileV1.model_validate(build_value)
    assert model.targets == build_value["targets"]


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
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


def test_build_environment_keys_are_closed(build_value: dict[str, Any]) -> None:
    build_value["environment"]["path_lists"] = {"PATH": "/usr/bin"}
    with pytest.raises(ValidationError):
        BuildProfileV1.model_validate(build_value)


@pytest.mark.parametrize(
    ("mode", "hip_compiler", "rocm_prefix"),
    [
        ("rocm", None, "/opt/rocm-10"),
        ("rocm", "/opt/rocm-10/bin/amdclang++", None),
        ("host", "/opt/rocm-10/bin/amdclang++", None),
    ],
)
def test_build_toolchain_mode_is_explicit(
    mode: str,
    hip_compiler: str | None,
    rocm_prefix: str | None,
    build_value: dict[str, Any],
) -> None:
    build_value["toolchain"].update(
        mode=mode,
        hip_compiler=hip_compiler,
        rocm_prefix=rocm_prefix,
    )
    with pytest.raises(ValidationError):
        BuildProfileV1.model_validate(build_value)


def test_old_build_profile_shape_fails_explicitly(build_value: dict[str, Any]) -> None:
    build_value.pop("toolchain")
    build_value.pop("execution")
    build_value["post_build_capture"] = ["cmake_cache"]

    with pytest.raises(ValidationError, match="toolchain|execution"):
        BuildProfileV1.model_validate(build_value)


def test_build_resolution_rejects_sensitive_interpolation(
    build_value: dict[str, Any],
) -> None:
    build_value["toolchain"]["cmake"] = "${API_TOKEN}/cmake"

    with pytest.raises(SensitiveInterpolationError):
        resolve_and_validate_manifest("build", build_value, {"API_TOKEN": "/secret"})


def test_build_resolution_revalidates_absolute_tool_paths(
    build_value: dict[str, Any],
) -> None:
    build_value["toolchain"]["cmake"] = "${SYSTEM_PREFIX}/bin/cmake"

    model = resolve_and_validate_manifest("build", build_value, {"SYSTEM_PREFIX": "/usr"})

    assert isinstance(model, BuildProfileV1)
    assert model.toolchain.cmake == "/usr/bin/cmake"


def test_raw_build_validation_accepts_a_leading_environment_path(
    build_value: dict[str, Any],
) -> None:
    build_value["toolchain"]["cmake"] = "${SYSTEM_PREFIX}/bin/cmake"

    assert validate_manifest("build", build_value).model_dump()["toolchain"]["cmake"] == (
        "${SYSTEM_PREFIX}/bin/cmake"
    )


@pytest.mark.parametrize("value", ["${UNTERMINATED", "${BAD-NAME}"])
def test_raw_build_validation_rejects_invalid_interpolation_grammar(
    value: str, build_value: dict[str, Any]
) -> None:
    build_value["cmake"]["VALUE"] = value

    with pytest.raises(ValueError, match="environment token"):
        validate_manifest("build", build_value)


def test_raw_build_validation_rejects_interpolation_in_closed_fields(
    build_value: dict[str, Any],
) -> None:
    build_value["id"] = "${BUILD_ID}"

    with pytest.raises(ValidationError):
        validate_manifest("build", build_value)


def test_build_cmake_strings_reject_nul_bytes(build_value: dict[str, Any]) -> None:
    build_value["cmake"]["VALUE"] = "before\x00after"

    with pytest.raises(ValidationError, match="NUL"):
        BuildProfileV1.model_validate(build_value)


def test_build_resolution_rejects_nul_from_environment(
    build_value: dict[str, Any],
) -> None:
    build_value["cmake"]["VALUE"] = "${CMAKE_VALUE}"

    with pytest.raises(ValidationError, match="NUL"):
        resolve_and_validate_manifest("build", build_value, {"CMAKE_VALUE": "before\x00after"})


def test_build_target_grammar(build_value: dict[str, Any]) -> None:
    build_value["targets"] = [" bad target "]
    with pytest.raises(ValidationError):
        BuildProfileV1.model_validate(build_value)

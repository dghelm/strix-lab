"""Versioned manifest models and registry."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self, cast
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    StringConstraints,
    WithJsonSchema,
    field_validator,
    model_validator,
)

from strixlab.capsule_contracts import CapsuleComparisonContractV1
from strixlab.config import iter_environment_references, resolve_environment
from strixlab.naming import ENV_NAME_PATTERN
from strixlab.secret_policy import reject_sensitive_interpolations

DASH_ID_PATTERN = r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$"
UNDERSCORE_ID_PATTERN = r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$"
TARGET_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._+-]*$"
COMMIT_PATTERN = r"^[0-9a-f]{40}$"

DashId = Annotated[str, StringConstraints(pattern=DASH_ID_PATTERN)]
UnderscoreId = Annotated[str, StringConstraints(pattern=UNDERSCORE_ID_PATTERN)]
EnvironmentKey = Annotated[str, StringConstraints(pattern=ENV_NAME_PATTERN)]
BuildTarget = Annotated[str, StringConstraints(pattern=TARGET_PATTERN)]
CommitSha = Annotated[str, StringConstraints(pattern=COMMIT_PATTERN)]


def _clean_string(value: str) -> str:
    if not value:
        raise ValueError("value cannot be empty")
    if value != value.strip():
        raise ValueError("value cannot have surrounding whitespace")
    return _nul_free_string(value)


def _nul_free_string(value: str) -> str:
    if "\x00" in value:
        raise ValueError("value cannot contain NUL bytes")
    return value


def _absolute_path(value: str) -> str:
    _clean_string(value)
    if not Path(value).is_absolute():
        raise ValueError("path must be absolute")
    return value


CleanString = Annotated[
    str,
    AfterValidator(_clean_string),
    WithJsonSchema(
        {
            "type": "string",
            "minLength": 1,
            "pattern": r"^(?!\s)(?![\s\S]*\s$)(?![\s\S]*\u0000)[\s\S]+$",
        }
    ),
]
NulFreeString = Annotated[
    str,
    AfterValidator(_nul_free_string),
    WithJsonSchema({"type": "string", "pattern": r"^[^\u0000]*$"}),
]
StrictNulFreeString = Annotated[
    StrictStr,
    AfterValidator(_nul_free_string),
    WithJsonSchema({"type": "string", "pattern": r"^[^\u0000]*$"}),
]
AbsolutePathString = Annotated[
    str,
    AfterValidator(_absolute_path),
    WithJsonSchema(
        {
            "type": "string",
            "minLength": 1,
            "pattern": r"^(?![\s\S]*\s$)(?![\s\S]*\u0000)/[\s\S]*$",
        }
    ),
]
type CMakeScalar = StrictBool | StrictInt | StrictFloat | StrictNulFreeString


class ManifestModel(BaseModel):
    """Strict base for every public manifest object."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


_GIT_URL_PATTERNS = (
    r"/[^\x00-\x1f\x7f]*",
    r"file:///[^\x00-\x1f\x7f?#]*",
    r"https://[^/@:#?\s]+(?::[0-9]+)?(?:/[^\x00-\x1f\x7f?#]*)?",
    r"ssh://(?:[^/@:#?\s]+@)?[^/@:#?\s]+(?::[0-9]+)?(?:/[^\x00-\x1f\x7f?#]*)?",
    r"(?:[A-Za-z_][A-Za-z0-9_.-]*@)?[A-Za-z0-9][A-Za-z0-9.-]*:[^/\\\s][^\\\s]*",
)


def _validate_git_url(value: str) -> str:
    if not value or value.startswith("-") or value != value.strip():
        raise ValueError("Git URL must be a nonempty, non-option value")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("Git URL cannot contain control characters")
    if not any(re.fullmatch(pattern, value) for pattern in _GIT_URL_PATTERNS):
        raise ValueError("unsupported or ambiguous Git URL")
    if value.startswith("/"):
        return value

    parsed = urlsplit(value)
    if "://" in value and parsed.scheme not in {"file", "https", "ssh"}:
        raise ValueError("unsupported Git URL scheme")
    if parsed.scheme in {"file", "https", "ssh"}:
        if parsed.query or parsed.fragment:
            raise ValueError("Git URL cannot contain a query or fragment")
        if parsed.username is not None and parsed.scheme in {"file", "https"}:
            raise ValueError("Git URL cannot contain user information")
        if parsed.password is not None:
            raise ValueError("Git URL cannot contain a password")
        if parsed.scheme == "file" and (parsed.netloc or not parsed.path.startswith("/")):
            raise ValueError("file Git URLs must contain a local absolute path")
        if parsed.scheme in {"https", "ssh"} and not parsed.hostname:
            raise ValueError("remote Git URLs must contain a host")
        return value

    if re.fullmatch(_GIT_URL_PATTERNS[-1], value):
        return value
    raise ValueError("unsupported or ambiguous Git URL")


GitUrl = Annotated[
    str,
    AfterValidator(_validate_git_url),
    WithJsonSchema(
        {
            "type": "string",
            "minLength": 1,
            "allOf": [
                {
                    "pattern": (
                        r"^(?!-)(?!\s)(?![\s\S]*\s$)"
                        r"(?![\s\S]*[\u0000-\u001f\u007f])[\s\S]+$"
                    )
                },
                {"anyOf": [{"pattern": f"^(?:{pattern})$"} for pattern in _GIT_URL_PATTERNS]},
            ],
            "description": (
                "Absolute local path or credential-free file, HTTPS, or SSH Git locator."
            ),
        }
    ),
]


class SourceLockV1(ManifestModel):
    schema_version: Literal[1]
    id: DashId
    kind: Literal["git"]
    url: GitUrl
    commit: CommitSha
    branch_hint: CleanString | None = None
    submodules: StrictBool
    adapter: UnderscoreId
    allowed_dirty_state: Literal[False]


class MachineExpectationV1(ManifestModel):
    gpu_arch: CleanString
    integrated_gpu: StrictBool
    memory_gib_min: Annotated[float, Field(gt=0)]


class ExclusiveLockV1(ManifestModel):
    path: AbsolutePathString


class TelemetryV1(ManifestModel):
    amd_smi: Literal["auto", "required", "disabled"]
    sample_interval_ms: Annotated[StrictInt, Field(gt=0)]


class MachineValidityV1(ManifestModel):
    require_ac_power: StrictBool
    max_background_gpu_busy_pct: Annotated[float, Field(ge=0, le=100)]
    min_available_memory_gib: Annotated[float, Field(ge=0)]
    temperature_warn_c: float


class MachineProfileV1(ManifestModel):
    schema_version: Literal[1]
    id: DashId
    expect: MachineExpectationV1
    exclusive_lock: ExclusiveLockV1
    telemetry: TelemetryV1
    validity: MachineValidityV1


BuildPathEnvironmentKey = Literal[
    "ROCM_PATH",
    "HIP_PATH",
    "LD_LIBRARY_PATH",
    "CMAKE_PREFIX_PATH",
    "PKG_CONFIG_PATH",
]
BuildLiteralEnvironmentKey = Literal["SOURCE_DATE_EPOCH"]


class BuildToolchainV1(ManifestModel):
    mode: Literal["host", "rocm"]
    prefixes: dict[UnderscoreId, AbsolutePathString]
    cmake: AbsolutePathString
    ninja: AbsolutePathString
    c_compiler: AbsolutePathString
    cxx_compiler: AbsolutePathString
    hip_compiler: AbsolutePathString | None
    rocm_prefix: AbsolutePathString | None
    path: Annotated[
        list[AbsolutePathString], Field(min_length=1, json_schema_extra={"uniqueItems": True})
    ]

    @field_validator("path")
    @classmethod
    def ordered_unique_path(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("toolchain PATH entries must be unique")
        return value

    @model_validator(mode="after")
    def valid_mode(self) -> Self:
        if not self.prefixes:
            raise ValueError("toolchain prefixes cannot be empty")
        if self.mode == "rocm":
            if self.hip_compiler is None or self.rocm_prefix is None:
                raise ValueError("ROCm mode requires hip_compiler and rocm_prefix")
            if self.prefixes.get("rocm") != self.rocm_prefix:
                raise ValueError("ROCm mode requires a matching 'rocm' prefix")
        elif self.hip_compiler is not None or self.rocm_prefix is not None:
            raise ValueError("host mode cannot declare hip_compiler or rocm_prefix")
        return self


class BuildEnvironmentV1(ManifestModel):
    path_lists: dict[BuildPathEnvironmentKey, NulFreeString]
    literals: dict[BuildLiteralEnvironmentKey, NulFreeString]


class BuildTimeoutsV1(ManifestModel):
    discovery_seconds: Annotated[float, Field(gt=0, le=3600)]
    configure_seconds: Annotated[float, Field(gt=0, le=86400)]
    build_seconds: Annotated[float, Field(gt=0, le=86400)]
    inspection_seconds: Annotated[float, Field(gt=0, le=3600)]
    capability_seconds: Annotated[float, Field(gt=0, le=3600)]


class BuildExecutionV1(ManifestModel):
    jobs: Annotated[StrictInt, Field(gt=0, le=1024)]
    timeouts: BuildTimeoutsV1


class BuildProfileV1(ManifestModel):
    schema_version: Literal[1]
    id: DashId
    source: DashId
    generator: Literal["Ninja"]
    build_type: Literal["Release", "RelWithDebInfo", "Debug"]
    toolchain: BuildToolchainV1
    execution: BuildExecutionV1
    environment: BuildEnvironmentV1
    cmake: dict[EnvironmentKey, CMakeScalar]
    targets: Annotated[
        list[BuildTarget], Field(min_length=1, json_schema_extra={"uniqueItems": True})
    ]

    @field_validator("targets")
    @classmethod
    def ordered_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("list entries must be unique")
        return value


def _raw_absolute_path(value: str) -> str:
    references = tuple(iter_environment_references(value))
    sentinel_environment = {name: f"/__strixlab_environment__/{name}" for name in references}
    resolved = resolve_environment(value, sentinel_environment)
    _absolute_path(resolved)
    return value


RawAbsolutePathString = Annotated[
    str,
    AfterValidator(_raw_absolute_path),
    WithJsonSchema(
        {
            "type": "string",
            "minLength": 1,
            "pattern": r"^(?:(?:\$\{[A-Za-z_][A-Za-z0-9_]*\})|/)[\s\S]*$",
        }
    ),
]


def _raw_environment_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, str):
        names.update(iter_environment_references(value))
    elif isinstance(value, list):
        for child in value:
            names.update(_raw_environment_names(child))
    elif isinstance(value, dict):
        for child in value.values():
            names.update(_raw_environment_names(child))
    return names


class _RawBuildToolchainV1(ManifestModel):
    mode: Literal["host", "rocm"]
    prefixes: dict[UnderscoreId, RawAbsolutePathString]
    cmake: RawAbsolutePathString
    ninja: RawAbsolutePathString
    c_compiler: RawAbsolutePathString
    cxx_compiler: RawAbsolutePathString
    hip_compiler: RawAbsolutePathString | None
    rocm_prefix: RawAbsolutePathString | None
    path: Annotated[
        list[RawAbsolutePathString], Field(min_length=1, json_schema_extra={"uniqueItems": True})
    ]

    @field_validator("path")
    @classmethod
    def ordered_unique_path(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("toolchain PATH entries must be unique")
        return value

    @model_validator(mode="after")
    def valid_mode(self) -> Self:
        if not self.prefixes:
            raise ValueError("toolchain prefixes cannot be empty")
        if self.mode == "rocm":
            if self.hip_compiler is None or self.rocm_prefix is None:
                raise ValueError("ROCm mode requires hip_compiler and rocm_prefix")
            if "rocm" not in self.prefixes:
                raise ValueError("ROCm mode requires a declared 'rocm' prefix")
        elif self.hip_compiler is not None or self.rocm_prefix is not None:
            raise ValueError("host mode cannot declare hip_compiler or rocm_prefix")
        return self


class _RawBuildProfileV1(ManifestModel):
    schema_version: Literal[1]
    id: DashId
    source: DashId
    generator: Literal["Ninja"]
    build_type: Literal["Release", "RelWithDebInfo", "Debug"]
    toolchain: _RawBuildToolchainV1
    execution: BuildExecutionV1
    environment: BuildEnvironmentV1
    cmake: dict[EnvironmentKey, CMakeScalar]
    targets: Annotated[
        list[BuildTarget], Field(min_length=1, json_schema_extra={"uniqueItems": True})
    ]

    @model_validator(mode="before")
    @classmethod
    def valid_interpolation_grammar(cls, value: Any) -> Any:
        names = _raw_environment_names(value)
        sentinel_environment = {name: f"/__strixlab_environment__/{name}" for name in names}
        resolve_environment(value, sentinel_environment)
        return value

    @field_validator("targets")
    @classmethod
    def ordered_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("list entries must be unique")
        return value


# --- Model manifest v1 --------------------------------------------------------

Sha256Lower = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
RepositoryId = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
]

# The inspector's closed GGUF value-type vocabulary (``gguf.GGUFValueType`` names).
GgufValueType = Literal[
    "UINT8",
    "INT8",
    "UINT16",
    "INT16",
    "UINT32",
    "INT32",
    "FLOAT32",
    "BOOL",
    "STRING",
    "ARRAY",
    "UINT64",
    "INT64",
    "FLOAT64",
]
_GGUF_INTEGER_TYPES = frozenset(
    {"UINT8", "INT8", "UINT16", "INT16", "UINT32", "INT32", "UINT64", "INT64"}
)
_GGUF_FLOAT_TYPES = frozenset({"FLOAT32", "FLOAT64"})


def _check_scalar_type_agreement(value_type: str, value: object) -> None:
    if value_type in _GGUF_INTEGER_TYPES:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"scalar_value must be an integer for value_type {value_type}")
    elif value_type in _GGUF_FLOAT_TYPES:
        if not isinstance(value, float):
            raise ValueError(f"scalar_value must be a float for value_type {value_type}")
    elif value_type == "BOOL":
        if not isinstance(value, bool):
            raise ValueError("scalar_value must be a boolean for value_type BOOL")
    else:  # STRING
        if not isinstance(value, str):
            raise ValueError("scalar_value must be a string for value_type STRING")


class MetadataPredicateV1(ManifestModel):
    """One strict, machine-checkable GGUF metadata predicate.

    Exactly one of ``scalar_value`` (type-strict exact equality against the inspector's
    scalar) or ``array_types`` (an exact match of the inspector's complete nested
    element-type tuple, never array contents) is declared. No coercion, substring, or
    numeric tolerance ever applies.
    """

    key: CleanString
    value_type: GgufValueType
    scalar_value: StrictBool | StrictInt | StrictFloat | StrictStr | None = None
    array_types: list[GgufValueType] | None = None

    @model_validator(mode="after")
    def _exactly_one_shape(self) -> Self:
        has_scalar = self.scalar_value is not None
        has_array = self.array_types is not None
        if has_scalar == has_array:
            raise ValueError("predicate must declare exactly one of scalar_value or array_types")
        if has_array:
            if self.value_type != "ARRAY":
                raise ValueError("an array predicate must use value_type ARRAY")
            if not self.array_types:
                raise ValueError("array_types must be nonempty")
        else:
            if self.value_type == "ARRAY":
                raise ValueError("a scalar predicate cannot use value_type ARRAY")
            assert self.scalar_value is not None
            _check_scalar_type_agreement(self.value_type, self.scalar_value)
        return self


def _unique_predicate_keys(value: list[MetadataPredicateV1]) -> list[MetadataPredicateV1]:
    keys = [predicate.key for predicate in value]
    if len(keys) != len(set(keys)):
        raise ValueError("metadata predicate keys must be unique")
    return value


def _check_sidecar_consistency(
    *, inspection: str, sidecar_format: str, kind: str, has_predicates: bool
) -> None:
    """Shared sidecar inspection/kind/predicate invariant for the raw and public models."""

    if inspection == "gguf":
        if sidecar_format != "gguf" or kind != "mmproj":
            raise ValueError("gguf inspection requires an mmproj sidecar in gguf format")
    elif has_predicates:
        raise ValueError("a hash-only sidecar cannot declare metadata predicates")


class ModelBaseV1(ManifestModel):
    repository: RepositoryId
    revision: CommitSha
    license: CleanString


class ModelArchitectureV1(ManifestModel):
    family: UnderscoreId
    moe: StrictBool
    gated_deltanet: StrictBool
    full_attention: StrictBool
    qsa: StrictBool
    mtp: StrictBool
    vision: StrictBool


class ModelFileIdentityV1(ManifestModel):
    repository: RepositoryId | None = None
    revision: CommitSha | None = None
    filename: CleanString | None = None
    local_path: AbsolutePathString | None = None
    size_bytes: Annotated[StrictInt, Field(gt=0)] | None = None
    sha256: Sha256Lower | None = None


class ModelArtifactV1(ManifestModel):
    format: Literal["gguf"]
    file: ModelFileIdentityV1
    metadata_predicates: Annotated[
        list[MetadataPredicateV1], AfterValidator(_unique_predicate_keys)
    ] = Field(default_factory=list)


class ModelSidecarV1(ManifestModel):
    id: DashId
    kind: Literal["mmproj", "imatrix", "opaque"]
    format: Literal["gguf", "opaque"]
    file: ModelFileIdentityV1
    inspection: Literal["gguf", "hash-only"]
    metadata_predicates: Annotated[
        list[MetadataPredicateV1], AfterValidator(_unique_predicate_keys)
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent_sidecar(self) -> Self:
        _check_sidecar_consistency(
            inspection=self.inspection,
            sidecar_format=self.format,
            kind=self.kind,
            has_predicates=bool(self.metadata_predicates),
        )
        return self


_QUANT_UNKNOWN = "unknown"


class ModelQuantizationV1(ManifestModel):
    format_family: CleanString
    storage_format: Literal["gguf"]
    measured_bits_per_weight: Annotated[float, Field(gt=0)] | None = None
    tensor_policy_id: CleanString
    tensor_policy_source: CleanString
    calibration_method: CleanString
    calibration_source: CleanString
    calibration_hash: Sha256Lower | Literal["unknown"]

    def is_fully_provenanced(self) -> bool:
        """True only when every quant-policy provenance field is explicitly known."""

        return _QUANT_UNKNOWN not in {
            self.tensor_policy_id,
            self.tensor_policy_source,
            self.calibration_method,
            self.calibration_source,
            self.calibration_hash,
        }


class ModelExecutionV1(ManifestModel):
    verification_status: Literal["unverified"]
    required_sources: Annotated[
        list[CleanString], Field(default_factory=list, json_schema_extra={"uniqueItems": True})
    ]
    required_features: Annotated[
        list[CleanString], Field(default_factory=list, json_schema_extra={"uniqueItems": True})
    ]

    @model_validator(mode="after")
    def _unique_requirements(self) -> Self:
        for name, values in (
            ("required_sources", self.required_sources),
            ("required_features", self.required_features),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} entries must be unique")
        return self


def _apply_model_manifest_invariants(manifest: ModelManifestV1) -> None:
    """Validate the registered/draft variant selected by ``registry_status``.

    Shared by the resolved public model and the raw pre-resolution model so both stages
    enforce the same identity discipline: a registered manifest pins full upstream and
    local identity and carries no draft reason; a draft carries a bounded reason and no
    local identity, receipt predicate, or sidecar.
    """

    artifact = manifest.artifact
    file = artifact.file
    predicates_present = bool(artifact.metadata_predicates) or any(
        sidecar.metadata_predicates for sidecar in manifest.sidecars
    )
    if manifest.registry_status == "registered":
        if manifest.draft_reason is not None:
            raise ValueError("a registered manifest cannot declare a draft_reason")
        if manifest.base_model is None:
            raise ValueError("a registered manifest requires base_model identity")
        if manifest.architecture is None:
            raise ValueError("a registered manifest requires an architecture")
        missing = [
            name
            for name, value in (
                ("repository", file.repository),
                ("revision", file.revision),
                ("filename", file.filename),
                ("local_path", file.local_path),
                ("size_bytes", file.size_bytes),
                ("sha256", file.sha256),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"a registered artifact requires {', '.join(missing)}")
        for sidecar in manifest.sidecars:
            side = sidecar.file
            if side.local_path is None or side.size_bytes is None or side.sha256 is None:
                raise ValueError("a registered sidecar requires local_path, size_bytes, and sha256")
    else:  # draft
        if manifest.draft_reason is None:
            raise ValueError("a draft manifest requires a draft_reason")
        if file.local_path is not None or file.size_bytes is not None or file.sha256 is not None:
            raise ValueError("a draft manifest cannot declare local artifact identity")
        if predicates_present:
            raise ValueError("a draft manifest cannot declare metadata predicates")
        if manifest.sidecars:
            raise ValueError("a draft manifest cannot declare sidecars")

    paths = [file.local_path] if file.local_path is not None else []
    ids = [sidecar.id for sidecar in manifest.sidecars]
    if len(ids) != len(set(ids)):
        raise ValueError("sidecar ids must be unique")
    for sidecar in manifest.sidecars:
        if sidecar.file.local_path is not None:
            if sidecar.file.local_path in paths:
                raise ValueError("sidecar local_path cannot alias another artifact")
            paths.append(sidecar.file.local_path)


class ModelManifestV1(ManifestModel):
    schema_version: Literal[1]
    id: DashId
    registry_status: Literal["registered", "draft"]
    base_model: ModelBaseV1 | None = None
    architecture: ModelArchitectureV1 | None = None
    artifact: ModelArtifactV1
    sidecars: list[ModelSidecarV1] = Field(default_factory=list)
    quantization: ModelQuantizationV1
    execution: ModelExecutionV1
    quality_reference: DashId | None = None
    draft_reason: CleanString | None = None

    @model_validator(mode="after")
    def _valid_variant(self) -> Self:
        _apply_model_manifest_invariants(self)
        return self


class _RawModelFileIdentityV1(ManifestModel):
    repository: RepositoryId | None = None
    revision: CommitSha | None = None
    filename: CleanString | None = None
    local_path: RawAbsolutePathString | None = None
    size_bytes: Annotated[StrictInt, Field(gt=0)] | None = None
    sha256: Sha256Lower | None = None


class _RawModelArtifactV1(ManifestModel):
    format: Literal["gguf"]
    file: _RawModelFileIdentityV1
    metadata_predicates: Annotated[
        list[MetadataPredicateV1], AfterValidator(_unique_predicate_keys)
    ] = Field(default_factory=list)


class _RawModelSidecarV1(ManifestModel):
    id: DashId
    kind: Literal["mmproj", "imatrix", "opaque"]
    format: Literal["gguf", "opaque"]
    file: _RawModelFileIdentityV1
    inspection: Literal["gguf", "hash-only"]
    metadata_predicates: Annotated[
        list[MetadataPredicateV1], AfterValidator(_unique_predicate_keys)
    ] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent_sidecar(self) -> Self:
        _check_sidecar_consistency(
            inspection=self.inspection,
            sidecar_format=self.format,
            kind=self.kind,
            has_predicates=bool(self.metadata_predicates),
        )
        return self


class _RawModelManifestV1(ManifestModel):
    schema_version: Literal[1]
    id: DashId
    registry_status: Literal["registered", "draft"]
    base_model: ModelBaseV1 | None = None
    architecture: ModelArchitectureV1 | None = None
    artifact: _RawModelArtifactV1
    sidecars: list[_RawModelSidecarV1] = Field(default_factory=list)
    quantization: ModelQuantizationV1
    execution: ModelExecutionV1
    quality_reference: DashId | None = None
    draft_reason: CleanString | None = None

    @model_validator(mode="before")
    @classmethod
    def valid_interpolation_grammar(cls, value: Any) -> Any:
        names = _raw_environment_names(value)
        sentinel_environment = {name: f"/__strixlab_environment__/{name}" for name in names}
        resolve_environment(value, sentinel_environment)
        return value

    @model_validator(mode="after")
    def _valid_variant(self) -> Self:
        _apply_model_manifest_invariants(cast("ModelManifestV1", self))
        return self


# --- Suite manifest v1 --------------------------------------------------------

# Aggregate bounds applied after per-field validation, before a run is allocated.
SUITE_MAX_PROMPTS = 4
SUITE_MAX_PERFORMANCE_CASES = 8
SUITE_MAX_ADAPTER_INVOCATIONS = 128
SUITE_MAX_BENCHMARK_REPETITIONS = 512
SUITE_MAX_PROMPT_BYTES = 16 * 1024
SUITE_MAX_PROMPT_AGGREGATE_BYTES = 32 * 1024
# The backend/server/bench case-id length ceiling (their DashId plus the 64-byte cap).
SUITE_ADAPTER_CASE_ID_MAX = 64
_SUITE_TOKEN_MAX = 1_048_576


def suite_warmup_case_id(index: int, case_id: str) -> str:
    """The deterministic warmup adapter case id for one ordered warmup round."""

    return f"warmup-{index:02d}-{case_id}"


def suite_measurement_case_id(window: int, case_id: str) -> str:
    """The deterministic measurement adapter case id for one ordered window."""

    return f"measure-{window:02d}-{case_id}"


def suite_greedy_case_id(greedy_id: str, prompt_id: str) -> str:
    """The deterministic server adapter case id for one greedy prompt."""

    return f"{greedy_id}-{prompt_id}"


_DASH_ID_RE = re.compile(DASH_ID_PATTERN)


def _valid_adapter_case_id(value: str) -> bool:
    return _DASH_ID_RE.fullmatch(value) is not None and len(value) <= SUITE_ADAPTER_CASE_ID_MAX


def _suite_backend(value: str) -> str:
    if not 1 <= len(value) <= 128:
        raise ValueError("backend must be 1 to 128 characters")
    if any(not 0x20 <= ord(character) <= 0x7E for character in value):
        raise ValueError("backend must be printable ASCII")
    return value


def _suite_params_regex(value: str) -> str:
    if not value:
        raise ValueError("params_regex must be non-empty")
    if len(value.encode("utf-8")) > 512:
        raise ValueError("params_regex exceeds 512 UTF-8 bytes")
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in value):
        raise ValueError("params_regex must not contain C0 or DEL control characters")
    return value


def _suite_prompt_text(value: str) -> str:
    size = len(value.encode("utf-8"))
    if size == 0 or size > SUITE_MAX_PROMPT_BYTES:
        raise ValueError("prompt text must contain 1 through 16384 UTF-8 bytes")
    if any((ord(char) < 32 and char not in "\n\t") or ord(char) == 127 for char in value):
        raise ValueError("prompt text contains a forbidden control character")
    return value


SuiteOperationName = Annotated[StrictStr, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]*$")]
SuiteBackendSelector = Annotated[StrictStr, AfterValidator(_suite_backend)]
SuiteParamsRegex = Annotated[StrictStr, AfterValidator(_suite_params_regex)]
SuitePromptText = Annotated[StrictStr, AfterValidator(_suite_prompt_text)]
SuiteTokenCount = Annotated[StrictInt, Field(ge=0, le=_SUITE_TOKEN_MAX)]
SuiteGfxArch = Annotated[StrictStr, StringConstraints(pattern=r"^gfx[0-9a-f]+$")]
SuiteTimeoutSeconds = Annotated[float, Field(gt=0, le=3600)]


class SuiteBuildRequirementV1(ManifestModel):
    source_id: DashId
    source_commit: CommitSha
    toolchain_mode: Literal["host", "rocm"]
    gfx_target: SuiteGfxArch


class SuiteBackendOpsV1(ManifestModel):
    id: DashId
    backend: SuiteBackendSelector
    operations: Annotated[list[SuiteOperationName], Field(min_length=1, max_length=32)]
    params_regex: SuiteParamsRegex

    @model_validator(mode="after")
    def _unique_operations(self) -> Self:
        if len(set(self.operations)) != len(self.operations):
            raise ValueError("operations must be unique")
        return self


class SuiteGreedyPromptV1(ManifestModel):
    id: DashId
    text: SuitePromptText


class SuiteGreedyV1(ManifestModel):
    id: DashId
    prompt_set_id: DashId
    prompts: Annotated[
        list[SuiteGreedyPromptV1],
        Field(min_length=1, max_length=SUITE_MAX_PROMPTS, json_schema_extra={"uniqueItems": True}),
    ]
    seed: Annotated[StrictInt, Field(ge=-(2**31), le=2**31 - 1)]
    output_tokens: Annotated[StrictInt, Field(ge=1, le=4096)]
    context_size: Annotated[StrictInt, Field(ge=1, le=1_048_576)]
    gpu_layers: Annotated[StrictInt, Field(ge=0, le=999)]

    @model_validator(mode="after")
    def _unique_prompts(self) -> Self:
        ids = [prompt.id for prompt in self.prompts]
        if len(set(ids)) != len(ids):
            raise ValueError("prompt ids must be unique")
        return self


class SuiteCorrectnessV1(ManifestModel):
    backend_ops: SuiteBackendOpsV1
    greedy: SuiteGreedyV1


class SuitePerformanceCaseV1(ManifestModel):
    id: DashId
    prompt_tokens: SuiteTokenCount
    generated_tokens: SuiteTokenCount

    @model_validator(mode="after")
    def _one_metric(self) -> Self:
        if (self.prompt_tokens > 0) == (self.generated_tokens > 0):
            raise ValueError("exactly one of prompt_tokens/generated_tokens must be nonzero")
        return self


class SuitePerformanceV1(ManifestModel):
    protocol: Literal["windowed-interleaved-v1"]
    warmup_runs: Annotated[StrictInt, Field(ge=0, le=16)]
    measurement_windows: Annotated[StrictInt, Field(ge=1, le=64)]
    repetitions_per_window: Annotated[StrictInt, Field(ge=1, le=32)]
    cases: Annotated[
        list[SuitePerformanceCaseV1],
        Field(
            min_length=1,
            max_length=SUITE_MAX_PERFORMANCE_CASES,
            json_schema_extra={"uniqueItems": True},
        ),
    ]

    @model_validator(mode="after")
    def _unique_cases(self) -> Self:
        ids = [case.id for case in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("performance case ids must be unique")
        return self


class SuiteTimeoutsV1(ManifestModel):
    capability_seconds: SuiteTimeoutSeconds
    backend_ops_seconds: SuiteTimeoutSeconds
    server_readiness_seconds: SuiteTimeoutSeconds
    server_request_seconds: SuiteTimeoutSeconds
    server_shutdown_seconds: SuiteTimeoutSeconds
    benchmark_seconds: SuiteTimeoutSeconds


class SuiteManifestV1(ManifestModel):
    schema_version: Literal[1]
    id: DashId
    machine: DashId
    model: DashId
    build: SuiteBuildRequirementV1
    correctness: SuiteCorrectnessV1
    performance: SuitePerformanceV1
    timeouts: SuiteTimeoutsV1

    @model_validator(mode="after")
    def _aggregate_limits(self) -> Self:
        prompts = self.correctness.greedy.prompts
        cases = self.performance.cases
        warmups = self.performance.warmup_runs
        windows = self.performance.measurement_windows
        reps = self.performance.repetitions_per_window

        aggregate_bytes = sum(len(prompt.text.encode("utf-8")) for prompt in prompts)
        if aggregate_bytes > SUITE_MAX_PROMPT_AGGREGATE_BYTES:
            raise ValueError("aggregate prompt text exceeds 32 KiB")

        invocations = 1 + len(prompts) + len(cases) * (warmups + windows)
        if invocations > SUITE_MAX_ADAPTER_INVOCATIONS:
            raise ValueError("suite exceeds the maximum adapter invocation budget")
        repetitions = len(cases) * (warmups + windows * reps)
        if repetitions > SUITE_MAX_BENCHMARK_REPETITIONS:
            raise ValueError("suite exceeds the maximum benchmark repetition budget")

        generated: list[str] = [
            suite_greedy_case_id(self.correctness.greedy.id, prompt.id) for prompt in prompts
        ]
        for case in cases:
            generated.extend(
                suite_warmup_case_id(index, case.id) for index in range(1, warmups + 1)
            )
            generated.extend(
                suite_measurement_case_id(window, case.id) for window in range(1, windows + 1)
            )
        for case_id in generated:
            if not _valid_adapter_case_id(case_id):
                raise ValueError(f"generated adapter case id is invalid: {case_id!r}")
        if len(set(generated)) != len(generated):
            raise ValueError("generated adapter case ids collide")
        return self


# --- Capsule manifest v1 ------------------------------------------------------


class _CapsuleManifestModel(ManifestModel):
    """Strict and transitively frozen base for the capsule manifest tree."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class CapsuleBuildRequirementV1(_CapsuleManifestModel):
    """Authoritative build coordinates required from the caller's verified lease."""

    source_id: DashId
    source_commit: CommitSha
    toolchain_mode: Literal["host", "rocm"]
    gfx_target: SuiteGfxArch
    target: BuildTarget


class CapsuleContractV1(_CapsuleManifestModel):
    """The fixed process protocol and immutable scenario identity."""

    protocol: Literal["native-capsule-v1"]
    scenario_sha256: Sha256Lower
    comparison: CapsuleComparisonContractV1


class CapsuleTimeoutsV1(_CapsuleManifestModel):
    """Independent hard wall-clock bounds for the three protocol children."""

    describe_seconds: SuiteTimeoutSeconds
    correctness_seconds: SuiteTimeoutSeconds
    benchmark_seconds: SuiteTimeoutSeconds


class CapsuleManifestV1(_CapsuleManifestModel):
    """Frozen library-only declaration for one native capsule candidate."""

    schema_version: Literal[1]
    id: DashId
    candidate: DashId
    machine: DashId
    build: CapsuleBuildRequirementV1
    contract: CapsuleContractV1
    timeouts: CapsuleTimeoutsV1


class UnknownManifestKind(ValueError):
    """Raised when no manifest kind is registered."""


class ManifestRegistry:
    """Extensible kind/version registry for public manifest models."""

    _models: ClassVar[dict[str, dict[int, type[ManifestModel]]]] = {}

    @classmethod
    def register(cls, kind: str, version: int, model: type[ManifestModel]) -> None:
        versions = cls._models.setdefault(kind, {})
        if version in versions:
            raise ValueError(f"manifest model already registered: {kind} v{version}")
        versions[version] = model

    @classmethod
    def kinds(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._models))

    @classmethod
    def model_for(cls, kind: str, version: Any) -> type[ManifestModel]:
        versions = cls._models.get(kind)
        if versions is None:
            raise UnknownManifestKind(f"unknown manifest kind: {kind}")
        if not isinstance(version, int) or isinstance(version, bool) or version not in versions:
            raise UnknownManifestKind(f"unsupported schema version for {kind}: {version!r}")
        return versions[version]

    @classmethod
    def validate(cls, kind: str, value: Mapping[str, Any]) -> ManifestModel:
        model = cls.model_for(kind, value.get("schema_version"))
        return model.model_validate(value)


ManifestRegistry.register("source-lock", 1, SourceLockV1)
ManifestRegistry.register("machine", 1, MachineProfileV1)
ManifestRegistry.register("build", 1, BuildProfileV1)
ManifestRegistry.register("model", 1, ModelManifestV1)
ManifestRegistry.register("suite", 1, SuiteManifestV1)
ManifestRegistry.register("capsule", 1, CapsuleManifestV1)

_RAW_MODELS: dict[str, type[ManifestModel]] = {
    "build": _RawBuildProfileV1,
    "model": _RawModelManifestV1,
}


def validate_manifest(kind: str, value: Mapping[str, Any]) -> ManifestModel:
    """Validate one already-parsed raw manifest without resolving its values."""

    raw_model = _RAW_MODELS.get(kind)
    if raw_model is not None:
        ManifestRegistry.model_for(kind, value.get("schema_version"))
        return raw_model.model_validate(value)
    return ManifestRegistry.validate(kind, value)


def resolve_and_validate_manifest(
    kind: str,
    value: Mapping[str, Any],
    environ: Mapping[str, str],
) -> ManifestModel:
    """Validate raw structure, resolve trusted values, then validate the result."""

    if kind in _RAW_MODELS:
        validate_manifest(kind, value)
        reject_sensitive_interpolations(value)
    resolved = resolve_environment(dict(value), environ)
    return ManifestRegistry.validate(kind, resolved)

"""Versioned manifest models and registry."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal
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
)

from strixlab.config import resolve_environment
from strixlab.naming import ENV_NAME_PATTERN

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
type CMakeScalar = StrictBool | StrictInt | StrictFloat | StrictStr


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


CaptureKind = Literal[
    "cmake_cache",
    "binary_hashes",
    "ldd",
    "elf_dynamic_section",
    "compile_commands",
]


class BuildProfileV1(ManifestModel):
    schema_version: Literal[1]
    id: DashId
    source: DashId
    generator: Literal["Ninja"]
    build_type: Literal["Release", "RelWithDebInfo", "Debug"]
    environment: dict[EnvironmentKey, NulFreeString]
    cmake: dict[EnvironmentKey, CMakeScalar]
    targets: Annotated[
        list[BuildTarget], Field(min_length=1, json_schema_extra={"uniqueItems": True})
    ]
    post_build_capture: Annotated[
        list[CaptureKind], Field(min_length=1, json_schema_extra={"uniqueItems": True})
    ]

    @field_validator("targets", "post_build_capture")
    @classmethod
    def ordered_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("list entries must be unique")
        return value


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


def validate_manifest(kind: str, value: Mapping[str, Any]) -> ManifestModel:
    """Validate one already-parsed manifest through the registry."""

    return ManifestRegistry.validate(kind, value)


def resolve_and_validate_manifest(
    kind: str,
    value: Mapping[str, Any],
    environ: Mapping[str, str],
) -> ManifestModel:
    """Resolve trusted environment values, then validate the resolved manifest."""

    resolved = resolve_environment(dict(value), environ)
    return validate_manifest(kind, resolved)

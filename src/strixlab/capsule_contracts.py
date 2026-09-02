"""Closed comparison contract shared by capsule manifests and scenarios."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator

__all__ = [
    "CapsuleArmDifference",
    "CapsuleComparisonContractV1",
]

CapsuleArmDifference = Literal["candidate-id", "source-candidate", "build-output"]

_CANDIDATE_ONLY: tuple[CapsuleArmDifference, ...] = ("candidate-id",)
_SOURCE_BUILD: tuple[CapsuleArmDifference, ...] = (
    "candidate-id",
    "source-candidate",
    "build-output",
)


class CapsuleComparisonContractV1(BaseModel):
    """The only generic paired-latency comparison policy admitted by capsule v1."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    policy: Literal["paired-latency-log-bootstrap-v1"]
    protected_regression_bps: Annotated[StrictInt, Field(ge=0, le=10_000)] | None
    permitted_arm_differences: Annotated[
        tuple[CapsuleArmDifference, ...],
        Field(
            json_schema_extra={
                "enum": [
                    ["candidate-id"],
                    ["candidate-id", "source-candidate", "build-output"],
                ]
            }
        ),
    ]

    @field_validator("permitted_arm_differences", mode="before")
    @classmethod
    def _yaml_sequence(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _closed_arm_differences(self) -> Self:
        if self.permitted_arm_differences not in {_CANDIDATE_ONLY, _SOURCE_BUILD}:
            raise ValueError("permitted arm differences are not a canonical capsule v1 tuple")
        return self

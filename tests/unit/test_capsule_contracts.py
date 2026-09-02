from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from strixlab.capsule_contracts import CapsuleComparisonContractV1

_ROOT = Path(__file__).parents[2]


def _value(**updates: Any) -> dict[str, Any]:
    return {
        "policy": "paired-latency-log-bootstrap-v1",
        "protected_regression_bps": 500,
        "permitted_arm_differences": [
            "candidate-id",
            "source-candidate",
            "build-output",
        ],
        **updates,
    }


@pytest.mark.parametrize(
    "differences",
    [
        ["candidate-id"],
        ["candidate-id", "source-candidate", "build-output"],
    ],
)
@pytest.mark.parametrize("protected_regression_bps", [None, 0, 500, 10_000])
def test_only_canonical_contracts_validate(
    differences: list[str], protected_regression_bps: int | None
) -> None:
    contract = CapsuleComparisonContractV1.model_validate(
        _value(
            permitted_arm_differences=differences,
            protected_regression_bps=protected_regression_bps,
        )
    )

    assert contract.policy == "paired-latency-log-bootstrap-v1"
    assert contract.permitted_arm_differences == tuple(differences)
    assert contract.protected_regression_bps == protected_regression_bps


@pytest.mark.parametrize(
    "differences",
    [
        [],
        ["source-candidate"],
        ["build-output"],
        ["candidate-id", "build-output"],
        ["candidate-id", "source-candidate"],
        ["source-candidate", "candidate-id", "build-output"],
        ["candidate-id", "candidate-id"],
    ],
)
def test_noncanonical_difference_tuples_are_rejected(differences: list[str]) -> None:
    with pytest.raises(ValidationError):
        CapsuleComparisonContractV1.model_validate(_value(permitted_arm_differences=differences))


@pytest.mark.parametrize("value", [-1, 10_001, True, 500.0, "500"])
def test_protected_regression_bps_is_nullable_bounded_strict_int(value: object) -> None:
    with pytest.raises(ValidationError):
        CapsuleComparisonContractV1.model_validate(_value(protected_regression_bps=value))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.pop("policy"),
        lambda value: value.pop("protected_regression_bps"),
        lambda value: value.pop("permitted_arm_differences"),
        lambda value: value.__setitem__("policy", "topk-paired-log-bootstrap-v1"),
        lambda value: value.__setitem__("bootstrap_replicates", 4096),
    ],
)
def test_contract_is_required_closed_and_has_no_statistics_knobs(mutate: Any) -> None:
    value = _value()
    mutate(value)
    with pytest.raises(ValidationError):
        CapsuleComparisonContractV1.model_validate(value)


def test_contract_is_frozen() -> None:
    contract = CapsuleComparisonContractV1.model_validate(_value())
    with pytest.raises(ValidationError, match="frozen"):
        contract.protected_regression_bps = None


def test_design_locks_bootstrap_seed_and_closed_normalization() -> None:
    design = (_ROOT / "docs" / "design.md").read_text(encoding="utf-8")
    seed = design.split("length_frame(\n", 1)[1].split("\n)\n```", 1)[0]
    labels = (
        '"policy_id"',
        '"baseline_record_sha256"',
        '"candidate_record_sha256"',
        '"case_id"',
        '"mode"',
        '"replicate"',
        '"draw"',
    )
    assert [seed.index(label) for label in labels] == sorted(seed.index(label) for label in labels)
    assert "case_set" not in seed
    assert "comparison_contract_sha256" not in seed

    source_row = next(
        line for line in design.splitlines() if line.startswith("| `source-candidate`")
    )
    assert "manifest.build.source_id" in source_row
    assert "manifest.build.source_commit" in source_row
    assert "are not normalized" in source_row
    for field in (
        "preparation_id",
        "request_digest",
        "root_tree",
        "content_tree_id",
        "candidate_id",
        "diff_file",
        "diff_sha256",
        "diff_size_bytes",
        "status",
        "patches",
        "created_at",
    ):
        assert field in source_row

    build_row = next(line for line in design.splitlines() if line.startswith("| `build-output`"))
    for field in (
        "recipe_id",
        "artifact_set_id",
        "targets[*].target_id",
        "inspections",
        "capture_tools",
        "cmake_cache_sha256",
        "compile_commands_sha256",
        "canonical_record_sha256",
        "executable_sha256",
    ):
        assert field in build_row
    assert "target topology" in build_row
    assert "profile_sha256" in build_row
    assert "requested targets" in build_row

    assert "candidate_median / baseline_median > 1 + bps / 10_000" in design
    assert "only a provisional aggregate `improvement` to `mixed`" in design

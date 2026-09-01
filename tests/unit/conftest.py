from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from _model_fixtures import build_verified_receipt

from strixlab import models


@pytest.fixture
def verified_receipt() -> Callable[..., models.ModelReceiptV1]:
    return build_verified_receipt


@pytest.fixture
def model_value() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": "qwen35-2b-smoke",
        "registry_status": "registered",
        "base_model": {
            "repository": "Qwen/Qwen3.5-2B",
            "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
            "license": "Apache-2.0",
        },
        "architecture": {
            "family": "qwen3_5",
            "moe": False,
            "gated_deltanet": True,
            "full_attention": True,
            "qsa": False,
            "mtp": True,
            "vision": True,
        },
        "artifact": {
            "format": "gguf",
            "file": {
                "repository": "bartowski/Qwen_Qwen3.5-2B-GGUF",
                "revision": "7d26695454df6de5fbcce2e58681e62dae06ce43",
                "filename": "Qwen_Qwen3.5-2B-Q4_K_M.gguf",
                "local_path": "${MODELS}/qwen35-2b/Qwen_Qwen3.5-2B-Q4_K_M.gguf",
                "size_bytes": 1396198496,
                "sha256": "57a1085840f497d764a7fc5d346922dbde961efb54cc792ea81d694fd846a1d8",
            },
            "metadata_predicates": [
                {"key": "general.architecture", "value_type": "STRING", "scalar_value": "qwen35"}
            ],
        },
        "sidecars": [],
        "quantization": {
            "format_family": "Q4_K",
            "storage_format": "gguf",
            "tensor_policy_id": "unknown",
            "tensor_policy_source": "unknown",
            "calibration_method": "unknown",
            "calibration_source": "unknown",
            "calibration_hash": "unknown",
        },
        "execution": {"verification_status": "unverified"},
    }


@pytest.fixture
def draft_model_value() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": "qwen38-27b",
        "registry_status": "draft",
        "artifact": {"format": "gguf", "file": {}},
        "quantization": {
            "format_family": "unknown",
            "storage_format": "gguf",
            "tensor_policy_id": "unknown",
            "tensor_policy_source": "unknown",
            "calibration_method": "unknown",
            "calibration_source": "unknown",
            "calibration_hash": "unknown",
        },
        "execution": {"verification_status": "unverified"},
        "draft_reason": "no reviewed conversion repository/revision or recipe is pinned yet",
    }


@pytest.fixture
def source_value() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": "strix-llama",
        "kind": "git",
        "url": "https://github.com/halo-box/strix-llama.cpp.git",
        "commit": "ca94157f70a2776e8da6b6849b50b45a083d0478",
        "branch_hint": "master",
        "submodules": False,
        "adapter": "llama_cpp",
        "allowed_dirty_state": False,
    }


@pytest.fixture
def build_value() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": "hip-rocm10-gfx1151",
        "source": "strix-llama",
        "generator": "Ninja",
        "build_type": "Release",
        "toolchain": {
            "mode": "rocm",
            "prefixes": {"system": "/usr", "rocm": "/opt/rocm-10"},
            "cmake": "/usr/bin/cmake",
            "ninja": "/usr/bin/ninja",
            "c_compiler": "/opt/rocm-10/bin/amdclang",
            "cxx_compiler": "/opt/rocm-10/bin/amdclang++",
            "hip_compiler": "/opt/rocm-10/bin/amdclang++",
            "rocm_prefix": "/opt/rocm-10",
            "path": ["/opt/rocm-10/bin", "/usr/bin"],
        },
        "execution": {
            "jobs": 8,
            "timeouts": {
                "discovery_seconds": 30.0,
                "configure_seconds": 300.0,
                "build_seconds": 1800.0,
                "inspection_seconds": 30.0,
                "capability_seconds": 30.0,
            },
        },
        "environment": {
            "path_lists": {"ROCM_PATH": "/opt/rocm-10"},
            "literals": {"SOURCE_DATE_EPOCH": "0"},
        },
        "cmake": {
            "GGML_HIP": True,
            "AMDGPU_TARGETS": "gfx1151",
            "GGML_NATIVE": False,
        },
        "targets": ["llama-bench", "llama-server", "test-backend-ops"],
    }


@pytest.fixture
def capsule_value() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "id": "rocm10-topk-capsule",
        "candidate": "baseline-hip",
        "machine": "strix-halo-128g",
        "build": {
            "source_id": "rocm10-topk",
            "source_commit": "a" * 40,
            "toolchain_mode": "rocm",
            "gfx_target": "gfx1151",
            "target": "topk-capsule",
        },
        "contract": {
            "protocol": "native-capsule-v1",
            "scenario_sha256": "b" * 64,
        },
        "timeouts": {
            "describe_seconds": 30.0,
            "correctness_seconds": 300.0,
            "benchmark_seconds": 1800.0,
        },
    }

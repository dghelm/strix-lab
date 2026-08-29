from __future__ import annotations

from typing import Any

import pytest


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

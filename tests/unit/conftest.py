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
        "environment": {"ROCM_PATH": "${ROCM10_PATH}"},
        "cmake": {
            "GGML_HIP": True,
            "AMDGPU_TARGETS": "gfx1151",
            "GGML_NATIVE": False,
        },
        "targets": ["llama-bench", "llama-server", "test-backend-ops"],
        "post_build_capture": [
            "cmake_cache",
            "binary_hashes",
            "ldd",
            "elf_dynamic_section",
            "compile_commands",
        ],
    }

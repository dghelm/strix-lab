"""Checked-in self-service input configs and their cross-binding agreement.

PILOT-002 PR1 ships a checked-in source lock and ROCm/gfx1151 build profile so a
fresh user can reproduce the source preparation and build the smoke suite already
assumes. These tests validate each config through the real registry and assert the
source, build, model, machine, and suite inputs agree on the source id, the pinned
commit, the gfx1151 target, and the three required build targets — offline, with no
GPU, weights, or network.
"""

from __future__ import annotations

from pathlib import Path

from strixlab.config import read_manifest
from strixlab.manifests import (
    BuildProfileV1,
    MachineProfileV1,
    ModelManifestV1,
    SourceLockV1,
    SuiteManifestV1,
    resolve_and_validate_manifest,
    validate_manifest,
)
from strixlab.suites import TARGET_BACKEND_OPS, TARGET_LLAMA_BENCH, TARGET_LLAMA_SERVER

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE = _REPO_ROOT / "configs" / "sources" / "strix-llama.yaml"
_ROCM72_BUILD = _REPO_ROOT / "configs" / "builds" / "hip-rocm-gfx1151.yaml"
_ROCM10_BUILD = _REPO_ROOT / "configs" / "builds" / "hip-rocm10-gfx1151.yaml"
_SUITE = _REPO_ROOT / "configs" / "suites" / "smoke-qwen35.yaml"
_MACHINE = _REPO_ROOT / "configs" / "machines" / "strix-halo-128g.yaml"
_MODELS_DIR = _REPO_ROOT / "configs" / "models"

_PINNED_COMMIT = "ca94157f70a2776e8da6b6849b50b45a083d0478"
_SOURCE_ID = "strix-llama"
_GFX_TARGET = "gfx1151"
_REQUIRED_TARGETS = frozenset({TARGET_BACKEND_OPS, TARGET_LLAMA_SERVER, TARGET_LLAMA_BENCH})
_BACKEND_PARAMS_REGEX = (
    r"^(type_a=f32,type_b=f32,m=1,n=1,k=2048,bs=\[1,1\],nr=\[1,1\],"
    r"per=\[0,1,2,3\],k_v=0,o=1,src_overlap=0|"
    r"type_a=f32,type_b=f32,n_mats=4,n_used=1,b=0,m=512,n=1,k=256|"
    r"type=f32,ne=\[1,2,1,3\],k=1,ties=1|"
    r"type=f32,ne=\[16,10,10,10\],order=0|"
    r"type=f32,ne=\[64,5,4,3\],v=0,eps=0\.000000,inplace=0)$"
)


def _source() -> SourceLockV1:
    value = validate_manifest("source-lock", read_manifest(_SOURCE))
    assert isinstance(value, SourceLockV1)
    return value


def _build(path: Path = _ROCM72_BUILD) -> BuildProfileV1:
    # ``resolve_and_validate_manifest`` validates the raw profile before resolving; the
    # pilot profile pins literal paths, so an empty environment leaves them unchanged.
    value = resolve_and_validate_manifest("build", read_manifest(path), {})
    assert isinstance(value, BuildProfileV1)
    return value


def _suite() -> SuiteManifestV1:
    value = validate_manifest("suite", read_manifest(_SUITE))
    assert isinstance(value, SuiteManifestV1)
    return value


def _machine() -> MachineProfileV1:
    value = validate_manifest("machine", read_manifest(_MACHINE))
    assert isinstance(value, MachineProfileV1)
    return value


def test_source_lock_pins_reviewed_strix_llama_constants() -> None:
    source = _source()
    assert source.id == _SOURCE_ID
    assert source.kind == "git"
    assert source.url == "https://github.com/halo-box/strix-llama.cpp.git"
    assert source.commit == _PINNED_COMMIT
    assert source.branch_hint == "master"
    assert source.submodules is False
    assert source.adapter == "llama_cpp"
    # v1 refuses any dirty-state allowance.
    assert source.allowed_dirty_state is False


def test_build_profile_targets_rocm_gfx1151_on_stable_paths() -> None:
    build = _build()
    assert build.source == _SOURCE_ID
    assert build.generator == "Ninja"
    assert build.build_type == "Release"

    assert build.toolchain.mode == "rocm"
    # The pilot machine's stable ROCm 7.2.4 root, never the test-only /opt/rocm-10 fixture.
    assert build.toolchain.rocm_prefix == "/opt/rocm"
    assert build.toolchain.prefixes["rocm"] == "/opt/rocm"
    all_paths = [
        build.toolchain.cmake,
        build.toolchain.ninja,
        build.toolchain.c_compiler,
        build.toolchain.cxx_compiler,
        build.toolchain.hip_compiler,
        build.toolchain.rocm_prefix,
        *build.toolchain.path,
        *build.toolchain.prefixes.values(),
        *build.environment.path_lists.values(),
    ]
    assert all("rocm-10" not in str(path) for path in all_paths)

    # The ROCm/gfx1151 knobs use the existing manifest spelling and scalar types.
    assert build.cmake["GGML_HIP"] is True
    assert build.cmake["AMDGPU_TARGETS"] == _GFX_TARGET
    assert build.cmake["GGML_NATIVE"] is False
    assert build.cmake["BUILD_SHARED_LIBS"] is False
    assert build.cmake["CMAKE_HIP_COMPILER_ROCM_ROOT"] == "/opt/rocm"
    assert set(build.targets) == _REQUIRED_TARGETS


def test_rocm10_build_profile_is_an_explicit_side_by_side_lane() -> None:
    control = _build()
    build = _build(_ROCM10_BUILD)

    assert build.id == "hip-rocm10-gfx1151"
    assert build.source == control.source == _SOURCE_ID
    assert build.generator == control.generator == "Ninja"
    assert build.build_type == control.build_type == "Release"
    assert build.execution == control.execution
    assert build.targets == control.targets

    assert build.toolchain.mode == "rocm"
    assert build.toolchain.prefixes == {"system": "/usr", "rocm": "/opt/rocm-10"}
    assert build.toolchain.rocm_prefix == "/opt/rocm-10"
    assert build.toolchain.c_compiler == "/opt/rocm-10/bin/amdclang"
    assert build.toolchain.cxx_compiler == "/opt/rocm-10/bin/amdclang++"
    assert build.toolchain.hip_compiler == "/opt/rocm-10/bin/amdclang++"
    assert build.toolchain.path == ["/opt/rocm-10/bin", "/usr/bin"]
    assert build.environment.path_lists == {
        "ROCM_PATH": "/opt/rocm-10",
        "LD_LIBRARY_PATH": "/opt/rocm-10/lib",
    }

    # Both arms ask CMake for the same build. Only the selected ROCm root differs.
    common_cmake = {
        "GGML_HIP": True,
        "AMDGPU_TARGETS": _GFX_TARGET,
        "GGML_NATIVE": False,
        "BUILD_SHARED_LIBS": False,
    }
    for name, expected in common_cmake.items():
        assert build.cmake[name] == control.cmake[name] == expected
    assert control.cmake["CMAKE_HIP_COMPILER_ROCM_ROOT"] == "/opt/rocm"
    assert build.cmake["CMAKE_HIP_COMPILER_ROCM_ROOT"] == "/opt/rocm-10"


def test_source_build_suite_agree_on_source_commit_and_gfx() -> None:
    source = _source()
    suite = _suite()
    machine = _machine()

    for build_path in (_ROCM72_BUILD, _ROCM10_BUILD):
        build = _build(build_path)
        # Source id agreement across source lock, build profile, and suite requirement.
        assert source.id == build.source == suite.build.source_id == _SOURCE_ID
        # Toolchain mode agreement between each profile and the suite requirement.
        assert build.toolchain.mode == suite.build.toolchain_mode == "rocm"
        # gfx1151 agreement across build CMake, suite target, and machine expectation.
        assert build.cmake["AMDGPU_TARGETS"] == suite.build.gfx_target
        assert build.cmake["AMDGPU_TARGETS"] == machine.expect.gpu_arch
        # Both build profiles produce exactly the three targets the suite composes.
        assert set(build.targets) == _REQUIRED_TARGETS

    # Pinned commit agreement between the source lock and the suite build requirement.
    assert source.commit == suite.build.source_commit == _PINNED_COMMIT
    assert suite.build.gfx_target == _GFX_TARGET
    # The suite binds a checked-in machine and registered model config.
    assert suite.machine == machine.id
    assert suite.correctness.backend_ops.params_regex == _BACKEND_PARAMS_REGEX
    model = resolve_and_validate_manifest(
        "model", read_manifest(_MODELS_DIR / f"{suite.model}.yaml"), {"MODELS": "/data/models"}
    )
    assert isinstance(model, ModelManifestV1)
    assert model.registry_status == "registered"

"""Execute the actual HIP kernel bodies using a CPU-only barrier/shuffle model.

This checks indexing, FP32 staging, reduction topology and dispatch contracts.
It does not emulate GPU rsqrt instructions or qualify compiled-backend parity.
"""

import resource
import shutil
import signal
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).parent
    output = tmp_path_factory.mktemp("gdn-norm-host") / "test"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-pthread",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{root / 'host_tests/gdn_hip'}",
            f"-I{root}",
            str(root / "host_tests/gdn_norm.cpp"),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return output


def test_gdn_norm_contract(binary):
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("PASS: 48 Q/K rows")


def test_wave32_required(binary):
    def disable_core_dump():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    result = subprocess.run(
        [str(binary), "wrong-wave"], capture_output=True, timeout=15, preexec_fn=disable_core_dump
    )
    assert result.returncode == -signal.SIGABRT
    assert b"warpSize == 32" in result.stderr

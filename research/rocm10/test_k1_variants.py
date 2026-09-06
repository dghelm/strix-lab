"""Compile and execute real variant kernel bodies through a synthetic host HIP.

No SDK or GPU access. Wave barriers require uniform physical-lane participation;
a subprocess timeout catches divergent shuffle/barrier deadlocks.
"""

import resource
import shutil
import signal
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def host_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    root = Path(__file__).parent
    binary = tmp_path_factory.mktemp("k1-variants-host") / "test"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-pthread",
            "-Wall",
            "-Wextra",
            "-Werror",
            f"-I{root / 'host_tests'}",
            f"-I{root}",
            str(root / "host_tests/k1_variants.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    return binary


def test_variants_correctness(host_binary):
    result = subprocess.run([str(host_binary)], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("PASS: 144 rows")


@pytest.mark.parametrize("argument", ["wrong-wave", "wrong-onewave"])
def test_wave_requires_32_lanes(host_binary, argument):
    def disable_core_dump():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    result = subprocess.run(
        [str(host_binary), argument],
        capture_output=True,
        timeout=15,
        preexec_fn=disable_core_dump,
    )
    assert result.returncode == -signal.SIGABRT
    assert b"warpSize == 32" in result.stderr

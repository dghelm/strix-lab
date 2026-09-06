"""Independent host oracle; native dependency failures are required test failures."""

import hashlib
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SHAPES = [
    (1, 128, 1),
    (1, 4096, 20),
    (2, 16381, 8),
    (8, 65536, 20),
    (32, 16384, 64),
    (128, 4096, 256),
    (1, 262144, 64),
    (2, 257, 256),
    (8, 16381, 20),
    (32, 65536, 64),
    (128, 262144, 20),
]
FIXED_DIGESTS = [
    "32599c17de87e67b515f62042724b0c790de5f86f65fe4104eb842689d89e32c",
    "8ec9a0380f8c58ca4adc6bc31180e092d6df9d2c7de4ce77bdf6ed46131e62fa",
    "320d12096ce27d29e242df01ef766748ab87556ce68935ec5cd5408b3230ccd7",
    "2bc47ef8391e0707352e821fde6d24392d7ea7ba5720ffafbf70e114cbbacd2a",
    "4e41f20f75f2b17deea9b8920c5667a46666143b721c5c7cf6b3717c2ac2f79b",
]


def run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=120).stdout


@pytest.fixture(scope="module")
def native(tmp_path_factory: pytest.TempPathFactory) -> Path:
    assert shutil.which("cmake"), "TOPK host tests require CMake >=3.20 (see native/topk/README.md)"
    build = tmp_path_factory.mktemp("topk-native")
    # Configuration fails explicitly if the C++ compiler or OpenSSL development files are absent.
    run("cmake", "-S", str(ROOT / "native/topk"), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release")
    run("cmake", "--build", str(build), "--parallel", "2")
    return build / "topk_reference_test"


def test_native_failure_vectors(native: Path) -> None:
    run(str(native))


def test_exact_matrix(native: Path) -> None:
    expected = []
    for i, (rows, columns, k) in enumerate(SHAPES):
        prefix, case_set = ("train", "training") if i < 6 else ("eval", "evaluation")
        expected.append(f"{prefix}-r{rows}-c{columns}-k{k} {case_set} {rows} {columns} {k}")
    assert run(str(native), "matrix").splitlines() == expected


def oracle(ordinal: int, family: int) -> list[int]:
    rows, columns, _ = SHAPES[ordinal - 1]
    state = 0x5354524958544F50 ^ (ordinal << 8) ^ family
    mask = (1 << 64) - 1
    words = []
    for i in range(rows * columns):
        if family in (1, 2):
            state = (state + 0x9E3779B97F4A7C15) & mask
            z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & mask
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
            z ^= z >> 31
            if family == 1:
                word = ((z >> 63) << 31) | (126 << 23) | (z & 0x7FFFFF)
            else:
                word = [0xC0400000, 0xBF800000, 0x80000000, 0, 0x3F800000, 0x40400000][z % 6]
        elif family in (3, 4):
            j = i % columns
            word = struct.unpack("<I", struct.pack("<f", j if family == 3 else columns - j))[0]
        else:
            word = [
                0xFF800000,
                0xBF800000,
                0x80000000,
                0,
                0x3F800000,
                0x7F800000,
                0xC0000000,
                0x40000000,
            ][i % 8]
        words.append(word)
    return words


@pytest.mark.parametrize("ordinal", [1, 3, 8])
@pytest.mark.parametrize("family", range(1, 6))
def test_generator_digest_and_selection(native: Path, ordinal: int, family: int) -> None:
    # Includes multi-row non-power-of-two rows, near-full K, and continuous PRNG state.
    identity, data, selection = run(str(native), str(ordinal), str(family)).splitlines()
    seed, digest = identity.split()
    assert int(seed) == 0x5354524958544F50 ^ (ordinal << 8) ^ family
    words = oracle(ordinal, family)
    assert list(map(int, data.split())) == words
    rows, columns, k = SHAPES[ordinal - 1]
    payload = b"strixlab.topk.input.v1\0" + struct.pack("<QQQ", rows, columns, k)
    payload += struct.pack(f"<{len(words)}I", *words)
    assert digest == hashlib.sha256(payload).hexdigest()
    if ordinal == 1:
        assert digest == FIXED_DIGESTS[family - 1]
    expected = []
    for row in range(rows):
        row_words = words[row * columns : (row + 1) * columns]
        values = struct.unpack(f"<{columns}f", struct.pack(f"<{columns}I", *row_words))
        # Full Python sort is independent of native partial_sort.
        indices = sorted(range(columns), key=lambda j: (-values[j], j))[:k]
        expected.extend(f"{j}:{row_words[j]}" for j in indices)
    assert selection.split() == expected

"""Inert host-only launcher checks; never run Bubblewrap, an SDK, or GPU code."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "research_launcher", Path(__file__).with_name("run.py")
)
assert spec is not None and spec.loader is not None
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_host_process_success(tmp_path):
    before = set(os.listdir("/proc/self/fd"))
    result = m.run_process([sys.executable, "-I", "-c", 'print("inert")'], tmp_path, 3)
    assert result["returncode"] == 0 and result["failure"] is None
    assert (tmp_path / "stdout.log").read_bytes() == b"inert\n"
    assert set(os.listdir("/proc/self/fd")) == before


def test_timeout_terminates_process(tmp_path):
    result = m.run_process(
        [sys.executable, "-I", "-c", "import time; time.sleep(10)"], tmp_path, 0.05
    )
    assert result["failure"] == "wall-time-limit"
    assert result["returncode"] != 0


def test_log_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "MAX_LOG", 100)
    result = m.run_process([sys.executable, "-I", "-c", 'print("x"*10000)'], tmp_path, 3)
    assert result["failure"] == "log-size-limit"
    assert (tmp_path / "stdout.log").stat().st_size == 100


def test_output_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "MAX_FILE", 10)
    (tmp_path / "big").write_bytes(b"x" * 11)
    with pytest.raises(RuntimeError, match="file-size"):
        m.run_process([sys.executable, "-I", "-c", "import time; time.sleep(10)"], tmp_path, 3)


def test_directory_swap_is_not_followed(tmp_path, monkeypatch):
    child = tmp_path / "dir"
    child.mkdir()
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    original = os.open

    def open_(path, flags, *a, **kw):
        if path == "dir" and kw.get("dir_fd") is not None:
            child.rmdir()
            child.symlink_to(outside, target_is_directory=True)
        return original(path, flags, *a, **kw)

    monkeypatch.setattr(os, "open", open_)
    with pytest.raises(OSError):
        m.tree_files(tmp_path)


def test_fixed_argv_boundary(tmp_path):
    for action in (
        "check",
        "compile-smoke",
        "compile-topk",
        "compile-k1",
        "run-smoke",
        "run-topk",
        "run-k1",
    ):
        args = m.bwrap_argv(action, tmp_path, tmp_path / "native", "1:2", {}, {}, True)
        assert args[0] == "/usr/bin/bwrap"
        assert "/native" in args
        assert "LD_LIBRARY_PATH" not in args
        assert "--diagnostic" in args
        assert ("--dev-bind" in args) == action.startswith("run-")
        assert "--gpu-devices" in args if action.startswith("run-") else "--gpu-devices" not in args
    assert "/native/baseline/adapter/hip_bitonic_topk.cu" in m.command_for("compile-topk")


@pytest.mark.parametrize("failure", [None, "process", "after"])
def test_receipt_inside_lease_and_release(tmp_path, monkeypatch, failure):
    for name in ("getuid", "geteuid", "getgid", "getegid"):
        monkeypatch.setattr(m.os, name, lambda: 1000)
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "preflight_exec.py").write_text("# inert")
    native = tmp_path / "native"
    native.mkdir()
    binary = tmp_path / "binary"
    binary.write_bytes(b"\x7fELFinert")
    expected = SimpleNamespace(root=SimpleNamespace(identity=SimpleNamespace(dev=1, ino=2)))
    monkeypatch.setattr(m, "baseline", lambda: expected)
    active = False

    @contextmanager
    def lease(path):
        nonlocal active
        active = True
        try:
            yield SimpleNamespace(path=path, status="acquired", reason=None, acquired=True)
        finally:
            active = False

    monkeypatch.setattr(m, "exclusive_lock", lease)

    def inspect(expected, output, label):
        assert active
        if failure == "after" and label == "after":
            raise RuntimeError("post failure")
        return {"test_label": label}

    monkeypatch.setattr(m, "inspect_baseline", inspect)
    monkeypatch.setattr(m, "gpu_evidence", lambda: {"nodes": {}})

    def process(*args):
        assert active
        return {"failure": None, "returncode": 1 if failure == "process" else 0}

    monkeypatch.setattr(m, "run_process", process)
    original = m.write_json

    def write(path, data):
        if path.name == "phase.json":
            assert active
        if path.name == "lock-release.json":
            assert not active
        return original(path, data)

    monkeypatch.setattr(m, "write_json", write)
    args = argparse.Namespace(
        action="run-smoke",
        output=tmp_path / "out",
        fixtures=fixtures,
        native=native,
        binary=binary,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        diagnostic=False,
    )
    result = m.main(args)
    assert result == (0 if failure is None else 1)
    receipt = json.loads((args.output / "phase.json").read_text())
    assert receipt["gpu_lock"]["held_at_receipt"] is True
    assert json.loads((args.output / "lock-release.json").read_text())["released"] is True
    assert not active


def test_k1_argv_and_cli(monkeypatch, tmp_path):
    command = m.command_for("compile-k1")
    assert command[0] == "/sdk/lib/llvm/bin/clang++"
    assert "-O3" in command and "--offload-arch=gfx1151" in command
    assert command[-2:] == ["-o", "/work/topk-k1-compare"]
    assert "/input/topk_k1_compare.cpp" in command
    assert "/native/reference.cpp" in command
    assert "/native/baseline/adapter/hip_bitonic_topk.cu" in command
    assert "-lcrypto" in command and "-I/native" in command
    assert m.command_for("run-k1") == ["/work/topk-k1-compare"]
    for action in ("compile-k1", "run-k1"):
        monkeypatch.setattr(sys, "argv", ["run.py", action, "--output", str(tmp_path / "out")])
        assert m.parse_args().action == action


@pytest.mark.parametrize("missing_header", [False, True])
def test_k1_header_copy_and_pin(tmp_path, monkeypatch, missing_header):
    for name in ("getuid", "geteuid", "getgid", "getegid"):
        monkeypatch.setattr(m.os, name, lambda: 1000)
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    contents = {
        "preflight_exec.py": b"# inert preflight",
        "topk_k1_compare.cpp": b"// inert source",
        "topk_k1.hpp": b"// pinned K1 header",
    }
    for name, data in contents.items():
        if name != "topk_k1.hpp" or not missing_header:
            (fixtures / name).write_bytes(data)
    native = tmp_path / "native"
    native.mkdir()
    expected = SimpleNamespace(root=SimpleNamespace(identity=SimpleNamespace(dev=1, ino=2)))
    monkeypatch.setattr(m, "baseline", lambda: expected)
    monkeypatch.setattr(m, "inspect_baseline", lambda *args: {"synthetic": True})
    invoked = False

    def process(argv, output, seconds):
        nonlocal invoked
        invoked = True
        assert seconds == 300 and "--dev-bind" not in argv and "--gpu-devices" not in argv
        assert argv[argv.index("--phase") + 1] == "compile"
        for name, data in contents.items():
            assert (output / "input" / name).read_bytes() == data
        (output / "work" / "topk-k1-compare").write_bytes(b"\x7fELFinert")
        return {"failure": None, "returncode": 0}

    monkeypatch.setattr(m, "run_process", process)
    args = argparse.Namespace(
        action="compile-k1",
        output=tmp_path / "out",
        fixtures=fixtures,
        native=native,
        binary=None,
        binary_sha256=None,
        diagnostic=False,
    )
    assert m.main(args) == (1 if missing_header else 0)
    assert invoked is not missing_header
    receipt = json.loads((args.output / "phase.json").read_text())
    if not missing_header:
        for name, data in contents.items():
            assert receipt["inputs"][name]["sha256"] == hashlib.sha256(data).hexdigest()
        assert any(item["path"] == "work/topk-k1-compare" for item in receipt["outputs"])
    else:
        assert receipt["status"] == "incomplete"


def test_k1_run_copies_only_pinned_binary(tmp_path, monkeypatch):
    for name in ("getuid", "geteuid", "getgid", "getegid"):
        monkeypatch.setattr(m.os, name, lambda: 1000)
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()
    (fixtures / "preflight_exec.py").write_text("# inert")
    native = tmp_path / "native"
    native.mkdir()
    binary = tmp_path / "prior"
    payload = b"\x7fELFinert"
    binary.write_bytes(payload)
    expected = SimpleNamespace(root=SimpleNamespace(identity=SimpleNamespace(dev=1, ino=2)))
    monkeypatch.setattr(m, "baseline", lambda: expected)
    monkeypatch.setattr(m, "inspect_baseline", lambda *args: {"synthetic": True})

    @contextmanager
    def lease(path):
        yield SimpleNamespace(path=path, status="acquired", reason=None, acquired=True)

    monkeypatch.setattr(m, "exclusive_lock", lease)
    monkeypatch.setattr(m, "gpu_evidence", lambda: {"nodes": {}})

    def process(argv, output, seconds):
        assert seconds == 30
        assert argv[-1] == "/work/topk-k1-compare"
        assert "--gpu-devices" in argv and "--dev-bind" in argv
        assert (output / "work" / "topk-k1-compare").read_bytes() == payload
        assert (output / "work" / "topk-k1-compare").stat().st_ino != binary.stat().st_ino
        assert sorted(p.name for p in (output / "input").iterdir()) == ["preflight_exec.py"]
        return {"failure": None, "returncode": 0}

    monkeypatch.setattr(m, "run_process", process)
    args = argparse.Namespace(
        action="run-k1",
        output=tmp_path / "out",
        fixtures=fixtures,
        native=native,
        binary=binary,
        binary_sha256=hashlib.sha256(payload).hexdigest(),
        diagnostic=False,
    )
    assert m.main(args) == 0
    receipt = json.loads((args.output / "phase.json").read_text())
    assert receipt["inputs"]["topk-k1-compare"]["sha256"] == hashlib.sha256(payload).hexdigest()

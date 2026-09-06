"""Bounded private ROCm research phases; this is not a capsule or SDK admission API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import selectors
import signal
import stat
import subprocess
import sys
import time
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import Any

from strixlab.locks import exclusive_lock
from strixlab.rocm_extract import GnuQuarantineResultV1
from strixlab.rocm_prefix import (
    _MAX_EVIDENCE_BYTES,
    PrefixInventoryV1,
    compare_prefixes,
    inspect_prefix,
    resolve_inventory_links,
)
from strixlab.serialization import canonical_json_bytes

STAGE = Path(
    "/home/dgh/.local/share/strixlab/toolchains/"
    ".stage-rocm-10.0.0-gfx1151-20260905T232532Z-99a93ed7"
)
RESULT_SHA256 = "08f153a04c921e4d8a9429e3e92b58a43c7134e710259d6e66eb01c37c927eb4"
ARCHIVE_SHA256 = "4feabd9f2da72352df37f6d714a54847d3fe913c0341fbe2a6542c1164024baf"
GPU_LOCK = Path("/tmp/strixlab-gpu.lock")
SYSFS = ("/sys/devices", "/sys/class/kfd", "/sys/class/drm", "/sys/bus/pci", "/sys/dev/char")
GPU_NODES = ("/dev/kfd", "/dev/dri/renderD128")
NAMESPACES = ("user", "mnt", "pid", "ipc", "net")
CHUNK = 65536
MAX_FILE = 512 * 1024**2
MAX_OUTPUT = 1024**3
MAX_LOG = 16 * 1024**2
MAX_FILES = 4096
HOST_ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TMPDIR": "/tmp"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def file_bytes(path: Path, limit: int) -> bytes:
    """Bounded, regular, no-follow input; reject observed descriptor/name drift."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= limit, "invalid input file")
        pieces = []
        remaining = before.st_size
        while remaining:
            data = os.read(fd, min(CHUNK, remaining))
            require(bool(data), "short input read")
            pieces.append(data)
            remaining -= len(data)
        require(not os.read(fd, 1), "input grew")
        require(identity(os.fstat(fd)) == identity(before), "input descriptor changed")
        require(
            identity(path.stat(follow_symlinks=False)) == identity(before), "input name changed"
        )
        return b"".join(pieces)
    finally:
        os.close(fd)


def write_bytes(path: Path, data: bytes, mode: int = 0o600) -> dict[str, Any]:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode)
    with os.fdopen(fd, "wb") as output:
        output.write(data)
    return {"path": path.name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_json(path: Path, value: Any) -> dict[str, Any]:
    return write_bytes(path, canonical_json_bytes(value))


def baseline() -> PrefixInventoryV1:
    raw = file_bytes(STAGE / "gnu-quarantine-result-v1.json", 128 * 1024**2)
    require(hashlib.sha256(raw).hexdigest() == RESULT_SHA256, "quarantine result digest mismatch")
    result = GnuQuarantineResultV1.model_validate_json(raw)
    require(result.archive.observed_sha256 == ARCHIVE_SHA256, "archive digest mismatch")
    return result.inventory


def inspect_baseline(expected: PrefixInventoryV1, output: Path, label: str) -> dict[str, Any]:
    require(_MAX_EVIDENCE_BYTES == 256 * 1024**2, "wrong prefix evidence capacity/source")
    fd = os.open(STAGE, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        current = inspect_prefix(fd, "prefix-2")
        after = os.fstat(fd)
        require((before.st_dev, before.st_ino) == (after.st_dev, after.st_ino), "stage changed")
        named = STAGE.stat(follow_symlinks=False)
        require((after.st_dev, after.st_ino) == (named.st_dev, named.st_ino), "stage rebound")
    finally:
        os.close(fd)
    record = write_bytes(output / f"prefix-{label}.json", current.canonical_bytes())
    comparison = compare_prefixes(expected, current)
    require(comparison.semantic_equal, "prefix semantic drift")
    old = {entry.path: entry for entry in (expected.root, *expected.entries)}
    for entry in (current.root, *current.entries):
        previous = old[entry.path]
        require(
            (entry.identity.dev, entry.identity.ino)
            == (previous.identity.dev, previous.identity.ino),
            "prefix physical identity drift",
        )
        metadata = entry.metadata
        require(
            metadata.list_status == "observed"
            and metadata.list_errno is None
            and metadata.name_list_size_bytes == 0
            and metadata.names_bytes_escaped == (),
            "prefix metadata not observed empty",
        )
        require((entry.uid, entry.gid) == (1000, 1000), "prefix ownership mismatch")
        require(entry.kind == "directory" or entry.nlink == 1, "prefix link count mismatch")
    require(current.root.mode == 0o700, "prefix root mode mismatch")
    links = resolve_inventory_links(current)
    require(all(item.status == "resolved" for item in links.links), "prefix link closure failed")
    return {
        **record,
        "semantic_equal": True,
        "physical_equal": True,
        "member_count": current.member_count,
        "resolved_links": len(links.links),
        "evidence_bytes_charged": current.evidence_bytes_charged,
    }


def gpu_evidence() -> dict[str, Any]:
    pci = Path("/sys/class/drm/renderD128/device").resolve(strict=True)
    require(pci.name == "0000:c2:00.0", "unexpected GPU PCI identity")
    vendor = int((pci / "vendor").read_text().strip(), 16)
    device = int((pci / "device").read_text().strip(), 16)
    require((vendor, device) == (4098, 5510), "unexpected GPU vendor/device")
    properties = Path("/sys/class/kfd/kfd/topology/nodes/1/properties").read_text()
    require(len(properties) <= 65536, "oversized KFD properties")
    values = dict(line.split(maxsplit=1) for line in properties.splitlines() if " " in line)
    for name, value in (
        ("drm_render_minor", "128"),
        ("gfx_target_version", "110501"),
        ("vendor_id", "4098"),
        ("device_id", "5510"),
    ):
        require(values.get(name) == value, f"unexpected KFD {name}")
    devices = {}
    for path in GPU_NODES:
        node = os.stat(path, follow_symlinks=False)
        require(stat.S_ISCHR(node.st_mode), "GPU path is not a character device")
        devices[path] = [os.major(node.st_rdev), os.minor(node.st_rdev)]
    require(devices["/dev/dri/renderD128"] == [226, 128], "unexpected render node")
    return {
        "pci": pci.name,
        "vendor": vendor,
        "device": device,
        "nodes": devices,
        "kfd_properties": values,
        "target": "gfx1151",
    }


def tree_files(root: Path, *, limit: int = MAX_OUTPUT) -> list[tuple[Path, os.stat_result]]:
    """Never follow phase-output links or open special files."""
    files: list[tuple[Path, os.stat_result]] = []
    count = total = 0

    def walk(fd: int, directory: Path, depth: int) -> None:
        nonlocal count, total
        require(depth <= 64, "output depth limit")
        with os.scandir(fd) as entries:
            for entry in entries:
                count += 1
                require(count <= MAX_FILES, "output member limit")
                value = entry.stat(follow_symlinks=False)
                path = directory / entry.name
                if stat.S_ISDIR(value.st_mode):
                    child = os.open(
                        entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
                    )
                    try:
                        held = os.fstat(child)
                        require(
                            (held.st_dev, held.st_ino) == (value.st_dev, value.st_ino),
                            "output directory changed",
                        )
                        walk(child, path, depth + 1)
                    finally:
                        os.close(child)
                else:
                    require(
                        stat.S_ISREG(value.st_mode) and value.st_nlink == 1,
                        "output is not an independent regular file",
                    )
                    require(value.st_size <= MAX_FILE, "output file-size limit")
                    total += value.st_size
                    require(total <= limit, "output total-size limit")
                    files.append((path, value))

    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        walk(fd, root, 0)
    finally:
        os.close(fd)
    return files


def native_evidence(root: Path) -> list[dict[str, Any]]:
    records = []
    for path, _ in sorted(tree_files(root, limit=32 * 1024**2)):
        data = file_bytes(path, 16 * 1024**2)
        records.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return records


def command_for(action: str) -> list[str]:
    if action == "check":
        return [
            "/usr/bin/python",
            "-I",
            "-c",
            'import json; print(json.dumps({"event":"host_check_pass","sdk_executed":False}))',
        ]
    if action.startswith("run-"):
        return ["/work/hip-smoke" if action == "run-smoke" else "/work/topk-bench"]
    compiler = [
        "/sdk/lib/llvm/bin/clang++",
        "-x",
        "hip",
        "--rocm-path=/sdk",
        "--hip-path=/sdk",
        "--offload-arch=gfx1151",
        "-std=c++17",
        "-O3",
    ]
    if action == "compile-smoke":
        return [*compiler, "/input/hip_smoke.cpp", "-o", "/work/hip-smoke"]
    return [
        *compiler,
        "-I/native",
        "/input/topk_bench.cpp",
        "/native/reference.cpp",
        "/native/baseline/adapter/hip_bitonic_topk.cu",
        "-lcrypto",
        "-o",
        "/work/topk-bench",
    ]


def bwrap_argv(
    action: str,
    output: Path,
    native: Path,
    sdk_identity: str,
    namespaces: dict[str, str],
    devices: dict[str, Any],
    diagnostic: bool,
) -> list[str]:
    phase = "check" if action == "check" else "gpu" if action.startswith("run-") else "compile"
    args = [
        "/usr/bin/bwrap",
        "--unshare-all",
        "--unshare-user",
        "--disable-userns",
        "--assert-userns-disabled",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--clearenv",
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/bin",
        "/bin",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib",
        "/lib64",
        "--ro-bind",
        str(STAGE / "prefix-2"),
        "/sdk",
        "--ro-bind",
        str(output / "input"),
        "/input",
        "--ro-bind",
        str(native),
        "/native",
        "--bind",
        str(output / "work"),
        "/work",
        "--ro-bind",
        str(output / "empty"),
        "/run/empty",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--size",
        "2147483648",
        "--tmpfs",
        "/tmp",
    ]
    if phase == "gpu":
        for path in GPU_NODES:
            args += ["--dev-bind", path, path]
        for path in SYSFS:
            args += ["--ro-bind", path, path]
    args += [
        "--chdir",
        "/run/empty",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "LC_ALL",
        "C",
        "--setenv",
        "TMPDIR",
        "/tmp",
        "/usr/bin/python",
        "-I",
        "/input/preflight_exec.py",
        "--phase",
        phase,
        "--host-namespaces",
        json.dumps(namespaces, sort_keys=True),
        "--sdk-identity",
        sdk_identity,
    ]
    if phase == "gpu":
        args += ["--gpu-devices", json.dumps(devices, sort_keys=True)]
    if diagnostic:
        args += ["--diagnostic"]
    return [*args, "--", *command_for(action)]


def terminate(process: subprocess.Popen[bytes]) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=2)
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=5)


def run_process(argv: list[str], output: Path, wall_seconds: int) -> dict[str, Any]:
    started = time.monotonic()
    reason = None
    used = 0
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=HOST_ENV,
        close_fds=True,
        start_new_session=True,
    )
    try:
        with selectors.DefaultSelector() as selector:
            assert process.stdout is not None and process.stderr is not None
            for stream in (process.stdout, process.stderr):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ)
            with (
                (output / "stdout.log").open("xb") as stdout,
                (output / "stderr.log").open("xb") as stderr,
            ):
                while selector.get_map() or process.poll() is None:
                    if time.monotonic() - started > wall_seconds:
                        reason = "wall-time-limit"
                        break
                    tree_files(output)
                    for key, _ in selector.select(timeout=0.1):
                        data = os.read(key.fd, CHUNK)
                        if not data:
                            selector.unregister(key.fileobj)
                            continue
                        remaining = MAX_LOG - used
                        target = stdout if key.fileobj is process.stdout else stderr
                        target.write(data[:remaining])
                        used += min(len(data), remaining)
                        if len(data) > remaining:
                            reason = "log-size-limit"
                            break
                    if reason:
                        break
        if reason:
            terminate(process)
        else:
            process.wait(timeout=2)
    except BaseException:
        terminate(process)
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    return {
        "returncode": process.returncode,
        "failure": reason,
        "elapsed_seconds": time.monotonic() - started,
        "log_bytes": used,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("check", "compile-smoke", "run-smoke", "compile-topk", "run-topk")
    )
    parser.add_argument("--output", type=Path, required=True, help="new exclusive phase directory")
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--native", type=Path, default=Path(__file__).resolve().parents[2] / "native/topk"
    )
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--binary-sha256")
    parser.add_argument(
        "--diagnostic", action="store_true", help="SDK-only LD_DEBUG; not for timing"
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    require(
        os.getuid() == os.geteuid() == 1000 and os.getgid() == os.getegid() == 1000,
        "launcher requires operator UID/GID 1000/1000",
    )
    gpu = args.action.startswith("run-")
    require(gpu == bool(args.binary and args.binary_sha256), "run requires binary and SHA-256")
    require(
        gpu or (args.binary is None and args.binary_sha256 is None), "compile/check rejects binary"
    )
    expected = baseline()
    output = args.output.absolute()
    output.mkdir(mode=0o700)  # Exclusive: no parents or exist_ok, no cleanup/reuse.
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "scope": "private-research-phase",
        "action": args.action,
        "status": "incomplete",
        "metadata_coverage": "unknown",
        "vendor_authenticity": "unverified",
        "quarantine_result_sha256": RESULT_SHA256,
        "archive_observed_sha256": ARCHIVE_SHA256,
        "output": str(output),
        "diagnostic": args.diagnostic,
        "limits": {
            "wall_seconds": 30 if gpu else 300,
            "cpu_seconds": 20 if gpu else 240,
            "compiler_virtual_bytes": None if gpu else 16 * 1024**3,
            "per_file_bytes": MAX_FILE,
            "total_output_bytes": MAX_OUTPUT,
            "log_bytes": MAX_LOG,
            "prefix_evidence_bytes": _MAX_EVIDENCE_BYTES,
            "requested_gpu_bytes": 256 * 1024**2 if gpu else 0,
        },
    }
    resources = ExitStack()
    lock_held = False
    try:
        for name in ("input", "work", "empty"):
            (output / name).mkdir(mode=0o700)
        inputs = {}
        names = ["preflight_exec.py"]
        if args.action.startswith("compile-"):
            names.append("hip_smoke.cpp" if args.action == "compile-smoke" else "topk_bench.cpp")
        for name in names:
            inputs[name] = write_bytes(
                output / "input" / name, file_bytes(args.fixtures / name, 1024**2)
            )
        if gpu:
            raw = file_bytes(args.binary, MAX_FILE)
            require(
                hashlib.sha256(raw).hexdigest() == args.binary_sha256, "run binary digest mismatch"
            )
            require(raw.startswith(b"\x7fELF"), "run input is not ELF")
            name = "hip-smoke" if args.action == "run-smoke" else "topk-bench"
            inputs[name] = write_bytes(output / "work" / name, raw, 0o700)
        receipt["inputs"] = inputs
        native = args.native.resolve(strict=True)
        native_before = native_evidence(native)
        receipt["native"] = {"path": str(native), "files": native_before}
        receipt["launcher_sha256"] = hashlib.sha256(file_bytes(Path(__file__), 1024**2)).hexdigest()
        receipt["bwrap_sha256"] = hashlib.sha256(
            file_bytes(Path("/usr/bin/bwrap"), 4 * 1024**2)
        ).hexdigest()
        namespaces = {name: os.readlink(f"/proc/self/ns/{name}") for name in NAMESPACES}
        receipt["host_namespaces"] = namespaces
        if gpu:
            lock = resources.enter_context(exclusive_lock(GPU_LOCK))
            receipt["gpu_lock"] = {
                "path": str(lock.path),
                "status": lock.status,
                "reason": lock.reason,
            }
            require(lock.acquired, "GPU lock unavailable")
            lock_held = True
        receipt["before"] = inspect_baseline(expected, output, "before")
        devices = gpu_evidence() if gpu else {}
        receipt["gpu"] = devices
        argv = bwrap_argv(
            args.action,
            output,
            native,
            f"{expected.root.identity.dev}:{expected.root.identity.ino}",
            namespaces,
            devices.get("nodes", {}),
            args.diagnostic,
        )
        receipt["argv"] = argv
        try:
            receipt["process"] = run_process(argv, output, 30 if gpu else 300)
        finally:
            receipt["after"] = inspect_baseline(expected, output, "after")
        require(native_evidence(native) == native_before, "native input drift")
        process = receipt["process"]
        require(process["failure"] is None and process["returncode"] == 0, "phase process failed")
        if args.action.startswith("compile-"):
            artifact = (
                output / "work" / ("hip-smoke" if args.action == "compile-smoke" else "topk-bench")
            )
            require(
                file_bytes(artifact, MAX_FILE).startswith(b"\x7fELF"),
                "compiler did not produce ELF",
            )
        receipt["status"] = "process-complete"
    except BaseException as exc:
        receipt["failure"] = {"type": type(exc).__name__, "message": str(exc)[:2048]}
    finally:
        try:
            if lock_held:
                receipt["gpu_lock"]["held_at_receipt"] = True
            # Hash bounded evidence on failures too; malformed outputs remain failures.
            try:
                receipt["outputs"] = [
                    {
                        "path": str(path.relative_to(output)),
                        "bytes": value.st_size,
                        "sha256": hashlib.sha256(file_bytes(path, MAX_FILE)).hexdigest(),
                    }
                    for path, value in sorted(tree_files(output))
                ]
            except (OSError, RuntimeError) as exc:
                receipt["status"] = "incomplete"
                receipt["output_evidence_failure"] = str(exc)[:2048]
            phase_record = write_json(output / "phase.json", receipt)
        finally:
            resources.close()
        if lock_held:
            write_json(
                output / "lock-release.json",
                {
                    "phase_sha256": phase_record["sha256"],
                    "path": str(GPU_LOCK),
                    "released": True,
                    "method": "exclusive_lock context exited",
                },
            )
    print(
        json.dumps(
            {"status": receipt["status"], "receipt": str(output / "phase.json")}, sort_keys=True
        )
    )
    return 0 if receipt["status"] == "process-complete" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(parse_args()))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"research launcher: {error}", file=sys.stderr)
        raise SystemExit(1) from error

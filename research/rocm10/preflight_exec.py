"""Trusted host-only checks, followed by explicit exec inside the research sandbox."""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import resource
import stat
import sys
from pathlib import Path

NAMESPACES = ("user", "mnt", "pid", "ipc", "net")
READ_ONLY = {"/usr", "/sdk", "/input", "/native", "/run/empty"}
SYSFS = {"/sys/devices", "/sys/class/kfd", "/sys/class/drm", "/sys/bus/pci", "/sys/dev/char"}
BASE_DEVICES = {"/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom", "/dev/tty"}
GPU_DEVICES = {"/dev/kfd", "/dev/dri/renderD128"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def unescape(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def mounts() -> list[dict[str, object]]:
    result = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        first, last = line.split(" - ", 1)
        fields = first.split()
        result.append(
            {
                "path": unescape(fields[4]),
                "flags": fields[5].split(","),
                "filesystem": last.split()[0],
            }
        )
    return result


def nested_userns_disabled() -> bool:
    # Test in a short-lived child so a surprising success cannot alter preflight.
    pid = os.fork()
    if pid == 0:
        try:
            os.unshare(os.CLONE_NEWUSER)
        except OSError as exc:
            os._exit(0 if exc.errno in (errno.EPERM, errno.ENOSPC, errno.EUSERS) else 2)
        except BaseException:
            os._exit(3)
        os._exit(1)
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status) == 0


def check(args: argparse.Namespace) -> dict[str, object]:
    require(os.getuid() == os.geteuid() == 1000, "unexpected uid")
    require(os.getgid() == os.getegid() == 1000, "unexpected gid")
    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    )
    for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
        require(int(status[key].strip(), 16) == 0, f"nonzero {key}")
    require(status["NoNewPrivs"].strip() == "1", "NoNewPrivs missing")
    host_ns = json.loads(args.host_namespaces)
    current_ns = {name: os.readlink(f"/proc/self/ns/{name}") for name in NAMESPACES}
    for name in NAMESPACES:
        require(name in host_ns and host_ns[name] != current_ns[name], f"shared {name} namespace")
    require(nested_userns_disabled(), "nested user namespaces not disabled")
    require(os.getcwd() == "/run/empty" and not os.listdir("."), "CWD must be empty /run/empty")
    require(
        set(os.environ) <= {"PATH", "LC_ALL", "TMPDIR", "PWD"}, "unexpected preflight environment"
    )
    require(os.environ.get("PWD") == "/run/empty", "unexpected preflight PWD")
    require(os.environ.get("PATH") == "/usr/bin:/bin", "preflight PATH is not host-only")
    for path in ("/home", "/opt", "/__w", "/run/user", "/run/dbus", "/etc/ld.so.preload"):
        require(not os.path.lexists(path), f"unexpected ambient path {path}")
    expected_identity = tuple(int(value) for value in args.sdk_identity.split(":"))
    sdk_stat = os.stat("/sdk", follow_symlinks=False)
    require((sdk_stat.st_dev, sdk_stat.st_ino) == expected_identity, "SDK root identity mismatch")
    records = mounts()
    seen = set()
    allowed = READ_ONLY | {"/", "/proc", "/dev", "/dev/pts", "/tmp", "/work"} | BASE_DEVICES
    if args.phase == "gpu":
        allowed |= SYSFS | GPU_DEVICES
    for record in records:
        path = str(record["path"])
        flags = set(record["flags"])
        require(path in allowed, f"unexpected mount or submount {path}")
        require(path not in seen, f"stacked mount {path}")
        seen.add(path)
        require("nosuid" in flags, f"mount allows suid: {path}")
        if path not in BASE_DEVICES | GPU_DEVICES | {"/dev/pts"}:
            require("nodev" in flags, f"mount allows devices: {path}")
        if path in READ_ONLY | SYSFS:
            require("ro" in flags, f"writable input mount: {path}")
    require(READ_ONLY | {"/proc", "/dev", "/tmp", "/work"} <= seen, "missing required mounts")
    devices = {}
    if args.phase == "gpu":
        require(seen >= SYSFS | GPU_DEVICES, "missing GPU mounts")
        expected_devices = json.loads(args.gpu_devices)
        require(set(expected_devices) == GPU_DEVICES, "GPU identity set mismatch")
        for path in GPU_DEVICES:
            value = os.stat(path)
            actual = [os.major(value.st_rdev), os.minor(value.st_rdev)]
            require(
                stat.S_ISCHR(value.st_mode) and actual == expected_devices[path],
                f"GPU node mismatch: {path}",
            )
            devices[path] = actual
        require(
            os.path.realpath("/sys/class/drm/renderD128/device").endswith("/0000:c2:00.0"),
            "GPU PCI mapping mismatch",
        )
        require(
            os.path.samefile("/sys/dev/char/226:128", "/sys/class/drm/renderD128"),
            "render sysfs alias mismatch",
        )
        require(
            Path("/sys/dev/char/226:128/device/drm").is_dir(), "missing libdrm classification path"
        )
        properties = Path("/sys/class/kfd/kfd/topology/nodes/1/properties").read_text()
        values = dict(line.split(maxsplit=1) for line in properties.splitlines() if " " in line)
        require(
            values.get("drm_render_minor") == "128"
            and values.get("gfx_target_version") == "110501",
            "KFD target mismatch",
        )
    else:
        require(
            not os.path.lexists("/dev/kfd")
            and not os.path.lexists("/dev/dri")
            and not os.path.lexists("/sys"),
            "GPU-free phase exposes GPU/sysfs",
        )
    require(os.readlink("/proc/self/fd/0") == "/dev/null", "stdin is not /dev/null")
    for fd in (1, 2):
        require(stat.S_ISFIFO(os.fstat(fd).st_mode), "output must use captured pipes")
    # Listing /proc/self/fd opens a transient descriptor; ignore entries gone on stat.
    for item in os.listdir("/proc/self/fd"):
        if int(item) > 2:
            try:
                os.fstat(int(item))
            except OSError as exc:
                if exc.errno == errno.EBADF:
                    continue
                raise
            raise RuntimeError(f"unexpected inherited descriptor {item}")
    return {
        "event": "preflight_pass",
        "phase": args.phase,
        "uid": os.getuid(),
        "gid": os.getgid(),
        "namespaces": current_ns,
        "mounts": records,
        "sdk_identity": list(expected_identity),
        "gpu_devices": devices,
        "metadata_coverage": "unknown",
        "vendor_authenticity": "unverified",
        "seccomp": status["Seccomp"].strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("check", "compile", "gpu"), required=True)
    parser.add_argument("--host-namespaces", required=True)
    parser.add_argument("--sdk-identity", required=True)
    parser.add_argument("--gpu-devices", default="{}")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    require(command and command[0].startswith("/"), "absolute command required")
    evidence = check(args)
    print(json.dumps(evidence, sort_keys=True), flush=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (512 * 1024**2, 512 * 1024**2))
    cpu_seconds = 20 if args.phase == "gpu" else 240
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    if args.phase != "gpu":
        resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    environment = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "TMPDIR": "/tmp"}
    if args.phase != "check":
        environment.update(
            PATH="/sdk/bin:/sdk/lib/llvm/bin:/usr/bin:/bin",
            HIP_PLATFORM="amd",
            HIP_PATH="/sdk",
            ROCM_PATH="/sdk",
            HIP_CLANG_PATH="/sdk/lib/llvm/bin",
            LD_LIBRARY_PATH="/sdk/lib:/sdk/lib/llvm/lib:/sdk/lib/rocm_sysdeps/lib",
        )
        if args.diagnostic:
            environment["LD_DEBUG"] = "libs,files"
    os.execve(command[0], command, environment)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            json.dumps({"event": "preflight_fail", "error": str(exc)}), file=sys.stderr, flush=True
        )
        sys.exit(125)

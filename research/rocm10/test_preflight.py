"""Host-only sandbox integration checks; never mount or execute the ROCm SDK."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.getuid() == 1000 and shutil.which("bwrap"), "requires local uid 1000 and Bubblewrap")
class PreflightTest(unittest.TestCase):
    def run_probe(self, defect=None):
        with tempfile.TemporaryDirectory(prefix="strix-preflight-test-") as temporary:
            root = Path(temporary)
            for name in ("sdk", "input", "native", "work", "empty"):
                (root / name).mkdir()
            shutil.copyfile(Path(__file__).with_name("preflight_exec.py"), root / "input/preflight_exec.py")
            host_ns = {name: os.readlink(f"/proc/self/ns/{name}") for name in ("user", "mnt", "pid", "ipc", "net")}
            sdk_stat = (root / "sdk").stat()
            identity = f"{sdk_stat.st_dev}:{sdk_stat.st_ino}"
            argv = ["/usr/bin/bwrap", "--unshare-all", "--unshare-user", "--disable-userns",
                    "--assert-userns-disabled", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
                    "--clearenv", "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
                    "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib", "/lib64"]
            for name, target in (("sdk", "/sdk"), ("input", "/input"), ("native", "/native"),
                                 ("work", "/work"), ("empty", "/run/empty")):
                writable = name == "work" or (name == "sdk" and defect == "writable_sdk")
                argv += ["--bind" if writable else "--ro-bind", str(root / name), target]
            argv += ["--proc", "/proc", "--dev", "/dev", "--size", "2147483648", "--tmpfs", "/tmp",
                     "--chdir", "/run/empty", "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "LC_ALL", "C",
                     "--setenv", "TMPDIR", "/tmp"]
            if defect == "extra_mount":
                argv += ["--ro-bind", str(root / "native"), "/unexpected"]
            if defect == "extra_environment":
                argv += ["--setenv", "STRIX_UNEXPECTED", "1"]
            if defect == "wrong_identity":
                identity = "0:0"
            argv += ["/usr/bin/python", "-I", "/input/preflight_exec.py", "--phase", "check",
                     "--host-namespaces", json.dumps(host_ns), "--sdk-identity", identity, "--",
                     "/usr/bin/python", "-I", "-c", "print('HOST_ONLY_EXECUTED')"]
            return subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, timeout=15, close_fds=True)

    def test_host_only_exec_after_checks(self):
        result = self.run_probe()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"event": "preflight_pass"', result.stdout)
        self.assertIn("HOST_ONLY_EXECUTED", result.stdout)

    def test_failed_gates_prevent_exec(self):
        for defect in ("writable_sdk", "extra_mount", "extra_environment", "wrong_identity"):
            with self.subTest(defect=defect):
                result = self.run_probe(defect)
                self.assertEqual(result.returncode, 125, result.stderr)
                self.assertIn('"event": "preflight_fail"', result.stderr)
                self.assertNotIn("HOST_ONLY_EXECUTED", result.stdout)


if __name__ == "__main__":
    unittest.main()

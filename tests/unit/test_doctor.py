from __future__ import annotations

import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from strixlab import cli
from strixlab import doctor as doctor_module
from strixlab.doctor import (
    CheckV1,
    DoctorReportV1,
    DoctorRun,
    RedactionContext,
    ReportWriteError,
    SensitiveInterpolationError,
    UnsafeDiagnosticError,
    _write_temp,
    build_report,
    canonical_report_bytes,
    evaluate_machine,
    publish_authoritative,
    publish_diagnostic,
    reject_sensitive_interpolations,
    run_doctor,
)
from strixlab.locks import LockAttempt, LockStatus, exclusive_lock
from strixlab.machine import (
    GPUIdentity,
    HostFacts,
    MachineProbe,
    MachineSnapshot,
    ProbeIssue,
    TelemetrySample,
    ToolFact,
    _bdf_from_id,
    _clock_hz,
    _failed_sample,
    _finite,
    _gfx_arch,
    _json_object,
    _nonnegative,
    _number,
    _parse_amd_smi_sample,
    _parse_rocm_smi_sample,
    _parse_rocminfo,
    _percentage,
    _read_optional_number,
)
from strixlab.manifests import MachineProfileV1
from strixlab.process import ProcessOutcome, ProcessResult
from strixlab.secret_policy import is_sensitive_name

runner = CliRunner()
NOW = datetime(2026, 8, 28, 20, 0, tzinfo=UTC)


def profile(*, mode: str = "auto", lock: str = "/tmp/strixlab-test.lock") -> MachineProfileV1:
    return MachineProfileV1.model_validate(
        {
            "schema_version": 1,
            "id": "test-machine",
            "expect": {
                "gpu_arch": "gfx1151",
                "integrated_gpu": True,
                "memory_gib_min": 120,
            },
            "exclusive_lock": {"path": lock},
            "telemetry": {"amd_smi": mode, "sample_interval_ms": 1},
            "validity": {
                "require_ac_power": True,
                "max_background_gpu_busy_pct": 10,
                "min_available_memory_gib": 8,
                "temperature_warn_c": 85,
            },
        }
    )


def gpu(*, integrated: bool | None = True) -> GPUIdentity:
    return GPUIdentity(
        node="1",
        arch="gfx1151",
        integrated=integrated,
        render_node="renderD128",
        pci_bdf="0000:c2:00.0",
        vendor_id=0x1002,
        device_id=0x1586,
        marketing_name="AMD Radeon 8060S Graphics",
    )


def host() -> HostFacts:
    return HostFacts(
        platform="linux",
        kernel="6.14",
        distro="Test Linux",
        cpu_model="Test CPU",
        cpu_count=32,
        memory_total_bytes=120 * 2**30,
        memory_available_bytes=8 * 2**30,
        swap_total_bytes=4 * 2**30,
        swap_free_bytes=3 * 2**30,
        ac_power=True,
    )


def tools() -> tuple[ToolFact, ...]:
    required = tuple(
        ToolFact(
            name=name,
            path=f"/tools/{name}",
            version="version 1",
            outcome="ok" if name != "rocminfo" else None,
        )
        for name in ("rocminfo", "hipcc", "cmake", "ninja", "git", "clang")
    )
    optional = tuple(
        ToolFact(name=name, path=f"/tools/{name}")
        for name in ("rocprof", "rocprofv3", "rocprof-compute", "rocprof-sys")
    )
    return (
        *required,
        *optional,
        ToolFact(name="amd-smi", path=None),
        ToolFact(name="rocm-smi", path="/tools/rocm-smi"),
    )


def samples(*, source: str = "rocm-smi", temperature: float = 50) -> tuple[TelemetrySample, ...]:
    return tuple(
        TelemetrySample(
            index, source, float(index + 1), temperature, 20.0, 1_000_000_000, 800_000_000
        )
        for index in range(3)
    )


def snapshot(*, source: str = "rocm-smi") -> MachineSnapshot:
    identity = gpu()
    return MachineSnapshot(
        host=host(),
        tools=tools(),
        gpu=identity,
        gpu_candidates=(identity,),
        rocminfo_arches=("gfx1151",),
        telemetry_source=source,
        samples=samples(source=source),
    )


def acquired(path: str = "/tmp/strixlab-test.lock") -> LockAttempt:
    return LockAttempt(LockStatus.ACQUIRED, Path(path))


def statuses(checks: tuple[Any, ...]) -> dict[str, str]:
    return {check.id: check.status for check in checks}


def test_ready_policy_and_equality_boundaries() -> None:
    checks = evaluate_machine(profile(), snapshot(), acquired())
    result = statuses(checks)

    assert "blocker" not in result.values()
    assert result["memory-total"] == "pass"
    assert result["memory-available"] == "pass"
    assert result["telemetry-source"] == "warning"


def test_policy_collects_independent_failures_and_warnings() -> None:
    value = snapshot()
    value = replace(
        value,
        host=replace(value.host, memory_available_bytes=1, ac_power=None),
        samples=(
            TelemetrySample(0, "rocm-smi", 11.0, 85.0, None, None, None),
            TelemetrySample(1, "rocm-smi", None, None, None, None, None, "bad"),
        ),
    )
    result = statuses(evaluate_machine(profile(), value, acquired()))

    assert result["memory-available"] == "blocker"
    assert result["ac-power"] == "blocker"
    assert result["gpu-busy"] == "blocker"
    assert result["gpu-temperature"] == "warning"


def test_unknown_integrated_classification_blocks() -> None:
    unknown = gpu(integrated=None)
    value = replace(snapshot(), gpu=None, gpu_candidates=(unknown,), samples=())
    result = statuses(evaluate_machine(profile(), value, acquired()))

    assert result["gpu-identity"] == "blocker"
    assert result["gpu-integrated"] == "blocker"


def test_non_owner_skips_gpu_checks() -> None:
    lock = LockAttempt(LockStatus.CONTENDED, Path("/tmp/lock"), "held")
    result = statuses(evaluate_machine(profile(), snapshot(), lock))

    assert result["exclusive-lock"] == "blocker"
    assert result["gpu-identity"] == "skipped"
    assert result["gpu-temperature"] == "skipped"


def test_required_and_disabled_telemetry_modes() -> None:
    required = statuses(evaluate_machine(profile(mode="required"), snapshot(), acquired()))
    incomplete_amd = replace(
        snapshot(source="amd-smi"),
        samples=(
            TelemetrySample(0, "amd-smi", 1.0, 40.0, None, None, None),
            TelemetrySample(1, "amd-smi", None, 40.0, None, None, None, "bad"),
            TelemetrySample(2, "amd-smi", 1.0, 40.0, None, None, None),
        ),
    )
    incomplete_required = statuses(
        evaluate_machine(profile(mode="required"), incomplete_amd, acquired())
    )
    disabled = statuses(
        evaluate_machine(profile(mode="disabled"), snapshot(source="sysfs"), acquired())
    )

    assert required["telemetry-source"] == "blocker"
    assert incomplete_required["telemetry-source"] == "blocker"
    assert incomplete_required["gpu-busy"] == "blocker"
    assert disabled["telemetry-source"] == "pass"


def test_report_round_trip_and_secret_redaction() -> None:
    base = snapshot()
    assert base.gpu is not None
    value = replace(
        base,
        tools=(
            *base.tools,
            ToolFact("extra", "/tool", "prefix top-secret suffix", "ok"),
        ),
        gpu=replace(base.gpu, marketing_name="top-secret gpu"),
        issues=(ProbeIssue("probe-error", "top-secret leaked"),),
    )
    report = build_report(
        profile(),
        value,
        acquired(),
        {"API_TOKEN": "top-secret", "PATH": "/tools"},
        NOW,
        NOW,
    )
    payload = canonical_report_bytes(report)

    assert b"top-secret" not in payload
    assert b"[REDACTED]" in payload
    assert DoctorReportV1.model_validate_json(payload) == report


def test_final_secret_scan_fails_closed() -> None:
    context = RedactionContext.from_environ({"API_TOKEN": "top-secret"})

    assert "top-secret" not in repr(context)
    with pytest.raises(UnsafeDiagnosticError):
        context.assert_payload_safe(b'{"path":"top-secret"}')


def test_sensitive_interpolation_is_rejected() -> None:
    with pytest.raises(SensitiveInterpolationError):
        reject_sensitive_interpolations({"path": "${API_TOKEN}"})
    reject_sensitive_interpolations({"path": "$${API_TOKEN}", "safe": "${ROCM_PATH}"})


def test_atomic_report_publication(tmp_path: Path) -> None:
    authoritative = tmp_path / "nested" / "private" / "doctor.json"
    publish_authoritative(authoritative, b"one\n")
    publish_authoritative(authoritative, b"two\n")
    first = publish_diagnostic(authoritative, b"diagnostic-one\n")
    second = publish_diagnostic(authoritative, b"diagnostic-two\n")

    assert authoritative.read_bytes() == b"two\n"
    assert first != second
    assert first.read_bytes() == b"diagnostic-one\n"
    assert second.read_bytes() == b"diagnostic-two\n"
    assert oct(authoritative.stat().st_mode & 0o777) == "0o600"
    assert authoritative.parent.stat().st_mode & 0o777 == 0o700
    assert authoritative.parent.parent.stat().st_mode & 0o777 == 0o700


def test_temp_and_publication_failures_clean_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "doctor.json"

    def fail_fsync(_: int) -> None:
        raise OSError("fsync boom")

    monkeypatch.setattr(doctor_module.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="fsync boom"):
        _write_temp(destination, b"payload")
    assert list(tmp_path.iterdir()) == []

    def fail_write(_: Path, __: bytes) -> Path:
        raise OSError("write boom")

    monkeypatch.setattr(doctor_module, "_write_temp", fail_write)
    with pytest.raises(ReportWriteError) as error:
        publish_authoritative(destination, b"payload")
    assert isinstance(error.value.__cause__, OSError)
    assert "write boom" in str(error.value.__cause__)
    assert list(tmp_path.iterdir()) == []


def test_exclusive_lock_acquisition_contention_and_safety(tmp_path: Path) -> None:
    path = tmp_path / "gpu.lock"
    with exclusive_lock(path) as first:
        assert first.status is LockStatus.ACQUIRED
        with exclusive_lock(path) as second:
            assert second.status is LockStatus.CONTENDED
    assert path.exists()
    assert path.stat().st_mode & 0o777 == 0o600

    broad = tmp_path / "broad.lock"
    broad.write_text("", encoding="utf-8")
    broad.chmod(0o644)
    with exclusive_lock(broad) as attempt:
        assert attempt.status is LockStatus.UNAVAILABLE

    target = tmp_path / "target"
    target.write_text("", encoding="utf-8")
    symlink = tmp_path / "link.lock"
    symlink.symlink_to(target)
    with exclusive_lock(symlink) as attempt:
        assert attempt.status is LockStatus.UNAVAILABLE


def test_lock_missing_parent_and_unsupported_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with exclusive_lock(tmp_path / "missing" / "gpu.lock") as attempt:
        assert attempt.status is LockStatus.UNAVAILABLE

    monkeypatch.setattr(sys, "platform", "darwin")
    with exclusive_lock(tmp_path / "darwin.lock") as attempt:
        assert attempt.status is LockStatus.UNAVAILABLE


def test_machine_parser_units_and_identity() -> None:
    assert _gfx_arch(110501) == "gfx1151"
    assert _bdf_from_id(49664) == "0000:c2:00.0"
    assert _percentage(0) == 0
    assert _percentage(100) == 100
    assert _percentage(101) is None
    assert _clock_hz("(1.5Ghz)") == 1_500_000_000
    records = _parse_rocminfo(
        """
  Name: gfx1151
  Marketing Name: AMD Radeon 8060S Graphics
  BDFID: 49664
"""
    )
    assert records == [
        {
            "arch": "gfx1151",
            "marketing": "AMD Radeon 8060S Graphics",
            "bdf": "0000:c2:00.0",
        }
    ]


def test_rocm_smi_parser_correlates_bdf() -> None:
    payload = json.dumps(
        {
            "card0": {
                "PCI Bus": "0000:C2:00.0",
                "GPU use (%)": "4",
                "Temperature (Sensor edge) (C)": "44.5",
                "Current Socket Graphics Package Power (W)": "32.7",
                "sclk clock speed:": "(1033Mhz)",
                "mclk clock speed:": "(1000Mhz)",
            }
        }
    )
    sample = _parse_rocm_smi_sample("warning\n" + payload, 0, gpu())

    assert sample.busy_pct == 4
    assert sample.temperature_c == 44.5
    assert sample.sclk_hz == 1_033_000_000


def write_fake_host(root: Path) -> tuple[Path, Path, Path]:
    proc = root / "proc"
    sysroot = root / "sys"
    etc = root / "etc"
    proc.mkdir()
    etc.mkdir()
    (proc / "meminfo").write_text(
        "MemTotal: 130000000 kB\nMemAvailable: 9000000 kB\nSwapTotal: 1 kB\nSwapFree: 0 kB\n",
        encoding="utf-8",
    )
    (proc / "cpuinfo").write_text("model name : Test CPU\n", encoding="utf-8")
    (etc / "os-release").write_text('PRETTY_NAME="Fixture Linux"\n', encoding="utf-8")
    ac = sysroot / "class" / "power_supply" / "AC"
    battery = sysroot / "class" / "power_supply" / "BAT0"
    ac.mkdir(parents=True)
    battery.mkdir()
    (ac / "type").write_text("Mains\n", encoding="utf-8")
    (ac / "online").write_text("1\n", encoding="utf-8")
    (battery / "type").write_text("Battery\n", encoding="utf-8")
    return proc, sysroot, etc


def process_result(argv: tuple[str, ...], stdout: str) -> ProcessResult:
    return ProcessResult(ProcessOutcome.EXITED, argv, 0, stdout, "", 0, 1, 1, None)


def test_production_probe_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proc, sysroot, etc = write_fake_host(tmp_path)
    properties = sysroot / "class" / "kfd" / "kfd" / "topology" / "nodes" / "1"
    properties.mkdir(parents=True)
    (properties / "properties").write_text(
        "gfx_target_version 110501\ndrm_render_minor 128\n", encoding="utf-8"
    )
    device = sysroot / "devices" / "pci0000:00" / "0000:c2:00.0"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n", encoding="utf-8")
    (device / "device").write_text("0x1586\n", encoding="utf-8")
    (device / "gpu_busy_percent").write_text("2\n", encoding="utf-8")
    hwmon = device / "hwmon" / "hwmon0"
    hwmon.mkdir(parents=True)
    (hwmon / "temp1_input").write_text("45000\n", encoding="utf-8")
    (hwmon / "power1_average").write_text("30000000\n", encoding="utf-8")
    drm = sysroot / "class" / "drm" / "renderD128"
    drm.mkdir(parents=True)
    (drm / "device").symlink_to(device)

    rocm_payload = json.dumps(
        {
            "card0": {
                "PCI Bus": "0000:C2:00.0",
                "GPU use (%)": "2",
                "Temperature (Sensor edge) (C)": "45",
                "Current Socket Graphics Package Power (W)": "30",
            }
        }
    )

    def fake_runner(argv: tuple[str, ...], **_: Any) -> ProcessResult:
        name = Path(argv[0]).name
        if name == "rocminfo":
            output = "Name: gfx1151\nMarketing Name: Fixture GPU\nBDFID: 49664\n"
        elif name == "rocm-smi":
            output = rocm_payload
        else:
            output = f"{name} version 1\n"
        return process_result(argv, output)

    probe = MachineProbe(
        environ={"PATH": "/tools"},
        proc_root=proc,
        sys_root=sysroot,
        etc_root=etc,
        cwd=tmp_path,
        runner=fake_runner,
        sleeper=lambda _: None,
    )
    base_tools = tuple(
        ToolFact(name, f"/tools/{name}")
        for name in (
            "hipcc",
            "cmake",
            "ninja",
            "git",
            "clang",
            "rocminfo",
            "rocm-smi",
            "amd-smi",
            "rocprof",
            "rocprofv3",
            "rocprof-compute",
            "rocprof-sys",
        )
        if name != "amd-smi"
    ) + (ToolFact("amd-smi", None),)
    pre = replace(probe.collect_prelock(), tools=base_tools)
    value = probe.collect_postlock(pre, profile())

    assert pre.host.ac_power is True
    assert pre.host.distro == "Fixture Linux"
    assert value.gpu is not None
    assert value.gpu.pci_bdf == "0000:c2:00.0"
    assert value.gpu.marketing_name == "Fixture GPU"
    assert value.telemetry_source == "rocm-smi"
    assert len(value.samples) == 3


class FakeProbe:
    def __init__(self, value: MachineSnapshot) -> None:
        self.value = value
        self.postlock_calls = 0

    def collect_prelock(self) -> MachineSnapshot:
        return replace(self.value, gpu=None, gpu_candidates=(), samples=(), rocminfo_arches=())

    def collect_postlock(self, _: MachineSnapshot, __: MachineProfileV1) -> MachineSnapshot:
        self.postlock_calls += 1
        return self.value

    def mark_gpu_skipped(self, value: MachineSnapshot, reason: str) -> MachineSnapshot:
        return replace(value, issues=(ProbeIssue("gpu-probes-skipped", reason),))


def write_profile(path: Path, lock_path: Path) -> None:
    path.write_text(
        f"""schema_version: 1
id: test-machine
expect:
  gpu_arch: gfx1151
  integrated_gpu: true
  memory_gib_min: 120
exclusive_lock:
  path: {lock_path}
telemetry:
  amd_smi: auto
  sample_interval_ms: 1
validity:
  require_ac_power: true
  max_background_gpu_busy_pct: 10
  min_available_memory_gib: 8
  temperature_warn_c: 85
""",
        encoding="utf-8",
    )


def test_run_doctor_authoritative_and_diagnostic(tmp_path: Path) -> None:
    manifest = tmp_path / "machine.yaml"
    lock_path = tmp_path / "gpu.lock"
    write_profile(manifest, lock_path)
    destination = tmp_path / "doctor.json"
    value = snapshot()

    ready = run_doctor(
        manifest,
        output=destination,
        environ={"PATH": "/tools"},
        probe=FakeProbe(value),  # type: ignore[arg-type]
        now=lambda: NOW,
    )
    assert ready.path == destination
    assert ready.ready is True

    @contextmanager
    def contended(_: Path) -> Any:
        yield LockAttempt(LockStatus.CONTENDED, lock_path, "held")

    blocked = run_doctor(
        manifest,
        output=destination,
        environ={"PATH": "/tools"},
        probe=FakeProbe(value),  # type: ignore[arg-type]
        lock_factory=contended,
        now=lambda: NOW,
    )
    assert blocked.ready is False
    assert blocked.path != destination
    assert destination.read_bytes() == canonical_report_bytes(ready.report)


def test_sensitive_output_path_fails_before_publication(tmp_path: Path) -> None:
    manifest = tmp_path / "machine.yaml"
    lock_path = tmp_path / "gpu.lock"
    write_profile(manifest, lock_path)
    destination = tmp_path / "top-secret" / "doctor.json"

    with pytest.raises(UnsafeDiagnosticError):
        run_doctor(
            manifest,
            output=destination,
            environ={"PATH": "/tools", "API_TOKEN": "top-secret"},
            probe=FakeProbe(snapshot()),  # type: ignore[arg-type]
            now=lambda: NOW,
        )
    assert destination.parent.exists() is False


def test_sensitive_diagnostic_suffix_fails_before_publication(tmp_path: Path) -> None:
    manifest = tmp_path / "machine.yaml"
    lock_path = tmp_path / "gpu.lock"
    write_profile(manifest, lock_path)
    destination = tmp_path / "safe" / "doctor.json"

    @contextmanager
    def contended(_: Path) -> Any:
        yield LockAttempt(LockStatus.CONTENDED, lock_path, "held")

    with pytest.raises(UnsafeDiagnosticError):
        run_doctor(
            manifest,
            output=destination,
            environ={"PATH": "/tools", "API_TOKEN": "diagnostic"},
            probe=FakeProbe(snapshot()),  # type: ignore[arg-type]
            lock_factory=contended,
            now=lambda: NOW,
        )
    assert list(destination.parent.glob("*.json")) == []


def test_cli_ready_and_blocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The CLI builds its RedactionContext from the real process environment. An
    # ambient sensitive-named var with a short value (e.g. CLAUDE_CODE_CHILD_SESSION=1
    # under Claude Code) would make redaction refuse the innocuous random tmp path.
    # Isolate this test from ambient sensitive-named vars; production redaction is
    # unchanged.
    for name in list(os.environ):
        if is_sensitive_name(name):
            monkeypatch.delenv(name, raising=False)
    report = build_report(profile(), snapshot(), acquired(), {"PATH": "/tools"}, NOW, NOW)
    path = tmp_path / "doctor.json"
    ready = DoctorRun(report=report, path=path)
    monkeypatch.setattr(cli, "run_doctor", lambda *_args, **_kwargs: ready)
    manifest = tmp_path / "machine.yaml"
    manifest.write_text("placeholder", encoding="utf-8")

    result = runner.invoke(cli.app, ["doctor", "--machine", str(manifest)])
    assert result.exit_code == 0
    assert "ready:" in result.stdout

    blocked_report = report.model_copy(
        update={
            "status": "blocked",
            "checks": (
                *report.checks,
                CheckV1(id="forced-blocker", status="blocker", message="blocked"),
            ),
        }
    )
    blocked_report = DoctorReportV1.model_validate(blocked_report.model_dump())
    monkeypatch.setattr(
        cli,
        "run_doctor",
        lambda *_args, **_kwargs: DoctorRun(report=blocked_report, path=path),
    )
    result = runner.invoke(cli.app, ["doctor", "--machine", str(manifest)])
    assert result.exit_code == 1
    assert "forced-blocker" in result.output


def test_cli_refuses_sensitive_report_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    report = build_report(profile(), snapshot(), acquired(), {"PATH": "/tools"}, NOW, NOW)
    sensitive = "terminal-path-secret"
    result_path = tmp_path / sensitive / "doctor.json"
    monkeypatch.setenv("API_TOKEN", sensitive)
    monkeypatch.setattr(
        cli,
        "run_doctor",
        lambda *_args, **_kwargs: DoctorRun(report=report, path=result_path),
    )
    manifest = tmp_path / "machine.yaml"
    manifest.write_text("placeholder", encoding="utf-8")

    result = runner.invoke(cli.app, ["doctor", "--machine", str(manifest)])
    assert result.exit_code == 1
    assert sensitive not in result.output
    assert "unable to safely render terminal output" in result.output


def test_amd_smi_parser_and_recursive_records() -> None:
    payload = json.dumps(
        {
            "metrics": [
                {
                    "gpu": {
                        "bdf": "0000:c2:00.0",
                        "gpu_busy_percent": 7,
                        "temperature_c": 46.5,
                        "power_w": 35,
                        "gfx_clock": "1.2Ghz",
                        "memory_clock": "900Mhz",
                    }
                }
            ]
        }
    )
    sample = _parse_amd_smi_sample(payload, 2, gpu())

    assert sample.index == 2
    assert sample.busy_pct == 7
    assert sample.temperature_c == 46.5
    assert sample.power_w == 35
    assert sample.sclk_hz == 1_200_000_000
    assert sample.mclk_hz == 900_000_000

    with pytest.raises(ValueError, match="identity"):
        _parse_amd_smi_sample('{"bdf":"0000:00:00.0"}', 0, gpu())


def test_numeric_and_json_parser_edges(tmp_path: Path) -> None:
    assert _number(None) is None
    assert _number(True) is None
    assert _number(3) == 3
    assert _number("1,234.5 watts") == 1234.5
    assert _number("none") is None
    assert _finite(math.nan) is None
    assert _nonnegative(-1) is None
    assert _clock_hz("2khz") == 2000
    assert _clock_hz("3Mhz") == 3_000_000
    assert _clock_hz(4) == 4
    assert _clock_hz(-1) is None
    assert _read_optional_number(tmp_path / "missing") is None
    assert _failed_sample(0, "none", "failed").error == "failed"
    with pytest.raises(json.JSONDecodeError):
        _json_object("not json")


def test_probe_unavailable_host_and_gpu_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = MachineProbe(
        environ={"PATH": ""},
        proc_root=tmp_path / "missing-proc",
        sys_root=tmp_path / "missing-sys",
        etc_root=tmp_path / "missing-etc",
        cwd=tmp_path,
        sleeper=lambda _: None,
    )
    value = probe.collect_prelock()

    assert value.host.memory_total_bytes is None
    assert value.host.cpu_model is None
    assert value.host.ac_power is None
    assert {issue.code for issue in value.issues} == {
        "meminfo-unavailable",
        "cpuinfo-unavailable",
    }
    skipped = probe.mark_gpu_skipped(value, "held")
    assert any(issue.code == "gpu-probes-skipped" for issue in skipped.issues)

    monkeypatch.setattr(sys, "platform", "darwin")
    issues: list[ProbeIssue] = []
    assert probe._gpu_candidates(issues) == ()
    assert issues[0].code == "unsupported-platform"


def test_probe_tool_and_sysfs_failure_paths(tmp_path: Path) -> None:
    def failed_runner(argv: tuple[str, ...], **_: Any) -> ProcessResult:
        return ProcessResult(
            ProcessOutcome.TIMED_OUT,
            argv,
            -15,
            "partial",
            "failure",
            0,
            1,
            1,
            None,
            True,
            False,
        )

    probe = MachineProbe(
        environ={"PATH": ""},
        proc_root=tmp_path,
        sys_root=tmp_path,
        etc_root=tmp_path,
        cwd=tmp_path,
        runner=failed_runner,
        sleeper=lambda _: None,
    )
    issues: list[ProbeIssue] = []
    versioned = probe._version_tools(
        (
            ToolFact("hipcc", "/tools/hipcc"),
            ToolFact("cmake", None),
            ToolFact("rocminfo", "/tools/rocminfo"),
        ),
        issues,
    )
    assert versioned[0].outcome == "timed_out"
    assert versioned[0].truncated is True
    assert issues[0].code == "hipcc-version-failed"

    rocminfo_issues: list[ProbeIssue] = []
    arches, marketing = probe._rocminfo(gpu(), versioned, rocminfo_issues)
    assert arches == ()
    assert marketing is None
    assert rocminfo_issues[0].code == "rocminfo-failed"

    sample = probe._read_sample("amd-smi", 0, gpu(), (ToolFact("amd-smi", None),))
    assert sample.error == "amd-smi is unavailable"
    sysfs_sample = probe._sysfs_sample(0, gpu())
    assert sysfs_sample.busy_pct is None
    assert sysfs_sample.error is not None

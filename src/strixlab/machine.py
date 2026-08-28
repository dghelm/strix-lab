"""Bounded, read-only host and AMD GPU observation."""

from __future__ import annotations

import json
import math
import os
import platform
import re
import shutil
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from strixlab.manifests import MachineProfileV1
from strixlab.process import ProcessOutcome, ProcessResult, run_process

PROBE_TIMEOUT_SECONDS = 5.0
PROBE_OUTPUT_LIMIT_BYTES = 64 * 1024
ROCM_SMI_ARGS = (
    "--showuse",
    "--showtemp",
    "--showpower",
    "--showclocks",
    "--showbus",
    "--json",
)
AMD_SMI_ARGS = ("metric", "--gpu", "all", "--json")
VERSION_ARGUMENTS: dict[str, tuple[str, ...]] = {
    "hipcc": ("--version",),
    "cmake": ("--version",),
    "ninja": ("--version",),
    "git": ("--version",),
    "clang": ("--version",),
}
PROFILER_TOOLS = ("rocprof", "rocprofv3", "rocprof-compute", "rocprof-sys")
KNOWN_INTEGRATED_AMD_DEVICES = frozenset({0x1586})
KNOWN_DISCRETE_AMD_DEVICES = frozenset(
    {
        0x73BF,  # Navi 21
        0x744C,  # Navi 31
        0x747E,  # Navi 32
        0x7480,  # Navi 33
    }
)


@dataclass(frozen=True, slots=True)
class ProbeIssue:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ToolFact:
    name: str
    path: str | None
    version: str | None = None
    outcome: str | None = None
    truncated: bool = False

    @property
    def available(self) -> bool:
        return self.path is not None


@dataclass(frozen=True, slots=True)
class HostFacts:
    platform: str
    kernel: str
    distro: str | None
    cpu_model: str | None
    cpu_count: int | None
    memory_total_bytes: int | None
    memory_available_bytes: int | None
    swap_total_bytes: int | None
    swap_free_bytes: int | None
    ac_power: bool | None


@dataclass(frozen=True, slots=True)
class GPUIdentity:
    node: str
    arch: str
    integrated: bool | None
    render_node: str
    pci_bdf: str
    vendor_id: int
    device_id: int
    marketing_name: str | None = None


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    index: int
    source: str
    busy_pct: float | None
    temperature_c: float | None
    power_w: float | None
    sclk_hz: int | None
    mclk_hz: int | None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class MachineSnapshot:
    host: HostFacts
    tools: tuple[ToolFact, ...]
    gpu: GPUIdentity | None = None
    gpu_candidates: tuple[GPUIdentity, ...] = ()
    rocminfo_arches: tuple[str, ...] = ()
    telemetry_source: str | None = None
    samples: tuple[TelemetrySample, ...] = ()
    issues: tuple[ProbeIssue, ...] = ()


@dataclass(slots=True)
class MachineProbe:
    """Production probe with injectable roots, environment, runner, and clock."""

    environ: Mapping[str, str]
    proc_root: Path = Path("/proc")
    sys_root: Path = Path("/sys")
    etc_root: Path = Path("/etc")
    cwd: Path = field(default_factory=Path.cwd)
    runner: Callable[..., ProcessResult] = run_process
    sleeper: Callable[[float], None] = time.sleep

    def collect_prelock(self) -> MachineSnapshot:
        issues: list[ProbeIssue] = []
        host = self._host_facts(issues)
        tools = tuple(self._discover_tools())
        return MachineSnapshot(host=host, tools=tools, issues=tuple(issues))

    def mark_gpu_skipped(self, snapshot: MachineSnapshot, reason: str) -> MachineSnapshot:
        return replace(
            snapshot,
            issues=(*snapshot.issues, ProbeIssue("gpu-probes-skipped", reason)),
        )

    def collect_postlock(
        self, snapshot: MachineSnapshot, profile: MachineProfileV1
    ) -> MachineSnapshot:
        issues = list(snapshot.issues)
        tools = tuple(self._version_tools(snapshot.tools, issues))
        candidates = self._gpu_candidates(issues)
        matching = tuple(
            candidate
            for candidate in candidates
            if candidate.arch == profile.expect.gpu_arch
            and candidate.integrated is profile.expect.integrated_gpu
        )
        selected = matching[0] if len(matching) == 1 else None
        rocminfo_arches: tuple[str, ...] = ()
        if selected is not None:
            rocminfo_arches, marketing = self._rocminfo(selected, tools, issues)
            if marketing is not None:
                selected = replace(selected, marketing_name=marketing)
        samples: tuple[TelemetrySample, ...] = ()
        telemetry_source: str | None = None
        if selected is not None:
            telemetry_source, samples = self._sample(profile, selected, tools, issues)
        return MachineSnapshot(
            host=snapshot.host,
            tools=tools,
            gpu=selected,
            gpu_candidates=candidates,
            rocminfo_arches=rocminfo_arches,
            telemetry_source=telemetry_source,
            samples=samples,
            issues=tuple(issues),
        )

    def _read_text(self, path: Path, *, limit: int = 128 * 1024) -> str:
        with path.open("rb") as stream:
            return stream.read(limit + 1)[:limit].decode("utf-8", errors="replace")

    def _host_facts(self, issues: list[ProbeIssue]) -> HostFacts:
        meminfo: dict[str, int] = {}
        try:
            for line in self._read_text(self.proc_root / "meminfo").splitlines():
                match = re.fullmatch(r"([A-Za-z_()]+):\s+(\d+)\s+kB", line)
                if match:
                    meminfo[match.group(1)] = int(match.group(2)) * 1024
        except OSError:
            issues.append(ProbeIssue("meminfo-unavailable", "unable to read memory facts"))

        cpu_model: str | None = None
        try:
            for line in self._read_text(self.proc_root / "cpuinfo").splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    cpu_model = line.split(":", 1)[1].strip() or None
                    break
        except OSError:
            issues.append(ProbeIssue("cpuinfo-unavailable", "unable to read CPU model"))

        distro: str | None = None
        try:
            values: dict[str, str] = {}
            for line in self._read_text(self.etc_root / "os-release", limit=16 * 1024).splitlines():
                if "=" in line:
                    name, value = line.split("=", 1)
                    values[name] = value.strip().strip('"')
            distro = values.get("PRETTY_NAME") or values.get("NAME")
        except OSError:
            pass

        return HostFacts(
            platform=sys.platform,
            kernel=platform.release(),
            distro=distro,
            cpu_model=cpu_model,
            cpu_count=os.cpu_count(),
            memory_total_bytes=meminfo.get("MemTotal"),
            memory_available_bytes=meminfo.get("MemAvailable"),
            swap_total_bytes=meminfo.get("SwapTotal"),
            swap_free_bytes=meminfo.get("SwapFree"),
            ac_power=self._ac_power(),
        )

    def _ac_power(self) -> bool | None:
        root = self.sys_root / "class" / "power_supply"
        readable: list[bool] = []
        try:
            supplies = sorted(root.iterdir())
        except OSError:
            return None
        for supply in supplies:
            try:
                supply_type = self._read_text(supply / "type", limit=256).strip()
                if not (supply_type in {"Mains", "Wireless"} or supply_type.startswith("USB")):
                    continue
                online = self._read_text(supply / "online", limit=32).strip()
                if online in {"0", "1"}:
                    readable.append(online == "1")
            except OSError:
                continue
        if not readable:
            return None
        return any(readable)

    def _discover_tools(self) -> list[ToolFact]:
        path = self.environ.get("PATH", "")
        names = (*VERSION_ARGUMENTS, "rocminfo", "amd-smi", "rocm-smi", *PROFILER_TOOLS)
        return [ToolFact(name=name, path=shutil.which(name, path=path)) for name in names]

    def _run(self, path: str, args: tuple[str, ...]) -> ProcessResult:
        return self.runner(
            (path, *args),
            cwd=self.cwd,
            timeout=PROBE_TIMEOUT_SECONDS,
            inherit_env=True,
            base_env=self.environ,
            output_limit_bytes=PROBE_OUTPUT_LIMIT_BYTES,
        )

    def _version_tools(
        self, tools: tuple[ToolFact, ...], issues: list[ProbeIssue]
    ) -> list[ToolFact]:
        result: list[ToolFact] = []
        for tool in tools:
            args = VERSION_ARGUMENTS.get(tool.name)
            if tool.path is None or args is None:
                result.append(tool)
                continue
            process = self._run(tool.path, args)
            summary = _bounded_summary(process.stdout or process.stderr)
            outcome = _process_outcome(process)
            result.append(
                ToolFact(
                    name=tool.name,
                    path=tool.path,
                    version=summary,
                    outcome=outcome,
                    truncated=process.stdout_truncated or process.stderr_truncated,
                )
            )
            if process.outcome is not ProcessOutcome.EXITED or process.returncode != 0:
                issues.append(
                    ProbeIssue(f"{tool.name}-version-failed", f"{tool.name} version probe failed")
                )
        return result

    def _gpu_candidates(self, issues: list[ProbeIssue]) -> tuple[GPUIdentity, ...]:
        if sys.platform != "linux":
            issues.append(ProbeIssue("unsupported-platform", "Linux is required"))
            return ()
        root = self.sys_root / "class" / "kfd" / "kfd" / "topology" / "nodes"
        candidates: list[GPUIdentity] = []
        try:
            nodes = sorted(root.iterdir(), key=lambda value: value.name)
        except OSError:
            issues.append(ProbeIssue("kfd-unavailable", "KFD topology is unavailable"))
            return ()
        for node in nodes:
            try:
                properties = _parse_properties(self._read_text(node / "properties"))
                target = properties.get("gfx_target_version", 0)
                if target <= 0:
                    continue
                render_minor = properties["drm_render_minor"]
                render_name = f"renderD{render_minor}"
                device_path = self.sys_root / "class" / "drm" / render_name / "device"
                resolved = device_path.resolve(strict=True)
                bdf = resolved.name.lower()
                vendor = _read_hex(device_path / "vendor")
                device = _read_hex(device_path / "device")
                if vendor != 0x1002:
                    continue
                candidates.append(
                    GPUIdentity(
                        node=node.name,
                        arch=_gfx_arch(target),
                        integrated=_classify_integrated(device),
                        render_node=render_name,
                        pci_bdf=bdf,
                        vendor_id=vendor,
                        device_id=device,
                    )
                )
            except (KeyError, OSError, ValueError):
                issues.append(ProbeIssue("kfd-node-invalid", f"KFD node {node.name} is incomplete"))
        return tuple(candidates)

    def _rocminfo(
        self,
        selected: GPUIdentity,
        tools: tuple[ToolFact, ...],
        issues: list[ProbeIssue],
    ) -> tuple[tuple[str, ...], str | None]:
        tool = _tool(tools, "rocminfo")
        if tool is None or tool.path is None:
            return (), None
        process = self._run(tool.path, ())
        if process.outcome is not ProcessOutcome.EXITED or process.returncode != 0:
            issues.append(ProbeIssue("rocminfo-failed", "rocminfo inventory failed"))
            return (), None
        records = _parse_rocminfo(process.stdout)
        matching = [record for record in records if record.get("bdf") == selected.pci_bdf]
        if len(matching) != 1:
            issues.append(
                ProbeIssue(
                    "rocminfo-correlation",
                    "rocminfo did not identify exactly one selected GPU",
                )
            )
            return tuple(sorted({str(record["arch"]) for record in records})), None
        record = matching[0]
        return (
            tuple(sorted({str(value["arch"]) for value in records})),
            str(record.get("marketing")) if record.get("marketing") else None,
        )

    def _sample(
        self,
        profile: MachineProfileV1,
        selected: GPUIdentity,
        tools: tuple[ToolFact, ...],
        issues: list[ProbeIssue],
    ) -> tuple[str | None, tuple[TelemetrySample, ...]]:
        mode = profile.telemetry.amd_smi
        sources: tuple[str, ...]
        if mode == "required":
            sources = ("amd-smi",)
        elif mode == "disabled":
            sources = ("sysfs",)
        else:
            sources = ("amd-smi", "rocm-smi", "sysfs")
        first: TelemetrySample | None = None
        source: str | None = None
        for candidate in sources:
            sample = self._read_sample(candidate, 0, selected, tools)
            if sample.busy_pct is not None and sample.error is None:
                first = sample
                source = candidate
                break
        if first is None or source is None:
            issues.append(ProbeIssue("telemetry-unavailable", "no usable telemetry source"))
            return None, ()
        samples = [first]
        for index in range(1, 3):
            self.sleeper(profile.telemetry.sample_interval_ms / 1000)
            samples.append(self._read_sample(source, index, selected, tools))
        return source, tuple(samples)

    def _read_sample(
        self,
        source: str,
        index: int,
        selected: GPUIdentity,
        tools: tuple[ToolFact, ...],
    ) -> TelemetrySample:
        if source == "sysfs":
            return self._sysfs_sample(index, selected)
        tool = _tool(tools, source)
        if tool is None or tool.path is None:
            return _failed_sample(index, source, f"{source} is unavailable")
        args = AMD_SMI_ARGS if source == "amd-smi" else ROCM_SMI_ARGS
        process = self._run(tool.path, args)
        if process.outcome is not ProcessOutcome.EXITED or process.returncode != 0:
            return _failed_sample(index, source, f"{source} query failed")
        try:
            if source == "rocm-smi":
                return _parse_rocm_smi_sample(process.stdout, index, selected)
            return _parse_amd_smi_sample(process.stdout, index, selected)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return _failed_sample(index, source, f"{source} returned invalid data")

    def _sysfs_sample(self, index: int, selected: GPUIdentity) -> TelemetrySample:
        device = self.sys_root / "class" / "drm" / selected.render_node / "device"
        busy = _read_optional_number(device / "gpu_busy_percent")
        temperature: float | None = None
        power: float | None = None
        try:
            hwmons = sorted((device / "hwmon").iterdir())
        except OSError:
            hwmons = []
        for hwmon in hwmons:
            if temperature is None:
                value = _read_optional_number(hwmon / "temp1_input")
                temperature = value / 1000 if value is not None else None
            if power is None:
                value = _read_optional_number(hwmon / "power1_average")
                power = value / 1_000_000 if value is not None else None
        busy = _percentage(busy)
        return TelemetrySample(
            index=index,
            source="sysfs",
            busy_pct=busy,
            temperature_c=_finite(temperature),
            power_w=_nonnegative(power),
            sclk_hz=None,
            mclk_hz=None,
            error=None if busy is not None else "sysfs busy value is unavailable",
        )


def _tool(tools: tuple[ToolFact, ...], name: str) -> ToolFact | None:
    return next((tool for tool in tools if tool.name == name), None)


def _process_outcome(result: ProcessResult) -> str:
    if result.outcome is ProcessOutcome.EXITED:
        return "ok" if result.returncode == 0 else "failed"
    return result.outcome.value


def _bounded_summary(value: str) -> str | None:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    return " | ".join(lines[:3])[:1024] or None


def _parse_properties(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for line in value.splitlines():
        parts = line.split()
        if len(parts) == 2:
            result[parts[0]] = int(parts[1], 0)
    return result


def _gfx_arch(target: int) -> str:
    major = target // 10_000
    minor = (target // 100) % 100
    stepping = target % 100
    return f"gfx{major}{minor}{stepping}"


def _classify_integrated(device_id: int) -> bool | None:
    if device_id in KNOWN_INTEGRATED_AMD_DEVICES:
        return True
    if device_id in KNOWN_DISCRETE_AMD_DEVICES:
        return False
    return None


def _read_hex(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip(), 0)


def _bdf_from_id(value: int) -> str:
    bus = (value >> 8) & 0xFF
    device = (value >> 3) & 0x1F
    function = value & 0x07
    domain = (value >> 16) & 0xFFFF
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function}"


def _parse_rocminfo(value: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    current: dict[str, object] = {}
    for line in value.splitlines():
        match = re.match(r"\s*([^:]+):\s*(.*?)\s*$", line)
        if not match:
            continue
        name, raw = match.groups()
        if name.strip() == "Name" and raw.startswith("gfx"):
            if current.get("arch"):
                records.append(current)
            current = {"arch": raw.split()[0]}
        elif current and name.strip() == "Marketing Name":
            current["marketing"] = raw
        elif current and name.strip() == "BDFID":
            current["bdf"] = _bdf_from_id(int(raw.split("(", 1)[0].strip()))
    if current.get("arch"):
        records.append(current)
    return records


def _json_object(value: str) -> Any:
    start_candidates = [
        position for position in (value.find("{"), value.find("[")) if position >= 0
    ]
    if not start_candidates:
        raise json.JSONDecodeError("no JSON value", value, 0)
    return json.loads(value[min(start_candidates) :])


def _parse_rocm_smi_sample(value: str, index: int, selected: GPUIdentity) -> TelemetrySample:
    payload = _json_object(value)
    if not isinstance(payload, dict):
        raise TypeError
    matched: Mapping[str, Any] | None = None
    for record in payload.values():
        if not isinstance(record, Mapping):
            continue
        bus = str(record.get("PCI Bus", "")).lower()
        if bus == selected.pci_bdf:
            matched = record
            break
    if matched is None:
        raise ValueError("GPU identity mismatch")
    busy = _percentage(_number(matched.get("GPU use (%)")))
    temperature = _finite(_number(matched.get("Temperature (Sensor edge) (C)")))
    power = _nonnegative(_number(matched.get("Current Socket Graphics Package Power (W)")))
    return TelemetrySample(
        index=index,
        source="rocm-smi",
        busy_pct=busy,
        temperature_c=temperature,
        power_w=power,
        sclk_hz=_clock_hz(matched.get("sclk clock speed:")),
        mclk_hz=_clock_hz(matched.get("mclk clock speed:")),
        error=None if busy is not None else "rocm-smi busy value is invalid",
    )


def _walk_mappings(value: Any) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        records.append(value)
        for child in value.values():
            records.extend(_walk_mappings(child))
    elif isinstance(value, list):
        for child in value:
            records.extend(_walk_mappings(child))
    return records


def _parse_amd_smi_sample(value: str, index: int, selected: GPUIdentity) -> TelemetrySample:
    records = _walk_mappings(_json_object(value))
    matched = next(
        (
            record
            for record in records
            if str(
                record.get("bdf")
                or record.get("BDF")
                or record.get("pci_bus")
                or record.get("PCI Bus")
                or ""
            ).lower()
            == selected.pci_bdf
        ),
        None,
    )
    if matched is None:
        raise ValueError("GPU identity mismatch")
    busy = _percentage(
        _number(
            matched.get("gpu_busy_percent")
            or matched.get("gfx_activity")
            or matched.get("GPU use (%)")
        )
    )
    temperature = _finite(
        _number(matched.get("temperature_c") or matched.get("temperature") or matched.get("temp"))
    )
    power = _nonnegative(
        _number(matched.get("power_w") or matched.get("power") or matched.get("socket_power"))
    )
    return TelemetrySample(
        index=index,
        source="amd-smi",
        busy_pct=busy,
        temperature_c=temperature,
        power_w=power,
        sclk_hz=_clock_hz(matched.get("sclk") or matched.get("gfx_clock")),
        mclk_hz=_clock_hz(matched.get("mclk") or matched.get("memory_clock")),
        error=None if busy is not None else "amd-smi busy value is invalid",
    )


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None


def _finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _nonnegative(value: float | None) -> float | None:
    value = _finite(value)
    return value if value is not None and value >= 0 else None


def _percentage(value: float | None) -> float | None:
    value = _finite(value)
    return value if value is not None and 0 <= value <= 100 else None


def _clock_hz(value: Any) -> int | None:
    number = _number(value)
    if number is None or number < 0:
        return None
    text = str(value).lower()
    if "ghz" in text:
        return round(number * 1_000_000_000)
    if "mhz" in text:
        return round(number * 1_000_000)
    if "khz" in text:
        return round(number * 1_000)
    return round(number)


def _read_optional_number(path: Path) -> float | None:
    try:
        return _number(path.read_text(encoding="utf-8").strip())
    except OSError:
        return None


def _failed_sample(index: int, source: str, error: str) -> TelemetrySample:
    return TelemetrySample(index, source, None, None, None, None, None, error)

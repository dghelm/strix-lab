"""Bounded ca94157 ``llama-server`` adapter for one two-request smoke case."""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import secrets
import selectors
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from strixlab.evidence import RunSession
from strixlab.executable_identity import (
    ExecutableIdentity,
    hash_executable,
    require_stable_executable,
)
from strixlab.manifests import AbsolutePathString, DashId
from strixlab.models import (
    ModelError,
    ModelLease,
    ModelReceiptEvidence,
    ModelReceiptV1,
    lease_verified_model,
    require_receipt_inputs_match,
)
from strixlab.process import ProcessOutcome, ProcessResult, run_process
from strixlab.serialization import canonical_json_bytes

__all__ = [
    "LlamaServerCapabilitiesV1",
    "LlamaServerCaseV1",
    "LlamaServerGrammarError",
    "LlamaServerInputsV1",
    "LlamaServerIntegrityError",
    "LlamaServerParseError",
    "LlamaServerSampleV1",
    "build_completion_request",
    "build_server_argv",
    "parse_completion_response",
    "parse_help_capabilities",
    "parse_version_capability",
    "run_llama_server_case",
]

SOURCE_COMMIT = "ca94157f70a2776e8da6b6849b50b45a083d0478"
PROFILE: Literal["ca94157-v1"] = "ca94157-v1"
EVIDENCE_ROOT = "adapters/llama-server"
STREAM_LIMIT_BYTES = 256 * 1024
HTTP_BODY_LIMIT_BYTES = 256 * 1024
_MAX_TIMEOUT = 3600.0
_MAX_SENTINEL_ATTEMPTS = 32
_REQUIRED_HELP_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("model", re.compile(r"(?m)^-m,\s+--model\s+FNAME\s+")),
    ("host", re.compile(r"(?m)^--host\s+HOST\s+")),
    ("port", re.compile(r"(?m)^--port\s+PORT\s+")),
    ("offline", re.compile(r"(?m)^--offline\s+")),
    ("ui", re.compile(r"(?m)^--ui,\s+--webui,\s+--no-ui,\s+--no-webui\s+")),
    (
        "cache-prompt",
        re.compile(r"(?m)^--cache-prompt,\s+--no-cache-prompt\s+"),
    ),
    ("parallel", re.compile(r"(?m)^-np,\s+--parallel\s+N\s+")),
    ("threads-http", re.compile(r"(?m)^--threads-http\s+N\s+")),
    ("ctx-size", re.compile(r"(?m)^-c,\s+--ctx-size\s+N\s+")),
    (
        "gpu-layers",
        re.compile(r"(?m)^-ngl,\s+--gpu-layers,\s+--n-gpu-layers\s+N\s+"),
    ),
)
_VERSION_RE = re.compile(r"^version: 0\.3\.0-dev \(build 1, commit ca94157\)$")
_SENTINEL_RE = re.compile(r"^[0-9a-f]{32}$")

Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
BuildId = Annotated[str, Field(pattern=r"^build-sha256:[0-9a-f]{64}$")]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
ProcessOutcomeLiteral = Literal["exited", "timed_out", "spawn_failed", "capture_failed"]
ProcessErrorCategory = Literal[
    "none",
    "capture-failed",
    "spawn-failed",
    "timed-out",
    "output-oversized",
    "encoding-failed",
    "nonzero-exit",
]
RequestErrorCategory = Literal[
    "none",
    "transport",
    "http-status",
    "oversized",
    "encoding",
    "json",
    "shape",
    "isolation",
]
SampleStatus = Literal[
    "success",
    "capability-failed",
    "capture-failed",
    "spawn-failed",
    "port-unavailable",
    "readiness-failed",
    "request-failed",
    "response-failed",
    "isolation-failed",
    "shutdown-failed",
]


class LlamaServerIntegrityError(RuntimeError):
    """The binary/model binding drifted and no truthful sample can be published."""


class LlamaServerGrammarError(ValueError):
    """Pinned capability output does not match ca94157-v1."""


class LlamaServerParseError(ValueError):
    """An HTTP response does not match the pinned bounded response grammar."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class LlamaServerCaseV1(_Model):
    schema_version: Literal[1] = 1
    id: DashId
    prompt: StrictStr
    n_predict: Annotated[StrictInt, Field(ge=1, le=4096)]
    seed: Annotated[StrictInt, Field(ge=-(2**31), le=2**31 - 1)]
    context_size: Annotated[StrictInt, Field(ge=1, le=1_048_576)]
    gpu_layers: Annotated[StrictInt, Field(ge=0, le=999)]

    @model_validator(mode="after")
    def _prompt_contract(self) -> Self:
        size = len(self.prompt.encode("utf-8"))
        if size == 0 or size > 16 * 1024:
            raise ValueError("prompt must contain 1 through 16384 UTF-8 bytes")
        if any((ord(char) < 32 and char not in "\n\t") or ord(char) == 127 for char in self.prompt):
            raise ValueError("prompt contains a forbidden control character")
        return self


class LlamaServerInputsV1(_Model):
    schema_version: Literal[1] = 1
    build_id: BuildId
    source_commit: Literal["ca94157f70a2776e8da6b6849b50b45a083d0478"] = (
        "ca94157f70a2776e8da6b6849b50b45a083d0478"
    )
    binary_path: AbsolutePathString
    binary_sha256: Sha256Hex
    model_id: DashId
    model_path: AbsolutePathString
    model_sha256: Sha256Hex
    model_digest_status: Literal["verified"] = "verified"
    model_receipt_sha256: Sha256Hex
    model_receipt_evidence: ModelReceiptEvidence


class LlamaServerCapabilitiesV1(_Model):
    schema_version: Literal[1] = 1
    profile: Literal["ca94157-v1"] = PROFILE
    version: Literal["0.3.0-dev"] = "0.3.0-dev"
    build: Literal[1] = 1
    short_commit: Literal["ca94157"] = "ca94157"
    toolchain: Annotated[StrictStr, Field(min_length=1, max_length=1024)]
    required_options: tuple[StrictStr, ...]


class ProcessProjectionV1(_Model):
    schema_version: Literal[1] = 1
    outcome: ProcessOutcomeLiteral
    returncode: int | None
    duration_seconds: Annotated[float, Field(ge=0)]
    stdout_bytes: NonNegativeInt
    stderr_bytes: NonNegativeInt
    stdout_sha256: Sha256Hex
    stderr_sha256: Sha256Hex
    stdout_truncated: bool
    stderr_truncated: bool
    error_category: ProcessErrorCategory


class CapabilityAttemptV1(_Model):
    schema_version: Literal[1] = 1
    version: ProcessProjectionV1
    help: ProcessProjectionV1
    capabilities: LlamaServerCapabilitiesV1 | None
    reason: StrictStr | None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.capabilities is None) != (self.reason is not None):
            raise ValueError("capability attempt must carry capabilities or a failure reason")
        return self


class ReadinessV1(_Model):
    schema_version: Literal[1] = 1
    attempts: NonNegativeInt
    elapsed_seconds: Annotated[float, Field(ge=0)]
    final_status: int | None
    final_body_sha256: Sha256Hex | None
    ready: bool
    reason: StrictStr | None


class CompletionResponseV1(_Model):
    schema_version: Literal[1] = 1
    prompt: Annotated[StrictStr, Field(max_length=65_536)]
    content: Annotated[StrictStr, Field(max_length=262_144)]
    tokens: tuple[Annotated[StrictInt, Field(ge=0, le=2**31 - 1)], ...]

    @model_validator(mode="after")
    def _token_bound(self) -> Self:
        if len(self.tokens) > 8192:
            raise ValueError("response contains too many token IDs")
        return self


class RequestAttemptV1(_Model):
    schema_version: Literal[1] = 1
    ordinal: Annotated[StrictInt, Field(ge=1, le=2)]
    sentinel: Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{32}$")]
    request_sha256: Sha256Hex
    response_status: int | None
    response_bytes: NonNegativeInt
    response_sha256: Sha256Hex | None
    response: CompletionResponseV1 | None
    isolated: bool | None
    error_category: RequestErrorCategory


class ServerLifecycleV1(_Model):
    schema_version: Literal[1] = 1
    argv: tuple[StrictStr, ...]
    pid: NonNegativeInt | None
    returncode: int | None
    stdout_bytes: NonNegativeInt
    stderr_bytes: NonNegativeInt
    stdout_sha256: Sha256Hex
    stderr_sha256: Sha256Hex
    stdout_truncated: bool
    stderr_truncated: bool
    capture_error: StrictStr | None
    sigterm_sent: bool
    sigkill_sent: bool
    unexpected_exit: bool
    shutdown_error: StrictStr | None
    shutdown_elapsed_seconds: Annotated[float, Field(ge=0)]


class EvidenceArtifactV1(_Model):
    schema_version: Literal[1] = 1
    path: StrictStr
    media_type: StrictStr
    size: NonNegativeInt
    sha256: Sha256Hex


class LlamaServerSampleV1(_Model):
    schema_version: Literal[1] = 1
    profile: Literal["ca94157-v1"] = PROFILE
    status: SampleStatus
    reason: StrictStr
    case: LlamaServerCaseV1
    inputs: LlamaServerInputsV1
    capability_attempt: CapabilityAttemptV1
    readiness: ReadinessV1 | None
    requests: tuple[RequestAttemptV1, ...]
    lifecycle: ServerLifecycleV1 | None
    artifacts: tuple[EvidenceArtifactV1, ...]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if len({artifact.path for artifact in self.artifacts}) != len(self.artifacts):
            raise ValueError("sample evidence paths must be unique")
        if self.status == "capability-failed":
            if self.capability_attempt.capabilities is not None:
                raise ValueError("capability-failed sample cannot carry capabilities")
            if self.readiness is not None or self.requests or self.lifecycle is not None:
                raise ValueError("capability-failed sample cannot carry server execution")
        elif self.capability_attempt.capabilities is None:
            raise ValueError("server execution requires discovered capabilities")
        if self.status == "success":
            if self.reason != "success" or self.readiness is None or not self.readiness.ready:
                raise ValueError("success sample requires successful readiness")
            if len(self.requests) != 2 or not all(item.isolated for item in self.requests):
                raise ValueError("success sample requires two isolated responses")
            if (
                self.lifecycle is None
                or not self.lifecycle.sigterm_sent
                or self.lifecycle.sigkill_sent
                or self.lifecycle.unexpected_exit
                or self.lifecycle.shutdown_error is not None
                or self.lifecycle.returncode != 0
                or self.lifecycle.capture_error is not None
                or self.lifecycle.stdout_truncated
                or self.lifecycle.stderr_truncated
            ):
                raise ValueError("success sample requires graceful server shutdown")
        elif self.reason == "success":
            raise ValueError("failed sample cannot carry the success reason")
        return self


@dataclass(frozen=True, slots=True)
class _HttpResult:
    status: int | None
    body: bytes
    error: str | None
    oversized: bool = False


@dataclass(slots=True)
class _StreamState:
    digest: Any
    retained: bytearray
    total: int = 0
    truncated: bool = False


class _StreamCollector:
    """Own and drain two nonblocking server pipes without waiting for inherited EOF."""

    def __init__(self, stdout: Any, stderr: Any) -> None:
        self._files = (stdout, stderr)
        self._states = (
            _StreamState(hashlib.sha256(), bytearray()),
            _StreamState(hashlib.sha256(), bytearray()),
        )
        self._stop = threading.Event()
        self.error: str | None = None
        self._thread = threading.Thread(
            target=self._run, name="strixlab-llama-server-capture", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float) -> bool:
        self._stop.set()
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def snapshot(self) -> tuple[_StreamState, _StreamState]:
        return self._states

    def _consume(self, index: int, data: bytes) -> None:
        state = self._states[index]
        state.total += len(data)
        state.digest.update(data)
        remaining = STREAM_LIMIT_BYTES - len(state.retained)
        if remaining > 0:
            state.retained.extend(data[:remaining])
        if len(data) > remaining:
            state.truncated = True

    def _drain_fd(self, index: int, fd: int) -> bool:
        while True:
            try:
                data = os.read(fd, 64 * 1024)
            except BlockingIOError:
                return True
            if not data:
                return False
            self._consume(index, data)

    def _run(self) -> None:
        selector = selectors.DefaultSelector()
        try:
            for index, file in enumerate(self._files):
                fd = file.fileno()
                os.set_blocking(fd, False)
                selector.register(fd, selectors.EVENT_READ, index)
            while selector.get_map() and not self._stop.is_set():
                for key, _events in selector.select(0.05):
                    if not self._drain_fd(key.data, key.fd):
                        selector.unregister(key.fd)
            for key in tuple(selector.get_map().values()):
                self._drain_fd(key.data, key.fd)
                selector.unregister(key.fd)
        except Exception:
            self.error = "server stream capture failed"
        finally:
            selector.close()
            for file in self._files:
                try:
                    file.close()
                except OSError:
                    self.error = "server stream capture failed"


@dataclass(slots=True)
class _Evidence:
    """Inventory complete-local evidence that may contain binary response bytes."""

    run: RunSession
    base: str
    artifacts: list[EvidenceArtifactV1]

    def write(self, relative: str, content: bytes, media_type: str) -> str:
        path = f"{self.base}/{relative}"
        digest = hashlib.sha256(content).hexdigest()
        self.run.write_evidence(path, content)
        self.artifacts.append(
            EvidenceArtifactV1(
                path=path,
                media_type=media_type,
                size=len(content),
                sha256=digest,
            )
        )
        return digest

    def json(self, relative: str, value: object) -> None:
        self.write(relative, canonical_json_bytes(value), "application/json")


def _write_stream_evidence(evidence: _Evidence, name: str, state: _StreamState) -> None:
    content = bytes(state.retained)
    if not state.truncated:
        try:
            content.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            pass
        else:
            evidence.write(f"server/{name}.txt", content, "text/plain")
            return
    evidence.write(f"server/{name}.bin", content, "application/octet-stream")


def _recheck(binary_path: Path, binary: ExecutableIdentity, lease: ModelLease) -> None:
    # A lease.verify() drift raises ModelError, translated to the adapter integrity error
    # by run_llama_server_case's outer boundary.
    require_stable_executable(
        binary_path, binary, error=LlamaServerIntegrityError, subject="server binary"
    )
    lease.verify()


def _validate_timeout(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive number")
    if not math.isfinite(float(value)) or value <= 0 or value > _MAX_TIMEOUT:
        raise ValueError(f"{name} must be finite, positive, and at most {_MAX_TIMEOUT:g} seconds")


def build_server_argv(
    *, binary_path: str, model_path: str, port: int, case: LlamaServerCaseV1
) -> tuple[str, ...]:
    if isinstance(port, bool) or not isinstance(port, int) or not 1024 <= port <= 65535:
        raise ValueError("port must be an integer from 1024 through 65535")
    return (
        binary_path,
        "--model",
        model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--offline",
        "--no-ui",
        "--no-cache-prompt",
        "--parallel",
        "1",
        "--threads-http",
        "1",
        "--ctx-size",
        str(case.context_size),
        "--gpu-layers",
        str(case.gpu_layers),
    )


def build_completion_request(case: LlamaServerCaseV1, sentinel: str) -> bytes:
    if _SENTINEL_RE.fullmatch(sentinel) is None or sentinel in case.prompt:
        raise ValueError("sentinel must be unique lowercase 128-bit hex")
    return canonical_json_bytes(
        {
            "cache_prompt": False,
            "n_predict": case.n_predict,
            "prompt": f"strixlab-request-v1:{sentinel}\n{case.prompt}",
            "return_tokens": True,
            "seed": case.seed,
            "stream": False,
            "temperature": 0.0,
        }
    )


def parse_version_capability(stderr: str) -> str:
    lines = stderr.splitlines()
    if len(lines) != 2 or _VERSION_RE.fullmatch(lines[0]) is None:
        raise LlamaServerGrammarError("version output does not match ca94157-v1")
    toolchain = lines[1]
    if (
        not toolchain
        or len(toolchain.encode("utf-8")) > 1024
        or any(ord(c) < 32 or ord(c) == 127 for c in toolchain)
    ):
        raise LlamaServerGrammarError("version toolchain line is invalid")
    return toolchain


def parse_help_capabilities(help_stdout: str, *, toolchain: str) -> LlamaServerCapabilitiesV1:
    if not help_stdout.startswith("----- common params -----\n"):
        raise LlamaServerGrammarError("help section header does not match ca94157-v1")
    options: list[str] = []
    for name, pattern in _REQUIRED_HELP_PATTERNS:
        if len(pattern.findall(help_stdout)) != 1:
            raise LlamaServerGrammarError(
                f"required server option {name!r} is missing or duplicated"
            )
        options.append(name)
    return LlamaServerCapabilitiesV1(toolchain=toolchain, required_options=tuple(options))


def _strict_json_object(body: bytes) -> dict[str, object]:
    try:
        text = body.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise LlamaServerParseError("response body is not UTF-8") from exc

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise LlamaServerParseError("response JSON contains a duplicate key")
            value[key] = item
        return value

    try:
        value = json.loads(text, object_pairs_hook=reject_duplicates)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise LlamaServerParseError("response body is not JSON") from exc
    if not isinstance(value, dict):
        raise LlamaServerParseError("response body is not a JSON object")
    return cast(dict[str, object], value)


def parse_completion_response(body: bytes) -> CompletionResponseV1:
    value = _strict_json_object(body)
    tokens = value.get("tokens")
    if not isinstance(tokens, list):
        raise LlamaServerParseError("completion response shape is invalid")
    try:
        return CompletionResponseV1.model_validate(
            {
                "prompt": value.get("prompt"),
                "content": value.get("content"),
                "tokens": tuple(tokens),
            }
        )
    except ValueError as exc:
        raise LlamaServerParseError("completion response shape is invalid") from exc


def _project_process(result: ProcessResult) -> tuple[ProcessProjectionV1, str | None, str | None]:
    outcomes: dict[ProcessOutcome, ProcessOutcomeLiteral] = {
        ProcessOutcome.EXITED: "exited",
        ProcessOutcome.TIMED_OUT: "timed_out",
        ProcessOutcome.SPAWN_FAILED: "spawn_failed",
        ProcessOutcome.CAPTURE_FAILED: "capture_failed",
    }
    stdout_bytes = result.stdout.encode()
    stderr_bytes = result.stderr.encode()
    stdout_exact = (
        not result.stdout_truncated
        and len(stdout_bytes) == result.stdout_bytes
        and hashlib.sha256(stdout_bytes).hexdigest() == result.stdout_sha256
    )
    stderr_exact = (
        not result.stderr_truncated
        and len(stderr_bytes) == result.stderr_bytes
        and hashlib.sha256(stderr_bytes).hexdigest() == result.stderr_sha256
    )
    if result.outcome is ProcessOutcome.CAPTURE_FAILED:
        category: ProcessErrorCategory = "capture-failed"
    elif result.outcome is ProcessOutcome.SPAWN_FAILED:
        category = "spawn-failed"
    elif result.outcome is ProcessOutcome.TIMED_OUT:
        category = "timed-out"
    elif result.stdout_truncated or result.stderr_truncated:
        category = "output-oversized"
    elif not stdout_exact or not stderr_exact:
        category = "encoding-failed"
    elif result.returncode != 0:
        category = "nonzero-exit"
    else:
        category = "none"
    projection = ProcessProjectionV1(
        outcome=outcomes[result.outcome],
        returncode=result.returncode,
        duration_seconds=max(0.0, result.duration),
        stdout_bytes=result.stdout_bytes,
        stderr_bytes=result.stderr_bytes,
        stdout_sha256=result.stdout_sha256,
        stderr_sha256=result.stderr_sha256,
        stdout_truncated=result.stdout_truncated,
        stderr_truncated=result.stderr_truncated,
        error_category=category,
    )
    return (
        projection,
        result.stdout if stdout_exact else None,
        result.stderr if stderr_exact else None,
    )


def _probe(
    *,
    argv: tuple[str, ...],
    cwd: Path,
    environment: Mapping[str, str],
    timeout: float,
    evidence: _Evidence,
    name: str,
) -> tuple[ProcessProjectionV1, str | None, str | None]:
    result = run_process(
        argv,
        cwd=cwd,
        timeout=timeout,
        inherit_env=False,
        base_env=environment,
        output_limit_bytes=STREAM_LIMIT_BYTES,
    )
    projection, stdout, stderr = _project_process(result)
    if stdout is not None:
        evidence.write(f"capabilities/{name}.stdout.txt", stdout.encode(), "text/plain")
    if stderr is not None:
        evidence.write(f"capabilities/{name}.stderr.txt", stderr.encode(), "text/plain")
    evidence.json(f"capabilities/{name}.process.json", projection.model_dump(mode="json"))
    return projection, stdout, stderr


def _http_exchange(
    *, port: int, method: str, path: str, body: bytes | None, timeout: float
) -> _HttpResult:
    deadline = time.monotonic() + timeout
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    collected = bytearray()
    status: int | None = None

    def remaining() -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError
        return value

    try:
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        connection.connect()
        assert connection.sock is not None
        connection.sock.settimeout(remaining())
        connection.request(method, path, body=body, headers=headers)
        connection.sock.settimeout(remaining())
        response = connection.getresponse()
        status = response.status
        while True:
            if connection.sock is not None:
                connection.sock.settimeout(remaining())
            chunk = response.read1(min(64 * 1024, HTTP_BODY_LIMIT_BYTES + 1 - len(collected)))
            if not chunk:
                break
            collected.extend(chunk)
            if len(collected) > HTTP_BODY_LIMIT_BYTES:
                return _HttpResult(
                    status,
                    bytes(collected[:HTTP_BODY_LIMIT_BYTES]),
                    "response body exceeded limit",
                    True,
                )
        return _HttpResult(status, bytes(collected), None)
    except (OSError, http.client.HTTPException, TimeoutError):
        return _HttpResult(status, bytes(collected), "HTTP transport failed")
    finally:
        connection.close()


def _sentinels(prompt: str, factory: Callable[[], str]) -> tuple[str, str]:
    values: list[str] = []
    for _attempt in range(_MAX_SENTINEL_ATTEMPTS):
        value = factory()
        if _SENTINEL_RE.fullmatch(value) and value not in prompt and value not in values:
            values.append(value)
            if len(values) == 2:
                return values[0], values[1]
    raise ValueError("unable to allocate distinct request sentinels")


def _port_available(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        probe.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _attempt_from_http(
    ordinal: int,
    sentinel: str,
    request_sha256: str,
    response_sha256: str | None,
    result: _HttpResult,
) -> RequestAttemptV1:
    response: CompletionResponseV1 | None = None
    if result.error is not None:
        category: RequestErrorCategory = "oversized" if result.oversized else "transport"
    elif result.status != 200:
        category = "http-status"
    else:
        try:
            response = parse_completion_response(result.body)
            category = "none"
        except LlamaServerParseError as exc:
            message = str(exc)
            category = (
                "encoding" if "UTF-8" in message else "json" if "JSON" in message else "shape"
            )
    return RequestAttemptV1(
        ordinal=ordinal,
        sentinel=sentinel,
        request_sha256=request_sha256,
        response_status=result.status,
        response_bytes=len(result.body),
        response_sha256=response_sha256,
        response=response,
        isolated=None,
        error_category=category,
    )


def _finalize(
    evidence: _Evidence,
    *,
    lease: ModelLease,
    status: SampleStatus,
    reason: str,
    case: LlamaServerCaseV1,
    inputs: LlamaServerInputsV1,
    capability: CapabilityAttemptV1,
    readiness: ReadinessV1 | None,
    requests: Sequence[RequestAttemptV1],
    lifecycle: ServerLifecycleV1 | None,
) -> LlamaServerSampleV1:
    sample = LlamaServerSampleV1(
        status=status,
        reason=reason,
        case=case,
        inputs=inputs,
        capability_attempt=capability,
        readiness=readiness,
        requests=tuple(requests),
        lifecycle=lifecycle,
        artifacts=tuple(evidence.artifacts),
    )
    # lease.verify() runs immediately before the terminal write, so finalizer-time drift
    # raises the integrity error (via the runner's outer boundary) before ``sample.json``
    # exists.
    lease.verify()
    # Publish the terminal sample.json twice with identical canonical bytes: once as local
    # evidence (so it sits beside the binary response siblings and is covered by the run
    # checksums) and once as a portable entry at the same logical path (so a consumer can
    # authenticate it against a content-addressed blob without reading the local tree).
    payload = canonical_json_bytes(sample.model_dump(mode="json"))
    sample_path = f"{evidence.base}/sample.json"
    evidence.run.write_evidence(sample_path, payload)
    evidence.run.write_portable(sample_path, payload, media_type="application/json", role="samples")
    return sample


def run_llama_server_case(
    *,
    case: LlamaServerCaseV1,
    inputs: LlamaServerInputsV1,
    receipt: ModelReceiptV1,
    run: RunSession,
    environment: Mapping[str, str],
    cwd: Path,
    port: int,
    capability_timeout: float,
    readiness_timeout: float,
    request_timeout: float,
    shutdown_timeout: float,
    nonce_factory: Callable[[], str] = lambda: secrets.token_hex(16),
) -> LlamaServerSampleV1:
    """Run one bounded two-request local-server case; the caller owns ``run``.

    The verified ``receipt`` is bound to ``inputs`` and held across the whole child
    lifetime through :func:`lease_verified_model`, so the long-lived ``llama-server``
    child opens the receipt-bound inode via ``/proc/self/fd/<fd>`` even if the public
    pathname is swapped mid-run. Integrity drift propagates without a ``sample.json``.
    """

    for name, value in (
        ("capability_timeout", capability_timeout),
        ("readiness_timeout", readiness_timeout),
        ("request_timeout", request_timeout),
        ("shutdown_timeout", shutdown_timeout),
    ):
        _validate_timeout(value, name)
    binary_path = Path(inputs.binary_path)
    binary = hash_executable(binary_path, error=LlamaServerIntegrityError, subject="server binary")
    if binary.sha256 != inputs.binary_sha256:
        raise LlamaServerIntegrityError("server binary SHA-256 does not match input binding")
    try:
        require_receipt_inputs_match(
            receipt,
            model_id=inputs.model_id,
            model_path=inputs.model_path,
            model_sha256=inputs.model_sha256,
            model_receipt_sha256=inputs.model_receipt_sha256,
            model_receipt_evidence=inputs.model_receipt_evidence,
        )
    except ModelError as exc:
        raise LlamaServerIntegrityError(str(exc)) from exc
    sentinels = _sentinels(case.prompt, nonce_factory)
    try:
        with lease_verified_model(receipt) as lease:
            return _drive_leased_server(
                case=case,
                inputs=inputs,
                lease=lease,
                binary_path=binary_path,
                binary=binary,
                sentinels=sentinels,
                run=run,
                environment=environment,
                cwd=cwd,
                port=port,
                capability_timeout=capability_timeout,
                readiness_timeout=readiness_timeout,
                request_timeout=request_timeout,
                shutdown_timeout=shutdown_timeout,
            )
    except ModelError as exc:
        raise LlamaServerIntegrityError(str(exc)) from exc


def _drive_leased_server(
    *,
    case: LlamaServerCaseV1,
    inputs: LlamaServerInputsV1,
    lease: ModelLease,
    binary_path: Path,
    binary: ExecutableIdentity,
    sentinels: tuple[str, str],
    run: RunSession,
    environment: Mapping[str, str],
    cwd: Path,
    port: int,
    capability_timeout: float,
    readiness_timeout: float,
    request_timeout: float,
    shutdown_timeout: float,
) -> LlamaServerSampleV1:
    # The model operand is the receipt-bound descriptor, not the public path.
    argv = build_server_argv(
        binary_path=inputs.binary_path, model_path=lease.descriptor_path, port=port, case=case
    )
    evidence = _Evidence(run, f"{EVIDENCE_ROOT}/{case.id}", [])

    version_p, version_out, version_err = _probe(
        argv=(inputs.binary_path, "--version"),
        cwd=cwd,
        environment=environment,
        timeout=capability_timeout,
        evidence=evidence,
        name="version",
    )
    _recheck(binary_path, binary, lease)
    help_p, help_out, help_err = _probe(
        argv=(inputs.binary_path, "--help"),
        cwd=cwd,
        environment=environment,
        timeout=capability_timeout,
        evidence=evidence,
        name="help",
    )
    capabilities: LlamaServerCapabilitiesV1 | None = None
    cap_reason: str | None = None
    if version_p.error_category != "none" or help_p.error_category != "none":
        cap_reason = "capability process failed"
    elif version_out != "" or help_err != "" or version_err is None or help_out is None:
        cap_reason = "capability stream contract failed"
    else:
        try:
            toolchain = parse_version_capability(version_err)
            capabilities = parse_help_capabilities(help_out, toolchain=toolchain)
        except LlamaServerGrammarError:
            cap_reason = "capability grammar failed"
    capability = CapabilityAttemptV1(
        version=version_p, help=help_p, capabilities=capabilities, reason=cap_reason
    )
    evidence.json("capabilities/attempt.json", capability.model_dump(mode="json"))
    _recheck(binary_path, binary, lease)
    if capabilities is None:
        return _finalize(
            evidence,
            lease=lease,
            status="capability-failed",
            reason=cap_reason or "capability failed",
            case=case,
            inputs=inputs,
            capability=capability,
            readiness=None,
            requests=(),
            lifecycle=None,
        )
    if not _port_available(port):
        _recheck(binary_path, binary, lease)
        return _finalize(
            evidence,
            lease=lease,
            status="port-unavailable",
            reason="loopback port is unavailable",
            case=case,
            inputs=inputs,
            capability=capability,
            readiness=None,
            requests=(),
            lifecycle=None,
        )

    _recheck(binary_path, binary, lease)
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            pass_fds=(lease.descriptor,),
        )
    except OSError:
        _recheck(binary_path, binary, lease)
        return _finalize(
            evidence,
            lease=lease,
            status="spawn-failed",
            reason="server spawn failed",
            case=case,
            inputs=inputs,
            capability=capability,
            readiness=None,
            requests=(),
            lifecycle=None,
        )
    assert process.stdout is not None and process.stderr is not None
    collector = _StreamCollector(process.stdout, process.stderr)
    collector.start()
    readiness: ReadinessV1 | None = None
    request_attempts: list[RequestAttemptV1] = []
    primary_status: SampleStatus = "success"
    primary_reason = "success"
    sigterm_sent = False
    sigkill_sent = False
    unexpected_exit = False
    shutdown_error: str | None = None
    shutdown_started = 0.0
    collector_done = False
    try:
        start = time.monotonic()
        deadline = start + readiness_timeout
        attempts = 0
        last = _HttpResult(None, b"", None)
        while time.monotonic() < deadline:
            if process.poll() is not None:
                primary_status, primary_reason = (
                    "readiness-failed",
                    "server exited before readiness",
                )
                break
            attempts += 1
            last = _http_exchange(
                port=port,
                method="GET",
                path="/health",
                body=None,
                timeout=min(1.0, max(0.001, deadline - time.monotonic())),
            )
            if last.status == 200:
                try:
                    health = _strict_json_object(last.body)
                except LlamaServerParseError:
                    health = {}
                if health == {"status": "ok"}:
                    break
                primary_status, primary_reason = "readiness-failed", "readiness body is invalid"
                break
            if last.status == 503:
                try:
                    loading = _strict_json_object(last.body)
                except LlamaServerParseError:
                    primary_status, primary_reason = "readiness-failed", "loading body is invalid"
                    break
                error = loading.get("error")
                if (
                    not isinstance(error, dict)
                    or error.get("code") != 503
                    or error.get("message") != "Loading model"
                ):
                    primary_status, primary_reason = "readiness-failed", "loading body is invalid"
                    break
            elif last.error is None:
                primary_status, primary_reason = "readiness-failed", "unexpected readiness status"
                break
            time.sleep(0.05)
        else:
            primary_status, primary_reason = "readiness-failed", "readiness deadline expired"
        ready = primary_status == "success"
        readiness = ReadinessV1(
            attempts=attempts,
            elapsed_seconds=max(0.0, time.monotonic() - start),
            final_status=last.status,
            final_body_sha256=hashlib.sha256(last.body).hexdigest() if last.body else None,
            ready=ready,
            reason=None if ready else primary_reason,
        )
        evidence.json("readiness/final.json", readiness.model_dump(mode="json"))

        if ready:
            for ordinal, sentinel in enumerate(sentinels, 1):
                _recheck(binary_path, binary, lease)
                request = build_completion_request(case, sentinel)
                request_sha256 = evidence.write(
                    f"requests/{ordinal:04d}/request.json", request, "application/json"
                )
                result = _http_exchange(
                    port=port,
                    method="POST",
                    path="/completion",
                    body=request,
                    timeout=request_timeout,
                )
                response_sha256 = evidence.write(
                    f"requests/{ordinal:04d}/response.body.bin",
                    result.body,
                    "application/octet-stream",
                )
                attempt = _attempt_from_http(
                    ordinal,
                    sentinel,
                    request_sha256,
                    response_sha256 if result.body or result.status is not None else None,
                    result,
                )
                request_attempts.append(attempt)
                if attempt.error_category != "none" and primary_status == "success":
                    primary_status = (
                        "request-failed"
                        if attempt.error_category in ("transport", "http-status", "oversized")
                        else "response-failed"
                    )
                    primary_reason = f"request {ordinal} {attempt.error_category}"
            if len(request_attempts) == 2 and all(
                item.response is not None for item in request_attempts
            ):
                revised: list[RequestAttemptV1] = []
                isolated = True
                for item in request_attempts:
                    assert item.response is not None
                    other = sentinels[1 if item.ordinal == 1 else 0]
                    verdict = (
                        item.response.prompt.count(item.sentinel) == 1
                        and other not in item.response.prompt
                    )
                    isolated &= verdict
                    revised.append(
                        item.model_copy(
                            update={
                                "isolated": verdict,
                                "error_category": "none" if verdict else "isolation",
                            }
                        )
                    )
                request_attempts = revised
                if not isolated:
                    primary_status, primary_reason = (
                        "isolation-failed",
                        "response/request sentinel isolation failed",
                    )
            for attempt in request_attempts:
                evidence.json(
                    f"requests/{attempt.ordinal:04d}/attempt.json",
                    attempt.model_dump(mode="json"),
                )
    finally:
        active_error = sys.exception()
        shutdown_started = time.monotonic()
        try:
            if process.poll() is None:
                sigterm_sent = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    unexpected_exit = True
                except OSError:
                    shutdown_error = "server SIGTERM failed"
                try:
                    process.wait(timeout=shutdown_timeout)
                except subprocess.TimeoutExpired:
                    sigkill_sent = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except OSError:
                        shutdown_error = "server SIGKILL failed"
                        with suppress(OSError):
                            process.kill()
                    try:
                        process.wait(timeout=shutdown_timeout)
                    except subprocess.TimeoutExpired:
                        shutdown_error = "server did not exit after SIGKILL"
            else:
                unexpected_exit = True
                process.wait(timeout=shutdown_timeout)
            if process.returncode not in (None, 0) and not sigkill_sent:
                unexpected_exit = True
        except (OSError, subprocess.SubprocessError):
            shutdown_error = "server shutdown failed"
            if active_error is not None:
                active_error.add_note(shutdown_error)
        finally:
            try:
                collector_done = collector.stop(max(0.1, min(shutdown_timeout, 5.0)))
            except RuntimeError:
                collector_done = False
            if not collector_done:
                note = "server stream collector did not stop"
                if active_error is not None:
                    active_error.add_note(note)
            try:
                _recheck(binary_path, binary, lease)
            except LlamaServerIntegrityError:
                if active_error is None:
                    raise
                active_error.add_note("server input identity also drifted during cleanup")

    if not collector_done:
        raise LlamaServerIntegrityError("server stream collector did not stop")

    stdout_state, stderr_state = collector.snapshot()
    lifecycle = ServerLifecycleV1(
        argv=argv,
        pid=process.pid,
        returncode=process.returncode,
        stdout_bytes=stdout_state.total,
        stderr_bytes=stderr_state.total,
        stdout_sha256=stdout_state.digest.hexdigest(),
        stderr_sha256=stderr_state.digest.hexdigest(),
        stdout_truncated=stdout_state.truncated,
        stderr_truncated=stderr_state.truncated,
        capture_error=collector.error if collector_done else "server stream collector did not stop",
        sigterm_sent=sigterm_sent,
        sigkill_sent=sigkill_sent,
        unexpected_exit=unexpected_exit,
        shutdown_error=shutdown_error,
        shutdown_elapsed_seconds=max(0.0, time.monotonic() - shutdown_started),
    )
    _write_stream_evidence(evidence, "stdout", stdout_state)
    _write_stream_evidence(evidence, "stderr", stderr_state)
    evidence.json("server/process.json", lifecycle.model_dump(mode="json"))
    _recheck(binary_path, binary, lease)
    if (
        lifecycle.capture_error is not None
        or lifecycle.stdout_truncated
        or lifecycle.stderr_truncated
    ):
        primary_status, primary_reason = "capture-failed", "server process capture is incomplete"
    elif primary_status == "success" and (
        sigkill_sent
        or unexpected_exit
        or shutdown_error is not None
        or lifecycle.returncode != 0
        or not sigterm_sent
    ):
        primary_status, primary_reason = "shutdown-failed", "server shutdown was not graceful"
    return _finalize(
        evidence,
        lease=lease,
        status=primary_status,
        reason=primary_reason,
        case=case,
        inputs=inputs,
        capability=capability,
        readiness=readiness,
        requests=request_attempts,
        lifecycle=lifecycle,
    )

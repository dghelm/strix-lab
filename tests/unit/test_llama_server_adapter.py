from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from _model_fixtures import build_verified_receipt
from pydantic import ValidationError

from strixlab.adapters import llama_server as llama_server_module
from strixlab.adapters.llama_server import (
    LlamaServerCaseV1,
    LlamaServerGrammarError,
    LlamaServerInputsV1,
    LlamaServerIntegrityError,
    LlamaServerParseError,
    LlamaServerSampleV1,
    build_completion_request,
    build_server_argv,
    parse_completion_response,
    parse_help_capabilities,
    parse_version_capability,
    run_llama_server_case,
)
from strixlab.models import ModelReceiptV1, receipt_evidence_digest

FIXTURES = Path(__file__).parents[1] / "fixtures" / "llama_server" / "ca94157"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
CASE = LlamaServerCaseV1(
    id="server-smoke",
    prompt="Return one token.",
    n_predict=1,
    seed=7,
    context_size=512,
    gpu_layers=0,
)


class _FakeRun:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.portable: dict[str, bytes] = {}

    def write_evidence(self, relative: str, content: bytes) -> Path:
        assert relative not in self.files
        self.files[relative] = content
        return Path("/fake") / relative

    def write_portable(
        self, logical_path: str, content: bytes, *, media_type: str, role: str
    ) -> None:
        assert logical_path not in self.portable
        self.portable[logical_path] = content


class _FailingRun(_FakeRun):
    def __init__(self, fail_suffix: str) -> None:
        super().__init__()
        self.fail_suffix = fail_suffix

    def write_evidence(self, relative: str, content: bytes) -> Path:
        if relative.endswith(self.fail_suffix):
            raise OSError("synthetic evidence failure")
        return super().write_evidence(relative, content)


def _write_server(tmp_path: Path) -> Path:
    path = tmp_path / "fake-llama-server"
    source = f"""#!{sys.executable}
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

if "--version" in sys.argv:
    sys.stderr.write(
        "version: 0.3.0-dev "
        "(build 0, commit ca94157f70a2776e8da6b6849b50b45a083d0478)\\n"
    )
    sys.stderr.write("built with tests for Linux x86_64\\n")
    raise SystemExit(0)
if "--help" in sys.argv:
    sys.stdout.write({(FIXTURES / "help.stdout.txt").read_text(encoding="utf-8")!r})
    raise SystemExit(0)

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--model")
parser.add_argument("--host")
parser.add_argument("--port", type=int)
parser.add_argument("--offline", action="store_true")
parser.add_argument("--no-ui", action="store_true")
parser.add_argument("--no-cache-prompt", action="store_true")
parser.add_argument("--parallel")
parser.add_argument("--threads-http")
parser.add_argument("--ctx-size")
parser.add_argument("--gpu-layers")
args = parser.parse_args()
mode = os.environ.get("FAKE_MODE", "success")
seen = []
health_checks = 0
pid_file = os.environ.get("PID_FILE")
if pid_file:
    with open(pid_file, "w", encoding="ascii") as handle:
        handle.write(str(os.getpid()))
if mode == "truncate-output":
    sys.stdout.buffer.write(b"x" * ({256 * 1024} + 1))
    sys.stdout.buffer.flush()
if mode == "inherit-pipe":
    subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.5)"],
        stdout=sys.stdout,
        stderr=sys.stderr,
        start_new_session=True,
    )

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def _send(self, status, value):
        body = json.dumps(value, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        global health_checks
        if self.path != "/health":
            self._send(404, {{}})
        elif health_checks == 0:
            health_checks += 1
            self._send(503, {{"error": {{"code": 503, "message": "Loading model"}}}})
        elif mode == "bad-health":
            self._send(200, {{"status": "wrong"}})
        else:
            self._send(200, {{"status": "ok"}})

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        request = json.loads(self.rfile.read(length))
        prompt = request["prompt"]
        ordinal = len(seen) + 1
        if mode == "stale" and seen:
            prompt = seen[0]
        seen.append(request["prompt"])
        if mode == "first-http-error" and ordinal == 1:
            self._send(500, {{"error": "first request failed"}})
            return
        if mode == "stall-headers":
            import time
            time.sleep(0.2)
        if mode == "mid-body" and ordinal == 1:
            partial = b'{{"prompt"'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(partial)
            self.wfile.flush()
            import time
            time.sleep(0.2)
            return
        if mode == "malformed":
            body = b"not-json"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if mode == "oversized":
            body = b"x" * ({256 * 1024} + 1)
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._send(200, {{"prompt": prompt, "content": "x", "tokens": [1]}})
        if mode == "drift-model" and ordinal == 2:
            with open(args.model, "ab") as handle:
                handle.write(b"drift")
        if mode == "early-death" and ordinal == 2:
            os._exit(3)

server = HTTPServer((args.host, args.port), Handler)
server.timeout = 0.05
stop = False

def on_term(_signum, _frame):
    global stop
    if mode != "ignore-term":
        stop = True

signal.signal(signal.SIGTERM, on_term)
while not stop:
    server.handle_request()
server.server_close()
"""
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def _receipt(model: Path) -> ModelReceiptV1:
    return build_verified_receipt(model.parent, model, model_id="tiny-model")


def _inputs(binary: Path, receipt: ModelReceiptV1) -> LlamaServerInputsV1:
    return LlamaServerInputsV1(
        build_id=f"build-sha256:{'1' * 64}",
        binary_path=str(binary),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        model_id=receipt.manifest_id,
        model_path=receipt.primary.local_path,
        model_sha256=receipt.primary.sha256,
        model_receipt_sha256=receipt_evidence_digest(receipt.evidence),
        model_receipt_evidence=receipt.evidence,
    )


def _port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _execute(
    tmp_path: Path,
    mode: str,
    *,
    request_timeout: float = 3.0,
    shutdown_timeout: float = 0.3,
) -> tuple[LlamaServerSampleV1, _FakeRun]:
    binary = _write_server(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    receipt = _receipt(model)
    run = _FakeRun()
    nonces = iter(("a" * 32, "b" * 32))
    sample = run_llama_server_case(
        case=CASE,
        inputs=_inputs(binary, receipt),
        receipt=receipt,
        run=run,  # type: ignore[arg-type]
        environment={"FAKE_MODE": mode},
        cwd=tmp_path,
        port=_port(),
        capability_timeout=3.0,
        readiness_timeout=3.0,
        request_timeout=request_timeout,
        shutdown_timeout=shutdown_timeout,
        nonce_factory=lambda: next(nonces),
    )
    return sample, run


def test_case_and_builders_are_strict() -> None:
    with pytest.raises(ValidationError):
        LlamaServerCaseV1(
            id="bad",
            prompt="bad\0prompt",
            n_predict=1,
            seed=0,
            context_size=1,
            gpu_layers=0,
        )
    argv = build_server_argv(
        binary_path="/bin/server",
        model_path="/models/m.gguf",
        port=12345,
        case=CASE,
    )
    assert argv == (
        "/bin/server",
        "--model",
        "/models/m.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "12345",
        "--offline",
        "--no-ui",
        "--no-cache-prompt",
        "--parallel",
        "1",
        "--threads-http",
        "1",
        "--ctx-size",
        "512",
        "--gpu-layers",
        "0",
    )
    request = build_completion_request(CASE, "a" * 32)
    assert b'"cache_prompt": false' in request
    assert b'"temperature": 0.0' in request
    assert request.endswith(b"\n")


def test_pinned_capability_fixtures() -> None:
    toolchain = parse_version_capability(
        (FIXTURES / "version.stderr.txt").read_text(encoding="utf-8")
    )
    capabilities = parse_help_capabilities(
        (FIXTURES / "help.stdout.txt").read_text(encoding="utf-8"),
        toolchain=toolchain,
    )
    assert capabilities.commit == "ca94157f70a2776e8da6b6849b50b45a083d0478"
    assert capabilities.required_options == (
        "model",
        "host",
        "port",
        "offline",
        "ui",
        "cache-prompt",
        "parallel",
        "threads-http",
        "ctx-size",
        "gpu-layers",
    )
    with pytest.raises(LlamaServerGrammarError):
        parse_help_capabilities(
            "----- common params -----\n--modelish FNAME\n", toolchain=toolchain
        )


def test_completion_response_is_strict_and_allows_immediate_eos() -> None:
    response = parse_completion_response(
        b'{"content":"","prompt":"strixlab-request-v1:abc\\nx","tokens":[]}'
    )
    assert response.tokens == ()
    with pytest.raises(LlamaServerParseError):
        parse_completion_response(b'{"content":"x","prompt":"p","tokens":[true]}')
    with pytest.raises(LlamaServerParseError):
        parse_completion_response(b'{"content":"x","content":"y","prompt":"p","tokens":[1]}')


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("success", "success"),
        ("stale", "isolation-failed"),
        ("bad-health", "readiness-failed"),
        ("ignore-term", "shutdown-failed"),
    ],
)
def test_managed_server_lifecycle(tmp_path: Path, mode: str, expected: str) -> None:
    sample, run = _execute(tmp_path, mode, shutdown_timeout=0.1)
    assert sample.status == expected
    assert sample.lifecycle is not None
    assert sample.lifecycle.returncode is not None
    sample_path = f"adapters/llama-server/{CASE.id}/sample.json"
    assert sample_path in run.files
    # The terminal sample.json is also published as a portable entry with identical bytes,
    # so a consumer can authenticate it against a content-addressed blob.
    assert run.portable.get(sample_path) == run.files[sample_path]
    if mode == "success":
        assert sample.readiness is not None and sample.readiness.attempts >= 2
        assert len(sample.requests) == 2
        assert all(request.isolated for request in sample.requests)
        for ordinal in (1, 2):
            attempt = json.loads(
                run.files[f"adapters/llama-server/{CASE.id}/requests/{ordinal:04d}/attempt.json"]
            )
            assert attempt["isolated"] is True
        assert sample.lifecycle.sigterm_sent
        assert not sample.lifecycle.sigkill_sent
    if mode == "ignore-term":
        assert sample.lifecycle.sigkill_sent


def test_second_request_runs_after_first_http_failure(tmp_path: Path) -> None:
    sample, _run = _execute(tmp_path, "first-http-error")
    assert sample.status == "request-failed"
    assert len(sample.requests) == 2
    assert sample.requests[0].error_category == "http-status"
    assert sample.requests[1].error_category == "none"


@pytest.mark.parametrize("mode", ["stall-headers", "mid-body"])
def test_http_deadline_is_bounded_and_records_both_attempts(tmp_path: Path, mode: str) -> None:
    sample, run = _execute(tmp_path, mode, request_timeout=0.05)
    assert sample.status == "request-failed"
    assert len(sample.requests) == 2
    assert all(item.error_category == "transport" for item in sample.requests)
    if mode == "mid-body":
        path = f"adapters/llama-server/{CASE.id}/requests/0001/response.body.bin"
        assert run.files[path] == b'{"prompt"'


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("malformed", "response-failed"), ("oversized", "request-failed")],
)
def test_invalid_completion_is_structured(tmp_path: Path, mode: str, expected: str) -> None:
    sample, _run = _execute(tmp_path, mode)
    assert sample.status == expected
    assert len(sample.requests) == 2


def test_unexpected_server_exit_disqualifies_success(tmp_path: Path) -> None:
    sample, _run = _execute(tmp_path, "early-death")
    assert sample.status == "shutdown-failed"
    assert sample.lifecycle is not None
    assert sample.lifecycle.returncode == 3


def test_truncated_server_output_retains_bounded_prefix(tmp_path: Path) -> None:
    sample, run = _execute(tmp_path, "truncate-output")
    assert sample.status == "capture-failed"
    assert sample.lifecycle is not None and sample.lifecycle.stdout_truncated
    path = f"adapters/llama-server/{CASE.id}/server/stdout.bin"
    assert run.files[path] == b"x" * (256 * 1024)


def test_inherited_server_pipe_does_not_block_collection(tmp_path: Path) -> None:
    sample, _run = _execute(tmp_path, "inherit-pipe")
    assert sample.status == "success"
    assert sample.lifecycle is not None and sample.lifecycle.capture_error is None


def test_port_refusal_is_structured(tmp_path: Path) -> None:
    binary = _write_server(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    receipt = _receipt(model)
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.bind(("127.0.0.1", 0))
    port = int(occupied.getsockname()[1])
    nonces = iter(("a" * 32, "b" * 32))
    try:
        sample = run_llama_server_case(
            case=CASE,
            inputs=_inputs(binary, receipt),
            receipt=receipt,
            run=_FakeRun(),  # type: ignore[arg-type]
            environment={},
            cwd=tmp_path,
            port=port,
            capability_timeout=3.0,
            readiness_timeout=3.0,
            request_timeout=3.0,
            shutdown_timeout=1.0,
            nonce_factory=lambda: next(nonces),
        )
    finally:
        occupied.close()
    assert sample.status == "port-unavailable"
    assert sample.lifecycle is None


def test_port_refusal_rechecks_input_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = _write_server(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    receipt = _receipt(model)
    run = _FakeRun()
    nonces = iter(("a" * 32, "b" * 32))

    def drift_before_refusal(_port: int) -> bool:
        model.write_bytes(b"drift")
        return False

    monkeypatch.setattr(llama_server_module, "_port_available", drift_before_refusal)
    with pytest.raises(LlamaServerIntegrityError):
        run_llama_server_case(
            case=CASE,
            inputs=_inputs(binary, receipt),
            receipt=receipt,
            run=run,  # type: ignore[arg-type]
            environment={},
            cwd=tmp_path,
            port=_port(),
            capability_timeout=3.0,
            readiness_timeout=3.0,
            request_timeout=3.0,
            shutdown_timeout=0.3,
            nonce_factory=lambda: next(nonces),
        )
    assert f"adapters/llama-server/{CASE.id}/sample.json" not in run.files


def test_server_spawn_failure_is_structured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = _write_server(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    receipt = _receipt(model)
    real_popen = subprocess.Popen
    calls = 0

    def fail_third_popen(*args: object, **kwargs: object) -> subprocess.Popen[bytes]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("synthetic spawn failure")
        return real_popen(*args, **kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(subprocess, "Popen", fail_third_popen)
    nonces = iter(("a" * 32, "b" * 32))
    sample = run_llama_server_case(
        case=CASE,
        inputs=_inputs(binary, receipt),
        receipt=receipt,
        run=_FakeRun(),  # type: ignore[arg-type]
        environment={},
        cwd=tmp_path,
        port=_port(),
        capability_timeout=3.0,
        readiness_timeout=3.0,
        request_timeout=3.0,
        shutdown_timeout=0.3,
        nonce_factory=lambda: next(nonces),
    )
    assert calls == 3
    assert sample.status == "spawn-failed"
    assert sample.lifecycle is None


def test_final_identity_drift_publishes_no_sample(tmp_path: Path) -> None:
    binary = _write_server(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    receipt = _receipt(model)
    run = _FakeRun()
    nonces = iter(("a" * 32, "b" * 32))
    with pytest.raises(LlamaServerIntegrityError):
        run_llama_server_case(
            case=CASE,
            inputs=_inputs(binary, receipt),
            receipt=receipt,
            run=run,  # type: ignore[arg-type]
            environment={"FAKE_MODE": "drift-model"},
            cwd=tmp_path,
            port=_port(),
            capability_timeout=3.0,
            readiness_timeout=3.0,
            request_timeout=3.0,
            shutdown_timeout=0.3,
            nonce_factory=lambda: next(nonces),
        )
    assert f"adapters/llama-server/{CASE.id}/sample.json" not in run.files


@pytest.mark.parametrize(
    ("mode", "fail_suffix"),
    [("success", "requests/0001/request.json"), ("ignore-term", "sample.json")],
)
def test_evidence_failure_preserves_exception_and_reaps_server(
    tmp_path: Path, mode: str, fail_suffix: str
) -> None:
    binary = _write_server(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    receipt = _receipt(model)
    pid_file = tmp_path / "server.pid"
    run = _FailingRun(fail_suffix)
    nonces = iter(("a" * 32, "b" * 32))
    with pytest.raises(OSError, match="synthetic evidence failure"):
        run_llama_server_case(
            case=CASE,
            inputs=_inputs(binary, receipt),
            receipt=receipt,
            run=run,  # type: ignore[arg-type]
            environment={"FAKE_MODE": mode, "PID_FILE": str(pid_file)},
            cwd=tmp_path,
            port=_port(),
            capability_timeout=3.0,
            readiness_timeout=3.0,
            request_timeout=3.0,
            shutdown_timeout=0.1,
            nonce_factory=lambda: next(nonces),
        )
    pid = int(pid_file.read_text(encoding="ascii"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert f"adapters/llama-server/{CASE.id}/sample.json" not in run.files


def test_fixture_empty_streams_are_exact() -> None:
    assert (FIXTURES / "help.stderr.txt").read_bytes() == b""
    assert (FIXTURES / "version.stdout.txt").read_bytes() == b""
    assert hashlib.sha256(b"").hexdigest() == EMPTY_SHA256


def test_server_receipt_input_mismatch_leaves_no_sample(tmp_path: Path) -> None:
    binary = _write_server(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    receipt = _receipt(model)
    inputs = _inputs(binary, receipt).model_copy(update={"model_sha256": "0" * 64})
    run = _FakeRun()
    with pytest.raises(LlamaServerIntegrityError):
        run_llama_server_case(
            case=CASE,
            inputs=inputs,
            receipt=receipt,
            run=run,  # type: ignore[arg-type]
            environment={},
            cwd=tmp_path,
            port=_port(),
            capability_timeout=3.0,
            readiness_timeout=3.0,
            request_timeout=3.0,
            shutdown_timeout=0.3,
            nonce_factory=lambda: "a" * 32,
        )
    assert f"adapters/llama-server/{CASE.id}/sample.json" not in run.files


def test_server_finalizer_lease_drift_leaves_no_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = _write_server(tmp_path)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"model")
    receipt = _receipt(model)
    original = llama_server_module._finalize

    def drifting(evidence: object, **kwargs: object) -> object:
        model.write_bytes(b"model-has-now-drifted")  # drift after teardown, before gate
        return original(evidence, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(llama_server_module, "_finalize", drifting)
    run = _FakeRun()
    nonces = iter(("a" * 32, "b" * 32))
    with pytest.raises(LlamaServerIntegrityError):
        run_llama_server_case(
            case=CASE,
            inputs=_inputs(binary, receipt),
            receipt=receipt,
            run=run,  # type: ignore[arg-type]
            environment={"FAKE_MODE": "success"},
            cwd=tmp_path,
            port=_port(),
            capability_timeout=3.0,
            readiness_timeout=3.0,
            request_timeout=3.0,
            shutdown_timeout=0.3,
            nonce_factory=lambda: next(nonces),
        )
    assert f"adapters/llama-server/{CASE.id}/sample.json" not in run.files

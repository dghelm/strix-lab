#!/usr/bin/env python3
"""Host-only executable fixture for the fixed native capsule protocol."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def scenario() -> dict[str, Any]:
    common = {
        "sample_count": 5,
        "warmup_count": 1,
    }
    return {
        "schema_version": 1,
        "comparison": {
            "policy": "paired-latency-log-bootstrap-v1",
            "protected_regression_bps": 500,
            "permitted_arm_differences": [
                "candidate-id",
                "source-candidate",
                "build-output",
            ],
        },
        "coordinates": [
            {
                **common,
                "case_id": "train-case",
                "case_set": "training",
                "coordinate_id": "train-forward",
                "input_id": "train-a",
                "input_sha256": "1" * 64,
                "mode": "forward",
                "order": 0,
            },
            {
                **common,
                "case_id": "eval-case",
                "case_set": "evaluation",
                "coordinate_id": "eval-reverse",
                "input_id": "eval-a",
                "input_sha256": "2" * 64,
                "mode": "reverse",
                "order": 1,
            },
        ],
    }


def response_binding(request: dict[str, Any], request_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": request["protocol"],
        "operation": request["operation"],
        "request_sha256": request_sha256,
        "capsule_id": request["capsule_id"],
        "candidate": request["candidate"],
        "scenario_sha256": request["scenario_sha256"],
        "manifest_sha256": request["manifest_sha256"],
        "executable_sha256": request["executable_sha256"],
        "prior_response_sha256": request["prior_response_sha256"],
        "scenario_contract_sha256": request["scenario_contract_sha256"],
        "opaque_payload": {"fixture": request["operation"]},
    }


def load_state(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text()) if path.exists() else []


def main() -> int:
    if (
        len(sys.argv) != 4
        or sys.argv[1] not in {"describe", "correctness", "benchmark"}
        or sys.argv[2] != "--request"
        or not sys.argv[3].startswith("/proc/self/fd/")
    ):
        return 90
    operation = sys.argv[1]
    descriptor_text = sys.argv[3].removeprefix("/proc/self/fd/")
    if not descriptor_text.isascii() or not descriptor_text.isdigit():
        return 90
    descriptor = int(descriptor_text)
    if sys.argv[3] != f"/proc/self/fd/{descriptor}":
        return 90
    if (fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE) != os.O_RDONLY:
        return 96
    required_seals = 0x0001 | 0x0002 | 0x0004 | 0x0008
    if fcntl.fcntl(descriptor, 1034) != required_seals:
        return 96
    mode = os.environ.get("FAKE_CAPSULE_MODE", "success")
    if mode == f"mutate-request-{operation}":
        writable = os.open(sys.argv[3], os.O_WRONLY | os.O_CLOEXEC)
        try:
            os.pwrite(writable, b"X", 0)
        except OSError:
            pass
        finally:
            os.close(writable)
    request_bytes = Path(sys.argv[3]).read_bytes()
    request = json.loads(request_bytes)
    if canonical(request) != request_bytes or request["operation"] != operation:
        return 91

    state_path = Path(os.environ["FAKE_CAPSULE_STATE"])
    state = load_state(state_path)
    expected = ("describe", "correctness", "benchmark")[len(state)]
    if operation != expected:
        return 92
    if operation == "describe":
        if request["prior_response_sha256"] is not None or request["scenario"] is not None:
            return 93
    elif request["prior_response_sha256"] != state[-1]["response_sha256"]:
        return 94

    if mode == "check-environment" and "STRIXLAB_CAPSULE_AMBIENT_SENTINEL" in os.environ:
        return 95
    if mode == f"timeout-{operation}":
        time.sleep(5)
    if mode == f"oversize-{operation}":
        sys.stdout.write("x" * (1024 * 1024 + 1))
        return 0
    if mode == f"oversize-stderr-{operation}":
        sys.stderr.write("x" * (256 * 1024 + 1))
        return 0
    if mode == f"secret-{operation}":
        sys.stdout.write(os.environ["API_TOKEN"])
        return 0
    if mode == f"interpolation-{operation}":
        sys.stdout.write("${API_TOKEN}")
        return 0
    if mode == f"interpolation-stderr-{operation}":
        sys.stderr.write("${API_TOKEN}")
    if mode == f"malformed-{operation}":
        sys.stdout.write("{malformed\n")
        return 0
    if mode == f"deep-json-{operation}":
        sys.stdout.write("[" * 2000 + "0" + "]" * 2000)
        return 0
    if mode == f"lone-surrogate-{operation}":
        sys.stdout.buffer.write(b'"\\ud800"')
        return 0

    binding = response_binding(request, digest(request_bytes))
    if operation == "describe":
        response = {**binding, "scenario": scenario()}
    elif operation == "correctness":
        coordinates = [
            {"coordinate": coordinate, "passed": True}
            for coordinate in request["scenario"]["coordinates"]
        ]
        if mode == "correctness-fail":
            coordinates[0]["passed"] = False
        if mode == "correctness-missing":
            coordinates.pop()
        elif mode == "correctness-duplicate":
            coordinates[1] = coordinates[0]
        elif mode == "correctness-reordered":
            coordinates.reverse()
        response = {**binding, "coordinates": coordinates}
    else:
        coordinates = [
            {
                "coordinate": coordinate,
                "latency_seconds": [0.01, 0.011, 0.012, 0.013, 0.014],
                "workspace_bytes": 4096 + 1024 * index,
            }
            for index, coordinate in enumerate(request["scenario"]["coordinates"])
        ]
        if mode == "benchmark-missing":
            coordinates.pop()
        elif mode == "benchmark-duplicate":
            coordinates[1] = coordinates[0]
        elif mode == "benchmark-reordered":
            coordinates.reverse()
        elif mode == "benchmark-incomplete-samples":
            coordinates[0]["latency_seconds"].pop()
        elif mode == "benchmark-nan":
            coordinates[0]["latency_seconds"][0] = float("nan")
        elif mode == "benchmark-inf":
            coordinates[0]["latency_seconds"][0] = float("inf")
        elif mode == "benchmark-nonpositive":
            coordinates[0]["latency_seconds"][0] = 0.0
        response = {**binding, "coordinates": coordinates}

    if mode == f"wrong-operation-{operation}":
        response["operation"] = "benchmark" if operation != "benchmark" else "describe"
    if mode == f"wrong-request-{operation}":
        response["request_sha256"] = "0" * 64
    if mode == f"wrong-candidate-{operation}":
        response["candidate"] = "wrong-candidate"
    if mode == f"wrong-scenario-{operation}":
        response["scenario_sha256"] = "0" * 64
    if mode == f"wrong-manifest-{operation}":
        response["manifest_sha256"] = "0" * 64
    if mode == f"wrong-executable-{operation}":
        response["executable_sha256"] = "0" * 64
    if mode == f"wrong-prior-{operation}":
        response["prior_response_sha256"] = "0" * 64
    if mode == f"wrong-contract-{operation}":
        response["scenario_contract_sha256"] = "0" * 64
    if mode == f"unknown-field-{operation}":
        response["unexpected"] = True
    if mode == "wrong-comparison-describe":
        response["scenario"]["comparison"]["protected_regression_bps"] = 501
    if mode == "coercible-describe":
        response["scenario"]["coordinates"][0]["order"] = "0"
    if mode == f"oversize-opaque-{operation}":
        response["opaque_payload"] = "x" * (256 * 1024 + 1)

    response_bytes = canonical(response)
    if mode == f"noncanonical-{operation}":
        response_bytes = (json.dumps(response, sort_keys=True) + "\n").encode()
    sys.stdout.buffer.write(response_bytes)
    sys.stdout.buffer.flush()
    if mode == f"nonzero-{operation}":
        return 7
    if mode == f"drift-{operation}":
        with Path(sys.argv[0]).open("ab") as stream:
            stream.write(b"\n# drift\n")

    state.append(
        {
            "operation": operation,
            "prior_response_sha256": request["prior_response_sha256"],
            "request_sha256": digest(request_bytes),
            "response_sha256": digest(response_bytes),
            "request_fd_readonly_and_sealed": True,
        }
    )
    state_path.write_text(json.dumps(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

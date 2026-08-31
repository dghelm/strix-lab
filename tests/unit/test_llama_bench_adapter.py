"""Tests for the bounded ca94157 ``llama-bench`` adapter (ADAPTER-001)."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from _model_fixtures import build_verified_receipt
from pydantic import ValidationError

import strixlab.adapters.llama_bench as lb
import strixlab.evidence as ev
from strixlab.bundles import export_bundle, verify_bundle
from strixlab.models import (
    ModelExecutionProjectionV1,
    ModelReceiptEvidenceV2,
    ModelReceiptV1,
    receipt_evidence_digest,
)
from strixlab.process import ProcessOutcome, ProcessResult, run_process
from strixlab.serialization import canonical_json_bytes

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "llama_bench" / "ca94157"
_HELP = (_FIXTURES / "help.stdout.txt").read_text()

_CHILD_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C"}
_RUN_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C"}
_SECRET = "supersecret-token-value"
_CLOCK = lambda: datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)  # noqa: E731

CASE_PP = lb.LlamaBenchCaseV1(
    id="pp-512",
    prompt_tokens=512,
    generated_tokens=0,
    repetitions=3,
    metric_kind="prompt-processing",
)


# --- helpers ------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_jsonl(
    model_path: str, *, n_prompt: int, n_gen: int, reps: int, avg: float = 118.5
) -> str:
    row = {
        "build_commit": "ca94157f",
        "model_filename": model_path,
        "n_prompt": n_prompt,
        "n_gen": n_gen,
        "avg_ts": avg,
        "stddev_ts": 1.25,
        "samples_ts": [round(avg + index * 0.5, 3) for index in range(reps)],
    }
    return json.dumps(row) + "\n"


def write_binary(
    directory: Path,
    *,
    name: str = "llama-bench",
    help_text: str = _HELP,
    help_rc: int = 0,
    version_rc: int = 1,
    version_stderr: str = "error: invalid parameter for argument: --version\n",
    bench_stdout: str = "",
    bench_stdout_bytes: bytes | None = None,
    bench_stderr: str = "",
    bench_rc: int = 0,
    bench_sleep: float = 0.0,
    executable: bool = True,
) -> tuple[Path, str]:
    """Write a tiny fake ``llama-bench`` and return its path and SHA-256."""

    raw = "None" if bench_stdout_bytes is None else repr(bench_stdout_bytes)
    script = (
        f"#!{sys.executable}\n"
        "import sys, time\n"
        "argv = sys.argv[1:]\n"
        f"if '--help' in argv:\n    sys.stdout.write({help_text!r})\n    sys.exit({help_rc})\n"
        f"if '--version' in argv:\n    sys.stderr.write({version_stderr!r})\n"
        f"    sys.exit({version_rc})\n"
        # Echo the effective -m operand (the /proc/self/fd path) as the tool would, so
        # '@MODEL@' in the benchmark stdout stands in for whatever the runner passed.
        "operand = argv[argv.index('-m') + 1] if '-m' in argv else ''\n"
        f"time.sleep({bench_sleep!r})\n"
        f"_raw = {raw}\n"
        "if _raw is not None:\n    sys.stdout.buffer.write(_raw)\n"
        f"else:\n    sys.stdout.write({bench_stdout!r}.replace('@MODEL@', operand))\n"
        f"sys.stderr.write({bench_stderr!r})\n"
        f"sys.exit({bench_rc})\n"
    )
    path = directory / name
    path.write_text(script)
    path.chmod(0o755 if executable else 0o644)
    return path, _sha256(path)


def make_inputs(binary: Path, binary_sha: str, receipt: ModelReceiptV1) -> lb.LlamaBenchInputsV1:
    return lb.LlamaBenchInputsV1(
        build_id="build-sha256:" + "a" * 64,
        source_commit=lb.SOURCE_ANCHOR_COMMIT,
        binary_path=str(binary),
        binary_sha256=binary_sha,
        model_id=receipt.manifest_id,
        model_path=receipt.primary.local_path,
        model_sha256=receipt.primary.sha256,
        model_receipt_sha256=receipt_evidence_digest(receipt.evidence),
        model_receipt_evidence=receipt.evidence,
    )


def receipt_for(model: Path) -> ModelReceiptV1:
    return build_verified_receipt(model.parent, model)


def begin(home: Path, *, environ: Mapping[str, str] | None = None) -> ev.RunSession:
    return ev.begin_run(
        "exp-adapter",
        b"suite: adapter\n",
        resolved={"case": "pp-512"},
        home=home,
        environ=environ or _RUN_ENV,
        clock=_CLOCK,
    )


def make_model(tmp_path: Path, content: bytes = b"gguf-model-bytes") -> Path:
    model = tmp_path / "model.gguf"
    model.write_bytes(content)
    return model


def run_case(
    run: ev.RunSession,
    binary: Path,
    sha: str,
    model: Path,
    *,
    case: lb.LlamaBenchCaseV1 = CASE_PP,
    environment: Mapping[str, str] | None = None,
    runner: lb.ProcessRunner = run_process,
    capability_timeout: float = 10.0,
    benchmark_timeout: float = 10.0,
    receipt: ModelReceiptV1 | None = None,
) -> lb.LlamaBenchSampleV1:
    receipt = receipt or receipt_for(model)
    inputs = make_inputs(binary, sha, receipt)
    return lb.run_llama_bench_case(
        case=case,
        inputs=inputs,
        receipt=receipt,
        run=run,
        environment=environment or _CHILD_ENV,
        cwd=binary.parent,
        capability_timeout=capability_timeout,
        benchmark_timeout=benchmark_timeout,
        runner=runner,
    )


def logical_paths(run: ev.RunSession) -> set[str]:
    entries = run.active / "portable" / "entries"
    if not entries.is_dir():
        return set()
    return {json.loads(path.read_text())["logical_path"] for path in entries.iterdir()}


def scripted_runner(overrides: Mapping[str, ProcessResult]) -> lb.ProcessRunner:
    def runner(argv: tuple[str, ...], **kwargs: object) -> ProcessResult:
        key = "help" if "--help" in argv else "version" if "--version" in argv else "bench"
        if key in overrides:
            return overrides[key]
        return run_process(argv, **kwargs)  # type: ignore[arg-type]

    return runner


def synthetic_result(argv: tuple[str, ...], outcome: ProcessOutcome) -> ProcessResult:
    now = 1.0
    return ProcessResult(
        outcome=outcome,
        argv=argv,
        returncode=None,
        stdout="",
        stderr="",
        started_at=now,
        ended_at=now,
        duration=0.0,
        error="synthetic",
    )


# --- case / inputs models -----------------------------------------------------


def test_case_requires_exactly_one_nonzero_metric() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        lb.LlamaBenchCaseV1(
            id="bad",
            prompt_tokens=0,
            generated_tokens=0,
            repetitions=1,
            metric_kind="prompt-processing",
        )
    with pytest.raises(ValidationError, match="exactly one"):
        lb.LlamaBenchCaseV1(
            id="bad",
            prompt_tokens=8,
            generated_tokens=8,
            repetitions=1,
            metric_kind="prompt-processing",
        )


def test_case_metric_kind_must_agree_with_nonzero_count() -> None:
    with pytest.raises(ValidationError, match="metric_kind must agree"):
        lb.LlamaBenchCaseV1(
            id="bad",
            prompt_tokens=0,
            generated_tokens=4,
            repetitions=1,
            metric_kind="prompt-processing",
        )
    tg = lb.LlamaBenchCaseV1(
        id="tg-128",
        prompt_tokens=0,
        generated_tokens=128,
        repetitions=1,
        metric_kind="text-generation",
    )
    assert tg.metric_kind == "text-generation"


def test_inputs_are_bound_to_the_pinned_source_revision() -> None:
    with pytest.raises(ValidationError):
        lb.LlamaBenchInputsV1(
            build_id="build-sha256:" + "a" * 64,
            source_commit="0" * 40,
            binary_path="/bin/llama-bench",
            binary_sha256="b" * 64,
            model_id="qwen35-4b-smoke",
            model_path="/models/model.gguf",
            model_sha256="c" * 64,
        )


def test_case_bounds_are_enforced() -> None:
    with pytest.raises(ValidationError):
        lb.LlamaBenchCaseV1(
            id="big",
            prompt_tokens=1_048_577,
            generated_tokens=0,
            repetitions=1,
            metric_kind="prompt-processing",
        )
    with pytest.raises(ValidationError):
        lb.LlamaBenchCaseV1(
            id="reps",
            prompt_tokens=1,
            generated_tokens=0,
            repetitions=33,
            metric_kind="prompt-processing",
        )
    with pytest.raises(ValidationError):
        lb.LlamaBenchCaseV1(
            id="reps0",
            prompt_tokens=1,
            generated_tokens=0,
            repetitions=0,
            metric_kind="prompt-processing",
        )


def test_inputs_label_model_digest_verified_and_reject_bad_fields(tmp_path: Path) -> None:
    receipt = receipt_for(make_model(tmp_path))
    inputs = make_inputs(Path("/bin/llama-bench"), "a" * 64, receipt)
    assert inputs.model_digest_status == "verified"
    assert inputs.model_receipt_sha256 == receipt_evidence_digest(receipt.evidence)
    with pytest.raises(ValidationError):
        make_inputs(Path("relative/llama-bench"), "a" * 64, receipt)
    with pytest.raises(ValidationError):
        make_inputs(Path("/bin/llama-bench"), "z" * 64, receipt)  # non-hex sha
    with pytest.raises(ValidationError):
        lb.LlamaBenchInputsV1(
            build_id="nope",
            source_commit=lb.SOURCE_ANCHOR_COMMIT,
            binary_path="/bin/llama-bench",
            binary_sha256="a" * 64,
            model_id="m",
            model_path="/models/x.gguf",
            model_sha256="b" * 64,
            model_receipt_sha256="a" * 64,
            model_receipt_evidence=receipt.evidence,
        )


def test_metric_kind_canonical_json_is_locked() -> None:
    tg = lb.LlamaBenchCaseV1(
        id="tg-128",
        prompt_tokens=0,
        generated_tokens=128,
        repetitions=5,
        metric_kind="text-generation",
    )
    assert b'"metric_kind": "prompt-processing"' in canonical_json_bytes(
        CASE_PP.model_dump(mode="json")
    )
    assert b'"metric_kind": "text-generation"' in canonical_json_bytes(tg.model_dump(mode="json"))


# --- command builder ----------------------------------------------------------


def test_build_benchmark_argv_is_exact_and_allowlisted() -> None:
    argv = lb.build_benchmark_argv(
        binary_path="/b/llama-bench", model_path="/m/x.gguf", case=CASE_PP
    )
    assert argv == (
        "/b/llama-bench",
        "-m",
        "/m/x.gguf",
        "-p",
        "512",
        "-n",
        "0",
        "-r",
        "3",
        "-o",
        "jsonl",
    )


# --- grammar parser -----------------------------------------------------------


def test_parse_capabilities_accepts_pinned_grammar() -> None:
    caps = lb.parse_capabilities(_HELP, binary_sha256="c" * 64)
    assert caps.output_mode == "jsonl"
    assert caps.advertised_output_modes == lb.EXPECTED_OUTPUT_MODES
    assert caps.required_flags.model == "-m/--model"
    assert caps.binary_sha256 == "c" * 64


@pytest.mark.parametrize(
    "help_text",
    [
        _HELP.replace("-m, --model", "--model-path"),  # missing required spelling
        _HELP.replace("<csv|json|jsonl|md|sql>", "<csv|md|sql>"),  # jsonl removed / weakened
        _HELP.replace(
            "-o, --output <csv|json|jsonl|md|sql>", "output stuff"
        ),  # no output flag line
        _HELP.replace("-r, --repetitions", "-r, --repeat"),  # weakened repetitions spelling
    ],
)
def test_parse_capabilities_rejects_weakened_grammar(help_text: str) -> None:
    with pytest.raises(lb.LlamaBenchGrammarError):
        lb.parse_capabilities(help_text, binary_sha256="c" * 64)


# --- JSONL parser -------------------------------------------------------------


def test_parse_jsonl_normalizes_single_metric_golden() -> None:
    model_path = "/opt/strixlab/models/qwen2.5-7b-instruct-q4_k_m.gguf"
    case = lb.LlamaBenchCaseV1(
        id="pp-512",
        prompt_tokens=512,
        generated_tokens=0,
        repetitions=5,
        metric_kind="prompt-processing",
    )
    text = (_FIXTURES / "single_metric.jsonl").read_text()
    measurement = lb.parse_jsonl_sample(text, case=case, model_path=model_path)
    assert len(measurement.samples_ts) == 5
    assert all(rate > 0 for rate in measurement.samples_ts)
    assert measurement.avg_ts > 0


def _base_row(model_path: str) -> dict[str, object]:
    return {
        "model_filename": model_path,
        "n_prompt": 512,
        "n_gen": 0,
        "avg_ts": 100.0,
        "stddev_ts": 1.0,
        "samples_ts": [100.0, 101.0, 102.0],
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda text, model: text.replace("102.0]", '102.0]\n{"extra": 1}'),  # extra row
        lambda text, model: '{"a": 1} garbage\n',  # trailing data
        lambda text, model: text.replace('"n_prompt": 512', '"n_prompt": "512"'),  # wrong type
        lambda text, model: text.replace(
            ', "samples_ts": [100.0, 101.0, 102.0]', ""
        ),  # missing field
        lambda text, model: text.replace("102.0]", "102.0, 103.0]"),  # samples != reps
        lambda text, model: text.replace('"avg_ts": 100.0', '"avg_ts": NaN'),  # non-finite
        lambda text, model: text.replace(
            '"n_prompt": 512, ', '"n_prompt": 512, "n_prompt": 512, '
        ),  # dup key
        lambda text, model: text.replace(
            '"stddev_ts": 1.0', '"stddev_ts": -1.0'
        ),  # negative stddev
        lambda text, model: text.replace(
            '"stddev_ts": 1.0', '"stddev_ts": "x"'
        ),  # stddev not a number
        lambda text, model: text.replace("100.0, 101.0", "0.0, 101.0"),  # non-positive rate
        lambda text, model: text.replace('"avg_ts": 100.0', '"avg_ts": "fast"'),  # avg not a number
        lambda text, model: text.replace("100.0, 101.0", '"x", 101.0'),  # sample not a number
        lambda text, model: text.replace(model, "/other/model.gguf"),  # model mismatch
        lambda text, model: text.replace('"n_gen": 0', '"n_gen": 7'),  # n_gen mismatch
    ],
)
def test_parse_jsonl_rejects_malformed_rows(mutate: Callable[[str, str], str]) -> None:
    model_path = "/models/x.gguf"
    text = json.dumps(_base_row(model_path)) + "\n"
    with pytest.raises(lb.LlamaBenchParseError):
        lb.parse_jsonl_sample(mutate(text, model_path), case=CASE_PP, model_path=model_path)


def test_parse_jsonl_rejects_decoder_overflow_anywhere_in_row() -> None:
    huge_integer = "9" * 5000
    with pytest.raises(lb.LlamaBenchParseError, match="invalid integer"):
        lb.parse_jsonl_sample(
            '{"model_filename":"/models/x.gguf","n_prompt":'
            + huge_integer
            + ',"n_gen":0,"avg_ts":1,"stddev_ts":0,"samples_ts":[1]}\n',
            case=CASE_PP,
            model_path="/models/x.gguf",
        )
    with pytest.raises(lb.LlamaBenchParseError, match="non-finite"):
        lb.parse_jsonl_sample(
            json.dumps(_base_row("/models/x.gguf"))[:-1] + ',"unused":1e999}\n',
            case=CASE_PP,
            model_path="/models/x.gguf",
        )


# --- fixture provenance -------------------------------------------------------


def test_fixture_digests_match_provenance() -> None:
    expected = {
        "help.stderr.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "help.stdout.txt": "e404a43db9f6bc7518e39b7c08a07a58bf00fc1e1968648863566f4b7cc5fc47",
        "version.stderr.txt": "019b6a93e02ac5e378d03caee4d5211ad17da382081737494c4d27e639b06a11",
        "version.stdout.txt": "e404a43db9f6bc7518e39b7c08a07a58bf00fc1e1968648863566f4b7cc5fc47",
        "readme.jsonl": "48c5fa392ba09fb21f37f4271e37f8c8d34d7feda06f84b602651b2a356803c5",
        "single_metric.jsonl": "50c7d832f4d2f80f76ea4c1596936c8b56c57610a3e29ab724dc749827396f0f",
    }
    for name, digest in expected.items():
        assert _sha256(_FIXTURES / name) == digest, name


def test_readme_documentation_fixture_is_not_adapter_valid() -> None:
    # The upstream example carries both a pp and a tg row, so it is two rows and
    # must be rejected by the one-metric parser.
    text = (_FIXTURES / "readme.jsonl").read_text()
    case = lb.LlamaBenchCaseV1(
        id="pp-512",
        prompt_tokens=512,
        generated_tokens=0,
        repetitions=5,
        metric_kind="prompt-processing",
    )
    with pytest.raises(lb.LlamaBenchParseError):
        lb.parse_jsonl_sample(text, case=case, model_path="models/Qwen2.5-7B-Instruct-Q4_K_M.gguf")


# --- orchestration: success ---------------------------------------------------


def test_success_writes_evidence_and_binds_case(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    stdout = valid_jsonl("@MODEL@", n_prompt=512, n_gen=0, reps=3)
    binary, sha = write_binary(tmp_path, bench_stdout=stdout)
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, model)
        paths = logical_paths(run)
        run.succeed()

    assert sample.status == "success"
    assert sample.reason == "success"
    assert sample.measurement is not None
    assert len(sample.measurement.samples_ts) == 3
    assert sample.inputs.model_digest_status == "verified"
    assert sample.invocation is not None
    assert sample.invocation.argv[-2:] == ("-o", "jsonl")
    assert sample.capabilities is not None
    base = f"adapters/llama-bench/{CASE_PP.id}"
    assert f"{base}/sample.json" in paths
    assert f"{base}/capabilities/attempt.json" in paths
    assert f"{base}/capabilities/help.process.json" in paths
    assert f"{base}/capabilities/version.process.json" in paths
    assert f"{base}/invocations/0001/process.json" in paths
    assert f"{base}/invocations/0001/stdout.txt" in paths  # exact stream published


def test_success_run_finalizes_exports_and_verifies_offline(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    stdout = valid_jsonl("@MODEL@", n_prompt=512, n_gen=0, reps=3)
    binary, sha = write_binary(tmp_path, bench_stdout=stdout)
    home = tmp_path / "home"
    with begin(home) as run:
        run_id = run.run_id
        run_case(run, binary, sha, model)
        run.succeed()

    destination = tmp_path / "bundle"
    export_bundle(run_id, destination, home=home, environ=_RUN_ENV)
    inspection = verify_bundle(destination)
    assert inspection.run_id == run_id
    assert inspection.outcome == "success"


def test_adapter_does_not_finalize_the_session(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    stdout = valid_jsonl("@MODEL@", n_prompt=512, n_gen=0, reps=3)
    binary, sha = write_binary(tmp_path, bench_stdout=stdout)
    home = tmp_path / "home"
    with begin(home) as run:
        run_case(run, binary, sha, model)
        # The session is still ACTIVE and writable — the adapter never finalized it.
        run.write_evidence("caller/extra.txt", b"still active\n")
        inspection = run.succeed()
    assert inspection.outcome == "success"


def test_runner_uses_bounded_limits_and_never_spools(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    stdout = valid_jsonl("@MODEL@", n_prompt=512, n_gen=0, reps=3)
    binary, sha = write_binary(tmp_path, bench_stdout=stdout)
    calls: list[dict[str, object]] = []

    def recording(argv: tuple[str, ...], **kwargs: object) -> ProcessResult:
        calls.append(kwargs)
        return run_process(argv, **kwargs)  # type: ignore[arg-type]

    home = tmp_path / "home"
    with begin(home) as run:
        run_case(run, binary, sha, model, runner=recording)
        run.succeed()

    assert len(calls) == 3  # two probes and one benchmark
    for kwargs in calls:
        assert kwargs["output_limit_bytes"] == lb.STREAM_LIMIT_BYTES
        assert kwargs["inherit_env"] is False
        assert (
            "stdout_spool" not in kwargs
            and "stderr_spool" not in kwargs
            and "spool_root" not in kwargs
        )


# --- orchestration: capability failures ---------------------------------------


def test_capability_failure_runs_no_benchmark_child(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    binary, sha = write_binary(tmp_path, help_text="usage: llama-bench\n(no supported grammar)\n")
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, model)
        paths = logical_paths(run)
        run.fail("capability failed")

    assert sample.status == "capability-failed"
    assert sample.reason == "capability-unsupported"
    assert sample.capabilities is None
    assert sample.invocation is None
    base = f"adapters/llama-bench/{CASE_PP.id}"
    assert f"{base}/sample.json" in paths
    # Both probes are captured before any benchmark decision.
    assert f"{base}/capabilities/help.process.json" in paths
    assert f"{base}/capabilities/version.process.json" in paths
    assert not any("invocations/" in path for path in paths)


def test_unexpected_version_success_fails_capability(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    # A --version that exits 0 violates the pinned expected unsupported outcome.
    binary, sha = write_binary(tmp_path, version_rc=0, version_stderr="")
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, model)
        run.fail("version")
    assert sample.status == "capability-failed"
    assert sample.reason == "capability-unsupported"


@pytest.mark.parametrize(
    ("version_rc", "version_stderr"),
    [
        (2, "error: invalid parameter for argument: --version\n"),
        (1, "fatal: unrelated failure\n"),
    ],
)
def test_unexpected_version_failure_shape_fails_capability(
    tmp_path: Path, version_rc: int, version_stderr: str
) -> None:
    model = make_model(tmp_path)
    binary, sha = write_binary(tmp_path, version_rc=version_rc, version_stderr=version_stderr)
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, model)
        run.fail("version")
    assert sample.status == "capability-failed"
    assert sample.reason == "capability-unsupported"


def test_probe_hard_failure_still_captures_both_probes(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    binary, sha = write_binary(tmp_path)
    overrides = {"help": synthetic_result((str(binary), "--help"), ProcessOutcome.SPAWN_FAILED)}
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, model, runner=scripted_runner(overrides))
        paths = logical_paths(run)
        run.fail("probe")
    assert sample.status == "capability-failed"
    assert sample.reason == "spawn-failed"
    base = f"adapters/llama-bench/{CASE_PP.id}"
    assert f"{base}/capabilities/version.process.json" in paths  # version still attempted


# --- orchestration: benchmark failures ----------------------------------------


def test_nonzero_benchmark_exit_is_process_failed(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    binary, sha = write_binary(tmp_path, bench_rc=2, bench_stderr="boom\n")
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, model)
        run.fail("nonzero")
    assert sample.status == "process-failed"
    assert sample.reason == "nonzero-exit"
    assert sample.measurement is None
    assert sample.invocation is not None


def test_benchmark_timeout_is_process_failed(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    binary, sha = write_binary(tmp_path, bench_sleep=3.0)
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, model, benchmark_timeout=0.3)
        run.fail("timeout")
    assert sample.status == "process-failed"
    assert sample.reason == "timed-out"


def test_spawn_and_capture_failures_are_process_failed(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    binary, sha = write_binary(tmp_path)
    for outcome, reason in (
        (ProcessOutcome.SPAWN_FAILED, "spawn-failed"),
        (ProcessOutcome.CAPTURE_FAILED, "capture-failed"),
    ):
        home = tmp_path / f"home-{reason}"
        with begin(home) as run:
            overrides = {"bench": synthetic_result(("argv",), outcome)}
            sample = run_case(run, binary, sha, model, runner=scripted_runner(overrides))
            run.fail(reason)
        assert sample.status == "process-failed"
        assert sample.reason == reason


def test_oversized_output_is_truncated_without_text_artifact(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    binary, sha = write_binary(tmp_path, bench_stdout="x\n" * (200 * 1024))
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, model)
        paths = logical_paths(run)
        run.fail("oversized")
    assert sample.status == "output-truncated"
    assert sample.reason == "output-oversized"
    assert sample.invocation is not None
    assert sample.invocation.process.stdout_truncated is True
    base = f"adapters/llama-bench/{CASE_PP.id}"
    assert f"{base}/invocations/0001/process.json" in paths
    assert f"{base}/invocations/0001/stdout.txt" not in paths  # truncated => no raw text artifact


def test_invalid_utf8_output_is_truncated_status(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    binary, sha = write_binary(tmp_path, bench_stdout_bytes=b"\xff\xfe not utf8 \x80\n")
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, model)
        paths = logical_paths(run)
        run.fail("encoding")
    assert sample.status == "output-truncated"
    assert sample.reason == "encoding-failed"
    assert sample.invocation is not None
    assert sample.invocation.process.stdout_publishable is False
    base = f"adapters/llama-bench/{CASE_PP.id}"
    assert f"{base}/invocations/0001/stdout.txt" not in paths


@pytest.mark.parametrize(
    "bench_stdout",
    [
        "not json at all\n",
        '{"model_filename": "X", "n_prompt": 512, "n_gen": 0}\n',  # missing measurement fields
    ],
)
def test_unparseable_or_mismatched_output_is_parse_failed(
    tmp_path: Path, bench_stdout: str
) -> None:
    model = make_model(tmp_path)
    binary, sha = write_binary(tmp_path, bench_stdout=bench_stdout)
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, model)
        run.fail("parse")
    assert sample.status == "parse-failed"
    assert sample.reason == "parse-failed"
    assert sample.measurement is None


def test_mismatched_case_binding_is_parse_failed(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    # Valid JSONL, but for a different token count than the requested case.
    stdout = valid_jsonl(str(model), n_prompt=256, n_gen=0, reps=3)
    binary, sha = write_binary(tmp_path, bench_stdout=stdout)
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, model)
        run.fail("mismatch")
    assert sample.status == "parse-failed"


# --- integrity ----------------------------------------------------------------


def test_binary_sha_mismatch_aborts_before_any_child(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    binary, _ = write_binary(tmp_path, bench_stdout="{}\n")
    home = tmp_path / "home"
    with begin(home) as run:
        with pytest.raises(lb.LlamaBenchIntegrityError):
            run_case(run, binary, "d" * 64, model)  # wrong asserted binary sha
        assert logical_paths(run) == set()
        run.fail("integrity")


def test_non_executable_binary_is_integrity_error(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    binary, sha = write_binary(tmp_path, executable=False)
    home = tmp_path / "home"
    with begin(home) as run:
        with pytest.raises(lb.LlamaBenchIntegrityError, match="not executable"):
            run_case(run, binary, sha, model)
        run.fail("integrity")


@pytest.mark.parametrize("changed_field", ["st_mode", "st_mtime_ns", "st_ctime_ns"])
def test_hash_binary_requires_complete_metadata_stability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_field: str
) -> None:
    binary, _ = write_binary(tmp_path, bench_stdout="{}\n")
    real_fstat = os.fstat
    calls = 0

    def changed_fstat(descriptor: int) -> object:
        nonlocal calls
        metadata = real_fstat(descriptor)
        calls += 1
        if calls == 1:
            return metadata
        values = {
            "st_dev": metadata.st_dev,
            "st_ino": metadata.st_ino,
            "st_size": metadata.st_size,
            "st_mode": metadata.st_mode,
            "st_mtime_ns": metadata.st_mtime_ns,
            "st_ctime_ns": metadata.st_ctime_ns,
        }
        values[changed_field] += 1
        return SimpleNamespace(**values)

    monkeypatch.setattr(os, "fstat", changed_fstat)
    with pytest.raises(lb.LlamaBenchIntegrityError, match="changed while hashing"):
        lb._hash_binary(binary)


def test_binary_drift_after_final_child_leaves_no_sample(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    stdout = valid_jsonl("@MODEL@", n_prompt=512, n_gen=0, reps=3)
    binary, sha = write_binary(tmp_path, bench_stdout=stdout)

    def drift_runner(argv: tuple[str, ...], **kwargs: object) -> ProcessResult:
        result = run_process(argv, **kwargs)  # type: ignore[arg-type]
        if "--help" not in argv and "--version" not in argv:
            binary.write_bytes(binary.read_bytes() + b"# mutated\n")  # drift after benchmark
        return result

    home = tmp_path / "home"
    with begin(home) as run:
        with pytest.raises(lb.LlamaBenchIntegrityError):
            run_case(run, binary, sha, model, runner=drift_runner)
        paths = logical_paths(run)
        run.fail("integrity")
    base = f"adapters/llama-bench/{CASE_PP.id}"
    assert f"{base}/invocations/0001/process.json" in paths  # evidence remains
    assert f"{base}/sample.json" not in paths  # no truthful binding can be written


def test_pre_launch_binary_drift_aborts_before_benchmark(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    binary, sha = write_binary(tmp_path, bench_stdout="{}\n")

    def drift_runner(argv: tuple[str, ...], **kwargs: object) -> ProcessResult:
        result = run_process(argv, **kwargs)  # type: ignore[arg-type]
        if "--version" in argv:
            binary.write_bytes(binary.read_bytes() + b"# mutated\n")  # drift during probe
        return result

    home = tmp_path / "home"
    with begin(home) as run:
        with pytest.raises(lb.LlamaBenchIntegrityError):
            run_case(run, binary, sha, model, runner=drift_runner)
        paths = logical_paths(run)
        run.fail("integrity")
    assert not any("invocations/" in path for path in paths)  # benchmark never launched


def test_capability_failure_still_runs_final_integrity_check(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    binary, sha = write_binary(tmp_path, help_text="unsupported grammar\n")

    def drift_runner(argv: tuple[str, ...], **kwargs: object) -> ProcessResult:
        result = run_process(argv, **kwargs)  # type: ignore[arg-type]
        if "--version" in argv:
            binary.write_bytes(binary.read_bytes() + b"# mutated\n")
        return result

    home = tmp_path / "home"
    with begin(home) as run:
        with pytest.raises(lb.LlamaBenchIntegrityError):
            run_case(run, binary, sha, model, runner=drift_runner)
        paths = logical_paths(run)
        run.fail("integrity")
    base = f"adapters/llama-bench/{CASE_PP.id}"
    assert f"{base}/capabilities/attempt.json" in paths
    assert f"{base}/sample.json" not in paths


def test_model_metadata_drift_is_detected(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    binary, sha = write_binary(tmp_path, bench_stdout="{}\n")

    def drift_runner(argv: tuple[str, ...], **kwargs: object) -> ProcessResult:
        result = run_process(argv, **kwargs)  # type: ignore[arg-type]
        if "--version" in argv:
            model.write_bytes(b"grown-model-bytes-larger")  # size/mtime drift during probe
        return result

    home = tmp_path / "home"
    with begin(home) as run:
        with pytest.raises(lb.LlamaBenchIntegrityError, match="drifted"):
            run_case(run, binary, sha, model, runner=drift_runner)
        run.fail("integrity")
    # The model digest is verified against a MODEL-001 receipt, never merely asserted.
    assert lb.LlamaBenchInputsV1.model_fields["model_digest_status"].default == "verified"


# --- secret boundary ----------------------------------------------------------


def test_sensitive_output_fails_through_evidence_secret_boundary(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    leaking = valid_jsonl("@MODEL@", n_prompt=512, n_gen=0, reps=3).replace(
        '"build_commit": "ca94157f"', f'"build_commit": "{_SECRET}"'
    )
    binary, sha = write_binary(tmp_path, bench_stdout=leaking)
    environ = {**_RUN_ENV, "API_TOKEN": _SECRET}
    home = tmp_path / "home"
    with begin(home, environ=environ) as run:
        with pytest.raises(ev.RunError):
            run_case(run, binary, sha, model)
        paths = logical_paths(run)
        run.fail("secret")
    base = f"adapters/llama-bench/{CASE_PP.id}"
    assert f"{base}/sample.json" not in paths  # never copied to portable evidence


# --- additional branch coverage -----------------------------------------------


def test_parse_jsonl_rejects_non_object_top_level() -> None:
    with pytest.raises(lb.LlamaBenchParseError, match="not a JSON object"):
        lb.parse_jsonl_sample("[1, 2, 3]\n", case=CASE_PP, model_path="/models/x.gguf")


def _projection(**overrides: object) -> lb.ProcessProjectionV1:
    fields: dict[str, object] = {
        "outcome": "exited",
        "returncode": 0,
        "duration_seconds": 0.1,
        "stdout_bytes": 0,
        "stderr_bytes": 0,
        "stdout_sha256": "0" * 64,
        "stderr_sha256": "0" * 64,
        "stdout_truncated": False,
        "stderr_truncated": False,
        "stdout_publishable": True,
        "stderr_publishable": True,
        "error_category": "none",
    }
    fields.update(overrides)
    return lb.ProcessProjectionV1(**fields)  # type: ignore[arg-type]


_CAPS = lb.LlamaBenchCapabilitiesV1(
    binary_sha256="c" * 64, advertised_output_modes=lb.EXPECTED_OUTPUT_MODES
)
_EVIDENCE = ModelReceiptEvidenceV2(
    manifest_id="qwen35-4b-smoke",
    manifest_sha256="d" * 64,
    execution=ModelExecutionProjectionV1(
        verification_status="unverified", required_sources=(), required_features=()
    ),
    primary_local_path="/models/x.gguf",
    primary_sha256="b" * 64,
    primary_size_bytes=16,
    endian="LITTLE",
    inspector_stdout_sha256="f" * 64,
    metadata_sha256="e" * 64,
    metadata_key_count=9,
    tensor_count=2,
    tensor_type_counts=(("F16", 1), ("F32", 1)),
    sidecars=(),
    compatibility="verified",
    publishable=False,
    inspector_python_sha256="1" * 64,
    inspector_script_sha256="2" * 64,
    source_preparation_id="prep-strix-llama-" + "a" * 24,
    source_candidate_id="candidate-sha256:" + "1" * 64,
    source_content_tree_id="content-tree-sha256:" + "2" * 64,
    source_base_commit=lb.SOURCE_ANCHOR_COMMIT,
)
_INPUTS = lb.LlamaBenchInputsV1(
    build_id="build-sha256:" + "a" * 64,
    source_commit=lb.SOURCE_ANCHOR_COMMIT,
    binary_path="/bin/llama-bench",
    binary_sha256="a" * 64,
    model_id="qwen35-4b-smoke",
    model_path="/models/x.gguf",
    model_sha256="b" * 64,
    model_receipt_sha256=receipt_evidence_digest(_EVIDENCE),
    model_receipt_evidence=_EVIDENCE,
)


def _attempt(**overrides: object) -> lb.LlamaBenchCapabilityAttemptV1:
    fields: dict[str, object] = {
        "status": "discovered",
        "reason": None,
        "help": _projection(),
        "version": _projection(returncode=1, error_category="nonzero-exit"),
        "capabilities": _CAPS,
    }
    fields.update(overrides)
    return lb.LlamaBenchCapabilityAttemptV1(**fields)  # type: ignore[arg-type]


def test_capability_attempt_invariants() -> None:
    with pytest.raises(ValidationError, match="discovered attempt"):
        _attempt(capabilities=None)
    with pytest.raises(ValidationError, match="failed attempt"):
        _attempt(status="failed", reason="spawn-failed")


def _measurement() -> lb.LlamaBenchMeasurementV1:
    return lb.LlamaBenchMeasurementV1(avg_ts=100.0, stddev_ts=1.0, samples_ts=(100.0, 101.0, 102.0))


def _invocation() -> lb.LlamaBenchInvocationV1:
    return lb.LlamaBenchInvocationV1(
        ordinal=1,
        argv=(
            "/bin/llama-bench",
            "-m",
            "/models/x.gguf",
            "-p",
            "512",
            "-n",
            "0",
            "-r",
            "3",
            "-o",
            "jsonl",
        ),
        process=_projection(),
    )


def _sample_kwargs(**overrides: object) -> dict[str, object]:
    attempt = _attempt()
    fields: dict[str, object] = {
        "status": "success",
        "reason": "success",
        "case": CASE_PP,
        "inputs": _INPUTS,
        "capability_attempt": attempt,
        "capabilities": attempt.capabilities,
        "invocation": _invocation(),
        "measurement": _measurement(),
        "artifacts": (),
    }
    fields.update(overrides)
    return fields


def test_sample_invariants_are_enforced() -> None:
    # A valid success sample constructs cleanly.
    lb.LlamaBenchSampleV1(**_sample_kwargs())  # type: ignore[arg-type]

    failed_attempt = _attempt(status="failed", reason="spawn-failed", capabilities=None)
    cases = [
        ("mirror the capability attempt", _sample_kwargs(capabilities=None)),
        ("success sample carries the success reason", _sample_kwargs(reason="parse-failed")),
        ("binds capabilities, invocation, and measurement", _sample_kwargs(measurement=None)),
        (
            "one rate per repetition",
            _sample_kwargs(
                measurement=lb.LlamaBenchMeasurementV1(
                    avg_ts=1.0, stddev_ts=0.0, samples_ts=(1.0, 2.0)
                )
            ),
        ),
        (
            "cannot carry the success reason",
            _sample_kwargs(status="parse-failed", measurement=None),
        ),
        (
            "only a success sample carries a measurement",
            _sample_kwargs(status="parse-failed", reason="parse-failed"),
        ),
        (
            "capability-failed sample runs no benchmark child",
            _sample_kwargs(
                status="capability-failed",
                reason="capability-unsupported",
                capability_attempt=failed_attempt,
                capabilities=None,
                measurement=None,
            ),
        ),
        (
            "benchmarked sample binds capabilities and an invocation",
            _sample_kwargs(
                status="parse-failed", reason="parse-failed", measurement=None, invocation=None
            ),
        ),
    ]
    for message, kwargs in cases:
        with pytest.raises(ValidationError, match=message):
            lb.LlamaBenchSampleV1(**kwargs)  # type: ignore[arg-type]


def test_help_nonzero_exit_is_capability_failed(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    binary, sha = write_binary(tmp_path, help_rc=3)
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, model)
        run.fail("help")
    assert sample.status == "capability-failed"
    assert sample.reason == "nonzero-exit"


def test_large_stderr_stream_is_output_truncated(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    stdout = valid_jsonl("@MODEL@", n_prompt=512, n_gen=0, reps=3)
    binary, sha = write_binary(tmp_path, bench_stdout=stdout, bench_stderr="e\n" * (200 * 1024))
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, model)
        paths = logical_paths(run)
        run.fail("oversized stderr")
    assert sample.status == "output-truncated"
    assert sample.reason == "output-oversized"
    base = f"adapters/llama-bench/{CASE_PP.id}"
    assert f"{base}/invocations/0001/stderr.txt" not in paths
    assert f"{base}/invocations/0001/stdout.txt" in paths


def test_integrity_rejects_missing_and_nonregular_paths(tmp_path: Path) -> None:
    good_binary, good_sha = write_binary(tmp_path, bench_stdout="{}\n")
    good_model = make_model(tmp_path)
    a_directory = tmp_path / "adir"
    a_directory.mkdir()

    home = tmp_path / "home"
    with begin(home) as run:
        # Missing binary path.
        with pytest.raises(lb.LlamaBenchIntegrityError, match="unavailable"):
            run_case(run, tmp_path / "missing-binary", good_sha, good_model)
        # Binary path is a directory.
        with pytest.raises(lb.LlamaBenchIntegrityError, match="not a regular file"):
            run_case(run, a_directory, good_sha, good_model)
        # A verified model that disappears before the lease opens fails closed.
        vanish_dir = tmp_path / "vanish"
        vanish_dir.mkdir()
        vanishing = make_model(vanish_dir)
        receipt = receipt_for(vanishing)
        vanishing.unlink()
        with pytest.raises(lb.LlamaBenchIntegrityError, match="model file is unavailable"):
            run_case(run, good_binary, good_sha, vanishing, receipt=receipt)
        # A verified model replaced by a directory fails closed.
        vanishing.mkdir()
        with pytest.raises(lb.LlamaBenchIntegrityError, match="model path is not a regular file"):
            run_case(run, good_binary, good_sha, vanishing, receipt=receipt)
        run.fail("integrity")


def test_receipt_input_mismatch_leaves_no_sample(tmp_path: Path) -> None:
    model = make_model(tmp_path)
    receipt = receipt_for(model)
    stdout = valid_jsonl("@MODEL@", n_prompt=512, n_gen=0, reps=3)
    binary, sha = write_binary(tmp_path, bench_stdout=stdout)
    # Inputs claim a different model SHA than the receipt substantiates.
    inputs = make_inputs(binary, sha, receipt).model_copy(update={"model_sha256": "0" * 64})
    home = tmp_path / "home"
    with begin(home) as run:
        with pytest.raises(lb.LlamaBenchIntegrityError):
            lb.run_llama_bench_case(
                case=CASE_PP,
                inputs=inputs,
                receipt=receipt,
                run=run,
                environment=_CHILD_ENV,
                cwd=binary.parent,
                capability_timeout=10.0,
                benchmark_timeout=10.0,
            )
        assert f"adapters/llama-bench/{CASE_PP.id}/sample.json" not in logical_paths(run)
        run.fail("integrity")


def test_finalizer_lease_drift_leaves_no_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = make_model(tmp_path)
    receipt = receipt_for(model)
    stdout = valid_jsonl("@MODEL@", n_prompt=512, n_gen=0, reps=3)
    binary, sha = write_binary(tmp_path, bench_stdout=stdout)
    original = lb._finalize_sample

    def drifting(evidence: object, **kwargs: object) -> object:
        model.write_bytes(b"x" * 99)  # drift after the final recheck, before the gate
        return original(evidence, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(lb, "_finalize_sample", drifting)
    home = tmp_path / "home"
    with begin(home) as run:
        with pytest.raises(lb.LlamaBenchIntegrityError):
            run_case(run, binary, sha, model, receipt=receipt)
        assert f"adapters/llama-bench/{CASE_PP.id}/sample.json" not in logical_paths(run)
        run.fail("integrity")

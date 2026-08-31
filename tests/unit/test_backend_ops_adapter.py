"""Tests for the bounded ca94157 ``test-backend-ops`` correctness adapter (ADAPTER-002)."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import strixlab.adapters.backend_ops as bo
import strixlab.adapters.llama_bench as lb
import strixlab.evidence as ev
from strixlab.bundles import export_bundle, verify_bundle
from strixlab.process import ProcessOutcome, ProcessResult, run_process

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures" / "backend_ops" / "ca94157"
_HELP = (_FIXTURES / "help.stdout.txt").read_text()
_LIST = (_FIXTURES / "list-ops.stdout.txt").read_text()
_ABS_CSV = (_FIXTURES / "abs-f32-cpu.csv").read_text()

_CHILD_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C"}
_RUN_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C"}
_SECRET = "supersecret-token-value"
_CLOCK = lambda: datetime(2026, 8, 30, 12, 0, 0, tzinfo=UTC)  # noqa: E731

CASE_ABS = bo.BackendOpsCaseV1(
    id="abs-cpu-smoke", operations=("ABS",), params_regex="type=f32", backend="CPU"
)


# --- helpers ------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_binary(
    directory: Path,
    *,
    name: str = "test-backend-ops",
    help_text: str = _HELP,
    help_rc: int = 1,
    help_stderr: str = "",
    list_text: str = _LIST,
    list_rc: int = 0,
    list_stderr: str = "",
    test_stdout: str = _ABS_CSV,
    test_stdout_bytes: bytes | None = None,
    test_stderr: str = "",
    test_rc: int = 0,
    test_sleep: float = 0.0,
    executable: bool = True,
) -> tuple[Path, str]:
    """Write a tiny fake ``test-backend-ops`` and return its path and SHA-256."""

    raw = "None" if test_stdout_bytes is None else repr(test_stdout_bytes)
    script = (
        f"#!{sys.executable}\n"
        "import sys, time\n"
        "argv = sys.argv[1:]\n"
        f"if '--help' in argv:\n"
        f"    sys.stdout.write({help_text!r})\n"
        f"    sys.stderr.write({help_stderr!r})\n"
        f"    sys.exit({help_rc})\n"
        f"if '--list-ops' in argv:\n"
        f"    sys.stdout.write({list_text!r})\n"
        f"    sys.stderr.write({list_stderr!r})\n"
        f"    sys.exit({list_rc})\n"
        f"time.sleep({test_sleep!r})\n"
        f"_raw = {raw}\n"
        "if _raw is not None:\n    sys.stdout.buffer.write(_raw)\n"
        f"else:\n    sys.stdout.write({test_stdout!r})\n"
        f"sys.stderr.write({test_stderr!r})\n"
        f"sys.exit({test_rc})\n"
    )
    path = directory / name
    path.write_text(script)
    path.chmod(0o755 if executable else 0o644)
    return path, _sha256(path)


def make_inputs(binary: Path, binary_sha: str) -> bo.BackendOpsInputsV1:
    return bo.BackendOpsInputsV1(
        build_id="build-abc-123", binary_path=str(binary), binary_sha256=binary_sha
    )


def begin(home: Path, *, environ: Mapping[str, str] | None = None) -> ev.RunSession:
    return ev.begin_run(
        "exp-adapter",
        b"suite: adapter\n",
        resolved={"case": "abs-cpu-smoke"},
        home=home,
        environ=environ or _RUN_ENV,
        clock=_CLOCK,
    )


def run_case(
    run: ev.RunSession,
    binary: Path,
    sha: str,
    *,
    case: bo.BackendOpsCaseV1 = CASE_ABS,
    environment: Mapping[str, str] | None = None,
    runner: bo.ProcessRunner = run_process,
    capability_timeout: float = 10.0,
    test_timeout: float = 10.0,
) -> bo.BackendOpsSampleV1:
    return bo.run_backend_ops_case(
        case=case,
        inputs=make_inputs(binary, sha),
        run=run,
        environment=environment or _CHILD_ENV,
        cwd=binary.parent,
        capability_timeout=capability_timeout,
        test_timeout=test_timeout,
        runner=runner,
    )


def logical_paths(run: ev.RunSession) -> set[str]:
    entries = run.active / "portable" / "entries"
    if not entries.is_dir():
        return set()
    return {json.loads(path.read_text())["logical_path"] for path in entries.iterdir()}


def artifact_paths(sample: bo.BackendOpsSampleV1) -> set[str]:
    return {artifact.path for artifact in sample.artifacts}


def scripted_runner(overrides: Mapping[str, ProcessResult]) -> bo.ProcessRunner:
    def runner(argv: tuple[str, ...], **kwargs: object) -> ProcessResult:
        key = "help" if "--help" in argv else "list" if "--list-ops" in argv else "test"
        if key in overrides:
            return overrides[key]
        return run_process(argv, **kwargs)  # type: ignore[arg-type]

    return runner


def craft(
    argv: tuple[str, ...],
    *,
    outcome: ProcessOutcome = ProcessOutcome.EXITED,
    returncode: int | None = 0,
    stdout: str = "",
    stderr: str = "",
    stdout_bytes: int | None = None,
    stderr_bytes: int | None = None,
    stdout_sha: str | None = None,
    stderr_sha: str | None = None,
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
) -> ProcessResult:
    """Build a precise :class:`ProcessResult` for stream-contract scenarios."""

    encoded_out = stdout.encode("utf-8")
    encoded_err = stderr.encode("utf-8")
    return ProcessResult(
        outcome=outcome,
        argv=argv,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        started_at=1.0,
        ended_at=1.0,
        duration=0.0,
        error=None,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        stdout_bytes=len(encoded_out) if stdout_bytes is None else stdout_bytes,
        stderr_bytes=len(encoded_err) if stderr_bytes is None else stderr_bytes,
        stdout_sha256=hashlib.sha256(encoded_out).hexdigest() if stdout_sha is None else stdout_sha,
        stderr_sha256=hashlib.sha256(encoded_err).hexdigest() if stderr_sha is None else stderr_sha,
    )


def row(
    *,
    backend: str = "CPU",
    op: str = "ABS",
    params: str = "type=f32,ne_a=[1]",
    mode: str = "test",
    supported: str = "1",
    error: str = "",
    reg: str = "",
) -> bo.BackendOpsRowV1:
    return bo.BackendOpsRowV1(
        backend_name=backend,
        op_name=op,
        op_params=params,
        test_mode=mode,
        supported=supported,  # type: ignore[arg-type]
        error_message=error,
        backend_reg_name=reg,
    )


# --- fixture provenance -------------------------------------------------------


_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_fixture_digests_match_provenance() -> None:
    expected = {
        "help.stdout.txt": "7e500f8a2ad11602bb7923e8e722b42db73b8a6799eecb26beff820286ed1f3e",
        "help.stderr.txt": _EMPTY_SHA256,
        "list-ops.stdout.txt": "bbee661695594e370097344af7943c3f262681900b0d1d751bbad2e8f4e2adc9",
        "list-ops.stderr.txt": _EMPTY_SHA256,
        "abs-f32-cpu.csv": "214421be9819dff6c439d10292b0f5b3d03bb3ea6d03ad989bf66250ecc16b7a",
        "abs-f32-cpu.stderr.txt": _EMPTY_SHA256,
    }
    for name, digest in expected.items():
        assert _sha256(_FIXTURES / name) == digest, name


# --- 1. strict model rejection ------------------------------------------------


def test_case_rejects_extra_fields_and_bad_ids() -> None:
    with pytest.raises(ValidationError):
        bo.BackendOpsCaseV1(
            id="abs", operations=("ABS",), params_regex="x", backend="CPU", extra="no"
        )
    with pytest.raises(ValidationError):
        bo.BackendOpsCaseV1(id="Bad_Id", operations=("ABS",), params_regex="x", backend="CPU")
    with pytest.raises(ValidationError, match="1 to 64"):
        bo.BackendOpsCaseV1(id="a" * 65, operations=("ABS",), params_regex="x", backend="CPU")


def test_case_rejects_bad_operation_sets() -> None:
    with pytest.raises(ValidationError):  # lowercase op name
        bo.BackendOpsCaseV1(id="c", operations=("abs",), params_regex="x", backend="CPU")
    with pytest.raises(ValidationError, match="unique"):  # duplicate
        bo.BackendOpsCaseV1(id="c", operations=("ABS", "ABS"), params_regex="x", backend="CPU")
    with pytest.raises(ValidationError):  # empty set
        bo.BackendOpsCaseV1(id="c", operations=(), params_regex="x", backend="CPU")
    with pytest.raises(ValidationError):  # too many
        bo.BackendOpsCaseV1(
            id="c",
            operations=tuple(f"OP{index}" for index in range(33)),
            params_regex="x",
            backend="CPU",
        )


def test_case_rejects_bad_params_regex_and_backend() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        bo.BackendOpsCaseV1(id="c", operations=("ABS",), params_regex="", backend="CPU")
    with pytest.raises(ValidationError, match="512"):
        bo.BackendOpsCaseV1(id="c", operations=("ABS",), params_regex="x" * 513, backend="CPU")
    with pytest.raises(ValidationError, match="control"):
        bo.BackendOpsCaseV1(id="c", operations=("ABS",), params_regex="a\tb", backend="CPU")
    with pytest.raises(ValidationError, match="control"):
        bo.BackendOpsCaseV1(id="c", operations=("ABS",), params_regex="a\x7f", backend="CPU")
    with pytest.raises(ValidationError, match="printable ASCII"):
        bo.BackendOpsCaseV1(id="c", operations=("ABS",), params_regex="x", backend="CPü")
    with pytest.raises(ValidationError, match="1 to 128"):
        bo.BackendOpsCaseV1(id="c", operations=("ABS",), params_regex="x", backend="B" * 129)


def test_case_rejects_wrong_fixed_values() -> None:
    for override in ({"mode": "perf"}, {"output": "sql"}, {"parallel_workers": 2}):
        with pytest.raises(ValidationError):
            bo.BackendOpsCaseV1(
                id="c",
                operations=("ABS",),
                params_regex="x",
                backend="CPU",
                **override,  # type: ignore[arg-type]
            )


def test_inputs_reject_bad_fields() -> None:
    with pytest.raises(ValidationError):  # wrong source commit
        bo.BackendOpsInputsV1(
            source_commit="0" * 40,
            build_id="b",
            binary_path="/bin/test-backend-ops",
            binary_sha256="a" * 64,
        )
    with pytest.raises(ValidationError):  # relative path
        bo.BackendOpsInputsV1(build_id="b", binary_path="rel/x", binary_sha256="a" * 64)
    with pytest.raises(ValidationError):  # non-hex / uppercase sha
        bo.BackendOpsInputsV1(build_id="b", binary_path="/bin/x", binary_sha256="A" * 64)
    with pytest.raises(ValidationError, match="256"):  # overlong build id
        bo.BackendOpsInputsV1(build_id="b" * 257, binary_path="/bin/x", binary_sha256="a" * 64)
    with pytest.raises(ValidationError, match="control"):  # control char in build id
        bo.BackendOpsInputsV1(build_id="b\x00", binary_path="/bin/x", binary_sha256="a" * 64)


# --- 2. pure argv construction ------------------------------------------------


def test_build_test_argv_is_exact_and_allowlisted() -> None:
    case = bo.BackendOpsCaseV1(
        id="two-op", operations=("ABS", "MUL_MAT"), params_regex="type=f32", backend="CPU"
    )
    argv = bo.build_test_argv(binary_path="/b/test-backend-ops", case=case)
    assert argv == (
        "/b/test-backend-ops",
        "test",
        "-o",
        "ABS,MUL_MAT",
        "-b",
        "CPU",
        "-p",
        "type=f32",
        "--output",
        "csv",
        "-j",
        "1",
    )


def test_build_test_argv_keeps_injection_shaped_values_as_single_items() -> None:
    case = bo.BackendOpsCaseV1(
        id="inject",
        operations=("ABS",),
        params_regex="type=f32; rm -rf / #$(whoami) `id`",
        backend="CPU:0 --output sql",
    )
    argv = bo.build_test_argv(binary_path="/b/test-backend-ops", case=case)
    assert argv[argv.index("-p") + 1] == "type=f32; rm -rf / #$(whoami) `id`"
    assert argv[argv.index("-b") + 1] == "CPU:0 --output sql"
    # The dangerous strings never split into extra argv tokens.
    assert len(argv) == 12


# --- 3. help and list-ops grammar ---------------------------------------------


def test_parse_help_accepts_pinned_grammar_and_normalizes_program_token() -> None:
    assert bo.parse_help_grammar(_HELP) is True
    # Any single program-path token after Usage: is normalized away.
    swapped = _HELP.replace("test-backend-ops", "/opt/some/other/path/test-backend-ops", 1)
    assert bo.parse_help_grammar(swapped) is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t.rstrip("\n") + " EXTRA\n",  # trailing suffix token
        lambda t: t.replace(" [--list-ops]", ""),  # dropped option (substring)
        lambda t: t.replace(
            "[-o <op,..>] [-b <backend>]", "[-b <backend>] [-o <op,..>]"
        ),  # reordered options
        lambda t: t.replace("<console|sql|csv>", "<console|sql>"),  # weakened choice
        lambda t: t.replace("Usage:", "Help:"),  # wrong leading token
        lambda t: "\n",  # empty
        lambda t: t.replace("valid modes:", "valid modes: extra"),  # extra token mid-line
    ],
)
def test_parse_help_rejects_lookalikes(mutate: Callable[[str], str]) -> None:
    assert bo.parse_help_grammar(mutate(_HELP)) is False


def test_parse_operations_accepts_pinned_list() -> None:
    operations = bo.parse_operations(_LIST)
    assert len(operations) == 128
    assert operations[0] == "ABS"
    assert "MUL_MAT" in operations


@pytest.mark.parametrize(
    "mutate",
    [
        lambda t: t.replace("  ABS\n", "  ABS\n  ABS\n", 1),  # duplicate (129 lines)
        lambda t: t.replace("  ABS\n", "", 1),  # 127 ops
        lambda t: t.replace("Total: 128 operations", "Total: 127 operations"),  # total mismatch
        lambda t: t.replace("GGML operations:", "GGML ops:"),  # wrong header
        lambda t: t.replace("  ABS\n", "  abs\n", 1),  # lowercase op
        lambda t: t.replace("  ABS\n", "    ABS\n", 1),  # wrong indentation
        lambda t: t.replace("\nTotal: 128 operations\n", "\nTotal: 128 operations"),  # no newline
        lambda t: t.replace("\n\nTotal", "\nTotal"),  # missing blank separator
        lambda t: t + "trailing\n",  # trailing content
        lambda t: t.replace("  ABS\n", "  ABS\r\n", 1),  # carriage return
    ],
)
def test_parse_operations_rejects_drift(mutate: Callable[[str], str]) -> None:
    with pytest.raises(bo.BackendOpsGrammarError):
        bo.parse_operations(mutate(_LIST))


# --- 4. CSV parser ------------------------------------------------------------


def test_parse_csv_accepts_exact_commit_fixture() -> None:
    rows = bo.parse_csv_rows(_ABS_CSV, case=CASE_ABS)
    assert len(rows) == 4
    assert all(r.backend_name == "CPU" and r.op_name == "ABS" for r in rows)
    assert all(r.supported == "1" and r.error_message == "" for r in rows)
    # Upstream row order is preserved.
    assert rows[0].op_params.endswith("v=0")
    assert rows[2].op_params.endswith("v=1")


_GOOD_HEADER = ",".join(f'"{column}"' for column in bo.CSV_HEADER)


def _csv(*rows: str) -> str:
    return _GOOD_HEADER + "\n" + "".join(line + "\n" for line in rows)


_GOOD_ROW = '"CPU","ABS","type=f32,ne_a=[1]","test","1","",""'


@pytest.mark.parametrize(
    "text",
    [
        _GOOD_HEADER.replace('"op_name","op_params"', '"op_params","op_name"') + "\n" + _GOOD_ROW,
        _GOOD_HEADER + ',"extra"\n' + _GOOD_ROW,  # extended header
        '"backend_name","op_name"\n' + _GOOD_ROW,  # short/missing header
        _csv(),  # zero result rows
        _csv('"CPU","ABS","p","test","1",""'),  # wrong column count (6)
        _csv('"CPU","ABS","p","grad","1","",""'),  # non-test mode
        _csv('"CPU","MUL","p","test","1","",""'),  # unrequested op
        _csv('"GPU","ABS","p","test","1","",""'),  # wrong backend
        _csv('"","ABS","p","test","1","",""'),  # empty backend
        _csv('"CPU","ABS","p","test","2","",""'),  # bad supported
        _csv(_GOOD_ROW, _GOOD_ROW),  # duplicate rows
        _csv('"CPU","ABS","p\x7f","test","1","",""'),  # DEL control char in field
        _GOOD_HEADER + "\r\n" + _GOOD_ROW + "\n",  # carriage return
        _GOOD_HEADER + "\n" + _GOOD_ROW + "\n\n",  # trailing blank record
        _GOOD_HEADER + "\n" + _GOOD_ROW + "\ntrailing-no-newline",  # trailing content
        _csv('"CPU","ABS","line1\nline2","test","1","",""'),  # embedded record newline
        '"CPU,"ABS"\n',  # malformed quoting
    ],
)
def test_parse_csv_rejects_structural_defects(text: str) -> None:
    with pytest.raises(bo.BackendOpsParseError):
        bo.parse_csv_rows(text, case=CASE_ABS)


def test_parse_csv_rejects_too_many_rows() -> None:
    rows = "\n".join(
        f'"CPU","ABS","type=f32,r={index}","test","1","",""'
        for index in range(bo.MAX_RESULT_ROWS + 1)
    )
    with pytest.raises(bo.BackendOpsParseError, match="maximum row count"):
        bo.parse_csv_rows(_GOOD_HEADER + "\n" + rows + "\n", case=CASE_ABS)


def test_parse_csv_translates_field_size_limit() -> None:
    oversized = '"CPU","ABS","' + "x" * (csv.field_size_limit() + 1) + '","test","1","",""'
    with pytest.raises(bo.BackendOpsParseError):
        bo.parse_csv_rows(_GOOD_HEADER + "\n" + oversized + "\n", case=CASE_ABS)
    # The process-global field-size limit is left unchanged.
    assert csv.field_size_limit() == 131072


# --- 5. pure hard-gate decisions ----------------------------------------------


def test_gate_passes_when_all_rows_supported() -> None:
    gate = bo.evaluate_gate((row(), row(params="type=f32,ne_a=[2]")), case=CASE_ABS)
    assert gate.passed is True
    assert gate.reason == "passed"
    assert gate.selected_count == 1
    assert gate.observed_rows == 2
    assert gate.observed_backends == ("CPU",)


@pytest.mark.parametrize(
    ("rows", "case", "reason"),
    [
        ((), CASE_ABS, "no-rows"),
        ((row(backend="GPU"),), CASE_ABS, "wrong-backend"),
        ((row(mode="grad"),), CASE_ABS, "wrong-mode"),
        (
            (row(op="ABS"),),
            bo.BackendOpsCaseV1(
                id="two", operations=("ABS", "ADD"), params_regex="x", backend="CPU"
            ),
            "missing-operation",
        ),
        ((row(supported="0"),), CASE_ABS, "unsupported-row"),
        ((row(error="mismatch: nan"),), CASE_ABS, "error-bearing-row"),
    ],
)
def test_gate_fails_closed(
    rows: tuple[bo.BackendOpsRowV1, ...], case: bo.BackendOpsCaseV1, reason: str
) -> None:
    gate = bo.evaluate_gate(rows, case=case)
    assert gate.passed is False
    assert gate.reason == reason


# --- 6. synthetic executable integration --------------------------------------


def test_pass_writes_evidence_and_binds_case(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path)
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha)
        paths = logical_paths(run)
        run.succeed()

    assert sample.status == "passed"
    assert sample.reason is None
    assert sample.rows is not None and len(sample.rows) == 4
    assert sample.gate is not None and sample.gate.passed
    assert sample.capabilities is not None and len(sample.capabilities.operations) == 128
    assert sample.invocation is not None
    assert sample.invocation.argv[-2:] == ("-j", "1")
    base = f"adapter/backend-ops/{CASE_ABS.id}"
    assert f"{base}/sample.json" in paths
    assert f"{base}/capabilities/help.stdout.txt" in paths
    assert f"{base}/capabilities/list-ops.stdout.txt" in paths
    assert f"{base}/capabilities/attempt.json" in paths
    assert f"{base}/invocations/0001/stdout.csv" in paths
    assert f"{base}/invocations/0001/process.json" in paths


def test_runner_uses_bounded_limits_and_never_spools(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path)
    calls: list[dict[str, object]] = []

    def recording(argv: tuple[str, ...], **kwargs: object) -> ProcessResult:
        calls.append(kwargs)
        return run_process(argv, **kwargs)  # type: ignore[arg-type]

    home = tmp_path / "home"
    with begin(home) as run:
        run_case(run, binary, sha, runner=recording)
        run.succeed()

    assert len(calls) == 3  # two probes and one test child
    for kwargs in calls:
        assert kwargs["output_limit_bytes"] == bo.STREAM_LIMIT_BYTES
        assert kwargs["inherit_env"] is False
        assert "stdout_spool" not in kwargs and "spool_root" not in kwargs


def test_capability_failure_runs_no_test_child(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path, help_text="Usage: test-backend-ops (weakened)\n")
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha)
        paths = logical_paths(run)
        run.fail("capability")
    assert sample.status == "capability-failed"
    assert sample.reason == "help-grammar"
    assert sample.capabilities is None
    assert sample.invocation is None
    base = f"adapter/backend-ops/{CASE_ABS.id}"
    assert f"{base}/capabilities/help.stdout.txt" in paths
    assert f"{base}/capabilities/list-ops.stdout.txt" in paths  # both probes attempted
    assert not any("invocations/" in path for path in paths)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"help_rc": 0}, "help-exit"),
        ({"help_stderr": "warn\n"}, "help-stream"),
        ({"help_text": "Usage: test-backend-ops nope\n"}, "help-grammar"),
        ({"list_rc": 3}, "list-exit"),
        ({"list_stderr": "warn\n"}, "list-stream"),
        ({"list_text": "GGML operations:\n  ABS\n\nTotal: 1 operations\n"}, "list-grammar"),
    ],
)
def test_capability_reasons(tmp_path: Path, kwargs: dict[str, object], reason: str) -> None:
    binary, sha = write_binary(tmp_path, **kwargs)  # type: ignore[arg-type]
    home = tmp_path / f"home-{reason}"
    with begin(home) as run:
        sample = run_case(run, binary, sha)
        run.fail("cap")
    assert sample.status == "capability-failed"
    assert sample.reason == reason


def test_requested_operation_absent_fails_capability(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path)
    case = bo.BackendOpsCaseV1(
        id="ghost", operations=("NOT_A_REAL_OP",), params_regex="type=f32", backend="CPU"
    )
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, case=case)
        run.fail("absent")
    assert sample.status == "capability-failed"
    assert sample.reason == "operation-absent"


def test_probe_spawn_failure_selects_process_status(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path)
    overrides = {
        "help": craft((str(binary), "--help"), outcome=ProcessOutcome.SPAWN_FAILED, returncode=None)
    }
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, runner=scripted_runner(overrides))
        paths = logical_paths(run)
        run.fail("spawn")
    assert sample.status == "spawn-failed"
    assert sample.reason is None
    assert sample.invocation is None
    base = f"adapter/backend-ops/{CASE_ABS.id}"
    assert f"{base}/capabilities/list-ops.process.json" in paths  # list still attempted


@pytest.mark.parametrize(
    ("help_result", "list_result", "expected"),
    [
        # capture outranks spawn regardless of probe (list has capture).
        (ProcessOutcome.SPAWN_FAILED, ProcessOutcome.CAPTURE_FAILED, "capture-failed"),
        # spawn outranks timeout regardless of probe (list has spawn).
        (ProcessOutcome.TIMED_OUT, ProcessOutcome.SPAWN_FAILED, "spawn-failed"),
        # same category ties select help before list-ops.
        (ProcessOutcome.SPAWN_FAILED, ProcessOutcome.SPAWN_FAILED, "spawn-failed"),
    ],
)
def test_mixed_probe_defect_precedence(
    tmp_path: Path,
    help_result: ProcessOutcome,
    list_result: ProcessOutcome,
    expected: str,
) -> None:
    binary, sha = write_binary(tmp_path)
    overrides = {
        "help": craft((str(binary), "--help"), outcome=help_result, returncode=None),
        "list": craft((str(binary), "--list-ops"), outcome=list_result, returncode=None),
    }
    home = tmp_path / f"home-{expected}"
    with begin(home) as run:
        sample = run_case(run, binary, sha, runner=scripted_runner(overrides))
        run.fail("mixed")
    assert sample.status == expected


def test_probe_invalid_utf8_stdout_is_encoding_failed(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path)
    # A help stdout whose recorded byte count cannot reproduce from its decoded text.
    overrides = {"help": craft((str(binary), "--help"), returncode=1, stdout="x", stdout_bytes=99)}
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, runner=scripted_runner(overrides))
        run.fail("encoding")
    assert sample.status == "encoding-failed"
    assert sample.invocation is None


def test_probe_oversized_stdout_is_oversized(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path)
    overrides = {
        "list": craft(
            (str(binary), "--list-ops"), returncode=0, stdout=_LIST, stdout_truncated=True
        )
    }
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, runner=scripted_runner(overrides))
        run.fail("oversized")
    assert sample.status == "oversized-output"


def test_test_child_timeout_is_timed_out(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path, test_sleep=3.0)
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, test_timeout=0.3)
        run.fail("timeout")
    assert sample.status == "timed-out"
    assert sample.invocation is not None  # capabilities were discovered first
    assert sample.rows is None


def test_test_child_oversized_output(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path, test_stdout=_GOOD_HEADER + "\n" + ("x\n" * (200 * 1024)))
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha)
        paths = logical_paths(run)
        run.fail("oversized")
    assert sample.status == "oversized-output"
    assert sample.invocation is not None
    assert sample.invocation.process.stdout_truncated is True
    base = f"adapter/backend-ops/{CASE_ABS.id}"
    assert f"{base}/invocations/0001/stdout.csv" not in paths  # no inexact text artifact


def test_test_child_invalid_utf8_output(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path, test_stdout_bytes=b"\xff\xfe not utf8 \x80\n")
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha)
        paths = logical_paths(run)
        run.fail("encoding")
    assert sample.status == "encoding-failed"
    assert sample.invocation is not None
    assert sample.invocation.process.stdout_publishable is False
    base = f"adapter/backend-ops/{CASE_ABS.id}"
    assert f"{base}/invocations/0001/stdout.csv" not in paths


def test_invalid_stderr_is_encoding_failed_but_retains_exact_stdout(tmp_path: Path) -> None:
    # An exact CSV stdout with an inexact (encoding) stderr is still parsed and its rows
    # retained and stdout artifact published, but the process has an encoding defect, so
    # the case cannot pass: status is encoding-failed with no gate.
    binary, sha = write_binary(tmp_path)
    argv = bo.build_test_argv(binary_path=str(binary), case=CASE_ABS)
    overrides = {
        "test": craft(argv, returncode=0, stdout=_ABS_CSV, stderr="x", stderr_bytes=42),
    }
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, runner=scripted_runner(overrides))
        paths = logical_paths(run)
        run.fail("encoding stderr")
    assert sample.status == "encoding-failed"
    assert sample.gate is None
    assert sample.rows is not None and len(sample.rows) == 4  # exact stdout still parsed
    base = f"adapter/backend-ops/{CASE_ABS.id}"
    assert f"{base}/invocations/0001/stdout.csv" in paths  # stdout still published
    assert f"{base}/invocations/0001/stderr.txt" not in paths  # inexact stderr omitted


def test_truncated_stderr_is_oversized_but_retains_exact_stdout(tmp_path: Path) -> None:
    # A truncated stderr is a process defect that outranks the gate, but exact CSV
    # stdout rows are still retained.
    binary, sha = write_binary(tmp_path)
    argv = bo.build_test_argv(binary_path=str(binary), case=CASE_ABS)
    overrides = {
        "test": craft(argv, returncode=0, stdout=_ABS_CSV, stderr="e\n", stderr_truncated=True),
    }
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, runner=scripted_runner(overrides))
        run.fail("oversized stderr")
    assert sample.status == "oversized-output"
    assert sample.gate is None
    assert sample.rows is not None and len(sample.rows) == 4


def test_stderr_defect_outranks_nonzero_child(tmp_path: Path) -> None:
    # A process defect (invalid stderr) wins over a nonzero exit, per the fixed
    # precedence: oversized/invalid-UTF-8 rank above nonzero-exit.
    binary, sha = write_binary(tmp_path)
    argv = bo.build_test_argv(binary_path=str(binary), case=CASE_ABS)
    overrides = {
        "test": craft(argv, returncode=2, stdout=_ABS_CSV, stderr="x", stderr_bytes=42),
    }
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha, runner=scripted_runner(overrides))
        run.fail("stderr over child")
    assert sample.status == "encoding-failed"  # not child-failed
    assert sample.rows is not None and len(sample.rows) == 4


def test_nonzero_child_wins_but_retains_parsed_rows(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path, test_rc=2, test_stderr="fatal\n")
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha)
        run.fail("nonzero")
    assert sample.status == "child-failed"
    assert sample.rows is not None and len(sample.rows) == 4  # rows retained
    assert sample.gate is None  # child-failed wins over the gate


def test_nonzero_child_with_unparseable_stdout(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path, test_rc=2, test_stdout="not a csv at all\n")
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha)
        run.fail("nonzero")
    assert sample.status == "child-failed"  # child-failed still wins over parse-failed
    assert sample.rows is None


def test_parse_failure_on_clean_exit(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path, test_stdout="garbage,not,the,header\n")
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha)
        run.fail("parse")
    assert sample.status == "parse-failed"
    assert sample.rows is None
    assert sample.gate is None


def test_hard_gate_failure_on_unsupported_row(tmp_path: Path) -> None:
    unsupported = (
        _GOOD_HEADER + "\n" + '"CPU","ABS","type=f32,ne_a=[1]","test","0","unsupported op",""\n'
    )
    binary, sha = write_binary(tmp_path, test_stdout=unsupported)
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha)
        run.fail("gate")
    assert sample.status == "hard-gate-failed"
    assert sample.rows is not None and len(sample.rows) == 1
    assert sample.gate is not None and sample.gate.reason == "unsupported-row"


# --- 7. binary integrity ------------------------------------------------------


def test_binary_sha_mismatch_aborts_before_any_child(tmp_path: Path) -> None:
    binary, _ = write_binary(tmp_path)
    home = tmp_path / "home"
    with begin(home) as run:
        with pytest.raises(bo.BackendOpsIntegrityError, match="does not match"):
            run_case(run, binary, "d" * 64)
        assert logical_paths(run) == set()
        run.fail("integrity")


def test_non_executable_binary_is_integrity_error(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path, executable=False)
    home = tmp_path / "home"
    with begin(home) as run:
        with pytest.raises(bo.BackendOpsIntegrityError, match="not executable"):
            run_case(run, binary, sha)
        run.fail("integrity")


def test_missing_and_nonregular_binary_paths(tmp_path: Path) -> None:
    good_binary, good_sha = write_binary(tmp_path)
    a_directory = tmp_path / "adir"
    a_directory.mkdir()
    home = tmp_path / "home"
    with begin(home) as run:
        with pytest.raises(bo.BackendOpsIntegrityError, match="unavailable"):
            run_case(run, tmp_path / "missing", good_sha)
        with pytest.raises(bo.BackendOpsIntegrityError, match="not a regular file"):
            run_case(run, a_directory, good_sha)
        run.fail("integrity")


@pytest.mark.parametrize("changed_field", ["st_mode", "st_mtime_ns", "st_ctime_ns"])
def test_hash_binary_requires_metadata_stability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_field: str
) -> None:
    binary, _ = write_binary(tmp_path)
    import strixlab.adapters._executable_identity as ident

    real_fstat = ident.os.fstat
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

    monkeypatch.setattr(ident.os, "fstat", changed_fstat)
    with pytest.raises(bo.BackendOpsIntegrityError, match="changed while hashing"):
        bo._hash_binary(binary)


def test_pre_test_child_drift_aborts_before_test(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path)

    def drift_runner(argv: tuple[str, ...], **kwargs: object) -> ProcessResult:
        result = run_process(argv, **kwargs)  # type: ignore[arg-type]
        if "--list-ops" in argv:
            binary.write_bytes(binary.read_bytes() + b"# mutated\n")  # drift during probe
        return result

    home = tmp_path / "home"
    with begin(home) as run:
        with pytest.raises(bo.BackendOpsIntegrityError):
            run_case(run, binary, sha, runner=drift_runner)
        paths = logical_paths(run)
        run.fail("integrity")
    assert not any("invocations/" in path for path in paths)


def test_post_test_child_drift_leaves_no_sample(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path)

    def drift_runner(argv: tuple[str, ...], **kwargs: object) -> ProcessResult:
        result = run_process(argv, **kwargs)  # type: ignore[arg-type]
        if "--help" not in argv and "--list-ops" not in argv:
            binary.write_bytes(binary.read_bytes() + b"# mutated\n")  # drift after test child
        return result

    home = tmp_path / "home"
    with begin(home) as run:
        with pytest.raises(bo.BackendOpsIntegrityError):
            run_case(run, binary, sha, runner=drift_runner)
        paths = logical_paths(run)
        run.fail("integrity")
    base = f"adapter/backend-ops/{CASE_ABS.id}"
    assert f"{base}/invocations/0001/process.json" in paths  # evidence remains
    assert f"{base}/sample.json" not in paths  # no truthful binding


# --- 8. evidence ordering and finalization ------------------------------------


def test_evidence_inventory_is_ordered_and_typed(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path)
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha)
        run_paths = logical_paths(run)
        base = f"adapter/backend-ops/{CASE_ABS.id}"
        # The recorded digests match the durable blobs (checked while ACTIVE).
        for artifact in sample.artifacts:
            assert (run.active / "portable" / "blobs" / artifact.sha256).is_file()
        run.succeed()

    paths = [artifact.path for artifact in sample.artifacts]
    assert paths == sorted(paths)  # deterministic ordering
    by_path = {artifact.path: artifact for artifact in sample.artifacts}
    assert by_path[f"{base}/invocations/0001/stdout.csv"].media_type == "text/csv"
    assert by_path[f"{base}/capabilities/help.stdout.txt"].media_type == "text/plain"
    assert by_path[f"{base}/capabilities/attempt.json"].media_type == "application/json"
    for artifact in sample.artifacts:
        assert artifact.role == "correctness"
        assert len(artifact.sha256) == 64
        assert artifact.size_bytes >= 0
    # sample.json is written but excluded from its own inventory.
    assert f"{base}/sample.json" not in artifact_paths(sample)
    assert f"{base}/sample.json" in run_paths


def test_empty_exact_stream_is_recorded_only_in_process_json(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path)
    home = tmp_path / "home"
    with begin(home) as run:
        sample = run_case(run, binary, sha)
        run.succeed()
    base = f"adapter/backend-ops/{CASE_ABS.id}"
    # No zero-byte text blob is written; the empty stream lives in process.json only.
    assert f"{base}/capabilities/help.stderr.txt" not in artifact_paths(sample)
    assert f"{base}/invocations/0001/stderr.txt" not in artifact_paths(sample)
    assert f"{base}/capabilities/help.process.json" in artifact_paths(sample)
    assert sample.capability_attempt.help.stderr_bytes == 0
    assert sample.capability_attempt.help.stderr_publishable is True


def test_failure_while_writing_sample_leaves_no_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary, sha = write_binary(tmp_path)
    original = ev.RunSession.write_portable

    def failing(self: ev.RunSession, logical_path: str, content: bytes, **kwargs: object) -> object:
        if logical_path.endswith("/sample.json"):
            raise ev.RunError("evidence boundary refused sample.json")
        return original(self, logical_path, content, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(ev.RunSession, "write_portable", failing)
    home = tmp_path / "home"
    with begin(home) as run:
        with pytest.raises(ev.RunError):
            run_case(run, binary, sha)
        paths = logical_paths(run)
        run.fail("boundary")
    base = f"adapter/backend-ops/{CASE_ABS.id}"
    assert f"{base}/invocations/0001/process.json" in paths  # prior evidence durable
    assert f"{base}/sample.json" not in paths


def test_adapter_does_not_finalize_the_session(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path)
    home = tmp_path / "home"
    with begin(home) as run:
        run_case(run, binary, sha)
        # Still ACTIVE and writable — the adapter never finalized the caller's run.
        run.write_evidence("caller/extra.txt", b"still active\n")
        inspection = run.succeed()
    assert inspection.outcome == "success"


def test_sensitive_output_fails_through_evidence_boundary(tmp_path: Path) -> None:
    leaking = _GOOD_HEADER + "\n" + f'"CPU","ABS","type=f32,{_SECRET}","test","1","",""\n'
    binary, sha = write_binary(tmp_path, test_stdout=leaking)
    environ = {**_RUN_ENV, "API_TOKEN": _SECRET}
    home = tmp_path / "home"
    with begin(home, environ=environ) as run:
        with pytest.raises(ev.RunError):
            run_case(run, binary, sha, environment={**_CHILD_ENV, "API_TOKEN": _SECRET})
        paths = logical_paths(run)
        run.fail("secret")
    base = f"adapter/backend-ops/{CASE_ABS.id}"
    assert f"{base}/sample.json" not in paths


# --- 9. export/verify a passing run offline -----------------------------------


def test_passing_run_finalizes_exports_and_verifies_offline(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path)
    home = tmp_path / "home"
    with begin(home) as run:
        run_id = run.run_id
        run_case(run, binary, sha)
        run.succeed()

    destination = tmp_path / "bundle"
    export_bundle(run_id, destination, home=home, environ=_RUN_ENV)
    inspection = verify_bundle(destination)
    assert inspection.run_id == run_id
    assert inspection.outcome == "success"


# --- 10. shared executable-identity primitive ---------------------------------


def test_shared_primitive_preserves_adapter_exception_translation(tmp_path: Path) -> None:
    binary, _ = write_binary(tmp_path, executable=False)
    # The same shared primitive raises each adapter's own integrity exception type,
    # rather than imposing a common exception type.
    with pytest.raises(bo.BackendOpsIntegrityError, match="operations binary is not executable"):
        bo._hash_binary(binary)
    with pytest.raises(lb.LlamaBenchIntegrityError, match="benchmark binary is not executable"):
        lb._hash_binary(binary)
    assert bo.BackendOpsIntegrityError is not lb.LlamaBenchIntegrityError


def test_shared_primitive_hashes_identically_for_both_adapters(tmp_path: Path) -> None:
    binary, sha = write_binary(tmp_path)
    bo_identity = bo._hash_binary(binary)
    lb_identity = lb._hash_binary(binary)
    # Extraction preserves the exact identity semantics for both adapters.
    assert bo_identity == lb_identity
    assert bo_identity.sha256 == sha


# --- model invariants ---------------------------------------------------------


def test_sample_reason_mirrors_capability_attempt() -> None:
    binary_inputs = bo.BackendOpsInputsV1(
        build_id="b", binary_path="/bin/test-backend-ops", binary_sha256="a" * 64
    )
    help_projection = _projection(returncode=1)
    list_projection = _projection(returncode=0)
    failed_attempt = bo.BackendOpsCapabilityAttemptV1(
        status="failed",
        reason="help-grammar",
        help=help_projection,
        list_ops=list_projection,
        capabilities=None,
    )
    kwargs = {
        "status": "capability-failed",
        "reason": "help-grammar",
        "case": CASE_ABS,
        "inputs": binary_inputs,
        "capability_attempt": failed_attempt,
        "capabilities": None,
        "invocation": None,
        "rows": None,
        "gate": None,
        "artifacts": (),
    }
    bo.BackendOpsSampleV1(**kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="mirror the capability attempt reason"):
        bo.BackendOpsSampleV1(**{**kwargs, "reason": "list-grammar"})  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="refines exactly a capability failure"):
        bo.BackendOpsSampleV1(**{**kwargs, "status": "parse-failed"})  # type: ignore[arg-type]


def _projection(**overrides: object) -> bo.ProcessProjectionV1:
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
    return bo.ProcessProjectionV1(**fields)  # type: ignore[arg-type]


def test_capability_attempt_invariants() -> None:
    caps = bo.BackendOpsCapabilitiesV1(
        binary_sha256="c" * 64, operations=tuple(f"OP{index}" for index in range(128))
    )
    with pytest.raises(ValidationError, match="discovered attempt"):
        bo.BackendOpsCapabilityAttemptV1(
            status="discovered",
            reason=None,
            help=_projection(),
            list_ops=_projection(),
            capabilities=None,
        )
    with pytest.raises(ValidationError, match="failed attempt carries no capabilities"):
        bo.BackendOpsCapabilityAttemptV1(
            status="failed",
            reason=None,
            help=_projection(),
            list_ops=_projection(),
            capabilities=caps,
        )


def test_gate_summary_rejects_inconsistent_passed() -> None:
    with pytest.raises(ValidationError, match="agree with the passed reason"):
        bo.BackendOpsGateSummaryV1(
            selected_count=1,
            observed_rows=1,
            observed_backends=("CPU",),
            passed=True,
            reason="no-rows",
        )


def test_capabilities_require_exactly_128_operations() -> None:
    with pytest.raises(ValidationError):
        bo.BackendOpsCapabilitiesV1(binary_sha256="c" * 64, operations=("ABS",))
    with pytest.raises(ValidationError, match="unique"):
        bo.BackendOpsCapabilitiesV1(
            binary_sha256="c" * 64,
            operations=("ABS", "ABS") + tuple(f"OP{index}" for index in range(126)),
        )


_CAPS = bo.BackendOpsCapabilitiesV1(
    binary_sha256="c" * 64, operations=tuple(f"OP{index}" for index in range(128))
)
_INPUTS = bo.BackendOpsInputsV1(
    build_id="b", binary_path="/bin/test-backend-ops", binary_sha256="a" * 64
)


def _discovered_attempt() -> bo.BackendOpsCapabilityAttemptV1:
    return bo.BackendOpsCapabilityAttemptV1(
        status="discovered",
        reason=None,
        help=_projection(returncode=1),
        list_ops=_projection(returncode=0),
        capabilities=_CAPS,
    )


def _failed_attempt() -> bo.BackendOpsCapabilityAttemptV1:
    return bo.BackendOpsCapabilityAttemptV1(
        status="failed",
        reason=None,
        help=_projection(returncode=1),
        list_ops=_projection(returncode=0),
        capabilities=None,
    )


def _invocation() -> bo.BackendOpsInvocationV1:
    argv = bo.build_test_argv(binary_path="/bin/test-backend-ops", case=CASE_ABS)
    return bo.BackendOpsInvocationV1(ordinal=1, argv=argv, process=_projection())


def _passing_sample_kwargs(**overrides: object) -> dict[str, object]:
    gate = bo.evaluate_gate((row(),), case=CASE_ABS)
    fields: dict[str, object] = {
        "status": "passed",
        "reason": None,
        "case": CASE_ABS,
        "inputs": _INPUTS,
        "capability_attempt": _discovered_attempt(),
        "capabilities": _CAPS,
        "invocation": _invocation(),
        "rows": (row(),),
        "gate": gate,
        "artifacts": (),
    }
    fields.update(overrides)
    return fields


def test_sample_invariants_are_enforced() -> None:
    bo.BackendOpsSampleV1(**_passing_sample_kwargs())  # type: ignore[arg-type]
    failing_gate = bo.evaluate_gate((), case=CASE_ABS)
    cases = [
        ("mirror the capability attempt", _passing_sample_kwargs(capabilities=None)),
        (
            "an invocation exists exactly",
            _passing_sample_kwargs(status="capture-failed", invocation=None, rows=None, gate=None),
        ),
        (
            "test-child status requires discovered capabilities",
            _passing_sample_kwargs(
                capability_attempt=_failed_attempt(),
                capabilities=None,
                invocation=None,
            ),
        ),
        (
            "post-parse status requires a test process with no defect",
            _passing_sample_kwargs(
                invocation=bo.BackendOpsInvocationV1(
                    ordinal=1,
                    argv=bo.build_test_argv(binary_path="/bin/test-backend-ops", case=CASE_ABS),
                    process=_projection(error_category="encoding-failed"),
                )
            ),
        ),
        ("binds rows and a gate summary", _passing_sample_kwargs(gate=None)),
        (
            "gate.passed must agree with the passed status",
            _passing_sample_kwargs(status="hard-gate-failed"),
        ),
        (
            "only a gated sample carries a gate summary",
            _passing_sample_kwargs(status="child-failed"),
        ),
        (
            "a parse-failed sample retains no rows",
            _passing_sample_kwargs(status="parse-failed", gate=None),
        ),
    ]
    for message, kwargs in cases:
        with pytest.raises(ValidationError, match=message):
            bo.BackendOpsSampleV1(**kwargs)  # type: ignore[arg-type]
    # A hard-gate-failed sample constructs cleanly with a failing gate.
    bo.BackendOpsSampleV1(
        **_passing_sample_kwargs(status="hard-gate-failed", rows=(), gate=failing_gate)  # type: ignore[arg-type]
    )


def test_parse_operations_rejects_same_length_defects() -> None:
    # 128 op lines but a duplicated operation (ACC replaced by a second ABS).
    with pytest.raises(bo.BackendOpsGrammarError, match="duplicate operation"):
        bo.parse_operations(_LIST.replace("  ACC\n", "  ABS\n", 1))
    # 129 op lines with no blank separator keeps the total count at 131 lines.
    no_blank = (
        "GGML operations:\n"
        + "".join(f"  OP{index}\n" for index in range(129))
        + "Total: 128 operations\n"
    )
    with pytest.raises(bo.BackendOpsGrammarError, match="blank separator"):
        bo.parse_operations(no_blank)


def test_parse_csv_rejects_empty_stdout() -> None:
    with pytest.raises(bo.BackendOpsParseError, match="empty"):
        bo.parse_csv_rows("", case=CASE_ABS)


@pytest.mark.parametrize(
    "fields",
    [
        {"mode": "perf"},  # wrong fixed mode
        {"output": "sql"},  # wrong fixed output
        {"parallel_workers": 4},  # wrong fixed worker count
        {"operations": ("abs",)},  # lowercase operation
        {"operations": ()},  # empty operation set
        {"backend": ""},  # empty backend
        {"params_regex": ""},  # empty regex
        {"params_regex": "a\x00b"},  # control character in regex
    ],
)
def test_build_test_argv_rejects_non_roundtrip_case(fields: dict[str, object]) -> None:
    base = {
        "id": "x",
        "operations": ("ABS",),
        "params_regex": "type=f32",
        "backend": "CPU",
    }
    broken = bo.BackendOpsCaseV1.model_construct(**{**base, **fields})
    with pytest.raises(ValueError, match="round-trip"):
        bo.build_test_argv(binary_path="/b/test-backend-ops", case=broken)

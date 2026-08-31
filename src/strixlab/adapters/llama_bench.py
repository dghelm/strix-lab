"""Bounded ``llama-bench`` adapter for one typed single-metric benchmark case.

This adapter executes a verified ``llama-bench`` binary for exactly one benchmark
case, preserves bounded raw process evidence through the run-evidence boundary,
and returns one versioned :class:`LlamaBenchSampleV1` whether the child succeeds,
exits nonzero, times out, cannot spawn, truncates output, or produces unparseable
output. The supported grammar is pinned to the checked-in
``tools/llama-bench/README.md`` of
``halo-box/strix-llama.cpp@ca94157f70a2776e8da6b6849b50b45a083d0478`` and its one
reachable JSONL profile. No suite, model registry, comparison, correctness gate,
or run command lives here; later layers call this adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self, cast

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
    "LlamaBenchCapabilitiesV1",
    "LlamaBenchCaseV1",
    "LlamaBenchGrammarError",
    "LlamaBenchInputsV1",
    "LlamaBenchIntegrityError",
    "LlamaBenchParseError",
    "LlamaBenchSampleV1",
    "build_benchmark_argv",
    "parse_capabilities",
    "parse_jsonl_sample",
    "run_llama_bench_case",
]

# --- Pinned profile constants -------------------------------------------------

PROFILE: Literal["ca94157-v1"] = "ca94157-v1"
SOURCE_ANCHOR_COMMIT = "ca94157f70a2776e8da6b6849b50b45a083d0478"
OUTPUT_MODE: Literal["jsonl"] = "jsonl"
EXPECTED_OUTPUT_MODES: tuple[str, ...] = ("csv", "json", "jsonl", "md", "sql")
VERSION_UNSUPPORTED_MARKERS: tuple[str, ...] = (
    "invalid parameter for argument",
    "--version",
)
EVIDENCE_ROLE = "samples"
EVIDENCE_ROOT = "adapters/llama-bench"
STREAM_LIMIT_BYTES = 256 * 1024
_MAX_TOKENS = 1_048_576
_MAX_REPETITIONS = 32

MetricKind = Literal["prompt-processing", "text-generation"]
SampleStatus = Literal[
    "success", "capability-failed", "process-failed", "output-truncated", "parse-failed"
]
Reason = Literal[
    "success",
    "spawn-failed",
    "timed-out",
    "capture-failed",
    "output-oversized",
    "encoding-failed",
    "nonzero-exit",
    "parse-failed",
    "capability-unsupported",
]
ErrorCategory = Literal[
    "none",
    "capture-failed",
    "spawn-failed",
    "timed-out",
    "output-oversized",
    "encoding-failed",
    "nonzero-exit",
]
ProcessOutcomeLiteral = Literal["exited", "timed_out", "spawn_failed", "capture_failed"]

ProcessRunner = Callable[..., ProcessResult]


# --- Adapter exceptions -------------------------------------------------------


class LlamaBenchIntegrityError(RuntimeError):
    """The verified binary or model binding drifted and can no longer be attested.

    This invalidates the whole operation: prior process evidence may remain, but no
    ``sample.json`` is written because no truthful binding can be claimed.
    """


class LlamaBenchGrammarError(ValueError):
    """Captured ``--help`` output does not match the pinned ca94157 grammar."""


class LlamaBenchParseError(ValueError):
    """A benchmark's JSONL output could not be normalized into the case binding."""


# --- Typed v1 models ----------------------------------------------------------


class _Model(BaseModel):
    """Strict, frozen base for every v1 adapter model."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
BuildId = Annotated[str, Field(pattern=r"^build-sha256:[0-9a-f]{64}$")]
TokenCount = Annotated[StrictInt, Field(ge=0, le=_MAX_TOKENS)]
Repetitions = Annotated[StrictInt, Field(ge=1, le=_MAX_REPETITIONS)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveRate = Annotated[float, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0)]


class LlamaBenchCaseV1(_Model):
    """One typed single-metric case; v1 represents exactly one ``pp`` or ``tg`` metric."""

    schema_version: Literal[1] = 1
    id: DashId
    prompt_tokens: TokenCount
    generated_tokens: TokenCount
    repetitions: Repetitions
    metric_kind: MetricKind

    @model_validator(mode="after")
    def _one_metric(self) -> Self:
        nonzero = (self.prompt_tokens > 0, self.generated_tokens > 0)
        if sum(nonzero) != 1:
            raise ValueError("exactly one of prompt_tokens/generated_tokens must be nonzero")
        expected: MetricKind = "prompt-processing" if self.prompt_tokens > 0 else "text-generation"
        if self.metric_kind != expected:
            raise ValueError("metric_kind must agree with the one nonzero token count")
        return self


class LlamaBenchInputsV1(_Model):
    """Binary and verified-model provenance bound to one benchmark case.

    The model digest is ``verified`` against a MODEL-001 :class:`ModelReceiptV1`: the
    embedded portable :class:`ModelReceiptEvidenceV1` projection independently
    substantiates that verification (its canonical digest equals
    ``model_receipt_sha256``) after the local registry disappears. ``model_path`` retains
    the public path; the runner drives every child through the receipt-bound descriptor.
    """

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


class LlamaBenchRequiredFlagsV1(_Model):
    """The pinned required flag spellings advertised by the ca94157 grammar."""

    model: Literal["-m/--model"] = "-m/--model"
    n_prompt: Literal["-p/--n-prompt"] = "-p/--n-prompt"
    n_gen: Literal["-n/--n-gen"] = "-n/--n-gen"
    repetitions: Literal["-r/--repetitions"] = "-r/--repetitions"
    output: Literal["-o/--output"] = "-o/--output"


class LlamaBenchCapabilitiesV1(_Model):
    """The one reachable, proven ca94157-v1 capability profile (JSONL)."""

    schema_version: Literal[1] = 1
    profile: Literal["ca94157-v1"] = PROFILE
    binary_sha256: Sha256Hex
    required_flags: LlamaBenchRequiredFlagsV1 = LlamaBenchRequiredFlagsV1()
    advertised_output_modes: tuple[str, ...]
    output_mode: Literal["jsonl"] = OUTPUT_MODE


class ProcessProjectionV1(_Model):
    """Neutral projection of one child :class:`ProcessResult`.

    ``error_category`` classifies the child by the stream the adapter consumes
    (``stdout``); ``stderr`` truncation/exactness is recorded separately and never
    changes the category. No raw exception prose is ever carried.
    """

    schema_version: Literal[1] = 1
    outcome: ProcessOutcomeLiteral
    returncode: int | None
    duration_seconds: NonNegativeFloat
    stdout_bytes: NonNegativeInt
    stderr_bytes: NonNegativeInt
    stdout_sha256: Sha256Hex
    stderr_sha256: Sha256Hex
    stdout_truncated: bool
    stderr_truncated: bool
    stdout_publishable: bool
    stderr_publishable: bool
    error_category: ErrorCategory


class LlamaBenchInvocationV1(_Model):
    """Exact benchmark argv, ordinal, output mode, and process projection."""

    schema_version: Literal[1] = 1
    ordinal: Annotated[StrictInt, Field(ge=1)]
    argv: tuple[StrictStr, ...]
    output_mode: Literal["jsonl"] = OUTPUT_MODE
    process: ProcessProjectionV1


class LlamaBenchCapabilityAttemptV1(_Model):
    """Capability status plus complete help/version process projections."""

    schema_version: Literal[1] = 1
    status: Literal["discovered", "failed"]
    reason: Reason | None
    help: ProcessProjectionV1
    version: ProcessProjectionV1
    capabilities: LlamaBenchCapabilitiesV1 | None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.status == "discovered":
            if self.capabilities is None or self.reason is not None:
                raise ValueError("a discovered attempt carries capabilities and no reason")
        elif self.capabilities is not None or self.reason is None:
            raise ValueError("a failed attempt carries a reason and no capabilities")
        return self


class LlamaBenchMeasurementV1(_Model):
    """Normalized execution evidence extracted from a valid single-metric row."""

    schema_version: Literal[1] = 1
    avg_ts: PositiveRate
    stddev_ts: NonNegativeFloat
    samples_ts: tuple[PositiveRate, ...]


class LlamaBenchSampleV1(_Model):
    """The terminal, versioned adapter sample for one benchmark case."""

    schema_version: Literal[1] = 1
    profile: Literal["ca94157-v1"] = PROFILE
    status: SampleStatus
    reason: Reason
    case: LlamaBenchCaseV1
    inputs: LlamaBenchInputsV1
    capability_attempt: LlamaBenchCapabilityAttemptV1
    capabilities: LlamaBenchCapabilitiesV1 | None
    invocation: LlamaBenchInvocationV1 | None
    measurement: LlamaBenchMeasurementV1 | None
    artifacts: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.capabilities != self.capability_attempt.capabilities:
            raise ValueError("sample capabilities must mirror the capability attempt")
        if self.status == "success":
            if self.reason != "success":
                raise ValueError("a success sample carries the success reason")
            if self.capabilities is None or self.invocation is None or self.measurement is None:
                raise ValueError("a success sample binds capabilities, invocation, and measurement")
            if len(self.measurement.samples_ts) != self.case.repetitions:
                raise ValueError("a success sample records one rate per repetition")
        else:
            if self.reason == "success":
                raise ValueError("a non-success sample cannot carry the success reason")
            if self.measurement is not None:
                raise ValueError("only a success sample carries a measurement")
        if self.status == "capability-failed":
            if self.invocation is not None or self.capabilities is not None:
                raise ValueError("a capability-failed sample runs no benchmark child")
        elif self.invocation is None or self.capabilities is None:
            raise ValueError("a benchmarked sample binds capabilities and an invocation")
        return self


# --- Pure command builder and parsers ----------------------------------------


def build_benchmark_argv(
    *, binary_path: str, model_path: str, case: LlamaBenchCaseV1
) -> tuple[str, ...]:
    """Return the only allowlisted argv for one JSONL benchmark invocation."""

    return (
        binary_path,
        "-m",
        model_path,
        "-p",
        str(case.prompt_tokens),
        "-n",
        str(case.generated_tokens),
        "-r",
        str(case.repetitions),
        "-o",
        OUTPUT_MODE,
    )


_FLAG_PATTERNS: tuple[tuple[str, str], ...] = (
    ("model", r"(?m)^\s*-m,\s*--model\b"),
    ("n_prompt", r"(?m)^\s*-p,\s*--n-prompt\b"),
    ("n_gen", r"(?m)^\s*-n,\s*--n-gen\b"),
    ("repetitions", r"(?m)^\s*-r,\s*--repetitions\b"),
)
_OUTPUT_PATTERN = re.compile(r"(?m)^\s*-o,\s*--output\s+<([a-z|]+)>")


def parse_capabilities(help_stdout: str, *, binary_sha256: str) -> LlamaBenchCapabilitiesV1:
    """Parse the pinned ca94157 grammar, requiring exact flag spellings and JSONL.

    The grammar is matched by exact token spellings, never fuzzy substrings. Any
    missing required flag, or an output-mode set that is not exactly the advertised
    ``csv|json|jsonl|md|sql``, raises :class:`LlamaBenchGrammarError`.
    """

    for name, pattern in _FLAG_PATTERNS:
        if re.search(pattern, help_stdout) is None:
            raise LlamaBenchGrammarError(
                f"required llama-bench {name} flag is missing from help output"
            )
    match = _OUTPUT_PATTERN.search(help_stdout)
    if match is None:
        raise LlamaBenchGrammarError("output-format flag is missing from help output")
    modes = tuple(match.group(1).split("|"))
    if modes != EXPECTED_OUTPUT_MODES:
        raise LlamaBenchGrammarError("advertised output modes do not match the pinned grammar")
    return LlamaBenchCapabilitiesV1(
        binary_sha256=binary_sha256,
        advertised_output_modes=modes,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    seen: dict[str, object] = {}
    for key, value in pairs:
        if key in seen:
            raise LlamaBenchParseError("JSONL row contains a duplicate key")
        seen[key] = value
    return seen


def _reject_nonfinite(_value: str) -> float:
    raise LlamaBenchParseError("JSONL row contains a non-finite number")


def _parse_json_int(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise LlamaBenchParseError("JSONL row contains an invalid integer") from exc


def _parse_json_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise LlamaBenchParseError("JSONL row contains an invalid number") from exc
    if not math.isfinite(parsed):
        raise LlamaBenchParseError("JSONL row contains a non-finite number")
    return parsed


def _require_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if type(value) is not int:
        raise LlamaBenchParseError(f"JSONL field {key!r} is missing or not an integer")
    return value


def _require_rate(value: object, key: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LlamaBenchParseError(f"JSONL field {key!r} is not a number")
    rate = float(value)
    if not math.isfinite(rate) or rate <= 0:
        raise LlamaBenchParseError(f"JSONL field {key!r} must be finite and positive")
    return rate


def parse_jsonl_sample(
    stdout: str, *, case: LlamaBenchCaseV1, model_path: str
) -> LlamaBenchMeasurementV1:
    """Normalize exactly one single-metric JSONL object bound to ``case``.

    Rejects duplicate keys, non-finite values, trailing data, wrong types, missing
    fields, extra result rows, a mismatched model path or token counts, and a sample
    count that is not ``case.repetitions``. Raw output is preserved as evidence
    separately; only the v1 execution fields are normalized here.
    """

    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise LlamaBenchParseError("expected exactly one nonblank JSONL object line")
    try:
        row = json.loads(
            lines[0],
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
            parse_int=_parse_json_int,
            parse_float=_parse_json_float,
        )
    except (json.JSONDecodeError, RecursionError) as exc:
        raise LlamaBenchParseError("JSONL row is not a single JSON object") from exc
    if not isinstance(row, dict):
        raise LlamaBenchParseError("JSONL row is not a JSON object")

    model_filename = row.get("model_filename")
    if model_filename != model_path:
        raise LlamaBenchParseError("JSONL model_filename does not match the invoked model path")
    if _require_int(row, "n_prompt") != case.prompt_tokens:
        raise LlamaBenchParseError("JSONL n_prompt does not match the case")
    if _require_int(row, "n_gen") != case.generated_tokens:
        raise LlamaBenchParseError("JSONL n_gen does not match the case")

    samples = row.get("samples_ts")
    if not isinstance(samples, list) or len(samples) != case.repetitions:
        raise LlamaBenchParseError("JSONL samples_ts count does not match repetitions")
    rates = tuple(_require_rate(value, "samples_ts") for value in samples)
    avg_ts = _require_rate(row.get("avg_ts"), "avg_ts")
    stddev = row.get("stddev_ts")
    if not isinstance(stddev, (int, float)) or isinstance(stddev, bool):
        raise LlamaBenchParseError("JSONL stddev_ts must be finite and nonnegative")
    stddev_value = float(stddev)
    if not math.isfinite(stddev_value) or stddev_value < 0:
        raise LlamaBenchParseError("JSONL stddev_ts must be finite and nonnegative")
    return LlamaBenchMeasurementV1(avg_ts=avg_ts, stddev_ts=stddev_value, samples_ts=rates)


# --- Binary / model integrity -------------------------------------------------


_BinaryIdentity = ExecutableIdentity
_BINARY_SUBJECT = "benchmark binary"


def _hash_binary(path: Path) -> _BinaryIdentity:
    """Stream-hash a non-symlink regular executable with pre/post metadata stability."""

    return hash_executable(path, error=LlamaBenchIntegrityError, subject=_BINARY_SUBJECT)


def _require_stable_binary(path: Path, expected: _BinaryIdentity) -> _BinaryIdentity:
    return require_stable_executable(
        path, expected, error=LlamaBenchIntegrityError, subject=_BINARY_SUBJECT
    )


# --- Process projection and stream exactness ---------------------------------

_OUTCOME_NAMES: dict[ProcessOutcome, ProcessOutcomeLiteral] = {
    ProcessOutcome.EXITED: "exited",
    ProcessOutcome.TIMED_OUT: "timed_out",
    ProcessOutcome.SPAWN_FAILED: "spawn_failed",
    ProcessOutcome.CAPTURE_FAILED: "capture_failed",
}


@dataclass(frozen=True, slots=True)
class _ProjectedProcess:
    projection: ProcessProjectionV1
    stdout_text: str | None
    stderr_text: str | None


def _exact_stream(text: str, byte_count: int, sha256: str, truncated: bool) -> str | None:
    """Return the publishable text of an exact stream, or ``None`` when inexact.

    A stream is exact and publishable only when it is not truncated and re-encoding
    the returned text as UTF-8 reproduces both the recorded byte count and SHA-256.
    """

    if truncated:
        return None
    encoded = text.encode("utf-8")
    if len(encoded) == byte_count and hashlib.sha256(encoded).hexdigest() == sha256:
        return text
    return None


def _project(result: ProcessResult) -> _ProjectedProcess:
    stdout_text = _exact_stream(
        result.stdout, result.stdout_bytes, result.stdout_sha256, result.stdout_truncated
    )
    stderr_text = _exact_stream(
        result.stderr, result.stderr_bytes, result.stderr_sha256, result.stderr_truncated
    )
    stdout_publishable = stdout_text is not None
    category = _error_category(
        result,
        stdout_publishable=stdout_publishable,
        stderr_publishable=stderr_text is not None,
    )
    return _ProjectedProcess(
        projection=ProcessProjectionV1(
            outcome=_OUTCOME_NAMES[result.outcome],
            returncode=result.returncode,
            duration_seconds=max(0.0, result.duration),
            stdout_bytes=result.stdout_bytes,
            stderr_bytes=result.stderr_bytes,
            stdout_sha256=result.stdout_sha256,
            stderr_sha256=result.stderr_sha256,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
            stdout_publishable=stdout_publishable,
            stderr_publishable=stderr_text is not None,
            error_category=category,
        ),
        stdout_text=stdout_text,
        stderr_text=stderr_text,
    )


def _error_category(
    result: ProcessResult, *, stdout_publishable: bool, stderr_publishable: bool
) -> ErrorCategory:
    if result.outcome is ProcessOutcome.CAPTURE_FAILED:
        return "capture-failed"
    if result.outcome is ProcessOutcome.SPAWN_FAILED:
        return "spawn-failed"
    if result.outcome is ProcessOutcome.TIMED_OUT:
        return "timed-out"
    if result.stdout_truncated or result.stderr_truncated:
        return "output-oversized"
    if not stdout_publishable or not stderr_publishable:
        return "encoding-failed"
    if result.returncode != 0:
        return "nonzero-exit"
    return "none"


# --- Evidence writer ----------------------------------------------------------


@dataclass(slots=True)
class _CaseEvidence:
    """Writes deterministic portable evidence under one case namespace."""

    run: RunSession
    base: str
    written: list[str]

    def json(self, relative: str, value: object) -> None:
        path = f"{self.base}/{relative}"
        self.run.write_portable(
            path, canonical_json_bytes(value), media_type="application/json", role=EVIDENCE_ROLE
        )
        self.written.append(path)

    def _text(self, relative: str, content: str) -> None:
        path = f"{self.base}/{relative}"
        self.run.write_portable(
            path, content.encode("utf-8"), media_type="text/plain", role=EVIDENCE_ROLE
        )
        self.written.append(path)

    def streams(self, prefix: str, process: _ProjectedProcess, *, sep: str) -> None:
        """Publish exact stdout/stderr text (when any) and the process projection.

        ``sep`` joins ``prefix`` to each member name: ``"."`` gives the flat
        ``capabilities/help.stdout.txt`` layout, ``"/"`` gives the per-ordinal
        ``invocations/0001/stdout.txt`` layout. Truncated or non-UTF-8 streams have no
        text artifact; ``process.json`` still preserves their complete byte counts,
        digests, truncation, and neutral category. Empty exact streams are recorded
        deterministically.
        """

        if process.stdout_text is not None:
            self._text(f"{prefix}{sep}stdout.txt", process.stdout_text)
        if process.stderr_text is not None:
            self._text(f"{prefix}{sep}stderr.txt", process.stderr_text)
        self.json(f"{prefix}{sep}process.json", process.projection.model_dump(mode="json"))


def _recheck(binary_path: Path, binary: _BinaryIdentity, lease: ModelLease) -> None:
    # A lease.verify() drift raises ModelError, translated to the adapter integrity error
    # by run_llama_bench_case's outer boundary.
    _require_stable_binary(binary_path, binary)
    lease.verify()


def _run_child(
    *,
    runner: ProcessRunner,
    argv: tuple[str, ...],
    cwd: Path,
    timeout: float,
    environment: Mapping[str, str],
    evidence: _CaseEvidence,
    prefix: str,
    sep: str,
    descriptor: int,
) -> _ProjectedProcess:
    result = runner(
        argv,
        cwd=cwd,
        timeout=timeout,
        inherit_env=False,
        base_env=environment,
        output_limit_bytes=STREAM_LIMIT_BYTES,
        pass_fds=(descriptor,),
    )
    projected = _project(result)
    evidence.streams(prefix, projected, sep=sep)
    return projected


# --- Orchestration ------------------------------------------------------------

# Highest-to-lowest precedence of hard child failures shared by both phases.
_HARD_ORDER: tuple[ErrorCategory, ...] = (
    "capture-failed",
    "spawn-failed",
    "timed-out",
    "output-oversized",
    "encoding-failed",
)


def _hard_reason(*projections: ProcessProjectionV1) -> Reason | None:
    for category in _HARD_ORDER:
        if any(projection.error_category == category for projection in projections):
            return cast(Reason, category)
    return None


def _evaluate_capability(
    help_projection: ProcessProjectionV1,
    help_stdout: str,
    version_projection: ProcessProjectionV1,
    version_stderr: str,
    *,
    binary_sha256: str,
) -> tuple[Reason | None, LlamaBenchCapabilitiesV1 | None]:
    """Decide whether both probes prove the pinned capability profile.

    A benchmark child runs only when the required help probe exits successfully, is
    valid UTF-8, and matches the supported grammar, and the advisory version attempt
    matches the pinned expected unsupported (nonzero, bounded) outcome.
    """

    hard = _hard_reason(help_projection, version_projection)
    if hard is not None:
        return hard, None
    if help_projection.error_category == "nonzero-exit":
        return "nonzero-exit", None
    try:
        capabilities = parse_capabilities(help_stdout, binary_sha256=binary_sha256)
    except LlamaBenchGrammarError:
        return "capability-unsupported", None
    # The pinned ca94157 source has no --version branch: its expected outcome is a
    # bounded nonzero exit. Any other shape (including an unexpected success) fails
    # closed rather than being treated as a capability.
    if (
        version_projection.error_category != "nonzero-exit"
        or version_projection.returncode != 1
        or any(marker not in version_stderr for marker in VERSION_UNSUPPORTED_MARKERS)
    ):
        return "capability-unsupported", None
    return None, capabilities


def _benchmark_outcome(projection: ProcessProjectionV1) -> tuple[SampleStatus, Reason] | None:
    """Map a benchmark child's category to a non-success status, or ``None`` to parse."""

    category = projection.error_category
    if category == "none":
        return None
    status: SampleStatus = (
        "output-truncated"
        if category in ("output-oversized", "encoding-failed")
        else "process-failed"
    )
    return status, cast(Reason, category)


def run_llama_bench_case(
    *,
    case: LlamaBenchCaseV1,
    inputs: LlamaBenchInputsV1,
    receipt: ModelReceiptV1,
    run: RunSession,
    environment: Mapping[str, str],
    cwd: Path,
    capability_timeout: float,
    benchmark_timeout: float,
    runner: ProcessRunner = run_process,
) -> LlamaBenchSampleV1:
    """Execute one benchmark case and return its terminal versioned sample.

    The verified ``receipt`` is bound to ``inputs`` and held across every child through
    :func:`lease_verified_model`, so each ``llama-bench`` child opens the receipt-bound
    inode via ``/proc/self/fd/<fd>`` even if the public pathname is swapped mid-run. Both
    probes and (on capability success) one benchmark child run against a ``environment``
    snapshot with ``inherit_env=False`` and an explicit ``cwd``. Every child outcome
    yields a structured sample. The caller owns ``run``: this adapter never finalizes it.
    Evidence-boundary refusals and integrity drift propagate as exceptions rather than
    being captured into a sample.
    """

    binary_path = Path(inputs.binary_path)
    base = f"{EVIDENCE_ROOT}/{case.id}"
    evidence = _CaseEvidence(run=run, base=base, written=[])

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
        raise LlamaBenchIntegrityError(str(exc)) from exc

    identity = _hash_binary(binary_path)
    if identity.sha256 != inputs.binary_sha256:
        raise LlamaBenchIntegrityError("benchmark binary SHA-256 does not match the input binding")

    try:
        with lease_verified_model(receipt) as lease:
            return _run_leased_case(
                case=case,
                inputs=inputs,
                lease=lease,
                binary_path=binary_path,
                identity=identity,
                evidence=evidence,
                environment=environment,
                cwd=cwd,
                capability_timeout=capability_timeout,
                benchmark_timeout=benchmark_timeout,
                runner=runner,
            )
    except ModelError as exc:
        raise LlamaBenchIntegrityError(str(exc)) from exc


def _run_leased_case(
    *,
    case: LlamaBenchCaseV1,
    inputs: LlamaBenchInputsV1,
    lease: ModelLease,
    binary_path: Path,
    identity: _BinaryIdentity,
    evidence: _CaseEvidence,
    environment: Mapping[str, str],
    cwd: Path,
    capability_timeout: float,
    benchmark_timeout: float,
    runner: ProcessRunner,
) -> LlamaBenchSampleV1:
    model_operand = lease.descriptor_path
    descriptor = lease.descriptor

    # Probe 1: required --help. The initial hash and lease snapshot are its integrity check.
    help_process = _run_child(
        runner=runner,
        argv=(str(binary_path), "--help"),
        cwd=cwd,
        timeout=capability_timeout,
        environment=environment,
        evidence=evidence,
        prefix="capabilities/help",
        sep=".",
        descriptor=descriptor,
    )

    # Probe 2: advisory --version. Always attempted so its evidence is complete,
    # unless an intervening integrity failure aborts before the child.
    _recheck(binary_path, identity, lease)
    version_process = _run_child(
        runner=runner,
        argv=(str(binary_path), "--version"),
        cwd=cwd,
        timeout=capability_timeout,
        environment=environment,
        evidence=evidence,
        prefix="capabilities/version",
        sep=".",
        descriptor=descriptor,
    )

    reason, capabilities = _evaluate_capability(
        help_process.projection,
        help_process.stdout_text or "",
        version_process.projection,
        version_process.stderr_text or "",
        binary_sha256=inputs.binary_sha256,
    )

    attempt = LlamaBenchCapabilityAttemptV1(
        status="failed" if capabilities is None else "discovered",
        reason=reason,
        help=help_process.projection,
        version=version_process.projection,
        capabilities=capabilities,
    )
    evidence.json("capabilities/attempt.json", attempt.model_dump(mode="json"))

    if capabilities is None:
        assert reason is not None
        _recheck(binary_path, identity, lease)
        return _finalize_sample(
            evidence,
            lease=lease,
            status="capability-failed",
            reason=reason,
            case=case,
            inputs=inputs,
            attempt=attempt,
            invocation=None,
            measurement=None,
        )

    # Benchmark child. The model operand is the receipt-bound descriptor, not the path.
    argv = build_benchmark_argv(binary_path=str(binary_path), model_path=model_operand, case=case)
    _recheck(binary_path, identity, lease)
    benchmark_process = _run_child(
        runner=runner,
        argv=argv,
        cwd=cwd,
        timeout=benchmark_timeout,
        environment=environment,
        evidence=evidence,
        prefix="invocations/0001",
        sep="/",
        descriptor=descriptor,
    )

    # Re-hash the binary and re-verify the lease after the final child. Drift leaves
    # no sample.json because the claimed binding can no longer be attested.
    _recheck(binary_path, identity, lease)

    invocation = LlamaBenchInvocationV1(ordinal=1, argv=argv, process=benchmark_process.projection)

    failure = _benchmark_outcome(benchmark_process.projection)
    if failure is not None:
        status, reason = failure
        return _finalize_sample(
            evidence,
            lease=lease,
            status=status,
            reason=reason,
            case=case,
            inputs=inputs,
            attempt=attempt,
            invocation=invocation,
            measurement=None,
        )

    try:
        measurement = parse_jsonl_sample(
            benchmark_process.stdout_text or "", case=case, model_path=model_operand
        )
    except LlamaBenchParseError:
        return _finalize_sample(
            evidence,
            lease=lease,
            status="parse-failed",
            reason="parse-failed",
            case=case,
            inputs=inputs,
            attempt=attempt,
            invocation=invocation,
            measurement=None,
        )

    return _finalize_sample(
        evidence,
        lease=lease,
        status="success",
        reason="success",
        case=case,
        inputs=inputs,
        attempt=attempt,
        invocation=invocation,
        measurement=measurement,
    )


def _finalize_sample(
    evidence: _CaseEvidence,
    *,
    lease: ModelLease,
    status: SampleStatus,
    reason: Reason,
    case: LlamaBenchCaseV1,
    inputs: LlamaBenchInputsV1,
    attempt: LlamaBenchCapabilityAttemptV1,
    invocation: LlamaBenchInvocationV1 | None,
    measurement: LlamaBenchMeasurementV1 | None,
) -> LlamaBenchSampleV1:
    """Build the sample and publish ``sample.json`` last, gated by a final lease check.

    ``lease.verify()`` runs immediately before the terminal write, so finalizer-time model
    drift raises the integrity error (via the runner's outer boundary) before any
    ``sample.json`` exists.
    """

    sample = LlamaBenchSampleV1(
        status=status,
        reason=reason,
        case=case,
        inputs=inputs,
        capability_attempt=attempt,
        capabilities=attempt.capabilities,
        invocation=invocation,
        measurement=measurement,
        artifacts=tuple(sorted(evidence.written)),
    )
    lease.verify()
    evidence.json("sample.json", sample.model_dump(mode="json"))
    return sample

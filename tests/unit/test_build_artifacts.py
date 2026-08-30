from __future__ import annotations

import json
from pathlib import Path

import pytest

import strixlab.build_artifacts as artifacts_module
from strixlab.build_artifacts import (
    BuildArtifactError,
    _file_api_targets,
    _normalize_relative,
    _parse_ldd,
    _read_regular_with_stat,
    prepare_file_api_query,
    verify_artifact_capture,
)
from strixlab.process import ProcessOutcome, ProcessResult


def _elf(e_type: int = 3) -> bytes:
    header = bytearray(64)
    header[0:4] = b"\x7fELF"
    header[4] = 2  # EI_CLASS: ELFCLASS64
    header[5] = 1  # EI_DATA: little-endian
    header[6] = 1  # EI_VERSION
    header[16:18] = e_type.to_bytes(2, "little")  # e_type
    return bytes(header)


def _proc(
    *,
    stdout: str = "",
    stderr: str = "",
    truncated: bool = False,
    outcome: ProcessOutcome = ProcessOutcome.EXITED,
    returncode: int | None = 0,
    stderr_truncated: bool = False,
    capture_error: str | None = None,
) -> ProcessResult:
    return ProcessResult(
        outcome=outcome,
        argv=("tool", "arg"),
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        started_at=0.0,
        ended_at=0.0,
        duration=0.0,
        error=None,
        stdout_truncated=truncated,
        stderr_truncated=stderr_truncated,
        capture_error=capture_error,
    )


_ELF = _elf(3)  # ET_DYN


def _write_reply(root: Path, *, targets: tuple[str, ...] = ("llama-bench",)) -> Path:
    reply = root / ".cmake" / "api" / "v1" / "reply"
    reply.mkdir(parents=True)
    for target in targets:
        (reply / f"target-{target}.json").write_text(
            json.dumps(
                {
                    "artifacts": [{"path": f"bin/{target}"}],
                    "id": f"{target}::@x",
                    "name": target,
                    "type": "EXECUTABLE",
                }
            )
        )
    (reply / "codemodel-v2.json").write_text(
        json.dumps(
            {
                "configurations": [
                    {
                        "name": "Release",
                        "targets": [
                            {"id": f"{t}::@x", "jsonFile": f"target-{t}.json", "name": t}
                            for t in targets
                        ],
                    }
                ],
                "kind": "codemodel",
                "version": {"major": 2, "minor": 0},
            }
        )
    )
    (reply / "index-test.json").write_text(
        json.dumps(
            {
                "reply": {
                    "client-strixlab": {
                        "query.json": {
                            "responses": [
                                {
                                    "jsonFile": "codemodel-v2.json",
                                    "kind": "codemodel",
                                    "version": {"major": 2, "minor": 0},
                                }
                            ]
                        }
                    }
                }
            }
        )
    )
    return reply


def test_normalize_relative_rejects_unsafe_paths() -> None:
    for value in ("/abs", "..", "a/../b", "", "./x", "a/"):
        with pytest.raises(BuildArtifactError):
            _normalize_relative(value)
    assert _normalize_relative("bin/x") == "bin/x"


def test_prepare_file_api_query_is_exclusive(tmp_path: Path) -> None:
    prepare_file_api_query(tmp_path)
    query = tmp_path / ".cmake" / "api" / "v1" / "query" / "client-strixlab" / "query.json"
    assert b"codemodel" in query.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_file_api_query(tmp_path)


def test_file_api_target_selection_and_errors(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    _write_reply(root)
    replies, targets = _file_api_targets(root, "Release", ("llama-bench",))
    assert targets[0].name == "llama-bench"
    assert targets[0].artifacts == ("bin/llama-bench",)
    assert any(reply.name == "codemodel-v2.json" for reply in replies)

    with pytest.raises(BuildArtifactError, match="absent or ambiguous"):
        _file_api_targets(root, "Release", ("missing",))


def test_file_api_requires_single_index(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    reply = _write_reply(root)
    (reply / "index-second.json").write_text("{}")
    with pytest.raises(BuildArtifactError, match="exactly one index"):
        _file_api_targets(root, "Release", ("llama-bench",))


def test_file_api_rejects_invalid_json(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    reply = _write_reply(root)
    (reply / "codemodel-v2.json").write_text("not json")
    with pytest.raises(BuildArtifactError, match="invalid JSON"):
        _file_api_targets(root, "Release", ("llama-bench",))


def test_file_api_rejects_symlinked_v1_ancestor(tmp_path: Path) -> None:
    # A symlinked .cmake/api/v1 ancestor must fail closed: reply reads are anchored
    # at build_root and traverse the whole ancestry descriptor-anchored no-follow.
    root = tmp_path / "build"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _write_reply(outside)  # real reply tree under outside/.cmake/api/v1/reply
    api = root / ".cmake" / "api"
    api.mkdir(parents=True)
    (api / "v1").symlink_to(outside / ".cmake" / "api" / "v1", target_is_directory=True)
    with pytest.raises(BuildArtifactError, match="unsafe directory"):
        _file_api_targets(root, "Release", ("llama-bench",))


def test_file_api_rejects_symlinked_dotcmake_ancestor(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _write_reply(outside)
    (root / ".cmake").symlink_to(outside / ".cmake", target_is_directory=True)
    with pytest.raises(BuildArtifactError, match="unsafe directory"):
        _file_api_targets(root, "Release", ("llama-bench",))


def test_file_api_rejects_unsupported_target_type(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    reply = _write_reply(root)
    (reply / "target-llama-bench.json").write_text(
        json.dumps(
            {
                "artifacts": [{"path": "bin/llama-bench"}],
                "id": "llama-bench::@x",
                "name": "llama-bench",
                "type": "X",
            }
        )
    )
    with pytest.raises(BuildArtifactError, match="no supported artifacts"):
        _file_api_targets(root, "Release", ("llama-bench",))


def test_parse_ldd_variants(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "lib" / "libin.so"
    inside.parent.mkdir()
    inside.write_bytes(_ELF)
    external = tmp_path / "external.so"
    external.write_bytes(_ELF)
    output = (
        f"\tlibin.so => {inside} (0x00007f00)\n"
        f"\tlibc.so.6 => {external} (0x00007f11)\n"
        "\tlinux-vdso.so.1 (0x00007fff)\n"
        f"\t{external} (0x00007f22)\n"
    )
    deps = _parse_ldd(_proc(stdout=output), root)
    by_name = {dep.name: dep for dep in deps}
    assert by_name["libin.so"].in_build_root is True
    assert by_name["libc.so.6"].in_build_root is False

    with pytest.raises(BuildArtifactError, match="unresolved dependency"):
        _parse_ldd(_proc(stdout="\tmissing.so => not found\n"), root)
    with pytest.raises(BuildArtifactError, match="not absolute"):
        _parse_ldd(_proc(stdout="\trel.so => relative/path.so (0x0)\n"), root)
    with pytest.raises(BuildArtifactError, match="malformed"):
        _parse_ldd(_proc(stdout="\tgarbage-with-no-arrow\n"), root)


def test_verify_artifact_capture_rejects_empty(tmp_path: Path) -> None:
    from strixlab.build_artifacts import BuildArtifactsV1

    empty = BuildArtifactsV1(
        artifact_set_id="artifact-set-sha256:" + "0" * 64,
        targets=(),
        artifacts=(),
        inspections=(),
        capture_tools=(),
        cmake_cache_sha256="0" * 64,
    )
    with pytest.raises(BuildArtifactError, match="empty"):
        verify_artifact_capture(tmp_path, empty, selections=(), toolchain_mode="host")


def test_read_regular_rejects_symlinked_parent(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_bytes(_ELF)
    (root / "dir").symlink_to(outside, target_is_directory=True)
    # Ancestor redirection: the symlinked directory component is opened nofollow and
    # rejected before the final open, so the read never escapes the trusted root.
    with pytest.raises(BuildArtifactError, match="unsafe directory"):
        _read_regular_with_stat(root, root / "dir" / "file", artifacts_module._FILE_LIMIT)


def test_read_regular_rejects_symlinked_final_file(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "secret.bin"
    outside.write_bytes(_ELF)
    (root / "file").symlink_to(outside)  # final component swapped for a symlink
    with pytest.raises(BuildArtifactError, match="cannot be opened"):
        _read_regular_with_stat(root, root / "file", artifacts_module._FILE_LIMIT)


def test_artifact_rejects_non_elf(tmp_path: Path) -> None:
    from strixlab.build_artifacts import _artifact

    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "x").write_bytes(b"not-elf")
    with pytest.raises(BuildArtifactError, match="unsupported format"):
        _artifact(tmp_path, "bin/x", targets=("x",))


def test_verify_artifact_capture_detects_change(tmp_path: Path) -> None:
    from strixlab.build_artifacts import ArtifactV1, BuildArtifactsV1

    (tmp_path / "bin").mkdir()
    payload = _ELF + b"body"
    (tmp_path / "bin" / "x").write_bytes(payload)
    good = ArtifactV1(
        path="bin/x",
        kind="elf",
        mode=0o644,
        size_bytes=len(payload),
        sha256=__import__("hashlib").sha256(payload).hexdigest(),
        targets=("x",),
    )
    evidence = BuildArtifactsV1(
        artifact_set_id="artifact-set-sha256:" + "0" * 64,
        targets=(),
        artifacts=(good.model_copy(update={"sha256": "9" * 64}),),
        inspections=(),
        capture_tools=(),
        cmake_cache_sha256="0" * 64,
    )
    with pytest.raises(BuildArtifactError, match="build artifact changed"):
        verify_artifact_capture(tmp_path, evidence, selections=(), toolchain_mode="host")


def test_verify_rejects_dangling_compile_commands_symlink(tmp_path: Path) -> None:
    import hashlib

    from strixlab.build_artifacts import BuildArtifactsV1, _artifact
    from strixlab.build_identity import ArtifactIdentity, IdentityEntry, artifact_set_id

    selections = tuple(
        IdentityEntry(name, value)
        for name, value in {
            "generator": "Ninja",
            "c_compiler": "/usr/bin/cc",
            "cxx_compiler": "/usr/bin/c++",
            "linker": "/usr/bin/ld",
            "archiver": "/usr/bin/ar",
            "toolchain_files": "",
            "sysroot": "",
        }.items()
    )
    (tmp_path / "bin").mkdir()
    payload = _ELF + b"body"
    (tmp_path / "bin" / "x").write_bytes(payload)
    (tmp_path / "CMakeCache.txt").write_bytes(b"cache")
    good = _artifact(tmp_path, "bin/x", targets=("x",))
    identity = ArtifactIdentity(good.path, good.mode, good.size_bytes, good.sha256)
    evidence = BuildArtifactsV1(
        artifact_set_id=artifact_set_id((identity,), selections, toolchain_mode="host"),
        targets=(),
        artifacts=(good,),
        inspections=(),
        capture_tools=(),
        cmake_cache_sha256=hashlib.sha256(b"cache").hexdigest(),
    )
    # A dangling symlink is divergent filesystem state, not an absent observation:
    # lexists sees it, and the expected-absent contract fails closed.
    (tmp_path / "compile_commands.json").symlink_to(tmp_path / "missing-target.json")
    with pytest.raises(BuildArtifactError, match="compile_commands.json appeared"):
        verify_artifact_capture(tmp_path, evidence, selections=selections, toolchain_mode="host")


def test_file_api_ambiguous_codemodel(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    reply = _write_reply(root)
    v2 = {"jsonFile": "codemodel-v2.json", "kind": "codemodel", "version": {"major": 2, "minor": 0}}
    (reply / "index-test.json").write_text(
        json.dumps({"reply": {"client-strixlab": {"query.json": {"responses": [v2, v2]}}}})
    )
    with pytest.raises(BuildArtifactError, match="codemodel response is ambiguous"):
        _file_api_targets(root, "Release", ("llama-bench",))


def _write_index_responses(reply: Path, responses: list[dict[str, object]]) -> None:
    (reply / "index-test.json").write_text(
        json.dumps({"reply": {"client-strixlab": {"query.json": {"responses": responses}}}})
    )


def test_file_api_rejects_wrong_major_response_version(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    reply = _write_reply(root)
    _write_index_responses(
        reply,
        [{"jsonFile": "codemodel-v2.json", "kind": "codemodel", "version": {"major": 3}}],
    )
    with pytest.raises(BuildArtifactError, match="codemodel response is ambiguous"):
        _file_api_targets(root, "Release", ("llama-bench",))


def test_file_api_rejects_malformed_response_version(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    reply = _write_reply(root)
    _write_index_responses(
        reply,
        [{"jsonFile": "codemodel-v2.json", "kind": "codemodel", "version": "2"}],
    )
    with pytest.raises(BuildArtifactError, match="codemodel response is ambiguous"):
        _file_api_targets(root, "Release", ("llama-bench",))


def test_file_api_rejects_wrong_major_codemodel_object(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    reply = _write_reply(root)
    # The index descriptor is a valid v2, but the loaded object claims major 3.
    (reply / "codemodel-v2.json").write_text(
        json.dumps(
            {
                "configurations": [{"name": "Release", "targets": []}],
                "kind": "codemodel",
                "version": {"major": 3, "minor": 0},
            }
        )
    )
    with pytest.raises(BuildArtifactError, match="unexpected kind or version"):
        _file_api_targets(root, "Release", ("llama-bench",))


def test_capture_tools_requires_path(tmp_path: Path) -> None:
    from strixlab.build_artifacts import _capture_tools
    from strixlab.process import run_process

    with pytest.raises(BuildArtifactError, match="PATH is missing"):
        _capture_tools({}, tmp_path, 5.0, tmp_path, run_process)
    with pytest.raises(BuildArtifactError, match="inspection tool is unavailable"):
        _capture_tools({"PATH": str(tmp_path)}, tmp_path, 5.0, tmp_path, run_process)


def test_read_regular_escape_and_limits(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(BuildArtifactError, match="escapes its trusted root"):
        _read_regular_with_stat(root, tmp_path / "elsewhere", artifacts_module._FILE_LIMIT)
    target = root / "file"
    target.write_bytes(_ELF)
    with pytest.raises(BuildArtifactError, match="exceeds the capture limit"):
        _read_regular_with_stat(root, target, 1)
    with pytest.raises(BuildArtifactError, match="resolves to its trusted root"):
        _read_regular_with_stat(root, root, artifacts_module._FILE_LIMIT)


def test_file_api_incomplete_target_reply(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    reply = _write_reply(root)
    (reply / "codemodel-v2.json").write_text(
        json.dumps(
            {
                "configurations": [
                    {
                        "name": "Release",
                        "targets": [{"name": "llama-bench", "id": "x"}],
                    }
                ],
                "kind": "codemodel",
                "version": {"major": 2, "minor": 0},
            }
        )
    )
    with pytest.raises(BuildArtifactError, match="reply is incomplete"):
        _file_api_targets(root, "Release", ("llama-bench",))


def test_file_api_ambiguous_configuration(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    reply = _write_reply(root)
    (reply / "codemodel-v2.json").write_text(
        json.dumps(
            {
                "configurations": [
                    {"name": "Release", "targets": []},
                    {"name": "Release", "targets": []},
                ],
                "kind": "codemodel",
                "version": {"major": 2, "minor": 0},
            }
        )
    )
    with pytest.raises(BuildArtifactError, match="configuration is ambiguous"):
        _file_api_targets(root, "Release", ("llama-bench",))


def test_file_api_target_without_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "build"
    root.mkdir()
    reply = _write_reply(root)
    (reply / "target-llama-bench.json").write_text(
        json.dumps(
            {
                "artifacts": [],
                "id": "llama-bench::@x",
                "name": "llama-bench",
                "type": "EXECUTABLE",
            }
        )
    )
    with pytest.raises(BuildArtifactError, match="no artifacts"):
        _file_api_targets(root, "Release", ("llama-bench",))


def test_parse_ldd_unresolvable_absolute_path(tmp_path: Path) -> None:
    with pytest.raises(BuildArtifactError, match="cannot be resolved"):
        _parse_ldd(_proc(stdout=f"\tlib.so => {tmp_path / 'nope.so'} (0x0)\n"), tmp_path)


def test_read_regular_missing_file(tmp_path: Path) -> None:
    with pytest.raises(BuildArtifactError, match="cannot be opened"):
        _read_regular_with_stat(tmp_path, tmp_path / "missing", artifacts_module._FILE_LIMIT)


def test_relative_file_rejects_escape(tmp_path: Path) -> None:
    from strixlab.build_artifacts import _relative_file

    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(BuildArtifactError, match="escapes the build root"):
        _relative_file(root, tmp_path / "outside" / "x")


def test_read_regular_rejects_foreign_owner_flag(tmp_path: Path) -> None:
    # require_owner=False path is exercised for capture tools; ensure it reads.
    target = tmp_path / "f"
    target.write_bytes(_ELF)
    content, meta = _read_regular_with_stat(
        tmp_path, target, artifacts_module._FILE_LIMIT, require_owner=False
    )
    assert content == _ELF


def test_target_format_mismatch_is_rejected(tmp_path: Path) -> None:
    from strixlab.build_artifacts import TargetArtifactsV1, _initial_artifacts

    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "lib.a").write_bytes(_elf(3))  # an ELF where an archive is required
    target = TargetArtifactsV1(
        name="lib", target_id="lib::@x", target_type="STATIC_LIBRARY", artifacts=("bin/lib.a",)
    )
    with pytest.raises(BuildArtifactError, match="expects an archive"):
        _initial_artifacts(tmp_path, (target,))

    (tmp_path / "bin" / "obj.o").write_bytes(_elf(1))  # ET_REL where an executable is required
    exe = TargetArtifactsV1(
        name="app", target_id="app::@x", target_type="EXECUTABLE", artifacts=("bin/obj.o",)
    )
    with pytest.raises(BuildArtifactError, match="incompatible"):
        _initial_artifacts(tmp_path, (exe,))


def test_artifact_streaming_ignores_file_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from strixlab.build_artifacts import _artifact

    (tmp_path / "bin").mkdir()
    payload = _elf(3) + b"\x00" * 4096
    (tmp_path / "bin" / "x").write_bytes(payload)
    monkeypatch.setattr(artifacts_module, "_FILE_LIMIT", 100)  # smaller than the artifact
    monkeypatch.setattr(artifacts_module, "_STREAM_CHUNK", 512)  # force chunked streaming

    art = _artifact(tmp_path, "bin/x", targets=("x",))
    assert art.sha256 == hashlib.sha256(payload).hexdigest()
    assert art.size_bytes == len(payload)
    assert art.elf_type == "ET_DYN"
    # The buffered reader would reject the same file at that limit; streaming does not.
    with pytest.raises(BuildArtifactError, match="capture limit"):
        _read_regular_with_stat(tmp_path, tmp_path / "bin" / "x", artifacts_module._FILE_LIMIT)


def _inspection_runner(readelf_out: str, ldd_out: str, calls: list[str]):
    def runner(argv: tuple[str, ...], **kwargs: object) -> ProcessResult:
        args = tuple(str(value) for value in argv)
        calls.append(args[0].rsplit("/", 1)[-1])
        if "-d" in args:
            return _proc(stdout=readelf_out)
        return _proc(stdout=ldd_out)

    return runner


def test_inspect_object_elf_skips_ldd(tmp_path: Path) -> None:
    from strixlab.build_artifacts import _inspect_dynamic

    calls: list[str] = []
    runner = _inspection_runner("", "", calls)
    inspection, results = _inspect_dynamic(
        tmp_path / "bin" / "x",
        "bin/x",
        "ET_REL",
        tmp_path,
        Path("/usr/bin/readelf"),
        Path("/usr/bin/ldd"),
        {"PATH": "/usr/bin"},
        5.0,
        tmp_path,
        runner,
    )
    assert inspection.dynamic is False
    assert inspection.ldd_sha256 is None
    assert inspection.dependencies == ()
    assert "ldd" not in calls  # ldd never invoked on a relocatable object
    assert [name for name, _ in results] == [
        "readelf-" + __import__("hashlib").sha256(b"bin/x").hexdigest()[:16]
    ]


def test_inspect_dynamic_parses_needed_and_runs_ldd(tmp_path: Path) -> None:
    from strixlab.build_artifacts import _inspect_dynamic

    lib = tmp_path / "libc.so.6"
    lib.write_bytes(_elf(3))
    readelf_out = (
        " 0x0000000000000001 (NEEDED) Shared library: [libc.so.6]\n"
        " 0x000000000000000e (SONAME) Library soname: [libfoo.so]\n"
        " 0x000000000000000f (RPATH) Library rpath: [/opt/lib:/usr/local/lib]\n"
        " 0x000000000000001d (RUNPATH) Library runpath: [$ORIGIN/../lib]\n"
    )
    ldd_out = f"\tlibc.so.6 => {lib} (0x00007f00)\n"
    calls: list[str] = []
    runner = _inspection_runner(readelf_out, ldd_out, calls)
    inspection, _results = _inspect_dynamic(
        tmp_path / "bin" / "x",
        "bin/x",
        "ET_DYN",
        tmp_path,
        Path("/usr/bin/readelf"),
        Path("/usr/bin/ldd"),
        {"PATH": "/usr/bin"},
        5.0,
        tmp_path,
        runner,
    )
    assert inspection.dynamic is True
    assert inspection.needed == ("libc.so.6",)
    assert inspection.soname == "libfoo.so"
    assert inspection.rpath == ("/opt/lib", "/usr/local/lib")
    assert inspection.runpath == ("$ORIGIN/../lib",)
    assert inspection.ldd_sha256 is not None
    assert [dep.name for dep in inspection.dependencies] == ["libc.so.6"]
    assert "ldd" in calls


def _ldd_result_runner(readelf_out: str, ldd_result: ProcessResult):
    def runner(argv: tuple[str, ...], **kwargs: object) -> ProcessResult:
        args = tuple(str(value) for value in argv)
        if "-d" in args:
            return _proc(stdout=readelf_out)
        return ldd_result

    return runner


def _run_inspect_dynamic(tmp_path: Path, runner: object):
    from strixlab.build_artifacts import _inspect_dynamic

    return _inspect_dynamic(
        tmp_path / "bin" / "x",
        "bin/x",
        "ET_DYN",
        tmp_path,
        Path("/usr/bin/readelf"),
        Path("/usr/bin/ldd"),
        {"PATH": "/usr/bin"},
        5.0,
        tmp_path,
        runner,
    )


_STATIC = "\tstatically linked\n"
_READELF = " 0x1 (NEEDED) Shared library: [libc.so.6]\n"


def test_inspect_dynamic_rejects_ldd_timeout_even_with_static_marker(tmp_path: Path) -> None:
    ldd = _proc(stderr=_STATIC, outcome=ProcessOutcome.TIMED_OUT, returncode=None)
    with pytest.raises(BuildArtifactError, match="ldd for bin/x failed"):
        _run_inspect_dynamic(tmp_path, _ldd_result_runner(_READELF, ldd))


def test_inspect_dynamic_rejects_ldd_signal_even_with_static_marker(tmp_path: Path) -> None:
    ldd = _proc(stderr=_STATIC, returncode=-9)
    with pytest.raises(BuildArtifactError, match="ldd for bin/x failed"):
        _run_inspect_dynamic(tmp_path, _ldd_result_runner(_READELF, ldd))


def test_inspect_dynamic_rejects_ldd_spawn_failure(tmp_path: Path) -> None:
    ldd = _proc(stderr=_STATIC, outcome=ProcessOutcome.SPAWN_FAILED, returncode=None)
    with pytest.raises(BuildArtifactError, match="ldd for bin/x failed"):
        _run_inspect_dynamic(tmp_path, _ldd_result_runner(_READELF, ldd))


def test_inspect_dynamic_rejects_ldd_capture_error(tmp_path: Path) -> None:
    ldd = _proc(stderr=_STATIC, returncode=1, capture_error="spool overflow")
    with pytest.raises(BuildArtifactError, match="ldd for bin/x failed"):
        _run_inspect_dynamic(tmp_path, _ldd_result_runner(_READELF, ldd))


def test_inspect_dynamic_rejects_truncated_static_marker(tmp_path: Path) -> None:
    ldd = _proc(stderr=_STATIC, returncode=1, stderr_truncated=True)
    with pytest.raises(BuildArtifactError, match="truncated"):
        _run_inspect_dynamic(tmp_path, _ldd_result_runner(_READELF, ldd))


def test_inspect_dynamic_accepts_valid_static_marker(tmp_path: Path) -> None:
    # The documented nonzero exit of the static form is accepted only once safe,
    # untruncated completion is proven.
    ldd = _proc(stderr="\tnot a dynamic executable\n", returncode=1)
    inspection, _results = _run_inspect_dynamic(tmp_path, _ldd_result_runner(_READELF, ldd))
    assert inspection.static is True
    assert inspection.dependencies == ()


def test_capture_build_artifacts_routes_timeouts(tmp_path: Path) -> None:
    from strixlab.build_artifacts import capture_build_artifacts
    from strixlab.build_identity import IdentityEntry

    root = tmp_path / "build"
    root.mkdir()
    _write_reply(root)
    (root / "bin").mkdir()
    (root / "bin" / "llama-bench").write_bytes(_elf(2))  # ET_EXEC: dynamic + executable
    (root / "CMakeCache.txt").write_bytes(b"cache")
    selections = tuple(
        IdentityEntry(name, value)
        for name, value in {
            "generator": "Ninja",
            "c_compiler": "/usr/bin/cc",
            "cxx_compiler": "/usr/bin/c++",
            "linker": "/usr/bin/ld",
            "archiver": "/usr/bin/ar",
            "toolchain_files": "",
            "sysroot": "",
        }.items()
    )
    records: list[tuple[str, str, float]] = []

    def runner(argv: tuple[str, ...], **kwargs: object) -> ProcessResult:
        args = tuple(str(value) for value in argv)
        base = args[0].rsplit("/", 1)[-1]
        flag = args[1] if len(args) > 1 else ""
        records.append((base, flag, float(kwargs["timeout"])))  # type: ignore[arg-type]
        if "-d" in args:
            return _proc(stdout=_READELF)
        if base in {"ldd", "readelf"} and flag == "--version":
            return _proc(stdout="tool v1")
        if base == "ldd":
            return _proc(stderr="\tstatically linked\n", returncode=1)  # static, no deps
        return _proc(stdout="usage")  # target --help/--version

    capture_build_artifacts(
        root,
        build_type="Release",
        requested_targets=("llama-bench",),
        selections=selections,
        toolchain_mode="host",
        environment={"PATH": "/usr/bin"},
        discovery_timeout=1.0,
        capability_timeout=2.0,
        inspection_timeout=3.0,
        process_root=tmp_path,
        runner=runner,
    )
    version_probes = {
        t for base, flag, t in records if flag == "--version" and base in {"ldd", "readelf"}
    }
    inspection = {
        t for base, flag, t in records if flag == "-d" or (base == "ldd" and flag != "--version")
    }
    capability = {t for base, flag, t in records if base == "llama-bench"}
    assert version_probes == {1.0}  # capture-tool --version → discovery timeout
    assert inspection == {3.0}  # readelf -d and ldd → inspection timeout
    assert capability == {2.0}  # target --help/--version → capability timeout


def test_parse_readelf_and_ldd_reject_truncation() -> None:
    from strixlab.build_artifacts import _parse_readelf_dynamic

    with pytest.raises(BuildArtifactError, match="truncated"):
        _parse_readelf_dynamic(_proc(stdout="(NEEDED) [libc.so.6]", truncated=True), "bin/x")
    with pytest.raises(BuildArtifactError, match="truncated"):
        _parse_ldd(_proc(stdout="\tlibc.so.6 => /lib/x (0x0)\n", truncated=True), Path("/"))


def test_parse_readelf_dynamic_rejects_malformed_and_duplicate_soname() -> None:
    from strixlab.build_artifacts import _parse_readelf_dynamic

    with pytest.raises(BuildArtifactError, match="NEEDED entry is malformed"):
        _parse_readelf_dynamic(_proc(stdout=" (NEEDED) Shared library: libc"), "bin/x")
    with pytest.raises(BuildArtifactError, match="multiple SONAME"):
        _parse_readelf_dynamic(
            _proc(stdout=" (SONAME) soname: [a.so]\n (SONAME) soname: [b.so]\n"), "bin/x"
        )


def test_parse_readelf_dynamic_absent_entries_default_empty() -> None:
    from strixlab.build_artifacts import _parse_readelf_dynamic

    parsed = _parse_readelf_dynamic(
        _proc(stdout=" 0x1 (NEEDED) Shared library: [libc.so.6]\n"), "x"
    )
    assert parsed.needed == ("libc.so.6",)
    assert parsed.soname is None
    assert parsed.rpath == ()
    assert parsed.runpath == ()

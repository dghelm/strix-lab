# ca94157 `test-backend-ops` golden fixtures — provenance

These fixtures pin the ADAPTER-002 capability grammar, operation list, and CSV result
parser to one exact upstream revision.

## Pinned source

- Fork repository: <https://github.com/halo-box/strix-llama.cpp>
- Clone URL: `https://github.com/halo-box/strix-llama.cpp.git`
- Full commit: `ca94157f70a2776e8da6b6849b50b45a083d0478`
- Target: `test-backend-ops`
- Build requirement: `LLAMA_BUILD_TESTS=ON`

The `test-backend-ops` correctness binary is only built when tests are enabled, so a
producing build profile must set `LLAMA_BUILD_TESTS=ON`. The ignored `.references/`
planning directory is not required to run these tests or consume these fixtures.

## Probe build and capture

The exact commit was configured with CMake, Ninja, and a CPU-only toolchain with the
tests target enabled:

```text
cmake -S repo -B build-tests -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF
cmake --build build-tests --target test-backend-ops -j 8
```

The captures were then made from `build-tests/bin` with:

```text
./test-backend-ops --help              > help.stdout.txt      2> help.stderr.txt
./test-backend-ops --list-ops          > list-ops.stdout.txt  2> list-ops.stderr.txt
./test-backend-ops test -o ABS -b CPU -p "type=f32" --output csv -j 1 \
                                       > abs-f32-cpu.csv       2> abs-f32-cpu.stderr.txt
```

- `--help` exited 1 and wrote the usage/option grammar to stdout; stderr was empty.
- `--list-ops` exited 0 and wrote `GGML operations:`, 128 indented uppercase
  operations, a blank line, and `Total: 128 operations`; stderr was empty.
- The filtered `test` run exited 0 and wrote the pinned seven-column CSV header plus
  four `ABS`/`type=f32` rows, all `supported=1` with empty `error_message`; stderr was
  empty.

## Normalization

`help.stdout.txt` is byte-for-byte from the capture **except** the single program-path
token immediately after `Usage:` was normalized from the local build path to the
stable basename `test-backend-ops`. The adapter's capability parser normalizes exactly
that one token before comparison, so this substitution is faithful to what the parser
observes and keeps the fixture free of an environment-specific absolute path.
`list-ops.stdout.txt` and `abs-f32-cpu.csv` are byte-for-byte captures with no
normalization. The three `*.stderr.txt` fixtures are the captured empty streams,
checked in so tests can prove the empty-stderr stream contract.

## Measurement scope and honesty

`abs-f32-cpu.csv` is a **CPU** correctness capture from the exact pinned commit. It is a
parser/provenance fixture only: it exercises the CSV grammar and the CPU reference
backend, **not** a gfx1151 or HIP correctness claim. A real HIP correctness result
requires running the pinned binary against a HIP backend build on gfx1151, which was
not performed here. Synthetic executables exercise adapter orchestration; parser
conformance is pinned by these exact-commit captures, and every fixture digest is
locked by tests.

## Fixture digests

| File | Bytes | SHA-256 |
| --- | --- | --- |
| `help.stdout.txt` | 962 | `7e500f8a2ad11602bb7923e8e722b42db73b8a6799eecb26beff820286ed1f3e` |
| `help.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `list-ops.stdout.txt` | 1419 | `bbee661695594e370097344af7943c3f262681900b0d1d751bbad2e8f4e2adc9` |
| `list-ops.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `abs-f32-cpu.csv` | 340 | `214421be9819dff6c439d10292b0f5b3d03bb3ea6d03ad989bf66250ecc16b7a` |
| `abs-f32-cpu.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

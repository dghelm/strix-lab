# ca94157 `llama-bench` golden fixtures — provenance

These fixtures pin the ADAPTER-001 capability grammar and JSONL parser to one exact
upstream revision.

## Pinned source

- Fork repository: <https://github.com/halo-box/strix-llama.cpp>
- Clone URL: `https://github.com/halo-box/strix-llama.cpp.git`
- Full commit: `ca94157f70a2776e8da6b6849b50b45a083d0478` (`master` at handoff)
- Documentation path in that tree: `tools/llama-bench/README.md`
- Raw source of the documentation:
  <https://raw.githubusercontent.com/halo-box/strix-llama.cpp/ca94157f70a2776e8da6b6849b50b45a083d0478/tools/llama-bench/README.md>

The source was fetched by full commit and built locally for the probe fixtures. The
ignored `.references/` planning directory is not required to run these tests or
consume these fixtures.

## Probe build and capture

The exact commit was configured with CMake 4.1.0, Ninja, GCC/G++ 16.2.1, and:

```text
cmake -S repo -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_VULKAN=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=ON
cmake --build build --target llama-bench -j 8
```

The captures were then made from `build/bin` with:

```text
./llama-bench --help > help.stdout.txt 2> help.stderr.txt
./llama-bench --version > version.stdout.txt 2> version.stderr.txt
```

`--help` exited 0. `--version` exited 1, wrote the same usage text to stdout,
and wrote `error: invalid parameter for argument: --version` to stderr.

## Measurement scope and honesty

No real `llama-bench` benchmark was executed to produce these fixtures. A valid JSONL
sample requires running the pinned binary against a real GGUF model on gfx1151, which
was not performed in this environment. Consequently:

- `help.stdout.txt` / `help.stderr.txt` — captured from the locally built pinned binary.
  They document the exact required flag spellings and output grammar the adapter
  matches; stderr is empty.
- `readme.jsonl` — the `### JSONL` fenced example, **byte-for-byte** from the pinned
  README. It is a documentation-protocol fixture only: it contains one `pp` row and one
  `tg` row, so it is **not** an adapter-valid single-metric sample.
- `single_metric.jsonl` — **documentation-derived, not locally measured**. It is the
  README's first JSONL row (`n_prompt: 512`, `n_gen: 0`, five `samples_ts`) with
  `model_filename` rebound to an absolute path so it satisfies the adapter's one-metric
  case binding. It conforms to the documented ca94157 JSONL protocol; its throughput
  numbers are the documented example's, not a measurement of any local build.
- `version.stdout.txt` / `version.stderr.txt` — captured from that same binary. The
  pinned source has no `--version` branch, so the advisory attempt prints usage, emits
  the invalid-parameter error above, and exits 1.

The `single_metric.jsonl` case binding it represents:

```
llama-bench -m /opt/strixlab/models/qwen2.5-7b-instruct-q4_k_m.gguf \
            -p 512 -n 0 -r 5 -o jsonl
```

## Fixture digests (SHA-256)

| file                  | bytes | sha256 |
| --------------------- | ----- | ------ |
| `help.stderr.txt`     |     0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `help.stdout.txt`     |  4312 | `e404a43db9f6bc7518e39b7c08a07a58bf00fc1e1968648863566f4b7cc5fc47` |
| `version.stderr.txt`  |    49 | `019b6a93e02ac5e378d03caee4d5211ad17da382081737494c4d27e639b06a11` |
| `version.stdout.txt`  |  4312 | `e404a43db9f6bc7518e39b7c08a07a58bf00fc1e1968648863566f4b7cc5fc47` |
| `readme.jsonl`        |  2170 | `48c5fa392ba09fb21f37f4271e37f8c8d34d7feda06f84b602651b2a356803c5` |
| `single_metric.jsonl` |  1090 | `50c7d832f4d2f80f76ea4c1596936c8b56c57610a3e29ab724dc749827396f0f` |

`tests/unit/test_llama_bench_adapter.py` re-derives and locks each digest, so a drifted
fixture fails the suite.

## Regenerate

Repeat the pinned build and probe commands above for the four `.txt` fixtures.
Fetch `tools/llama-bench/README.md` from the full commit for `readme.jsonl`, which is
the `### JSONL` fenced example with one trailing newline. `single_metric.jsonl` is
the first README JSONL row with `model_filename` rebound to the absolute path above.

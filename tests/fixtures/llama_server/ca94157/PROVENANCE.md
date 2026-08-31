# ca94157 `llama-server` golden fixtures — provenance

These files are byte-exact capability captures from a local CPU build of:

- repository: `halo-box/strix-llama.cpp`
- commit: `ca94157f70a2776e8da6b6849b50b45a083d0478`
- target: `llama-server`

The temporary checkout and build used:

```sh
cmake -S . -B build -G Ninja \
  -DGGML_CUDA=OFF \
  -DGGML_HIP=OFF \
  -DGGML_VULKAN=OFF \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TOOLS=ON
cmake --build build --target llama-server
build/bin/llama-server --help \
  > help.stdout.txt 2> help.stderr.txt
build/bin/llama-server --version \
  > version.stdout.txt 2> version.stderr.txt
```

The build system fetched its optional Web UI asset while producing this temporary
binary. That asset is irrelevant to these CLI captures and is not included here.
Adapter execution itself uses `--offline --no-ui` and performs no network access.

| file | bytes | SHA-256 |
|---|---:|---|
| `help.stdout.txt` | 58,787 | `887c3620d40312790a7701b7f8cd7d91d1ab58b172f9fa9b9ccbc018e6210fc1` |
| `help.stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `version.stdout.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `version.stderr.txt` | 84 | `7e5139bafc47a68c01ceff4d42ab8c4afde50ebce88b85e026cc4e3bc1249d80` |

These fixtures prove only the pinned CLI grammar and stream placement. The unit
suite uses a synthetic loopback server for lifecycle behavior; it does not claim
model, GPU, or performance evidence.

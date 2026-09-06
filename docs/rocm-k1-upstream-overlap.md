# K1 overlap with the Halo llama fork

Read-only source inspection on 2026-09-05 (America/Chicago), against
[`halo-box/strix-llama.cpp` master at c7af5c6](https://github.com/halo-box/strix-llama.cpp/commit/c7af5c6c29902eb1f7b3bd7952607e2349e1c668).
The local Strixlab source mirror points to this repository, but its five locked
worktrees remain detached at `ca94157f7`; they are older than the inspected master.
Current source and PR status were retrieved through authenticated GitHub access.
No inference or GPU benchmark was run for this inspection.

## Already present, and still different

The fork already contains the main reduction techniques explored here. Its HIP
backend compiles the [shared CUDA source files](https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/ggml/src/ggml-hip/CMakeLists.txt#L60-L95).
[GPU argmax](https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/ggml/src/ggml-cuda/argmax.cu#L8-L90)
uses strided row reads, value/index warp shuffles, shared wave-leader partials,
one block barrier, and a final reduction in wave 0. It launches columns rounded
up to 32 threads, capped at 1024: 32 threads for 32 columns, 256 for 256 columns,
and 1024 for 1024 columns. Thus short rows already use one wave; Strixlab's new
one-wave variant instead keeps 32 threads for every tested width.

These are algorithmic overlaps, not a Strixlab code port. No Strixlab K1 entry
points occur in the inspected fork. The implementations also differ semantically:
its argmax uses strict-greater comparisons and initializes `-FLT_MAX`/index `-1`,
whereas the research kernels explicitly handle invalid lanes and lower-original-
index ties, including signed zero. They are not interchangeable without checking
contracts.

The separate [HIP TOP_K implementation](https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/ggml/src/ggml-cuda/top-k.cu#L56-L108)
has no K=1 shortcut. With hipCUB enabled, it uses bitonic sort for eligible widths
up to 1024 and a CUB/hipCUB argsort path otherwise, then copies the first K indices.
Vulkan has its own [subgroup-sized shared-memory argmax](https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/ggml/src/ggml-vulkan/vulkan-shaders/argmax.comp#L15-L58)
and [tournament/radix TOP_K dispatch](https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L14102-L14135).
A HIP header does not change those Vulkan paths.

## Which llama workloads reach these operations?

- Backend greedy sampling already calls [ARGMAX](https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/src/llama-sampler.cpp#L1076-L1089).
  The backend [nonpositive-temperature path](https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/src/llama-sampler.cpp#L2036-L2046)
  does too. These do not need a TOP_K K=1 shortcut. Earlier samplers in a configured
  chain can still run TOP_K before temperature selection.
- Explicit backend top-k sampling calls [TOP_K with the configured K](https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/src/llama-sampler.cpp#L1477-L1487).
  [K=1 is accepted without substitution](https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/src/llama-sampler.cpp#L1519-L1532),
  so this is a reachable K=1 TOP_K path when backend sampling is active. That is
  a source-level possibility, not evidence that a measured application used it.
  Logit widths depend on the vocabulary and preceding filters; the research
  limit of 1024 columns is not a vocabulary-wide qualification. CPU sampling's
  [partial-sort implementation](https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/src/llama-sampler.cpp#L322-L338)
  does not invoke this HIP kernel.
- Qwen MoE routing uses [ARGSORT plus a top-K view](https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/src/llama-graph.cpp#L2032-L2054),
  with model-dependent expert/group counts. Even if an expert selection uses K=1,
  changing only TOP_K would not affect it:
  [ggml_argsort_top_k builds ARGSORT](https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/ggml/src/ggml.c#L5389-L5402).
- Flash-Next/Qwen4 QSA uses [context-dependent selection width](https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/src/models/qwen4exp.cpp#L893-L899),
  while [DeepSeek indexer selection](https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/src/models/deepseek32.cpp#L323-L326)
  and [DFlash selection](https://github.com/halo-box/strix-llama.cpp/blob/c7af5c6c29902eb1f7b3bd7952607e2349e1c668/src/models/dflash.cpp#L482-L495)
  use configured K values. These call sites do not establish a hot K=1 workload.
  Masks and logit suppression can also introduce infinities outside the research
  fixture's finite-input contract.

## PR status and interpretation

[PR 18](https://github.com/halo-box/strix-llama.cpp/pull/18), including HIP argsort/
hipCUB changes, and [PR 20](https://github.com/halo-box/strix-llama.cpp/pull/20),
including deterministic Vulkan radix selection, are merged into the inspected
master. Open [PR 17](https://github.com/halo-box/strix-llama.cpp/pull/17) concerns
Vulkan attention work; open [PR 16](https://github.com/halo-box/strix-llama.cpp/pull/16)
also includes Vulkan radix-selection changes. Their reviewed file lists show no
HIP argmax or TOP_K K=1 port. Open PR 7 concerns Vulkan matvec, not this operation.

The Strixlab measurements compare specific implementations and call boundaries.
They are neither evidence that warp reductions are novel nor evidence of faster
llama inference. In particular, the initial sort-versus-K1 ratio cannot be applied
to the fork's existing greedy argmax. Before proposing a port, capture actual
backend operations, K, row/column shapes and input semantics, then benchmark
against the existing implementation and measure the affected application.

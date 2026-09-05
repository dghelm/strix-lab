# Field report: rocprofv3 deferred HIP argument string lifetime

- **Report kind:** repository validation (read-only profiler evidence investigation)
- **StrixLab commit:** `9eb7a96cdd11c723ae6ae804710c05e8d06b413f`
- **Date:** 2026-09-05

## Outcome

- **Result:** inconclusive for the exact allocation history; upstream lifetime hazard identified in pinned source.

The malformed PP512 and PP2048 JSON reported in the
[baseline profile](2026-09-05-qwen35-baseline-profile.md) is consistent with a
rocprofiler-SDK defect: buffered HIP records retain argument pointers, but later
JSON serialization dereferences string pointers. The string formatter bypasses
the buffered iterator's zero-dereference limit for strings. A caller's name
buffer can therefore be read after it changes or ceases to exist.

No StrixLab JSON rewriting was found. The saved launcher directly invokes
rocprofv3, which writes its own trace files. Build/model leases were recorded as
verified before and after the original executions. This investigation performed
no GPU execution, rerun, source/build/model mutation, toolkit change, or raw repair.

## Environment and provenance

Linux, Strix Halo 128G class, gfx1151. Installed Arch package `rocprofiler 7.2.4-1`
reports SDK **1.1.0**, ROCm **7.2.4**, revision
[`97f5574fe2fdc7bef44fb01545347912ee9f1779`](https://github.com/ROCm/rocm-systems/commit/97f5574fe2fdc7bef44fb01545347912ee9f1779).
The package database records signature validation. Current SHA-256 values match
its stored package manifest:

| Installed component | SHA-256 |
|---|---|
| `bin/rocprofv3` | `8db152ac9e03b920245f8b17b589b6bb7fab0f94faf3e55d4960e7a6841b32e4` |
| `lib/librocprofiler-sdk.so.1.1.0` | `b0511fa108c78e17ee4776aac47cb94fd52c2f5092f8cbd47fb662ae384c1d15` |
| `lib/rocprofiler-sdk/librocprofiler-sdk-tool.so.1.1.0` | `4c07af0fb2c9b2a1318d8551d7d166eec5e1b06e0984223d2fd5bc1dde21b436` |

The launcher hash also matches both saved PP invocations. Original runs did not
record profiler shared-library hashes, so present library verification cannot
prove their historical loaded bytes. `pacman -Qkk rocprofiler` reported only UID
and GID mismatches (996 each); these were not repaired. Explicit component digest
checks above supply content evidence independently of ownership metadata.

Pinned source blobs were retrieved read-only through authenticated GitHub API
calls and their Git blob IDs verified against the revision's tree. The installed
`cxx/serialization/save.hpp` is byte-identical to that pinned source blob.

## Source-grounded mechanism

All links below pin the installed revision in `ROCm/rocm-systems`:

1. The tool selects extended buffered HIP tracing in
   [tool.cpp:2012](https://github.com/ROCm/rocm-systems/blob/97f5574fe2fdc7bef44fb01545347912ee9f1779/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk-tool/tool.cpp#L2012-L2017).
2. The HIP wrapper copies `tracer_data.args` into `extended_record.args`, retaining
   pointer values, in [hip.cpp:313](https://github.com/ROCm/rocm-systems/blob/97f5574fe2fdc7bef44fb01545347912ee9f1779/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hip/hip.cpp#L313-L328).
   The tool copies the record into its ring buffer in
   [tool.cpp:1056](https://github.com/ROCm/rocm-systems/blob/97f5574fe2fdc7bef44fb01545347912ee9f1779/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk-tool/tool.cpp#L1056-L1066).
3. JSON output later consumes stored records in
   [tool.cpp:2692](https://github.com/ROCm/rocm-systems/blob/97f5574fe2fdc7bef44fb01545347912ee9f1779/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk-tool/tool.cpp#L2692-L2718).
   [save.hpp:589](https://github.com/ROCm/rocm-systems/blob/97f5574fe2fdc7bef44fb01545347912ee9f1779/projects/rocprofiler-sdk/source/include/rocprofiler-sdk/cxx/serialization/save.hpp#L589-L595)
   calls `get_buffer_tracing_args`, which invokes the buffered argument iterator
   at [save.hpp:140](https://github.com/ROCm/rocm-systems/blob/97f5574fe2fdc7bef44fb01545347912ee9f1779/projects/rocprofiler-sdk/source/include/rocprofiler-sdk/cxx/serialization/save.hpp#L113-L142).
4. The buffered iterator supplies `max_deref=0` in
   [hip.cpp:635](https://github.com/ROCm/rocm-systems/blob/97f5574fe2fdc7bef44fb01545347912ee9f1779/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hip/hip.cpp#L635-L647),
   but the string branch constructs `std::string{_v}` without consulting that
   limit in [stringize_arg.hpp:59](https://github.com/ROCm/rocm-systems/blob/97f5574fe2fdc7bef44fb01545347912ee9f1779/projects/rocprofiler-sdk/source/lib/common/stringize_arg.hpp#L54-L79).

This establishes an upstream deferred-read hazard. It does not reconstruct the
original name, its owning library, its allocation/free history, or the exact
moment of corruption in the saved traces. No claim is made that a later ROCm
release fixes this path.

## Raw evidence checks

Both PP files fail strict UTF-8 decoding in
`rocprofiler-sdk-tool[0].buffer_records.hip_api[39861].args[2].value` and
`hip_api[39863].args[2].value`: `hipModuleGetFunction.kname`, correlations 39862
and 39864, recorded return values 500 and 0 respectively. Within each run the
two strings contain identical six-byte values. First invalid byte offsets are
37260469 (PP512) and 40189203 (PP2048), zero-based.

Forensic parsing used `surrogateescape` with an asserted exact byte roundtrip;
no decoded or repaired trace was written. Interpreting each six-byte value as
a little-endian integer gives a value near the following
`hipExtModuleLaunchKernel` function pointer: differences are **+0x5a0** for PP512
and **+0xa0** for PP2048. This is consistent with reused storage containing a
pointer, not proof of a particular allocator mechanism. Raw addresses and the
address-bearing byte strings are deliberately omitted.

| Preserved raw file | SHA-256 |
|---|---|
| `pp512/pp512_results.json` | `af42c53ddca0294a01a7a303e2a137bc6bddfef5902d38933f3cde36082c6363` |
| `pp2048/pp2048_results.json` | `34abfad58b65d315568aa7eab852a9573c861baba6aa1b9fd51b54b401784704` |

Existing CSV timestamp/count agreement remains limited timing evidence. This
investigation does not reclassify either PP JSON bundle as valid, prove all
other argument strings correct, or establish performance gains.

## Commands run and next capture configuration

Read-only checks included `pacman -Qi rocprofiler`, `pacman -Qkk rocprofiler`,
component SHA-256/package-manifest comparisons, saved launcher/config inspection,
byte-preserving raw inspection, and pinned upstream tree/blob retrieval with
`gh api`. No application or profiler workload was executed.

Original invocation structure, with private paths redacted:

```text
rocprofv3 --hip-trace --kernel-trace --memory-copy-trace \
  --memory-allocation-trace --stats --output-config \
  --output-format csv json pftrace --perfetto-backend inprocess \
  --output-directory <fresh-case-directory> --output-file <case> -- \
  <authenticated-build>/bin/llama-bench -m <leased-model-fd> \
  -p <512|2048> -n 0 -r 1 -o jsonl
```

For a future coordinator-owned capture, use **`--output-format csv`**, omit the
Perfetto-only option, and retain the tracing domains, stats, saved config,
exclusive lock, fresh output directory, and authenticated leases. This avoids
the JSON argument-formatting branch; it is **not a root-cause fix** and does not
recover argument text or repair old evidence. No additional capture is proposed
for this investigation. The CSV-only configuration is source-supported, not
newly validated by GPU execution here. Any future capture still requires its
own exit/status, strict CSV, count/statistics, and dropped-record checks.

An upstream fix should either respect zero dereferences for buffered string
pointers or own a copy made while the argument is valid. UTF-8 replacement alone
would hide the symptom without addressing lifetime. A future upstream regression
test could change a still-live name buffer after a successful lookup, avoiding
undefined behavior while testing whether serialization reads the later contents.
No such reproducer was executed; no issue was posted.

## Privacy confirmation

- [x] This report contains no model weights, credentials, private data, raw
      StrixLab home, raw memory addresses, or evidence bundle; identifying values
      are redacted. Original evidence remains unchanged.

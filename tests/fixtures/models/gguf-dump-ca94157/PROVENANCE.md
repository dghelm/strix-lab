# gguf-dump ca94157 fixture provenance

These fixtures pin the exact JSON contract of the source-compatible GGUF inspector
used by MODEL-001:

```
gguf-py/gguf/scripts/gguf_dump.py <model> --json
```

from the pinned preparation of
`strix-llama.cpp@ca94157f70a2776e8da6b6849b50b45a083d0478`.

## Files

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `tiny-qwen35.gguf` | 448 | `4b922fe8fbd661fd639f07b481165e24ec4475e38bf456cc2cc6824d733f36ab` |
| `gguf_dump.json` | 965 | `9ee6115a9cf39b64a0eafe267efde07000cce7c66ebf0e64963af350683c6f1c` |

## How these were produced

`tiny-qwen35.gguf` was written with the leased ca94157 `gguf` Python package
(`gguf.GGUFWriter`, `arch="qwen35"`). It is a synthetic, text-free GGUF carrying:

- scalar metadata across several GGUF value types (`STRING`, `UINT32`, `UINT64`,
  `FLOAT32`, `BOOL`), including `general.architecture = "qwen35"`;
- one array-typed metadata field (`qwen35.tiny_list`, `INT32` elements);
- two tiny tensors (`a.weight` `F32`, `b.weight` `F16`).

`gguf_dump.json` is the byte-exact stdout of

```
PYTHONPATH=<lease>/gguf-py python3 <lease>/gguf-py/gguf/scripts/gguf_dump.py model.gguf --json
```

run from this directory (so the tool's `filename` field is `model.gguf`). MODEL-001's
normalization drops `filename` and the positional `index`/`offset` fields, so the
portable metadata projection is independent of where the file lived when dumped.

No model weights, model-card text, tokenizer arrays (`--json-array` is never used), or
network fetches are involved. The GGUF contains only zero-filled tensors.

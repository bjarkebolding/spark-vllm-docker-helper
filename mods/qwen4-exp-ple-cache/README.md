# qwen4-exp-ple-cache

A GPU-resident LRU row cache for the `VLLM_PLE_MMAP` n-gram gather in
Qwen3.8-Flash-Next.

## Why

`MmapNgramEmbedding.forward` reads every n-gram row it needs from the mmap'd
PLE table on every decode step. On a DGX Spark / GB10 the table is 47.7 GiB —
far larger than the page cache that is left after the ~79 GiB of resident
weights and ~23 GiB of KV cache — so those reads are cold NVMe random reads.
Measured at ~3–7 ms per decode step (the `copy_ms` field of the
`PLE mmap: ... copy_ms=` log line), i.e. ~10–20 % of the token time.

## What it does

Keeps the most-recently-gathered rows resident on the GPU (a ring buffer keyed
by global row id) and only reads the **misses** from disk. Table rows are
immutable, so a cached row is always valid; eviction is a plain ring.

- `VLLM_PLE_MMAP_CACHE_ROWS=N` — cache N rows. Default `1048576` (~168 MiB GPU
  for a 160-byte row).
- `VLLM_PLE_MMAP_CACHE_ROWS=0` — disable; fall back to the stock gather.

Logs `PLE cache: hit_rate=... (hits=... miss=...)` once per 60 s. If the
hit rate is low for your workload, set it to `0` — the cache is then pure
overhead.

## How

`run.sh` is a guarded, idempotent text patch of
`vllm/models/qwen4_exp/nvidia/ple_mmap.py` (`MmapNgramEmbedding.forward`).
It anchors on the stock `ids.detach().to("cpu", ...)` line and refuses to
patch an unrecognised layout. AST-syntax-checked before writing.

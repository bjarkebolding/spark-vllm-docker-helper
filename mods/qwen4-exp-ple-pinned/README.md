# qwen4-exp-ple-pinned

Pinned-host staging for the `VLLM_PLE_MMAP` PLE gather.

## What it changes

`MmapNgramEmbedding.forward` in `vllm/models/qwen4_exp/nvidia/ple_mmap.py`
gathers PLE rows into a **pageable** numpy buffer, then does a pageable
host→device copy. On GB10's coherent memory a pageable H2D is much slower than
a pinned one and it blocks the calling thread. This mod stages the gathered
rows through a persistent pinned buffer so the H2D is a real async DMA.

The patch is one free function plus a two-line call site. It does not touch the
gather itself, the readahead pre-pass, or CUDA-graph behaviour.

## What to expect

| workload | effect |
|---|---|
| single-stream decode | small — the copy is a few KB/step |
| prefill | meaningful — the copy is MB-scale, and pageable MB copies are the slow ones |
| concurrent serving | meaningful — copy size grows with batch × tokens |

The thing that actually fixes slow single-stream decode on this model is the
recipe's `VLLM_PLE_MMAP_WORKERS` and `gpu_memory_utilization` — see the recipe
comments. This mod is a second-order polish on top of that.

## Use

```bash
./run-recipe.sh /abs/path/spark-vllm-docker-helper/recipes/qwen3.8-flash-next-nvfp4.yaml \
  --solo --setup --earlyoom -d \
  --apply-mod /abs/path/spark-vllm-docker-helper/mods/qwen4-exp-ple-pinned
```

Runtime toggle: `VLLM_PLE_MMAP_PINNED=0` restores the stock pageable copy
without un-applying the mod.

## Safety

`apply.py` is idempotent and shape-guarded: it verifies the exact stock code
shape before writing and aborts otherwise, so a future vLLM that reworks this
path fails the mod loudly instead of being silently mis-patched. Any pinned
allocation failure at runtime falls back to the stock pageable copy.

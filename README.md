# Qwen3.8-Flash-Next (NVFP4) — recipe for spark-vllm-docker

Runs **Qwen3.8-Flash-Next** on **one DGX Spark / GB10** at 262k context, via
[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) — which it
does not modify or wrap. It's just a recipe.

```
recipes/qwen3.8-flash-next-nvfp4.yaml     the recipe (NVFP4 experts, bf16 dense side)
recipes/qwen3.8-flash-next-fp8hybrid.yaml faster — dense side also fp8 (~+18% decode)
patches/qwen38-flash-next-ple-mmap.patch  fallback — only if PR #54129 stops applying
mods/qwen4-exp-fp8-hybrid/                for the fp8hybrid recipe (builds the checkpoint)
mods/qwen4-exp-ple-pinned/                optional — pinned-host staging for the PLE gather
```

## Run it

Needs a GB10 Spark (128 GB, driver **580.x**), Docker, ~140 GB free NVMe.

```bash
git clone https://github.com/bjarkebolding/spark-vllm-docker-helper

cd /path/to/spark-vllm-docker            # your checkout, untouched
./run-recipe.sh /path/to/spark-vllm-docker-helper/recipes/qwen3.8-flash-next-nvfp4.yaml \
    --solo --setup --earlyoom -d         # build (~66 min) + download (~126 GiB) + serve
docker logs -f vllm_node                 # ready at "Application startup complete"
```

Drop `--setup` on later runs. API on `localhost:8000/v1` — it's a thinking model,
send `chat_template_kwargs: {"enable_thinking": false}` for short answers.

## What it builds

Official `vllm-project/vllm` @ `16d6c376` **+ PR #54129** (`--apply-vllm-pr 54129`,
no fork). PR #54129 adds `VLLM_PLE_MMAP`, which serves the 47.7 GiB PLE n-gram
table from disk via `np.memmap` instead of GPU/RAM — that plus an NVFP4 body is
what fits the model on one GB10 at TP1. GB10 tuning lives in the recipe.

## Measured (single GB10)

Non-streaming wall clock (a streaming client adds ~40 ms/token of its own overhead
and reads roughly 2× low):

| | |
|---|---|
| prose decode | **~27 tok/s** |
| code decode | **~32 tok/s** |
| concurrent | ~80 tok/s @ 4 streams · ~122 @ 8 |
| prefill | ~2450 tok/s, flat to 120k |
| KV pool | 522k tokens · 35k needle passes · greedy-deterministic |

**~27–32 single-stream is the NVFP4-lossless ceiling on this hardware.** Every
independent single-Spark report for this model lands in the same band (best
measured anywhere: 32.2). Decode is not GPU-bound and not quite memory-bound —
~20 ms of each ~35 ms step is fixed launch/scheduler/indexer overhead that neither
vLLM nor llama.cpp removes today. Going faster needs a re-quantized checkpoint
(FP8 dense side → ~31–33, quality-neutral; W4A16 on the QSA projections → ~40,
unverified on GB10) or a lossy PLE table — not a config change.

### `mods/qwen4-exp-ple-pinned` (optional)

Routes the PLE gather's host→device copy through a pinned buffer.
`--apply-mod /abs/path/mods/qwen4-exp-ple-pinned`. Correctness-verified
(deterministic, needle passes); **no measurable single-stream effect** — its
value is prefill and concurrent serving, where the copy is MB-scale. Skip it
unless you're serving multiple streams.

## Checkpoint & license

[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
(NVIDIA ModelOpt NVFP4). Apache-2.0; the vendored patch is derived from vLLM
PR #54129 (Apache-2.0).

# Qwen3.8-Flash-Next (NVFP4) — recipe for spark-vllm-docker

Runs **Qwen3.8-Flash-Next** on **one DGX Spark / GB10** at 262k context, via
[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) — which it
does not modify or wrap. It's just a recipe.

```
recipes/qwen3.8-flash-next-nvfp4.yaml     the recipe
patches/qwen38-flash-next-ple-mmap.patch  fallback — only if PR #54129 stops applying
tools/bench.py                            optional benchmark
RESULTS.md                                tuning notes: what helped, what didn't
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

~29 tok/s prose · ~35 code · ~42 aggregate at 4 concurrent · ~877k KV tokens ·
64k+128k needle retrieval passes · greedy-deterministic.

## Checkpoint & license

[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
(NVIDIA ModelOpt NVFP4). Apache-2.0; the vendored patch is derived from vLLM
PR #54129 (Apache-2.0).

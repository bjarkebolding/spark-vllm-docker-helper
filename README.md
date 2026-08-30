# Qwen3.8-Flash-Next (NVFP4) — recipe for spark-vllm-docker

Runs **Qwen3.8-Flash-Next** on **one DGX Spark / GB10** at 262k context, via
[`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) — which it
does not modify or wrap. It's just a recipe.

Three recipes, fastest last:

```
recipes/qwen3.8-flash-next-nvfp4.yaml      NVFP4 experts, bf16 dense side. ~27 prose / ~32 code. Deterministic.
recipes/qwen3.8-flash-next-fp8hybrid.yaml  + fp8 dense side.               ~32 / ~36.  Deterministic. +mod fp8-hybrid.
recipes/qwen3.8-flash-next-w4a16.yaml      int4 experts + int8 head + fp8. ~40 / ~48.  NON-deterministic (Marlin). +mods below.
```

```
patches/qwen38-flash-next-ple-mmap.patch  fallback — only if PR #54129 stops applying
mods/qwen4-exp-fp8-hybrid/     fp8 dense side          (fp8hybrid + w4a16 recipes)
mods/qwen4-exp-w4a16-gptq-fp8/ int4 GPTQ + fp8 dispatch (w4a16 recipe; also links the PLE shards in)
mods/qwen4-exp-int8-lmhead/    int8 lm_head + MTP head  (w4a16 recipe)
mods/qwen4-exp-fla-gb10/       GB10 FLA kernel fixes    (w4a16 recipe)
mods/qwen4-exp-ple-pinned/     optional — pinned-host PLE staging (helps prefill/concurrency)
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

~27–32 is the NVFP4 ceiling — decode is launch/scheduler-bound, not a config
knob. Re-quantized checkpoints go further:

| recipe | prose / code | KV | notes |
|---|---|---|---|
| `nvfp4` | ~27 / ~32 | 522k | NVFP4 experts, bf16 dense side. Deterministic. |
| `fp8hybrid` | ~32 / ~36 | 594k | + fp8 dense side. Deterministic. Quality-neutral. |
| `w4a16` | **~40 / ~48** | **747k** | int4 experts (Marlin) + int8 head + fp8 dense. **Non-deterministic** (Marlin kernel jitter — outputs vary within correct answers). MTP=3. |

`w4a16` needs `Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid` — see that recipe's
header. Credit: the int4/int8/fp8 recipe and shims are
[@Saren-Arterius](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound) /
[blazux](https://github.com/blazux/qwen3.8-Flash-DGX) (Apache-2.0).

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

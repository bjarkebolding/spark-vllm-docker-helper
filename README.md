# Qwen3.8-Flash-Next on one DGX Spark (GB10)

Recipes for running **Qwen3.8-Flash-Next** (~180 B, 262 k context) on a **single GB10 /
DGX Spark** via [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker) —
unmodified. Official vLLM + one PR, no fork.

## Recipes — pick one (fastest last)

| recipe | prose / code | KV pool | deterministic | needs |
|---|---:|---:|:---:|---|
| `qwen3.8-flash-next-nvfp4.yaml` | ~27 / ~32 | 522 k | yes | RadixArk NVFP4 checkpoint |
| `qwen3.8-flash-next-fp8hybrid.yaml` | ~32 / ~36 | 594 k | yes | + `fp8-hybrid` mod (builds the checkpoint) |
| **`qwen3.8-flash-next-w4a16.yaml`** | **~40 / ~48** | **747 k** | **no** *(Marlin jitter)* | Saren W4A16 checkpoint + 4 mods |

Non-streaming wall-clock decode. `w4a16` matches the best published single-Spark figure.
See [`RESULTS.md`](RESULTS.md) for the full numbers, how, and what didn't work.

## Run

```bash
git clone https://github.com/bjarkebolding/spark-vllm-docker-helper H
cd /path/to/spark-vllm-docker            # your checkout, untouched

# fastest — w4a16
./run-recipe.sh $PWD/../H/recipes/qwen3.8-flash-next-w4a16.yaml --solo --setup --earlyoom -d \
  --apply-mod $PWD/../H/mods/qwen4-exp-w4a16-gptq-fp8 \
  --apply-mod $PWD/../H/mods/qwen4-exp-int8-lmhead \
  --apply-mod $PWD/../H/mods/qwen4-exp-fla-gb10 \
  --apply-mod $PWD/../H/mods/qwen4-exp-qsa-exact-topk \
  --apply-mod $PWD/../H/mods/qwen4-exp-ple-pinned

# deterministic — fp8hybrid
./run-recipe.sh $PWD/../H/recipes/qwen3.8-flash-next-fp8hybrid.yaml --solo --setup --earlyoom -d \
  --apply-mod $PWD/../H/mods/qwen4-exp-fp8-hybrid \
  --apply-mod $PWD/../H/mods/qwen4-exp-ple-pinned

# conservative — nvfp4
./run-recipe.sh $PWD/../H/recipes/qwen3.8-flash-next-nvfp4.yaml --solo --setup --earlyoom -d \
  --apply-mod $PWD/../H/mods/qwen4-exp-ple-pinned
```

Ready when `docker logs -f vllm_node` shows `Application startup complete`. API on
`localhost:8000/v1` — it's a thinking model, send `chat_template_kwargs: {"enable_thinking": false}`
for short answers.

Requires: a GB10 Spark (128 GB, driver 580.x), Docker, ~130–250 GB free NVMe depending on
the recipe. Checkpoints download on first `--setup`.

## Contents

```
recipes/    the 3 recipes above
mods/       guarded source patches (each applied with --apply-mod; run.sh + README each)
  qwen4-exp-ple-pinned       pinned-host staging for the PLE gather   (all recipes, optional)
  qwen4-exp-fp8-hybrid       fp8 dense side + checkpoint builder       (fp8hybrid)
  qwen4-exp-w4a16-gptq-fp8   int4-GPTQ + fp8 dispatch shim            (w4a16)
  qwen4-exp-int8-lmhead      int8 lm_head in model + MTP head         (w4a16)
  qwen4-exp-fla-gb10         GB10 FLA kernel fixes                    (w4a16)
  qwen4-exp-qsa-exact-topk   deterministic QSA top-k (correctness)    (w4a16)
patches/    frozen diff of vLLM PR #54129 — fallback if the PR is rebased and stops applying
RESULTS.md  numbers, method, negative results
```

## What it builds

Official `vllm-project/vllm` @ `16d6c376` **+ PR #54129** (`VLLM_PLE_MMAP` — serves the
47.7 GiB PLE n-gram table from NVMe via mmap instead of resident memory; that's what fits
the model on one GB10 at TP1). No fork. Everything else is checkpoint quantization + GB10
kernel/config workarounds, all as `--apply-mod` patches.

## Credits

[@Saren-Arterius](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound) (the
int4/int8/fp8 recipe, quantizers, and shims), [blazux](https://github.com/blazux/qwen3.8-Flash-DGX)
(the disk-mmap-PLE foundation + fp8 hybrid), the vLLM PR #54129 author, Intel (W4A16
AutoRound base), RadixArk (NVFP4 checkpoint). Apache-2.0.

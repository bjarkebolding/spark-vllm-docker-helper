# spark-vllm-docker-helper

> Run **Qwen3.8-Flash-Next** (NVFP4) on a single **DGX Spark / GB10** with vLLM —
> without patching [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker).

Qwen3.8-Flash-Next is a Qwen4-preview: a ~125B ultra-sparse MoE (6B active) with a
separate **~51B PLE n-gram table** and a hybrid Gated-DeltaNet + Qwen-Sparse-Attention
stack. No released vLLM image runs it on a GB10, and the documented
`VLLM_PLE_CPU_OFFLOAD` flag deadlocks at TP=1. This repo carries the one config that
works — a recipe plus a helper script that drives an **unmodified** `spark-vllm-docker`
checkout.

```
recipes/qwen3.8-flash-next-nvfp4.yaml    # the recipe
patches/qwen38-flash-next-ple-mmap.patch # frozen vLLM diff (fallback only)
flash-next.sh                            # the helper
```

## Quick start

Needs: a DGX Spark / GB10 (128 GB, sm_121, driver **580.x**), Docker, ~140 GB free
NVMe, `git`, `python3`.

```bash
git clone https://github.com/bjarkebolding/spark-vllm-docker-helper
cd spark-vllm-docker-helper

./flash-next.sh setup        # clone spark-vllm-docker, build image (~66 min),
                             # download model (~126 GiB), start serving
./flash-next.sh logs         # wait for "Application startup complete" (~12 min)
./flash-next.sh test         # one chat request against localhost:8000
```

`setup` is idempotent — re-run it and it skips the build and download if they're
already done. Other commands: `serve`, `build`, `download`, `dry-run`, `logs`,
`stop`. Point `SPARK_VLLM_DOCKER=/path` at an existing checkout to reuse it.

The OpenAI-compatible API is then on `http://localhost:8000/v1`. It's a thinking
model — send `chat_template_kwargs: {"enable_thinking": false}` for short answers.

## How it works

`flash-next.sh` runs `spark-vllm-docker/run-recipe.sh` with the recipe passed by
absolute path, so `spark-vllm-docker` itself is never touched. The recipe's
`build_args` build the image from **official `vllm-project/vllm`**:

| | |
|---|---|
| `--vllm-ref 16d6c376…` | official `vllm-project/vllm` main @ 2026-08-29 |
| `--apply-vllm-pr 54129` | [PR #54129](https://github.com/vllm-project/vllm/pull/54129) (disk-mmap PLE), fetched from `vllm-project/vllm`'s own PR endpoint |

`16d6c376` is PR #54129's merge-base, so the diff applies cleanly. PR #54129 stacks
the model PR [#53896](https://github.com/vllm-project/vllm/pull/53896), so this one
flag brings the whole `qwen4_exp` architecture plus `VLLM_PLE_MMAP`.

At runtime `VLLM_PLE_MMAP=1` serves the **47.7 GiB FP8 PLE table straight from the
checkpoint's safetensors via `np.memmap`** — page-cache backed, never resident on
GPU or in a worker. That's what makes the model fit: ~79 GiB of resident weights
leaves ~20 GiB for a bf16 KV cache (~730k tokens).

### Why not the usual approaches

| | |
|---|---|
| FP8 body | doesn't fit — main model alone ~115 GiB |
| `VLLM_PLE_CPU_OFFLOAD=1` | **deadlocks at CUDA-graph warmup** at TP=1 on GB10 — [vLLM #53960](https://github.com/vllm-project/vllm/issues/53960) |
| `--enable-prefix-caching` | crashes the GDN path on sm_121 — [#54173](https://github.com/vllm-project/vllm/issues/54173) |
| `vllm/vllm-openai:qwen38-flash-next` | stale — pre-rename `Qwen3_8FlashNext…`, won't load current checkpoints |
| `--moe-backend marlin` | errors — the model has unquantized BF16 MoE layers |

### If PR #54129 is rebased and stops applying

Use the frozen diff in `patches/` (`sha256 3273ad05…`, `git diff 16d6c376..PR-head`):

```bash
git clone https://github.com/vllm-project/vllm && cd vllm
git checkout 16d6c376bc55e2e44ce012d26104c8af11e27c25
git apply ../spark-vllm-docker-helper/patches/qwen38-flash-next-ple-mmap.patch
git add -A && git -c user.email=x -c user.name=x commit -m qwen4_exp-ple-mmap
```

then in `recipes/qwen3.8-flash-next-nvfp4.yaml` replace the `--apply-vllm-pr / "54129"`
lines with `--vllm-source-dir / /abs/path/to/vllm`.

## Measured

DGX Spark, GB10, driver 580.173.02, onboard NVMe:

| | |
|---|---|
| Resident weights | **79.3 GiB** (NVFP4 routed experts + BF16 attention/QSA/GDN/MTP) |
| PLE table | 47.68 GiB on disk, 128 shards — **0 B resident** |
| KV cache (bf16) | ~21 GiB → **~760k tokens**, 2.9× concurrency at full 262k |
| Startup | ~12 min (weight stream + PLE prewarm) |
| Decode, 1 stream, prose | **~27–29 tok/s** (MTP acceptance 65–75%) |
| Prefill | ~1.3k tok/s ; 64k-token prompt end-to-end ~46 s |
| 64k needle-in-haystack | retrieves correctly |
| Concurrency 4 | ~37 tok/s aggregate |

Memory runs tight (~116/121 GiB, a few GiB swap under load; `earlyoom` armed).
Good for one-to-few long-context streams; heavy concurrency is memory-bound.
`sudo nvidia-smi -lgc 0,3003` before serving recovers ~15–20% — the GB10
otherwise boosts to only ~2.4 of its 3.0 GHz.

## Checkpoint

[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
— NVIDIA ModelOpt NVFP4 (routed experts only). `config.json` declares
`Qwen4ExpForConditionalGeneration`, matching PR #53896. No `nvidia/` or `RedHatAI/`
NVFP4 of this model exists yet.

## License

Apache-2.0. The vendored patch is derived from vLLM PR #54129 (Apache-2.0).

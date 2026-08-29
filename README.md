# Qwen3.8-Flash-Next (NVFP4) — recipe add-on for spark-vllm-docker

A tuned [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker)
**recipe** that runs **Qwen3.8-Flash-Next** (Qwen4-preview, ~125B ultra-sparse
MoE + a ~51B PLE n-gram table, Gated-DeltaNet + Qwen-Sparse-Attention hybrid) on
a **single DGX Spark / GB10** at full 262k context.

This is an add-on, not a wrapper. `spark-vllm-docker` is used exactly as you
already use it — you just point `run-recipe.sh` at the recipe here.

```
recipes/qwen3.8-flash-next-nvfp4.yaml     the recipe
patches/qwen38-flash-next-ple-mmap.patch  frozen vLLM diff (only if PR #54129 stops applying)
tools/bench.py                            optional: prose / code / needle / concurrency bench
RESULTS.md                                what was tuned, what didn't work, the decode ceiling
```

## Use it

Needs: DGX Spark / GB10 (128 GB, sm_121, driver **580.x** — 590.x has a CUDA-graph
deadlock), Docker, ~140 GB free NVMe.

```bash
git clone https://github.com/bjarkebolding/spark-vllm-docker-helper
cd /path/to/your/spark-vllm-docker          # your existing checkout, unmodified

./run-recipe.sh /path/to/spark-vllm-docker-helper/recipes/qwen3.8-flash-next-nvfp4.yaml \
    --solo --setup --earlyoom -d
#   --setup = build the image (~66 min) + download the model (~126 GiB) + serve
docker logs -f vllm_node                    # wait for "Application startup complete"
```

`--setup` is idempotent. Drop `--setup` on later runs. The OpenAI API is then on
`localhost:8000/v1`. It's a thinking model — send
`chat_template_kwargs: {"enable_thinking": false}` for short answers.

Prefer keeping recipes in-tree? Copy the file into `spark-vllm-docker/recipes/`
and run `./run-recipe.sh qwen3.8-flash-next-nvfp4 --solo --setup` — same thing.

Optional, after startup, to compile the QSA/indexer Triton kernels before real
traffic (otherwise the first few requests spike):

```bash
python3 /path/to/spark-vllm-docker-helper/tools/bench.py --mode warmup
python3 .../tools/bench.py --mode all      # prose + needle + concurrency
```

## What the recipe does

The image builds from **official `vllm-project/vllm`** pinned to `16d6c376`
(2026-08-29 main) plus **PR #54129** (`--apply-vllm-pr 54129`, fetched from
`vllm-project/vllm`'s own PR endpoint). `16d6c376` is that PR's merge-base, so the
diff applies cleanly. PR #54129 (disk-mmap PLE, `VLLM_PLE_MMAP`) stacks the model
PR [#53896](https://github.com/vllm-project/vllm/pull/53896), so this one flag
brings the whole `qwen4_exp` architecture. No third-party vLLM repo.

At runtime `VLLM_PLE_MMAP=1` serves the 47.7 GiB FP8 PLE table straight from the
checkpoint's safetensors via `np.memmap` — never resident on GPU or in a worker.
That's what makes the model fit: ~79 GiB of resident weights leaves ~24 GiB for a
bf16 KV cache (~877k tokens).

### Why not the obvious approaches

| | |
|---|---|
| FP8 body | doesn't fit — main model alone ~115 GiB |
| `VLLM_PLE_CPU_OFFLOAD=1` (the documented flag) | **deadlocks at CUDA-graph warmup** at TP=1 on GB10 — [vLLM #53960](https://github.com/vllm-project/vllm/issues/53960) |
| `--enable-prefix-caching` | crashes the GDN path on sm_121 — [#54173](https://github.com/vllm-project/vllm/issues/54173) |
| `vllm/vllm-openai:qwen38-flash-next` | stale image — pre-rename `Qwen3_8FlashNext…`, won't load current checkpoints |
| `--moe-backend b12x` / `marlin` | error — the MTP draft has unquantized BF16 MoE layers |
| `--kv-cache-dtype fp8` | QSA refuses it |

### If PR #54129 is rebased and stops applying to `16d6c376`

```bash
git clone https://github.com/vllm-project/vllm && cd vllm
git checkout 16d6c376bc55e2e44ce012d26104c8af11e27c25
git apply /path/to/spark-vllm-docker-helper/patches/qwen38-flash-next-ple-mmap.patch
git add -A && git -c user.email=x -c user.name=x commit -m qwen4_exp-ple-mmap
```

then in the recipe's `build_args` replace `--apply-vllm-pr / "54129"` with
`--vllm-source-dir / /abs/path/to/vllm`. (`sha256(patch) = 3273ad05…`)

## Measured

DGX Spark, GB10, driver 580.173.02, onboard NVMe, fresh boot:

| | |
|---|---|
| Resident weights | 79.3 GiB (NVFP4 routed experts + BF16 rest) |
| PLE table | 47.7 GiB on disk — **0 B resident** |
| KV cache (bf16) | ~24 GiB → **~877k tokens**, 3.3× concurrency at 262k |
| Startup — `init engine` | ~43 s (after weight load) |
| Decode, 1 stream — prose / code | **~29 / ~35 tok/s** (MTP acceptance 65–95%) |
| Aggregate, 4 concurrent | **~42 tok/s** |
| Prefill | ~2.3–2.4k tok/s ; 128k prompt end-to-end ~56 s |
| 64k / 128k needle | retrieves correctly |
| Greedy determinism | byte-identical across runs |

Full tuning breakdown and the decode ceiling: **[RESULTS.md](RESULTS.md)**.
Memory is tight (~118/121 GiB, ~3 GiB swap under load) — good for one-to-few
long-context streams; heavy concurrency is memory-bound. GPU clock locking is
*not* a decode win (memory-bandwidth-bound); a 2000–2200 MHz `nvidia-smi -lgc`
cap is worth it only for stability against the GB10's throttle-lock behaviour.

## Checkpoint

[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
— NVIDIA ModelOpt NVFP4 (routed experts only). `config.json` declares
`Qwen4ExpForConditionalGeneration`, matching PR #53896. No `nvidia/` or
`RedHatAI/` NVFP4 of this model exists yet.

## License

Apache-2.0. The vendored patch is derived from vLLM PR #54129 (Apache-2.0).

# Qwen3.8-Flash-Next (NVFP4) — recipe add-on for spark-vllm-docker

Runs **Qwen3.8-Flash-Next** (Qwen4-preview, ~125B MoE + a ~51B PLE n-gram table,
hybrid Gated-DeltaNet / Qwen-Sparse-Attention) on **one DGX Spark / GB10** at full
262k context — via [`eugr/spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker),
which it does **not** modify or wrap. It's just a recipe you point `run-recipe.sh` at.

```
recipes/qwen3.8-flash-next-nvfp4.yaml     the recipe
patches/qwen38-flash-next-ple-mmap.patch  frozen vLLM diff — only needed if PR #54129 stops applying
tools/bench.py                            optional: prose / code / needle / concurrency bench
RESULTS.md                                the tuning: what helped, what didn't, the decode ceiling
```

## Use it

Needs a GB10 Spark (128 GB, driver **580.x**), Docker, ~140 GB free NVMe.

```bash
git clone https://github.com/bjarkebolding/spark-vllm-docker-helper

cd /path/to/your/spark-vllm-docker          # your existing checkout, untouched
./run-recipe.sh /path/to/spark-vllm-docker-helper/recipes/qwen3.8-flash-next-nvfp4.yaml \
    --solo --setup --earlyoom -d
#   --setup = build the image (~66 min) + download the model (~126 GiB) + serve
docker logs -f vllm_node                    # wait for "Application startup complete"
```

Drop `--setup` on later runs. Or copy the recipe into `spark-vllm-docker/recipes/`
and run `./run-recipe.sh qwen3.8-flash-next-nvfp4 --solo --setup` — same result.

OpenAI API on `localhost:8000/v1`. It's a thinking model — pass
`chat_template_kwargs: {"enable_thinking": false}` for short answers. After
startup, `python3 tools/bench.py --mode warmup` compiles the QSA kernels so the
first requests don't spike.

## What it builds

Official `vllm-project/vllm` @ `16d6c376` (2026-08-29 main) **+ PR #54129**
(`--apply-vllm-pr 54129`) — no third-party fork. PR #54129 adds `VLLM_PLE_MMAP`,
which serves the 47.7 GiB PLE table straight from the checkpoint via `np.memmap`
instead of GPU/RAM. That, plus an NVFP4 body (~79 GiB) and the GB10-specific flags
in the recipe, is what makes the model fit and run at TP1.

The recipe carries the tuning too — `VLLM_PLE_MMAP_READAHEAD`, `WORKERS`,
`--no-enable-flashinfer-autotune`, `gpu-memory-utilization`. See **RESULTS.md**.

## Measured (single GB10, fresh boot)

~29 tok/s decode (prose) · ~35 (code) · ~42 (4 concurrent) · ~877k KV tokens ·
~43 s `init engine` · 64k+128k needle retrieval passes · greedy-deterministic.

## If PR #54129 gets rebased and stops applying

Apply `patches/qwen38-flash-next-ple-mmap.patch` (`sha256 3273ad05…`,
`git diff 16d6c376..PR-head`) to a local `vllm` checkout at `16d6c376` and, in the
recipe's `build_args`, swap `--apply-vllm-pr 54129` for `--vllm-source-dir <path>`.

## Checkpoint

[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
— NVIDIA ModelOpt NVFP4, routed experts only; declares
`Qwen4ExpForConditionalGeneration` (matches PR #53896). No `nvidia/` or
`RedHatAI/` NVFP4 of this model exists yet.

## License

Apache-2.0. The vendored patch is derived from vLLM PR #54129 (Apache-2.0).

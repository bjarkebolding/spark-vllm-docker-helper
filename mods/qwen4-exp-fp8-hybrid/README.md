# qwen4-exp-fp8-hybrid

Stores Qwen3.8-Flash-Next's **dense side layers** as blockwise fp8 while the routed
experts stay NVFP4. The dense side (Gated-DeltaNet in/out projections, QSA q/k/v/o,
shared-expert MLP — ~15 GiB of bf16, 300 tensors) is read in full on every decoded
token, so halving it is a real decode win.

**Measured on one GB10 (non-streaming) vs the plain NVFP4 recipe:**

| | NVFP4 | fp8-hybrid |
|---|---|---|
| prose decode | ~27 tok/s | **~32** |
| code decode | ~32 tok/s | **~36** |
| KV pool | 534k tokens | **594k** |
| determinism / 35k needle | pass | pass |

~+18% decode, quality-neutral (3.5% worst-case per-tensor quant error; arithmetic,
sorting, code, word-problems all correct).

## Use

Serve with the paired recipe (it sets `VLLM_FP8_HYBRID=1` and `VLLM_USE_DEEP_GEMM=0`):

```bash
./run-recipe.sh /abs/path/recipes/qwen3.8-flash-next-fp8hybrid.yaml --solo --setup --earlyoom -d \
  --apply-mod /abs/path/mods/qwen4-exp-fp8-hybrid
```

The mod's `run.sh` builds the hybrid checkpoint on first run (`make_hybrid.py`,
~2 min, ~8 GiB — hardlinks the unchanged shards) as a sibling snapshot
`…/snapshots/<hash>-fp8hybrid` plus a stable `…/snapshots/fp8hybrid` symlink that the
recipe points at. Then it installs the shim.

## How it works

- `make_hybrid.py` — rewrites the 300 dense-side bf16 projections to blockwise
  fp8-e4m3 (DeepSeek layout: fp8 `weight` + fp32 `weight_scale_inv`, 128×128 blocks).
  Quantizer by [@Saren-Arterius](https://github.com/Saren-Arterius) (Apache-2.0).
- `vllm_fp8_hybrid_modelopt.py` — patches `ModelOptNvFp4Config` so the fp8 layers
  route to vLLM's blockwise `Fp8Config` instead of the unquantized bf16 path;
  redirects `qsa.py`'s `without_modelopt_fp4` for the QSA `qkv_proj`.
  Port of Saren-Arterius's shim via blazux/qwen3.8-Flash-DGX (Apache-2.0).

## Why `VLLM_USE_DEEP_GEMM=0` is required

Without it, vLLM routes blockwise-fp8 to `DeepGemmFp8BlockScaledMMKernel`, which emits
constant-garbage output on sm_121 (GB10). Disabling DeepGEMM selects the CUTLASS
blockwise-fp8 kernel, which is correct and fast. The recipe sets this.

## Safety

`apply.py` is idempotent and shape-guarded (aborts if the stock code shape isn't what
it expects). The shim is a runtime no-op unless `VLLM_FP8_HYBRID=1`.

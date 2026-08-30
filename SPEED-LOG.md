# Qwen3.8-Flash-Next NVFP4 on one GB10 — speed log

Running record for the 30-min SPEED-LOOP cron (session job `09a2bbc1`).
Benchmark = non-streaming wall clock (`scratchpad/bench.py`); a streaming client reads ~2x low.

## Baseline (shipped recipe `qwen3.8-flash-next-nvfp4.yaml` + `mods/qwen4-exp-ple-pinned`)

| metric | value |
|---|---|
| prose decode, single-stream | ~27 tok/s |
| code decode, single-stream | ~32 tok/s |
| concurrent | ~80 @ 4 streams · ~122 @ 8 |
| prefill | ~2450 tok/s, flat to 120k |
| KV pool | ~535k tokens (gpu-mem 0.80) |
| MTP acceptance | ~2.4 mean, pos-0 ~0.85 / pos-1 ~0.58 |
| GPU during decode | 2515 MHz, ~31 W, 91% util → launch/sync-bound, not compute- or purely BW-bound |

Per 0xBakeer's analysis: ~20 ms of the ~35 ms decode step is fixed overhead
(per-step PLE gather CPU op + H2D sync ~4 ms; QSA-state attn-metadata rebuild between
MTP draft steps; piecewise-graph boundaries because PLE mmap can't be in a full graph).

## Tried — did NOT help (do not re-run without a new angle)

| lever | result |
|---|---|
| SGLang single-Spark | no lossless TP1 recipe exists; single-Spark = ~18-19 prose (worse) |
| llama.cpp | ~25-28 prose, not faster; ngram-mod only helps file-rewrite tasks |
| recipe retune (WORKERS 1→24, READAHEAD →16384, gpu-mem 0.88→0.80) | robustness only, single-stream unchanged. KEPT (shipped). |
| `mods/qwen4-exp-ple-pinned` (pinned H2D staging) | prefill/concurrency only, no single-stream change. KEPT (shipped, optional). |
| FLA shared-mem gate 102400→101376 | no-op — our vLLM has a CUDA GDN *decode* kernel; the gate only affects Triton *prefill* |
| fp8-hybrid checkpoint (blazux `fp8_convert` + `vllm_fp8_hybrid_modelopt` shim) | BROKEN — vLLM 0.28 routes blockwise-fp8 to DeepGemm; incompatible weight-load with blazux's fp32-scale converter → constant-garbage output, 0% MTP. E8M0 toggle no fix. Reverted + deleted. |
| `num_speculative_tokens: 3` | wash — pos-2 acceptance 0.33, extra draft-step cost cancels it |
| `custom_ops: ["+rms_norm","+silu_and_mul"]` | ignored — sm_121 platform default forces `rms_norm=['native']` |
| `cudagraph_capture_sizes` incl 3/5/6 (MTP verify batch) | no measurable effect |
| GPU clock lock | can't — needs sudo password |

## 2026-08-30 tick — MAJOR LEAD: Saren-Arterius AutoRound hybrid

`Saren-Arterius/qwen3.8-Flash-DGX-AutoRound` reports **~49 tok/s single-stream on GB10**
(1.8x our 27), vLLM, pre-built checkpoints:
`Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid` (68 shards, INT4 experts + fp8 side +
int8 lm_head) + `Saren/Qwen3.8-Flash-Next-ple-table-fp8` (33 shards, the PLE table).

Their recipe: W4A16 AutoRound on the **MoE experts** (Marlin kernel — faster than
FLASHINFER_CUTLASS NVFP4 on sm_121), blockwise-fp8 side layers (GDN/QSA/shared, **Triton
fallback**), int8 GPTQ lm_head, MTP=3.

**KEY: they set `VLLM_USE_DEEP_GEMM=0`** — DeepGEMM fails on sm_121 (`CUDA_ERROR_LAUNCH_FAILED`).
This session's fp8-hybrid produced garbage precisely because DeepGEMM was active — I only
tried `VLLM_USE_DEEP_GEMM_E8M0=0`, not the full disable.

### RESULT — fp8-hybrid + `VLLM_USE_DEEP_GEMM=0` → KEEP ✓

`Selected CutlassFp8BlockScaledMMKernel`. Coherent, deterministic 3/3, 35k needle pass,
quality probes correct (arithmetic/sort/code/word-problems).

| | NVFP4 baseline | fp8-hybrid + DEEP_GEMM=0 |
|---|---|---|
| prose | ~27 | **~32** (31.5-32.6) |
| code | ~32 | **~36** (34-37) |
| KV | 534k | **594k** |
| MTP accept | 2.4 | 2.2-2.4 |

**Shipped** as `recipes/qwen3.8-flash-next-fp8hybrid.yaml` + `mods/qwen4-exp-fp8-hybrid`
(the mod builds the checkpoint on first run). Server switched to this config.

### NEXT: Saren's full W4A16 (INT4 experts via GPTQ-Marlin) → target ~49 tok/s, MTP=3

**Tick 3: download kicked** (`scratchpad/saren-dl.log`, pid running):
`Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid` + `Saren/Qwen3.8-Flash-Next-ple-table-fp8`
(~122 GiB total, ~30-60 min). Repo cloned to `scratchpad/saren/`.

Saren measured **~49 tok/s single-stream (Q&A 46 / Code 49 / JSON 58 / Math 49), MTP=3**,
weights ~71 GiB, on GB10. Base = `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound` (int4 experts).

**Precision map:** experts int4 GPTQ-Marlin g128 · lm_head + MTP head int8 GPTQ-Marlin ·
GDN in/out + QSA qkvo + shared-expert = fp8 blockwise (we already do this) · embed/norms/
gates/fc_hidden bf16 · PLE fp8 mmap · KV bf16.

**Our-vLLM compat CONFIRMED this tick:**
- checkpoint `config.json` → `quant_method: gptq` → our vLLM resolves `gptq`/`gptq_marlin`
  to **`AutoGPTQConfig`** — exactly what Saren's `src/vllm_fp8_hybrid.py` shim wraps. ✓
- `dynamic` rules supported: `vllm/.../quantization/utils/gptq_utils.py::get_dynamic_override`
  reads `config.dynamic`. ✓ (the `-:` excludes for linear_attn/self_attn/shared_expert/ple/
  embed/gate/fc_hidden/layers.48 will apply)
- `ParallelLMHead(` at `qwen4_exp/nvidia/model.py:628` and `mtp.py:394` — the sed points for
  the int8-head patch (patch 3). Without the mtp.py half, MTP>=3 crashes at load.

**Integration steps (next tick, once download done):**
1. Symlink `Saren/...ple-table-fp8` shards into the Saren checkpoint snapshot dir (our
   PR#54129 ple_mmap globs the model dir for `*.safetensors` matching the ngram regex; it
   has no `VLLM_PLE_MMAP_DIR`). Verify `config.json` doesn't have the ngram in a
   `.index.json` that would try to load it as a weight (Saren's strip_ngram_index handles
   this on their side — check ours).
2. New mod `qwen4-exp-int8-lmhead`: sed `quant_config=quant_config` into both
   `ParallelLMHead(` call sites (model.py:628, mtp.py:394). Guarded.
3. New mod `qwen4-exp-fp8-hybrid-gptq` (or extend the existing fp8-hybrid mod): install
   Saren's `src/vllm_fp8_hybrid.py` (wraps `AutoGPTQConfig`, not ModelOptNvFp4Config),
   hook the qsa.py `without_modelopt_fp4` redirect the same way.
4. Recreate `qwen4-exp-fla-gb10` mod (FLA shmem gate — no-op for our decode but Saren
   ships it; the num_warps pin is a real correctness fix).
5. Variant recipe `qwen3.8-flash-next-w4a16.yaml`: model = Saren checkpoint path,
   `VLLM_FP8_HYBRID=1`, `VLLM_USE_DEEP_GEMM=0`, `VLLM_PLE_MMAP=1`, `num_speculative_tokens=3`,
   `--kv-cache-memory-bytes 20g` + `gpu-memory-utilization 0.01` (Saren's deterministic
   sizing to avoid unified-pool oversubscription freezes).
6. Serve → validate coherence + determinism + 35k needle + quality probes → benchmark.
   KEEP if clearly faster + quality intact, else REVERT to the fp8-hybrid config.

RISK: GPTQ-Marlin int4 experts + the bf16 MTP-draft-MoE coexistence (this session
`--moe-backend marlin` errored on "unquantized BF16 MoE" — but Saren's checkpoint has the
MTP layer's experts int4 too, or excludes via `layers.48`; TBD). If experts won't load,
fall back to: keep NVFP4 experts + just add int8 lm_head on top of the current fp8-hybrid
(smaller win, ~1.3 GiB less + a bf16 GEMV/token removed).

Patches they need (port to our vLLM 0.28 + PR#54129): FLA shmem gate (have it), int8 lm_head
(model.py+mtp.py `quant_config` kwarg to ParallelLMHead), AutoGPTQ+Fp8 dispatch shim,
`VLLM_USE_DEEP_GEMM=0`.

## Open plans (loop: pick up here)

1. **Marlin INT4 (W4A16) on the GDN in/out projections + shared experts.** Different kernel
   from the broken fp8/DeepGemm path — Marlin is the INT4 path reported working on sm_121
   (`VLLM_MARLIN_USE_ATOMIC_ADD=1`). GDN is 36/48 layers and the dominant per-token read.
   Build: `llm-compressor` W4A16 g128 on those tensor families (regex like the fp8 converter
   but exclude QSA q/k/v/o first — validate GDN-only, then add). Integrate: compressed-tensors
   config group OR a shim like `vllm_fp8_hybrid_modelopt` but routing to `CompressedTensorsW4A16`.
   Validate: coherence + determinism + 35k needle + a ~10-question quality probe. Risk: same
   modelopt-`ignore` routing problem as fp8; GDN below INT8 may hurt (recurrent state is fragile —
   keep the GDN *conv1d / A_log / dt_bias* BF16, only the big matmul projections go INT4).
   Est: ~1-2 days, payoff maybe +10-20% (0xBakeer says not fully BW-bound).

2. **Structural PLE prefetch.** `ple_mmap.py` still does a blocking `ids.detach().to("cpu",
   non_blocking=False)` per step. Pin that D2H + use a CUDA event instead of a full stream sync;
   optionally kick the CPU gather from an earlier splitting-op boundary to overlap with layers 0-1.
   Contained mod. Est: ~1 ms/step, maybe +1 tok/s.

3. **Watch upstream:** vLLM PR #54371 (UVA PLE — could de-split the graph → FULL_AND_PIECEWISE),
   #54070 (disk offload), and any qwen4_exp / QSA fused-multi-step-draft work. Check each tick.

## Cycle log

_(loop appends: date · change · prose/code tok/s · KV · MTP · verdict)_

- 2026-08-30 · session baseline established · 27 / 32 · 535k · 2.4 · —
- 2026-08-30 tick · fp8-hybrid dense side + VLLM_USE_DEEP_GEMM=0 (CUTLASS kernel) · 32 / 36 · 594k · 2.3 · **KEEP, shipped**

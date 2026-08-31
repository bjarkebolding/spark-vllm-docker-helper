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

## Tick 4 (2026-08-30) — W4A16 integration built + load test launched

Download done: Saren W4A16 checkpoint (68 shards) + config `quant_method: gptq`, ple
geometry matches RadixArk (ple_layer_ids [2], embed_dim 2560, split 128). Saren's
ple-table-fp8 download was incomplete/mis-organized → instead **symlinked RadixArk's 10
`model-plefp8-*.safetensors` into the Saren snapshot** (same fp8 table, tensor names the
PR#54129 ple_mmap regex expects).

Built + dry-run-tested 3 mods + recipe (committed `dcc4004`):
- `qwen4-exp-int8-lmhead` — `quant_config=vllm_config.quant_config` into ParallelLMHead
  at model.py:628 + mtp.py:394 (without the mtp half, MTP>=3 crashes at load)
- `qwen4-exp-w4a16-gptq-fp8` — Saren's `vllm_fp8_hybrid.py` (wraps `AutoGPTQConfig`,
  routes the 300 F8_E4M3 dense layers to Fp8Config). Our vLLM: `gptq`→`AutoGPTQConfig`
  (matches), `dynamic` rules supported (`gptq_utils.get_dynamic_override`), qsa qkv_proj
  routes through AutoGPTQConfig for a gptq ckpt so no qsa hook needed.
- `qwen4-exp-fla-gb10` — recreated (FLA shmem gate + num_warps pin)
- `recipes/qwen3.8-flash-next-w4a16.yaml` — VLLM_FP8_HYBRID=1, VLLM_USE_DEEP_GEMM=0,
  VLLM_MARLIN_USE_ATOMIC_ADD=1, MTP=3

**Load test #1 failed**: PLE symlinks used absolute host paths (`/home/bolding/...`),
invalid in the container (`/root/...` mount). Fixed with relative symlinks
(`../../../models--RadixArk--.../snapshots/.../model-plefp8-N.safetensors`), verified
resolvable in the container. Retrying (`relaunch9.log`). Shim active: "300 blockwise-fp8 layers
detected". Known risks: MTP experts are bf16 in the checkpoint (`mtp.layers.N.mlp.experts
.N.{gate,up,down}_proj.weight`, unfused vs RadixArk's fused `gate_up_proj`) — the
bf16-MTP-MoE + GPTQ-Marlin-main coexistence + unfused loading may break. If it fails,
next tick reverts to the fp8-hybrid config (32/36) and falls back to int8-lmhead-only.

## Tick 4/5 — W4A16 LOADS AND WORKS: ~40 prose / ~48 code (+47%)

Saren checkpoint (Intel W4A16 int4 experts + int8 head + fp8 side) loaded on our vLLM
0.28 + PR#54129 with the 4 mods. `MarlinExperts` + `MarlinLinearKernel` (head) +
`CutlassFp8BlockScaledMMKernel` (dense side); the bf16 MTP-draft MoE coexists on
FlashInfer. **The load risk didn't materialize.**

| | baseline | fp8-hybrid | **W4A16 (ATOMIC_ADD=1)** |
|---|---|---|---|
| prose | 27 | 32 | **39-41** |
| code | 32 | 36 | **44-49** |
| KV | 534k | 594k | **741k** |
| needle 35k | pass | pass | **pass** |
| quality probes | — | — | **7/7** (arith/physics/calc/code/bio/geo/sort) |
| determinism | pass | pass | **FAIL — Marlin atomic-add jitter** |

Only failure: byte-determinism. `VLLM_MARLIN_USE_ATOMIC_ADD=0` tested → **no fix, same
speed** — the nondeterminism is deeper (Marlin MoE reduction order and/or QSA topk).
Set back to =1 (research-blessed for sm_121). Final validation (ATOMIC_ADD=1):
capital ✓, 35k needle ✓, 6/7 quality probes (7th was a max_tokens artifact of the probe,
not the model), **38-41 prose / 48-51 code**, MTP 2.5.

**SHIPPED as `recipes/qwen3.8-flash-next-w4a16.yaml` + 4 mods** (`9aaca9f`). Server on it.
Non-deterministic by design — the README table marks it so; fp8-hybrid stays the
deterministic option.

## Tick 6 — research: nothing faster than ~49 exists on a single Spark. QSA-exact-topk test.

WebSearch: no new lever; ~40/48 matches Saren's ceiling. `PLE_PREFETCH` confirmed not
worth it (Saren appendix); RAM-disk table won't fit (44 GB table, ~14 GB free RAM).

Confirmed our qsa.py forces `persistent_topk` on sm_121 (`use_cooperative_topk` gated
`not is_device_capability_family(120)`) — that's the nondeterminism source (vllm#51782).
Added `mods/qwen4-exp-qsa-exact-topk` (blazux patcher, `VLLM_QSA_EXACT_TOPK=1` → exact
torch.topk). Relaunching w4a16 with it (`relaunch12.log`). Expected: deterministic;
blazux says it "costs prefill/decode time" so watch the tok/s.

### RESULT: QSA_EXACT_TOPK does NOT fix determinism (4 distinct outputs).
Decode unchanged (38-39 prose / 48-53 code), prefill ~1840 (down ~8% from ~2000).
Combined with the earlier `ATOMIC_ADD=0` null result → **the nondeterminism is the
Marlin MoE expert reduction (non-associative float accumulation), not QSA and not
atomic-add. It is inherent to the int4-Marlin path and not recoverable via flags.**

**FINAL: w4a16 ships non-deterministic (`recipes/qwen3.8-flash-next-w4a16.yaml`, ~40/48).
fp8-hybrid (~32/36) is the deterministic tier. nvfp4 (~27/32) is the conservative
baseline.** Keeping `VLLM_QSA_EXACT_TOPK=1` in the w4a16 recipe anyway — it's a
correctness win (persistent_topk can drop real top-k candidates, vllm#51782) at ~zero
decode cost. `mods/qwen4-exp-qsa-exact-topk` kept as a general opt-in tool.

Loop status: 27->40 prose (+48%), 32->48 code (+50%) since it started. At the known
single-Spark ceiling (Saren ~49). Remaining levers all marginal (<2 tok/s) or need
upstream vLLM work (fused multi-step QSA draft, PLE in full graphs). Expect
"no new lever" ticks from here unless something new lands upstream.
- If deterministic AND speed holds (>=38 prose) → w4a16 becomes the single recommended recipe.
- If deterministic but slow → revert the env, note the cost, w4a16 stays non-det + fp8hybrid is the det option.
- If still nondeterministic → source is Marlin MoE reduction, revert, note it.

(superseded) Next lever (for determinism, not speed): port blazux/Saren's QSA-exact-topk patch
(`patch_qsa_exact_topk.py` — torch.topk over visible cols instead of the nondeterministic
`persistent_topk`, vllm#51782) as a guarded mod, test whether it makes w4a16 deterministic.
Our vLLM has no `VLLM_QSA_EXACT_TOPK` env so it must be a source patch.



## Tick 7 (2026-08-31) — no new decode lever

Research: checked Saren's latest commits (Aug 30: PLE_PREFETCH experiment, README fixes,
"update numbers" — no recipe change, no number above 49) and their `serve-intel-ar.sh`.
**Our w4a16 config matches Saren's** on every decode-relevant flag (MTP=3, DEEP_GEMM=0,
MARLIN_ATOMIC_ADD=1, FLASHINFER_SAMPLER=1, autotune off, PIECEWISE). Only diffs are
boot-only (`LOAD_FORMAT=fastsafetensors`) or cosmetic (tool parser).

**We are at the published single-Spark ceiling: ~40 prose / ~48 code = Saren's ~49.**

Parked candidates (none is a single-stream *decode* win):
- **Prefix caching** (`patch_mamba_align_split.py` — scheduler.py + mamba_hybrid.py core
  patches). Big TTFT/throughput win for repeated system prompts = the 24/7-agent workload,
  NOT single-stream decode. High risk: core-scheduler anchor drift on our vLLM 0.28 + the
  #54173 GDN-prefix-cache crash. Would need careful cache-hit correctness validation
  (greedy output changed on hits before Saren's fix). Defer unless the user asks for
  multi-agent throughput specifically.
- **MTP=2 vs 3 A/B on w4a16** — cheap (~1 restart) but earlier k=3 was a wash on NVFP4;
  low expected value. Run only on an otherwise-idle tick.
- **aixiaoma/Qwen3.8-Flash-Next-W4A16** — more aggressive (QSA also int4). No GB10 number,
  ~168 GiB download, integration risk. Only if a tick has nothing else.
- **Upstream**: watch for fused multi-step QSA draft (would fix the per-draft-step metadata
  rebuild) and PLE-in-full-graph (#54371). Check each tick.

Server left on w4a16 (healthy, ~41 tok/s). No restart this tick.

## LOOP STATUS: at the ceiling, idling (2026-08-31)

Loop delivered **27 -> 40 prose (+48%) / 32 -> 48 code (+50%)** over ticks 1-6, landing at the
published single-Spark ceiling (Saren ~49). Ticks 7-9 found no new decode lever. From here
the loop keeps the server healthy and re-checks each tick for:
- a new/faster Flash-Next checkpoint on HF
- vLLM PRs merging: #54371 (UVA PLE / de-split graph), fused multi-step QSA draft, PLE-in-full-graph
- new single-Spark forum recipes above ~49 tok/s
It will only restart the server for a change with a real expected decode win.

## Cycle log

_(loop appends: date · change · prose/code tok/s · KV · MTP · verdict)_

- 2026-08-30 · session baseline established · 27 / 32 · 535k · 2.4 · —
- 2026-08-30 tick · fp8-hybrid dense side + VLLM_USE_DEEP_GEMM=0 (CUTLASS kernel) · 32 / 36 · 594k · 2.3 · **KEEP, shipped**
- 2026-08-30 tick · W4A16 int4 experts (Marlin) + int8 head + fp8 side, MTP=3 · **40 / 48** · 747k · 2.6 · **KEEP, SHIPPED** (non-deterministic — Marlin MoE reduction, inherent)
- 2026-08-31 tick 7 · research — no new decode lever; at Saren's ~49 ceiling · — · — · — · idle
- 2026-08-31 tick 8 · research (PR#54371 still Draft) + MTP=2 A/B on w4a16 → **WASH** (36.5/49.3 median vs ~38/50 at MTP=3). REVERT to MTP=3.


## Tick 10 (2026-08-31) — no new lever; prefix-cache feasibility checked

Prefix caching = the top remaining lever for the **concurrent** path (24/7 agents sharing
a system prompt: TTFT seconds -> ~0.3s, prefill compute saved). Feasibility:
- `mamba_hybrid.py:122` seed-bug anchor — **EXACT match** to blazux's patch. ✓
- `scheduler.py:411` anchor has **drifted** — our `_mamba_block_aligned_split` now has
  Eagle-specific handling that blazux/Saren's version didn't; can't apply their patch
  verbatim. Needs a fresh read of whether our newer scheduler still uses the min-across-
  groups block_size wrongly here, or already aligns to mamba's 1600.
- Also unknown: whether #54173 (GDN prefix-cache crash on sm_121) still reproduces on our
  vLLM + the w4a16 fp8 side layers, or is fixed upstream.
- vLLM has no `VLLM_MAMBA*`/`VLLM_PREFIX*` env to toggle the align mode — it's `--mamba-cache-mode`.

Not a same-tick change (evolved core-scheduler code + needs a load test to check #54173).
Deferred. It's concurrency, not single-stream decode. Do it only on a tick with appetite
for a ~30-min investigation, and only if the user asks for multi-agent throughput.

No restart. Server healthy on shipped w4a16 (~36-40 tok/s this cycle).

- 2026-08-31 tick 9 · **no new lever** — nothing new upstream/HF; #54371 still Draft. Idling.


## Tick 11 (2026-08-31) — no new lever; aixiaoma crossed off

Checked `aixiaoma/Qwen3.8-Flash-Next-W4A16`: int4 experts + **int4 QSA q/k/v/o**, but
**GDN kept BF16** and lm_head BF16. GDN is 36/48 layers and the dominant per-token read —
BF16 GDN means MORE bytes/token than our Saren-based w4a16 (fp8 GDN + int8 head). Likely
**slower**, not faster; 8×3090-only, no GB10 data, 168 GiB. **Not worth pursuing.**

Our w4a16 (Saren base) is the best-quantized decode config that exists for GB10:
experts int4-Marlin · GDN/QSA/shared fp8-CUTLASS · lm_head int8-Marlin · PLE fp8 mmap.

### Top remaining speed lever (multi-tick, risky): INT4 the GDN *projections*

The GDN in/out matmul projections (`linear_attn.in_proj_qkv / in_proj_z / out_proj`,
36 layers) are currently fp8. They are plain linear matmuls — the recurrent state
(`conv1d`, `A_log`, `dt_bias`) is separate and small and would stay BF16. Marlin int4 on
just the projections would cut the dominant read further. Neither Saren nor aixiaoma did
this — likely a quality concern on the GDN path, or just conservatism.
Plan: start from Intel's base checkpoint, add `linear_attn.(in_proj_qkv|in_proj_z|out_proj)`
to the int4 `dynamic` set (leave conv1d/A_log/dt_bias out), keep everything else as Saren's.
Build with the same GPTQ/AutoRound path. Validate HARD: perplexity vs bf16 + full quality
suite + long-context needle (GDN is the long-range memory). Est: ~1 day, payoff maybe
+10-20% if quality holds, could be a quality bust. Do only if the user wants to push further.

No restart. Server healthy on shipped w4a16 (~41 tok/s this cycle).

- 2026-08-31 tick 10 · **no new lever** — prefix-cache anchors partly drifted; deferred. Idling.
- 2026-08-31 tick 11 · **no new lever** — aixiaoma crossed off. INT4-GDN-projections = last speed lever (multi-tick). Idling.
- 2026-08-31 tick 12 · **no new lever** — GDN-int4 confirmed not a same-tick change (custom RTN checkpoint, quality risk, Saren+aixiaoma both avoided it). Needs a dedicated session + perplexity harness. Idling, server healthy (~38 tok/s).

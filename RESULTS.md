# Results — Qwen3.8-Flash-Next on one DGX Spark (GB10)

**2026-08-31.** Hardware: NVIDIA GB10 / DGX Spark, 128 GB unified LPDDR5X (~273 GB/s),
sm_121, aarch64, driver 580.173.02. All decode figures are **non-streaming wall clock**
(a line-iterating SSE client adds ~40 ms/token of its own overhead and reads ~2× low).

## Headline

| config | prose | code | concurrent (×8) | KV pool | prefill | deterministic |
|---|---:|---:|---:|---:|---:|:---:|
| `nvfp4` (tuned baseline) | ~27 | ~32 | ~122 tok/s | 522k tok | ~2450 tok/s | yes |
| `fp8hybrid` | ~32 | ~36 | — | 594k tok | ~2400 tok/s | yes |
| **`w4a16` (shipped)** | **~40** | **~48** | **~196 tok/s** | **747k tok** | ~1750 tok/s | **no** |

**Net: +48% prose, +50% code, +40% KV** over the starting point, on stock open-source
software (official vLLM + one merged-track PR, no fork). `w4a16` matches the best
published single-Spark figure (Saren-Arterius: Q&A 46 / Code 49 / JSON 58 / Math 49).

Model: ~180 B params (125 B compute + 51 B n-gram table, ~6 B active/token), 512-expert
MoE top-10, hybrid Gated-DeltaNet + Qwen-Sparse-Attention, MTP draft head, 262 k context.

---

## `w4a16` measured in detail (2026-08-31, non-streaming wall clock, MTP=3)

### Single-stream decode by task shape

| task | tok/s |
|---|---:|
| prose / long-form explanation | ~40 |
| JSON / structured output | ~43 |
| Q&A / factual | ~45 |
| code — refactor existing | ~47 |
| math / multi-step (thinking on) | ~51 |
| code — write new | ~55 |

Range ~40–55; structured and code-heavy output decodes fastest (the MTP draft
predicts predictable text better).

### Decode vs context depth (single stream)

| prompt | decode | TTFT |
|---:|---:|---:|
| ~1 k | ~37 tok/s | 0.8 s |
| ~5 k | ~37 | 2.5 s |
| ~18 k | ~36 | 10 s |

Decode is flat with context; only TTFT (prefill) grows.

### Prefill throughput

| prompt | tok/s | wall |
|---:|---:|---:|
| 2 k | 1840 | 1.1 s |
| 8 k | 1900 | 4.2 s |
| 32 k | 1810 | 18 s |
| 96 k | 1760 | 55 s |
| 200 k | 1660 | 120 s |

Small prompts ramp up: ~300 tok → ~1000 tok/s, ~1.2 k → ~1600.

### Concurrency (aggregate)

| streams | aggregate | per stream |
|---:|---:|---:|
| 1 | 52 tok/s | 52 |
| 2 | 80 | 40 |
| 4 | 127 | 32 |
| 8 | **196** | 25 |
| 16 | 190 | 12 |

Scales cleanly to ~200 tok/s aggregate at 8 concurrent streams, then flattens.

### Thinking on vs off (same question)

`enable_thinking:false` ~56 tok/s; `true` ~50 tok/s per token but often fewer
total tokens to a good answer — worth it only for genuinely hard reasoning.

---

## How

### 1. Base — fit the model at all (`nvfp4`)

The NVFP4 checkpoint is ~126 GiB; 47.7 GiB of that is the PLE n-gram embedding table — a
pure lookup a token reads ~a few KB from, never multiplied.

- **Official `vllm-project/vllm` @ `16d6c376` + PR #54129** (`VLLM_PLE_MMAP`): serves the
  PLE table from NVMe via `np.memmap` / page cache instead of resident memory. No fork.
- Resident weights ~82 GiB, leaving room for a ~522 k-token KV cache in the 128 GB pool.
- **GB10 tuning:** `VLLM_PLE_MMAP_WORKERS 1→24` (parallelize cold shard reads),
  `READAHEAD 2048→16384`, `gpu-memory-utilization 0.88→0.80` (page-cache headroom for the
  table). `VLLM_PLE_MMAP_PINNED=1` — pinned-host gather staging (native to PR #54129).
- Also required on sm_121: `--no-enable-prefix-caching` (#54173), `--no-enable-flashinfer-autotune`
  (FI #4003), bf16 KV (QSA refuses fp8), `cudagraph_mode PIECEWISE` (the mmap gather is a
  splitting op).

**Result: ~27 prose / ~32 code**, 35k+128k needle pass, greedy-deterministic. This is the
NVFP4 ceiling — every independent single-Spark report lands here.

### 2. Quantize the dense side (`fp8hybrid`, +18% / +12%)

~15 GiB of the model (GDN in/out projections, QSA q/k/v/o, shared-expert MLP — 300
tensors) was still bf16 and is read in full every token.

- **`make_hybrid.py`** (@Saren-Arterius quantizer, Apache-2.0) rewrites those 300 tensors
  to blockwise fp8-e4m3 (128×128, DeepSeek layout). Worst per-tensor error 3.5%. Runs in
  ~2 min, hardlinks the unchanged shards (~8 GiB new).
- **`mods/qwen4-exp-fp8-hybrid`** — a shim patching `ModelOptNvFp4Config` to route those
  layers to vLLM's blockwise `Fp8Config`.
- **The unlock: `VLLM_USE_DEEP_GEMM=0`.** Without it, vLLM routes blockwise-fp8 to
  `DeepGemmFp8BlockScaledMMKernel`, which emits **constant garbage** on GB10 (an identical
  first attempt failed exactly here). Disabling DeepGEMM selects the CUTLASS
  blockwise-fp8 kernel — correct and fast.

**Result: ~32 prose / ~36 code**, 594 k KV, deterministic, quality-neutral (3.5% error;
arithmetic/sort/code/word-problems all correct).

### 3. INT4 the experts (`w4a16`, +48% / +50% total)

- Base: **`Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid`** (from
  `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`). Precision map:
  - routed experts (512/layer × 48): **INT4 GPTQ-Marlin** g128
  - `lm_head` + MTP draft head: **INT8 GPTQ-Marlin**
  - GDN / QSA / shared-expert projections: fp8 blockwise (CUTLASS, as `fp8hybrid`)
  - PLE table: fp8, mmap from NVMe · KV: bf16 · norms/gates/embeds: bf16
- Ported to our vLLM 0.28 + PR #54129 with 4 guarded mods:
  - `qwen4-exp-w4a16-gptq-fp8` — @Saren-Arterius `AutoGPTQConfig` shim; also fetches the
    fp8 PLE-table shards from RadixArk and links them into the Saren checkpoint dir
  - `qwen4-exp-int8-lmhead` — `quant_config` kwarg into `ParallelLMHead` in `model.py`
    *and* `mtp.py` (without the second, MTP≥3 crashes at load)
  - `qwen4-exp-fla-gb10` — FLA shared-mem gate + `chunk_delta_h` num_warps pin
  - `qwen4-exp-qsa-exact-topk` — `VLLM_QSA_EXACT_TOPK=1`, exact `torch.topk` (correctness;
    `persistent_topk` can drop candidates on sm_121, vllm#51782)
- Env: `VLLM_FP8_HYBRID=1`, `VLLM_USE_DEEP_GEMM=0`, `VLLM_MARLIN_USE_ATOMIC_ADD=1`, MTP=3.
- Resident weights drop to ~71 GiB → **747 k-token KV pool**.

**Result: ~40 prose / ~48 code**, MTP mean acceptance 2.4–2.9, 35 k needle pass, quality
probes all correct (arithmetic, calculus, sorting, code, biology, geography), matches
Saren's ~86% tournament score.

**Not deterministic.** Greedy output varies run-to-run — the Marlin MoE expert reduction
is a non-associative float accumulation. Not fixable via `VLLM_MARLIN_USE_ATOMIC_ADD=0`
or `VLLM_QSA_EXACT_TOPK=1` (both tested). The variation is *within correct answers*
("kernel jitter"), not wrong output. `fp8hybrid` (32/36) remains the deterministic tier.

---

## What did not work

| tried | result |
|---|---|
| SGLang, single Spark | no lossless TP1 recipe exists; the one datapoint is ~18–19 prose (worse). All 47–70 tok/s SGLang numbers are 2× Spark. |
| llama.cpp | ~25–28 prose (not faster). Its 88 tok/s is `ngram-mod` on file-reproduction tasks only. |
| fp8-hybrid via DeepGEMM | constant garbage on sm_121 — fixed only by `VLLM_USE_DEEP_GEMM=0` → CUTLASS. |
| FLA shared-mem gate (102400→101376) | no-op for our decode — our vLLM uses a CUDA GDN *decode* kernel; the gate only affects the Triton *prefill* path. |
| `num_speculative_tokens` 3 vs 2 | wash on every checkpoint (pos-2 acceptance ~0.3; extra draft-step cost cancels the gain). |
| `custom_ops: ["+rms_norm","+silu_and_mul"]` | silently overridden — sm_121 platform default forces `rms_norm=['native']`. |
| `cudagraph_capture_sizes` incl. 3/5/6 (MTP verify batch) | no measurable effect. |
| GPU clock lock | needs sudo; also documented as not a decode win (bandwidth/launch-bound). |
| `aixiaoma/Qwen3.8-Flash-Next-W4A16` | keeps GDN in bf16 (36/48 layers, the dominant read) → would be *slower*; not pursued. |
| **INT4 the GDN projections** (RTN, on top of `w4a16`) | **built + served + FAILED.** RTN gives ~15% weight error on those matmuls (vs ~1–2% for fp8); output was garbage ("sumsum…"), 0/10 probes, needles fail. The GDN linear-attention path can't take it — same reason Saren and aixiaoma stopped at fp8. Calibrated AutoRound *might* survive but needs a real calibration run. **Lever closed.** |

---

## The bottleneck (why not faster)

Decode is **launch / scheduler-bound**, not GPU-compute-bound (91% util at only ~31 W of
GB10's ~100 W+) and not purely memory-bandwidth-bound (independent finding: a smaller
K-quant is *slower* than a larger one). Roughly **~20 ms of the ~35 ms decode step is
fixed overhead**:

- the per-step PLE gather (CPU op + host→device sync, ~4 ms),
- the QSA-state attention-metadata rebuild between MTP draft steps,
- piecewise CUDA-graph boundaries — the PLE mmap gather cannot run inside a FULL graph.

Every layer that can be quantized is quantized as far as it goes (experts INT4, head INT8,
GDN/QSA/shared fp8, PLE fp8, KV bf16 — INT4 GDN was tried and breaks the model). Getting
past ~40/48 now needs **upstream vLLM work** — a fused multi-step QSA draft (removes the
per-draft-step metadata rebuild) or the PLE gather running inside a full CUDA graph
(PR #54371, still Draft). Both target the ~20 ms/step fixed overhead.

---

## Reproduce

```bash
git clone https://github.com/bjarkebolding/spark-vllm-docker-helper
cd /path/to/spark-vllm-docker          # eugr/spark-vllm-docker, unmodified

# fastest (w4a16) — --setup + the first mod pull ~118 GiB (Saren checkpoint + PLE-table shards)
./run-recipe.sh /path/to/helper/recipes/qwen3.8-flash-next-w4a16.yaml --solo --setup --earlyoom -d \
  --apply-mod /path/to/helper/mods/qwen4-exp-w4a16-gptq-fp8 \
  --apply-mod /path/to/helper/mods/qwen4-exp-int8-lmhead \
  --apply-mod /path/to/helper/mods/qwen4-exp-fla-gb10 \
  --apply-mod /path/to/helper/mods/qwen4-exp-qsa-exact-topk

# deterministic (fp8hybrid): recipes/qwen3.8-flash-next-fp8hybrid.yaml + qwen4-exp-fp8-hybrid
# conservative (nvfp4):      recipes/qwen3.8-flash-next-nvfp4.yaml (recipe only)
```

**Credits:** [@Saren-Arterius](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)
(int4/int8/fp8 recipe + quantizers + shims), [blazux](https://github.com/blazux/qwen3.8-Flash-DGX)
(the disk-mmap-PLE foundation + fp8 hybrid), vLLM PR #54129 author (`VLLM_PLE_MMAP`),
Intel (W4A16 AutoRound base), RadixArk (NVFP4 checkpoint). All Apache-2.0.

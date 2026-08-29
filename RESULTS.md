# Tuning results — Qwen3.8-Flash-Next NVFP4 on one GB10

Single DGX Spark (GB10, driver 580.173.02, onboard NVMe),
`RadixArk/Qwen3.8-Flash-Next-NVFP4`, vLLM `0.28.1rc1.dev80+g496580261`
(official `vllm-project/vllm@16d6c376` + PR #54129), TP1, MTP=2, bf16 KV,
PIECEWISE cudagraphs, prefix caching off. Prose = 8 varied explanatory prompts
(300 tok, greedy, `enable_thinking:false`). Code = 4 coding prompts (400 tok).

## The recipe vs the stock defaults

| | Stock (before) | This recipe |
|---|---|---|
| `init engine` | ~190 s | **~43 s** |
| KV cache | 20.1 GiB / 728k tok | **24.2 GiB / 877k tok** |
| Decode, 1 stream — prose | ~27–28 tok/s | ~29 tok/s |
| Decode, 1 stream — code | ~31 tok/s | **~35 tok/s** |
| Aggregate, 4 concurrent | ~37 tok/s | **~42 tok/s** |
| `PLE mmap … copy_ms` / step | ~5 ms fresh, drifts to ~11 ms | **~3 ms**, stable |
| 64k / 128k needle | pass | pass |
| Greedy determinism | — | byte-identical |

## What each change does

| Change | Effect | Why |
|---|---|---|
| `--no-enable-flashinfer-autotune` | **−150 s startup**, removes a latent crash | NVFP4-MoE autotune crashes on sm_121 (FlashInfer #4003). No measurable decode cost. |
| `VLLM_PLE_MMAP_READAHEAD` 128 → 2048 | `copy_ms` stable instead of drifting | at 128 the `posix_fadvise(WILLNEED)` pre-pass was **skipped every decode step** ("512 coalesced runs exceed 128" in the log). |
| `VLLM_PLE_MMAP_WORKERS` 32 → 1 | slightly lower gather latency at batch 1 | a decode gather is ~40–128 hash-scattered rows → ~1 row per pool task, so the 32-thread pool was pure dispatch overhead. Raise it for many concurrent streams. |
| `--gpu-memory-utilization` 0.85 → 0.88 | **+149k KV tokens**, +~5 tok/s at 4-way | more room for KV. 0.88 is near the edge on 128 GB (a few GiB of swap under load); use 0.80–0.85 for 24/7 service. |

## Things that don't work here

| | |
|---|---|
| `--moe-backend b12x` / `marlin` | **fail** — the MTP draft has unquantized BF16 MoE layers, and neither backend supports unquantized MoE (`ValueError`). Auto → `FLASHINFER_CUTLASS` is the only usable NVFP4 path. |
| `num_speculative_tokens: 3` | no change (~29 tok/s) — the extra accepted tokens are offset by the QSA-state backend rebuilding draft attention metadata every step at k ≥ 3. Stay at 2. |
| `"moe_backend":"triton"` / `"index_share_for_mtp_iteration":true` in `--speculative-config` | silently ignored for the MTP path — the draft still loads `FlashInfer CUTLASS Unquantized MoE`. |
| `--async-scheduling` | correctness hazard — with prefix caching off (`mamba_cache_mode=none`) vLLM skips the accepted-token sync, corrupting GDN spec rollback. Only safe with prefix-caching+`align`, which crashes GDN on sm_121 (#54173). |
| `--kv-cache-dtype fp8` | QSA hard-refuses: `NotImplementedError: … QSA requires a BF16 main KV cache`. |
| a GPU LRU cache over the PLE gather | measured n-gram row locality is only ~20 % on prose / ~30 % on code — at that hit rate the cache's per-step cost cancels the saved reads. Not shipped. |
| GPU clock locking (`nvidia-smi -lgc`) | **not a decode win** — decode is memory-bandwidth-bound (~273 GB/s LPDDR5x); capping the clock only lowers it. A 2000–2200 MHz cap is worth it for *stability* (the GB10 otherwise throttle-locks to 513/650/721 MHz and power-offs at ~90 W under sustained load), not speed. Persist it with a systemd unit. |

## The decode ceiling

`PLE mmap: … copy_ms ≈ 3 ms` per decode step is the cost of ~40 hash-scattered
**cold NVMe random reads** — the 47.7 GiB FP8 PLE table is far larger than the
~15 GiB of page cache left after weights + KV, and the PLE layer sits at position
2 of 48 so there is nothing to overlap the reads with. That ~3 ms is ~10 % of the
token time.

Getting past ~29 tok/s prose / ~35 tok/s code on this hardware needs the PLE
table off NVMe entirely — NVFP4-packed resident (~29 GiB) or hash-compressed
(~13 GiB, lossy). Both are out of scope for this lossless recipe.
~29 / ~35 / ~42 (4-way) matches every other single-GB10 report for this model.

## Quality — tool-eval-bench

[`SeraphimSerapis/tool-eval-bench`](https://github.com/SeraphimSerapis/tool-eval-bench)
`run --seed 42 --hardmode` (88 scenarios, temp 0, `--parallel 1`, thinking
**disabled**). Full result: `benchmarks/tool-eval-bench.seed42-hardmode.json`.

| | |
|---|---|
| **Final score** | **81 / 100 · ★★★★ Good** — 142/176 points |
| Scenarios | 64 pass · 14 partial · 10 fail (all 88 completed, no timeouts) |
| Runtime | ~21 min · median turn 3.4 s |
| Safety gate | **fail** — TC-42: injected extra params despite `additionalProperties: false` |
| 100% | Parameter Precision · Restraint & Refusal · Error Recovery · Localization · Instruction Following · Creative Composition |
| Weak | Autonomous Planning 50% · Context & State 65% · Structured Reasoning 67% · Hard Mode 71% · Multi-Step Chains 75% |

All failures are model behaviour (planning, over-eager tool calls, param
injection), not serving — every scenario completed cleanly and deterministically.
Thinking was disabled to keep the run to ~20 min; a thinking-on run would likely
score higher in the weak (planning / multi-step) categories.

# qwen4-exp-fla-gb10

Two GB10 fixes for the flash-linear-attention (GDN) kernels:
1. **Shared-mem gate** `102400 -> 101376` — GB10 reports 99 KiB/block; the gate was
   forcing small-tile Triton GDN kernels. (Affects the prefill Triton path; the
   current vLLM decode path uses a CUDA GDN kernel so decode is unchanged — kept for
   parity with Saren/blazux + the prefill path.)
2. **`chunk_delta_h` num_warps `[2,4] -> [2]`** — fla#953 Blackwell `tl.dot` race
   (correctness).

Both edits guarded + idempotent, skip cleanly if upstream has fixed them.
Credit: @Saren-Arterius / Entrpi's `patch_fla_shmem.py`.

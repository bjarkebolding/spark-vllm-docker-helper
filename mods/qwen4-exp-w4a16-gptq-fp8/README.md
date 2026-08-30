# qwen4-exp-w4a16-gptq-fp8

For the `w4a16` recipe (Saren's `Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid` checkpoint:
int4 GPTQ-Marlin experts + int8 head + blockwise-fp8 dense side).

Two jobs:
1. **Links the PLE shards in** — the Saren checkpoint has the n-gram table stripped;
   this relative-symlinks RadixArk's `model-plefp8-*.safetensors` into the Saren
   snapshot so PR#54129's `VLLM_PLE_MMAP` finds them (same fp8 table, same geometry).
2. **fp8 dispatch shim** — wraps `AutoGPTQConfig` so the 300 F8_E4M3 dense-side layers
   route to `Fp8Config` (CUTLASS blockwise) instead of unquantized bf16.
   `src/vllm_fp8_hybrid.py` by @Saren-Arterius (Apache-2.0), verbatim.

Needs `VLLM_FP8_HYBRID=1` + `VLLM_USE_DEEP_GEMM=0` (the recipe sets both). Idempotent.

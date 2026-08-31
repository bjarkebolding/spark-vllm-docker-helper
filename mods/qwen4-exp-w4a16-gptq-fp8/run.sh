#!/bin/bash
# qwen4-exp-w4a16-gptq-fp8 — for the Saren W4A16 checkpoint (int4 GPTQ-Marlin experts
# + int8 head), whose dense side layers are blockwise fp8. Stock AutoGPTQConfig inits
# those excluded layers as bf16 and can't load the fp8 tensors. This shim wraps
# AutoGPTQConfig.get_quant_method to route the F8_E4M3 layers to Fp8Config instead
# (qsa.py's qkv_proj also routes through AutoGPTQConfig for a gptq checkpoint, so no
# separate qsa hook is needed).
# No-op unless VLLM_FP8_HYBRID=1. Needs VLLM_USE_DEEP_GEMM=0 on sm_121.
# Shim by @Saren-Arterius (Apache-2.0), src/vllm_fp8_hybrid.py verbatim.
set -euo pipefail

PREFIX="[qwen4-exp-w4a16-gptq-fp8]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command -v python3 >/dev/null 2>&1 || { echo "$PREFIX python3 required" >&2; exit 1; }
VLLM_PACKAGE_ROOT="${VLLM_PACKAGE_ROOT:-$(python3 - <<'PY'
import importlib.util
spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("vLLM is not installed for this interpreter")
print(next(iter(spec.submodule_search_locations)))
PY
)}"
SP="$(dirname "$VLLM_PACKAGE_ROOT")"

# The Saren checkpoint has the n-gram table stripped; PR#54129's ple_mmap globs the
# model dir for model-plefp8-*.safetensors. Fetch just those ~48 GiB of shards from
# RadixArk/Qwen3.8-Flash-Next-NVFP4 (nothing else from that repo) and relative-symlink
# them into the Saren snapshot. Relative so the links resolve in any mount namespace.
HF="${HF_HOME:-/root/.cache/huggingface}"
SAREN_SNAP="$(ls -d "$HF"/hub/models--Saren--Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid/snapshots/*/ 2>/dev/null | head -1)"
if [ -z "$SAREN_SNAP" ]; then
  echo "$PREFIX Saren checkpoint not found under $HF — run with --setup." >&2; exit 1
fi
RADIX_SNAP="$(ls -d "$HF"/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/*/ 2>/dev/null | grep -v -- '-' | head -1)"
if [ -z "$RADIX_SNAP" ] || ! ls "$RADIX_SNAP"model-plefp8-*.safetensors >/dev/null 2>&1; then
  echo "$PREFIX fetching PLE-table shards from RadixArk/Qwen3.8-Flash-Next-NVFP4 (~48 GiB)"
  hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 --include 'model-plefp8-*.safetensors' >/dev/null
  RADIX_SNAP="$(ls -d "$HF"/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/*/ 2>/dev/null | grep -v -- '-' | head -1)"
fi
rh="$(basename "${RADIX_SNAP%/}")"
n=0
for f in "$RADIX_SNAP"model-plefp8-*.safetensors; do
  bn="$(basename "$f")"
  ln -sfn "../../../models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/$rh/$bn" "${SAREN_SNAP%/}/$bn"
  n=$((n + 1))
done
echo "  linked $n PLE shards into $(basename "${SAREN_SNAP%/}")"

install -m 644 "$MOD_DIR/vllm_fp8_hybrid.py" "$SP/vllm_fp8_hybrid.py"
echo "  installed $SP/vllm_fp8_hybrid.py"

python3 - "$SP" <<'PY'
import ast, sys
AG = f"{sys.argv[1]}/vllm/model_executor/layers/quantization/auto_gptq.py"
s = open(AG).read()
if "_w4a16_fp8_hybrid_apply" in s:
    print("  auto_gptq.py already hooked"); raise SystemExit(0)
if "class AutoGPTQConfig" not in s:
    print("  ABORT: AutoGPTQConfig not found"); raise SystemExit(1)
s = s.rstrip("\n") + (
    "\n\n# --- qwen4-exp-w4a16-gptq-fp8 ---\n"
    "from vllm_fp8_hybrid import apply as _w4a16_fp8_hybrid_apply\n"
    "_w4a16_fp8_hybrid_apply()\n"
)
ast.parse(s)
open(AG, "w").write(s)
print("  hooked auto_gptq.py")
PY
echo "$PREFIX done"

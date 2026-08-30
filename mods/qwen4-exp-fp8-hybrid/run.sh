#!/bin/bash
# qwen4-exp-fp8-hybrid — NVFP4 experts + blockwise-fp8 dense side layers.
#
# Builds the hybrid checkpoint (once, ~2 min) if absent, then installs the shim.
# Serve with the fp8hybrid recipe: it sets VLLM_FP8_HYBRID=1 and VLLM_USE_DEEP_GEMM=0
# (DeepGEMM's blockwise-fp8 path emits garbage on sm_121; CUTLASS is correct + fast).
#
# Measured on one GB10 vs the plain NVFP4 recipe: ~27->32 prose, ~32->36 code,
# +60k KV tokens, greedy-deterministic, 35k needle pass.
set -euo pipefail

PREFIX="[qwen4-exp-fp8-hybrid]"
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

# --- build the hybrid checkpoint if missing -------------------------------------
HF="${HF_HOME:-/root/.cache/huggingface}"
REPO_DIR="$HF/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4"
SNAP="$(ls -d "$REPO_DIR"/snapshots/*/ 2>/dev/null | grep -v -- '-fp8hybrid' | head -1 || true)"
if [ -n "$SNAP" ]; then
  SNAP="${SNAP%/}"
  DST="${SNAP}-fp8hybrid"
  if [ -f "$DST/.prepared" ]; then
    echo "$PREFIX hybrid checkpoint present: $DST"
  else
    echo "$PREFIX building hybrid checkpoint (~2 min, ~8 GiB)"
    python3 "$MOD_DIR/make_hybrid.py" "$SNAP" "$DST"
  fi
  # stable path for the recipe (independent of the snapshot hash)
  ln -sfn "$DST" "$REPO_DIR/snapshots/fp8hybrid"
  echo "$PREFIX stable path: $REPO_DIR/snapshots/fp8hybrid"
else
  echo "$PREFIX NOTE: RadixArk NVFP4 snapshot not found under $REPO_DIR;"
  echo "$PREFIX       download the model first, then re-run with --apply-mod."
fi

# --- install the vLLM shim -----------------------------------------------------
python3 "$MOD_DIR/apply.py" "$(dirname "$VLLM_PACKAGE_ROOT")"
echo "$PREFIX done"

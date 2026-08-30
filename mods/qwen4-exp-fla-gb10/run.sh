#!/bin/bash
# qwen4-exp-fla-gb10 — GB10 fixes for the flash-linear-attention (GDN) kernels.
#  1. Shared-mem gate 102400 -> 101376: GB10 has 99 KiB/block; the gate was
#     forcing small-tile Triton GDN kernels. (No-op if the decode path uses the
#     CUDA GDN kernel, as on current vLLM — kept for the prefill path + parity
#     with Saren/blazux.)
#  2. chunk_delta_h num_warps [2,4] -> [2]: fla#953 Blackwell tl.dot race (correctness).
# Idempotent, both edits guarded.
set -euo pipefail
PREFIX="[qwen4-exp-fla-gb10]"
command -v python3 >/dev/null 2>&1 || { echo "$PREFIX python3 required" >&2; exit 1; }
VLLM_PACKAGE_ROOT="${VLLM_PACKAGE_ROOT:-$(python3 - <<'PY'
import importlib.util
spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("vLLM is not installed for this interpreter")
print(next(iter(spec.submodule_search_locations)))
PY
)}"
FLA="$VLLM_PACKAGE_ROOT/third_party/flash_linear_attention/ops"
U="$FLA/utils.py"; C="$FLA/chunk_delta_h.py"
[ -f "$U" ] || { echo "$PREFIX not found: $U" >&2; exit 1; }

if grep -q 'DEFAULT = 102400  # Default' "$U"; then
  sed -i 's/    DEFAULT = 102400  # Default/    DEFAULT = 101376  # qwen4-exp-fla-gb10/' "$U"
  grep -q qwen4-exp-fla-gb10 "$U" || { echo "$PREFIX sed failed on $U" >&2; exit 1; }
  echo "$PREFIX FLA shared-mem gate -> 101376"
elif grep -q qwen4-exp-fla-gb10 "$U"; then echo "$PREFIX utils.py already patched"
else echo "$PREFIX WARNING: FLA gate line not found (upstream may have fixed it)"; fi

if [ -f "$C" ] && grep -q 'for num_warps in \[2, 4\]' "$C"; then
  sed -i 's/for num_warps in \[2, 4\]/for num_warps in [2]  # qwen4-exp-fla-gb10 fla#953/' "$C"
  grep -q qwen4-exp-fla-gb10 "$C" || { echo "$PREFIX sed failed on $C" >&2; exit 1; }
  echo "$PREFIX chunk_delta_h num_warps -> [2]"
fi
python3 -c "import ast; ast.parse(open('$U').read())" || { echo "$PREFIX $U broken" >&2; exit 1; }
echo "$PREFIX done"

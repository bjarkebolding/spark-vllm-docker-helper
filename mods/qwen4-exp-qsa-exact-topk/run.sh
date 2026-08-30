#!/bin/bash
# qwen4-exp-qsa-exact-topk — opt-in deterministic QSA block top-k.
# On sm_121 the QSA indexer falls to torch.ops._C.persistent_topk, which is
# non-deterministic run-to-run (vllm#51782). With VLLM_QSA_EXACT_TOPK=1 the
# indexer uses an exact torch.topk over the visible columns instead.
# `fill` = -inf-mask the never-written columns then the stock kernel (diagnostic).
# No-op unless VLLM_QSA_EXACT_TOPK is set. Patcher by blazux/qwen3.8-Flash-DGX (Apache-2.0).
set -euo pipefail

PREFIX="[qwen4-exp-qsa-exact-topk]"
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
TARGET="$VLLM_PACKAGE_ROOT/models/qwen4_exp/nvidia/ops/qsa.py"
[ -f "$TARGET" ] || { echo "$PREFIX not found: $TARGET" >&2; exit 1; }

if grep -q "VLLM_QSA_EXACT_TOPK" "$TARGET"; then
  echo "$PREFIX already patched"; exit 0
fi
python3 "$MOD_DIR/patch_qsa_exact_topk.py" "$TARGET"
echo "$PREFIX done"

#!/bin/bash
# qwen4-exp-ple-pinned — pinned-host staging for the VLLM_PLE_MMAP gather. See README.md.
set -euo pipefail

PREFIX="[qwen4-exp-ple-pinned]"
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

TARGET="$VLLM_PACKAGE_ROOT/models/qwen4_exp/nvidia/ple_mmap.py"
[ -f "$TARGET" ] || { echo "$PREFIX not found: $TARGET" >&2; exit 1; }

python3 "$MOD_DIR/apply.py" "$TARGET"
echo "$PREFIX done — $TARGET"

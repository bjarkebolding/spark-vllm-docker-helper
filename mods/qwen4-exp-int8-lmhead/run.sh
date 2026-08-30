#!/bin/bash
# qwen4-exp-int8-lmhead — let ParallelLMHead pick up the checkpoint's quantized
# (int8 GPTQ) packing in both the main model and the MTP draft head.
# Upstream builds ParallelLMHead without quant_config -> forced bf16 head
# (1.27 GiB + a bf16 GEMV/token over a 248320 vocab). Without the mtp.py half,
# MTP>=3 crashes at load ("no parameter named 'lm_head.qweight'").
# No-op on a checkpoint whose lm_head is not quantized. Idempotent, guarded.
set -euo pipefail

PREFIX="[qwen4-exp-int8-lmhead]"
command -v python3 >/dev/null 2>&1 || { echo "$PREFIX python3 required" >&2; exit 1; }
VLLM_PACKAGE_ROOT="${VLLM_PACKAGE_ROOT:-$(python3 - <<'PY'
import importlib.util
spec = importlib.util.find_spec("vllm")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("vLLM is not installed for this interpreter")
print(next(iter(spec.submodule_search_locations)))
PY
)}"

python3 - "$VLLM_PACKAGE_ROOT" <<'PY'
import sys
root = sys.argv[1]
OLD = (
    "        self.lm_head = ParallelLMHead(\n"
    "            config.vocab_size,\n"
    "            config.hidden_size,\n"
    "            prefix=maybe_prefix(prefix, \"lm_head\"),\n"
    "        )"
)
NEW = (
    "        self.lm_head = ParallelLMHead(\n"
    "            config.vocab_size,\n"
    "            config.hidden_size,\n"
    "            prefix=maybe_prefix(prefix, \"lm_head\"),\n"
    "            quant_config=vllm_config.quant_config,  # qwen4-exp-int8-lmhead\n"
    "        )"
)
MTP_OLD = (
    "                self.lm_head = ParallelLMHead(\n"
    "                    config.vocab_size,\n"
    "                    config.hidden_size,\n"
    "                    prefix=maybe_prefix(prefix, \"lm_head\"),\n"
    "                )"
)
MTP_NEW = (
    "                self.lm_head = ParallelLMHead(\n"
    "                    config.vocab_size,\n"
    "                    config.hidden_size,\n"
    "                    prefix=maybe_prefix(prefix, \"lm_head\"),\n"
    "                    quant_config=vllm_config.quant_config,  # qwen4-exp-int8-lmhead\n"
    "                )"
)
import ast
for rel, old, new in [
    ("models/qwen4_exp/nvidia/model.py", OLD, NEW),
    ("models/qwen4_exp/nvidia/mtp.py", MTP_OLD, MTP_NEW),
]:
    p = f"{root}/{rel}"
    s = open(p).read()
    if "qwen4-exp-int8-lmhead" in s:
        print(f"  {rel}: already patched"); continue
    if old not in s:
        print(f"  ABORT: {rel}: ParallelLMHead call site not in expected shape"); raise SystemExit(1)
    s = s.replace(old, new, 1)
    ast.parse(s)
    open(p, "w").write(s)
    print(f"  patched {rel}")
PY
echo "$PREFIX done"

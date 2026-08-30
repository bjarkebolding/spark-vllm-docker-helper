#!/usr/bin/env python3
"""Install the fp8-hybrid shim: NVFP4 experts + blockwise-fp8 dense side layers.
Idempotent, guarded. No-op at runtime unless VLLM_FP8_HYBRID=1. Needs VLLM_USE_DEEP_GEMM=0
on sm_121 (DeepGEMM blockwise-fp8 fails / emits garbage on GB10).
Shim ported from @Saren-Arterius via blazux/qwen3.8-Flash-DGX (Apache-2.0)."""

from __future__ import annotations

import ast
import os
import shutil
import sys

MOD_DIR = os.path.dirname(os.path.abspath(__file__))
SP = sys.argv[1]

SHIM = f"{SP}/vllm_fp8_hybrid_modelopt.py"
MODELOPT = f"{SP}/vllm/model_executor/layers/quantization/modelopt.py"
QSA = f"{SP}/vllm/models/qwen4_exp/nvidia/qsa.py"

MODELOPT_HOOK = (
    "\n\n# --- qwen4-exp-fp8-hybrid ---\n"
    "from vllm_fp8_hybrid_modelopt import apply as _fp8_hybrid_apply\n"
    "_fp8_hybrid_apply()\n"
)
QSA_IMPORT = "from vllm_fp8_hybrid_modelopt import excluded_quant_config as _fp8_hybrid_excluded"
QSA_OLD = "quant_config=model.without_modelopt_fp4(quant_config)"
QSA_NEW = "quant_config=_fp8_hybrid_excluded(quant_config)"


def _fail(msg):
    print(f"  ABORT: {msg}")
    raise SystemExit(1)


def main():
    for p in (MODELOPT, QSA):
        if not os.path.isfile(p):
            _fail(f"not found: {p}")

    shutil.copyfile(f"{MOD_DIR}/vllm_fp8_hybrid_modelopt.py", SHIM)
    print(f"  installed {SHIM}")

    mo = open(MODELOPT).read()
    if "_fp8_hybrid_apply" not in mo:
        if "class ModelOptNvFp4Config" not in mo:
            _fail("modelopt.py: ModelOptNvFp4Config not found")
        mo = mo.rstrip("\n") + MODELOPT_HOOK
        ast.parse(mo)
        open(MODELOPT, "w").write(mo)
        print("  hooked modelopt.py")
    else:
        print("  modelopt.py already hooked")

    qs = open(QSA).read()
    if "_fp8_hybrid_excluded" not in qs:
        if QSA_OLD not in qs:
            _fail(f"qsa.py: expected call site not found ({QSA_OLD!r})")
        if "\nfrom . import model\n" not in qs:
            _fail("qsa.py: 'from . import model' not found")
        qs = qs.replace("\nfrom . import model\n", f"\nfrom . import model\n{QSA_IMPORT}\n", 1)
        qs = qs.replace(QSA_OLD, QSA_NEW, 1)
        ast.parse(qs)
        open(QSA, "w").write(qs)
        print("  hooked qsa.py")
    else:
        print("  qsa.py already hooked")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

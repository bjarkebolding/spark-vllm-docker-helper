#!/usr/bin/env python3
"""Build the fp8-hybrid snapshot: copy the RadixArk NVFP4 snapshot, rewrite the
300 dense-side bf16 projections to blockwise fp8-e4m3 (DeepSeek layout).

Unchanged shards are hardlinked (instant, no extra disk). Only the 4 touched
shards + the index become real new files.

Quantizer (block_quant / TARGETS) is @Saren-Arterius's fp8_convert.py (Apache-2.0),
via blazux/qwen3.8-Flash-DGX.

    python3 make_hybrid.py <src_snapshot_dir> <dst_dir>
"""
import json
import os
import re
import sys

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open

SRC, DST = sys.argv[1], sys.argv[2]
BLOCK = 128
FP8_MAX = 448.0

TARGETS = re.compile(
    r"model\.language_model\.layers\.\d+\.("
    r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
    r"|self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.shared_expert\.(gate_proj|up_proj|down_proj)"
    r")\.weight$"
)


def block_quant(w: torch.Tensor):
    out, inn = w.shape
    assert out % BLOCK == 0 and inn % BLOCK == 0, w.shape
    wf = w.float().reshape(out // BLOCK, BLOCK, inn // BLOCK, BLOCK)
    absmax = wf.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    scale = absmax / FP8_MAX
    q = (wf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    deq = q.float() * scale
    rel = ((deq - wf).abs().amax() / wf.abs().amax()).item()
    return q.reshape(out, inn), scale.squeeze(1).squeeze(-1).contiguous(), rel


def main():
    os.makedirs(DST, exist_ok=True)
    idx = json.load(open(f"{SRC}/model.safetensors.index.json"))
    wm = idx["weight_map"]
    targets = {n: f for n, f in wm.items() if TARGETS.search(n)}
    by_file = {}
    for name, fname in targets.items():
        by_file.setdefault(fname, []).append(name)
    print(f"{len(targets)} tensors across {len(by_file)} shards -> fp8 blockwise")

    all_shards = sorted(set(wm.values()))
    for fname in all_shards:
        s, d = f"{SRC}/{fname}", f"{DST}/{fname}"
        if fname in by_file:
            continue
        if os.path.exists(d):
            os.remove(d)
        try:
            os.link(os.path.realpath(s), d)
        except OSError:
            import shutil

            shutil.copy2(s, d)
    # non-shard files (config.json, tokenizer, etc.)
    for f in os.listdir(SRC):
        if f.endswith(".safetensors") or f == "model.safetensors.index.json":
            continue
        s, d = f"{SRC}/{f}", f"{DST}/{f}"
        if os.path.isfile(s) and not os.path.exists(d):
            try:
                os.link(os.path.realpath(s), d)
            except OSError:
                import shutil

                shutil.copy2(s, d)

    worst = 0.0
    for i, (fname, names) in enumerate(sorted(by_file.items())):
        src_path = os.path.realpath(f"{SRC}/{fname}")
        tensors = {}
        with safe_open(src_path, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        for name in names:
            w = tensors.pop(name)
            if w.shape[0] % BLOCK or w.shape[1] % BLOCK:
                print(f"  SKIP (shape) {name} {tuple(w.shape)}")
                tensors[name] = w
                continue
            q, scale, rel = block_quant(w)
            worst = max(worst, rel)
            tensors[name] = q
            sname = name.replace(".weight", ".weight_scale_inv")
            tensors[sname] = scale
            wm[sname] = fname
        save_file(tensors, f"{DST}/{fname}", metadata={"format": "pt"})
        print(f"[{i+1}/{len(by_file)}] {fname}: {len(names)} tensors, worst rel err so far {worst:.4f}")

    json.dump(idx, open(f"{DST}/model.safetensors.index.json", "w"))
    assert worst < 0.10, f"fp8 roundtrip error too large: {worst}"
    open(f"{DST}/.prepared", "w").write("fp8hybrid\n")
    print(f"done. worst per-tensor max rel err: {worst:.4f}")


if __name__ == "__main__":
    main()

#!/bin/bash
# mods/qwen4-exp-ple-cache — GPU-resident LRU row cache for the VLLM_PLE_MMAP
# n-gram gather.
#
# The stock MmapNgramEmbedding.forward reads every needed n-gram row from the
# mmap'd table on every decode step. On a GB10 the 47.7 GiB table is far bigger
# than the page cache, so those are cold NVMe random reads (~3-7 ms/step,
# "copy_ms" in the PLE mmap stats line). This keeps the most-recently-gathered
# rows resident on the GPU and only reads the misses from disk.
#
#   VLLM_PLE_MMAP_CACHE_ROWS=0   disable (fall back to the stock gather)
#   VLLM_PLE_MMAP_CACHE_ROWS=N   cache N rows (default 1048576 ~= 168 MiB GPU)
#
# Logs "PLE cache: hit_rate=... " once per 60 s. Rows are immutable, so a
# cached row is always valid; eviction is a plain ring buffer.
set -euo pipefail
PYROOT="${PYTHON_ROOT:-/usr/local/lib/python3.12/dist-packages}"
F="$PYROOT/vllm/models/qwen4_exp/nvidia/ple_mmap.py"
[ -f "$F" ] || { echo "[ple-cache] $F not found — not a qwen4_exp mmap build" >&2; exit 1; }
command -v python3 >/dev/null || { echo "[ple-cache] python3 required" >&2; exit 1; }

python3 - "$F" <<'PY'
import ast, re, sys
p = sys.argv[1]
src = open(p).read()
if "# ple-cache mod" in src:
    print("[ple-cache] already applied; skipping.")
    sys.exit(0)

anchor = 'ids_np = ids.detach().to("cpu", non_blocking=False).numpy().reshape(-1)'
if anchor not in src:
    sys.exit("[ple-cache] stock MmapNgramEmbedding.forward not found — refusing to patch")

m = re.search(
    r"\n    def forward\(self, ids: torch\.Tensor\) -> torch\.Tensor:\n.*?\n\ndef set_weight_scale",
    src, re.S)
if not m:
    sys.exit("[ple-cache] could not delimit MmapNgramEmbedding.forward")

NEW = '''
    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # ple-cache mod
        table = self.table
        if table is None:
            if self.weights_streamed:
                raise RuntimeError(
                    "PLE mmap table not initialized — load_weights ran but "
                    "build_tables did not"
                )
            return torch.zeros(
                (*ids.shape, self.embedding_dim),
                dtype=self.torch_dtype,
                device=ids.device,
            )
        row_bytes = table.row_bytes
        if row_bytes != self.embedding_dim * table.itemsize:
            raise ValueError(
                f"PLE mmap: table row_bytes={row_bytes} does not match "
                f"embedding_dim={self.embedding_dim} * itemsize={table.itemsize}"
            )
        import os as _os, time as _time
        cap = int(_os.environ.get("VLLM_PLE_MMAP_CACHE_ROWS", "1048576"))
        dev = ids.device
        shape = ids.shape
        flat = ids.reshape(-1)

        if cap <= 0 or flat.numel() == 0 or dev.type != "cuda":
            ids_np = flat.detach().to("cpu", non_blocking=False).numpy()
            rows = table.gather(ids_np)
            out = torch.from_numpy(rows).view(table.torch_dtype).to(dev, non_blocking=True)
            return out.reshape(*shape, self.embedding_dim)

        pc = getattr(self, "_pc", None)
        if pc is None:
            pc = self._pc = {
                "cap": cap,
                "rows": torch.zeros((cap, row_bytes), dtype=torch.uint8, device=dev),
                "map": {},                 # global row id -> slot
                "slot_id": [-1] * cap,     # slot -> global row id
                "next": 0,
                "hits": 0, "miss": 0, "log": _time.monotonic(),
            }

        uniq, inv = torch.unique(flat, return_inverse=True)
        uniq_l = uniq.tolist()
        cmap = pc["map"]
        slots = [cmap.get(r, -1) for r in uniq_l]
        miss_pos = [i for i, s in enumerate(slots) if s < 0]
        pc["hits"] += len(uniq_l) - len(miss_pos)

        if miss_pos:
            miss_ids = np.asarray([uniq_l[i] for i in miss_pos], dtype=np.int64)
            mrows = torch.from_numpy(table.gather(miss_ids)).to(dev, non_blocking=True)
            n = len(miss_pos)
            base = pc["next"]
            dst = [(base + k) % cap for k in range(n)]
            for k, s in enumerate(dst):
                old = pc["slot_id"][s]
                if old >= 0:
                    cmap.pop(old, None)
                rid = int(miss_ids[k])
                pc["slot_id"][s] = rid
                cmap[rid] = s
                slots[miss_pos[k]] = s
            pc["next"] = (base + n) % cap
            pc["rows"].index_copy_(
                0, torch.as_tensor(dst, dtype=torch.int64, device=dev), mrows)
            pc["miss"] += n

        slot_t = torch.as_tensor(slots, dtype=torch.int64, device=dev)
        out = pc["rows"].index_select(0, slot_t)[inv].view(table.torch_dtype)

        now = _time.monotonic()
        if now - pc["log"] > 60.0:
            tot = pc["hits"] + pc["miss"]
            if tot:
                logger.info(
                    "PLE cache: hit_rate=%.1f%% (hits=%d miss=%d) cap=%d",
                    100.0 * pc["hits"] / tot, pc["hits"], pc["miss"], cap)
            pc["hits"] = pc["miss"] = 0
            pc["log"] = now

        return out.reshape(*shape, self.embedding_dim)


def set_weight_scale'''

src2 = src[:m.start()] + NEW + src[m.end():]
ast.parse(src2)
open(p, "w").write(src2)
print("[ple-cache] applied GPU LRU row cache to MmapNgramEmbedding.forward")
PY

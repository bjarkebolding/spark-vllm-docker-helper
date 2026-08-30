#!/usr/bin/env python3
"""Idempotent, shape-guarded patch: route the VLLM_PLE_MMAP H2D through pinned memory."""

from __future__ import annotations

import ast
import sys

MARKER = "# --- qwen4-exp-ple-pinned ---"

STOCK_TAIL = '''        out = torch.from_numpy(rows).view(table.torch_dtype)
        # non_blocking=True has no effect here: `rows` (from table.gather)
        # is pageable host memory, not pinned, so this H2D copy is
        # effectively synchronous. Pinned staging (Phase-4 lever 5) is a
        # separate, not-yet-pulled lever, not something this line hides.
        out = out.to(ids.device, non_blocking=True)
        return out.reshape(*ids.shape, self.embedding_dim)
'''

PATCHED_TAIL = '''        out = _ple_pinned_h2d(self, rows, table.torch_dtype, ids.device)
        return out.reshape(*ids.shape, self.embedding_dim)
'''

HELPER = '''

''' + MARKER + '''
def _ple_pinned_h2d(module, rows, torch_dtype, device):
    """H2D of gathered PLE rows via a persistent pinned buffer; pageable fallback."""
    import os

    n, row_bytes = int(rows.shape[0]), int(rows.shape[1])
    if os.environ.get("VLLM_PLE_MMAP_PINNED", "1") == "0" or n == 0:
        return torch.from_numpy(rows).view(torch_dtype).to(device, non_blocking=True)

    buf = getattr(module, "_ple_pin_buf", None)
    if (
        buf is None
        or getattr(module, "_ple_pin_row_bytes", None) != row_bytes
        or buf.shape[0] < n
    ):
        new_cap = max(8192, 1 << max(0, (n - 1)).bit_length())
        try:
            buf = torch.empty((new_cap, row_bytes), dtype=torch.uint8, pin_memory=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning_once(
                "PLE mmap: pinned staging unavailable (%s); using pageable H2D", exc
            )
            module._ple_pin_buf = None
            return (
                torch.from_numpy(rows).view(torch_dtype).to(device, non_blocking=True)
            )
        module._ple_pin_buf = buf
        module._ple_pin_row_bytes = row_bytes

    staged = buf[:n]
    staged.numpy()[:] = rows
    return staged.view(torch_dtype).to(device, non_blocking=True)
'''


def main(path: str) -> int:
    src = open(path).read()

    if MARKER in src:
        print("  already patched; nothing to do")
        return 0

    guards = [
        'ids_np = ids.detach().to("cpu", non_blocking=False).numpy().reshape(-1)',
        "rows = table.gather(ids_np)",
        "class MmapNgramEmbedding(nn.Module):",
        STOCK_TAIL,
    ]
    for g in guards:
        if g not in src:
            print(f"  ABORT: expected code not found:\n    {g.splitlines()[0]}")
            return 1
    if src.count(STOCK_TAIL) != 1:
        print("  ABORT: forward() tail is not unique")
        return 1

    patched = src.replace(STOCK_TAIL, PATCHED_TAIL, 1).rstrip("\n") + HELPER + "\n"

    try:
        ast.parse(patched)
    except SyntaxError as exc:
        print(f"  ABORT: patched file does not parse: {exc}")
        return 1

    open(path, "w").write(patched)
    print("  patched MmapNgramEmbedding.forward -> pinned H2D staging")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))

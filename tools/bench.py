#!/usr/bin/env python3
"""Benchmark / smoke / warmup a running Qwen3.8-Flash-Next server (OpenAI API)."""
import argparse, json, sys, time, urllib.request, concurrent.futures

MODEL = "qwen3.8-flash-next"


def chat(port, content, max_tokens, temperature=0.0, think=False, timeout=600):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": think},
    }
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t0
    u = r["choices"][0]["message"]
    return {
        "text": (u.get("content") or ""),
        "out": r["usage"]["completion_tokens"],
        "prompt": r["usage"]["prompt_tokens"],
        "dt": dt,
        "finish": r["choices"][0].get("finish_reason"),
    }


PROSE = [
    "Explain how a mixture-of-experts layer routes tokens and why only a fraction of parameters activate per token.",
    "Write a 150-word technical blog intro about running large language models on unified-memory hardware.",
    "Compare linear attention with standard softmax attention across memory, quality, and long-context behaviour.",
    "Describe the steps to quantize a transformer's weights to 4-bit NVFP4, including what stays higher precision.",
    "Summarize the tradeoffs of speculative decoding with a small draft model versus multi-token prediction heads.",
    "Give a clear explanation of KV cache memory growth with context length and three techniques to reduce it.",
    "Outline a benchmarking methodology for measuring tokens per second of an LLM server.",
    "Explain what an OS page cache is and how memory-mapped file reads interact with it under memory pressure.",
]


def bench_prose(port):
    print("=== prose, single-stream, no thinking ===")
    tot_o = tot_t = 0.0
    ttfts = []
    for p in PROSE:
        r = chat(port, p, 300, think=False)
        tps = r["out"] / r["dt"]
        tot_o += r["out"]; tot_t += r["dt"]
        print(f"  {r['out']:4d} tok / {r['dt']:6.2f}s = {tps:6.2f} tok/s")
    print(f"TOTAL: {tot_o:.0f} tok / {tot_t:.1f}s = {tot_o/tot_t:.2f} tok/s single-stream\n")
    return tot_o / tot_t


def bench_needle(port, ntokens=64000):
    print(f"=== needle-in-haystack (~{ntokens//1000}k) ===")
    filler = "The survey crew mapped the ridge line each dawn while the fog still clung to the valley floor. "
    reps = max(1, ntokens // 20)
    hay = filler * reps
    needle = "  KEY DETAIL: the vault code is 47-19-83.  "
    doc = hay[: len(hay) // 2] + needle + hay[len(hay) // 2 :]
    r = chat(port, doc + "\n\nWhat is the vault code stated above? Reply with only the digits.",
             24, think=False)
    ok = "47-19-83" in r["text"]
    pf = r["prompt"] / max(r["dt"], 1e-6)
    print(f"  prompt={r['prompt']} tok  answer={r['text']!r}  e2e={r['dt']:.1f}s  ~prefill={pf:.0f} tok/s  {'PASS' if ok else 'FAIL'}\n")
    return ok


def bench_concurrency(port, n=4):
    print(f"=== concurrency {n}, ~4k prompt, 256 out ===")
    prompt = ("Analyze the following notes and produce a summary. " + "detail item alpha bravo charlie delta. " * 300)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        rs = list(ex.map(lambda _: chat(port, prompt, 256, think=False), range(n)))
    wall = time.time() - t0
    agg = sum(r["out"] for r in rs) / wall
    print(f"  {n} streams: {sum(r['out'] for r in rs)} tok / {wall:.1f}s wall = {agg:.1f} tok/s aggregate\n")
    return agg


def warmup(port):
    print("=== warmup sweep (compile QSA / indexer / rejection kernels) ===")
    sizes = [200, 2000, 16000, 64000, 180000]
    for s in sizes:
        body = "word " * (s // 2)
        r = chat(port, body + " Reply with OK.", 8, think=False)
        print(f"  ctx~{s:>7}: prompt={r['prompt']} out={r['out']} {r['dt']:.1f}s")
    # a few concurrent to hit batch-size buckets
    for n in (2, 4, 8):
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
            list(ex.map(lambda _: chat(port, "Count to five.", 16, think=False), range(n)))
        print(f"  concurrency {n}: done")
    print("warmup complete\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--mode", default="all",
                    choices=["all", "smoke", "warmup", "prose", "needle", "concurrency"])
    ap.add_argument("--needle-tokens", type=int, default=64000)
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()

    if a.mode == "smoke":
        r = chat(a.port, "List the first 10 prime numbers.", 80, think=False)
        print(r["text"])
        sys.exit(0 if "2, 3, 5, 7, 11, 13, 17, 19, 23" in r["text"].replace(" and", "") or "29" in r["text"] else 1)
    if a.mode == "warmup":
        warmup(a.port); return
    if a.mode == "prose":
        bench_prose(a.port); return
    if a.mode == "needle":
        sys.exit(0 if bench_needle(a.port, a.needle_tokens) else 1)
    if a.mode == "concurrency":
        bench_concurrency(a.port, a.concurrency); return

    tps = bench_prose(a.port)
    ok = bench_needle(a.port, a.needle_tokens)
    agg = bench_concurrency(a.port, a.concurrency)
    print(f"SUMMARY  single-stream {tps:.1f} tok/s | needle {'PASS' if ok else 'FAIL'} | conc{a.concurrency} {agg:.1f} tok/s")


if __name__ == "__main__":
    main()

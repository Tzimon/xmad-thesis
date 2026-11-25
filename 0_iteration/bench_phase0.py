#!/usr/bin/env python3
"""Throughput/latency benchmark for the served model.

Sends N chat-completion requests to the local OpenAI-compatible endpoint and
records, per request, the wall-clock latency and the prompt/completion token
counts reported by vLLM. Writes a per-request CSV and an aggregate JSON summary
(throughput in req/s and tokens/s, plus latency mean/p50/p90/p99).

Only depends on `requests`, which ships as a vLLM dependency.
"""
import argparse, time, json, csv, statistics
import concurrent.futures as cf
import requests

# A small, self-contained prompt pool (no retrieval needed in Phase 0);
# cycled up to N so runs are comparable across iterations.
PROMPTS = [
    "Reply in one sentence: what is a large language model?",
    "Name three programming languages.",
    "What is the capital of France?",
    "Explain in one sentence what a GPU does.",
    "Give a short definition of retrieval-augmented generation.",
    "What is the boiling point of water at sea level in Celsius?",
    "Summarise the theory of relativity in one sentence.",
    "List two benefits of on-premise model hosting.",
    "What is the difference between a dense and a sparse model? One sentence.",
    "Translate 'good morning' into German.",
    "What is 17 multiplied by 23?",
    "Give one example of a multi-hop question.",
    "What does the acronym API stand for?",
    "Name a common vector database.",
    "In one sentence, what is quantisation in machine learning?",
    "What is the primary role of an attention mechanism?",
    "State one advantage of open-weight models.",
    "What is a token in the context of language models?",
    "Give a one-sentence description of the Model Context Protocol.",
    "Explain what latency means for an inference server.",
]

def one_request(base_url, model, prompt, max_tokens):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.7}
    t = time.time()
    try:
        r = requests.post(url, json=payload, timeout=180)
        lat = time.time() - t
        r.raise_for_status()
        u = r.json().get("usage", {}) or {}
        return {"ok": True, "latency": lat,
                "prompt_tokens": u.get("prompt_tokens", 0),
                "completion_tokens": u.get("completion_tokens", 0)}
    except Exception as e:
        return {"ok": False, "latency": time.time() - t, "error": str(e),
                "prompt_tokens": 0, "completion_tokens": 0}

def run(base_url, model, prompts, concurrency, max_tokens):
    results = [None] * len(prompts)
    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
        fut = {ex.submit(one_request, base_url, model, p, max_tokens): i
               for i, p in enumerate(prompts)}
        for f in cf.as_completed(fut):
            results[fut[f]] = f.result()
    return results

def pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, int(round(q / 100.0 * (len(xs) - 1))))
    return xs[k]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-summary", required=True)
    a = ap.parse_args()

    pool = (PROMPTS * ((a.n // len(PROMPTS)) + 1))
    warm = pool[:a.warmup]
    work = pool[:a.n]

    if a.warmup:
        print(f"[bench] warmup: {a.warmup} requests")
        run(a.base_url, a.model, warm, a.concurrency, a.max_tokens)

    print(f"[bench] measuring: n={a.n} concurrency={a.concurrency}")
    t0 = time.time()
    res = run(a.base_url, a.model, work, a.concurrency, a.max_tokens)
    wall = time.time() - t0

    with open(a.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "ok", "latency_s", "prompt_tokens", "completion_tokens"])
        for i, r in enumerate(res):
            w.writerow([i, r["ok"], f"{r['latency']:.4f}",
                        r["prompt_tokens"], r["completion_tokens"]])

    ok = [r for r in res if r["ok"]]
    lat = [r["latency"] for r in ok]
    comp_tok = sum(r["completion_tokens"] for r in ok)
    summary = {
        "model": a.model, "n": a.n, "concurrency": a.concurrency,
        "max_tokens": a.max_tokens, "warmup": a.warmup,
        "success": len(ok), "success_rate": len(ok) / a.n if a.n else 0,
        "wall_time_s": round(wall, 3),
        "throughput_req_s": round(len(ok) / wall, 3) if wall else None,
        "throughput_completion_tokens_s": round(comp_tok / wall, 1) if wall else None,
        "latency_mean_s": round(statistics.mean(lat), 4) if lat else None,
        "latency_p50_s": round(pct(lat, 50), 4),
        "latency_p90_s": round(pct(lat, 90), 4),
        "latency_p99_s": round(pct(lat, 99), 4),
    }
    json.dump(summary, open(a.out_summary, "w"), indent=2)
    print("[bench] summary:", json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

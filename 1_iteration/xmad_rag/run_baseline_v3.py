"""
xmad/1_iteration/xmad_rag/run_baseline.py

Building block 4: the single-agent RAG baseline (C5.1).

Runs the agent loop over N HotpotQA distractor questions through the shared MCP
tool pool, scores answers with the official HotpotQA exact-match and F1 metrics,
and records latency (model time vs. tool time), token consumption, per-question
tool traces, and node GPU/VRAM utilisation.

Each run writes a timestamped folder under logs/ containing:
    baseline_results.csv  -- one row per question, written incrementally so a
                             crash mid-run keeps everything completed so far
    summary.json          -- config + aggregate metrics (means/percentiles/totals),
                             including a GPU summary (active GPU, peak VRAM, util)
    gpu_metrics.csv        -- nvidia-smi samples over the run (if nvidia-smi present)

Note: the run is sequential (one question at a time). The MCP server holds one
active corpus (load_corpus sets it per question), so questions cannot yet run
concurrently without racing on that corpus -- concurrency needs per-question
corpus isolation, which is a later change.

Requires, on the same node:
  * vLLM serving Qwen with tool calling  (port 7555)
  * the MCP server running                (port 7600)

    python -m xmad_rag.run_baseline --n 3700          # ~50% of the dev split
    python -m xmad_rag.run_baseline --n 500 --start 0 # a quick sanity slice
"""

import argparse
import asyncio
import csv
import json
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from fastmcp import Client as MCPClient
from openai import AsyncOpenAI

from xmad_rag.baseline_agent_loop import run_agent
from xmad_rag.metrics import exact_match, f1_score
from xmad_rag.retriever import build_corpus

MCP_URL = "http://127.0.0.1:7600/mcp"
VLLM_URL = "http://127.0.0.1:7555/v1"
MODEL = "qwen"

CSV_COLUMNS = [
    "idx", "id", "question", "gold", "pred", "em", "f1",
    "latency_s", "llm_time_s", "tool_time_s",
    "tool_calls", "llm_calls",
    "prompt_tokens", "completion_tokens", "total_tokens",
    "tool_trace", "pred_raw",
]


# --------------------------------------------------------------------------- #
# GPU/VRAM sampling
# --------------------------------------------------------------------------- #
def start_gpu_sampler(path):
    try:
        f = open(path, "w")
        proc = subprocess.Popen(
            ["nvidia-smi",
             "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,"
             "memory.used,memory.total",
             "--format=csv,nounits", "-lms", "1000"],
            stdout=f, stderr=subprocess.DEVNULL,
        )
        return proc, f
    except FileNotFoundError:
        return None, None


def stop_gpu_sampler(proc, f):
    if proc is not None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    if f is not None:
        f.close()


def summarize_gpu(path):
    """Reduce the raw nvidia-smi samples to the active GPU's peak VRAM and
    mean/peak utilisation. Returns None if the file is missing/empty."""
    try:
        per = {}
        with open(path) as f:
            reader = csv.reader(f)
            next(reader, None)  # header
            for row in reader:
                if len(row) < 6:
                    continue
                try:
                    idx = int(row[1]); util = float(row[2])
                    mem_used = float(row[4]); mem_total = float(row[5])
                except ValueError:
                    continue
                g = per.setdefault(idx, {"util": [], "mem": [], "total": mem_total})
                g["util"].append(util); g["mem"].append(mem_used); g["total"] = mem_total
        if not per:
            return None
        active = max(per, key=lambda k: max(per[k]["mem"]))
        g = per[active]
        return {
            "active_gpu": active,
            "vram_peak_mib": int(max(g["mem"])),
            "vram_total_mib": int(g["total"]),
            "util_gpu_mean_pct": round(sum(g["util"]) / len(g["util"]), 1),
            "util_gpu_peak_pct": int(max(g["util"])),
            "samples": len(g["util"]),
        }
    except FileNotFoundError:
        return None


# --------------------------------------------------------------------------- #
# The run (writes each row to the CSV as it completes)
# --------------------------------------------------------------------------- #
def _row_for_csv(row):
    out = dict(row)
    out["tool_trace"] = json.dumps(row["tool_trace"], separators=(",", ":"))
    out["pred_raw"] = row["pred_raw"].replace("\n", " ").strip()
    return out


async def run_baseline(n, start, run_dir):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rows = []
    llm = AsyncOpenAI(base_url=VLLM_URL, api_key="not-needed")

    csv_path = run_dir / "baseline_results.csv"
    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        cf.flush()

        async with MCPClient(MCP_URL) as mcp:  # ONE connection for the whole run
            for i, ex in enumerate(ds.select(range(start, start + n))):
                await mcp.call_tool("load_corpus", {"paragraphs": build_corpus(ex)})

                t0 = time.perf_counter()
                res = await run_agent(ex["question"], model=MODEL, mcp=mcp, llm=llm)
                latency = time.perf_counter() - t0

                em = exact_match(res["answer"], ex["answer"])
                f1 = f1_score(res["answer"], ex["answer"])
                row = {
                    "idx": start + i,
                    "id": ex["id"],
                    "question": ex["question"],
                    "gold": ex["answer"],
                    "pred": res["answer"],
                    "em": em,
                    "f1": round(f1, 4),
                    "latency_s": round(latency, 3),
                    "llm_time_s": res["llm_time_s"],
                    "tool_time_s": res["tool_time_s"],
                    "tool_calls": res["tool_calls"],
                    "llm_calls": res["llm_calls"],
                    "prompt_tokens": res["prompt_tokens"],
                    "completion_tokens": res["completion_tokens"],
                    "total_tokens": res["total_tokens"],
                    "tool_trace": res["tool_trace"],
                    "pred_raw": res["raw"],
                }
                rows.append(row)
                writer.writerow(_row_for_csv(row))
                cf.flush()  # durable after every question

                if (i + 1) % 20 == 0 or i == 0:
                    print(f"[{i + 1}/{n}] EM={em} F1={f1:.2f} {latency:5.2f}s "
                          f"(llm {res['llm_time_s']:.2f}+tool {res['tool_time_s']:.2f}) "
                          f"tools={res['tool_calls']}  gold={ex['answer']!r} pred={res['answer']!r}")
    return rows


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100)
    lo = int(k); hi = min(lo + 1, len(sorted_vals) - 1); frac = k - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _stats(vals):
    s = sorted(vals); n = len(s) or 1
    return {"mean": round(sum(s) / n, 4), "p50": round(_pct(s, 50), 4),
            "p90": round(_pct(s, 90), 4), "p99": round(_pct(s, 99), 4),
            "total": round(sum(s), 4)}


def build_summary(rows, run_dir):
    n = len(rows) or 1
    usage = Counter()
    for r in rows:
        for e in r["tool_trace"]:
            usage[e["name"]] += 1
    return {
        "config": {
            "n": len(rows),
            "model": MODEL,
            "dataset": "hotpotqa distractor (validation split)",
            "force_first_search": True,
            "max_turns": 6,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "metrics": {
            "questions": len(rows),
            "exact_match": round(sum(r["em"] for r in rows) / n, 4),
            "f1": round(sum(r["f1"] for r in rows) / n, 4),
            "latency_s": _stats([r["latency_s"] for r in rows]),
            "llm_time_s": _stats([r["llm_time_s"] for r in rows]),
            "tool_time_s": _stats([r["tool_time_s"] for r in rows]),
            "tool_calls": _stats([r["tool_calls"] for r in rows]),
            "llm_calls": _stats([r["llm_calls"] for r in rows]),
            "tokens": {
                "prompt_total": sum(r["prompt_tokens"] for r in rows),
                "completion_total": sum(r["completion_tokens"] for r in rows),
                "total": sum(r["total_tokens"] for r in rows),
                "total_mean": round(sum(r["total_tokens"] for r in rows) / n, 1),
            },
            "tool_usage": dict(usage),
        },
        "gpu": summarize_gpu(run_dir / "gpu_metrics.csv"),
    }


def print_summary(summary):
    m = summary["metrics"]; g = summary["gpu"]
    print("\n==== single-agent RAG baseline ====")
    print(f"questions          : {m['questions']}")
    print(f"exact match (EM)   : {m['exact_match']:.3f}")
    print(f"F1                 : {m['f1']:.3f}")
    print(f"latency mean/p90   : {m['latency_s']['mean']:.2f} / {m['latency_s']['p90']:.2f} s"
          f"   (llm {m['llm_time_s']['mean']:.2f} + tool {m['tool_time_s']['mean']:.2f})")
    print(f"tool calls mean    : {m['tool_calls']['mean']:.2f}   usage: {m['tool_usage']}")
    print(f"tokens total       : {m['tokens']['total']:,}"
          f"  (mean {m['tokens']['total_mean']:.0f}/question)")
    if g:
        print(f"GPU {g['active_gpu']} VRAM peak    : {g['vram_peak_mib']:,} / "
              f"{g['vram_total_mib']:,} MiB   util mean/peak {g['util_gpu_mean_pct']}/"
              f"{g['util_gpu_peak_pct']}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of questions")
    ap.add_argument("--start", type=int, default=0, help="first question index")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("logs") / f"baseline_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    gpu_proc, gpu_file = start_gpu_sampler(run_dir / "gpu_metrics.csv")
    try:
        rows = asyncio.run(run_baseline(args.n, args.start, run_dir))
    finally:
        stop_gpu_sampler(gpu_proc, gpu_file)

    summary = build_summary(rows, run_dir)
    print_summary(summary)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nresults  -> {run_dir / 'baseline_results.csv'}")
    print(f"summary  -> {run_dir / 'summary.json'}")
    print(f"gpu      -> {run_dir / 'gpu_metrics.csv'}")

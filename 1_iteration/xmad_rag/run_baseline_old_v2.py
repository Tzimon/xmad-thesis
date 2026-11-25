"""
xmad/1_iteration/xmad_rag/run_baseline.py

Building block 4: the single-agent RAG baseline (C5.1).

Runs the agent loop over N HotpotQA distractor questions through the shared MCP
tool pool, scores answers with the official HotpotQA exact-match and F1 metrics,
and records latency and token consumption -- the reference point the debate is
later measured against (C5.1, C5.3).

Every run writes its per-question results into a timestamped folder under logs/,
matching the Phase-0 verification scheme so results stay comparable across phases.

Requires, on the same node:
  * vLLM serving Qwen with tool calling  (port 7555)
  * the MCP server running                (port 7600)

    python -m xmad_rag.run_baseline --n 50
"""

import argparse
import asyncio
import csv
import time
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from fastmcp import Client as MCPClient

from xmad_rag.baseline_agent_loop import run_agent
from xmad_rag.metrics import exact_match, f1_score
from xmad_rag.retriever import build_corpus

MCP_URL = "http://127.0.0.1:7600/mcp"


async def run_baseline(n):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rows = []
    async with MCPClient(MCP_URL) as mcp:
        for i, ex in enumerate(ds.select(range(n))):
            # Harness step: set this question's corpus on the shared pool.
            await mcp.call_tool("load_corpus", {"paragraphs": build_corpus(ex)})

            t0 = time.perf_counter()
            res = await run_agent(ex["question"])
            latency = time.perf_counter() - t0

            em = exact_match(res["answer"], ex["answer"])
            f1 = f1_score(res["answer"], ex["answer"])
            rows.append({
                "id": ex["id"],
                "question": ex["question"],
                "gold": ex["answer"],
                "pred": res["answer"],
                "pred_raw": res["raw"],
                "em": em,
                "f1": round(f1, 4),
                "latency_s": round(latency, 3),
                "tool_calls": res["tool_calls"],
                "llm_calls": res["llm_calls"],
                "prompt_tokens": res["prompt_tokens"],
                "completion_tokens": res["completion_tokens"],
                "total_tokens": res["total_tokens"],
            })
            print(f"[{i + 1}/{n}] EM={em} F1={f1:.2f} {latency:5.1f}s "
                  f"tools={res['tool_calls']}  gold={ex['answer']!r} pred={res['answer']!r}")
    return rows


def summarize(rows):
    n = len(rows)
    mean = lambda key: sum(r[key] for r in rows) / n if n else 0.0
    print("\n==== single-agent RAG baseline ====")
    print(f"questions          : {n}")
    print(f"exact match (EM)   : {mean('em'):.3f}")
    print(f"F1                 : {mean('f1'):.3f}")
    print(f"latency mean       : {mean('latency_s'):.2f} s")
    print(f"tool calls mean    : {mean('tool_calls'):.2f}")
    print(f"tokens mean (total): {mean('total_tokens'):.0f}"
          f"  (prompt {mean('prompt_tokens'):.0f} + completion {mean('completion_tokens'):.0f})")


def write_csv(rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "baseline_results.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"per-question results -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of questions")
    args = ap.parse_args()

    rows = asyncio.run(run_baseline(args.n))
    summarize(rows)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_csv(rows, Path("logs") / f"baseline_{stamp}")

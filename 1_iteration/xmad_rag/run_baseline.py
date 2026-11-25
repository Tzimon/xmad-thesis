"""
xmad/1_iteration/xmad_rag/run_baseline.py

Single-agent RAG baseline (C5.1), instrumented for the cross-model sweep.

Runs the agent loop over N HotpotQA distractor questions through the shared MCP
tool pool; scores answers with HotpotQA exact-match and F1; records latency
(model vs. tool time), tokens, per-question tool traces, and node GPU/VRAM/POWER.
From the power samples it integrates GPU energy (Wh) and, given a price, the
electricity cost -- the local-cost side of the local-vs-API comparison.

Power method: GPU power via nvidia-smi power.draw (same NVML source hypno uses),
no privileges required. This is GPU energy, not full-node wall power; the platform
overhead (CPU/RAM/NICs/fans/PSU) is not captured and would add a roughly constant
term across models. See the sweep notes.

Each run writes logs/baseline_<label>_<ts>/ with:
    baseline_results.csv  -- per-question rows, written incrementally
    summary.json          -- config + aggregate metrics + GPU energy/cost
    gpu_metrics.csv        -- per-second nvidia-smi samples incl. power.draw

Requires on the same node: vLLM (served as alias 'qwen', port 7555) + MCP server
(port 7600). The orchestrator serves each model under the same alias, so this
script is model-agnostic; --label/--load-time-s/--tp only record what was served.

    python -m xmad_rag.run_baseline --label 7B --load-time-s 42.1 --tp 1
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
ALIAS = "qwen"  # every model is served under this alias by the orchestrator

CSV_COLUMNS = [
    "idx", "id", "question", "gold", "pred", "error", "em", "f1",
    "latency_s", "llm_time_s", "tool_time_s",
    "tool_calls", "llm_calls",
    "prompt_tokens", "completion_tokens", "total_tokens",
    "tool_trace", "pred_raw",
]


# --------------------------------------------------------------------------- #
# GPU sampling: utilisation, memory, and POWER (for energy)
# --------------------------------------------------------------------------- #
def start_gpu_sampler(path):
    try:
        f = open(path, "w")
        proc = subprocess.Popen(
            ["nvidia-smi",
             "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,"
             "memory.used,memory.total,power.draw",
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


def summarize_gpu(path, kwh_price_eur):
    """Reduce nvidia-smi samples to the active GPU(s): peak VRAM, utilisation,
    mean/peak power, and integrated energy (Wh) + its electricity cost."""
    try:
        per = {}
        with open(path) as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) < 7:
                    continue
                try:
                    idx = int(row[1]); util = float(row[2]); mem = float(row[4])
                    total = float(row[5]); power = float(row[6])
                    ts = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                except (ValueError, IndexError):
                    continue
                g = per.setdefault(idx, {"ts": [], "util": [], "mem": [],
                                         "power": [], "total": total})
                g["ts"].append(ts); g["util"].append(util); g["mem"].append(mem)
                g["power"].append(power); g["total"] = total
        if not per:
            return None
        # active = GPUs the model actually occupies (idle GPUs sit at ~4 MiB)
        active = [i for i in per if max(per[i]["mem"]) > 2000] \
            or [max(per, key=lambda k: max(per[k]["mem"]))]
        ts0 = per[active[0]]["ts"]
        duration_s = (ts0[-1] - ts0[0]).total_seconds() if len(ts0) > 1 else 0.0
        sum_mean_power = sum(sum(per[i]["power"]) / len(per[i]["power"]) for i in active)
        peak_power = sum(max(per[i]["power"]) for i in active)
        energy_wh = round(sum_mean_power * duration_s / 3600, 3)
        g0 = per[active[0]]
        return {
            "active_gpus": sorted(active),
            "vram_peak_mib": int(max(g0["mem"])),
            "vram_total_mib": int(g0["total"]),
            "util_gpu_mean_pct": round(sum(g0["util"]) / len(g0["util"]), 1),
            "util_gpu_peak_pct": int(max(g0["util"])),
            "power_mean_w": round(sum_mean_power, 1),
            "power_peak_w": round(peak_power, 1),
            "duration_s": round(duration_s, 1),
            "energy_wh": energy_wh,
            "kwh_price_eur": kwh_price_eur,
            "energy_cost_eur": round(energy_wh / 1000 * kwh_price_eur, 5),
            "samples": len(g0["util"]),
        }
    except FileNotFoundError:
        return None


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def _row_for_csv(row):
    out = dict(row)
    out["tool_trace"] = json.dumps(row["tool_trace"], separators=(",", ":"))
    out["pred_raw"] = row["pred_raw"].replace("\n", " ").strip()
    return out


async def run_baseline(n, start, run_dir):
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    total = len(ds)
    n = (total - start) if n is None else max(0, min(n, total - start))
    print(f"running {n} questions (start={start}; dev split has {total})")
    rows = []
    llm = AsyncOpenAI(base_url=VLLM_URL, api_key="not-needed")

    with open(run_dir / "baseline_results.csv", "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_COLUMNS)
        writer.writeheader(); cf.flush()
        async with MCPClient(MCP_URL) as mcp:
            for i, ex in enumerate(ds.select(range(start, start + n))):
                await mcp.call_tool("load_corpus", {"paragraphs": build_corpus(ex)})
                t0 = time.perf_counter()
                try:
                    res = await run_agent(ex["question"], model=ALIAS, mcp=mcp, llm=llm)
                    error = ""
                except Exception as e:
                    # A malformed tool call (common for tiny models) or any other
                    # failure marks this question wrong and the run continues.
                    error = type(e).__name__
                    res = {"answer": "", "raw": f"[ERROR] {error}: {str(e)[:200]}",
                           "tool_trace": [], "llm_calls": 0, "tool_calls": 0,
                           "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                           "llm_time_s": 0.0, "tool_time_s": 0.0}
                latency = time.perf_counter() - t0
                em = exact_match(res["answer"], ex["answer"])
                f1 = f1_score(res["answer"], ex["answer"])
                row = {
                    "idx": start + i, "id": ex["id"], "question": ex["question"],
                    "gold": ex["answer"], "pred": res["answer"], "error": error,
                    "em": em, "f1": round(f1, 4), "latency_s": round(latency, 3),
                    "llm_time_s": res["llm_time_s"], "tool_time_s": res["tool_time_s"],
                    "tool_calls": res["tool_calls"], "llm_calls": res["llm_calls"],
                    "prompt_tokens": res["prompt_tokens"],
                    "completion_tokens": res["completion_tokens"],
                    "total_tokens": res["total_tokens"],
                    "tool_trace": res["tool_trace"], "pred_raw": res["raw"],
                }
                rows.append(row)
                writer.writerow(_row_for_csv(row)); cf.flush()
                if (i + 1) % 50 == 0 or i == 0:
                    print(f"[{i + 1}/{n}] EM={em} F1={f1:.2f} {latency:4.2f}s "
                          f"gold={ex['answer']!r} pred={res['answer']!r}")
    return rows


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _pct(sv, p):
    if not sv:
        return 0.0
    k = (len(sv) - 1) * (p / 100); lo = int(k); hi = min(lo + 1, len(sv) - 1)
    return sv[lo] * (1 - (k - lo)) + sv[hi] * (k - lo)


def _stats(vals):
    s = sorted(vals); n = len(s) or 1
    return {"mean": round(sum(s) / n, 4), "p50": round(_pct(s, 50), 4),
            "p90": round(_pct(s, 90), 4), "p99": round(_pct(s, 99), 4),
            "total": round(sum(s), 4)}


def build_summary(rows, run_dir, label, load_time_s, tp, kwh_price_eur):
    n = len(rows) or 1
    usage = Counter()
    for r in rows:
        for e in r["tool_trace"]:
            usage[e["name"]] += 1
    return {
        "config": {
            "label": label, "served_alias": ALIAS, "tensor_parallel": tp,
            "load_time_s": load_time_s,
            "dataset": "hotpotqa distractor (validation split)",
            "n": len(rows), "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "metrics": {
            "questions": len(rows),
            "exact_match": round(sum(r["em"] for r in rows) / n, 4),
            "f1": round(sum(r["f1"] for r in rows) / n, 4),
            "latency_s": _stats([r["latency_s"] for r in rows]),
            "llm_time_s": _stats([r["llm_time_s"] for r in rows]),
            "tool_time_s": _stats([r["tool_time_s"] for r in rows]),
            "tool_calls": _stats([r["tool_calls"] for r in rows]),
            "tokens": {
                "prompt_total": sum(r["prompt_tokens"] for r in rows),
                "completion_total": sum(r["completion_tokens"] for r in rows),
                "total": sum(r["total_tokens"] for r in rows),
                "total_mean": round(sum(r["total_tokens"] for r in rows) / n, 1),
            },
            "tool_usage": dict(usage),
            "errors": {
                "total": sum(1 for r in rows if r["error"]),
                "rate": round(sum(1 for r in rows if r["error"]) / n, 4),
                "by_type": dict(Counter(r["error"] for r in rows if r["error"])),
            },
        },
        "gpu": summarize_gpu(run_dir / "gpu_metrics.csv", kwh_price_eur),
    }


def print_summary(s):
    m = s["metrics"]; g = s["gpu"]; c = s["config"]
    print(f"\n==== baseline [{c['label']}] ====")
    print(f"questions   : {m['questions']}")
    print(f"EM / F1     : {m['exact_match']:.3f} / {m['f1']:.3f}")
    print(f"load time   : {c['load_time_s']} s")
    print(f"latency mean: {m['latency_s']['mean']:.2f} s "
          f"(llm {m['llm_time_s']['mean']:.2f} + tool {m['tool_time_s']['mean']:.3f})")
    print(f"tokens total: {m['tokens']['total']:,} "
          f"(prompt {m['tokens']['prompt_total']:,} + compl {m['tokens']['completion_total']:,})")
    e = m["errors"]
    print(f"errors      : {e['total']} ({e['rate']*100:.1f}%)  {e['by_type']}")
    if g:
        print(f"GPU {g['active_gpus']} : power mean/peak {g['power_mean_w']}/{g['power_peak_w']} W, "
              f"energy {g['energy_wh']} Wh, cost EUR {g['energy_cost_eur']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="questions; omit for whole split")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--label", default="model", help="model label for the record")
    ap.add_argument("--load-time-s", type=float, default=None, help="vLLM load time (from orchestrator)")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size served")
    ap.add_argument("--kwh-price-eur", type=float, default=0.30, help="electricity price EUR/kWh")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("logs") / f"baseline_{args.label}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    gpu_proc, gpu_file = start_gpu_sampler(run_dir / "gpu_metrics.csv")
    try:
        rows = asyncio.run(run_baseline(args.n, args.start, run_dir))
    finally:
        stop_gpu_sampler(gpu_proc, gpu_file)

    summary = build_summary(rows, run_dir, args.label, args.load_time_s,
                            args.tp, args.kwh_price_eur)
    print_summary(summary)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n-> {run_dir}/  (summary.json, baseline_results.csv, gpu_metrics.csv)")

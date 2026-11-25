"""
xmad/2_iteration/run_debate.py

The debate harness (C5): runs the debate over N HotpotQA distractor questions,
scores the final answer (consensus or judge) against gold, and records timing,
per-agent token/call stats, round counts, judge usage, and GPU/energy.

Connections (LLM clients, both MCP client sessions) are opened ONCE and reused
across the whole run -- opening fresh connections per question exhausted the
MCP server in Phase 1. Each question's debate is wrapped in try/except: a
failure (e.g. a malformed tool call from a weaker model) is recorded as an
error and scored as wrong, and the run continues -- the same lesson from the
Phase 1 cross-model sweep, where an uncaught error would otherwise abort a
multi-hour run.

Each run writes logs/debate_<label>_<ts>/ containing:
    debate_results.csv   -- one row per question, written incrementally
    summary.json         -- aggregate metrics, including a GPU/energy summary
    gpu_metrics.csv       -- per-second nvidia-smi samples (power-aware)
    sessions/<idx>.json   -- the full per-question DebateSession (if
                             logging.save_full_session is true in the config)

Requires, on the same node (see start_phase2_services.sh):
  * vLLM serving agent A                  (config: agents.agent_a.url)
  * vLLM serving agent B                  (config: agents.agent_b.url)
  * the MCP server                        (config: tools.mcp_url)

    python run_debate.py                              # config/debate.yaml, whole split
    python run_debate.py --n 50                        # a quick sanity slice
    python run_debate.py --config config/other.yaml
"""

import argparse
import asyncio
import csv
import json
import subprocess
import time
from random import Random
from collections import Counter
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from fastmcp import Client as MCPClient
from openai import AsyncOpenAI

from xmad_rag.metrics import exact_match, f1_score
from xmad_rag.retriever import build_corpus

from xmad_debate.config import load_config
from xmad_debate.debate_loop import run_debate
from xmad_debate.session import DebateSession

CSV_COLUMNS = [
    "idx", "id", "question", "gold", "pred", "source", "em", "f1",
    "rounds", "consensus_round", "stop_reason", "used_judge", "judge_confidence",
    "wall_time_s",
    "a_llm_calls", "a_tool_calls", "a_total_tokens",
    "b_llm_calls", "b_tool_calls", "b_total_tokens",
    "error", "error_detail",
]


# --------------------------------------------------------------------------- #
# GPU/energy sampling -- same method as the Phase 1 cross-model sweep. Already
# sums power across every GPU currently above the idle threshold, which is
# exactly what a heterogeneous debate needs: agent A and agent B typically run
# on two DIFFERENT GPUs at once (not one model split across GPUs, but two
# models active simultaneously), and both must be counted.
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
        active = [i for i in per if max(per[i]["mem"]) > 2000] \
            or [max(per, key=lambda k: max(per[k]["mem"]))]
        ts0 = per[active[0]]["ts"]
        duration_s = (ts0[-1] - ts0[0]).total_seconds() if len(ts0) > 1 else 0.0
        sum_mean_power = sum(sum(per[i]["power"]) / len(per[i]["power"]) for i in active)
        peak_power = sum(max(per[i]["power"]) for i in active)
        energy_wh = round(sum_mean_power * duration_s / 3600, 3)
        return {
            "active_gpus": sorted(active),
            "power_mean_w": round(sum_mean_power, 1),
            "power_peak_w": round(peak_power, 1),
            "duration_s": round(duration_s, 1),
            "energy_wh": energy_wh,
            "kwh_price_eur": kwh_price_eur,
            "energy_cost_eur": round(energy_wh / 1000 * kwh_price_eur, 5),
        }
    except FileNotFoundError:
        return None


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
def _row_for_csv(row):
    out = dict(row)
    return out


def random_sample_indices(total, n, seed):
    """A reproducible, uniformly random sample of n indices out of
    range(total), sorted for a stable, readable order in the results CSV.
    Same seed -> the exact same sample every time, so every debate condition
    (homogeneous, heterogeneous, ...) reuses identical questions just by
    passing the same --seed. Chosen over a contiguous "first N" slice because
    the dataset's on-disk ordering is not documented as randomised -- a plain
    random sample avoids any risk of a contiguous block happening to cluster
    by topic or collection batch."""
    rng = Random(seed)
    idxs = list(range(total))
    rng.shuffle(idxs)
    return sorted(idxs[:n])


async def run_all(cfg, run_dir, seed=None):
    ds = load_dataset(cfg.dataset.name, cfg.dataset.config, split=cfg.dataset.split)
    total = len(ds)

    if seed is not None:
        # Reproducible random sample instead of a contiguous slice.
        n = cfg.run.n_questions or 1000
        indices = random_sample_indices(total, n, seed)
        print(f"random sample: {len(indices)} of {total} questions (seed={seed})")
        questions = [ds[i] for i in indices]
    else:
        start = cfg.run.start
        n = (total - start) if cfg.run.n_questions is None else max(0, min(cfg.run.n_questions, total - start))
        print(f"running {n} questions (start={start}; split has {total})")
        indices = list(range(start, start + n))
        questions = list(ds.select(indices))

    n = len(indices)  # used by the progress print below, regardless of which path was taken

    debate_system_prompt = cfg.read_prompt(cfg.prompts.debate_system_prompt_file)
    judge_prompt = cfg.read_prompt(cfg.judge.prompt_file) if cfg.judge.enabled else None
    allowed_tools = set(cfg.tools.agent_tools)

    # one LLM client per UNIQUE endpoint (so the homogeneous condition, where
    # agent_a and agent_b share a url, reuses a single client, not two)
    urls = {cfg.agents.agent_a.url, cfg.agents.agent_b.url}
    if cfg.judge.enabled:
        urls.add(cfg.judge.url)
    llm_clients = {u: AsyncOpenAI(base_url=u, api_key="not-needed") for u in urls}

    sessions_dir = run_dir / "sessions"
    if cfg.logging.save_full_session:
        sessions_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    csv_path = run_dir / "debate_results.csv"
    with open(csv_path, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_COLUMNS)
        writer.writeheader(); cf.flush()

        # TWO dedicated MCP connections, one per agent, opened once and reused
        # for the whole run (see debate_loop.py's module docstring for why).
        async with MCPClient(cfg.tools.mcp_url) as mcp_a, MCPClient(cfg.tools.mcp_url) as mcp_b:
            for i, (idx, ex) in enumerate(zip(indices, questions)):
                await mcp_a.call_tool("load_corpus", {"paragraphs": build_corpus(ex)})

                t0 = time.perf_counter()
                error, error_detail = "", ""
                session = DebateSession(
                    ex["question"],
                    {"agent_a": cfg.agents.agent_a.alias, "agent_b": cfg.agents.agent_b.alias},
                    cfg.debate.max_rounds,
                    stalemate_patience=cfg.debate.stalemate_patience,
                )
                try:
                    session = await run_debate(
                        ex["question"], session,
                        cfg.agents.agent_a, cfg.agents.agent_b,
                        llm_clients, mcp_a, mcp_b,
                        cfg.decoding, allowed_tools, debate_system_prompt,
                        cfg.debate, judge_cfg=cfg.judge, judge_prompt=judge_prompt,
                    )
                    pred, source = session.final_answer()
                except Exception as e:
                    # NOTE: `session` is NOT discarded here -- run_debate mutates it
                    # in place via record_turn(), so any rounds completed before the
                    # failure are still present and still get saved/logged below.
                    error = type(e).__name__
                    error_detail = str(e)[:300]
                    pred, source = "", "error"
                wall_time = time.perf_counter() - t0

                em = exact_match(pred, ex["answer"]) if pred else 0
                f1 = f1_score(pred, ex["answer"]) if pred else 0.0

                a_stats = _agent_totals(session, "agent_a") if session else _zero_stats()
                b_stats = _agent_totals(session, "agent_b") if session else _zero_stats()

                row = {
                    "idx": idx, "id": ex["id"], "question": ex["question"],
                    "gold": ex["answer"], "pred": pred, "source": source,
                    "em": em, "f1": round(f1, 4),
                    "rounds": len(session.rounds) if session else 0,
                    "consensus_round": session.consensus_round if session else None,
                    "stop_reason": session.stop_reason() if session else "error",
                    "used_judge": bool(session and session.judge_verdict is not None),
                    "judge_confidence": (session.judge_verdict or {}).get("confidence") if session else None,
                    "wall_time_s": round(wall_time, 3),
                    "a_llm_calls": a_stats["llm_calls"], "a_tool_calls": a_stats["tool_calls"],
                    "a_total_tokens": a_stats["total_tokens"],
                    "b_llm_calls": b_stats["llm_calls"], "b_tool_calls": b_stats["tool_calls"],
                    "b_total_tokens": b_stats["total_tokens"],
                    "error": error,
                    "error_detail": error_detail,
                }
                rows.append(row)
                writer.writerow(_row_for_csv(row)); cf.flush()

                if session is not None and cfg.logging.save_full_session:
                    with open(sessions_dir / f"{idx}.json", "w") as sf:
                        sf.write(session.to_json())

                if (i + 1) % 20 == 0 or i == 0:
                    print(f"[{i + 1}/{n}] EM={em} F1={f1:.2f} src={source} "
                          f"rounds={row['rounds']} {wall_time:5.2f}s  "
                          f"gold={ex['answer']!r} pred={pred!r}"
                          + (f"  ERROR={error}" if error else ""))
    return rows


def _agent_totals(session, key):
    stats = {"llm_calls": 0, "tool_calls": 0, "total_tokens": 0}
    for rnd in session.rounds:
        turn = rnd.turns.get(key)
        if turn:
            stats["llm_calls"] += turn.llm_calls
            stats["tool_calls"] += turn.tool_calls
            stats["total_tokens"] += turn.total_tokens
    return stats


def _zero_stats():
    return {"llm_calls": 0, "tool_calls": 0, "total_tokens": 0}


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
            "p90": round(_pct(s, 90), 4), "total": round(sum(s), 4)}


def build_summary(rows, cfg, run_dir, kwh_price_eur, seed=None):
    n = len(rows) or 1
    errs = Counter(r["error"] for r in rows if r["error"])
    return {
        "config": {
            "label": cfg.run.label,
            "agent_a": {"alias": cfg.agents.agent_a.alias, "url": cfg.agents.agent_a.url},
            "agent_b": {"alias": cfg.agents.agent_b.alias, "url": cfg.agents.agent_b.url},
            "homogeneous": cfg.agents.agent_a.url == cfg.agents.agent_b.url,
            "max_rounds": cfg.debate.max_rounds,
            "temperature": cfg.decoding.temperature,
            "n": len(rows),
            "question_selection": (
                {"mode": "random", "seed": seed}
                if seed is not None else
                {"mode": "contiguous", "start": cfg.run.start}
            ),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        "metrics": {
            "questions": len(rows),
            "exact_match": round(sum(r["em"] for r in rows) / n, 4),
            "f1": round(sum(r["f1"] for r in rows) / n, 4),
            "consensus_rate": round(sum(1 for r in rows if r["source"] == "consensus") / n, 4),
            "judge_rate": round(sum(1 for r in rows if r["source"] == "judge") / n, 4),
            "stop_reason_counts": dict(Counter(r["stop_reason"] for r in rows if r["stop_reason"])),
            "rounds": _stats([r["rounds"] for r in rows]),
            "wall_time_s": _stats([r["wall_time_s"] for r in rows]),
            "tokens": {
                "agent_a_total": sum(r["a_total_tokens"] for r in rows),
                "agent_b_total": sum(r["b_total_tokens"] for r in rows),
            },
            "errors": {
                "total": sum(errs.values()),
                "rate": round(sum(errs.values()) / n, 4),
                "by_type": dict(errs),
            },
        },
        "gpu": summarize_gpu(run_dir / "gpu_metrics.csv", kwh_price_eur),
    }


def print_summary(s):
    m = s["metrics"]; c = s["config"]; g = s["gpu"]
    print(f"\n==== debate [{c['label']}] "
          f"({'homogeneous' if c['homogeneous'] else 'heterogeneous'}) ====")
    print(f"questions        : {m['questions']}")
    print(f"exact match / F1 : {m['exact_match']:.3f} / {m['f1']:.3f}")
    print(f"consensus rate   : {m['consensus_rate']:.3f}   judge rate: {m['judge_rate']:.3f}")
    print(f"rounds mean/p90  : {m['rounds']['mean']:.2f} / {m['rounds']['p90']:.2f}")
    print(f"tokens (a / b)   : {m['tokens']['agent_a_total']:,} / {m['tokens']['agent_b_total']:,}")
    print(f"errors           : {m['errors']['total']} ({m['errors']['rate']*100:.1f}%)  {m['errors']['by_type']}")
    if g:
        print(f"GPU {g['active_gpus']}: power mean {g['power_mean_w']} W, "
              f"energy {g['energy_wh']} Wh, cost EUR {g['energy_cost_eur']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/debate.yaml")
    ap.add_argument("--n", type=int, default=None, help="override run.n_questions")
    ap.add_argument("--start", type=int, default=None, help="override run.start")
    ap.add_argument("--kwh-price-eur", type=float, default=0.069)
    ap.add_argument("--seed", type=int, default=None,
                    help="If set, run a reproducible RANDOM sample of --n questions "
                         "from the whole split instead of a contiguous slice at --start. "
                         "Use the same --n and --seed for every condition (homogeneous, "
                         "heterogeneous, ...) so they all run on identical questions.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.n is not None:
        cfg.run.n_questions = args.n
    if args.start is not None:
        cfg.run.start = args.start

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.resolve(cfg.logging.log_dir) / f"debate_{cfg.run.label}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    gpu_proc, gpu_file = start_gpu_sampler(run_dir / "gpu_metrics.csv")
    try:
        rows = asyncio.run(run_all(cfg, run_dir, args.seed))
    finally:
        stop_gpu_sampler(gpu_proc, gpu_file)

    summary = build_summary(rows, cfg, run_dir, args.kwh_price_eur, args.seed)
    print_summary(summary)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nresults  -> {run_dir / 'debate_results.csv'}")
    print(f"summary  -> {run_dir / 'summary.json'}")
    print(f"sessions -> {run_dir / 'sessions'}/  (per-question full transcripts)")
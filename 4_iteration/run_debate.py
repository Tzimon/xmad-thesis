"""
xmad/4_iteration/run_debate.py

Debate harness for ONE run folder. Normally started by phase4.py, which has
already written <run_dir>/run.yaml and started the model servers and the MCP
server. Can also be started by hand for debugging:

    python run_debate.py --run-dir results/P4-MSQ-001_qwen2.5-7b__qwen2.5-7b
    python run_debate.py --run-dir ... --n 20          # quick slice (first 20 of the sample)
    python run_debate.py --run-dir ... --summarize-only  # rebuild summary.json (after the judge pass)

Same structure as Phase 2 (connections opened once, per-question try/except,
incremental CSV, one session JSON per question). Phase 3 additions:
  * Resume: questions already in debate_results.csv are skipped and new rows
    appended, so a run that was stopped continues where it stopped. Each
    attempt writes its own gpu_metrics_<n>.csv.
  * GPU energy only on THIS run's GPUs (nvidia-smi -i), reported per agent.
    Fixes the Phase 2 issue where a loaded but unused model on another GPU
    was counted. Homogeneous runs: one server, so agent A == agent B GPUs.
  * Round-1 answers of both agents (matched single-agent baseline) and the
    round-1 evidence overlap are written into the CSV.
  * Debates without consensus get source "pending_judge" (judge pass later).

Phase 4 changes (everything else is the Phase 3 harness, line for line):
  * Benchmark-independent: the questions come from xmad_bench (load_benchmark),
    selected by the `dataset` block of run.yaml (name = benchmark). The harness
    only sees Example objects (question, answers, paragraphs, gold titles), so
    a new benchmark needs an adapter, not a change here (C3.3).
  * Scoring is the max over all gold answers (MuSiQue aliases); the gold list is
    stored in the CSV (gold_answers) so the judge pass can re-score.
  * Prompt and completion tokens per agent are written to the CSV (needed for
    API-equivalent prices; in Phase 3 they had to be recovered from the sessions).
  * Evidence recall: share of the supporting paragraphs among the titles an
    agent retrieved, in round 1 (per agent and both together) and over all rounds.
"""

import argparse
import asyncio
import csv
import glob
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "1_iteration"))   # xmad_rag, if not installed editable

from xmad_bench import em_max, evidence_recall, f1_max, load_benchmark  # noqa: E402
from xmad_debate.config import load_run_config  # noqa: E402
from xmad_debate.session import NO_ANSWER, DebateSession  # noqa: E402

TOTALS = ("llm_calls", "tool_calls", "total_tokens", "prompt_tokens", "completion_tokens",
          "truncated", "api_retries", "forced_final", "invalid_tool_calls")

CSV_COLUMNS = [
    "idx", "id", "benchmark", "q_meta", "question", "gold", "gold_answers", "pred", "source", "em", "f1",
    "rounds", "consensus_round", "stop_reason", "judge_confidence", "wall_time_s",
    "a_r1_pred", "a_r1_em", "a_r1_f1", "b_r1_pred", "b_r1_em", "b_r1_f1", "r1_jaccard",
    "a_r1_recall", "b_r1_recall", "r1_recall_union", "recall_all_rounds", "n_gold_paragraphs",
    *[f"a_{k}" for k in TOTALS],
    *[f"b_{k}" for k in TOTALS],
    "error", "error_detail",
]


def sample_indices(total, n, seed):
    """Identical to Phase 2: seed 42, n 1000 -> the same 1000 questions."""
    rng = Random(seed)
    idxs = list(range(total))
    rng.shuffle(idxs)
    return sorted(idxs[:n])


def read_rows(path):
    if not Path(path).exists():
        return []
    csv.field_size_limit(10**9)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------- #
# GPU sampling, restricted to this run's GPUs
# --------------------------------------------------------------------------- #
def start_gpu_sampler(path, gpus):
    cmd = ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw",
           "--format=csv,nounits", "-lms", "1000", "-i", ",".join(str(g) for g in gpus)]
    try:
        f = open(path, "w")
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.DEVNULL), f
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


def energy_per_gpu(run_dir):
    """{gpu: (energy_wh, mean_w, peak_vram_mib)} summed over all attempts."""
    out = {}
    for path in sorted(glob.glob(str(Path(run_dir) / "gpu_metrics_*.csv"))):
        per = {}
        with open(path) as fh:
            for row in csv.reader(fh):
                try:
                    ts = datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
                    g, mem, p = int(row[1]), float(row[3]), float(row[4])
                except (ValueError, IndexError):
                    continue
                per.setdefault(g, []).append((ts, mem, p))
        for g, s in per.items():
            dur = (s[-1][0] - s[0][0]).total_seconds() if len(s) > 1 else 0.0
            mean_w = sum(x[2] for x in s) / len(s)
            e, _, vram = out.get(g, (0.0, 0.0, 0))
            out[g] = (e + mean_w * dur / 3600, mean_w, max(vram, int(max(x[1] for x in s))))
    return out


# --------------------------------------------------------------------------- #
# Per-question helpers
# --------------------------------------------------------------------------- #
def _jaccard(a, b):
    a, b = set(a), set(b)
    return None if not a or not b else len(a & b) / len(a | b)


def _titles(turn):
    return [t for c in turn.tool_trace for t in c.get("retrieved_titles", [])]


def _r(x):
    return "" if x is None else round(x, 4)


def round1_fields(session, ex):
    out = dict.fromkeys(["a_r1_pred", "a_r1_em", "a_r1_f1", "b_r1_pred", "b_r1_em", "b_r1_f1", "r1_jaccard",
                         "a_r1_recall", "b_r1_recall", "r1_recall_union", "recall_all_rounds"], "")
    out["n_gold_paragraphs"] = len(ex.gold_titles)
    done = session.completed_rounds()
    if not done:
        return out
    r1 = done[0]
    for key, p in (("agent_a", "a"), ("agent_b", "b")):
        ans = r1.turns[key].final_answer
        out[f"{p}_r1_pred"] = ans
        out[f"{p}_r1_em"] = em_max(ans, ex.answers)
        out[f"{p}_r1_f1"] = round(f1_max(ans, ex.answers), 4)
    titles = {k: _titles(r1.turns[k]) for k in ("agent_a", "agent_b")}
    j = _jaccard(titles["agent_a"], titles["agent_b"])
    out["r1_jaccard"] = "" if j is None else round(j, 4)
    # evidence recall: which supporting paragraphs did the agents actually see?
    out["a_r1_recall"] = _r(evidence_recall(titles["agent_a"], ex.gold_titles))
    out["b_r1_recall"] = _r(evidence_recall(titles["agent_b"], ex.gold_titles))
    out["r1_recall_union"] = _r(evidence_recall(titles["agent_a"] + titles["agent_b"], ex.gold_titles))
    every = [t for rnd in done for k in ("agent_a", "agent_b") if k in rnd.turns for t in _titles(rnd.turns[k])]
    out["recall_all_rounds"] = _r(evidence_recall(every, ex.gold_titles))
    return out


def agent_totals(session, key):
    t = Counter()
    for rnd in session.rounds:
        turn = rnd.turns.get(key)
        if turn:
            for f in TOTALS:
                t[f] += getattr(turn, f)
    return t


def row_for(idx, ex, benchmark, session, pred, source, wall, error="", error_detail=""):
    em = em_max(pred, ex.answers)
    f1 = f1_max(pred, ex.answers)
    a, b = agent_totals(session, "agent_a"), agent_totals(session, "agent_b")
    return {
        "idx": idx, "id": ex.id, "benchmark": benchmark, "q_meta": json.dumps(ex.meta),
        "question": ex.question, "gold": ex.answer, "gold_answers": json.dumps(ex.answers),
        "pred": pred, "source": source, "em": em, "f1": round(f1, 4),
        "rounds": len(session.completed_rounds()), "consensus_round": session.consensus_round,
        "stop_reason": session.stop_reason() or ("error" if error else ""),
        "judge_confidence": (session.judge_verdict or {}).get("confidence", ""),
        "wall_time_s": round(wall, 3),
        **round1_fields(session, ex),
        **{f"a_{k}": a[k] for k in TOTALS},
        **{f"b_{k}": b[k] for k in TOTALS},
        "error": error, "error_detail": error_detail,
    }


CONNECTION_LOST = ("not connected", "Connection closed", "ReadError", "ClosedResourceError")


def connection_lost(e_or_text):
    """True if the connection to the MCP tool server is gone. Added 2026-10-31: in
    P4-2WK-023 one MCP session dropped after 258 questions and every later question
    failed in 0 s ("Client is not connected", 742 rows). Such a run now stops instead
    of writing error rows; the questions done before stay valid."""
    text = e_or_text if isinstance(e_or_text, str) else f"{type(e_or_text).__name__}: {e_or_text}"
    return any(s in text for s in CONNECTION_LOST)


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #
async def run_all(cfg, run_dir, n_override=None):
    from fastmcp import Client as MCPClient
    from openai import AsyncOpenAI
    from xmad_debate.debate_loop import run_debate

    bench = cfg.dataset.name
    ds = load_benchmark(bench, cfg.dataset.config, cfg.dataset.split)
    indices = sample_indices(len(ds), cfg.run.n_questions, cfg.run.seed)
    if n_override:
        indices = indices[:n_override]

    csv_path = run_dir / "debate_results.csv"
    done = {int(r["idx"]) for r in read_rows(csv_path)}
    todo = [i for i in indices if i not in done]
    print(f"[{cfg.run.run_id}] {len(indices)} questions in sample, {len(done)} already done, "
          f"{len(todo)} to go", flush=True)

    a, b = cfg.agents.agent_a, cfg.agents.agent_b
    llm = {u: AsyncOpenAI(base_url=u, api_key="not-needed", timeout=900) for u in {a.url, b.url}}
    (run_dir / "sessions").mkdir(exist_ok=True)

    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as cf:
        writer = csv.DictWriter(cf, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader(); cf.flush()
        async with MCPClient(cfg.tools.mcp_url) as mcp_a, MCPClient(cfg.tools.mcp_url) as mcp_b:
            for i, idx in enumerate(todo):
                ex = ds[idx]
                await mcp_a.call_tool("load_corpus", {"paragraphs": ex.paragraphs})
                session = DebateSession(ex.question, {"agent_a": a.alias, "agent_b": b.alias},
                                        cfg.debate.max_rounds,
                                        stalemate_patience=cfg.debate.stalemate_patience)
                t0 = time.perf_counter()
                try:
                    session = await run_debate(ex.question, session, cfg, llm, mcp_a, mcp_b)
                    pred, source = session.final_answer()
                    row = row_for(idx, ex, bench, session, pred, source, time.perf_counter() - t0)
                except Exception as e:
                    if connection_lost(e):
                        raise          # no row: the run fails and `run --retry-failed` resumes at this question
                    row = row_for(idx, ex, bench, session, "", "error", time.perf_counter() - t0,
                                  type(e).__name__, str(e)[:300])
                writer.writerow(row); cf.flush()
                with open(run_dir / "sessions" / f"{idx}.json", "w", encoding="utf-8") as sf:
                    sf.write(session.to_json())
                n_done = len(done) + i + 1
                if n_done % 25 == 0 or i == 0:
                    print(f"[{cfg.run.run_id}] {n_done}/{len(indices)}  EM so far "
                          f"{sum(int(r['em']) for r in read_rows(csv_path)) / n_done:.3f}  "
                          f"last: src={row['source']} rounds={row['rounds']} {row['wall_time_s']}s"
                          + (f"  ERROR={row['error']}" if row["error"] else ""), flush=True)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _num(v, cast=float):
    try:
        return cast(v)
    except (TypeError, ValueError):
        return 0


def _pct(vals, p):
    s = sorted(vals)
    if not s:
        return 0.0
    k = (len(s) - 1) * p / 100; lo = int(k); hi = min(lo + 1, len(s) - 1)
    return s[lo] * (1 - (k - lo)) + s[hi] * (k - lo)


def build_summary(cfg, run_dir, kwh_price_eur):
    rows = read_rows(run_dir / "debate_results.csv")
    n = len(rows) or 1
    src = Counter(r["source"] for r in rows)
    jacc = [_num(r["r1_jaccard"]) for r in rows if r["r1_jaccard"] not in ("", None)]

    def mean_of(col):
        v = [_num(r.get(col)) for r in rows if r.get(col) not in ("", None)]
        return round(sum(v) / len(v), 4) if v else None
    wall = [_num(r["wall_time_s"]) for r in rows]
    a, b = cfg.agents.agent_a, cfg.agents.agent_b
    e = energy_per_gpu(run_dir)
    e_a = sum(e[g][0] for g in a.gpus if g in e)
    e_b = 0.0 if b.gpus == a.gpus else sum(e[g][0] for g in b.gpus if g in e)
    repair = {}
    if (run_dir / "repair.json").exists():
        repair = json.loads((run_dir / "repair.json").read_text(encoding="utf-8"))
        e_a += repair["v1_energy_agent_a_wh"] * repair["kept_fraction"]
        e_b += repair["v1_energy_agent_b_wh"] * repair["kept_fraction"]
    consensus = [r for r in rows if r["source"] == "consensus"]
    return {
        "run_id": cfg.run.run_id,
        "label": cfg.run.label,
        "benchmark": cfg.dataset.name,
        "agent_a": a.model_key, "agent_b": b.model_key,
        "homogeneous": a.model_key == b.model_key,
        "questions": len(rows),
        "complete": len(rows) >= cfg.run.n_questions,
        "judge_pending": src.get("pending_judge", 0),
        "exact_match": round(sum(_num(r["em"], int) for r in rows) / n, 4),
        "f1": round(sum(_num(r["f1"]) for r in rows) / n, 4),
        "exact_match_consensus_only": round(sum(_num(r["em"], int) for r in consensus) / len(consensus), 4)
                                      if consensus else None,
        "source_counts": dict(src),
        "consensus_rate": round(src.get("consensus", 0) / n, 4),
        "judge_rate": round((src.get("judge", 0) + src.get("pending_judge", 0)) / n, 4),
        "stop_reason_counts": dict(Counter(r["stop_reason"] for r in rows)),
        "round1_single_agent_em": {"agent_a": round(sum(_num(r["a_r1_em"], int) for r in rows) / n, 4),
                                   "agent_b": round(sum(_num(r["b_r1_em"], int) for r in rows) / n, 4)},
        "evidence_jaccard_round1": round(sum(jacc) / len(jacc), 4) if jacc else None,
        "evidence_recall": {"agent_a_round1": mean_of("a_r1_recall"), "agent_b_round1": mean_of("b_r1_recall"),
                            "union_round1": mean_of("r1_recall_union"), "all_rounds": mean_of("recall_all_rounds")},
        "rounds_mean": round(sum(_num(r["rounds"], int) for r in rows) / n, 3),
        "wall_time_s": {"mean": round(sum(wall) / n, 3), "p50": round(_pct(wall, 50), 3),
                        "p90": round(_pct(wall, 90), 3), "p99": round(_pct(wall, 99), 3),
                        "total": round(sum(wall), 1)},
        "tokens": {"agent_a": sum(_num(r["a_total_tokens"], int) for r in rows),
                   "agent_b": sum(_num(r["b_total_tokens"], int) for r in rows)},
        "tokens_split": {f"{p}_{k}": sum(_num(r[f"{p}_{k}_tokens"], int) for r in rows)
                         for p in ("a", "b") for k in ("prompt", "completion")},
        "truncated_replies": {"agent_a": sum(_num(r["a_truncated"], int) for r in rows),
                              "agent_b": sum(_num(r["b_truncated"], int) for r in rows)},
        "api_retries": sum(_num(r["a_api_retries"], int) + _num(r["b_api_retries"], int) for r in rows),
        "forced_final_answers": {"agent_a": sum(_num(r.get("a_forced_final"), int) for r in rows),
                                 "agent_b": sum(_num(r.get("b_forced_final"), int) for r in rows)},
        "no_answer_round1": {"agent_a": sum(1 for r in rows if r["a_r1_pred"] in ("", NO_ANSWER)),
                             "agent_b": sum(1 for r in rows if r["b_r1_pred"] in ("", NO_ANSWER))},
        "no_answer_final": sum(1 for r in rows if r["pred"] == NO_ANSWER),
        "invalid_tool_calls": {"agent_a": sum(_num(r.get("a_invalid_tool_calls"), int) for r in rows),
                               "agent_b": sum(_num(r.get("b_invalid_tool_calls"), int) for r in rows)},
        "errors": dict(Counter(r["error"] for r in rows if r["error"])),
        "energy": {
            "gpus_agent_a": a.gpus, "gpus_agent_b": b.gpus,
            "agent_a_wh": round(e_a, 2), "agent_b_wh": round(e_b, 2),
            "total_wh": round(e_a + e_b, 2),
            "per_gpu": {str(g): {"wh": round(v[0], 2), "vram_peak_mib": v[2]} for g, v in sorted(e.items())},
            "kwh_price_eur": kwh_price_eur,
            "cost_eur": round((e_a + e_b) / 1000 * kwh_price_eur, 5),
            "note": "GPU power only (nvidia-smi), only the GPUs assigned to this run"
                    + ("; repaired run: re-run part measured + kept share of the first attempt (estimate)"
                       if repair else ""),
        },
        "updated": datetime.now().isoformat(timespec="seconds"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n", type=int, default=None, help="only the first n questions of the sample")
    ap.add_argument("--summarize-only", action="store_true")
    ap.add_argument("--kwh-price-eur", type=float, default=0.069)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg = load_run_config(run_dir / "run.yaml", HERE)

    if not args.summarize_only:
        gpus = sorted(set(cfg.agents.agent_a.gpus) | set(cfg.agents.agent_b.gpus))
        attempt = len(glob.glob(str(run_dir / "gpu_metrics_*.csv")))
        proc, fh = start_gpu_sampler(run_dir / f"gpu_metrics_{attempt}.csv", gpus)
        try:
            asyncio.run(run_all(cfg, run_dir, args.n))
        finally:
            stop_gpu_sampler(proc, fh)

    summary = build_summary(cfg, run_dir, args.kwh_price_eur)
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("run_id", "questions", "exact_match", "f1",
                                              "consensus_rate", "judge_pending")}), flush=True)


if __name__ == "__main__":
    main()

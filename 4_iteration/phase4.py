"""
xmad/4_iteration/phase4.py

Runs the Phase 4 experiment on ONE node with 8 GPUs and keeps the results
readable for a human. Built from 3_iteration/phase3.py; the scheduler, the run
folders, the judge pass and the guards are unchanged. Phase 4 changes:
  * every run in runs.yaml names a `benchmark` (musique, 2wiki, hotpotqa); its
    dataset block comes from config/base.yaml -> benchmarks (C3.3)
  * run IDs carry the benchmark: P4-MSQ-001, P4-2WK-001 (P4-HPQ-001 for regression checks)
  * `run --benchmark musique` selects the runs of one benchmark
  * the judge pass re-scores against all gold answers (MuSiQue aliases)
  * `repair` (a one-off fix of Phase 3) is not carried over

    python phase4.py status                          # rebuild results/OVERVIEW.md, print counts
    python phase4.py run --dry-run                   # show which runs would start on which GPUs
    python phase4.py run                             # work through runs.yaml (must, should, could)
    python phase4.py run --benchmark musique         # only the MuSiQue runs
    python phase4.py run --only P4-MSQ-001           # selected runs
    python phase4.py run --n 20 --only P4-MSQ-001    # quick test: 20 questions (folder suffix _test)
    python phase4.py judge                           # judge pass over all finished runs
    python phase4.py remaining --priority must       # runs still to do (used by run_phase4.sbatch)
    python phase4.py drop-errors                     # remove questions hit by a lost MCP connection, then
                                                     # run --retry-failed computes them again

Pre-test before the real runs (homogeneous pairs, 30 questions, folders *_test):
    python phase4.py run --homogeneous --n 30
    python phase4.py sanity --test                   # flags implausible models (see SANITY_RULES)

How `run` works
  * The runs of runs.yaml are taken in file order (priority, then the order of
    the file). A run starts as soon as enough GPUs are free, so several small
    runs share the node while a large one waits. If the next run in line has
    waited for more than 30 minutes, no smaller runs are started past it until
    it got its GPUs (so large runs are not starved).
  * Each run gets its own GPUs, ports and MCP server. Its model servers are
    started, the debate runs, the servers are stopped, the GPUs are freed.
  * Only models with a PASS in smoke/ are used; other runs show as "blocked".
  * Everything of a run is in results/<run_id>_<label>[_rep<n>]/:
        run.yaml            exact configuration
        status.json         state, GPUs, ports, times, runs that ran at the same time
        debate.log          progress of the debate harness
        server_logs/        vLLM and MCP logs
        debate_results.csv  one row per question
        summary.json        metrics of the run (EM, F1, consensus, energy per agent ...)
        sessions/           full debate per question (not in git)
  * Stopping (Ctrl+C or job end) is safe: unfinished runs show as
    "interrupted" and continue where they stopped on the next `run`.

How `judge` works
  Debates without consensus end as "pending_judge". The judge pass starts the
  judge model once (models.yaml, role: judge), decides all pending debates of
  all finished runs, writes the verdict into each session file and the CSV,
  and rebuilds summary.json and the overview.
"""

import argparse
import asyncio
import csv
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from types import SimpleNamespace
from pathlib import Path

import yaml

import servers

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SMOKE = HERE / "smoke"
PRIORITIES = ["must", "should", "could"]
STARVATION_S = 30 * 60
POLL_S = 10

_lock = threading.Lock()
_procs = {}          # run_id -> list of Popen (for clean shutdown)
_stop = threading.Event()


def now():
    return datetime.now().isoformat(timespec="seconds")


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Plan: runs.yaml + models.yaml + smoke results
# --------------------------------------------------------------------------- #
def load_runs():
    return servers.load_yaml(HERE / "runs.yaml")


def folder_name(run, test=False):
    name = f"{run['run_id']}_{run['label']}"
    if run.get("replicate", 1) > 1:
        name += f"_rep{run['replicate']}"
    return name + ("_test" if test else "")


def run_dir(run, test=False):
    return RESULTS / folder_name(run, test)


def smoke_result(key):
    p = SMOKE / f"{key}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, data):
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def state_of(run, test=False):
    st = read_json(run_dir(run, test) / "status.json", {})
    return st.get("state", "pending")


_MODELS = None


def blocked_reason(run):
    global _MODELS
    _MODELS = _MODELS or servers.load_models()
    for k in {run["agent_a"], run["agent_b"]}:
        if _MODELS[k].get("excluded"):
            return f"{k}: excluded ({_MODELS[k]['excluded'][:60]})"
        s = smoke_result(k)
        if s is None:
            return f"{k}: no smoke test"
        if not s["ok"]:
            return f"{k}: smoke FAIL"
    return ""


def dataset_of(run, base):
    """The dataset block of a run's benchmark (config/base.yaml -> benchmarks)."""
    b = run.get("benchmark")
    if b not in base["benchmarks"]:
        raise KeyError(f"{run['run_id']}: unknown benchmark {b!r}; known: {sorted(base['benchmarks'])}")
    return base["benchmarks"][b]


def build_run_config(run, gpus_a, gpus_b, ports, base, models, sample, n_override):
    def agent(key, gpus, port):
        m, s = models[key], smoke_result(key)
        par = bool(s["supports_parallel_tool_calls"])
        return {"model_key": key, "alias": key, "url": f"http://127.0.0.1:{port}/v1", "gpus": gpus,
                "supports_parallel_tool_calls": par,
                "max_turns_per_round": base["turns_per_round"]["default" if par else "no_parallel_tool_calls"],
                "request_extra_body": m.get("request_extra_body") or {},
                "system_prompt_suffix": m.get("system_prompt_suffix") or ""}
    homo = run["agent_a"] == run["agent_b"]
    return {
        "run": {"run_id": run["run_id"], "label": run["label"],
                "n_questions": n_override or sample["n_questions"], "seed": sample["seed"]},
        "dataset": dataset_of(run, base),
        "agents": {"agent_a": agent(run["agent_a"], gpus_a, ports["a"]),
                   "agent_b": agent(run["agent_b"], gpus_a if homo else gpus_b,
                                    ports["a"] if homo else ports["b"])},
        "decoding": base["decoding"],
        "debate": base["debate"],
        "tools": {"mcp_url": f"http://127.0.0.1:{ports['mcp']}/mcp", "agent_tools": base["tools"]["agent_tools"]},
        "debate_system_prompt_file": base["debate_system_prompt_file"],
        "benchmark": run["benchmark"],
        "block": run["block"], "priority": run["priority"], "replicate": run.get("replicate", 1),
    }


# --------------------------------------------------------------------------- #
# One run (executed in its own thread)
# --------------------------------------------------------------------------- #
def execute_run(run, gpus_a, gpus_b, slot, base, models, sample, n_override, concurrent, on_done):
    rd = run_dir(run, bool(n_override))
    (rd / "server_logs").mkdir(parents=True, exist_ok=True)
    pb = base["serving"]["port_base"] + 10 * slot
    ports = {"a": pb, "b": pb + 1, "mcp": pb + 5}
    homo = run["agent_a"] == run["agent_b"]
    cfg = build_run_config(run, gpus_a, gpus_b, ports, base, models, sample, n_override)
    with open(rd / "run.yaml", "w", encoding="utf-8") as f:
        f.write(f"# written by phase4.py at {now()} -- exact configuration of this run\n")
        yaml.safe_dump(cfg, f, sort_keys=False)

    status = read_json(rd / "status.json", {})
    status.update({"run_id": run["run_id"], "state": "running", "started": now(),
                   "node": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                   "gpus_agent_a": gpus_a,
                   "gpus_agent_b": gpus_a if homo else gpus_b, "ports": ports,
                   "running_at_the_same_time": concurrent,
                   "attempts": status.get("attempts", 0) + 1, "message": ""})
    write_json(rd / "status.json", status)
    log(f"{run['run_id']} START {run['label']}  GPUs A={gpus_a} B={gpus_a if homo else gpus_b}")

    procs, t0, state, message = [], time.time(), "failed", ""
    _procs[run["run_id"]] = procs
    try:
        keys = [(run["agent_a"], gpus_a, ports["a"])] + ([] if homo else [(run["agent_b"], gpus_b, ports["b"])])
        for key, gpus, port in keys:
            procs.append(servers.start_vllm(key, models[key], gpus, port, smoke_result(key)["tool_call_parser"],
                                            base["serving"], rd / "server_logs" / f"vllm_{key}.log"))
        for (key, _, port), p in zip(keys, procs):
            if not servers.wait_vllm_ready(port, p, base["serving"]["ready_timeout_s"]):
                raise RuntimeError(f"vLLM for {key} did not start, see server_logs/vllm_{key}.log:\n"
                                   + servers.tail(rd / "server_logs" / f"vllm_{key}.log", 8))
            if _stop.is_set():
                raise InterruptedError()
        status["models_ready_after_s"] = round(time.time() - t0, 1)
        write_json(rd / "status.json", status)

        mcp = servers.start_mcp(gpus_a[0], ports["mcp"], rd / "server_logs" / "mcp.log")
        procs.append(mcp)
        if not servers.wait_port_open(ports["mcp"], mcp):
            raise RuntimeError("MCP server did not start, see server_logs/mcp.log")

        cmd = [sys.executable, str(HERE / "run_debate.py"), "--run-dir", str(rd)]
        if n_override:
            cmd += ["--n", str(n_override)]
        with open(rd / "debate.log", "a") as dl:
            dl.write(f"\n===== attempt {status['attempts']} started {now()} =====\n"); dl.flush()
            harness = subprocess.Popen(cmd, stdout=dl, stderr=subprocess.STDOUT, start_new_session=True)
            procs.append(harness)
            rc = harness.wait()
        summary = read_json(rd / "summary.json", {})
        if _stop.is_set():
            state, message = "interrupted", "stopped by user / job end -- resumes on next run"
        elif rc == 0 and summary.get("complete"):
            state = "done"
            message = f"{summary.get('judge_pending', 0)} debates wait for the judge" if summary.get("judge_pending") else ""
        else:
            message = f"harness exit code {rc}; last lines of debate.log:\n" + servers.tail(rd / "debate.log", 8)
    except InterruptedError:
        state, message = "interrupted", "stopped before the debate started"
    except Exception as e:
        state, message = ("interrupted", "stopped by user / job end") if _stop.is_set() else ("failed", str(e))
    finally:
        for p in reversed(procs):
            servers.stop(p)
        _procs.pop(run["run_id"], None)
        status.update({"state": state, "finished": now(),
                       "duration_min": round((time.time() - t0) / 60, 1), "message": message})
        write_json(rd / "status.json", status)
        log(f"{run['run_id']} {state.upper()} after {status['duration_min']} min"
            + (f" -- {message.splitlines()[0]}" if message else ""))
        on_done(run, gpus_a + ([] if homo else gpus_b))


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #
def gpu_need(run, models):
    a, b = run["agent_a"], run["agent_b"]
    return models[a]["gpus"] if a == b else models[a]["gpus"] + models[b]["gpus"]


def select_runs(plan, args):
    runs = plan["runs"]
    if getattr(args, "benchmark", None):
        runs = [r for r in runs if r.get("benchmark") == args.benchmark]
    if getattr(args, "homogeneous", False):
        runs = [r for r in runs if r["agent_a"] == r["agent_b"] and r.get("replicate", 1) == 1]
    if args.only:
        runs = [r for r in runs if r["run_id"] in args.only]
    else:
        allowed = PRIORITIES[: PRIORITIES.index(args.priority) + 1]
        runs = [r for r in runs if r["priority"] in allowed]
    out = []
    for r in runs:
        st = state_of(r, bool(args.n))
        if st == "done" or (st == "failed" and not args.retry_failed):
            continue
        if st == "running" and _still_running(r, bool(args.n)):
            log(f"{r['run_id']} skipped: still running in SLURM job "
                f"{read_json(run_dir(r, bool(args.n)) / 'status.json', {}).get('slurm_job_id')}")
            continue
        out.append(r)
    return out


def _still_running(run, test=False):
    """True if the run is marked running by a SLURM job that still exists --
    then a second process must not touch it (this happened on 2026-10-09 and
    wrote questions twice). Without a job id (old status files, interactive
    runs) the run is treated as stale and may be resumed."""
    job = read_json(run_dir(run, test) / "status.json", {}).get("slurm_job_id")
    if not job or job == os.environ.get("SLURM_JOB_ID"):
        return False
    try:
        out = subprocess.run(["squeue", "-h", "-j", str(job)], capture_output=True, text=True, timeout=30)
        return bool(out.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def cmd_run(args):
    plan, models, base = load_runs(), servers.load_models(), servers.load_base()
    gpus_all = [int(g) for g in args.gpus.split(",")]
    queue = select_runs(plan, args)
    blocked = [r for r in queue if blocked_reason(r)]
    queue = [r for r in queue if not blocked_reason(r)]
    for r in blocked:
        log(f"{r['run_id']} blocked ({blocked_reason(r)})")
    log(f"{len(queue)} runs to do on GPUs {gpus_all}, {len(blocked)} blocked")

    if args.dry_run:
        for r in queue:
            print(f"  {r['run_id']} {r['priority']:6} needs {gpu_need(r, models)} GPUs  {folder_name(r)}")
        return

    free, active, slots = set(gpus_all), {}, set()
    head_wait_since = {}

    def on_done(run, gpus):
        with _lock:
            free.update(gpus)
            slots.discard(active.pop(run["run_id"])["slot"])
        write_overview()

    def handle_stop(signum, frame):
        log("stop requested -- stopping servers of all running runs (they resume on the next start)")
        _stop.set()
        for procs in list(_procs.values()):
            for p in procs:
                servers.stop(p, grace_s=10)
    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    write_overview()
    while (queue or active) and not _stop.is_set():
        with _lock:
            started_any = True
            while started_any and queue:
                started_any = False
                head = queue[0]
                head_wait_since.setdefault(head["run_id"], time.time())
                head_starving = time.time() - head_wait_since[head["run_id"]] > STARVATION_S
                for r in list(queue):
                    if r is not head and head_starving:
                        break          # keep GPUs free for the waiting head run
                    need = gpu_need(r, models)
                    if need > len(free):
                        continue
                    pick = sorted(free)[:need]
                    ga = pick[: models[r["agent_a"]]["gpus"]]
                    gb = pick[models[r["agent_a"]]["gpus"]:]
                    slot = min(set(range(len(gpus_all))) - slots)
                    free.difference_update(pick); slots.add(slot)
                    concurrent = sorted(active)
                    active[r["run_id"]] = {"slot": slot}
                    queue.remove(r)
                    threading.Thread(target=execute_run, daemon=True,
                                     args=(r, ga, gb, slot, base, models, plan["sample"], args.n,
                                           concurrent, on_done)).start()
                    started_any = True
                    break
        time.sleep(POLL_S)
    while active:          # after a stop request: wait until all threads cleaned up
        time.sleep(min(POLL_S, 2))
    write_overview()
    log("scheduler finished" + (" (stopped)" if _stop.is_set() else ""))


# --------------------------------------------------------------------------- #
# Judge pass
# --------------------------------------------------------------------------- #
def cmd_judge(args):
    sys.path.insert(0, str(HERE)); sys.path.append(str(HERE.parent / "1_iteration"))
    from openai import AsyncOpenAI
    from run_debate import CSV_COLUMNS, build_summary, read_rows, write_rows
    from xmad_debate.config import load_run_config
    from xmad_debate.debate_loop import run_judge
    from xmad_debate.session import DebateSession
    from xmad_bench import em_max, f1_max

    plan, models, base = load_runs(), servers.load_models(), servers.load_base()
    jcfg = base["judge"]; key = jcfg["model_key"]; jm = models[key]
    todo = []
    for r in plan["runs"]:
        if args.only and r["run_id"] not in args.only:
            continue
        rd = run_dir(r)
        s = read_json(rd / "summary.json", {})
        if state_of(r) == "done" and s.get("judge_pending", 0) > 0:
            todo.append(r)
    log(f"judge pass: {len(todo)} runs with pending debates")
    if not todo:
        return
    smoke = smoke_result(key)
    if not smoke or not smoke["ok"]:
        sys.exit(f"judge model {key} has no smoke PASS -- run: python smoke_test.py --keys {key}")

    gpus = [int(g) for g in args.gpus.split(",")][: jm["gpus"]]
    port = base["serving"]["port_base"] + 95
    RESULTS.mkdir(exist_ok=True)
    proc = servers.start_vllm(key, jm, gpus, port, smoke["tool_call_parser"], base["serving"],
                              RESULTS / "judge_server.log")
    try:
        if not servers.wait_vllm_ready(port, proc, base["serving"]["ready_timeout_s"]):
            sys.exit("judge server did not start, see results/judge_server.log")
        prompt = (HERE / jcfg["prompt_file"]).read_text(encoding="utf-8")
        dec = SimpleNamespace(**jcfg["decoding"])
        async def judge_session(path, patience, client, sem):
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("judge_verdict"):            # already judged in an earlier, interrupted pass
                return int(path.stem), d["judge_verdict"]
            sess = DebateSession.from_dict(d, stalemate_patience=patience)
            try:
                async with sem:
                    verdict = await run_judge(sess, prompt, client, key, dec, jm.get("request_extra_body"))
            except Exception as e:                 # one failed request must not stop the pass
                log(f"  judge failed for {path.parent.parent.name}/{path.name}: {type(e).__name__}: {str(e)[:120]}")
                return int(path.stem), None
            d["judge_verdict"] = verdict
            path.write_text(json.dumps(d, indent=2), encoding="utf-8")
            return int(path.stem), verdict

        for r in todo:
            rd = run_dir(r)
            cfg = load_run_config(rd / "run.yaml", HERE)
            rows = read_rows(rd / "debate_results.csv")
            pending = [row for row in rows if row["source"] == "pending_judge"]
            log(f"{r['run_id']}: judging {len(pending)} debates ...")
            async def _all():
                # client and semaphore are created inside the event loop of THIS run:
                # asyncio.run() starts a new loop per run, and an HTTP client bound to an
                # earlier, closed loop fails with "Event loop is closed".
                client = AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="not-needed",
                                     timeout=1800)
                sem = asyncio.Semaphore(jcfg.get("parallel_requests", 8))
                try:
                    return await asyncio.gather(*[judge_session(rd / "sessions" / f"{row['idx']}.json",
                                                                cfg.debate.stalemate_patience, client, sem)
                                                  for row in pending])
                finally:
                    await client.close()
            verdicts = dict(asyncio.run(_all()))
            for row in rows:
                v = verdicts.get(int(row["idx"]))
                if v is None:
                    continue
                row["pred"], row["source"] = v["final_answer"], "judge"
                answers = json.loads(row["gold_answers"]) if row.get("gold_answers") else [row["gold"]]
                row["em"] = em_max(v["final_answer"], answers)
                row["f1"] = round(f1_max(v["final_answer"], answers), 4)
                row["judge_confidence"] = v["confidence"]
            write_rows(rd / "debate_results.csv", [{c: row.get(c, "") for c in CSV_COLUMNS} for row in rows])
            summary = build_summary(cfg, rd, 0.069)
            summary["judge_model"] = key
            write_json(rd / "summary.json", summary)
            st = read_json(rd / "status.json", {})
            st["judged"] = now(); st["message"] = ""
            write_json(rd / "status.json", st)
            log(f"{r['run_id']}: judged, EM now {summary['exact_match']}")
    finally:
        servers.stop(proc)
    write_overview()


# --------------------------------------------------------------------------- #
# Overview for humans
# --------------------------------------------------------------------------- #
def write_overview():
    with _lock:
        _write_overview()


def _write_overview():
    plan = load_runs()
    RESULTS.mkdir(exist_ok=True)
    rows, by_label = [], {}
    for r in plan["runs"]:
        rd = run_dir(r)
        st = read_json(rd / "status.json", {})
        sm = read_json(rd / "summary.json", {})
        state = st.get("state", "pending")
        if state == "pending" and blocked_reason(r):
            state = "blocked"
        row = {"run_id": r["run_id"], "benchmark": r.get("benchmark", ""), "block": r["block"], "priority": r["priority"],
               "agent_a": r["agent_a"], "agent_b": r["agent_b"], "replicate": r.get("replicate", 1),
               "pair_type": r["pair_type"], "status": state,
               "questions": sm.get("questions", ""), "judge_pending": sm.get("judge_pending", ""),
               "em": sm.get("exact_match", ""), "f1": sm.get("f1", ""),
               "consensus": sm.get("consensus_rate", ""), "jaccard_r1": sm.get("evidence_jaccard_round1", ""),
               "rounds": sm.get("rounds_mean", ""), "duration_min": st.get("duration_min", ""),
               "energy_wh": (sm.get("energy") or {}).get("total_wh", ""),
               "forced": "/".join(str(v) for v in (sm.get("forced_final_answers") or {}).values()),
               "invalid_calls": "/".join(str(v) for v in (sm.get("invalid_tool_calls") or {}).values()),
               "gpus": "+".join(str(x) for x in st.get("gpus_agent_a", [])) +
                       ("" if st.get("gpus_agent_b") == st.get("gpus_agent_a") or not st.get("gpus_agent_b")
                        else " / " + "+".join(str(x) for x in st["gpus_agent_b"])),
               "folder": folder_name(r), "message": (st.get("message") or "").splitlines()[0][:80]
               if st.get("message") else ""}
        rows.append(row)
        if row["replicate"] == 1:
            by_label[(r.get("benchmark"), r["agent_a"], r["agent_b"])] = row

    # delta to the better homogeneous self-pair (heterogeneous runs only)
    def final_em(row):
        return row["em"] if row["status"] == "done" and row["judge_pending"] == 0 else None
    for row in rows:
        row["delta_em"] = ""
        if row["agent_a"] != row["agent_b"] and row["replicate"] == 1:
            ha = by_label.get((row["benchmark"], row["agent_a"], row["agent_a"]))
            hb = by_label.get((row["benchmark"], row["agent_b"], row["agent_b"]))
            vals = [final_em(x) for x in (row, ha, hb) if x]
            if len(vals) == 3 and None not in vals:
                row["delta_em"] = round(vals[0] - max(vals[1], vals[2]), 4)

    cols = ["run_id", "benchmark", "status", "agent_a", "agent_b", "replicate", "pair_type", "block", "priority",
            "questions", "judge_pending", "em", "delta_em", "f1", "consensus", "jaccard_r1", "rounds",
            "duration_min", "energy_wh", "forced", "invalid_calls", "gpus", "folder", "message"]
    with open(RESULTS / "overview.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    lines = [
        "# Phase 4 -- run overview", "",
        f"Updated {datetime.now():%Y-%m-%d %H:%M}. Generated by `phase4.py`; do not edit.", "",
        "Status: " + ", ".join(f"**{k}** {v}" for k, v in sorted(counts.items())), "",
        "EM = exact match over the 1000 questions (final, after the judge pass). "
        "ΔEM = EM of the pair minus the EM of the better of its two homogeneous self-pairs "
        "(same benchmark; shown once all three are done and judged). Jaccard r1 = overlap of the evidence "
        "both agents retrieved in round 1. Pair type: DxD dense+dense, DxM dense+MoE, MxM MoE+MoE. "
        "Forced A/B = questions in which the agent had to be asked for its answer after using up its "
        "search budget. Invalid A/B = tool calls with an invalid tool name (dropped).", "",
        "| ID | Benchmark | Status | Agent A | Agent B | Type | Prio | EM | ΔEM | F1 | Consensus | Jaccard r1 | "
        "Rounds | Judge pending | Forced A/B | Invalid A/B | Min | Wh | GPUs | Folder | Note |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        rep = f" (rep {row['replicate']})" if row["replicate"] > 1 else ""
        lines.append(
            f"| {row['run_id']} | {row['benchmark']} | {row['status']} | {row['agent_a']} | {row['agent_b']}{rep} | {row['pair_type']} | "
            f"{row['priority']} | {row['em']} | {row['delta_em']} | {row['f1']} | {row['consensus']} | "
            f"{row['jaccard_r1']} | {row['rounds']} | {row['judge_pending']} | {row['forced']} | "
            f"{row['invalid_calls']} | {row['duration_min']} | "
            f"{row['energy_wh']} | {row['gpus']} | [{row['folder']}]({row['folder']}/) | {row['message']} |")

    # replicate spread = run-to-run noise
    groups = {}
    for row in rows:
        if row["agent_a"] == row["agent_b"] and row["em"] != "":
            groups.setdefault(f"{row['agent_a']} ({row['benchmark']})", []).append(row["em"])
    groups = {k: v for k, v in groups.items() if len(v) > 1}
    if groups:
        lines += ["", "## Replicates (run-to-run noise)", "",
                  "| Model (homogeneous) | EM per replicate | max - min |", "|---|---|---|"]
        for k, v in groups.items():
            lines.append(f"| {k} | {', '.join(str(x) for x in v)} | {round(max(v) - min(v), 4)} |")
    (RESULTS / "OVERVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_status(args):
    write_overview()
    counts = {}
    with open(RESULTS / "overview.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print(f"-> {RESULTS / 'OVERVIEW.md'}")


# Minimum plausible round-1 EM per benchmark. HotpotQA: Phase 3 rule. MuSiQue is
# much harder (2-4 hops, Trivedi et al. 2022), so its threshold is lower. (The first retriever
# check behind this choice measured BM25 by mistake, see retriever_check.py.)
MIN_R1_EM = {"hotpotqa": 0.30, "2wiki": 0.30, "musique": 0.15}
MIN_RECALL_ALL_ROUNDS = 0.60   # agreed 2026-10-24: below this, discuss k (passages per search)

SANITY_RULES = {   # flag a run if ...
    "no answer (round 1) > 1%": lambda s: max(s.get("no_answer_round1", {}).values() or [0]) > 0.01 * s["questions"],
    "errors > 0.5%": lambda s: sum(s["errors"].values()) > 0.005 * s["questions"],
    "truncated > 2%": lambda s: max(s["truncated_replies"].values()) > 0.02 * s["questions"],
    "low round-1 EM": lambda s: min(s["round1_single_agent_em"].values()) < MIN_R1_EM.get(s.get("benchmark"), 0.30),
    "evidence recall (all rounds) < 0.60":
        lambda s: (s.get("evidence_recall") or {}).get("all_rounds") is not None
        and s["evidence_recall"]["all_rounds"] < MIN_RECALL_ALL_ROUNDS,
}


def _fmt(x):
    return "-" if x is None else f"{x:.2f}"


def cmd_sanity(args):
    """One line per run with the numbers that reveal a broken model/setup."""
    pattern = "P4-*_test" if args.test else "P4-*"
    print(f"{'run':52} {'n':>4} {'EM':>5} {'r1A':>5} {'r1B':>5} {'noAnsA':>6} {'noAnsB':>6} "
          f"{'forcA':>5} {'forcB':>5} {'trunc':>5} {'err':>4} {'s/q':>5} {'recR1':>5} {'recAll':>6}  flags")
    bad = 0
    for p in sorted(RESULTS.glob(f"{pattern}/summary.json")):
        if not args.test and p.parent.name.endswith("_test"):
            continue
        s = read_json(p, {})
        na, ff = s.get("no_answer_round1", {}), s.get("forced_final_answers", {})
        flags = [k for k, rule in SANITY_RULES.items() if rule(s)]
        bad += bool(flags)
        print(f"{p.parent.name[:52]:52} {s['questions']:>4} {s['exact_match']:5.3f} "
              f"{s['round1_single_agent_em']['agent_a']:5.3f} {s['round1_single_agent_em']['agent_b']:5.3f} "
              f"{na.get('agent_a', '-'):>6} {na.get('agent_b', '-'):>6} {ff.get('agent_a', '-'):>5} "
              f"{ff.get('agent_b', '-'):>5} {sum(s['truncated_replies'].values()):>5} "
              f"{sum(s['errors'].values()):>4} {s['wall_time_s']['mean']:5.1f} "
              f"{_fmt((s.get('evidence_recall') or {}).get('union_round1')):>5} "
              f"{_fmt((s.get('evidence_recall') or {}).get('all_rounds')):>6}  {'; '.join(flags)}")
    print(f"\n{bad} run(s) flagged" + (" -- look at their debate_results.csv before the real runs" if bad else ""))


def cmd_dedupe(args):
    """Keep one row per question (the last one written) in every run; backup as
    debate_results_with_duplicates.csv; rebuild summary.json. Energy of such a
    run stays doubled (two processes measured) -- the summary notes it."""
    from run_debate import CSV_COLUMNS, build_summary, read_rows, write_rows
    from xmad_debate.config import load_run_config
    for r in load_runs()["runs"]:
        rd = run_dir(r)
        if not (rd / "debate_results.csv").exists():
            continue
        rows = read_rows(rd / "debate_results.csv")
        last = {}
        for x in rows:
            last[int(x["idx"])] = x
        dup = len(rows) - len(last)
        if not dup:
            continue
        print(f"{rd.name}: {len(rows)} rows -> {len(last)} ({dup} duplicates removed)")
        if args.dry_run:
            continue
        shutil.copy2(rd / "debate_results.csv", rd / "debate_results_with_duplicates.csv")
        write_rows(rd / "debate_results.csv", [{c: x.get(c, "") for c in CSV_COLUMNS}
                                               for _, x in sorted(last.items())])
        s = build_summary(load_run_config(rd / "run.yaml", HERE), rd, 0.069)
        s["energy"]["note"] += f"; WARNING: {dup} questions were run twice by two processes, energy overstated"
        s["duplicate_rows_removed"] = dup
        write_json(rd / "summary.json", s)
    if not args.dry_run:
        write_overview()


def tool_call_errors(session_path):
    """Error texts of the failed tool calls in a saved session (only the `error` field
    of each tool_trace entry -- passages and answers may contain words like "not connected")."""
    if not session_path.exists():
        return []
    d = json.loads(session_path.read_text(encoding="utf-8"))
    return [t["error"] for rnd in d.get("rounds", []) for turn in rnd.values() if isinstance(turn, dict)
            for t in turn.get("tool_trace") or [] if t.get("error")]


def cmd_drop_errors(args):
    """Remove the questions hit by a lost MCP connection, so that `run --retry-failed`
    computes them again (resume skips every question that has a row). A question
    counts as hit if its row has a connection error, or if its session file shows a
    tool call that failed with one (the question during which the connection dropped).
    Backup: debate_results_with_errors.csv; the removed idx are listed in drop_errors.json."""
    from run_debate import CSV_COLUMNS, connection_lost, read_rows, write_rows
    for r in load_runs()["runs"]:
        if args.only and r["run_id"] not in args.only:
            continue
        rd = run_dir(r)
        if not (rd / "debate_results.csv").exists():
            continue
        rows = read_rows(rd / "debate_results.csv")
        bad = set()
        for x in rows:
            if (x.get("error") and connection_lost(f"{x['error']}: {x.get('error_detail', '')}")) or \
                    any(connection_lost(e) for e in tool_call_errors(rd / "sessions" / f"{x['idx']}.json")):
                bad.add(int(x["idx"]))
        if not bad:
            continue
        print(f"{rd.name}: {len(rows)} rows, {len(bad)} hit by a lost MCP connection"
              + (" (dry run, nothing changed)" if args.dry_run else " -> removed"))
        if args.dry_run:
            continue
        shutil.copy2(rd / "debate_results.csv", rd / "debate_results_with_errors.csv")
        write_rows(rd / "debate_results.csv", [{c: x.get(c, "") for c in CSV_COLUMNS}
                                               for x in rows if int(x["idx"]) not in bad])
        write_json(rd / "drop_errors.json", {"date": now(), "reason": "MCP connection lost", "rows_before": len(rows),
                                             "removed": len(bad), "removed_idx": sorted(bad)})
        st = read_json(rd / "status.json", {})
        st.update({"state": "failed", "message": f"drop-errors: {len(bad)} questions to re-run (MCP connection lost)"})
        write_json(rd / "status.json", st)


def cmd_remaining(args):
    """Print how many selected runs are not done yet and could run (not blocked)."""
    args.n, args.retry_failed = None, False
    todo = [r for r in select_runs(load_runs(), args) if not blocked_reason(r)]
    print(len(todo))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    r = sub.add_parser("run")
    r.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    r.add_argument("--priority", choices=PRIORITIES, default="could", help="include runs up to this priority")
    r.add_argument("--only", nargs="+", help="run IDs, e.g. P4-MSQ-001 P4-2WK-001")
    r.add_argument("--benchmark", help="only the runs of one benchmark (musique, 2wiki, hotpotqa)")
    r.add_argument("--n", type=int, default=None, help="test mode: only the first n questions (folder suffix _test)")
    r.add_argument("--retry-failed", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--homogeneous", action="store_true", help="only the homogeneous self-pairs (pre-test)")
    dd = sub.add_parser("dedupe")
    dd.add_argument("--dry-run", action="store_true")
    de = sub.add_parser("drop-errors", help="remove questions hit by a lost MCP connection (then run --retry-failed)")
    de.add_argument("--only", nargs="+")
    de.add_argument("--dry-run", action="store_true")
    sa = sub.add_parser("sanity")
    sa.add_argument("--test", action="store_true", help="check the *_test folders of the pre-test")
    rem = sub.add_parser("remaining")
    rem.add_argument("--priority", choices=PRIORITIES, default="could")
    rem.add_argument("--only", nargs="+")
    rem.add_argument("--benchmark")
    j = sub.add_parser("judge")
    j.add_argument("--gpus", default="0,1,2,3")
    j.add_argument("--only", nargs="+")
    args = ap.parse_args()
    {"status": cmd_status, "run": cmd_run, "judge": cmd_judge, "remaining": cmd_remaining,
     "sanity": cmd_sanity, "dedupe": cmd_dedupe, "drop-errors": cmd_drop_errors}[args.cmd](args)


if __name__ == "__main__":
    main()

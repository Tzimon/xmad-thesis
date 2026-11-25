"""
xmad/3_iteration/tests/test_offline.py

Offline checks of the Phase 3 code: no GPU, no vLLM, no network. Model servers
and the MCP server are replaced by small fakes. Runs with plain Python
(pytest is not needed):

    python tests/test_offline.py
"""

import asyncio
import csv
import json
import shutil
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "1_iteration"))

import yaml  # noqa: E402

import phase3  # noqa: E402
import run_debate  # noqa: E402
from xmad_debate import debate_loop  # noqa: E402
from xmad_debate.config import load_run_config  # noqa: E402
from xmad_debate.session import DebateSession  # noqa: E402

NS = types.SimpleNamespace


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
def _resp(content=None, tool_calls=None, finish="stop", tokens=10):
    msg = NS(content=content, tool_calls=tool_calls)
    return NS(choices=[NS(message=msg, finish_reason=finish)],
              usage=NS(prompt_tokens=tokens, completion_tokens=tokens, total_tokens=2 * tokens))


def _call(i, query):
    return NS(id=f"c{i}", function=NS(name="search_knowledge_base", arguments=json.dumps({"query": query})))


class FakeLLM:
    """Searches once, then answers `answer` (with a <think> block to be stripped)."""
    def __init__(self, answer, fail_first=0):
        self.answer, self.fail_first = answer, fail_first
        self.chat = NS(completions=NS(create=self.create))
        self.requests = []

    async def create(self, **kw):
        self.requests.append(kw)
        if self.fail_first:
            self.fail_first -= 1
            raise type("APIConnectionError", (Exception,), {})("down")
        if not any(m["role"] == "tool" for m in kw["messages"]):
            return _resp(tool_calls=[_call(1, "q1"), _call(2, "q2")])
        return _resp(content=f"<think>hmm</think>Because.\nReasoning summary: found it.\nFinal answer: {self.answer}")


class FakeMCP:
    def __init__(self, titles):
        self.titles = titles

    async def list_tools(self):
        return [NS(name="search_knowledge_base", description="search", inputSchema={"type": "object"}),
                NS(name="load_corpus", description="ctl", inputSchema={"type": "object"})]

    async def call_tool(self, name, args):
        return NS(data=[{"title": t, "text": "x", "score": 1.0} for t in self.titles])


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def _setup_tmp():
    tmp = Path(tempfile.mkdtemp())
    phase3.RESULTS = tmp / "results"
    phase3.SMOKE = tmp / "smoke"
    phase3.SMOKE.mkdir(parents=True)
    models = phase3.servers.load_models()
    for k in models:
        (phase3.SMOKE / f"{k}.json").write_text(json.dumps(
            {"ok": True, "tool_call_parser": "x", "supports_parallel_tool_calls": not k.startswith("llama")}))
    return tmp


def _cfg_for(run, tmp, n=3):
    base, models, plan = phase3.servers.load_base(), phase3.servers.load_models(), phase3.load_runs()
    cfg = phase3.build_run_config(run, [0], [1], {"a": 8000, "b": 8001, "mcp": 8005},
                                  base, models, plan["sample"], n)
    rd = tmp / "run"; rd.mkdir(exist_ok=True)
    (rd / "run.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return load_run_config(rd / "run.yaml", HERE), rd


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_strip_reasoning():
    s = debate_loop._strip_reasoning
    _check(s("<think>a</think>Final answer: x") == "Final answer: x", "closed think block removed")
    _check(s("reasoning...</think>Final answer: x") == "Final answer: x", "dangling closing tag")
    _check(s("Final answer: x<think>cut off") == "Final answer: x", "dangling opening tag")
    _check(s(None) == "", "None content")


def test_debate_and_session_roundtrip(tmp):
    plan = phase3.load_runs()
    run = next(r for r in plan["runs"] if r["agent_a"] == "qwen2.5-7b" and r["agent_b"] == "llama3.1-8b")
    cfg, _ = _cfg_for(run, tmp)
    _check(cfg.agents.agent_b.max_turns_per_round == 12 and not cfg.agents.agent_b.supports_parallel_tool_calls,
           "no-parallel model gets 1 call per reply and 12 turns")
    llm = {cfg.agents.agent_a.url: FakeLLM("Paris"), cfg.agents.agent_b.url: FakeLLM("Paris")}
    s = DebateSession("q?", {"agent_a": "a", "agent_b": "b"}, 10)
    s = asyncio.run(debate_loop.run_debate("q?", s, cfg, llm, FakeMCP(["A", "B"]), FakeMCP(["B", "C"])))
    _check(s.final_answer() == ("Paris", "consensus"), "consensus after round 1, think block stripped")
    _check(s.rounds[0].turns["agent_a"].tool_calls == 2 and s.rounds[0].turns["agent_b"].tool_calls == 1,
           "parallel calls kept for A, truncated to 1 for B")
    fields = run_debate.round1_fields(s, "Paris")
    _check(fields["r1_jaccard"] == round(1 / 3, 4) and fields["a_r1_em"] == 1, "round-1 baseline + Jaccard")

    llm = {cfg.agents.agent_a.url: FakeLLM("Paris"), cfg.agents.agent_b.url: FakeLLM("Rome")}
    s = DebateSession("q?", {"agent_a": "a", "agent_b": "b"}, 10)
    s = asyncio.run(debate_loop.run_debate("q?", s, cfg, llm, FakeMCP(["A"]), FakeMCP(["A"])))
    _check(s.final_answer() == ("", "pending_judge") and s.stop_reason() == "stalemate",
           "no consensus -> stalemate -> pending_judge")
    s2 = DebateSession.from_dict(json.loads(s.to_json()))
    _check(s2.final_answer() == ("", "pending_judge") and len(s2.rounds) == len(s.rounds), "from_dict round trip")
    s2.judge_verdict = {"final_answer": "Rome", "confidence": 0.9}
    _check(s2.final_answer() == ("Rome", "judge"), "judge verdict becomes the answer")

    request = llm[cfg.agents.agent_a.url].requests[0]
    _check(request["temperature"] == 0.0 and request["max_tokens"] == 8192, "shared decoding block applied")


def test_retry(tmp):
    debate_loop.RETRY_WAITS_S = (0, 0)
    plan = phase3.load_runs()
    run = next(r for r in plan["runs"] if r["agent_a"] == r["agent_b"] == "qwen2.5-7b")
    cfg, _ = _cfg_for(run, tmp)
    fake = FakeLLM("Paris", fail_first=1)
    s = DebateSession("q?", {"agent_a": "a", "agent_b": "b"}, 10)
    s = asyncio.run(debate_loop.run_debate("q?", s, cfg, {cfg.agents.agent_a.url: fake},
                                           FakeMCP(["A"]), FakeMCP(["A"])))
    retries = sum(t.api_retries for t in s.rounds[0].turns.values())
    _check(retries == 1 and s.final_answer()[1] == "consensus", "connection error retried and counted")


class LoopingLLM(FakeLLM):
    """Keeps searching until it is asked without tools (like Nemotron-3-Nano-30B in the smoke test)."""
    async def create(self, **kw):
        self.requests.append(kw)
        if "tools" not in kw:          # the forced final request comes without tools
            return _resp(content=f"Reasoning summary: enough.\nFinal answer: {self.answer}")
        return _resp(tool_calls=[_call(len(self.requests), "again")], finish="tool_calls")


def test_forced_final_answer(tmp):
    plan = phase3.load_runs()
    run = next(r for r in plan["runs"] if r["agent_a"] == r["agent_b"] == "qwen2.5-7b")
    cfg, _ = _cfg_for(run, tmp)
    fake = LoopingLLM("Paris")
    s = DebateSession("q?", {"agent_a": "a", "agent_b": "b"}, 10)
    s = asyncio.run(debate_loop.run_debate("q?", s, cfg, {cfg.agents.agent_a.url: fake},
                                           FakeMCP(["A"]), FakeMCP(["A"])))
    t = s.rounds[0].turns["agent_a"]
    _check(t.forced_final == 1 and t.final_answer == "Paris" and t.tool_calls == 6,
           "after 6 searches the agent is asked once without tools and answers")


class EmptyNameLLM(FakeLLM):
    """First reply: one valid call plus one with an empty name (Mistral-Small-4 in run P3-013)."""
    async def create(self, **kw):
        self.requests.append(kw)
        if not any(m["role"] == "tool" for m in kw["messages"]):
            bad = NS(id="c9", function=NS(name="", arguments="{}"))
            return _resp(tool_calls=[_call(1, "q1"), bad], finish="tool_calls")
        return _resp(content="Reasoning summary: ok.\nFinal answer: Paris")


class NoAnswerLLM(FakeLLM):
    async def create(self, **kw):
        self.requests.append(kw)
        if not any(m["role"] == "tool" for m in kw["messages"]):
            return _resp(tool_calls=[_call(1, "q1")], finish="tool_calls")
        return _resp(content="")


def test_invalid_calls_and_no_answer(tmp):
    plan = phase3.load_runs()
    run = next(r for r in plan["runs"] if r["agent_a"] == r["agent_b"] == "qwen2.5-7b")
    cfg, _ = _cfg_for(run, tmp)
    fake = EmptyNameLLM("Paris")
    s = DebateSession("q?", {"agent_a": "a", "agent_b": "b"}, 10)
    s = asyncio.run(debate_loop.run_debate("q?", s, cfg, {cfg.agents.agent_a.url: fake},
                                           FakeMCP(["A"]), FakeMCP(["A"])))
    replayed = [m for m in fake.requests[-1]["messages"] if m["role"] == "assistant"][0]["tool_calls"]
    _check(len(replayed) == 1 and s.final_answer() == ("Paris", "consensus"),
           "tool call with empty name dropped, conversation not rejected")
    s = DebateSession("q?", {"agent_a": "a", "agent_b": "b"}, 3)
    s = asyncio.run(debate_loop.run_debate("q?", s, cfg, {cfg.agents.agent_a.url: NoAnswerLLM("x")},
                                           FakeMCP(["A"]), FakeMCP(["A"])))
    _check(s.consensus_round is None and len(s.rounds) == 3 and s.final_answer()[1] == "pending_judge",
           "two missing answers are no consensus and no stalemate -> rounds continue")


def test_mistral_and_suffix_config(tmp):
    plan = phase3.load_runs()
    run = next(r for r in plan["runs"] if "muse-glimmer-30b" in (r["agent_a"], r["agent_b"])
               and "mistral-small4-119b" not in (r["agent_a"], r["agent_b"]))
    cfg, _ = _cfg_for(run, tmp)
    muse = cfg.agents.agent_a if cfg.agents.agent_a.model_key == "muse-glimmer-30b" else cfg.agents.agent_b
    _check(cfg.system_prompt_for(muse).rstrip().endswith("Reasoning strength: low"), "prompt-side reasoning switch")
    cmd = phase3.servers.vllm_command("qwen3.5-9b", phase3.servers.load_models()["qwen3.5-9b"], 8000,
                                      "qwen3_coder", phase3.servers.load_base()["serving"])
    _check("--generation-config" in cmd and cmd[cmd.index("--generation-config") + 1] == "vllm",
           "every server runs without model sampling defaults")
    _check('{"enable_thinking": false}' in cmd, "server-side thinking switch for Qwen3.5")


def test_summary_energy(tmp):
    plan = phase3.load_runs()
    for homo in (True, False):
        run = next(r for r in plan["runs"] if (r["agent_a"] == r["agent_b"]) == homo)
        cfg, rd = _cfg_for(run, tmp)
        for f in rd.glob("gpu_metrics_*.csv"):
            f.unlink()
        lines = ["timestamp, index, utilization.gpu [%], memory.used [MiB], power.draw [W]"]
        for t in range(0, 3601, 60):
            ts = f"2026/10/05 1{t // 3600}:{(t % 3600) // 60:02d}:00.000"
            lines += [f"{ts}, 0, 90, 50000, 400", f"{ts}, 1, 90, 50000, 200"]
        (rd / "gpu_metrics_0.csv").write_text("\n".join(lines))
        rows = [{c: "" for c in run_debate.CSV_COLUMNS} | {"idx": i, "em": 1, "f1": 1.0, "source": "consensus",
                                                            "rounds": 1, "wall_time_s": 1.0} for i in range(3)]
        run_debate.write_rows(rd / "debate_results.csv", rows)
        s = run_debate.build_summary(cfg, rd, 0.1)
        if homo:
            _check(s["energy"]["total_wh"] == 400 and s["energy"]["agent_b_wh"] == 0,
                   "homogeneous: only the shared server GPU counts (400 Wh)")
        else:
            _check(s["energy"]["agent_a_wh"] == 400 and s["energy"]["agent_b_wh"] == 200,
                   "heterogeneous: energy per agent on its own GPU")
        _check(s["complete"] and s["exact_match"] == 1.0, "summary complete")


def test_overview_delta(tmp):
    plan = phase3.load_runs()
    ids = {}
    for r in plan["runs"]:
        if r.get("replicate", 1) == 1 and {r["agent_a"], r["agent_b"]} <= {"qwen3.5-27b", "gemma4-31b"}:
            ids[(r["agent_a"], r["agent_b"])] = r
    em = {("qwen3.5-27b",) * 2: 0.50, ("gemma4-31b",) * 2: 0.55}
    for key, r in ids.items():
        rd = phase3.run_dir(r); rd.mkdir(parents=True, exist_ok=True)
        phase3.write_json(rd / "status.json", {"state": "done"})
        phase3.write_json(rd / "summary.json", {"exact_match": em.get(key, 0.60), "judge_pending": 0,
                                                "questions": 1000})
    phase3.write_overview()
    with open(phase3.RESULTS / "overview.csv", encoding="utf-8") as f:
        rows = {r["run_id"]: r for r in csv.DictReader(f)}
    het = next(r for k, r in ids.items() if k[0] != k[1])
    _check(abs(float(rows[het["run_id"]]["delta_em"]) - 0.05) < 1e-9, "delta EM = 0.60 - max(0.50, 0.55)")
    _check((phase3.RESULTS / "OVERVIEW.md").exists(), "OVERVIEW.md written")


def test_scheduler_packing(tmp):
    """Fake execute_run: checks that GPUs are never over-allocated and every run starts once."""
    phase3.POLL_S = 0.02
    models = phase3.servers.load_models()
    in_use, peak, started, lock = set(), [0], [], threading.Lock()

    def fake_execute(run, ga, gb, slot, base, models_, sample, n, concurrent, on_done):
        gpus = ga + ([] if run["agent_a"] == run["agent_b"] else gb)
        with lock:
            assert not (in_use & set(gpus)), f"GPU clash {gpus} vs {in_use}"
            in_use.update(gpus); peak[0] = max(peak[0], len(in_use)); started.append(run["run_id"])
        time.sleep(0.05 * len(gpus))
        with lock:
            in_use.difference_update(gpus)
        on_done(run, gpus)

    phase3.execute_run = fake_execute
    args = NS(gpus="0,1,2,3,4,5,6,7", priority="could", only=None, n=None, retry_failed=False, dry_run=False)
    phase3.cmd_run(args)
    plan = phase3.load_runs()
    expected = [r["run_id"] for r in phase3.select_runs(plan, args) if not phase3.blocked_reason(r)]
    _check(sorted(started) == sorted(expected), f"all {len(expected)} open runs started, each exactly once")
    _check(peak[0] <= 8, f"never more than 8 GPUs in use (peak {peak[0]})")
    _check(max(phase3.gpu_need(r, models) for r in plan["runs"]) <= 8, "every run fits on one node")


if __name__ == "__main__":
    tmp = _setup_tmp()
    try:
        for name, fn in list(globals().items()):
            if name.startswith("test_"):
                print(name)
                fn(tmp) if fn.__code__.co_argcount else fn()
        print("\nALL OFFLINE TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

"""
xmad/4_iteration/tests/test_phase4_offline.py

Offline checks of the Phase 4 additions: no GPU, no vLLM, no network. The debate
itself is the unchanged Phase 3 code (checked by 3_iteration/tests/test_offline.py);
here the new parts are tested: benchmark adapters, scoring over aliases, evidence
recall, the harness end to end on a tiny fake benchmark, and phase4.py config/overview.

    python tests/test_phase4_offline.py            # synthetic fixtures only
    python tests/test_phase4_offline.py --real     # also check the downloaded files in data/
"""

import asyncio
import json
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "1_iteration"))

import yaml  # noqa: E402

import xmad_bench  # noqa: E402
from xmad_bench import em_max, evidence_recall, f1_max, load_benchmark  # noqa: E402
from xmad_bench.base import unique_titles  # noqa: E402

NS = types.SimpleNamespace


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


# --------------------------------------------------------------------------- #
# fixtures in the official formats (field names from the authors' repositories)
# --------------------------------------------------------------------------- #
MSQ = [{"id": "2hop__1_2", "question": "Who founded the company that makes the iPhone?",
        "answer": "Steve Jobs", "answer_aliases": ["Steven Paul Jobs"], "answerable": True,
        "paragraphs": [
            {"idx": 0, "title": "iPhone", "paragraph_text": "The iPhone is made by Apple.", "is_supporting": True},
            {"idx": 1, "title": "Apple Inc.", "paragraph_text": "Apple was founded by Steve Jobs.", "is_supporting": True},
            {"idx": 2, "title": "Apple Inc.", "paragraph_text": "Apple is in Cupertino.", "is_supporting": False},
            {"idx": 3, "title": "Pear", "paragraph_text": "A pear is a fruit.", "is_supporting": False}]},
       {"id": "3hop1__9_8_7", "question": "Unanswerable?", "answer": "", "answer_aliases": [],
        "answerable": False, "paragraphs": [{"idx": 0, "title": "X", "paragraph_text": "x", "is_supporting": False}]}]

WIKI = [{"_id": "abc", "type": "comparison", "question": "Who was born first, A or B?", "answer": "A",
         "context": [["A", ["A was born in 1900.", "He was a poet."]], ["B", ["B was born in 1950."]],
                     ["C", ["Unrelated."]]],
         "supporting_facts": [["A", 0], ["B", 0]], "evidences": [], "entity_ids": "Q1_Q2"}]


def _write_fixtures(root):
    (root / "musique").mkdir(parents=True)
    (root / "2wiki").mkdir(parents=True)
    (root / "musique" / "musique_ans_v1.0_dev.jsonl").write_text("\n".join(json.dumps(x) for x in MSQ), encoding="utf-8")
    (root / "2wiki" / "dev.json").write_text(json.dumps(WIKI), encoding="utf-8")


# --------------------------------------------------------------------------- #
def test_adapters(data):
    msq = load_benchmark("musique", "answerable", "dev", data)
    _check(len(msq) == 1, "MuSiQue: unanswerable question skipped")
    ex = msq[0]
    _check([p["title"] for p in ex.paragraphs] == ["iPhone", "Apple Inc.", "Apple Inc. (2)", "Pear"],
           "MuSiQue: paragraphs in idx order, repeated title made unique")
    _check(ex.gold_titles == {"iPhone", "Apple Inc."}, "MuSiQue: gold = supporting paragraphs")
    _check(ex.answers == ["Steve Jobs", "Steven Paul Jobs"] and ex.meta["hops"] == 2, "MuSiQue: aliases, hops")

    wk = load_benchmark("2wiki", "april7", "dev", data)
    ex = wk[0]
    _check(ex.paragraphs[0] == {"title": "A", "text": "A was born in 1900. He was a poet."}, "2Wiki: sentences joined")
    _check(ex.gold_titles == {"A", "B"} and ex.meta["type"] == "comparison", "2Wiki: gold titles, type")

    try:
        load_benchmark("musique", "answerable", "dev", data / "nowhere")
        _check(False, "missing file raises")
    except FileNotFoundError:
        _check(True, "missing file -> clear FileNotFoundError")


def test_scoring():
    _check(em_max("Steven Paul Jobs", ["Steve Jobs", "Steven Paul Jobs"]) == 1, "EM = max over aliases")
    _check(em_max("the Steve Jobs.", ["Steve Jobs"]) == 1, "Phase 1 normalisation kept")
    _check(abs(f1_max("Steve", ["Steve Jobs"]) - 2 / 3) < 1e-9 and em_max("", ["x"]) == 0, "F1, empty prediction")
    _check(evidence_recall(["A", "C"], {"A", "B"}) == 0.5 and evidence_recall(["A"], set()) is None, "evidence recall")
    out, m = unique_titles([("T", "1"), ("T", "2"), ("U", "3")])
    _check([p["title"] for p in out] == ["T", "T (2)", "U"] and m["T"] == ["T", "T (2)"], "unique titles")


# --------------------------------------------------------------------------- #
# harness end to end with fake model and MCP servers
# --------------------------------------------------------------------------- #
def _resp(content=None, tool_calls=None, tokens=10):
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls), finish_reason="stop")],
              usage=NS(prompt_tokens=3 * tokens, completion_tokens=tokens, total_tokens=4 * tokens))


class FakeLLM:
    def __init__(self, answer):
        self.answer = answer
        self.chat = NS(completions=NS(create=self.create))

    async def create(self, **kw):
        if not any(m["role"] == "tool" for m in kw["messages"]):
            call = NS(id="c1", function=NS(name="search_knowledge_base", arguments=json.dumps({"query": "q"})))
            return _resp(tool_calls=[call])
        return _resp(content=f"Reasoning summary: ok.\nFinal answer: {self.answer}")

    async def close(self):
        pass


class FakeMCP:
    loaded = []

    def __init__(self, url):
        self.titles = ["iPhone", "Pear"] if "a" not in FakeMCP.loaded else ["Apple Inc."]
        FakeMCP.loaded.append("a")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def list_tools(self):
        return [NS(name="search_knowledge_base", description="search", inputSchema={"type": "object"}),
                NS(name="load_corpus", description="ctl", inputSchema={"type": "object"})]

    async def call_tool(self, name, args):
        if name == "load_corpus":
            _check(len(args["paragraphs"]) == 4, "harness loads the benchmark paragraphs into the MCP server")
            return NS(data={"loaded": 4})
        return NS(data=[{"title": t, "text": "x", "score": 1.0} for t in self.titles])


def test_harness(data, tmp):
    import phase4
    import run_debate
    sys.modules["fastmcp"] = NS(Client=FakeMCP)
    answers = iter(["Steven Paul Jobs", "Steve Jobs"])
    sys.modules["openai"] = NS(AsyncOpenAI=lambda **kw: FakeLLM(next(answers)))
    run_debate.load_benchmark = lambda n, c, s: load_benchmark(n, c, s, data)

    base, models = phase4.servers.load_base(), phase4.servers.load_models()
    run = {"run_id": "P4-MSQ-999", "label": "x__y", "benchmark": "musique", "block": "T", "priority": "must",
           "agent_a": "qwen2.5-7b", "agent_b": "llama3.1-8b"}
    phase4.SMOKE = tmp / "smoke"; phase4.SMOKE.mkdir()
    for k in ("qwen2.5-7b", "llama3.1-8b"):
        (phase4.SMOKE / f"{k}.json").write_text(json.dumps({"ok": True, "tool_call_parser": "x",
                                                            "supports_parallel_tool_calls": True}))
    cfg = phase4.build_run_config(run, [0], [1], {"a": 8000, "b": 8001, "mcp": 8005}, base, models,
                                  {"n_questions": 1, "seed": 42}, None)
    _check(cfg["dataset"]["name"] == "musique" and cfg["benchmark"] == "musique", "run.yaml gets the benchmark's dataset")
    rd = tmp / "run"; rd.mkdir()
    (rd / "run.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    rc = run_debate.load_run_config(rd / "run.yaml", HERE)
    asyncio.run(run_debate.run_all(rc, rd))
    row = run_debate.read_rows(rd / "debate_results.csv")[0]
    _check(row["benchmark"] == "musique" and json.loads(row["gold_answers"])[1] == "Steven Paul Jobs", "CSV: benchmark, gold list")
    _check(row["source"] == "pending_judge", "two different answer strings -> no consensus -> waits for the judge")
    _check(row["a_r1_em"] == "1" and row["b_r1_em"] == "1", "round-1 answers: alias and main answer both score EM 1")
    _check(float(row["a_r1_recall"]) == 0.5 and float(row["b_r1_recall"]) == 0.5 and float(row["r1_recall_union"]) == 1.0,
           "evidence recall: each agent 0.5, both together 1.0 (complementary evidence)")
    answers = json.loads(row["gold_answers"])
    _check(em_max("Steven Paul Jobs", answers) == 1, "judge pass re-scores the verdict against all gold answers")
    _check(int(row["a_prompt_tokens"]) > 0 and int(row["a_completion_tokens"]) > 0, "prompt/completion tokens in CSV")
    s = run_debate.build_summary(rc, rd, 0.069)
    _check(s["benchmark"] == "musique" and "evidence_recall" in s and "tokens_split" in s, "summary: benchmark, recall, split")


def _dist(values):
    from collections import Counter
    c = Counter(values)
    return ", ".join(f"{k}: {v}" for k, v in sorted(c.items(), key=lambda kv: (kv[0] is None, kv[0])))


def _check_corpus(name, exs):
    """What must hold for every question, whatever the benchmark."""
    _check(all(e.question.strip() and e.answers and e.answers[0].strip() for e in exs),
           f"{name}: every question and gold answer non-empty")
    _check(all(e.paragraphs for e in exs), f"{name}: every question has paragraphs")
    _check(all(e.gold_titles for e in exs), f"{name}: every question has supporting paragraphs")
    _check(all(e.gold_titles <= {p["title"] for p in e.paragraphs} for e in exs),
           f"{name}: supporting paragraphs are part of the corpus of their question")
    _check(all(len({p["title"] for p in e.paragraphs}) == len(e.paragraphs) for e in exs),
           f"{name}: titles unique within each question (after renaming repeats)")
    print(f"      paragraphs per question: {_dist(len(e.paragraphs) for e in exs)}")
    print(f"      supporting paragraphs per question: {_dist(len(e.gold_titles) for e in exs)}")


def test_real_files():
    print("checking the downloaded files in data/ ...")
    msq = load_benchmark("musique", "answerable", "dev")
    _check(len(msq) > 2000, f"MuSiQue-Ans dev: {len(msq)} answerable questions")
    _check_corpus("MuSiQue", msq)
    print(f"      hops: {_dist(e.meta['hops'] for e in msq)}")
    print(f"      questions with aliases: {sum(len(e.answers) > 1 for e in msq)}")
    wk = load_benchmark("2wiki", "april7", "dev")
    _check(len(wk) > 10000, f"2Wiki dev: {len(wk)} questions")
    _check_corpus("2Wiki", wk)
    types_ = {e.meta["type"] for e in wk}
    _check(types_ == {"comparison", "inference", "compositional", "bridge_comparison"}, f"2Wiki types: {sorted(types_)}")
    print(f"      types: {_dist(e.meta['type'] for e in wk)}")


if __name__ == "__main__":
    tmp = Path(tempfile.mkdtemp())
    data = tmp / "data"
    _write_fixtures(data)
    print("adapters"); test_adapters(data)
    print("scoring"); test_scoring()
    print("harness"); test_harness(data, tmp)
    if "--real" in sys.argv:
        test_real_files()
    print("\nall offline checks passed")

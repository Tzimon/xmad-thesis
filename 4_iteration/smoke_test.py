"""
xmad/4_iteration/smoke_test.py (copied from 3_iteration, only comments changed)

Five-minute check per model BEFORE it enters a scored run. For each model it
starts the vLLM server on the given GPUs and checks:

  T1 tool call      a forced tool call comes back as a parsable
                    search_knowledge_base(query=...) call
  T2 parallel calls asked for two searches at once: how many calls come back,
                    and can a reply with all of them be sent back to the model?
                    -> supports_parallel_tool_calls for the runs
  T3 answer         one full debate turn (real system prompt, a given tool
                    result; further searches are answered as in the debate)
                    ends with "Final answer: ..." and no reasoning text

If T1 fails with the first tool-call parser of models.yaml, the next one is
tried. The judge model gets only a judge-format check (J1).

Output (human-readable), in smoke/:
    <model>.json              all details, read by phase4.py
    <model>_server.log        vLLM log of the last attempt
    SMOKE_RESULTS.md          one line per model, rebuilt after every model

    python smoke_test.py --gpus 0,1,2,3                    # all models, one after another
    python smoke_test.py --keys qwen3.5-9b gemma4-12b --gpus 0
    python smoke_test.py --missing --gpus 0,1,2,3          # only models without a PASS yet
    python smoke_test.py --priority must --missing         # models of the must-runs + judge, no PASS yet

Two terminals with disjoint GPUs (e.g. --gpus 0,1,2,3 and --gpus 4,5,6,7, and
--port 9100 / 9200) can test different models at the same time.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import servers

HERE = Path(__file__).resolve().parent
SMOKE = HERE / "smoke"

SEARCH_TOOL = [{
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Search the knowledge base for passages relevant to the query.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string", "description": "A natural-language search query."},
                                      "k": {"type": "integer", "description": "How many passages to return (default 3)."}},
                       "required": ["query"]},
    },
}]
QUESTION = "Were Scott Derrickson and Ed Wood of the same nationality?"
PASSAGES = ("[Scott Derrickson] Scott Derrickson (born July 16, 1966) is an American director, "
            "screenwriter and producer.\n\n[Ed Wood] Edward Davis Wood Jr. (October 10, 1924 - "
            "December 10, 1978) was an American filmmaker, actor, writer, producer, and director.")
CALL_ID = "a1b2c3d4e"
MAX_T3_TURNS = 6      # replies allowed in T3 = the debate's default turn budget per round   # 9 alphanumeric characters: accepted by every chat template incl. Mistral's


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _reasoning_text(msg):
    extra = getattr(msg, "model_extra", None) or {}
    return (getattr(msg, "reasoning_content", None) or extra.get("reasoning_content")
            or extra.get("reasoning") or "")


def run_checks(client, key, model, base):
    from xmad_debate.debate_loop import FINAL_PROMPT, _strip_reasoning
    sys_prompt = (HERE / base["debate_system_prompt_file"]).read_text(encoding="utf-8")
    if model.get("system_prompt_suffix"):
        sys_prompt = f"{sys_prompt.rstrip()}\n\n{model['system_prompt_suffix']}"
    dec = base["decoding"]
    kw = {"model": key, "temperature": dec["temperature"], "top_p": dec["top_p"],
          "max_tokens": dec["max_tokens"]}
    if model.get("request_extra_body"):
        kw["extra_body"] = model["request_extra_body"]
    res = {}

    # T1 forced tool call
    try:
        r = client.chat.completions.create(
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": QUESTION}],
            tools=SEARCH_TOOL, tool_choice="required", **kw)
        calls = r.choices[0].message.tool_calls or []
        ok = bool(calls) and calls[0].function.name == "search_knowledge_base" \
            and "query" in json.loads(calls[0].function.arguments or "{}")
        res["T1_tool_call"] = {"pass": ok, "calls": [c.function.arguments for c in calls]}
    except Exception as e:
        res["T1_tool_call"] = {"pass": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    if not res["T1_tool_call"]["pass"]:
        return res

    # T2 parallel tool calls + replay
    try:
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": "Search the knowledge base for Scott Derrickson and, in a separate "
                                            "search, for Ed Wood. Issue both searches now, in this one reply."}]
        r = client.chat.completions.create(messages=msgs, tools=SEARCH_TOOL, tool_choice="auto", **kw)
        calls = r.choices[0].message.tool_calls or []
        replay_ok = None
        if len(calls) > 1:
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                        for c in calls]})
            for c in calls:
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": PASSAGES})
            try:
                client.chat.completions.create(messages=msgs, tools=SEARCH_TOOL, tool_choice="auto", **kw)
                replay_ok = True
            except Exception as e:
                replay_ok = False
                res["T2_replay_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        res["T2_parallel"] = {"calls_in_one_reply": len(calls), "replay_ok": replay_ok,
                              "supports_parallel_tool_calls": len(calls) > 1 and bool(replay_ok)}
    except Exception as e:
        res["T2_parallel"] = {"calls_in_one_reply": 0, "supports_parallel_tool_calls": False,
                              "error": f"{type(e).__name__}: {str(e)[:300]}"}

    # T3 one full debate turn. Like the real debate loop, further searches are
    # answered (up to MAX_T3_TURNS replies); one call per reply is replayed if
    # the model cannot take several (T2), exactly as the debate loop does.
    parallel_ok = res.get("T2_parallel", {}).get("supports_parallel_tool_calls", False)
    try:
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": QUESTION},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": CALL_ID, "type": "function",
                                 "function": {"name": "search_knowledge_base",
                                              "arguments": json.dumps({"query": "Scott Derrickson Ed Wood nationality"})}}]},
                {"role": "tool", "tool_call_id": CALL_ID, "content": PASSAGES}]
        extra_searches, queries = 0, []
        for _ in range(MAX_T3_TURNS):
            r = client.chat.completions.create(messages=msgs, tools=SEARCH_TOOL, tool_choice="auto", **kw)
            m = r.choices[0].message
            if not m.tool_calls:
                break
            calls = m.tool_calls if parallel_ok else m.tool_calls[:1]
            extra_searches += len(calls)
            queries += [c.function.arguments for c in calls]
            msgs.append({"role": "assistant", "content": _strip_reasoning(m.content),
                         "tool_calls": [{"id": c.id, "type": "function",
                                         "function": {"name": c.function.name, "arguments": c.function.arguments}}
                                        for c in calls]})
            for c in calls:
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": PASSAGES})
        forced = bool(m.tool_calls) or not _strip_reasoning(m.content)
        if forced:   # same rule as the debate loop: FINAL_PROMPT, no tools
            r = client.chat.completions.create(messages=msgs + [{"role": "user", "content": FINAL_PROMPT}], **kw)
            m = r.choices[0].message
        content = m.content or ""
        answer = _strip_reasoning(content)
        res["T3_answer"] = {
            "forced_final": forced,
            "pass": "final answer:" in answer.lower() and not m.tool_calls,
            "final_lines": answer.strip().splitlines()[-2:],
            "extra_searches_before_answer": extra_searches,
            "extra_search_queries": queries,
            "last_content": content[-300:],
            "think_tag_in_content": "<think>" in content.lower(),
            "reasoning_chars": len(_reasoning_text(m)),
            "completion_tokens": r.usage.completion_tokens if r.usage else None,
            "finish_reason": r.choices[0].finish_reason,
        }
    except Exception as e:
        res["T3_answer"] = {"pass": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    return res


def run_judge_check(client, key, base):
    from xmad_debate.debate_loop import _extract_judge_fields, _strip_reasoning
    j = base["judge"]
    prompt = (HERE / j["prompt_file"]).read_text(encoding="utf-8")
    user = (f"Question: {QUESTION}\n\nRound 1:\n  Agent A reasoning summary: Both are described as American.\n"
            f"  Agent A answer: yes\n  Agent B reasoning summary: Only Derrickson's nationality was found.\n"
            f"  Agent B answer: no\n\nAgent A's final answer: yes\nAgent B's final answer: no")
    try:
        r = client.chat.completions.create(model=key, messages=[{"role": "system", "content": prompt},
                                                                {"role": "user", "content": user}],
                                           temperature=j["decoding"]["temperature"], top_p=j["decoding"]["top_p"],
                                           max_tokens=j["decoding"]["max_tokens"])
        conf, final = _extract_judge_fields(_strip_reasoning(r.choices[0].message.content))
        return {"J1_judge_format": {"pass": final.lower() in ("yes", "no"), "final_answer": final,
                                    "confidence": conf, "finish_reason": r.choices[0].finish_reason}}
    except Exception as e:
        return {"J1_judge_format": {"pass": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}}


def smoke_one(key, model, gpus, port, base):
    from openai import OpenAI
    is_judge = model.get("role") == "judge"
    attempts = []
    for parser in model.get("tool_call_parsers") or [None]:   # next parser if T1 or T3 fails
        log(f"{key}: starting vLLM on GPUs {gpus} with tool parser {parser} ...")
        t0 = time.time()
        proc = servers.start_vllm(key, model, gpus, port, parser, base["serving"], SMOKE / f"{key}_server.log")
        try:
            if not servers.wait_vllm_ready(port, proc, base["serving"]["ready_timeout_s"]):
                attempts.append({"tool_call_parser": parser, "server_started": False,
                                 "log_tail": servers.tail(SMOKE / f"{key}_server.log")})
                log(f"{key}: server did not start (see smoke/{key}_server.log)")
                continue
            load_s = round(time.time() - t0, 1)
            client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="not-needed", timeout=600)
            checks = run_judge_check(client, key, base) if is_judge else run_checks(client, key, model, base)
            attempts.append({"tool_call_parser": parser, "server_started": True, "load_s": load_s, **checks})
            if is_judge or (checks["T1_tool_call"]["pass"] and checks.get("T3_answer", {}).get("pass")):
                break
            log(f"{key}: failed with parser {parser} -- trying the next one, if any")
        finally:
            servers.stop(proc)

    def _ok(a):
        if is_judge:
            return a.get("server_started", False) and a.get("J1_judge_format", {}).get("pass", False)
        return a.get("server_started", False) and a.get("T1_tool_call", {}).get("pass", False)             and a.get("T3_answer", {}).get("pass", False)
    last = next((a for a in attempts if _ok(a)), attempts[-1])
    if is_judge:
        ok = last.get("server_started", False) and last.get("J1_judge_format", {}).get("pass", False)
    else:
        ok = last.get("server_started", False) and last.get("T1_tool_call", {}).get("pass", False) \
             and last.get("T3_answer", {}).get("pass", False)
    result = {"model": key, "hf_id": model["hf_id"], "ok": ok,
              "tool_call_parser": last.get("tool_call_parser"),
              "supports_parallel_tool_calls": last.get("T2_parallel", {}).get("supports_parallel_tool_calls", False),
              "tested": datetime.now().isoformat(timespec="seconds"), "gpus": gpus, "attempts": attempts}
    with open(SMOKE / f"{key}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log(f"{key}: {'PASS' if ok else 'FAIL'}")
    return result


def write_overview(models):
    lines = ["# Smoke test results", "",
             f"Updated {datetime.now():%Y-%m-%d %H:%M}. Details per model in `<model>.json`, vLLM log in `<model>_server.log`.",
             "Only models with **PASS** are used by `phase4.py run`.", "",
             "| Model | Result | Tool parser | Parallel calls | Final answer line | Reasoning in output | Load (s) | Tested |",
             "|---|---|---|---|---|---|---|---|"]
    for key in models:
        p = SMOKE / f"{key}.json"
        if not p.exists():
            lines.append(f"| {key} | not tested | | | | | | |")
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        a = r["attempts"][-1]
        t3 = a.get("T3_answer", {})
        j1 = a.get("J1_judge_format", {})
        final = " / ".join(t3.get("final_lines", [])) or j1.get("final_answer", "") or a.get("T1_tool_call", {}).get("error", "")
        if t3.get("forced_final"):
            final = "[answer forced after 6 searches] " + final
        reasoning = (f"{t3.get('reasoning_chars', 0)} chars" + (", <think> in content" if t3.get("think_tag_in_content") else "")) if t3 else ""
        lines.append(f"| {key} | {'PASS' if r['ok'] else '**FAIL**'} | {r.get('tool_call_parser') or ''} | "
                     f"{'yes' if r.get('supports_parallel_tool_calls') else 'no'} | {final[:80].replace('|', '/')} | "
                     f"{reasoning} | {a.get('load_s', '')} | {r['tested'][:16]} |")
    (SMOKE / "SMOKE_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3", help="GPUs this test may use (largest model needs 4)")
    ap.add_argument("--keys", nargs="+", help="models to test (default: all in models.yaml)")
    ap.add_argument("--missing", action="store_true", help="only models without a PASS yet")
    ap.add_argument("--priority", choices=["must", "should", "could"],
                    help="only models used by runs.yaml runs up to this priority (plus the judge)")
    ap.add_argument("--port", type=int, default=9100)
    args = ap.parse_args()

    SMOKE.mkdir(exist_ok=True)
    models, base = servers.load_models(), servers.load_base()
    gpus_avail = [int(g) for g in args.gpus.split(",")]
    keys = args.keys or list(models)
    if args.priority:
        order = ["must", "should", "could"]
        allowed = order[: order.index(args.priority) + 1]
        runs = servers.load_yaml(HERE / "runs.yaml")["runs"]
        used = {k for r in runs if r["priority"] in allowed for k in (r["agent_a"], r["agent_b"])}
        keys = [k for k in keys if k in used or models[k].get("role") == "judge"]
    if args.missing:
        keys = [k for k in keys if not ((SMOKE / f"{k}.json").exists()
                                        and json.loads((SMOKE / f"{k}.json").read_text())["ok"])]
    keys = [k for k in keys if k in models and not models[k].get("excluded")]
    for key in sorted(keys, key=lambda k: models[k]["gpus"]):
        need = models[key]["gpus"]
        if need > len(gpus_avail):
            log(f"{key}: needs {need} GPUs, only {len(gpus_avail)} given -- skipped")
            continue
        smoke_one(key, models[key], gpus_avail[:need], args.port, base)
        write_overview(list(models))
    write_overview(list(models))
    log(f"done -> {SMOKE / 'SMOKE_RESULTS.md'}")


if __name__ == "__main__":
    main()

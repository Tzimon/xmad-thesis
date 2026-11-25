"""
xmad/3_iteration/xmad_debate/debate_loop.py

The orchestrator (C1). Debate logic unchanged from Phase 2 (parallel rounds,
one shared decoding block, one MCP connection per agent, per-agent
parallel-tool-call flag, forced first search in round 1 only); see
2_iteration/xmad_debate/debate_loop.py for the rationale.

Phase 3 changes:
  * Reasoning switch per agent: request_extra_body is merged into every request
    of that agent (the server-side switch lives in the vLLM flags, the
    prompt-side one in the system prompt; see models.yaml).
  * _strip_reasoning: a <think>...</think> block left in the content would
    otherwise be parsed as the answer.
  * Truncated replies (finish_reason == "length") are counted.
  * Transient connection errors to the model server are retried (3 attempts,
    15/45 s back-off) and counted, instead of failing the question at once.
  * Final answer when the turn budget is used up: an agent that is still
    searching after its last turn of the round gets ONE more request, with the
    fixed FINAL_PROMPT as a user message and WITHOUT tools, and has to answer
    with what it found. Same rule and same text for every agent; counted per
    agent as `forced_final`. (Phase 2 kept "(no final reply within max_turns)"
    as the answer in that case. A first version used tool_choice="none" with
    the tools still listed; Qwen and Gemma then returned empty replies.)
  * Tool calls with a name that is not an offered tool (e.g. an empty name,
    seen with Mistral-Small-4) are dropped: sending them back would make the
    server reject the whole conversation (HTTP 400).
  * A request that would exceed the context window is retried once with a
    smaller max_tokens instead of failing the question.
  * No judge inside the debate: a debate without consensus ends with
    needs_judge() True; run_judge() is called later by the judge pass.
"""

import asyncio
import json
import re
import time

from xmad_debate.session import NO_ANSWER

AGENT_KEYS = ("agent_a", "agent_b")
FINAL_PROMPT = ("You have used all searches available in this round. Do not search again. "
                "Answer now from the evidence you have, ending with the two required lines "
                "(Reasoning summary and Final answer).")
SMALL_MAX_TOKENS = 1024
RETRY_ERRORS = {"APIConnectionError", "APITimeoutError", "InternalServerError"}
RETRY_WAITS_S = (15, 45)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text):
    """Remove reasoning that a chat template left in the content: closed
    <think>...</think> blocks; after a dangling closing tag keep only what
    follows; before a dangling opening tag keep only what precedes."""
    if not text:
        return ""
    out = _THINK_BLOCK.sub("", text)
    low = out.lower()
    if "</think>" in low:
        out = out[low.rfind("</think>") + len("</think>"):]
    elif "<think>" in low:
        out = out[:low.find("<think>")]
    return out.strip()


def _create_kwargs(decoding, extra_body):
    kwargs = {"temperature": decoding.temperature, "top_p": decoding.top_p}
    if decoding.max_tokens is not None:
        kwargs["max_tokens"] = decoding.max_tokens
    if extra_body:
        kwargs["extra_body"] = dict(extra_body)
    return kwargs


async def _chat_with_retry(llm_client, stats, **kwargs):
    """One chat completion; transient server errors are retried and counted.
    A context-length overflow is retried once with max_tokens=SMALL_MAX_TOKENS."""
    attempt = 0
    while True:
        try:
            return await llm_client.chat.completions.create(**kwargs)
        except Exception as e:
            name, text = type(e).__name__, str(e)
            if (name == "BadRequestError" and "maximum context length" in text
                    and kwargs.get("max_tokens", 0) > SMALL_MAX_TOKENS):
                kwargs["max_tokens"] = SMALL_MAX_TOKENS
                stats["api_retries"] += 1
                continue
            if name not in RETRY_ERRORS or attempt >= len(RETRY_WAITS_S):
                raise
            stats["api_retries"] += 1
            await asyncio.sleep(RETRY_WAITS_S[attempt])
            attempt += 1


def _mcp_tools_to_openai(mcp_tools, allowed_names):
    return [{"type": "function",
             "function": {"name": t.name, "description": t.description or "",
                          "parameters": getattr(t, "input_schema", None) or t.inputSchema}}
            for t in mcp_tools if t.name in allowed_names]


def _tool_result_to_text(result):
    data = result.data
    if isinstance(data, list):  # search_knowledge_base -> [{title, text, score}]
        blocks = [f"[{p.get('title', '')}] {p.get('text', '')}" for p in data]
        return "\n\n".join(blocks) if blocks else "No results."
    return json.dumps(data)


# --------------------------------------------------------------------------- #
# One agent's turn within one round
# --------------------------------------------------------------------------- #
async def run_agent_turn(agent_key, agent, llm_client, mcp_client, decoding, allowed_tools,
                         system_prompt, context_prefix, question, round_num, force_first_search):
    stats = {"llm_calls": 0, "tool_calls": 0, "truncated": 0, "api_retries": 0, "forced_final": 0,
             "invalid_tool_calls": 0,
             "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    trace = []
    tools = _mcp_tools_to_openai(await mcp_client.list_tools(), allowed_tools)

    user_content = f"{context_prefix}\n\nQuestion: {question}" if context_prefix else question
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}]
    create_kwargs = _create_kwargs(decoding, agent.request_extra_body)

    def _count(resp):
        stats["llm_calls"] += 1
        if resp.usage is not None:
            stats["prompt_tokens"] += resp.usage.prompt_tokens or 0
            stats["completion_tokens"] += resp.usage.completion_tokens or 0
            stats["total_tokens"] += resp.usage.total_tokens or 0
        if resp.choices[0].finish_reason == "length":
            stats["truncated"] += 1

    raw = None
    for turn in range(agent.max_turns_per_round):
        tool_choice = "required" if (turn == 0 and round_num == 1 and force_first_search) else "auto"
        resp = await _chat_with_retry(llm_client, stats, model=agent.alias, messages=messages,
                                      tools=tools, tool_choice=tool_choice, **create_kwargs)
        _count(resp)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            raw = _strip_reasoning(msg.content) or None
            break

        offered = {t["function"]["name"] for t in tools}
        tool_calls = [tc for tc in msg.tool_calls if tc.function.name in offered]
        for tc in msg.tool_calls:
            if tc.function.name not in offered:
                stats["invalid_tool_calls"] += 1
                trace.append({"name": tc.function.name, "args": tc.function.arguments, "ms": 0.0,
                              "retrieved_titles": [], "error": "dropped: not an offered tool"})
        if not tool_calls:          # only invalid calls: treat the reply like a final reply
            raw = _strip_reasoning(msg.content) or None
            break
        if len(tool_calls) > 1 and not agent.supports_parallel_tool_calls:
            tool_calls = tool_calls[:1]   # model cannot replay >1 call per turn (Phase 2, block 3b)

        messages.append({
            "role": "assistant",
            "content": _strip_reasoning(msg.content),
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in tool_calls],
        })
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            t1 = time.perf_counter()
            try:
                result = await mcp_client.call_tool(tc.function.name, args)
                titles = [p.get("title", "") for p in result.data] if isinstance(result.data, list) else []
                trace.append({"name": tc.function.name, "args": args,
                              "ms": round((time.perf_counter() - t1) * 1000, 1), "retrieved_titles": titles})
                tool_content = _tool_result_to_text(result)
            except Exception as e:
                # a malformed tool call must not abort the question (Phase 2, block 3c)
                trace.append({"name": tc.function.name, "args": args,
                              "ms": round((time.perf_counter() - t1) * 1000, 1), "retrieved_titles": [],
                              "error": str(e)[:200]})
                tool_content = f"Error: tool call failed ({type(e).__name__}: {e})"
            stats["tool_calls"] += 1
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_content})

    if raw is None:
        # no answer yet (budget used up, or an empty reply): one request with the
        # fixed FINAL_PROMPT and without tools
        resp = await _chat_with_retry(llm_client, stats, model=agent.alias,
                                      messages=messages + [{"role": "user", "content": FINAL_PROMPT}],
                                      **create_kwargs)
        _count(resp)
        stats["forced_final"] = 1
        raw = _strip_reasoning(resp.choices[0].message.content) or NO_ANSWER

    return {"raw": raw, "tool_trace": trace, **stats}


# --------------------------------------------------------------------------- #
# The full debate for one question (no judge -- see module docstring)
# --------------------------------------------------------------------------- #
async def run_debate(question, session, cfg, llm_clients, mcp_a, mcp_b):
    a, b = cfg.agents.agent_a, cfg.agents.agent_b
    prompt_a, prompt_b = cfg.system_prompt_for(a), cfg.system_prompt_for(b)
    allowed = set(cfg.tools.agent_tools)

    while not session.is_done():
        round_num = session.next_round_number()
        ctx_a = session.render_context_for("agent_a")
        ctx_b = session.render_context_for("agent_b")
        turn_a, turn_b = await asyncio.gather(
            run_agent_turn("agent_a", a, llm_clients[a.url], mcp_a, cfg.decoding, allowed,
                           prompt_a, ctx_a, question, round_num, cfg.debate.force_first_search),
            run_agent_turn("agent_b", b, llm_clients[b.url], mcp_b, cfg.decoding, allowed,
                           prompt_b, ctx_b, question, round_num, cfg.debate.force_first_search),
        )
        for key, t in (("agent_a", turn_a), ("agent_b", turn_b)):
            session.record_turn(key, t["raw"], tool_trace=t["tool_trace"],
                                llm_calls=t["llm_calls"], tool_calls=t["tool_calls"],
                                truncated=t["truncated"], api_retries=t["api_retries"],
                                forced_final=t["forced_final"], invalid_tool_calls=t["invalid_tool_calls"],
                                prompt_tokens=t["prompt_tokens"],
                                completion_tokens=t["completion_tokens"],
                                total_tokens=t["total_tokens"])
    return session


# --------------------------------------------------------------------------- #
# The judge (C4) -- used by the judge pass (phase3.py judge)
# --------------------------------------------------------------------------- #
def _extract_judge_fields(raw):
    """'Confidence: <float>' and 'Final answer: <text>' (unchanged from Phase 2)."""
    def _after_marker(text, marker):
        idx = text.lower().rfind(marker.lower())
        if idx == -1:
            return None
        tail = text[idx + len(marker):].strip()
        cut = tail.lower().find("final answer:") if marker.lower() != "final answer:" else -1
        if cut != -1:
            tail = tail[:cut].strip()
        return tail.strip().strip('"').strip(".").strip()

    conf_text = _after_marker(raw, "Confidence:")
    final = _after_marker(raw, "Final answer:")
    try:
        confidence = float(conf_text) if conf_text is not None else 0.5
    except ValueError:
        confidence = 0.5
    if final is None:
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        final = lines[-1] if lines else raw.strip()
    return confidence, final


async def run_judge(session, judge_prompt, llm_client, alias, decoding, extra_body=None):
    """Judge one debate that ended without consensus (no tools, no gold answer)."""
    last = session.completed_rounds()[-1]
    user_content = (
        f"Question: {session.question}\n\n"
        f"{session.render_transcript_for_judge()}\n\n"
        f"Agent A's final answer: {last.turns['agent_a'].final_answer}\n"
        f"Agent B's final answer: {last.turns['agent_b'].final_answer}"
    )
    stats = {"api_retries": 0}
    resp = await _chat_with_retry(
        llm_client, stats, model=alias,
        messages=[{"role": "system", "content": judge_prompt},
                  {"role": "user", "content": user_content}],
        **_create_kwargs(decoding, extra_body))
    raw = _strip_reasoning(resp.choices[0].message.content)
    confidence, final_answer = _extract_judge_fields(raw)
    return {"raw": raw, "confidence": confidence, "final_answer": final_answer,
            "judge_model": alias, "finish_reason": resp.choices[0].finish_reason}

"""
xmad/1_iteration/xmad_rag/baseline_agent_loop.py

Building block 3 (as used by the single-agent RAG baseline): the agentic loop.

The model searches the knowledge base, reasons over the retrieved passages, and
ends with a short final answer. No agent framework -- the loop is explicit so its
behaviour is attributable to the design, not to a library.

Design points that make this a correct, scorable, well-instrumented RAG baseline:
  * answers only from retrieved passages (system prompt);
  * the first turn forces a tool call (tool_choice="required"), so every question
    is grounded in retrieval, not parametric knowledge;
  * the model reasons, then emits "Final answer: <phrase>" which is extracted for
    exact-match/F1 scoring;
  * every LLM call and every tool call is timed, and each tool call is recorded in
    an ordered trace (name, arguments, milliseconds), so the latency can be split
    into model time vs. tool time and the tool usage can be analysed even once more
    tools exist.

Connection handling: run_agent reuses an MCP client and an AsyncOpenAI client
passed in by the harness (one connection for the whole run); it opens its own only
when none is given (single-question test).

run_agent returns:
    {answer, raw, llm_calls, tool_calls, prompt_tokens, completion_tokens,
     total_tokens, llm_time_s, tool_time_s, tool_trace}
"""

import json
import time

from fastmcp import Client as MCPClient
from openai import AsyncOpenAI

AGENT_TOOLS = {"search_knowledge_base"}

SYSTEM_PROMPT = (
    "You are a research assistant answering multi-hop questions from a knowledge "
    "base. Always use the search_knowledge_base tool to gather evidence before "
    "answering, and base your answer only on the retrieved passages, not on prior "
    "knowledge. You may search several times with different queries, for example one "
    "entity at a time. Reason step by step through the retrieved evidence, then end "
    "your reply with a single final line in exactly this format:\n"
    "Final answer: <shortest possible phrase: a single word, name, number, or yes/no>"
)


def _extract_answer(text):
    marker = "final answer:"
    idx = text.lower().rfind(marker)
    if idx != -1:
        return text[idx + len(marker):].strip().strip('"').strip(".").strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else text.strip()


def _mcp_tools_to_openai(mcp_tools):
    out = []
    for t in mcp_tools:
        if t.name not in AGENT_TOOLS:
            continue
        out.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        })
    return out


def _tool_result_to_text(result):
    data = result.data
    if isinstance(data, list):
        blocks = [f"[{p.get('title', '')}] {p.get('text', '')}" for p in data]
        return "\n\n".join(blocks) if blocks else "No results."
    return json.dumps(data)


async def _run_loop(question, mcp, llm, model, system_prompt, max_turns,
                    force_first_search, verbose):
    stats = {"llm_calls": 0, "tool_calls": 0,
             "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
             "llm_time_s": 0.0, "tool_time_s": 0.0}
    trace = []  # ordered [{name, args, ms}] across all tool calls

    def _accumulate(usage):
        if usage is None:
            return
        stats["prompt_tokens"] += usage.prompt_tokens or 0
        stats["completion_tokens"] += usage.completion_tokens or 0
        stats["total_tokens"] += usage.total_tokens or 0

    tools = _mcp_tools_to_openai(await mcp.list_tools())
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    raw = "(no final answer within max_turns)"
    for turn in range(max_turns):
        tool_choice = "required" if (turn == 0 and force_first_search) else "auto"

        t0 = time.perf_counter()
        resp = await llm.chat.completions.create(
            model=model, messages=messages, tools=tools, tool_choice=tool_choice,
        )
        stats["llm_time_s"] += time.perf_counter() - t0
        stats["llm_calls"] += 1
        _accumulate(resp.usage)
        msg = resp.choices[0].message

        if not msg.tool_calls:
            raw = msg.content or ""
            break

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            if verbose:
                print(f"  [tool] {tc.function.name}({args})")

            t0 = time.perf_counter()
            result = await mcp.call_tool(tc.function.name, args)
            dt = time.perf_counter() - t0

            stats["tool_time_s"] += dt
            stats["tool_calls"] += 1
            trace.append({"name": tc.function.name, "args": args,
                          "ms": round(dt * 1000, 1)})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": _tool_result_to_text(result),
            })

    stats["llm_time_s"] = round(stats["llm_time_s"], 4)
    stats["tool_time_s"] = round(stats["tool_time_s"], 4)
    return {"answer": _extract_answer(raw), "raw": raw, "tool_trace": trace, **stats}


async def run_agent(
    question,
    model="qwen",
    vllm_url="http://127.0.0.1:7555/v1",
    mcp_url="http://127.0.0.1:7600/mcp",
    mcp=None,
    llm=None,
    system_prompt=SYSTEM_PROMPT,
    max_turns=6,
    force_first_search=True,
    verbose=False,
):
    if llm is None:
        llm = AsyncOpenAI(base_url=vllm_url, api_key="not-needed")

    if mcp is not None:
        return await _run_loop(question, mcp, llm, model, system_prompt,
                               max_turns, force_first_search, verbose)

    async with MCPClient(mcp_url) as own_mcp:
        return await _run_loop(question, own_mcp, llm, model, system_prompt,
                               max_turns, force_first_search, verbose)

"""
xmad/1_iteration/xmad_rag/baseline_agent_loop.py

Building block 3 (as used by the single-agent RAG baseline): the agentic loop.

A single tool-capable model (served by vLLM) is driven through a hand-written
loop: it searches the knowledge base, reasons over the retrieved passages, and
ends with a short final answer. No agent framework -- the loop is explicit so its
behaviour is attributable to the design, not to a library.

Three design points make this a correct, scorable RAG baseline:
  * the system prompt requires the model to answer only from retrieved passages;
  * the first turn forces a tool call (tool_choice="required"), so every question
    is grounded in retrieval rather than parametric knowledge;
  * the model reasons step by step (needed for multi-hop questions), then emits a
    "Final answer: <phrase>" line that is extracted for exact-match/F1 scoring.
    Forbidding the reasoning outright makes the model guess and lowers accuracy;
    letting it reason and then extracting the short span gives both correctness
    and a metric-ready answer.

The agent is shown only search_knowledge_base; load_corpus is filtered out. The
per-question corpus must already be loaded on the MCP server. run_agent returns a
dict with the extracted answer, the full raw reply, and per-question cost stats.
"""

import json

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
    """Pull the short answer from the model's 'Final answer:' line; fall back to
    the last non-empty line if the marker is absent."""
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
    if isinstance(data, list):  # search_knowledge_base -> [{title, text, score}]
        blocks = [f"[{p.get('title', '')}] {p.get('text', '')}" for p in data]
        return "\n\n".join(blocks) if blocks else "No results."
    return json.dumps(data)


async def run_agent(
    question,
    model="qwen",
    vllm_url="http://127.0.0.1:7555/v1",
    mcp_url="http://127.0.0.1:7600/mcp",
    system_prompt=SYSTEM_PROMPT,
    max_turns=6,
    force_first_search=True,
    verbose=False,
):
    """Run one question through the agentic loop.

    Returns a dict:
        {answer, raw, llm_calls, tool_calls, prompt_tokens, completion_tokens,
         total_tokens}
    where `answer` is the extracted short span and `raw` is the full final reply.
    """
    llm = AsyncOpenAI(base_url=vllm_url, api_key="not-needed")

    stats = {"llm_calls": 0, "tool_calls": 0,
             "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _accumulate(usage):
        if usage is None:
            return
        stats["prompt_tokens"] += usage.prompt_tokens or 0
        stats["completion_tokens"] += usage.completion_tokens or 0
        stats["total_tokens"] += usage.total_tokens or 0

    async with MCPClient(mcp_url) as mcp:
        tools = _mcp_tools_to_openai(await mcp.list_tools())

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        raw = "(no final answer within max_turns)"
        for turn in range(max_turns):
            tool_choice = "required" if (turn == 0 and force_first_search) else "auto"
            resp = await llm.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
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
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                if verbose:
                    print(f"  [tool] {tc.function.name}({args})")
                result = await mcp.call_tool(tc.function.name, args)
                stats["tool_calls"] += 1
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _tool_result_to_text(result),
                })

    return {"answer": _extract_answer(raw), "raw": raw, **stats}

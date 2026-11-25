"""
xmad/1_iteration/xmad_rag/baseline_agent_loop.py

Building block 3: the agentic loop.

A single tool-capable model (served by vLLM) is driven through a hand-written
loop: it is given the question and the tool schema, decides to call
search_knowledge_base, receives the retrieved passages, and continues until it
produces a final answer. No agent framework -- the loop is explicit so its
behaviour is attributable to the design, not to a library.

It connects to two services:
  * the vLLM OpenAI-compatible endpoint (the model), via the openai client;
  * the shared MCP tool pool (building block 2), via the fastmcp client.

The agent is shown only search_knowledge_base; load_corpus (the harness control
tool) is filtered out of the advertised tool list.

The per-question corpus must already be loaded on the MCP server (the harness
calls load_corpus before run_agent).

run_agent returns a dict with the answer and the per-question cost statistics
(LLM calls, tool calls, token counts) that the baseline records for C5.1/C5.3.
"""

import json

from fastmcp import Client as MCPClient
from openai import AsyncOpenAI

# Tools the agent is allowed to see. load_corpus is control-plane, not agent-facing.
AGENT_TOOLS = {"search_knowledge_base"}

# The final answer must be short so the HotpotQA EM/F1 metrics apply. The model may
# still reason and search freely; only the final message is constrained.
SYSTEM_PROMPT = (
    "You are a careful research assistant answering multi-hop questions. Use the "
    "search_knowledge_base tool to look up evidence before answering; you may search "
    "more than once with different queries. When you have enough evidence, reply with "
    "the shortest possible answer -- a single word, name, number, or 'yes'/'no' -- and "
    "nothing else. Do not restate the question or explain your reasoning in the final "
    "answer."
)


def _mcp_tools_to_openai(mcp_tools):
    """Translate MCP tool definitions to the OpenAI tools format, keeping only
    the agent-facing tools."""
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
    """Turn an MCP tool result into a string the model can read."""
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
    verbose=False,
):
    """Run one question through the agentic loop.

    Returns a dict:
        {answer, llm_calls, tool_calls, prompt_tokens, completion_tokens, total_tokens}
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

        answer = "(no final answer within max_turns)"
        for _ in range(max_turns):
            resp = await llm.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            stats["llm_calls"] += 1
            _accumulate(resp.usage)
            msg = resp.choices[0].message

            # No tool call -> final answer.
            if not msg.tool_calls:
                answer = msg.content or ""
                break

            # Record the assistant turn (with its tool calls) before the results.
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

            # Execute each tool call via the shared MCP pool and append the result.
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

    return {"answer": answer, **stats}

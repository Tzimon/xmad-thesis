"""
xmad/1_iteration/xmad_rag/agent_loop.py

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
"""

import json

from fastmcp import Client as MCPClient
from openai import AsyncOpenAI

# Tools the agent is allowed to see. load_corpus is control-plane, not agent-facing.
AGENT_TOOLS = {"search_knowledge_base"}

SYSTEM_PROMPT = (
    "You are a careful research assistant. Use the search_knowledge_base tool to "
    "look up evidence before answering; you may search more than once with "
    "different queries. When you have enough evidence, give a short, direct final "
    "answer to the question."
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
    max_turns=6,
    verbose=False,
):
    """Run one question through the agentic loop; return the final answer text."""
    llm = AsyncOpenAI(base_url=vllm_url, api_key="not-needed")

    async with MCPClient(mcp_url) as mcp:
        tools = _mcp_tools_to_openai(await mcp.list_tools())

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        for _ in range(max_turns):
            resp = await llm.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            msg = resp.choices[0].message

            # No tool call -> final answer.
            if not msg.tool_calls:
                return msg.content or ""

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
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _tool_result_to_text(result),
                })

        return "(no final answer within max_turns)"

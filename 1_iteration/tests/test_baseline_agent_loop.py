"""
xmad/1_iteration/tests/test_baseline_agent_loop.py

One-question end-to-end test of the REAL baseline agent loop (building block 3,
as used by the single-agent RAG baseline). Mirrors one iteration of run_baseline.

Requires, on the same node:
  * vLLM serving Qwen with tool calling  (port 7555)
  * the MCP server running                (port 7600)

    python tests/test_baseline_agent_loop.py
"""

import asyncio

from datasets import load_dataset
from fastmcp import Client as MCPClient

from xmad_rag.baseline_agent_loop import run_agent
from xmad_rag.retriever import build_corpus

MCP_URL = "http://127.0.0.1:7600/mcp"


async def main():
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    ex = ds[0]

    async with MCPClient(MCP_URL) as mcp:
        r = await mcp.call_tool("load_corpus", {"paragraphs": build_corpus(ex)})
        print("load_corpus:", r.data)

    print("\nQ:", ex["question"])
    print("Gold answer:", ex["answer"])
    print("--- running agent ---")
    res = await run_agent(ex["question"], verbose=True)
    print("--- full reply (with reasoning) ---")
    print(res["raw"])
    print("--- extracted answer ---")
    print(repr(res["answer"]))
    print(f"(tool calls: {res['tool_calls']}, total tokens: {res['total_tokens']})")


if __name__ == "__main__":
    asyncio.run(main())

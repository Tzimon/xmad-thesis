"""
xmad/1_iteration/tests/test_agent_loop.py

One-question end-to-end test of the agent loop (building block 3).

Requires, on the same node:
  * vLLM serving Qwen with tool calling  (port 7555)
  * the MCP server running                (port 7600)

    python tests/test_agent_loop.py

This mimics one iteration of the eval harness: set the question's corpus via
load_corpus, then run the agent and print its answer next to the gold answer.
"""

import asyncio

from datasets import load_dataset
from fastmcp import Client as MCPClient

from xmad_rag.agent_loop import run_agent
from xmad_rag.retriever import build_corpus

MCP_URL = "http://127.0.0.1:7600/mcp"


async def main():
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    ex = ds[0]

    # Harness step: load this question's 10 paragraphs onto the MCP server.
    async with MCPClient(MCP_URL) as mcp:
        r = await mcp.call_tool("load_corpus", {"paragraphs": build_corpus(ex)})
        print("load_corpus:", r.data)

    print("\nQ:", ex["question"])
    print("Gold answer:", ex["answer"])
    print("--- running agent ---")
    answer = await run_agent(ex["question"], verbose=True)
    print("--- agent answer ---")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())

"""
xmad/1_iteration/test_mcp_client.py

Verifies the MCP server (building block 2) end to end over Streamable HTTP.
No LLM/agent yet -- this only confirms the tool pool works.

Run the server and this client ON THE SAME NODE (localhost from the login node
to a compute node does not route -- use a second shell on the allocation:
    srun --jobid=<ID> --overlap --pty bash

Shell A:  python mcp_server.py      # wait for the "serving tools" line
Shell B:  python test_mcp_client.py
"""

import asyncio

from fastmcp import Client

URL = "http://127.0.0.1:7600/mcp"


async def main():
    async with Client(URL) as client:
        tools = await client.list_tools()
        print("Tools:", [t.name for t in tools])

        # 1) search the demo corpus loaded at startup
        r = await client.call_tool("search_knowledge_base",
                                   {"query": "river in Paris", "k": 2})
        print("\nsearch (demo corpus):")
        for p in r.data:
            print(f"  [{p['score']}] {p['title']}")

        # 2) load a fresh corpus, then search it -- this is exactly what the eval
        #    harness will do per HotpotQA question
        await client.call_tool("load_corpus", {"paragraphs": [
            {"title": "Ada Lovelace",
             "text": "Ada Lovelace was a 19th-century mathematician regarded as the first computer programmer."},
            {"title": "Charles Babbage",
             "text": "Charles Babbage designed the Analytical Engine, a mechanical general-purpose computer."},
        ]})
        r = await client.call_tool("search_knowledge_base",
                                   {"query": "who was the first computer programmer", "k": 1})
        print("\nsearch (after load_corpus):")
        for p in r.data:
            print(f"  [{p['score']}] {p['title']}: {p['text'][:60]}")


if __name__ == "__main__":
    asyncio.run(main())

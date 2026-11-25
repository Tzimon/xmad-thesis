"""
xmad/1_iteration/mcp_server.py

Building block 2: the MCP server -- the shared MCP tool pool (C2.1).

Exposes the retrieval tool over Streamable HTTP, so any agent (the single-agent
baseline now, the debate agents later) reaches it through one shared,
standardized interface that is separate from the agents. Backed by the
DenseKnowledgeBase from building block 1.

Tools:
  * search_knowledge_base(query, k) -- agent-facing (C2.1 / C3.1).
  * load_corpus(paragraphs)         -- control tool the eval harness calls to set
                                       the per-question corpus before an agent
                                       runs. This is eval plumbing: we simply do
                                       not put it in the agent's advertised tool
                                       list, so the debate agents never see it.

Transport: Streamable HTTP (the current MCP standard; SSE is deprecated). The
pool is one service behind a URL that many agent clients can reach concurrently
-- exactly what C2.1 requires and what Phase 2's multiple agents need.

Run (on a GPU node, alongside the vLLM servers):
    pip install fastmcp        # verified against fastmcp 3.4.7; pin for repro
    python mcp_server.py       # serves http://127.0.0.1:7600/mcp

Depends on: fastmcp, plus block 1's retriever_dense (transformers/torch).

Note: search_knowledge_base runs a GPU forward pass synchronously, which blocks
the event loop. Fine for the sequential Phase-1 baseline; for concurrent debate
agents in Phase 2 we can make it async / offload to a thread.
"""

from fastmcp import FastMCP

from xmad_rag.retriever_dense import DenseKnowledgeBase

HOST = "127.0.0.1"
PORT = 7600

# One knowledge base; the embedding model is loaded once and kept in memory.
_kb = DenseKnowledgeBase()

# Small demo corpus so the server is testable standalone, before the harness
# wires in real HotpotQA corpora. Replaced per question via load_corpus().
_DEMO_CORPUS = [
    {"title": "Eiffel Tower", "text": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France."},
    {"title": "Seine", "text": "The Seine is a river that flows through Paris into the English Channel."},
    {"title": "Mount Everest", "text": "Mount Everest is the highest mountain above sea level."},
]
_kb.load(_DEMO_CORPUS)

mcp = FastMCP("xmad-tools")


@mcp.tool
def search_knowledge_base(query: str, k: int = 3) -> list[dict]:
    """Search the knowledge base for passages relevant to the query.

    Args:
        query: A natural-language search query.
        k: How many passages to return (default 3).

    Returns:
        A list of {title, text, score}, most relevant first.
    """
    return _kb.search(query, k=k)


@mcp.tool
def load_corpus(paragraphs: list[dict]) -> dict:
    """Control tool (eval harness only): set the active per-question corpus.

    Args:
        paragraphs: A list of {title, text} dicts for the current question.

    Returns:
        {"loaded": n} confirming how many passages are now active.
    """
    _kb.load(paragraphs)
    return {"loaded": len(paragraphs)}


if __name__ == "__main__":
    print(f"[mcp] knowledge base ready on device: {_kb.device}")
    print(f"[mcp] serving tools at http://{HOST}:{PORT}/mcp")
    mcp.run(transport="http", host=HOST, port=PORT)

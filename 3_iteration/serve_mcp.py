"""
xmad/3_iteration/serve_mcp.py

Starts the unchanged Phase 1 MCP tool server (xmad_rag.mcp_server: dense bge
retriever, tools search_knowledge_base + load_corpus) on a given port, so that
several runs on the same node can each have their own server. The server holds
one corpus at a time, which is why every run needs its own.

    CUDA_VISIBLE_DEVICES=3 python serve_mcp.py --port 8005
"""

import argparse

from xmad_rag import mcp_server   # loads the retriever on import

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    print(f"[mcp] knowledge base on device {mcp_server._kb.device}, port {args.port}", flush=True)
    mcp_server.mcp.run(transport="http", host="127.0.0.1", port=args.port)

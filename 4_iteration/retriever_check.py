"""
xmad/4_iteration/retriever_check.py

Recall@k of the unchanged Phase 1 dense retriever (bge-base) on each benchmark,
as in thesis Table "tab:retriever-recall" (Phase 1). Needed before the runs:
MuSiQue has up to about 20 paragraphs per question (HotpotQA / 2Wiki: 10), and the agent tool
returns k = 3 passages per call -- this checks whether that is still enough.

    CUDA_VISIBLE_DEVICES=0 python retriever_check.py --n 50
    -> analysis/retriever_recall.csv  (benchmark, k, recall@k)

Query = the question itself (as in Phase 1). Recall@k = share of the
supporting paragraphs among the top-k passages.
"""

import argparse
import csv
import sys
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "1_iteration"))

from xmad_bench import evidence_recall, load_benchmark  # noqa: E402
import servers  # noqa: E402

KS = [1, 2, 3, 5, 10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--benchmarks", nargs="+", default=["hotpotqa", "musique", "2wiki"])
    args = ap.parse_args()
    # the retriever of the MCP server (xmad_rag/mcp_server.py), not the BM25 KnowledgeBase of
    # xmad_rag/retriever.py -- fixed 2026-11-03: the first run imported the BM25 class by mistake
    # (its result is kept as analysis/retriever_recall_bm25.csv)
    from xmad_rag.retriever_dense import DenseKnowledgeBase

    base = servers.load_base()
    kb = DenseKnowledgeBase()
    rows = []
    for name in args.benchmarks:
        b = base["benchmarks"][name]
        exs = load_benchmark(b["name"], b["config"], b["split"])
        idx = list(range(len(exs))); Random(0).shuffle(idx)
        sums = {k: 0.0 for k in KS}
        for i in idx[: args.n]:
            ex = exs[i]
            kb.load(ex.paragraphs)
            hits = [p["title"] for p in kb.search(ex.question, k=max(KS))]
            for k in KS:
                sums[k] += evidence_recall(hits[:k], ex.gold_titles) or 0.0
        for k in KS:
            rows.append({"benchmark": name, "k": k, "recall": round(sums[k] / args.n, 3), "n": args.n})
        print(name, " ".join(f"R@{k}={sums[k] / args.n:.2f}" for k in KS), flush=True)
    out = HERE / "analysis" / "retriever_recall.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["benchmark", "k", "recall", "n"]); w.writeheader(); w.writerows(rows)
    print("->", out)


if __name__ == "__main__":
    main()

"""
xmad/1_iteration/test_retriever.py

Sanity check for building block 1. No MCP, no agent, no LLM.

Validates that BM25 retrieval over the HotpotQA-distractor corpus surfaces the
gold paragraphs. If recall is high, the retriever is good enough to wrap as the
search_knowledge_base MCP tool in building block 2. If recall@k is disappointing,
that is the signal to switch BM25 for a dense retriever (sentence-transformers).

Run:  python test_retriever.py
"""

from datasets import load_dataset

from xmad_rag.retriever import KnowledgeBase, build_corpus, gold_titles


def recall_at_k(dataset, n_questions=50, ks=(2, 3, 5)):
    """Micro-averaged gold-paragraph recall at several values of k."""
    kb = KnowledgeBase()
    hits = {k: 0 for k in ks}
    total = 0
    for ex in dataset.select(range(n_questions)):
        kb.load(build_corpus(ex))
        gold = gold_titles(ex)
        total += len(gold)
        for k in ks:
            retrieved = {p["title"] for p in kb.search(ex["question"], k=k)}
            hits[k] += len(retrieved & gold)
    return {k: (hits[k] / total if total else 0.0) for k in ks}


if __name__ == "__main__":
    print("Loading HotpotQA (distractor, validation split) ...")
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    print(f"Loaded {len(ds)} questions.\n")

    # Qualitative: look at one example
    kb = KnowledgeBase()
    ex = ds[0]
    kb.load(build_corpus(ex))
    print("Q:", ex["question"])
    print("Gold paragraphs:", sorted(gold_titles(ex)))
    print("Top-3 retrieved:", [p["title"] for p in kb.search(ex["question"], k=3)])
    print()

    # Quantitative: recall sanity check
    scores = recall_at_k(ds, n_questions=50, ks=(2, 3, 5))
    print("Gold-paragraph recall (micro-avg over 50 questions):")
    for k, r in scores.items():
        print(f"  recall@{k}: {r:.3f}")

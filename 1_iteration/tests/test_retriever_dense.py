"""
xmad/1_iteration/test_retriever_dense.py

Same recall sanity check as test_retriever.py, but with the dense retriever.
The embedding model is loaded once and reused across questions.

Run on a GPU node for speed (e.g. via srun); it falls back to CPU otherwise,
which is still fine for 50 questions.

    python test_retriever_dense.py
"""

from datasets import load_dataset

from xmad_rag.retriever import build_corpus, gold_titles
# from xmad_rag.retriever_dense import DenseKnowledgeBase    #BGE-en-v1.5
from xmad_rag.qwen_retriever_dense import DenseKnowledgeBase #Nehmen für Qwen3-Embedding-0.6B


def recall_at_k(kb, dataset, n_questions=50, ks=(2, 3, 5)):
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
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    print(f"Loaded {len(ds)} questions.")

    print("Loading embedding model (once) ...")
    kb = DenseKnowledgeBase()  # Qwen/Qwen3-Embedding-0.6B
    print(f"Device: {kb.device}\n")

    # Qualitative: the same example that BM25 got wrong
    ex = ds[0]
    kb.load(build_corpus(ex))
    print("Q:", ex["question"])
    print("Gold paragraphs:", sorted(gold_titles(ex)))
    print("Top-3 retrieved:", [p["title"] for p in kb.search(ex["question"], k=3)])
    print()

    scores = recall_at_k(kb, ds, n_questions=50, ks=(2, 3, 5))
    print("Gold-paragraph recall (micro-avg over 50 questions):")
    for k, r in scores.items():
        print(f"  recall@{k}: {r:.3f}")

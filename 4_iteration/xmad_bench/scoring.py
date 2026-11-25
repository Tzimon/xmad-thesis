"""
xmad/4_iteration/xmad_bench/scoring.py

Answer scoring for all benchmarks: the normalized exact match and token F1 of
Phase 1-3 (xmad_rag.metrics, unchanged), taken as the maximum over all gold
answers. With one gold answer (HotpotQA, 2Wiki) this is identical to Phase 3.
MuSiQue lists aliases; its official script scores the max over
[answer] + answer_aliases in the same way (StonyBrookNLP/musique, evaluate_v1.0.py).

Evidence recall: share of the supporting paragraphs whose title was retrieved.
"""

from xmad_rag.metrics import exact_match, f1_score


def em_max(pred, answers):
    return max((exact_match(pred, a) for a in answers), default=0) if pred else 0


def f1_max(pred, answers):
    return max((f1_score(pred, a) for a in answers), default=0.0) if pred else 0.0


def evidence_recall(retrieved_titles, gold_titles):
    """None if the question has no supporting paragraphs (should not happen)."""
    gold = set(gold_titles)
    if not gold:
        return None
    return len(gold & set(retrieved_titles)) / len(gold)

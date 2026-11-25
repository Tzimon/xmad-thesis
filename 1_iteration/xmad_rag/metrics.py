"""
xmad/1_iteration/xmad_rag/metrics.py

Official HotpotQA / SQuAD answer scoring: normalized exact match and token F1.
Normalisation lowercases, drops punctuation and the articles a/an/the, and
collapses whitespace, so "Yes." and "The Walt Disney Company" score correctly.
"""

import re
import string
from collections import Counter


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred, gold):
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_toks)
    recall = same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)

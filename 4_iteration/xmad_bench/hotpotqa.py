"""
xmad/4_iteration/xmad_bench/hotpotqa.py

HotpotQA, distractor setting, exactly as in Phase 2/3: Hugging Face
hotpotqa/hotpot_qa, 10 paragraphs per question, one gold answer. Kept so that
the Phase 4 harness can be checked against Phase 3 (same sample, same scores).
"""

from .base import Example


def load(config="distractor", split="validation", data_dir=None):
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", config, split=split)
    out = []
    for ex in ds:
        paragraphs = [{"title": t, "text": " ".join(s).strip()}
                      for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])]
        out.append(Example(id=ex["id"], question=ex["question"], answers=[ex["answer"]],
                           paragraphs=paragraphs, gold_titles=set(ex["supporting_facts"]["title"]),
                           meta={"type": ex.get("type", ""), "level": ex.get("level", "")}))
    return out

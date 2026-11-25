"""
xmad/4_iteration/xmad_bench/twowiki.py

2WikiMultiHopQA (Ho et al., COLING 2020), official release of the authors
(github.com/Alab-NII/2wikimultihop, data_ids_april7.zip -- the version with the
fixed sentence segmentation), downloaded by download_benchmarks.py to data/2wiki/.

Same format as HotpotQA (README "Dataset Contents"):
    _id, question, answer, type, context: [[title, [sentences]]],
    supporting_facts: [[title, sent_id]], evidences, entity_ids
10 paragraphs per question, one gold answer (the official v1.0 script scores
against this single answer). Question types: comparison, inference,
compositional, bridge_comparison -- kept in meta for a per-type breakdown.
"""

import json
from pathlib import Path

from .base import Example, unique_titles

FILES = {("april7", "dev"): "dev.json", ("april7", "train"): "train.json"}


def load(config="april7", split="dev", data_dir="data"):
    path = Path(data_dir) / "2wiki" / FILES[(config, split)]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run: python download_benchmarks.py 2wiki")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for ex in data:
        paragraphs, mapping = unique_titles([(t, " ".join(s)) for t, s in ex["context"]])
        # a supporting fact names a title; if the title occurs twice, count its first paragraph
        gold = {mapping[t][0] for t, _ in ex["supporting_facts"] if t in mapping}
        out.append(Example(id=ex["_id"], question=ex["question"], answers=[ex["answer"]],
                           paragraphs=paragraphs, gold_titles=gold, meta={"type": ex.get("type", "")}))
    return out

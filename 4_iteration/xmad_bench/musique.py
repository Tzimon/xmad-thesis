"""
xmad/4_iteration/xmad_bench/musique.py

MuSiQue-Ans (Trivedi et al., TACL 2022), official release of the authors
(github.com/StonyBrookNLP/musique, musique_v1.0.zip, CC BY 4.0), downloaded by
download_benchmarks.py to data/musique/.

Per question: up to about 20 paragraphs (several supporting), one answer plus aliases.
The exact distribution is printed by `python tests/test_phase4_offline.py --real`.
Only answerable questions are used (MuSiQue-Ans), so EM/F1 stay the metric.

Fields used (one JSON object per line):
    id, question, answer, answer_aliases, answerable,
    paragraphs: [{idx, title, paragraph_text, is_supporting}]
`id`, `paragraphs[].idx/is_supporting`, `answer`, `answer_aliases` and
`answerable` are confirmed by the official evaluate_v1.0.py; `question`,
`title` and `paragraph_text` were confirmed on the real file (2,417 answerable
dev questions read by `python tests/test_phase4_offline.py --real`).
"""

import json
import re
from pathlib import Path

from .base import Example, unique_titles

FILES = {("answerable", "dev"): "musique_ans_v1.0_dev.jsonl",
         ("answerable", "train"): "musique_ans_v1.0_train.jsonl"}


def hops(qid):
    """'2hop__...' -> 2, '3hop1__...' -> 3, '4hop2__...' -> 4; None if the id has another form."""
    m = re.match(r"(\d)hop", qid or "")
    return int(m.group(1)) if m else None


def load(config="answerable", split="dev", data_dir="data"):
    path = Path(data_dir) / "musique" / FILES[(config, split)]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run: python download_benchmarks.py musique")
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            if not ex.get("answerable", True):
                continue
            paras = sorted(ex["paragraphs"], key=lambda p: p["idx"])
            paragraphs, mapping = unique_titles([(p["title"], p["paragraph_text"]) for p in paras])
            gold = {paragraphs[i]["title"] for i, p in enumerate(paras) if p.get("is_supporting")}
            out.append(Example(id=ex["id"], question=ex["question"],
                               answers=[ex["answer"]] + [a for a in ex.get("answer_aliases", []) if a],
                               paragraphs=paragraphs, gold_titles=gold,
                               meta={"hops": hops(ex["id"])}))
    return out

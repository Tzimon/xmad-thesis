"""
xmad/4_iteration/xmad_bench/base.py

One question of any benchmark, in the form the debate harness needs. The
harness (run_debate.py) only sees Example objects, never the raw benchmark
format, so a new benchmark needs an adapter but no change to the harness, the
agents or the debate (C3.3, C2.2).
"""

from dataclasses import dataclass, field


@dataclass
class Example:
    id: str                          # benchmark id of the question
    question: str
    answers: list[str]               # gold answer first, then aliases (scored as max over all)
    paragraphs: list[dict]           # [{title, text}] -> load_corpus of the MCP server, unique titles
    gold_titles: set[str]            # titles of the supporting paragraphs (for evidence recall)
    meta: dict = field(default_factory=dict)   # e.g. question type or number of hops

    @property
    def answer(self) -> str:
        return self.answers[0]


def unique_titles(pairs):
    """[(title, text)] -> [{title, text}] with unique titles.

    Some benchmarks (MuSiQue, 2Wiki) contain several paragraphs with the same
    title. Retrieval does not care, but evidence overlap (Jaccard) and evidence
    recall are computed on titles, so two different paragraphs must not count as
    the same one. Repeats get a suffix: "Title", "Title (2)", "Title (3)" ...
    Returns the paragraph list and, per original title, the list of unique titles.
    """
    seen, out, mapping = {}, [], {}
    for title, text in pairs:
        title = (title or "").strip()
        seen[title] = seen.get(title, 0) + 1
        t = title if seen[title] == 1 else f"{title} ({seen[title]})"
        out.append({"title": t, "text": (text or "").strip()})
        mapping.setdefault(title, []).append(t)
    return out, mapping

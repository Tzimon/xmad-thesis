"""
xmad/1_iteration/retriever.py

Phase 1 -- Building block 1: the local retriever (the "R" in the RAG chain).

This module is transport- and model-independent. It is imported by:
  * test_retriever.py -- the retrieval sanity check (this building block)
  * the MCP server    -- building block 2, which exposes search_knowledge_base()

Distractor setting: the knowledge base is *per question*. Each HotpotQA-distractor
question ships 10 paragraphs (2 gold + 8 distractors). KnowledgeBase.load() sets the
active corpus for one question; KnowledgeBase.search() returns the top-k paragraphs
for a query. That is exactly what the MCP tool wraps later, and it is where the
per-question corpus gets set before an agent runs.

Only external dependency: `datasets` (used by the helpers/test; pulled over the HF
CDN). BM25 is implemented inline so nothing else needs offline installation.
"""

import math
import re
from collections import Counter


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Minimal Okapi BM25. The corpus is tiny (10 paragraphs), so rebuilding it
    per question is cheap."""

    def __init__(self, tokenized_docs, k1=1.5, b=0.75):
        self.docs = tokenized_docs
        self.k1, self.b = k1, b
        self.N = len(tokenized_docs)
        self.avgdl = sum(len(d) for d in tokenized_docs) / max(self.N, 1)
        df = Counter()
        for d in tokenized_docs:
            for term in set(d):
                df[term] += 1
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }
        self.tf = [Counter(d) for d in tokenized_docs]

    def scores(self, query_tokens):
        out = []
        for i in range(self.N):
            dl = len(self.docs[i])
            s = 0.0
            for t in query_tokens:
                f = self.tf[i].get(t, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                    s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / denom
            out.append(s)
        return out


class KnowledgeBase:
    """Per-question knowledge base.

    Usage (also the exact shape the MCP server + eval harness will use):
        kb = KnowledgeBase()
        kb.load(paragraphs)            # paragraphs = [{"title": ..., "text": ...}, ...]
        hits = kb.search("some query", k=3)
    """

    def __init__(self):
        self._corpus = []
        self._bm25 = None

    def load(self, paragraphs):
        """Set the active corpus for one question.

        The title is indexed alongside the body. In HotpotQA the gold-paragraph
        titles are almost always the named entities that appear verbatim in the
        question, so making the title searchable is a strong and fair signal --
        every paragraph (gold and distractor alike) gets the same treatment. It
        also mirrors real retrieval, where document titles are indexed."""
        self._corpus = paragraphs
        self._bm25 = BM25(
            [tokenize(p.get("title", "") + " " + p["text"]) for p in paragraphs]
        )

    def search(self, query, k=3):
        """Return the top-k paragraphs for a query, highest BM25 score first."""
        if self._bm25 is None:
            raise RuntimeError("KnowledgeBase.load() must be called before search().")
        scores = self._bm25.scores(tokenize(query))
        ranked = sorted(zip(scores, self._corpus), key=lambda x: x[0], reverse=True)
        return [
            {"title": p["title"], "text": p["text"], "score": round(float(s), 4)}
            for s, p in ranked[:k]
        ]


# --- HotpotQA-distractor helpers (used by the test and later the eval harness) ---
def build_corpus(example):
    """Turn one example's 10 context entries into paragraph dicts."""
    titles = example["context"]["title"]
    sentence_lists = example["context"]["sentences"]
    return [
        {"title": t, "text": " ".join(sents).strip()}
        for t, sents in zip(titles, sentence_lists)
    ]


def gold_titles(example):
    """Titles of the paragraphs that support the answer."""
    return set(example["supporting_facts"]["title"])

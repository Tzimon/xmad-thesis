"""
xmad/1_iteration/retriever_dense.py

Dense drop-in for the retriever. Same load()/search() interface as
retriever.KnowledgeBase, so nothing downstream (MCP tool, agent, eval) changes.

Why dense: BM25 is purely lexical. On HotpotQA it plateaus -- it cannot
disambiguate "Ed Wood" (person) from "Ed Wood (film)", and comparison questions
share almost no vocabulary with the passages beyond the entity names. A sentence
embedding model matches on meaning and fixes both.

Dependency note: this uses `transformers` + `torch`, which are ALREADY in the
vLLM environment -- no sentence-transformers is installed, so nothing upgrades
`transformers` and breaks the vLLM tokenizer pin. The embedding model (bge, a
BERT-architecture encoder) loads on any recent `transformers` version.

Model: BAAI/bge-base-en-v1.5 -- CLS pooling, cosine similarity on L2-normalized
vectors, with the query instruction recommended for bge retrieval. Swap to
BAAI/bge-small-en-v1.5 for a lighter/faster variant.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# bge-v1.5 retrieval convention: instruction on the QUERY only; passages as-is.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class DenseKnowledgeBase:
    """Per-question knowledge base backed by dense embeddings.

    Interface is identical to retriever.KnowledgeBase:
        kb = DenseKnowledgeBase()      # loads the embedding model once
        kb.load(paragraphs)            # embeds this question's corpus
        hits = kb.search("query", k=3)

    Instantiate ONCE and reuse across questions -- the model load is the
    expensive part; load()/search() per question are cheap (10 short passages).
    """

    def __init__(self, model_name="BAAI/bge-base-en-v1.5", device=None, max_length=512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.max_length = max_length
        self._corpus = []
        self._emb = None  # (N, d) L2-normalized passage embeddings

    @torch.no_grad()
    def _embed(self, texts):
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        out = self.model(**batch)
        emb = out.last_hidden_state[:, 0]      # CLS pooling (bge)
        emb = F.normalize(emb, p=2, dim=1)     # so a dot product == cosine
        return emb

    def load(self, paragraphs):
        """Embed and store this question's corpus."""
        self._corpus = paragraphs
        texts = [f"{p.get('title', '')} {p['text']}".strip() for p in paragraphs]
        self._emb = self._embed(texts)

    def search(self, query, k=3):
        """Return the top-k passages by cosine similarity, highest first."""
        if self._emb is None:
            raise RuntimeError("DenseKnowledgeBase.load() must be called before search().")
        q = self._embed([QUERY_INSTRUCTION + query])   # (1, d)
        scores = (self._emb @ q.T).squeeze(1)          # (N,)
        k = min(k, len(self._corpus))
        top = torch.topk(scores, k=k)
        results = []
        for score, idx in zip(top.values.tolist(), top.indices.tolist()):
            p = self._corpus[idx]
            results.append(
                {"title": p["title"], "text": p["text"], "score": round(float(score), 4)}
            )
        return results

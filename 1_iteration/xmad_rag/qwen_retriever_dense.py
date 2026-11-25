"""
xmad/1_iteration/retriever_dense.py

Dense drop-in for the retriever. Same load()/search() interface as
retriever.KnowledgeBase, so nothing downstream (MCP tool, agent, eval) changes.

Model: Qwen/Qwen3-Embedding-0.6B -- selected on MTEB as the final retriever
(highest retrieval score under the size limit among models that also keep a high
zero-shot score; Apache 2.0; same family as the served Qwen2.5-7B-Instruct).

Qwen3-Embedding differs from bge in two ways that MUST be handled correctly:
  1. Pooling: it is a decoder model and uses LAST-TOKEN pooling, not CLS. The
     tokenizer therefore pads on the LEFT, so the final position is a real token.
  2. Query format: queries are wrapped in an instruction ("Instruct: ...\nQuery:");
     passages get NO instruction.

Dependency note: uses `transformers` + `torch` only, already present in the vLLM
environment (Qwen3 support requires a recent transformers, which the vLLM 0.11
stack already has). No sentence-transformers, so nothing upgrades the pinned
transformers. (sentence-transformers would handle pooling/prompt automatically but
must live in its own venv -- see the note at the bottom of the module.)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Qwen3-Embedding retrieval convention: instruct wraps the QUERY only.
QUERY_TASK = "Given a web search query, retrieve relevant passages that answer the query"


def _format_query(query):
    return f"Instruct: {QUERY_TASK}\nQuery:{query}"


def _last_token_pool(last_hidden_states, attention_mask):
    """Take the hidden state of the last real (non-padding) token per sequence."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    seq_len = attention_mask.sum(dim=1) - 1
    bs = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(bs, device=last_hidden_states.device), seq_len]


class DenseKnowledgeBase:
    """Per-question knowledge base backed by dense embeddings.

    Interface is identical to retriever.KnowledgeBase:
        kb = DenseKnowledgeBase()      # loads the embedding model once
        kb.load(paragraphs)            # embeds this question's corpus
        hits = kb.search("query", k=3)

    Instantiate ONCE and reuse across questions -- the model load is the expensive
    part; load()/search() per question are cheap (10 short passages).
    """

    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B", device=None, max_length=512):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # left padding is required for correct last-token pooling
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
            .to(self.device)
            .eval()
        )
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
        emb = _last_token_pool(out.last_hidden_state, batch["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)     # so a dot product == cosine
        return emb

    def load(self, paragraphs):
        """Embed and store this question's corpus (passages get no instruction)."""
        self._corpus = paragraphs
        texts = [f"{p.get('title', '')} {p['text']}".strip() for p in paragraphs]
        self._emb = self._embed(texts)

    def search(self, query, k=3):
        """Return the top-k passages by cosine similarity, highest first."""
        if self._emb is None:
            raise RuntimeError("DenseKnowledgeBase.load() must be called before search().")
        q = self._embed([_format_query(query)])        # (1, d)
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


# ---------------------------------------------------------------------------
# Alternative: sentence-transformers (handles pooling + prompt automatically).
# Cleaner, but pulls its own transformers -> keep it in a SEPARATE venv from vLLM.
#
#   from sentence_transformers import SentenceTransformer
#   m = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
#   doc_emb   = m.encode(passages, normalize_embeddings=True)
#   query_emb = m.encode([query], prompt_name="query", normalize_embeddings=True)
#
# Both paths produce the same embeddings; use whichever fits the environment.
# ---------------------------------------------------------------------------
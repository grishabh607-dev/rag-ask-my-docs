"""
bm25_store.py — BM25 sparse keyword retrieval.

BM25 excels at exact keyword matching — product codes, names, acronyms —
where dense vectors underperform. Hybrid = dense + sparse fusion.

Index is serialized to disk so it survives restarts without re-ingesting.
"""
import pickle
import os
from typing import List, Tuple
from ingestion.loader import Document


def tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer. Swap for NLTK if needed."""
    import re
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


class BM25Store:
    def __init__(self, index_path: str = "data/indexes/bm25.pkl"):
        self.index_path = index_path
        self.bm25 = None
        self.chunks: List[Document] = []

        if os.path.exists(index_path):
            self._load()

    def build(self, chunks: List[Document]):
        """Build BM25 index from a list of Document chunks."""
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        tokenized = [tokenize(c.page_content) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        self._save()
        print(f"  ✓ BM25 index built ({len(chunks)} docs)")

    def search(self, query: str, top_k: int = 20) -> List[Tuple[Document, float]]:
        """Return top_k chunks ranked by BM25 score."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Run ingestion first.")
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Get top_k indices sorted by score descending
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.chunks[idx], float(scores[idx])))
        return results

    def _save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)

    def _load(self):
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        print(f"  ✓ BM25 index loaded from {self.index_path}")

"""
embedder.py — Generate dense vector embeddings for chunks.

Model: all-MiniLM-L6-v2 (384-dim, fast, good quality for English docs)
Production alternative: text-embedding-3-small (OpenAI) or embed-english-v3.0 (Cohere)

Design note: Embedder is stateless and batched.
Batch size 64 balances GPU memory and throughput.
"""
from typing import List
import numpy as np


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"  ✓ Embedder loaded: {model_name} (dim={self.dimension})")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts, returning list of float vectors."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,  # cosine similarity via dot product
                show_progress_bar=False,
            )
            all_embeddings.extend(embeddings.tolist())
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        return self.embed([query])[0]

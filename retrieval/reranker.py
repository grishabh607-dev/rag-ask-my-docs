"""
reranker.py — Cross-encoder reranking as the final retrieval stage.

Why rerank?
Bi-encoders (embedding models) trade accuracy for speed — they embed query
and doc independently. Cross-encoders see the (query, doc) pair together
and score relevance much more accurately. Too slow to run on the whole corpus,
but perfect as a post-filtering step on the top-20 hybrid candidates.

Two backends:
  - Cohere Rerank API (production, pay-per-use)
  - Local cross-encoder model (free, slightly slower)
"""
from typing import List, Tuple
from ingestion.loader import Document
import os


class CohereReranker:
    def __init__(self, model: str = "rerank-english-v3.0", top_n: int = 5):
        import cohere
        self.client = cohere.Client(os.environ["COHERE_API_KEY"])
        self.model = model
        self.top_n = top_n

    def rerank(self, query: str, candidates: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        docs_text = [doc.page_content for doc, _ in candidates]
        response = self.client.rerank(
            query=query,
            documents=docs_text,
            model=self.model,
            top_n=self.top_n,
        )
        reranked = []
        for result in response.results:
            doc, _ = candidates[result.index]
            reranked.append((doc, result.relevance_score))
        return reranked


class LocalReranker:
    """Uses a local cross-encoder model. No API key needed."""
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_n: int = 5):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model)
        self.top_n = top_n

    def rerank(self, query: str, candidates: List[Tuple[Document, float]]) -> List[Tuple[Document, float]]:
        pairs = [(query, doc.page_content) for doc, _ in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [(doc, float(score)) for (doc, _), score in ranked[:self.top_n]]


def get_reranker(backend: str = "local", top_n: int = 5):
    """Factory. Use 'cohere' in production, 'local' for free dev usage."""
    if backend == "cohere":
        return CohereReranker(top_n=top_n)
    return LocalReranker(top_n=top_n)

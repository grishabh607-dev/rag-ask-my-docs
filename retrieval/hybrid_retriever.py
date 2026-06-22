"""
hybrid_retriever.py — Fuse dense + sparse results using Reciprocal Rank Fusion (RRF).

RRF Formula: score(d) = Σ 1/(k + rank(d))
k=60 is standard; dampens the effect of very high rankings.

Why RRF over score normalization?
- BM25 and cosine scores are on different scales and can't be directly summed.
- RRF is parameter-light, robust, and empirically matches more complex fusion methods.
"""
from typing import List, Tuple, Dict
from ingestion.loader import Document
from retrieval.vector_store import VectorStore
from retrieval.bm25_store import BM25Store
from ingestion.embedder import Embedder


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[Document, float]]],
    k: int = 60,
) -> List[Tuple[Document, float]]:
    """
    Fuse multiple ranked lists of (Document, score) via RRF.
    Documents are identified by their source+chunk_index metadata.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, (doc, _) in enumerate(ranked_list):
            doc_id = f"{doc.metadata.get('source', '')}::{doc.metadata.get('chunk_index', 0)}"
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[doc_id], score) for doc_id, score in fused]


class HybridRetriever:
    def __init__(
        self,
        collection_name: str = "ask_my_docs",
        bm25_index_path: str = "data/indexes/ask_my_docs_bm25.pkl",
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
        final_top_k: int = 10,
    ):
        self.embedder = Embedder()
        self.vector_store = VectorStore(collection_name=collection_name)
        self.bm25_store = BM25Store(index_path=bm25_index_path)
        self.vector_top_k = vector_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """
        Retrieve top candidates via hybrid search.
        Returns final_top_k results sorted by RRF score.
        These are passed to the reranker for a final pass.
        """
        # Dense retrieval
        query_vec = self.embedder.embed_query(query)
        dense_results = self.vector_store.search(query_vec, top_k=self.vector_top_k)

        # Sparse retrieval
        sparse_results = self.bm25_store.search(query, top_k=self.bm25_top_k)

        # Fuse
        fused = reciprocal_rank_fusion([dense_results, sparse_results])
        return fused[:self.final_top_k]

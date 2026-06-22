"""
vector_store.py — Qdrant-backed dense vector retrieval.

Qdrant runs locally via Docker (see docker-compose.yml).
Production: swap URL to Qdrant Cloud, add API key.

Uses cosine similarity (COSINE distance on normalized vectors).
"""
from typing import List, Tuple
from ingestion.loader import Document


class VectorStore:
    def __init__(
        self,
        collection_name: str = "ask_my_docs",
        dimension: int = 384,
        url: str = "http://localhost:6333",
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.dimension = dimension

        # Create collection if not exists
        existing = [c.name for c in self.client.get_collections().collections]
        if collection_name not in existing:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )
            print(f"  ✓ Created Qdrant collection: {collection_name}")

    def upsert(self, chunks: List[Document], embeddings: List[List[float]]):
        from qdrant_client.models import PointStruct
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            points.append(PointStruct(
                id=i,
                vector=embedding,
                payload={
                    "text": chunk.page_content,
                    **chunk.metadata,
                }
            ))
        # Batch upsert in groups of 100
        for i in range(0, len(points), 100):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i:i + 100],
            )
        print(f"  ✓ Upserted {len(points)} points into Qdrant")

    def search(self, query_vector: List[float], top_k: int = 20) -> List[Tuple[Document, float]]:
        """Return top_k chunks with their similarity scores."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        docs_with_scores = []
        for r in results:
            doc = Document(
                page_content=r.payload.pop("text"),
                metadata=r.payload,
            )
            docs_with_scores.append((doc, r.score))
        return docs_with_scores

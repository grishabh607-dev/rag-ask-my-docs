"""
pipeline.py — Orchestrates the full ingestion flow.

Flow: Load → Chunk → Embed → Store (Vector + BM25)

This is the entry point called by scripts/ingest.py.
Idempotent: re-running won't duplicate chunks (uses upsert).
"""
import time
from typing import Optional
from ingestion.loader import load_directory
from ingestion.chunker import chunk_documents
from ingestion.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.bm25_store import BM25Store


def run_ingestion(
    docs_dir: str,
    collection_name: str = "ask_my_docs",
    chunk_size: int = 512,
    overlap: int = 64,
):
    start = time.time()
    print(f"\n{'='*60}")
    print(f"  RAG Ingestion Pipeline")
    print(f"{'='*60}")

    # Step 1: Load
    print(f"\n[1/4] Loading documents from: {docs_dir}")
    raw_docs = load_directory(docs_dir)
    print(f"  → {len(raw_docs)} raw segments loaded")

    # Step 2: Chunk
    print(f"\n[2/4] Chunking (size={chunk_size}, overlap={overlap})")
    chunks = chunk_documents(raw_docs, chunk_size=chunk_size, overlap=overlap)
    print(f"  → {len(chunks)} chunks created")

    # Step 3: Embed
    print(f"\n[3/4] Generating embeddings")
    embedder = Embedder()
    texts = [c.page_content for c in chunks]
    embeddings = embedder.embed(texts)
    print(f"  → {len(embeddings)} embeddings generated (dim={len(embeddings[0])})")

    # Step 4: Store
    print(f"\n[4/4] Storing in vector store + BM25 index")
    vector_store = VectorStore(collection_name=collection_name, dimension=len(embeddings[0]))
    vector_store.upsert(chunks, embeddings)

    bm25_store = BM25Store(index_path=f"data/indexes/{collection_name}_bm25.pkl")
    bm25_store.build(chunks)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  ✓ Ingestion complete in {elapsed:.1f}s")
    print(f"  Chunks stored: {len(chunks)}")
    print(f"  Collection: {collection_name}")
    print(f"{'='*60}\n")

    return {"chunks": len(chunks), "elapsed_seconds": elapsed}

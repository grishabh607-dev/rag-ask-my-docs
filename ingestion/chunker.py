"""
chunker.py — Split documents into retrieval-optimized chunks.

Strategy: Recursive character splitting with overlap.
chunk_size=512 tokens is the sweet spot for most RAG use cases.
Overlap=64 ensures context doesn't get cut at boundaries.

Key insight: chunk size affects both retrieval precision AND generation quality.
Too small → context fragments. Too large → diluted relevance scores.
"""
import re
from typing import List
from ingestion.loader import Document


def count_tokens(text: str) -> int:
    """Approximate token count without loading tiktoken (4 chars ≈ 1 token)."""
    return len(text) // 4


def recursive_split(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Split text using a hierarchy of separators:
    paragraphs → sentences → words → characters.
    Falls back to coarser splits only when needed.
    """
    separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(text: str, separators: List[str]) -> List[str]:
        if not separators:
            # Character-level split as last resort
            chunks = []
            for i in range(0, len(text), chunk_size * 4):
                chunks.append(text[i:i + chunk_size * 4])
            return chunks

        sep = separators[0]
        splits = text.split(sep) if sep else list(text)

        chunks = []
        current = ""
        for split in splits:
            candidate = (current + sep + split).strip() if current else split.strip()
            if count_tokens(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # If a single split is still too big, recurse with next separator
                if count_tokens(split) > chunk_size:
                    chunks.extend(_split(split, separators[1:]))
                    current = ""
                else:
                    current = split
        if current:
            chunks.append(current)
        return chunks

    raw_chunks = _split(text, separators)

    # Apply overlap: prepend tail of previous chunk
    final_chunks = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            final_chunks.append(chunk)
        else:
            prev_tail = raw_chunks[i - 1].split()[-overlap // 4:]  # rough token overlap
            overlap_text = " ".join(prev_tail)
            final_chunks.append((overlap_text + " " + chunk).strip())

    return [c for c in final_chunks if c.strip()]


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[Document]:
    """
    Chunk a list of Documents. Preserves and enriches metadata.
    Adds chunk_index and total_chunks so citations can say "Page 3, Chunk 2 of 5".
    """
    chunked = []
    for doc in documents:
        chunks = recursive_split(doc.page_content, chunk_size, overlap)
        for i, chunk_text in enumerate(chunks):
            chunked.append(Document(
                page_content=chunk_text,
                metadata={
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "token_count": count_tokens(chunk_text),
                }
            ))
    return chunked

"""
citations.py — Parse and validate citations from LLM output.

After generation, we verify:
1. All [N] citations in the answer map to a valid source index.
2. No fabricated sources (citation index out of range).
3. Extract structured citation list for the API response.

This is the "citation enforcement" piece of the project.
"""
import re
from typing import List, Dict, Tuple
from ingestion.loader import Document


def extract_citations(text: str) -> List[int]:
    """Extract all [N] citation indices from generated text."""
    pattern = r'\[(\d+)\]'
    matches = re.findall(pattern, text)
    return list(set(int(m) for m in matches))


def validate_citations(
    answer: str,
    sources: List[Tuple[Document, float]],
) -> Tuple[bool, List[int]]:
    """
    Check that all [N] citations refer to valid source indices.
    Returns (is_valid, list_of_invalid_indices).
    """
    cited_indices = extract_citations(answer)
    max_valid = len(sources)
    invalid = [i for i in cited_indices if i < 1 or i > max_valid]
    return len(invalid) == 0, invalid


def build_citation_objects(
    answer: str,
    sources: List[Tuple[Document, float]],
) -> List[Dict]:
    """
    Build structured citation objects for the API response.
    Only includes sources that were actually cited in the answer.
    """
    cited_indices = extract_citations(answer)
    citations = []
    for idx in sorted(cited_indices):
        if 1 <= idx <= len(sources):
            doc, score = sources[idx - 1]
            citations.append({
                "index": idx,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page"),
                "chunk_index": doc.metadata.get("chunk_index"),
                "relevance_score": round(score, 3),
                "excerpt": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            })
    return citations

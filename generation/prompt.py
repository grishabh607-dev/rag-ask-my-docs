"""
prompt.py — Prompt templates for the RAG chain.

Key design decisions:
1. System prompt enforces citation behavior — model MUST cite [1], [2], etc.
2. "Only answer from context" instruction reduces hallucination.
3. "If you don't know, say so" handles out-of-domain queries cleanly.
4. Context is injected with numbered source blocks for citation tracking.
"""

SYSTEM_PROMPT = """You are a precise, helpful assistant that answers questions strictly based on the provided context documents.

RULES:
1. Answer ONLY using information from the numbered [Source N] blocks below.
2. Every factual claim MUST include an inline citation like [1] or [2].
3. If the context does not contain enough information to answer, say: "I don't have enough information in the provided documents to answer this question."
4. Do NOT make up facts, infer beyond what's stated, or use prior knowledge.
5. Be concise and direct. Use bullet points for lists.
6. At the end of your answer, include a "Sources" section listing the cited sources.

FORMAT:
Answer: <your answer with inline [N] citations>

Sources:
- [1] <filename>, page <page>
- [2] <filename>, chunk <chunk_index>
"""


def build_context_block(chunks_with_scores) -> str:
    """Format retrieved chunks into numbered source blocks."""
    lines = []
    for i, (doc, score) in enumerate(chunks_with_scores, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        chunk_idx = doc.metadata.get("chunk_index", "")
        source_label = f"{source}"
        if page:
            source_label += f", page {page}"
        if chunk_idx != "":
            source_label += f", chunk {chunk_idx}"

        lines.append(f"[Source {i}] ({source_label}, relevance={score:.2f})")
        lines.append(doc.page_content.strip())
        lines.append("")  # blank line between sources
    return "\n".join(lines)


def build_user_message(query: str, context: str) -> str:
    return f"""Context Documents:
{context}

Question: {query}"""

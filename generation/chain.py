"""
chain.py — The full RAG chain: Retrieve → Rerank → Generate → Validate.

This is the main entry point for answering a question.
Returns a structured RAGResponse with answer, citations, sources, and metadata.
"""
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterator
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import get_reranker
from generation.prompt import SYSTEM_PROMPT, build_context_block, build_user_message
from generation.llm import ClaudeLLM
from generation.citations import validate_citations, build_citation_objects


@dataclass
class RAGResponse:
    answer: str
    citations: List[Dict]
    sources_used: int
    citations_valid: bool
    latency_ms: Dict[str, float]
    token_usage: Dict[str, int]
    query: str


class RAGChain:
    def __init__(
        self,
        collection_name: str = "ask_my_docs",
        reranker_backend: str = "local",  # "local" or "cohere"
        top_k_retrieve: int = 10,
        top_k_rerank: int = 5,
    ):
        self.retriever = HybridRetriever(
            collection_name=collection_name,
            final_top_k=top_k_retrieve,
        )
        self.reranker = get_reranker(backend=reranker_backend, top_n=top_k_rerank)
        self.llm = ClaudeLLM()

    def query(self, question: str) -> RAGResponse:
        """Full RAG pipeline. Blocking (non-streaming)."""
        timings = {}

        # Step 1: Retrieve
        t0 = time.time()
        candidates = self.retriever.retrieve(question)
        timings["retrieval_ms"] = round((time.time() - t0) * 1000, 1)

        # Step 2: Rerank
        t1 = time.time()
        reranked = self.reranker.rerank(question, candidates)
        timings["reranking_ms"] = round((time.time() - t1) * 1000, 1)

        # Step 3: Build prompt
        context = build_context_block(reranked)
        user_msg = build_user_message(question, context)

        # Step 4: Generate
        t2 = time.time()
        result = self.llm.generate(system=SYSTEM_PROMPT, user=user_msg)
        timings["generation_ms"] = round((time.time() - t2) * 1000, 1)
        timings["total_ms"] = round((time.time() - t0) * 1000, 1)

        answer = result["text"]

        # Step 5: Validate citations
        is_valid, invalid = validate_citations(answer, reranked)
        citations = build_citation_objects(answer, reranked)

        return RAGResponse(
            answer=answer,
            citations=citations,
            sources_used=len(reranked),
            citations_valid=is_valid,
            latency_ms=timings,
            token_usage={
                "input": result["input_tokens"],
                "output": result["output_tokens"],
            },
            query=question,
        )

    def stream(self, question: str) -> Iterator[str]:
        """Streaming version for the UI. Yields text chunks."""
        candidates = self.retriever.retrieve(question)
        reranked = self.reranker.rerank(question, candidates)
        context = build_context_block(reranked)
        user_msg = build_user_message(question, context)
        yield from self.llm.stream(system=SYSTEM_PROMPT, user=user_msg)

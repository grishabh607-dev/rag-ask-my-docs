"""
models.py — Pydantic request/response models for the FastAPI layer.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="The question to answer")
    collection: str = Field(default="ask_my_docs", description="Document collection to query")
    reranker: str = Field(default="local", description="Reranker backend: 'local' or 'cohere'")


class CitationObject(BaseModel):
    index: int
    source: str
    page: Optional[int] = None
    chunk_index: Optional[int] = None
    relevance_score: float
    excerpt: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationObject]
    sources_used: int
    citations_valid: bool
    latency_ms: Dict[str, float]
    token_usage: Dict[str, int]
    query: str


class IngestRequest(BaseModel):
    docs_dir: str = Field(..., description="Path to documents directory")
    collection: str = Field(default="ask_my_docs")
    chunk_size: int = Field(default=512, ge=128, le=2048)
    overlap: int = Field(default=64, ge=0, le=256)


class IngestResponse(BaseModel):
    chunks: int
    elapsed_seconds: float
    collection: str


class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"

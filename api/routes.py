"""
routes.py — FastAPI route handlers.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from api.models import (
    QueryRequest, QueryResponse, CitationObject,
    IngestRequest, IngestResponse, HealthResponse
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Answer a question using the RAG pipeline."""
    try:
        from generation.chain import RAGChain
        chain = RAGChain(
            collection_name=request.collection,
            reranker_backend=request.reranker,
        )
        result = chain.query(request.question)
        return QueryResponse(
            answer=result.answer,
            citations=[CitationObject(**c) for c in result.citations],
            sources_used=result.sources_used,
            citations_valid=result.citations_valid,
            latency_ms=result.latency_ms,
            token_usage=result.token_usage,
            query=result.query,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Streaming answer endpoint for the UI."""
    from generation.chain import RAGChain
    chain = RAGChain(collection_name=request.collection)

    def generate():
        for chunk in chain.stream(request.question):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """Trigger document ingestion."""
    try:
        from ingestion.pipeline import run_ingestion
        result = run_ingestion(
            docs_dir=request.docs_dir,
            collection_name=request.collection,
            chunk_size=request.chunk_size,
            overlap=request.overlap,
        )
        return IngestResponse(**result, collection=request.collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

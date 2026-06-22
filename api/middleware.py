"""
middleware.py — Request logging, latency tracking, error handling.
This is the foundation for Project 3 (Observability).
"""
import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("rag")


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        latency_ms = round((time.time() - start) * 1000, 1)

        logger.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} "
            f"[{latency_ms}ms]"
        )
        response.headers["X-Latency-Ms"] = str(latency_ms)
        return response

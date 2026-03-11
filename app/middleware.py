"""
app/middleware.py
─────────────────
Two middleware layers:

1. RequestLoggingMiddleware
   - Attaches a unique X-Request-ID to every request
   - Logs method, path, status code, and latency on every response
   - Structured output so logs are grep/query friendly

2. CORS is registered in main.py via FastAPI's built-in CORSMiddleware
   (it needs to be the outermost layer, so it lives there)
"""
from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("churn_api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    For every request:
      - Generate a request_id if the client didn't send X-Request-ID
      - Attach it to the response so clients can correlate logs
      - Log: method | path | status | latency | request_id
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())[:8]

        # Make request_id available to route handlers via request.state
        request.state.request_id = request_id

        start = time.perf_counter()
        response: Response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        # Attach request_id to every response header for client-side tracing
        response.headers["x-request-id"] = request_id

        logger.info(
            "request completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": latency_ms,
            },
        )

        return response
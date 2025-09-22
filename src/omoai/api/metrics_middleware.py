from __future__ import annotations

import logging
import time
from typing import Any

from litestar import get
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_METRICS: dict[str, Any] = {
    "request_total": 0,
    "request_latency_sum": 0.0,
}


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:  # type: ignore[override]
        start = time.perf_counter()
        try:
            response = await call_next(request)
            return response
        finally:
            elapsed = time.perf_counter() - start
            try:
                _METRICS["request_total"] += 1
                _METRICS["request_latency_sum"] += float(elapsed)
            except (TypeError, ValueError) as err:
                logging.getLogger(__name__).debug("Failed to update metrics", exc_info=err)


@get(path="/metrics")
async def metrics_endpoint() -> Response:
    """Return very basic Prometheus-like metrics text."""
    try:
        total = int(_METRICS.get("request_total", 0))
        latency_sum = float(_METRICS.get("request_latency_sum", 0.0))
    except (TypeError, ValueError):
        total = 0
        latency_sum = 0.0
    lines = [
        "# HELP request_total Total HTTP requests",
        "# TYPE request_total counter",
        f"request_total {total}",
        "# HELP request_latency_seconds_sum Cumulative request latency",
        "# TYPE request_latency_seconds_sum counter",
        f"request_latency_seconds_sum {latency_sum:.6f}",
    ]
    body = "\n".join(lines) + "\n"
    return Response(body, media_type="text/plain; charset=utf-8")

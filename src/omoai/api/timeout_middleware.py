from __future__ import annotations

import asyncio

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class RequestTimeoutMiddleware(BaseHTTPMiddleware):
    """Simple request timeout middleware using asyncio.wait_for.

    Applies only to HTTP requests. WebSockets and other scopes are passed through.
    """

    def __init__(self, app, timeout_seconds: float) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.timeout_seconds = float(timeout_seconds)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if self.timeout_seconds and self.timeout_seconds > 0:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout_seconds)
        return await call_next(request)


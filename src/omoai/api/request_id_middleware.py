from __future__ import annotations

import logging
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a request ID to request.state and response headers.

    - Uses incoming 'X-Request-ID' header if present; else generates a UUID4.
    - Exposes the value via 'X-Request-ID' response header and request.state.request_id.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:  # type: ignore[override]
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        # Attach to request state for logging / error envelopes
        request.state.request_id = request_id
        response = await call_next(request)
        try:
            response.headers["X-Request-ID"] = request_id
        except (TypeError, ValueError) as err:
            logging.getLogger(__name__).debug("Failed to set X-Request-ID header", exc_info=err)
        return response


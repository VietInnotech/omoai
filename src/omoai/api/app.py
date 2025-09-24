import logging
import os

import uvicorn
from litestar import Litestar, Router, get
from litestar.exceptions import HTTPException
from litestar.middleware import DefineMiddleware
from litestar.response import Redirect, Response

from omoai.api.asr_controller import ASRController
from omoai.api.exceptions import AudioProcessingException
from omoai.api.health import health_check
from omoai.api.jobs import JobsController
from omoai.api.main_controller import MainController
from omoai.api.metrics_middleware import MetricsMiddleware, metrics_endpoint
from omoai.api.postprocess_controller import PostprocessController
from omoai.api.preprocess_controller import PreprocessController
from omoai.api.request_id_middleware import RequestIDMiddleware
from omoai.api.timeout_middleware import RequestTimeoutMiddleware
from omoai.config.schemas import get_config
from omoai.logging_system.logger import setup_logging


def global_exception_handler(request, exc):
    """Global exception handler for all uncaught exceptions with structured logs."""
    logger = logging.getLogger(__name__)

    # Basic request context (best-effort)
    req_path = str(
        getattr(getattr(request, "url", None), "path", getattr(request, "url", ""))
    )
    method = getattr(request, "method", "")

    # Request/trace ID from middleware
    trace_id = getattr(getattr(request, "state", None), "request_id", None) or ""

    def envelope(code: str, message: str, status_code: int, details: dict | None = None):
        payload = {
            "code": code,
            "message": message,
            "trace_id": trace_id,
        }
        if details:
            payload["details"] = details
        return Response(content=payload, status_code=status_code)

    if isinstance(exc, AudioProcessingException):
        logger.error(
            "Audio processing error",
            exc_info=exc,
            extra={
                "type": "AudioProcessingException",
                "status_code": getattr(exc, "status_code", 500),
                "detail": str(exc),
                "path": req_path,
                "method": method,
                "trace_id": trace_id,
            },
        )
        return envelope(
            code="audio_processing_error",
            message=str(exc),
            status_code=getattr(exc, "status_code", 500),
            details={"path": req_path, "method": method},
        )
    elif isinstance(exc, HTTPException):
        logger.error(
            "HTTP error",
            exc_info=exc,
            extra={
                "type": "HTTPException",
                "status_code": getattr(exc, "status_code", 500),
                "detail": str(exc),
                "path": req_path,
                "method": method,
                "trace_id": trace_id,
            },
        )
        return envelope(
            code="http_error",
            message=str(exc),
            status_code=getattr(exc, "status_code", 500),
            details={"path": req_path, "method": method},
        )
    else:
        logger.exception(
            "Unhandled exception",
            extra={
                "type": type(exc).__name__,
                "path": req_path,
                "method": method,
                "trace_id": trace_id,
            },
        )
        return envelope(
            code="internal_error",
            message="Internal server error",
            status_code=500,
            details={"path": req_path, "method": method},
        )


@get(path="/")
def root_redirect() -> Redirect:
    """Root endpoint that redirects to the OpenAPI schema documentation."""
    return Redirect(path="/schema")


def create_app() -> Litestar:
    """Create the Litestar application with configuration-based settings."""
    config = get_config()
    # Configure the unified logging system once, using config.yaml
    setup_logging()

    # Configure request timeout middleware from config
    timeout_seconds = float(config.api.request_timeout_seconds)

    api_router = Router(
        path="/v1",
        route_handlers=[
            MainController,
            PreprocessController,
            ASRController,
            PostprocessController,
            JobsController,
            health_check,
            metrics_endpoint,
        ],
    )

    return Litestar(
        route_handlers=[root_redirect, api_router],
        on_startup=[],
        # Use unified stdlib logging configured by setup_logging()
        logging_config=None,
        middleware=[
            DefineMiddleware(RequestIDMiddleware),
            DefineMiddleware(MetricsMiddleware),
            DefineMiddleware(RequestTimeoutMiddleware, timeout_seconds=timeout_seconds),
        ],
        request_max_body_size=config.api.max_body_size_mb
        * 1024
        * 1024,  # Convert MB to bytes,
        exception_handlers={
            Exception: global_exception_handler,
            AudioProcessingException: global_exception_handler,
            HTTPException: global_exception_handler,
        },
    )


def main():
    # Configure logging before starting uvicorn to ensure consistent behavior
    setup_logging()

    # Get configuration for server settings
    config = get_config()

    # Determine log level based on config
    log_level = "info"
    if config.logging and config.logging.debug_mode:
        log_level = "debug"
    elif config.logging and config.logging.level:
        log_level = config.logging.level.lower()

    uvicorn.run(
        "omoai.api.app:create_app",
        factory=True,
        host=config.api.host,
        port=config.api.port,
        reload=True,
        reload_dirs=["src", "scripts"],
        reload_excludes=[".venv/*"],
        workers=int(os.environ.get("UVICORN_WORKERS", 1)),
        log_config=None,  # Disable uvicorn's logging config, let Loguru handle it
        log_level=log_level,
    )


if __name__ == "__main__":
    main()

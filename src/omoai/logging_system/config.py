"""
Logging configuration for OMOAI structured logging.
"""

import logging
import os
import sys
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from ..config import get_config


class LoggingConfig(BaseModel):
    """Structured logging configuration."""

    # Logging levels
    level: str = Field(default="INFO", description="Root logging level")

    # Output configuration
    format_type: str = Field(
        default="structured",
        description="Logging format: 'structured', 'json', or 'simple'",
    )
    enable_console: bool = Field(default=True, description="Enable console output")
    enable_file: bool = Field(default=False, description="Enable file output")
    # Optional human-readable file output (non-JSON)
    enable_text_file: bool = Field(
        default=False, description="Enable text file output (human-readable)"
    )
    text_log_file: Path | None = Field(
        default=None, description="Text log file path (non-JSON)"
    )

    # File logging
    log_file: Path | None = Field(default=None, description="Log file path")
    max_file_size: int = Field(
        default=10 * 1024 * 1024, description="Maximum log file size in bytes"
    )
    backup_count: int = Field(default=5, description="Number of backup log files")
    # Loguru-native advanced options
    rotation: str | None = Field(
        default=None, description="Rotation policy, e.g. '10 MB' or '00:00'"
    )
    retention: object | None = Field(
        default=None, description="Retention policy, e.g. '14 days' or number of files"
    )
    compression: str | None = Field(
        default=None, description="Compression for rotated files, e.g. 'gz'"
    )
    enqueue: bool = Field(
        default=True, description="Enable async logging queue for sinks"
    )

    # Performance logging
    enable_performance_logging: bool = Field(
        default=True, description="Enable performance metrics"
    )
    performance_threshold_ms: float = Field(
        default=100.0, description="Log performance if above threshold (ms)"
    )

    # Request tracing
    enable_request_tracing: bool = Field(
        default=True, description="Enable request ID tracing"
    )
    trace_headers: bool = Field(
        default=False, description="Include request headers in traces"
    )

    # Error tracking
    enable_error_tracking: bool = Field(
        default=True, description="Enable structured error tracking"
    )
    include_stacktrace: bool = Field(
        default=True, description="Include stack traces in error logs"
    )

    # Metrics
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_interval: int = Field(
        default=60, description="Metrics collection interval (seconds)"
    )

    # Environment-based overrides
    debug_mode: bool = Field(default=False, description="Enable debug mode logging")
    quiet_mode: bool = Field(default=False, description="Suppress non-error output")

    # Instance helpers
    def get_log_level(self) -> int:
        """Get numeric log level."""
        if self.debug_mode:
            return logging.DEBUG
        if self.quiet_mode:
            return logging.ERROR
        return getattr(logging, (self.level or "INFO").upper(), logging.INFO)

    def get_level_name(self) -> str:
        """Get textual level name suitable for sinks like Loguru."""
        if self.debug_mode:
            return "DEBUG"
        if self.quiet_mode:
            return "ERROR"
        return (self.level or "INFO").upper()

    def should_log_performance(self, duration_ms: float) -> bool:
        """Check if performance should be logged based on threshold."""
        try:
            thr = float(self.performance_threshold_ms)
        except Exception:
            thr = 100.0
        return bool(self.enable_performance_logging) and float(duration_ms) >= thr

    @classmethod
    def from_environment(cls) -> "LoggingConfig":
        """Create logging config from environment variables.

        Supports a simple alias for log paths: values starting with "@logs/"
        are rewritten to the local "logs/" directory.
        """
        log_file_env = os.environ.get("OMOAI_LOG_FILE")
        if log_file_env and log_file_env.startswith("@logs/"):
            # Normalize alias to local logs directory
            log_file_env = os.path.join("logs", log_file_env[len("@logs/") :])

        text_log_file_env = os.environ.get("OMOAI_LOG_TEXT_FILE")
        if text_log_file_env and text_log_file_env.startswith("@logs/"):
            text_log_file_env = os.path.join("logs", text_log_file_env[len("@logs/") :])

        return cls(
            level=os.environ.get("OMOAI_LOG_LEVEL", "INFO").upper(),
            format_type=os.environ.get("OMOAI_LOG_FORMAT", "structured"),
            enable_console=os.environ.get("OMOAI_LOG_CONSOLE", "true").lower()
            == "true",
            enable_file=os.environ.get("OMOAI_LOG_FILE_ENABLED", "false").lower()
            == "true",
            log_file=Path(log_file_env) if log_file_env else None,
            enable_text_file=os.environ.get(
                "OMOAI_LOG_TEXT_FILE_ENABLED", "false"
            ).lower()
            == "true",
            text_log_file=Path(text_log_file_env) if text_log_file_env else None,
            rotation=os.environ.get("OMOAI_LOG_ROTATION") or None,
            retention=(
                int(os.environ["OMOAI_LOG_RETENTION"])
                if os.environ.get("OMOAI_LOG_RETENTION", "").isdigit()
                else os.environ.get("OMOAI_LOG_RETENTION")
            ),
            compression=os.environ.get("OMOAI_LOG_COMPRESSION") or None,
            enqueue=os.environ.get("OMOAI_LOG_ENQUEUE", "true").lower() == "true",
            enable_performance_logging=os.environ.get(
                "OMOAI_LOG_PERFORMANCE", "true"
            ).lower()
            == "true",
            performance_threshold_ms=float(
                os.environ.get("OMOAI_LOG_PERF_THRESHOLD", "100.0")
            ),
            enable_request_tracing=os.environ.get("OMOAI_LOG_TRACING", "true").lower()
            == "true",
            trace_headers=os.environ.get("OMOAI_LOG_TRACE_HEADERS", "false").lower()
            == "true",
            enable_error_tracking=os.environ.get("OMOAI_LOG_ERRORS", "true").lower()
            == "true",
            include_stacktrace=os.environ.get("OMOAI_LOG_STACKTRACE", "true").lower()
            == "true",
            enable_metrics=os.environ.get("OMOAI_LOG_METRICS", "true").lower()
            == "true",
            metrics_interval=int(os.environ.get("OMOAI_LOG_METRICS_INTERVAL", "60")),
            debug_mode=os.environ.get("OMOAI_DEBUG", "false").lower() == "true",
            quiet_mode=os.environ.get("OMOAI_QUIET", "false").lower() == "true",
        )


def _add_json_sink_with_custom_serializer(
    logger_obj,
    file_path: str,
    level_name: str,
    rotation,
    retention,
    compression,
    enqueue: bool,
):
    """Add JSON sink using deterministic flat serializer to maintain test contract."""
    from .serializers import flat_json_serializer

    try:
        return logger_obj.add(
            file_path,
            level=level_name,
            format=flat_json_serializer,  # Custom deterministic serializer
            serialize=False,  # We handle serialization ourselves
            rotation=rotation,
            retention=retention,
            compression=compression,
            enqueue=enqueue,
            backtrace=False,
            diagnose=False,
        )
    except (PermissionError, OSError):
        # Sandbox environments may deny multiprocessing primitives used by enqueue
        return logger_obj.add(
            file_path,
            level=level_name,
            format=flat_json_serializer,
            serialize=False,
            rotation=rotation,
            retention=retention,
            compression=compression,
            enqueue=False,
            backtrace=False,
            diagnose=False,
        )


def get_level_name(config: LoggingConfig) -> str:
    """Get textual level name suitable for sinks like Loguru."""
    if config.debug_mode:
        return "DEBUG"
    if config.quiet_mode:
        return "ERROR"
    return (config.level or "INFO").upper()


def get_logging_config() -> LoggingConfig:
    """Get logging configuration with clear precedence.

    Precedence (highest last):
    1) config.yaml logging (and nested env OMOAI_LOGGING__*)
    2) legacy env OMOAI_LOG_* (overrides YAML only)
    3) nested env OMOAI_LOGGING__* (wins over both)
    """
    import os

    # Base from main config (includes YAML and nested env OMOAI_LOGGING__*)
    base = LoggingConfig()
    try:
        main_config = get_config()
        if hasattr(main_config, "logging") and main_config.logging is not None:
            logging_dict = (
                main_config.logging.model_dump()
                if hasattr(main_config.logging, "model_dump")
                else main_config.logging.__dict__
            )
            for key, value in logging_dict.items():
                if hasattr(base, key) and value is not None:
                    setattr(base, key, value)
    except Exception:
        # If main config unavailable, keep defaults
        pass

    # Legacy env overlay (OMOAI_LOG_*) built from environment
    env_direct = LoggingConfig.from_environment()

    # Mapping of fields to direct and nested env var names
    direct_map = {
        "level": "OMOAI_LOG_LEVEL",
        "format_type": "OMOAI_LOG_FORMAT",
        "enable_console": "OMOAI_LOG_CONSOLE",
        "enable_file": "OMOAI_LOG_FILE_ENABLED",
        "log_file": "OMOAI_LOG_FILE",
        "enable_text_file": "OMOAI_LOG_TEXT_FILE_ENABLED",
        "text_log_file": "OMOAI_LOG_TEXT_FILE",
        "rotation": "OMOAI_LOG_ROTATION",
        "retention": "OMOAI_LOG_RETENTION",
        "compression": "OMOAI_LOG_COMPRESSION",
        "enqueue": "OMOAI_LOG_ENQUEUE",
        "enable_performance_logging": "OMOAI_LOG_PERFORMANCE",
        "performance_threshold_ms": "OMOAI_LOG_PERF_THRESHOLD",
        "enable_request_tracing": "OMOAI_LOG_TRACING",
        "trace_headers": "OMOAI_LOG_TRACE_HEADERS",
        "enable_error_tracking": "OMOAI_LOG_ERRORS",
        "include_stacktrace": "OMOAI_LOG_STACKTRACE",
        "enable_metrics": "OMOAI_LOG_METRICS",
        "metrics_interval": "OMOAI_LOG_METRICS_INTERVAL",
        "debug_mode": "OMOAI_DEBUG",
        "quiet_mode": "OMOAI_QUIET",
    }

    # Apply legacy env values only if set AND no nested env present for same field
    for field, env_key in direct_map.items():
        if env_key in os.environ:
            nested_key = f"OMOAI_LOGGING__{field.upper()}"
            if nested_key not in os.environ:
                setattr(base, field, getattr(env_direct, field))

    return base


class InterceptHandler(logging.Handler):
    """Route stdlib logging records to Loguru while preserving extras."""

    _exclude: ClassVar[set[str]] = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
    }

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from loguru import logger as _logger
        except Exception:
            return

        try:
            level = _logger.level(record.levelname).name
        except Exception:
            level = record.levelno

        # Find caller depth for correct source
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        # Bind record extras (including module name)
        extras = {k: v for k, v in record.__dict__.items() if k not in self._exclude}
        if "name" not in extras:
            extras["name"] = record.name
        _logger.bind(**extras).opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _add_sink_with_fallback(logger_obj, *args, enqueue: bool, **kwargs):
    """Add a Loguru sink with enqueue, falling back to non-enqueue if denied."""
    try:
        return logger_obj.add(*args, enqueue=enqueue, **kwargs)
    except (PermissionError, OSError):
        # Sandbox environments may deny multiprocessing primitives used by enqueue
        return logger_obj.add(*args, enqueue=False, **kwargs)


def configure_python_logging(config: LoggingConfig) -> None:
    """Configure logging via Loguru and intercept stdlib logging."""
    from loguru import logger as _logger

    # Intercept stdlib logging
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.setLevel(logging.NOTSET)
    root.addHandler(InterceptHandler())

    # Reset Loguru sinks
    try:
        _logger.remove()
    except Exception:
        pass

    level_name = get_level_name(config)

    # Console sink
    if config.enable_console and not config.quiet_mode:
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan> | "
            "<level>{message}</level>"
        )
        _add_sink_with_fallback(
            _logger,
            sys.stdout,
            level=level_name,
            format=fmt,
            colorize=True,
            enqueue=config.enqueue,
            backtrace=False,
            diagnose=False,
        )

    # File sink (human-readable text)
    if config.enable_text_file and config.text_log_file:
        config.text_log_file.parent.mkdir(parents=True, exist_ok=True)
        rotation = config.rotation or config.max_file_size
        retention = (
            config.retention if config.retention is not None else config.backup_count
        )
        text_fmt = (
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name} | "
            "{message} | "
            "{extra}"
        )
        _add_sink_with_fallback(
            _logger,
            str(config.text_log_file),
            level=level_name,
            format=text_fmt,
            colorize=False,
            rotation=rotation,
            retention=retention,
            compression=config.compression,
            enqueue=config.enqueue,
            backtrace=False,
            diagnose=False,
        )

    # File sink (JSON) with deterministic schema
    if config.enable_file and config.log_file:
        config.log_file.parent.mkdir(parents=True, exist_ok=True)
        rotation = config.rotation or config.max_file_size
        retention = (
            config.retention if config.retention is not None else config.backup_count
        )
        _add_json_sink_with_custom_serializer(
            _logger,
            str(config.log_file),
            get_level_name(config),
            rotation,
            retention,
            config.compression,
            config.enqueue,
        )

    # Configure third-party loggers (levels) via stdlib
    _configure_third_party_loggers(config)


def _configure_third_party_loggers(config: LoggingConfig) -> None:
    """Configure logging levels for third-party libraries."""
    # Reduce noise from verbose libraries
    noisy_loggers = [
        "urllib3.connectionpool",
        "requests.packages.urllib3",
        "httpx",
        "httpcore",
        "torch",
        "transformers",
        "pydantic",
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "litestar.middleware",
        # vLLM loggers (not previously included): ensure visibility when debug_mode is enabled
        "vllm",
        "vllm.engine",
        "vllm.worker",
        "vllm.model_executor",
    ]

    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        if config.debug_mode:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

    # Ensure vLLM loggers propagate to root so our sinks capture them
    vllm_loggers = [
        "vllm",
        "vllm.engine",
        "vllm.worker",
        "vllm.model_executor",
    ]
    for name in vllm_loggers:
        logging.getLogger(name).propagate = True

    # Special handling for specific loggers
    if not config.debug_mode:
        # Quiet down transformers tokenizer warnings
        logging.getLogger("transformers.tokenization_utils_base").setLevel(
            logging.ERROR
        )

        # Reduce torch compilation noise
        logging.getLogger("torch._inductor").setLevel(logging.WARNING)

        # Reduce pydantic validation noise in production
        logging.getLogger("pydantic.main").setLevel(logging.WARNING)

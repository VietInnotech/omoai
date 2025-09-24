"""
Pydantic schemas for configuration validation.

This module provides comprehensive validation for the OMOAI configuration
with security defaults and type safety.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AlignmentConfig(BaseModel):
    """Configuration for phonetic alignment."""

    enabled: bool = Field(default=False, description="Enable or disable phonetic alignment.")
    language: str | Literal["auto"] = Field(
        default="auto",
        description='Language code for the alignment model. Use "auto" to derive from ASR language.',
    )
    device: Literal["cpu", "cuda", "auto"] = Field(
        default="cpu", description="Device to use for alignment (cpu, cuda, or auto-detect)."
    )
    align_model: str | None = Field(
        default=None, description="Optional override for the Hugging Face model name."
    )
    return_char_alignments: bool = Field(
        default=False, description="Whether to compute and return character-level timestamps."
    )
    interpolate_method: Literal["nearest", "linear", "ignore"] = Field(
        default="nearest", description="Method for interpolating word timestamps."
    )
    print_progress: bool = Field(
        default=False, description="Whether to print alignment progress to the console."
    )

    @field_validator("device")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        if v == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except (ImportError, RuntimeError):
                return "cpu"
        return v


class LoggingSettings(BaseModel):
    """Top-level logging settings, configurable via config.yaml.

    These fields mirror the logging_system LoggingConfig so values can merge in.
    """

    # Base behavior
    level: str = Field(default="INFO", description="Root logging level")
    format_type: str = Field(
        default="structured", description="structured | json | simple"
    )
    enable_console: bool = Field(default=True, description="Enable console output")
    enable_file: bool = Field(default=False, description="Enable file output")
    # Optional human-readable text log alongside JSON
    enable_text_file: bool = Field(
        default=False, description="Enable text log file output"
    )

    # File logging
    log_file: Path | None = Field(
        default=Path("logs/app.log"), description="Log file path"
    )
    text_log_file: Path | None = Field(
        default=Path("logs/app.log"), description="Text log file path"
    )
    max_file_size: int = Field(
        default=10 * 1024 * 1024, description="Max file size in bytes"
    )
    backup_count: int = Field(default=5, description="Number of rotated backups")

    # Advanced (for future Loguru migration compatibility)
    rotation: str | None = Field(
        default=None, description="Rotation policy (e.g., '10 MB')"
    )
    retention: str | None = Field(
        default=None, description="Retention policy (e.g., '14 days')"
    )
    compression: str | None = Field(
        default=None, description="Compression (e.g., 'gz')"
    )
    enqueue: bool | None = Field(default=None, description="Async logging if supported")

    # Performance / tracing / errors / metrics
    enable_performance_logging: bool = Field(default=True)
    performance_threshold_ms: float = Field(default=100.0)
    enable_request_tracing: bool = Field(default=True)
    trace_headers: bool = Field(default=False)
    enable_error_tracking: bool = Field(default=True)
    include_stacktrace: bool = Field(default=True)
    enable_metrics: bool = Field(default=True)
    metrics_interval: int = Field(default=60)

    # Modes
    debug_mode: bool = Field(default=False)
    quiet_mode: bool = Field(default=False)

    @field_validator("log_file")
    @classmethod
    def normalize_log_file(cls, v: Path | None) -> Path | None:
        """Normalize @logs/ alias to ./logs/ for convenience."""
        if v is None:
            return v
        s = str(v)
        if s.startswith("@logs/"):
            return Path("logs") / s[len("@logs/") :]
        return v

    @field_validator("text_log_file")
    @classmethod
    def normalize_text_log_file(cls, v: Path | None) -> Path | None:
        if v is None:
            return v
        s = str(v)
        if s.startswith("@logs/"):
            return Path("logs") / s[len("@logs/") :]
        return v


class PathsConfig(BaseModel):
    """Configuration for file and directory paths."""

    chunkformer_dir: Path = Field(
        description="Path to the ChunkFormer source directory",
    )
    chunkformer_checkpoint: Path = Field(
        description="Path to the ChunkFormer model checkpoint",
    )
    out_dir: Path = Field(
        default=Path("data/output"),
        description="Output directory for artifacts",
    )

    @field_validator("chunkformer_dir", "chunkformer_checkpoint")
    @classmethod
    def validate_required_paths_exist(cls, v: Path) -> Path:
        """Validate that required paths exist."""
        if not v.exists():
            raise ValueError(f"Required path does not exist: {v}")
        return v

    @field_validator("out_dir")
    @classmethod
    def ensure_out_dir_exists(cls, v: Path) -> Path:
        """Ensure output directory exists or can be created."""
        try:
            v.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot create output directory {v}: {e}") from e
        return v


class ASRConfig(BaseModel):
    """Configuration for Automatic Speech Recognition."""

    total_batch_duration_s: int = Field(
        default=1800,
        ge=60,  # Minimum 1 minute
        le=7200,  # Maximum 2 hours
        description="Maximum audio duration (seconds) per batch",
    )
    chunk_size: int = Field(
        default=64,
        ge=16,
        le=512,
        description="ChunkFormer chunk size",
    )
    left_context_size: int = Field(
        default=128,
        ge=0,
        le=1024,
        description="Left context size for ChunkFormer",
    )
    right_context_size: int = Field(
        default=128,
        ge=0,
        le=1024,
        description="Right context size for ChunkFormer",
    )
    device: Literal["cpu", "cuda", "auto"] = Field(
        default="auto",
        description="Device for ASR inference",
    )
    autocast_dtype: Literal["fp32", "bf16", "fp16"] | None = Field(
        default="fp16",
        description="Autocast dtype for mixed precision",
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device setting."""
        if v == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return v


class LLMConfig(BaseModel):
    """Configuration for Large Language Model inference."""

    model_id: str | None = Field(
        default=None,
        description="Hugging Face model identifier",
    )
    quantization: str | None = Field(
        default="auto",
        description="Quantization method (auto, awq, gptq, etc.)",
    )
    max_model_len: int = Field(
        default=8192,
        ge=512,
        le=200000,
        description="Maximum model context length",
    )
    gpu_memory_utilization: float = Field(
        default=0.85,
        ge=0.1,
        le=1.0,
        description="GPU memory utilization ratio",
    )
    max_num_seqs: int = Field(
        default=1,
        ge=1,
        le=256,
        description="Maximum number of concurrent sequences",
    )
    max_num_batched_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of batched tokens",
    )
    trust_remote_code: bool = Field(
        default=False,  # SECURITY: Default to False
        description="Whether to trust remote code execution (SECURITY RISK)",
    )

    @field_validator("trust_remote_code")
    @classmethod
    def warn_trust_remote_code(cls, v: bool) -> bool:
        """Warn about security implications of trust_remote_code."""
        if v:
            import warnings

            warnings.warn(
                "trust_remote_code=True enables arbitrary code execution from model repositories. "
                "This is a SECURITY RISK. Only enable if you trust the model source.",
                UserWarning,
                stacklevel=3,
            )
        return v


class SamplingConfig(BaseModel):
    """Configuration for LLM sampling parameters."""

    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )


class VADWebrtcConfig(BaseModel):
    mode: int = Field(default=2, ge=0, le=3, description="Aggressiveness 0..3")
    frame_ms: int = Field(default=20, description="Frame size in ms (10/20/30)")
    # Hangover guidance — primarily covered by overlap_s globally
    start_hangover_frames: int = Field(default=3, ge=0)
    end_hangover_frames: int = Field(default=5, ge=0)


class VADSileroConfig(BaseModel):
    threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    min_speech_duration_ms: int = Field(default=250, ge=0)
    min_silence_duration_ms: int = Field(default=100, ge=0)
    max_speech_duration_s: float = Field(default=30.0, ge=1.0)
    speech_pad_ms: int = Field(default=30, ge=0)
    window_size_samples: int = Field(default=512, ge=128)


class VADPyannoteConfig(BaseModel):
    min_duration_on: float = Field(default=0.1, ge=0.0)
    min_duration_off: float = Field(default=0.1, ge=0.0)
    pad_onset: float = Field(default=0.0, ge=0.0)
    pad_offset: float = Field(default=0.0, ge=0.0)


class VADConfig(BaseModel):
    """Configuration for Voice Activity Detection pre-segmentation."""

    enabled: bool = Field(default=False, description="Enable VAD pre-segmentation")
    method: Literal["webrtc", "silero", "pyannote"] = Field(
        default="silero", description="VAD method"
    )
    chunk_size: float = Field(
        default=30.0, ge=5.0, le=300.0, description="Max speech window length (s)"
    )
    overlap_s: float = Field(
        default=0.4, ge=0.0, le=2.0, description="Window overlap for context (s)"
    )
    vad_onset: float = Field(default=0.50, ge=0.0, le=1.0)
    vad_offset: float = Field(default=0.363, ge=0.0, le=1.0)
    min_speech_s: float = Field(default=0.30, ge=0.0, le=2.0)
    min_silence_s: float = Field(default=0.30, ge=0.0, le=2.0)
    device: Literal["cpu", "cuda", "auto"] = Field(default="auto")
    hf_token_env: str | None = Field(
        default="HUGGINGFACE_TOKEN",
        description="Env var name for HF token (pyannote)",
    )

    webrtc: VADWebrtcConfig = Field(default_factory=VADWebrtcConfig)
    silero: VADSileroConfig = Field(default_factory=VADSileroConfig)
    pyannote: VADPyannoteConfig = Field(default_factory=VADPyannoteConfig)

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        if v == "auto":
            try:
                import torch  # type: ignore

                return "cuda" if torch.cuda.is_available() else "cpu"
            except (ImportError, RuntimeError):
                return "cpu"
        return v

class PunctuationConfig(BaseModel):
    """Configuration for punctuation restoration."""

    llm: LLMConfig
    preserve_original_words: bool = Field(
        default=True,
        description="Whether to preserve original word order",
    )
    auto_switch_ratio: float = Field(
        default=0.98,
        ge=0.5,
        le=1.0,
        description="Ratio of context used for token budget",
    )
    auto_margin_tokens: int = Field(
        default=128,
        ge=16,
        le=1024,
        description="Safety margin tokens",
    )
    enable_paragraphs: bool = Field(
        default=True,
        description="Enable paragraph breaks based on timing",
    )
    join_separator: str = Field(
        default=" ",
        description="Separator for joining segments",
    )
    paragraph_gap_seconds: float = Field(
        default=3.0,
        ge=0.5,
        le=30.0,
        description="Time gap threshold for paragraph breaks",
    )
    system_prompt: str = Field(
        description="System prompt for punctuation model",
    )
    user_prompt: str | None = Field(
        default=None,
        description="A user-defined prompt to guide the punctuation model.",
    )

    @field_validator("system_prompt", "user_prompt", mode="before")
    @classmethod
    def resolve_prompt_file_refs(cls, v):
        """Allow @file:/path/to/file to load prompt content from file.

        Useful for overriding large prompts via environment variables.
        """
        if isinstance(v, str) and v.startswith("@file:"):
            path = Path(v[len("@file:"):])
            try:
                return path.read_text(encoding="utf-8")
            except Exception as e:
                raise ValueError(f"Failed to read prompt file {path}: {e}") from e
        return v
    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig,
        description="Sampling configuration",
    )


class SummarizationConfig(BaseModel):
    """Configuration for text summarization."""

    llm: LLMConfig
    map_reduce: bool = Field(
        default=False,
        description="Use map-reduce for very long texts",
    )
    auto_switch_ratio: float = Field(
        default=0.98,
        ge=0.5,
        le=1.0,
        description="Auto-switch to map-reduce ratio",
    )
    auto_margin_tokens: int = Field(
        default=256,
        ge=16,
        le=1024,
        description="Safety margin tokens",
    )
    system_prompt: str = Field(
        description="System prompt for summarization model",
    )
    user_prompt: str | None = Field(
        default=None,
        description="A user-defined prompt to guide the summarization model.",
    )

    @field_validator("system_prompt", "user_prompt", mode="before")
    @classmethod
    def resolve_prompt_file_refs(cls, v):
        if isinstance(v, str) and v.startswith("@file:"):
            path = Path(v[len("@file:"):])
            try:
                return path.read_text(encoding="utf-8")
            except Exception as e:
                raise ValueError(f"Failed to read prompt file {path}: {e}") from e
        return v
    sampling: SamplingConfig = Field(
        default_factory=lambda: SamplingConfig(temperature=0.7),
        description="Sampling configuration",
    )


class TimestampedSummaryConfig(BaseModel):
    """Configuration for timestamped summarization."""

    llm: LLMConfig = Field(
        description="LLM configuration for timestamped summarization",
    )
    auto_margin_tokens: int = Field(
        default=256, ge=0, description="Automatic margin tokens for context"
    )
    return_raw: bool = Field(
        default=False, description="Return raw LLM response alongside parsed result"
    )
    system_prompt: str | None = Field(
        default=None, description="Custom system prompt for timestamped summarization"
    )
    user_prompt: str | None = Field(
        default=None, description="Custom user prompt template for timestamped summarization"
    )
    sampling: SamplingConfig = Field(
        default_factory=SamplingConfig, description="Sampling parameters"
    )

    @field_validator("system_prompt", "user_prompt", mode="before")
    @classmethod
    def resolve_prompt_file_refs(cls, v):
        if isinstance(v, str) and v.startswith("@file:"):
            path = Path(v[len("@file:"):])
            try:
                return path.read_text(encoding="utf-8")
            except Exception as e:
                raise ValueError(f"Failed to read prompt file {path}: {e}") from e
        return v


class TranscriptOutputConfig(BaseModel):
    """Configuration for transcript outputs and file naming."""

    include_raw: bool = Field(
        default=True, description="Include raw transcript in outputs"
    )
    include_punct: bool = Field(
        default=True, description="Include punctuated transcript in outputs"
    )
    include_segments: bool = Field(
        default=True, description="Include segment list in outputs"
    )
    timestamps: Literal["none", "s", "ms", "clock"] = Field(
        default="clock", description="Timestamp format"
    )
    wrap_width: int = Field(
        default=100, ge=0, description="Text wrapping width (0 = no wrapping)"
    )

    # Filenames (when writing to disk)
    file_raw: str = Field(default="transcript.raw.txt")
    file_punct: str = Field(default="transcript.punct.txt")
    file_srt: str = Field(default="transcript.srt")
    file_vtt: str = Field(default="transcript.vtt")
    file_segments: str = Field(default="segments.json")


class SummaryOutputConfig(BaseModel):
    """Configuration for summary outputs and file naming."""

    mode: Literal["bullets", "abstract", "both", "none"] = Field(default="both")
    bullets_max: int = Field(default=7, ge=0)
    abstract_max_chars: int = Field(default=1000, ge=0)
    language: str = Field(default="vi")
    file: str = Field(default="summary.md")


class APIDefaultsConfig(BaseModel):
    """Defaults for API response shaping when query params are not provided.

    Mirrors OutputFormatParams so admin can configure default includes.
    """

    formats: list[Literal["json", "text", "srt", "vtt", "md"]] | None = None
    include: list[Literal["transcript_raw", "transcript_punct", "segments", "timestamped_summary", "summary"]] | None = None
    ts: Literal["none", "s", "ms", "clock"] | None = None
    summary: Literal["bullets", "abstract", "both", "none"] | None = None
    summary_bullets_max: int | None = None
    summary_lang: str | None = None
    include_quality_metrics: bool | None = None
    include_diffs: bool | None = None
    return_summary_raw: bool | None = None
    include_vad: bool | None = None
    summary_fields: list[Literal["title", "abstract", "bullets", "raw"]] | None = None
    timestamped_summary_fields: list[Literal["summary_text", "timestamps", "raw"]] | None = None


class OutputConfig(BaseModel):
    """Top-level output configuration.

    Includes a few legacy compatibility fields used by scripts/post.py. New code
    should prefer structured `formats`, `transcript`, and `summary` sections.
    """

    # Legacy compatibility (kept for scripts/post.py behavior)
    write_separate_files: bool = Field(default=False)
    transcript_file: str = Field(default="transcript.txt")
    summary_file: str = Field(default="summary.txt")
    wrap_width: int = Field(default=0, ge=0)

    # Structured configuration for programmatic outputs
    formats: list[Literal["json", "text", "srt", "vtt", "md"]] = Field(
        default_factory=lambda: ["json"],
        description="Which formats to generate",
    )
    transcript: TranscriptOutputConfig = Field(default_factory=TranscriptOutputConfig)
    summary: SummaryOutputConfig = Field(default_factory=SummaryOutputConfig)
    final_json: str = Field(default="final.json")

    # API persistence controls
    save_on_api: bool = Field(default=False)
    save_formats_on_api: list[Literal["final_json", "segments", "transcripts"]] = Field(
        default_factory=lambda: ["final_json"],
    )
    api_output_dir: Path | None = Field(default=None)

    # API default response shaping (no query params)
    api_defaults: APIDefaultsConfig = Field(default_factory=APIDefaultsConfig)

    @model_validator(mode="after")
    def validate_legacy_mapping(self) -> "OutputConfig":
        """Provide a simple consistency mapping between legacy and structured fields.

        - If wrap_width is set (>0), prefer it over transcript.wrap_width when writing plain text.
        """
        try:
            if self.wrap_width and self.wrap_width > 0 and self.transcript:
                # Do not overwrite if transcript.wrap_width explicitly different in configs
                if (self.transcript.wrap_width or 0) != self.wrap_width:
                    self.transcript.wrap_width = int(self.wrap_width)
        except (ValueError, AttributeError) as e:
            logging.getLogger(__name__).debug("Failed to sync wrap_width: %s", e)
        return self


class APIConfig(BaseModel):
    """Configuration for API server."""

    host: str = Field(
        default="127.0.0.1",  # SECURITY: Default to localhost
        description="API server host",
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="API server port",
    )
    max_body_size_mb: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum request body size in MB",
    )
    request_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=3600,
        description="Request timeout in seconds",
    )
    temp_dir: Path = Field(
        default_factory=lambda: Path(tempfile.gettempdir()),
        description="Temporary files directory",
    )
    cleanup_temp_files: bool = Field(
        default=True,
        description="Automatically cleanup temporary files",
    )
    # Runtime UX controls for subprocessed scripts (ASR, postprocess)
    stream_subprocess_output: bool = Field(
        default=False,
        description="Stream child process stdout/stderr to the API terminal",
    )
    verbose_scripts: bool = Field(
        default=False,
        description="Pass --verbose to helper scripts for richer logs",
    )
    enable_progress_output: bool = Field(
        default=False,  # SECURITY: Default to False for production
        description="Enable progress output (may leak information)",
    )
    # Response formatting / content negotiation
    default_response_format: Literal["json", "text"] = Field(
        default="json", description="Default response format when no explicit preference"
    )
    allow_accept_override: bool = Field(
        default=True,
        description="Allow Accept header (text/plain) to override default response",
    )
    allow_query_format_override: bool = Field(
        default=True,
        description="Allow query param formats=text to force text response",
    )
    health_check_dependencies: list[str] = Field(
        default_factory=lambda: ["ffmpeg", "config_file"],
        description="Dependencies to check in health endpoint",
    )

    @field_validator("temp_dir")
    @classmethod
    def validate_temp_dir(cls, v: Path) -> Path:
        """Validate temporary directory."""
        try:
            v.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = v / ".omoai_access_test"
            test_file.touch()
            test_file.unlink()
        except (OSError, PermissionError) as e:
            raise ValueError(f"Temporary directory {v} is not accessible: {e}") from e
        return v


class OmoAIConfig(BaseSettings):
    """Main OMOAI configuration with environment variable support."""

    model_config = SettingsConfigDict(
        env_prefix="OMOAI_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="forbid",  # Reject unknown fields
    )

    paths: PathsConfig
    logging: LoggingSettings | None = Field(
        default=None, description="Logging configuration"
    )
    asr: ASRConfig = Field(default_factory=ASRConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    llm: LLMConfig = Field(
        description="Base LLM configuration (model_id required)",
    )
    punctuation: PunctuationConfig
    summarization: SummarizationConfig
    timestamped_summary: TimestampedSummaryConfig | None = Field(
        default=None, description="Configuration for timestamped summarization"
    )
    output: OutputConfig | None = Field(
        default=None, description="Output configuration"
    )
    api: APIConfig = Field(default_factory=APIConfig)

    @model_validator(mode="after")
    def validate_config_consistency(self) -> "OmoAIConfig":
        """Validate cross-field consistency."""
        # Validate base LLM has model_id
        if not self.llm.model_id:
            raise ValueError("Base llm.model_id is required")

        # Ensure punctuation, summarization, and timestamped_summary inherit base LLM model_id if not set
        if not self.punctuation.llm.model_id:
            self.punctuation.llm.model_id = self.llm.model_id

        if not self.summarization.llm.model_id:
            self.summarization.llm.model_id = self.llm.model_id

        if self.timestamped_summary and not self.timestamped_summary.llm.model_id:
            self.timestamped_summary.llm.model_id = self.llm.model_id

        # Validate trust_remote_code consistency
        if (
            self.llm.trust_remote_code
            or self.punctuation.llm.trust_remote_code
            or self.summarization.llm.trust_remote_code
            or (self.timestamped_summary and self.timestamped_summary.llm.trust_remote_code)
        ):
            import warnings

            warnings.warn(
                "At least one LLM configuration has trust_remote_code=True. "
                "This enables arbitrary code execution. Ensure you trust all model sources.",
                UserWarning,
                stacklevel=2,
            )

        return self

    @classmethod
    def load_from_yaml(cls, config_path: Path) -> "OmoAIConfig":
        """Load configuration from YAML file with validation."""
        import os

        import yaml

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file {config_path}: {e}") from e

        if not raw_config:
            raise ValueError(f"Empty configuration file: {config_path}")

        # Allow targeted env overrides for large prompt fields (robust even if
        # nested env processing is limited by BaseSettings semantics)
        punct = raw_config.setdefault("punctuation", {})
        summ = raw_config.setdefault("summarization", {})
        for env_key, dst in (
            ("OMOAI_PUNCTUATION__SYSTEM_PROMPT", (punct, "system_prompt")),
            ("OMOAI_PUNCTUATION__USER_PROMPT", (punct, "user_prompt")),
            ("OMOAI_SUMMARIZATION__SYSTEM_PROMPT", (summ, "system_prompt")),
            ("OMOAI_SUMMARIZATION__USER_PROMPT", (summ, "user_prompt")),
        ):
            val = os.environ.get(env_key)
            if val:
                dst[0][dst[1]] = val

        try:
            return cls(**raw_config)
        except Exception as e:
            raise ValueError(
                f"Configuration validation failed for {config_path}: {e}"
            ) from e

    def model_dump_yaml(self, **kwargs) -> str:
        """Export configuration as YAML string."""
        import yaml

        # Convert Path objects to strings for YAML serialization
        data = self.model_dump(mode="python", **kwargs)

        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj

        data = convert_paths(data)
        return yaml.dump(data, default_flow_style=False, sort_keys=False)

    def save_to_yaml(self, config_path: Path, **kwargs) -> None:
        """Save configuration to YAML file."""
        yaml_content = self.model_dump_yaml(**kwargs)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)


# Legacy compatibility functions
def load_config(config_path: str | Path | None = None) -> OmoAIConfig:
    """Load and validate configuration from YAML file."""
    if config_path is None:
        # Search for config.yaml in common locations
        search_paths = [
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent.parent.parent / "config.yaml",
        ]

        env_config = os.environ.get("OMOAI_CONFIG")
        if env_config:
            search_paths.insert(0, Path(env_config))

        config_path = None
        for path in search_paths:
            if path.exists():
                config_path = path
                break

        if config_path is None:
            raise FileNotFoundError(
                f"Configuration file not found. Searched: {search_paths}. "
                f"Set OMOAI_CONFIG environment variable or create config.yaml",
            )

    return OmoAIConfig.load_from_yaml(Path(config_path))


# Module-level singleton container for the validated configuration.
# Using a module-level variable avoids attaching attributes to functions,
# which resolves static-analyzer (Pylance) warnings and is simpler to reason about.
_GLOBAL_CONFIG: OmoAIConfig | None = None


def get_config() -> OmoAIConfig:
    """Get the global configuration instance (singleton pattern).

    Loads the configuration once and returns the same validated instance on
    subsequent calls. The configuration is stored in the module-level
    _GLOBAL_CONFIG variable to keep static analysis tools happy.
    """
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        _GLOBAL_CONFIG = load_config()
    return _GLOBAL_CONFIG


def reload_config(config_path: str | Path | None = None) -> OmoAIConfig:
    """Reload the global configuration instance.

    If config_path is provided, it will be used as the OMOAI_CONFIG location
    for this reload call (temporarily set in environment).
    """
    global _GLOBAL_CONFIG
    if config_path is not None:
        os.environ["OMOAI_CONFIG"] = str(config_path)
    # Clear the cached instance so the next get_config() call reloads from file
    _GLOBAL_CONFIG = None
    return get_config()

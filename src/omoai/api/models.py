from typing import Any, Literal

from litestar.datastructures import UploadFile
from pydantic import BaseModel, ConfigDict


# Request Models
class PipelineRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    audio_file: UploadFile


class PreprocessRequest(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    audio_file: UploadFile


class ASRRequest(BaseModel):
    preprocessed_path: str


class PostprocessRequest(BaseModel):
    asr_output: dict


# Enhanced Output Parameters (Query Parameters)
class OutputFormatParams(BaseModel):
    """Query parameters for controlling output formats and options."""

    # Format selection
    formats: list[Literal["json", "text", "srt", "vtt", "md"]] | None = None

    # Transcript options
    include: list[Literal["transcript_raw", "transcript_punct", "segments", "timestamped_summary", "summary"]] | None = (
        None
    )
    ts: Literal["none", "s", "ms", "clock"] | None = None

    # Summary options
    summary: Literal["bullets", "abstract", "both", "none"] | None = None
    summary_bullets_max: int | None = None
    summary_lang: str | None = None
    summary_fields: list[Literal["title", "abstract", "bullets", "raw"]] | None = None
    timestamped_summary_fields: list[Literal["summary_text", "timestamps", "raw"]] | None = None

    # Quality metrics and diff options
    include_quality_metrics: bool | None = None
    include_diffs: bool | None = None
    # Simple raw summary option
    return_summary_raw: bool | None = None
    # Option to include raw LLM response for timestamped summary
    return_timestamped_summary_raw: bool | None = None
    # Include VAD metadata (if available) in the saved or returned JSON
    include_vad: bool | None = None


# Response Models
class QualityMetrics(BaseModel):
    """Quality metrics for punctuation alignment."""

    wer: float | None = None
    cer: float | None = None
    per: float | None = None
    uwer: float | None = None
    fwer: float | None = None
    alignment_confidence: float | None = None


class HumanReadableDiff(BaseModel):
    """Human-readable diff for quality assurance."""

    original_text: str | None = None
    punctuated_text: str | None = None
    diff_output: str | None = None
    alignment_summary: str | None = None


class PipelineResponse(BaseModel):
    # Summary is a structured dict with the shape:
    # {
    #   "title": str,
    #   "abstract": str,
    #   "bullets": List[str]
    # }
    summary: dict
    segments: list
    # Punctuated transcript text for convenience (used for default text/plain responses)
    # Note: Raw transcript is excluded by default for privacy and data minimization
    transcript_punct: str | None = None
    transcript_raw: str | None = None
    quality_metrics: QualityMetrics | None = None
    diffs: HumanReadableDiff | None = None
    # Optional raw LLM summary text (unparsed), included only on request
    # Optional metadata (e.g., VAD) included only when requested
    # Optional timestamped summary with [HH:MM:SS] formatted timestamps
    # Can include an optional "raw" field with the unparsed LLM response
    timestamped_summary: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class PreprocessResponse(BaseModel):
    output_path: str


class ASRResponse(BaseModel):
    segments: list
    transcript_raw: str | None = None


class PostprocessResponse(BaseModel):
    # Postprocess now returns a structured summary dict (see PipelineResponse.summary)
    summary: dict
    segments: list  # Include segments with punctuated text
    quality_metrics: QualityMetrics | None = None
    diffs: HumanReadableDiff | None = None

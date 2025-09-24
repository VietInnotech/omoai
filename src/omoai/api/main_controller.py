import logging
from typing import Annotated, Literal

from litestar import Controller, get, post
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.response import Redirect, Response
from starlette.requests import Request

from omoai.api.models import OutputFormatParams, PipelineRequest, PipelineResponse
from omoai.api.services import run_full_pipeline
from omoai.config.schemas import get_config

# Set up logger
logger = logging.getLogger(__name__)


class MainController(Controller):
    """
    MainController handles the main API endpoint for running the entire audio processing pipeline.
    """

    @get("/")
    async def root(self) -> Redirect:
        """
        Root endpoint that redirects to the OpenAPI schema documentation.

        This provides a convenient way for users to access the API documentation
        when they visit the root URL of the application.
        """
        return Redirect(path="/schema")

    @post("/pipeline")
    async def pipeline(
        self,
        request: Request,
        data: Annotated[
            PipelineRequest, Body(media_type=RequestEncodingType.MULTI_PART)
        ],
        # Query parameters for output formatting
        formats: list[Literal["json", "text", "srt", "vtt", "md"]] | None = None,
        include: list[Literal["transcript_raw", "transcript_punct", "segments", "timestamped_summary", "summary"]]
        | None = None,
        ts: Literal["none", "s", "ms", "clock"] | None = None,
        summary: Literal["bullets", "abstract", "both", "none"] | None = None,
        summary_bullets_max: int | None = None,
        summary_lang: str | None = None,
        return_summary_raw: bool | None = None,
        return_timestamped_summary_raw: bool | None = None,
        include_vad: bool | None = None,
        # Quality metrics and diff options
        include_quality_metrics: bool | None = None,
        include_diffs: bool | None = None,
        # Async processing option
        async_: bool | None = None,
    ) -> PipelineResponse | Response[str] | Response[dict[str, str]]:
        """
        Endpoint to run the entire audio processing pipeline.

        This endpoint executes the full workflow:
        1. Preprocess: Take the uploaded audio file and preprocess it.
        2. ASR: Pass the preprocessed file path to the ASR logic to get the raw transcript and segments.
        3. Post-process: Pass the ASR output to the post-processing logic to get the final punctuated transcript and summary.

        Default Response (no query parameters):
        - transcript_punct: Punctuated transcript text
        - summary.bullets: Bullet point summary
        - summary.abstract: Abstract summary
        - segments: Empty array (excluded by default)

        Query Parameters (optional):
        - formats: List of output formats (json, text, srt, vtt, md)
        - include: What to include (transcript_raw, transcript_punct, segments, timestamped_summary)
        - ts: Timestamp format (none, s, ms, clock)
        - summary: Summary type (both=default, bullets, abstract, none)
        - summary_bullets_max: Maximum number of bullet points
        - summary_lang: Summary language

        Examples:
        - Default: POST /pipeline (returns bullets + abstract + punctuated transcript)
        - With segments: POST /pipeline?include=segments
        - Bullets only: POST /pipeline?summary=bullets
        - Full response: POST /pipeline?include=transcript_raw,segments
        """
        # Create OutputFormatParams object from query parameters
        output_params = OutputFormatParams(
            formats=formats,
            include=include,
            ts=ts,
            summary=summary,
            summary_bullets_max=summary_bullets_max,
            summary_lang=summary_lang,
            include_quality_metrics=include_quality_metrics,
            include_diffs=include_diffs,
            return_summary_raw=return_summary_raw,
            return_timestamped_summary_raw=return_timestamped_summary_raw,
            include_vad=include_vad,
        )

        # Apply defaults from configuration when not provided via query
        try:
            cfg = get_config()
            out_cfg = getattr(cfg, "output", None)
            sum_cfg = getattr(out_cfg, "summary", None)
            api_def = getattr(out_cfg, "api_defaults", None)
            # Prefer output.api_defaults if provided; fall back to output.summary for summary defaults
            if out_cfg:
                if api_def:
                    # Set defaults only when not provided via query
                    if output_params.formats is None and getattr(api_def, "formats", None) is not None:
                        output_params.formats = list(api_def.formats)  # type: ignore[assignment]
                    if output_params.include is None and getattr(api_def, "include", None) is not None:
                        output_params.include = list(api_def.include)  # type: ignore[assignment]
                    if output_params.ts is None and getattr(api_def, "ts", None) is not None:
                        output_params.ts = api_def.ts  # type: ignore[assignment]
                    if output_params.summary is None and getattr(api_def, "summary", None) is not None:
                        output_params.summary = api_def.summary  # type: ignore[assignment]
                    if (
                        output_params.summary_bullets_max is None
                        and getattr(api_def, "summary_bullets_max", None) is not None
                    ):
                        output_params.summary_bullets_max = int(api_def.summary_bullets_max)  # type: ignore[assignment]
                    if output_params.summary_lang is None and getattr(api_def, "summary_lang", None) is not None:
                        output_params.summary_lang = str(api_def.summary_lang)  # type: ignore[assignment]
                    if (
                        output_params.include_quality_metrics is None
                        and getattr(api_def, "include_quality_metrics", None) is not None
                    ):
                        output_params.include_quality_metrics = bool(api_def.include_quality_metrics)  # type: ignore[assignment]
                    if (
                        output_params.include_diffs is None
                        and getattr(api_def, "include_diffs", None) is not None
                    ):
                        output_params.include_diffs = bool(api_def.include_diffs)  # type: ignore[assignment]
                    if (
                        output_params.return_summary_raw is None
                        and getattr(api_def, "return_summary_raw", None) is not None
                    ):
                        output_params.return_summary_raw = bool(api_def.return_summary_raw)  # type: ignore[assignment]
                    if (
                        output_params.include_vad is None
                        and getattr(api_def, "include_vad", None) is not None
                    ):
                        output_params.include_vad = bool(api_def.include_vad)  # type: ignore[assignment]
                    if (
                        output_params.summary_fields is None
                        and getattr(api_def, "summary_fields", None) is not None
                    ):
                        output_params.summary_fields = list(api_def.summary_fields)  # type: ignore[assignment]
                    if (
                        output_params.timestamped_summary_fields is None
                        and getattr(api_def, "timestamped_summary_fields", None) is not None
                    ):
                        output_params.timestamped_summary_fields = list(api_def.timestamped_summary_fields)  # type: ignore[assignment]
                # Additionally, support legacy summary defaults from output.summary
                if sum_cfg:
                    if output_params.summary is None and getattr(sum_cfg, "mode", None):
                        output_params.summary = sum_cfg.mode  # type: ignore[assignment]
                    if (
                        output_params.summary_bullets_max is None
                        and getattr(sum_cfg, "bullets_max", None) is not None
                    ):
                        output_params.summary_bullets_max = int(sum_cfg.bullets_max)  # type: ignore[assignment]
                    if (
                        output_params.summary_lang is None
                        and getattr(sum_cfg, "language", None)
                    ):
                        output_params.summary_lang = str(sum_cfg.language)  # type: ignore[assignment]
        except (AttributeError, TypeError) as e:
            logger.debug(f"Failed to set summary language: {e}")

        # Debug logging to validate query parameter parsing
        logger.info(
            f"Received query parameters - summary: {summary}, summary_bullets_max: {summary_bullets_max}, include: {include}"
        )
        logger.info(f"Created output_params: {output_params}")
        if output_params:
            logger.info(f"output_params.summary: {output_params.summary}")
            logger.info(
                f"output_params.summary_bullets_max: {output_params.summary_bullets_max}"
            )
            logger.info(f"output_params.include: {output_params.include}")
            logger.info(
                f"output_params.include_quality_metrics: {output_params.include_quality_metrics}, "
                f"include_diffs: {output_params.include_diffs}, return_summary_raw: {output_params.return_summary_raw}"
            )

        # Run pipeline or enqueue async job
        params = output_params if output_params else None
        if async_:
            # Enqueue background job using in-memory manager
            from omoai.api import services as _svc
            from omoai.api.jobs import job_manager

            # Read upload now to decouple from request lifecycle
            file_bytes = await data.audio_file.read()
            upload = _svc._BytesUploadFile(
                data=file_bytes,
                filename=getattr(data.audio_file, "filename", "upload.bin"),
                content_type=getattr(
                    data.audio_file, "content_type", "application/octet-stream"
                ),
            )
            # Construct a new request to satisfy type checks (jobs manager will rebuild)
            data = PipelineRequest(audio_file=upload)
            job_id = await job_manager.submit_pipeline_job(
                file_bytes=file_bytes,
                filename=upload.filename,
                content_type=upload.content_type,
                output_params=params,
            )
            return Response(
                {
                    "job_id": job_id,
                    "status": "queued",
                    "status_url": f"/v1/jobs/{job_id}",
                },
                status_code=202,
            )

        # Synchronous inline processing
        result = await run_full_pipeline(data, params)

        # Content negotiation with configurable defaults
        accept = (request.headers.get("accept") or "").lower()
        api_cfg = None
        try:
            api_cfg = getattr(get_config(), "api", None)
        except (AttributeError, ImportError) as e:
            logger.debug(f"Failed to get API config: {e}")
            api_cfg = None

        # Prefer JSON if client indicates application/json, */*, or no Accept header
        prefer_json = (
            ("application/json" in accept)
            or ("*/*" in accept)
            or (not accept)
        )

        # Start with default from config
        default_fmt = getattr(api_cfg, "default_response_format", "json")
        wants_text = str(default_fmt).lower() == "text"

        # Query override (formats=text)
        allow_query = bool(getattr(api_cfg, "allow_query_format_override", True))
        if allow_query and formats and "text" in formats:
            wants_text = True

        # Accept header override
        allow_accept = bool(getattr(api_cfg, "allow_accept_override", True))
        if allow_accept:
            if ("text/plain" in accept) and not prefer_json:
                wants_text = True

        if wants_text:
            summary_data = getattr(result, "summary", {}) or {}
            # If raw summary requested and available, return only the raw LLM output
            if return_summary_raw:
                raw_text = str(summary_data.get("raw", "") or "").strip()
                if raw_text:
                    return Response(raw_text, media_type="text/plain; charset=utf-8")
            # Otherwise compose plain text with transcript and structured summary (title, abstract, bullets)
            parts: list[str] = []
            if result.transcript_punct:
                parts.append(result.transcript_punct)
            title = summary_data.get("title", "")
            abstract_text = summary_data.get("summary", "") or summary_data.get(
                "abstract", ""
            )
            bullets = summary_data.get("bullets", []) or []
            if title:
                parts.append(f"\n\n# {title}\n")
            if abstract_text:
                parts.append(str(abstract_text))
            if bullets:
                parts.append("\n\n# Points\n")
                parts.extend([f"- {p}" for p in bullets])
            text_response = "\n".join(parts)
            return Response(text_response, media_type="text/plain; charset=utf-8")

        # Default: return service result as-is (services handle include/gating)
        return result

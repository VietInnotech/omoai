"""WhisperX transcription using faster-whisper with internal VAD and alignment."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from faster_whisper import WhisperModel  # type: ignore
from pydub import AudioSegment  # type: ignore

from omoai.config.schemas import OmoAIConfig
from omoai.integrations.vad import apply_overlap, detect_speech, merge_chunks
from omoai.logging_system.logger import get_logger


logger = get_logger(__name__)


def _resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except (ImportError, RuntimeError):
            return "cpu"
    return device


def _resolve_language(cfg: OmoAIConfig, detected: str | None = None) -> str:
    lang = getattr(getattr(cfg.asr, "whisperx", None), "language", None) or "auto"
    if str(lang).lower() == "auto":
        if detected:
            return detected
        return ""
    return str(lang)


def _get_vad_windows(
    audio_path: Path,
    audio_duration: float,
    sample_rate: int,
    cfg: OmoAIConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vad_cfg = getattr(cfg, "vad", None)
    if vad_cfg is None or not bool(getattr(vad_cfg, "enabled", False)):
        window = {"start": 0.0, "end": float(audio_duration), "segments": []}
        metadata = {
            "enabled": False,
            "method": str(getattr(vad_cfg, "method", "none")) if vad_cfg else "none",
            "chunk_size": float(getattr(vad_cfg, "chunk_size", audio_duration)),
            "overlap_s": float(getattr(vad_cfg, "overlap_s", 0.0)),
            "windows": 1,
            "speech_ratio": 1.0 if audio_duration > 0 else 0.0,
        }
        return [window], metadata

    method = str(getattr(vad_cfg, "method", "webrtc"))
    chunk_size = float(getattr(vad_cfg, "chunk_size", 30.0))
    overlap_s = float(getattr(vad_cfg, "overlap_s", 0.4))

    intervals = detect_speech(
        str(audio_path),
        method=method,
        sample_rate=sample_rate,
        vad_onset=float(getattr(vad_cfg, "vad_onset", 0.5)),
        vad_offset=float(getattr(vad_cfg, "vad_offset", 0.363)),
        min_speech_s=float(getattr(vad_cfg, "min_speech_s", 0.30)),
        min_silence_s=float(getattr(vad_cfg, "min_silence_s", 0.30)),
        chunk_size=chunk_size,
        webrtc_mode=int(getattr(getattr(vad_cfg, "webrtc", None), "mode", 2)),
        frame_ms=int(getattr(getattr(vad_cfg, "webrtc", None), "frame_ms", 20)),
        speech_pad_ms=int(getattr(getattr(vad_cfg, "silero", None), "speech_pad_ms", 30)),
        window_size_samples=int(getattr(getattr(vad_cfg, "silero", None), "window_size_samples", 512)),
        device=str(getattr(vad_cfg, "device", "cpu")),
    )

    if not intervals:
        logger.info("[whisperx] VAD returned no intervals; using full audio")
        window = {"start": 0.0, "end": float(audio_duration), "segments": []}
        metadata = {
            "enabled": True,
            "method": method,
            "chunk_size": chunk_size,
            "overlap_s": overlap_s,
            "windows": 1,
            "speech_ratio": 1.0 if audio_duration > 0 else 0.0,
        }
        return [window], metadata

    windows = merge_chunks(intervals, chunk_size=chunk_size)
    windows = apply_overlap(windows, overlap_s=overlap_s, audio_duration=audio_duration)

    speech_seconds = 0.0
    normalized: list[dict[str, Any]] = []
    for w in windows:
        start = max(0.0, float(w.get("start", 0.0)))
        end = min(float(audio_duration), float(w.get("end", start)))
        if end <= start:
            continue
        speech_seconds += end - start
        normalized.append({"start": start, "end": end, "segments": w.get("segments", [])})

    if not normalized:
        normalized = [{"start": 0.0, "end": float(audio_duration), "segments": []}]
        speech_seconds = float(audio_duration)

    metadata = {
        "enabled": True,
        "method": method,
        "chunk_size": chunk_size,
        "overlap_s": overlap_s,
        "windows": len(normalized),
        "speech_ratio": (speech_seconds / max(1e-6, float(audio_duration))) if audio_duration > 0 else 0.0,
    }
    return normalized, metadata


def _apply_alignment(
    segments: list[dict[str, Any]],
    cfg: OmoAIConfig,
    audio_path: Path,
    detected_language: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, dict[str, Any] | None]:
    alignment_cfg = getattr(cfg, "alignment", None)
    if not alignment_cfg or not bool(getattr(alignment_cfg, "enabled", False)):
        return segments, None, None

    try:
        from omoai.integrations.alignment import (
            align_segments,
            load_alignment_model,
            merge_alignment_back,
            to_whisperx_segments,
        )

        alignment_language = alignment_cfg.language
        if alignment_language == "auto":
            alignment_language = (
                detected_language
                or str(getattr(getattr(cfg.asr, "whisperx", None), "language", "")).strip()
                or "en"
            )

        alignment_device = alignment_cfg.device
        if alignment_device == "auto":
            alignment_device = _resolve_device("auto")

        align_model, align_metadata = load_alignment_model(
            language=alignment_language,
            device=alignment_device,
            model_name=alignment_cfg.align_model,
        )

        wx_segments = to_whisperx_segments(segments)
        if not wx_segments:
            return segments, None, {
                "enabled": True,
                "status": "skipped",
                "reason": "no_segments",
            }

        aligned_result = align_segments(
            wx_segments=wx_segments,
            audio_path_or_array=str(audio_path),
            model=align_model,
            metadata=align_metadata,
            device=alignment_device,
            return_char_alignments=alignment_cfg.return_char_alignments,
            interpolate_method=alignment_cfg.interpolate_method,
            print_progress=alignment_cfg.print_progress,
        )

        enriched_segments, word_segments = merge_alignment_back(segments, aligned_result)
        alignment_meta = {
            "enabled": True,
            "status": "success",
            "language": alignment_language,
            "device": alignment_device,
            "model": alignment_cfg.align_model or "default",
            "segments_aligned": len(enriched_segments),
            "words_aligned": len(word_segments or []),
            "return_char_alignments": alignment_cfg.return_char_alignments,
            "interpolate_method": alignment_cfg.interpolate_method,
        }
        return enriched_segments, word_segments, alignment_meta

    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("[whisperx] Alignment failed: %s", exc, exc_info=True)
        return segments, None, {
            "enabled": True,
            "status": "failed",
            "error": str(exc),
        }


def run_whisperx(audio_path: Path, cfg: OmoAIConfig) -> dict[str, Any]:
    """Run faster-whisper transcription using internal VAD and alignment."""
    audio_segment = AudioSegment.from_file(str(audio_path))
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    sr = audio_segment.frame_rate
    duration_s = float(len(audio_segment) / 1000.0)

    # Convert to float32 [-1,1]
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    samples /= 32768.0

    windows, vad_metadata = _get_vad_windows(audio_path, duration_s, sr, cfg)

    wx_cfg = getattr(cfg.asr, "whisperx")
    model_path = Path(wx_cfg.model_path)
    device = _resolve_device(wx_cfg.device)
    compute_type = wx_cfg.compute_type
    valid_compute = {"float16", "float32", "int8", "int8_float16", "int8_float32"}
    if compute_type not in valid_compute:
        compute_type = "float16"

    logger.info(
        "[whisperx] Loading faster-whisper model: path=%s device=%s compute_type=%s",
        model_path,
        device,
        compute_type,
    )
    model = WhisperModel(
        str(model_path),
        device=device,
        compute_type=compute_type,
    )

    # Ensure feature extractor matches CT2 model expectations (e.g., EraX uses 128 mel bins)
    try:
        model.model.load_model()
    except Exception:  # pragma: no cover - CT2 loads lazily
        pass
    try:
        ct2_n_mels = int(getattr(model.model, "n_mels", 0) or 0)
    except Exception:
        ct2_n_mels = 0
    if ct2_n_mels and ct2_n_mels != model.feature_extractor.mel_filters.shape[0]:
        from faster_whisper.feature_extractor import FeatureExtractor

        feat_kwargs = dict(model.feat_kwargs) if model.feat_kwargs else {}
        feat_kwargs.update({
            "feature_size": ct2_n_mels,
            "sampling_rate": feat_kwargs.get("sampling_rate", 16000),
            "hop_length": feat_kwargs.get("hop_length", 160),
            "chunk_length": feat_kwargs.get("chunk_length", 30),
            "n_fft": feat_kwargs.get("n_fft", 400),
        })
        model.feat_kwargs = feat_kwargs
        model.feature_extractor = FeatureExtractor(**feat_kwargs)
        logger.info(
            "[whisperx] Adjusted feature extractor to %s mel bins for CT2 model",
            ct2_n_mels,
        )

    language_override = _resolve_language(cfg)
    beam_size = int(wx_cfg.beam_size)

    segments: list[dict[str, Any]] = []
    transcript_parts: list[str] = []
    detected_language = None

    schedule: list[float] | None = None
    if getattr(wx_cfg, "temperature_schedule", None):
        try:
            schedule = [
                float(t)
                for t in wx_cfg.temperature_schedule
                if t is not None and math.isfinite(float(t))
            ]
        except (TypeError, ValueError):  # pragma: no cover - defensive cast
            schedule = None
        if schedule:
            schedule = [t for t in schedule if 0.0 <= t <= 2.0]
        if schedule is not None and not schedule:
            schedule = None

    base_temperature = float(getattr(wx_cfg, "temperature", 0.0) or 0.0)
    temp_increment = float(getattr(wx_cfg, "temperature_increment_on_fallback", 0.2) or 0.0)
    max_temperature = 2.0
    resolved_temperatures: list[float] = []
    if schedule:
        resolved_temperatures = [t for t in schedule if 0.0 <= float(t) <= max_temperature]
    if not resolved_temperatures:
        resolved_temperatures = [base_temperature]
        if temp_increment > 0.0:
            temp_cursor = base_temperature + temp_increment
            steps = 0
            while temp_cursor <= max_temperature and steps < 10:
                resolved_temperatures.append(round(temp_cursor, 3))
                temp_cursor += temp_increment
                steps += 1
    resolved_temperatures = [max(0.0, min(max_temperature, float(t))) for t in resolved_temperatures]
    # Remove duplicates while preserving order
    unique_temps: list[float] = []
    for t in resolved_temperatures:
        if not unique_temps or abs(unique_temps[-1] - t) > 1e-6:
            unique_temps.append(t)
    resolved_temperatures = unique_temps or [base_temperature]

    quality_thresholds = {
        "compression_ratio": (
            float(wx_cfg.compression_ratio_threshold)
            if getattr(wx_cfg, "compression_ratio_threshold", None) is not None
            else None
        ),
        "logprob": (
            float(wx_cfg.logprob_threshold)
            if getattr(wx_cfg, "logprob_threshold", None) is not None
            else None
        ),
        "no_speech": (
            float(wx_cfg.no_speech_threshold)
            if getattr(wx_cfg, "no_speech_threshold", None) is not None
            else None
        ),
    }

    quality_rejects: list[dict[str, Any]] = []

    max_chunk_duration = getattr(wx_cfg, "max_chunk_duration_s", None)
    if max_chunk_duration is not None:
        try:
            max_chunk_duration = float(max_chunk_duration)
        except (TypeError, ValueError):
            max_chunk_duration = None
    if not max_chunk_duration or max_chunk_duration <= 0:
        try:
            total_limit = float(getattr(cfg.asr, "total_batch_duration_s", 0) or 0)
        except (TypeError, ValueError):
            total_limit = 0.0
        max_chunk_duration = total_limit if total_limit > 0 else None

    def _should_drop_segment(
        text_value: str,
        seg_obj: Any,
        chunk_start_time: float,
        rel_start_time: float,
        rel_end_time: float,
    ) -> bool:
        reasons: list[str] = []
        compression_ratio = getattr(seg_obj, "compression_ratio", None)
        avg_logprob = getattr(seg_obj, "avg_logprob", None)
        no_speech_prob = getattr(seg_obj, "no_speech_prob", None)

        if isinstance(compression_ratio, (int, float)) and not math.isfinite(compression_ratio):
            compression_ratio = None
        if isinstance(avg_logprob, (int, float)) and not math.isfinite(avg_logprob):
            avg_logprob = None
        if isinstance(no_speech_prob, (int, float)) and not math.isfinite(no_speech_prob):
            no_speech_prob = None

        cr_th = quality_thresholds.get("compression_ratio")
        if cr_th is not None and compression_ratio is not None and compression_ratio > cr_th:
            reasons.append("compression_ratio")

        log_th = quality_thresholds.get("logprob")
        if log_th is not None and avg_logprob is not None and avg_logprob < log_th:
            reasons.append("avg_logprob")

        ns_th = quality_thresholds.get("no_speech")
        if ns_th is not None and no_speech_prob is not None:
            if log_th is None:
                if no_speech_prob > ns_th and not text_value.strip():
                    reasons.append("no_speech")
            else:
                if avg_logprob is not None and avg_logprob < log_th and no_speech_prob > ns_th:
                    reasons.append("no_speech")

        if reasons:
            quality_rejects.append(
                {
                    "start": chunk_start_time + rel_start_time,
                    "end": chunk_start_time + rel_end_time,
                    "text": text_value,
                    "reasons": reasons,
                    "avg_logprob": avg_logprob,
                    "no_speech_prob": no_speech_prob,
                    "compression_ratio": compression_ratio,
                }
            )
            return True
        return False

    temperature_argument: float | list[float]
    if len(resolved_temperatures) == 1:
        temperature_argument = resolved_temperatures[0]
    else:
        temperature_argument = resolved_temperatures

    mel_bins = ct2_n_mels or model.feature_extractor.mel_filters.shape[0]

    for window in windows:
        start_s = max(0.0, float(window.get("start", 0.0)))
        end_s = min(float(duration_s), float(window.get("end", start_s)))
        if end_s <= start_s:
            continue

        chunk_start = start_s
        while chunk_start < end_s:
            chunk_end = end_s
            if max_chunk_duration and max_chunk_duration > 0:
                chunk_end = min(end_s, chunk_start + max_chunk_duration)

            start_idx = max(0, int(chunk_start * sr))
            end_idx = min(len(samples), int(chunk_end * sr))
            if end_idx <= start_idx:
                chunk_start = chunk_end
                continue

            chunk_audio = samples[start_idx:end_idx]
            if not np.any(chunk_audio):
                chunk_start = chunk_end
                continue

            segments_iter, info = model.transcribe(
                chunk_audio,
                language=language_override or None,
                beam_size=beam_size,
                vad_filter=False,
                word_timestamps=False,
                temperature=temperature_argument,
                compression_ratio_threshold=None,
                log_prob_threshold=None,
                no_speech_threshold=None,
            )
            if info and getattr(info, "language", None) and not detected_language:
                detected_language = info.language

            for seg in segments_iter:
                text = str(getattr(seg, "text", "")).strip()
                if not text:
                    continue
                rel_start = max(0.0, float(getattr(seg, "start", 0.0) or 0.0))
                rel_end = float(getattr(seg, "end", rel_start) or rel_start)
                if not math.isfinite(rel_end) or rel_end < rel_start:
                    rel_end = rel_start

                if _should_drop_segment(text, seg, chunk_start, rel_start, rel_end):
                    continue

                seg_start = chunk_start + rel_start
                seg_end = chunk_start + rel_end
                segment_dict = {
                    "start": float(seg_start),
                    "end": float(seg_end),
                    "text_raw": text,
                    "text_punct": text,
                }
                segments.append(segment_dict)
                transcript_parts.append(text)

            chunk_start = chunk_end

    del model  # free resources

    segments.sort(key=lambda s: s.get("start", 0.0))
    transcript = " ".join(transcript_parts).strip()

    segments, word_segments, alignment_meta = _apply_alignment(
        segments,
        cfg,
        audio_path,
        detected_language,
    )

    metadata: dict[str, Any] = {
        "asr_model": f"whisperx: {model_path}",
        "params": {
            "device": device,
            "compute_type": compute_type,
            "language": language_override or detected_language or "auto",
            "beam_size": beam_size,
            "temperature": base_temperature,
            "temperature_schedule": schedule,
            "temperature_increment_on_fallback": temp_increment,
            "temperature_resolved": resolved_temperatures,
            "max_chunk_duration_s": max_chunk_duration,
            "vad_method": vad_metadata.get("method"),
            "vad_onset": getattr(getattr(cfg, "vad", None), "vad_onset", 0.5),
            "vad_offset": getattr(getattr(cfg, "vad", None), "vad_offset", 0.363),
            "mel_bins": mel_bins,
        },
        "vad": vad_metadata,
        "quality_filters": {
            "thresholds": quality_thresholds,
            "dropped_segments": quality_rejects,
        },
    }
    if detected_language:
        metadata["language"] = detected_language
    if alignment_meta:
        metadata["alignment"] = alignment_meta

    result: dict[str, Any] = {
        "audio": {
            "sr": sr,
            "path": str(audio_path),
            "duration_s": duration_s,
        },
        "segments": segments,
        "transcript_raw": transcript,
        "transcript_punct": transcript,
        "metadata": metadata,
    }
    if word_segments:
        result["word_segments"] = word_segments

    return result

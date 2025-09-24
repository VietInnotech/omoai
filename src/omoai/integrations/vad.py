from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

"""
Lightweight VAD utilities for optional pre-segmentation.

Current implementation:
- webrtc: uses `webrtcvad` if installed; otherwise falls back to [] and lets caller skip VAD.
- silero: placeholder; returns [] unless dependencies are available (kept minimal by design).

All functions are best-effort and must never raise; callers can fall back to full-audio decode.
"""

VADMethod = Literal["webrtc", "silero", "pyannote"]


@dataclass
class VADParams:
    method: VADMethod = "webrtc"
    sample_rate: int = 16000
    vad_onset: float = 0.5
    vad_offset: float = 0.363
    min_speech_s: float = 0.30
    min_silence_s: float = 0.30
    chunk_size: float = 30.0
    overlap_s: float = 0.4
    device: str = "cpu"


def _read_pcm16_mono_wav(path: str | Path) -> tuple[bytes, int]:
    import wave

    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
        if nch != 1 or sampwidth != 2 or sr not in (8000, 16000, 32000, 48000):
            # Caller is expected to pass preprocessed 16k mono PCM16 WAV.
            # We still return raw bytes for best-effort VAD; WebRTC requires 8/16/32/48k mono 16-bit.
            pass
        return raw, sr


def detect_speech(
    audio_path: str | Path,
    *,
    method: VADMethod = "webrtc",
    sample_rate: int = 16000,
    vad_onset: float = 0.5,
    vad_offset: float = 0.363,  # unused for webrtc
    min_speech_s: float = 0.30,
    min_silence_s: float = 0.30,
    chunk_size: float = 30.0,
    # webrtc specifics
    webrtc_mode: int = 2,
    frame_ms: int = 20,
    # silero specifics
    speech_pad_ms: int = 30,
    window_size_samples: int = 512,
    device: str = "cpu",
) -> list[tuple[float, float]]:
    """Return list of (start_s, end_s) speech intervals. Never raises."""
    try:
        if method == "webrtc":
            try:
                import webrtcvad  # type: ignore
            except ImportError:
                return []

            raw, sr = _read_pcm16_mono_wav(audio_path)
            if sr != sample_rate:
                # WebRTC VAD is tolerant to 8/16/32/48k; prefer the given sample_rate but continue.
                sample_rate = sr

            # 20 ms frame size in samples/bytes
            bytes_per_sample = 2
            frame_bytes = int(sample_rate * frame_ms / 1000) * bytes_per_sample
            if frame_bytes <= 0:
                return []

            try:
                vad = webrtcvad.Vad(int(webrtc_mode))
            except (ValueError, TypeError):
                vad = webrtcvad.Vad(2)
            frames = [raw[i : i + frame_bytes] for i in range(0, len(raw), frame_bytes)]
            # Drop last partial frame
            if frames and len(frames[-1]) != frame_bytes:
                frames = frames[:-1]

            flags = [vad.is_speech(f, sample_rate) for f in frames]

            # Group contiguous frames with smoothing
            intervals: list[tuple[float, float]] = []
            in_speech = False
            start_idx = 0
            for i, flag in enumerate(flags):
                if flag and not in_speech:
                    in_speech = True
                    start_idx = i
                elif not flag and in_speech:
                    in_speech = False
                    end_idx = i
                    s = start_idx * frame_ms / 1000.0
                    e = end_idx * frame_ms / 1000.0
                    intervals.append((s, e))
            if in_speech:
                e = len(flags) * frame_ms / 1000.0
                s = start_idx * frame_ms / 1000.0
                intervals.append((s, e))

            # Remove short speech
            intervals = [(s, e) for (s, e) in intervals if (e - s) >= float(min_speech_s)]

            # Merge gaps shorter than min_silence_s
            merged: list[tuple[float, float]] = []
            for seg in intervals:
                if not merged:
                    merged.append(seg)
                    continue
                ps, pe = merged[-1]
                s, e = seg
                if s - pe <= float(min_silence_s):
                    merged[-1] = (ps, e)
                else:
                    merged.append(seg)
            return merged

        if method == "silero":
            try:
                import torch  # type: ignore
            except ImportError:
                return []

            # Prefer PyPI package API if available
            ts = None
            try:
                from silero_vad import (
                    get_speech_timestamps,
                    load_silero_vad,
                    read_audio,
                )  # type: ignore

                model = load_silero_vad(onnx=False)
                wav = read_audio(str(audio_path), sampling_rate=sample_rate)
                ts = get_speech_timestamps(
                    wav,
                    model,
                    sampling_rate=sample_rate,
                    threshold=float(vad_onset),
                    min_speech_duration_ms=int(max(0.0, float(min_speech_s)) * 1000),
                    min_silence_duration_ms=int(max(0.0, float(min_silence_s)) * 1000),
                    max_speech_duration_s=float(max(1.0, float(chunk_size))),
                    speech_pad_ms=int(max(0, int(speech_pad_ms))),
                    window_size_samples=int(max(128, int(window_size_samples))),
                )
            except (ImportError, RuntimeError):
                # Fallback to torch.hub loader (same repo)
                try:
                    vad_model, vad_utils = torch.hub.load(
                        repo_or_dir="snakers4/silero-vad",
                        model="silero_vad",
                        force_reload=False,
                        onnx=False,
                        trust_repo=True,
                    )
                    get_speech_timestamps = vad_utils[0]
                    # Load raw PCM16 and convert to float tensor in [-1, 1]
                    raw, sr = _read_pcm16_mono_wav(audio_path)
                    import array

                    arr = array.array("h")
                    arr.frombytes(raw)
                    if sr != sample_rate:
                        sample_rate = sr
                    if len(arr) == 0:
                        return []
                    waveform = torch.tensor(arr, dtype=torch.float32) / 32768.0

                    ts = get_speech_timestamps(
                        waveform,
                        model=vad_model,
                        sampling_rate=sample_rate,
                        threshold=float(vad_onset),
                        min_speech_duration_ms=int(max(0.0, float(min_speech_s)) * 1000),
                        min_silence_duration_ms=int(max(0.0, float(min_silence_s)) * 1000),
                        max_speech_duration_s=float(max(1.0, float(chunk_size))),
                        speech_pad_ms=int(max(0, int(speech_pad_ms))),
                    )
                except (ImportError, RuntimeError):
                    ts = None

            out: list[tuple[float, float]] = []
            for t in (ts or []):
                try:
                    s = float(t.get("start", 0)) / float(sample_rate)
                    e = float(t.get("end", 0)) / float(sample_rate)
                except (ValueError, TypeError, KeyError):
                    continue
                if e > s:
                    out.append((s, e))
            # Simple smoothing consistent with method-agnostic behavior
            if not out:
                return []
            merged: list[tuple[float, float]] = []
            for seg in out:
                if not merged:
                    merged.append(seg)
                    continue
                ps, pe = merged[-1]
                s, e = seg
                if s - pe <= float(min_silence_s):
                    merged[-1] = (ps, e)
                else:
                    merged.append(seg)
            # Drop micro segments
            merged = [(s, e) for (s, e) in merged if (e - s) >= float(min_speech_s)]
            return merged

        # pyannote not implemented here to avoid heavy dependencies
        return []
    except (OSError, ValueError, TypeError):
        return []


def merge_chunks(
    segments: Iterable[tuple[float, float]],
    *,
    chunk_size: float,
) -> list[dict]:
    """WhisperX-style merge into bounded windows with segment lists.

    Returns a list of dicts: {"start": float, "end": float, "segments": list[(s,e)]}
    """
    segments_list = list(segments)
    if not segments_list:
        return []
    merged: list[dict] = []
    curr_start = segments_list[0][0]
    curr_end = segments_list[0][1]
    curr_segs: list[tuple[float, float]] = [(curr_start, curr_end)]
    for s, e in segments_list[1:]:
        # If adding this seg exceeds chunk_size and current has positive length, flush
        if (e - curr_start) > float(chunk_size) and (curr_end - curr_start) > 0:
            merged.append({"start": curr_start, "end": curr_end, "segments": curr_segs})
            curr_start = s
            curr_end = e
            curr_segs = [(s, e)]
        else:
            curr_end = e
            curr_segs.append((s, e))
    merged.append({"start": curr_start, "end": curr_end, "segments": curr_segs})
    return merged


def apply_overlap(
    windows: list[dict], *, overlap_s: float, audio_duration: float
) -> list[dict]:
    if not windows:
        return []
    if overlap_s <= 0:
        return windows
    out: list[dict] = []
    n = len(windows)
    for i, w in enumerate(windows):
        s = w["start"] - overlap_s
        e = w["end"] + overlap_s
        if i == 0:
            s = max(0.0, s)
        else:
            s = max(windows[i - 1]["end"], s)
        if i == n - 1:
            e = min(audio_duration, e)
        else:
            e = min(windows[i + 1]["start"], e)
        out.append({"start": s, "end": e, "segments": w.get("segments", [])})
    return out

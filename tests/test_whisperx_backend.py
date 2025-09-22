import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.asr_whisperx as asr_whisperx


def test_whisperx_runner_converts_output(monkeypatch, tmp_path):
    audio_path = tmp_path / "sample.wav"
    import wave
    import struct

    with wave.open(str(audio_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<h", 0))
    out_path = tmp_path / "asr.json"

    dummy_cfg = SimpleNamespace(asr=SimpleNamespace(engine="whisperx"))
    monkeypatch.setattr(asr_whisperx, "load_config", lambda path: dummy_cfg)

    fake_output = {
        "audio": {
            "sr": 16000,
            "path": str(audio_path),
            "duration_s": 1.23,
        },
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text_raw": "Hello WhisperX.",
                "text_punct": "Hello WhisperX.",
            },
        ],
        "transcript_raw": "Hello WhisperX.",
        "transcript_punct": "Hello WhisperX.",
        "metadata": {
            "asr_model": "whisperx: models/whisperx/EraX",
        },
    }

    monkeypatch.setattr(asr_whisperx, "run_whisperx", lambda audio_path, cfg: fake_output)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "asr_whisperx.py",
            "--audio",
            str(audio_path),
            "--out",
            str(out_path),
        ],
    )

    asr_whisperx.main()

    result = json.loads(out_path.read_text(encoding="utf-8"))

    assert result["metadata"]["asr_model"].startswith("whisperx:")
    assert "models/whisperx/EraX" in result["metadata"]["asr_model"]
    assert result["transcript_raw"] == "Hello WhisperX."
    assert result["transcript_punct"] == result["transcript_raw"]
    assert result["segments"][0]["text_punct"] == "Hello WhisperX."

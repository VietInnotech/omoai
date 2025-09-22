"""Shim script that runs WhisperX CLI and converts output to OMOAI ASR schema."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydub import AudioSegment  # type: ignore

from omoai.config.schemas import OmoAIConfig, load_config
from omoai.logging_system.logger import get_logger, setup_logging


setup_logging()
logger = get_logger(__name__)


def _load_config(path: str | None) -> OmoAIConfig:
    cfg_path: str | Path | None = Path(path).resolve() if path else None
    return load_config(cfg_path)


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _construct_command(
    audio_path: Path,
    output_dir: Path,
    cfg: OmoAIConfig,
) -> tuple[list[str], dict[str, str]]:
    wx = cfg.asr.whisperx
    compute_type = wx.compute_type
    valid_cli_compute = {"float16", "float32", "int8"}
    if compute_type not in valid_cli_compute:
        # Fallback to a supported CLI option; int8_float16 is a hybrid mode available via API but not CLI
        compute_type = "float16"
    repo_root = _resolve_repo_root()
    ref_path = repo_root / "Ref" / "whisperX"

    env = os.environ.copy()
    pythonpath_entries = [str(ref_path)]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath_entries.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    cmd = [
        sys.executable,
        "-m",
        "whisperx",
        "--model",
        str(Path(wx.model_path)),
        "--model_cache_only",
        "True",
        "--device",
        wx.device,
        "--compute_type",
        compute_type,
        "--language",
        wx.language,
        "--vad_method",
        wx.vad_method,
        "--vad_onset",
        f"{wx.vad_onset:.3f}",
        "--vad_offset",
        f"{wx.vad_offset:.3f}",
        "--beam_size",
        str(wx.beam_size),
        "--output_format",
        "json",
        "--output_dir",
        str(output_dir),
        "--task",
        "transcribe",
        str(audio_path),
    ]
    return cmd, env


def _load_audio_metadata(audio_path: Path) -> tuple[int, float]:
    audio = AudioSegment.from_file(str(audio_path))
    sr = int(audio.frame_rate)
    duration_s = float(len(audio) / 1000.0)
    return sr, duration_s


def _convert_segments(raw_segments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    segments: list[dict[str, Any]] = []
    running: list[str] = []
    for seg in raw_segments or []:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        segment = {
            "start": start,
            "end": end,
            "text_raw": text,
            "text_punct": text,
        }
        segments.append(segment)
        running.append(text)
    transcript = " ".join(running).strip()
    return segments, transcript


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ASR wrapper for WhisperX (JSON output)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config.yaml",
    )
    parser.add_argument(
        "--audio", type=str, required=True, help="Path to input audio file"
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Path to output JSON file"
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if cfg.asr.engine != "whisperx":
        logger.warning(
            "WhisperX script invoked while asr.engine=%s; continuing anyway",
            cfg.asr.engine,
        )

    audio_path = Path(args.audio).resolve()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    repo_root = _resolve_repo_root()

    with tempfile.TemporaryDirectory(prefix="whisperx-", dir=None) as tmpdir:
        output_dir = Path(tmpdir)
        cmd, env = _construct_command(audio_path, output_dir, cfg)
        logger.info("Running WhisperX command: %s", " ".join(cmd))
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "WhisperX execution failed with return code "
                f"{result.returncode}\nstdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            )

        json_path = output_dir / f"{audio_path.stem}.json"
        if not json_path.exists():
            candidates = list(output_dir.glob("*.json"))
            if len(candidates) == 1:
                json_path = candidates[0]
            else:
                raise FileNotFoundError(
                    f"Unable to locate WhisperX JSON output in {output_dir}"
                )

        with open(json_path, encoding="utf-8") as f:
            whisperx_output = json.load(f)

    raw_segments = whisperx_output.get("segments", []) if isinstance(whisperx_output, dict) else []
    segments, transcript = _convert_segments(raw_segments)

    sr, duration_s = _load_audio_metadata(audio_path)

    metadata = {
        "asr_model": f"whisperx: {Path(cfg.asr.whisperx.model_path)}",
        "params": {
            "device": cfg.asr.whisperx.device,
            "compute_type": cfg.asr.whisperx.compute_type,
            "language": cfg.asr.whisperx.language,
            "beam_size": cfg.asr.whisperx.beam_size,
            "vad_method": cfg.asr.whisperx.vad_method,
            "vad_onset": cfg.asr.whisperx.vad_onset,
            "vad_offset": cfg.asr.whisperx.vad_offset,
        },
    }
    if isinstance(whisperx_output, dict) and whisperx_output.get("language"):
        metadata["language"] = whisperx_output.get("language")

    final_output: dict[str, Any] = {
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

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

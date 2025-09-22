"""WhisperX ASR script using faster-whisper runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omoai.asr.whisperx_runner import run_whisperx
from omoai.config.schemas import load_config
from omoai.logging_system.logger import get_logger, setup_logging


setup_logging()
logger = get_logger(__name__)


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

    cfg_path: Path | None = Path(args.config).resolve() if args.config else None
    cfg = load_config(cfg_path)
    if cfg.asr.engine != "whisperx":
        logger.warning(
            "WhisperX script invoked while asr.engine=%s; continuing anyway",
            cfg.asr.engine,
        )

    audio_path = Path(args.audio).resolve()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_whisperx(audio_path, cfg)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

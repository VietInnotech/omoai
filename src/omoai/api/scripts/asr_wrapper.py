"""ASR script wrapper ensuring correct working directory and command structure.

On non-zero exit codes, raises AudioProcessingException with a detailed message
including return code, stdout and stderr. This aligns wrapper behavior with
tests that call the wrapper directly.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    # Prefer centralized config; fall back gracefully if unavailable
    from omoai.config.schemas import get_config, load_config  # type: ignore
except (ImportError, AttributeError):  # pragma: no cover - defensive import
    get_config = None  # type: ignore
    load_config = None  # type: ignore

from omoai.api.exceptions import AudioProcessingException


def run_asr_script(
    audio_path: str | Path,
    output_path: str | Path,
    config_path: str | Path | None = None,
    timeout_seconds: float | None = None,
) -> None:
    """
    Invoke the top-level ASR script as a module with cwd set to project root.

    Expected command pattern (validated by tests):
      [sys.executable, "-m", "scripts.asr", "--audio", audio_path, "--out", output_path, ("--config", config_path)?]
    """
    audio_path = str(audio_path)
    output_path = str(output_path)
    cfg = None
    engine = "chunkformer"

    if config_path and load_config is not None:
        try:
            cfg = load_config(config_path)
        except Exception:
            cfg = None
    elif get_config is not None:
        try:
            cfg = get_config()
        except Exception:
            cfg = None

    if cfg is not None:
        try:
            engine = getattr(cfg.asr, "engine", "chunkformer")
        except Exception:
            engine = "chunkformer"

    module_name = "scripts.asr_whisperx" if engine == "whisperx" else "scripts.asr"
    cmd = [
        sys.executable,
        "-m",
        module_name,
        "--audio",
        audio_path,
        "--out",
        output_path,
    ]
    if config_path:
        cmd.extend(["--config", str(config_path)])

    # Project root: src/omoai/api/scripts/asr_wrapper.py -> .../src -> project root
    project_root = Path(__file__).resolve().parents[4]

    # Let CalledProcessError propagate to callers for test assertions
    # Optional: stream child output directly to this terminal based on config.yaml
    stream = False
    try:
        if get_config is not None:
            cfg = get_config()
            stream = bool(getattr(cfg.api, "stream_subprocess_output", False))
    except (AttributeError, TypeError):
        stream = False
    env = os.environ.copy()
    if config_path:
        env["OMOAI_CONFIG"] = str(config_path)
    if stream:
        # Encourage unbuffered output from the child
        env.setdefault("PYTHONUNBUFFERED", "1")
        result = subprocess.run(
            cmd,
            cwd=project_root,
            text=True,
            env=env,
            timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
    else:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None,
        )
    if result.returncode != 0:
        # Provide detailed error context expected by tests
        message = (
            f"ASR processing failed with return code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )
        raise AudioProcessingException(message)

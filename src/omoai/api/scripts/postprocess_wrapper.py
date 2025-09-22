"""Postprocess script wrapper ensuring correct working directory and command structure."""

from __future__ import annotations

import logging
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run_postprocess_script(
    asr_json_path: str | Path,
    output_path: str | Path,
    config_path: str | Path | None = None,
    timeout_seconds: float | None = None,
    timestamped_summary: bool = False,
) -> None:
    """
    Invoke the top-level postprocess script as a module with cwd set to project root.

    Expected command pattern (validated by tests):
      [sys.executable, "-m", "scripts.post", "--asr-json", asr_json_path, "--out", output_path, ("--config", config_path)?]
    """
    # Log diagnostic information for CUDA multiprocessing issue
    # Keep diagnostics lightweight; avoid importing heavy CUDA libraries here
    try:
        logger.info(
            f"Current multiprocessing start method: {multiprocessing.get_start_method()}"
        )
    except Exception:
        logger.info("Current multiprocessing start method: unknown")

    asr_json_path = str(asr_json_path)
    output_path = str(output_path)
    cmd = [
        sys.executable,
        "-m",
        "scripts.post",
        "--asr-json",
        asr_json_path,
        "--out",
        output_path,
    ]
    if config_path:
        cmd.extend(["--config", str(config_path)])

    # Add timestamped_summary flag if requested
    if timestamped_summary:
        cmd.append("--timestamped_summary")

    # Load centralized config to drive runtime toggles from config.yaml
    cfg = None
    try:
        from omoai.config.schemas import get_config  # type: ignore

        cfg = get_config()
    except Exception:
        cfg = None

    if cfg is not None:
        try:
            if getattr(cfg.api, "enable_progress_output", False):
                cmd.append("--progress")
            # Verbose: via explicit toggle or debug logging mode
            want_verbose = bool(getattr(cfg.api, "verbose_scripts", False))
            try:
                log_cfg = getattr(cfg, "logging", None)
                if log_cfg and getattr(log_cfg, "debug_mode", False):
                    want_verbose = True
                elif log_cfg and str(getattr(log_cfg, "level", "")).upper() == "DEBUG":
                    want_verbose = True
            except Exception as err:
                logger.debug("postprocess_wrapper: unable to read logging config", exc_info=err)
            if want_verbose:
                cmd.append("--verbose")
        except Exception as err:
            logger.debug("postprocess_wrapper: unable to apply api verbosity flags", exc_info=err)

    # Project root: src/omoai/api/scripts/postprocess_wrapper.py -> .../src -> project root
    project_root = Path(__file__).resolve().parents[4]

    # Set environment variable to force spawn method for CUDA compatibility
    env = os.environ.copy()
    env["MULTIPROCESSING_START_METHOD"] = "spawn"

    # Add CUDA isolation to prevent re-initialization issues
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"  # vLLM multiprocessing method
    env["TOKENIZERS_PARALLELISM"] = "false"  # Reduce tokenizer thread overhead

    # Startup-time win: ensure HF cache on fast local disk to avoid repeated downloads
    # Prefer user-provided env; otherwise default to project_root/models/hf-cache
    try:
        default_hf_cache = project_root / "models" / "hf-cache"
        default_hf_cache.mkdir(parents=True, exist_ok=True)
        env.setdefault("HF_HOME", str(default_hf_cache))
        env.setdefault("TRANSFORMERS_CACHE", str(default_hf_cache))
    except Exception as err:
        logger.debug("postprocess_wrapper: unable to set default HF cache", exc_info=err)

    # PyTorch CUDA allocator tuning (per PyTorch docs) to reduce fragmentation and
    # speed up pinned memory operations during per-request launches
    # Reference: PYTORCH_CUDA_ALLOC_CONF options (expandable_segments, gc threshold, pinned_*).
    if "PYTORCH_CUDA_ALLOC_CONF" not in env:
        env["PYTORCH_CUDA_ALLOC_CONF"] = (
            "expandable_segments:True,"
            "garbage_collection_threshold:0.8,"
            "pinned_use_cuda_host_register:True,"
            "pinned_num_register_threads:8,"
            "pinned_use_background_threads:True"
        )

    logger.info(f"Running postprocess command: {' '.join(cmd)}")
    logger.info(f"Working directory: {project_root}")
    logger.info(
        f"CUDA environment: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', 'not set')}"
    )

    # Optional: stream child output directly to this terminal based on config.yaml
    want_stream = False
    try:
        if cfg is not None:
            want_stream = bool(getattr(cfg.api, "stream_subprocess_output", False))
    except Exception:
        want_stream = False
    if want_stream:
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
        raise RuntimeError(
            f"Postprocess script failed with return code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
        )

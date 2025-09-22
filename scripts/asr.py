import argparse
import json
import os
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

import torch

from omoai.logging_system.logger import get_logger, setup_logging

# Initialize unified logging once for scripts
setup_logging()

# Environment flag for debug GPU memory clearing
DEBUG_EMPTY_CACHE = os.environ.get("OMOAI_DEBUG_EMPTY_CACHE", "false").lower() == "true"


def ensure_chunkformer_on_path(chunkformer_dir: Path) -> None:
    logger = get_logger(__name__)

    logger.info(f"chunkformer_dir: {chunkformer_dir}")
    logger.info(f"chunkformer_dir.exists(): {chunkformer_dir.exists()}")

    # Add the chunkformer directory to path if it exists
    if chunkformer_dir.exists() and str(chunkformer_dir) not in sys.path:
        sys.path.insert(0, str(chunkformer_dir))
        logger.info(f"Added chunkformer_dir to sys.path: {chunkformer_dir}")

    # Derive repository root from the chunkformer directory (repo_root/chunkformer)
    repo_root = chunkformer_dir.parent
    src_dir = repo_root / "src"
    logger.info(f"src_dir: {src_dir}")
    logger.info(f"src_dir.exists(): {src_dir.exists()}")

    # Tests expect the repo's src path to be added to sys.path for imports.
    # Add it unconditionally if not already present so imports can be resolved
    # in test environments where the path may not physically exist.
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
        logger.info(f"Added src_dir to sys.path: {src_dir} (exists={src_dir.exists()})")

    logger.info(f"sys.path: {sys.path}")


def run_asr(
    audio_path: Path,
    model_checkpoint: Path,
    out_path: Path,
    total_batch_duration: int,
    chunk_size: int,
    left_context_size: int,
    right_context_size: int,
    device_str: str,
    autocast_dtype: str | None,
    chunkformer_dir: Path,
) -> None:
    logger = get_logger(__name__)

    logger.info(f"Starting ASR processing for audio: {audio_path}")
    logger.info(f"Model checkpoint path: {model_checkpoint}")
    logger.info(f"Output path: {out_path}")
    logger.info(f"Device: {device_str}")
    logger.info(f"Chunkformer directory: {chunkformer_dir}")

    # Check if model checkpoint exists
    if not model_checkpoint.exists():
        logger.error(f"Model checkpoint does not exist: {model_checkpoint}")
        raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint}")

    logger.info(
        f"Model checkpoint exists, size: {model_checkpoint.stat().st_size} bytes"
    )

    ensure_chunkformer_on_path(chunkformer_dir)

    # Local imports after sys.path adjustment
    try:
        from chunkformer import decode as cfdecode  # type: ignore

        logger.info("Successfully imported chunkformer decode module")
    except ImportError as e:
        logger.error(f"Failed to import chunkformer decode module: {e}")
        raise

    try:
        import torchaudio.compliance.kaldi as kaldi  # type: ignore

        logger.info("Successfully imported torchaudio kaldi")
    except ImportError as e:
        logger.error(f"Failed to import torchaudio kaldi: {e}")
        raise

    try:
        from chunkformer.model.utils.ctc_utils import (
            get_output_with_timestamps,
        )  # type: ignore

        logger.info("Successfully imported ctc_utils")
    except ImportError as e:
        logger.error(f"Failed to import ctc_utils: {e}")
        raise

    try:
        from pydub import AudioSegment  # type: ignore

        logger.info("Successfully imported pydub AudioSegment")
    except ImportError as e:
        logger.error(f"Failed to import pydub: {e}")
        raise

    # Check CUDA availability
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")

    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    dtype_map = {
        None: None,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    amp_dtype = dtype_map.get(autocast_dtype, None)
    logger.info(f"AMP dtype: {amp_dtype}")

    # Initialize model and char dict
    logger.info("Initializing model and character dictionary...")
    try:
        model, char_dict = cfdecode.init(str(model_checkpoint), device)
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

    # Compute internal parameters in the same way as decode.py
    subsampling_factor = model.encoder.embed.subsampling_factor
    conv_lorder = model.encoder.cnn_module_kernel // 2

    # Maximum duration (seconds) the GPU can handle in one batch
    max_length_limited_context = total_batch_duration
    max_length_limited_context = int(max_length_limited_context // 0.01) // 2  # in 10ms

    multiply_n = max_length_limited_context // chunk_size // subsampling_factor
    truncated_context_size = chunk_size * multiply_n

    # Relative right context in frames
    def get_max_input_context(c: int, r: int, n: int) -> int:
        return r + max(c, r) * (n - 1)

    rel_right_context_size = get_max_input_context(
        chunk_size,
        max(right_context_size, conv_lorder),
        model.encoder.num_blocks,
    )
    rel_right_context_size = rel_right_context_size * subsampling_factor

    # Load and standardize audio to 16kHz mono PCM16
    audio = AudioSegment.from_file(str(audio_path))
    audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)
    audio_duration_s: float = len(audio) / 1000.0
    waveform_full = torch.as_tensor(
        audio.get_array_of_samples(), dtype=torch.float32
    ).unsqueeze(0)

    def _sec_to_hhmmssms(seconds: float) -> str:
        from chunkformer.model.utils.ctc_utils import milliseconds_to_hhmmssms as _msfmt

        ms = int(max(0.0, float(seconds)) * 1000.0)
        return _msfmt(ms)

    def _decode_waveform_slice(waveform_slice: torch.Tensor) -> list[dict[str, Any]]:
        # Extract log-mel filterbank features (Kaldi fbank) like decode.py
        xs_local = kaldi.fbank(
            waveform_slice,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            energy_floor=0.0,
            sample_frequency=16000,
        ).unsqueeze(0)

        # Prepare caches per window
        offset_l = torch.zeros(1, dtype=torch.int, device=device)
        att_cache_l = torch.zeros(
            (
                model.encoder.num_blocks,
                left_context_size,
                model.encoder.attention_heads,
                model.encoder._output_size * 2 // model.encoder.attention_heads,
            ),
        ).to(device)
        cnn_cache_l = torch.zeros(
            (model.encoder.num_blocks, model.encoder._output_size, conv_lorder),
        ).to(device)

        hyps_local: list[torch.Tensor] = []
        ctx = torch.autocast(device.type, dtype=amp_dtype, enabled=(amp_dtype is not None))
        with torch.inference_mode(), ctx:
            for idx, _ in enumerate(
                range(0, xs_local.shape[1], truncated_context_size * subsampling_factor)
            ):
                start = max(truncated_context_size * subsampling_factor * idx, 0)
                end = min(
                    truncated_context_size * subsampling_factor * (idx + 1) + 7,
                    xs_local.shape[1],
                )

                x = xs_local[:, start : end + rel_right_context_size]
                x_len = torch.tensor([x[0].shape[0]], dtype=torch.int).to(device)

                (
                    encoder_outs,
                    encoder_lens,
                    _,
                    att_cache_l,
                    cnn_cache_l,
                    offset_l,
                ) = model.encoder.forward_parallel_chunk(
                    xs=x,
                    xs_origin_lens=x_len,
                    chunk_size=chunk_size,
                    left_context_size=left_context_size,
                    right_context_size=right_context_size,
                    att_cache=att_cache_l,
                    cnn_cache=cnn_cache_l,
                    truncated_context_size=truncated_context_size,
                    offset=offset_l,
                )

                encoder_outs = encoder_outs.reshape(1, -1, encoder_outs.shape[-1])[
                    :, :encoder_lens
                ]
                if (
                    chunk_size * multiply_n * subsampling_factor * idx
                    + rel_right_context_size
                    < xs_local.shape[1]
                ):
                    # exclude the output of relative right context
                    encoder_outs = encoder_outs[:, :truncated_context_size]

                offset_l = offset_l - encoder_lens + encoder_outs.shape[1]

                hyp = model.encoder.ctc_forward(encoder_outs).squeeze(0)
                hyps_local.append(hyp)
                if DEBUG_EMPTY_CACHE and device.type == "cuda":
                    torch.cuda.empty_cache()
                if (
                    chunk_size * multiply_n * subsampling_factor * idx
                    + rel_right_context_size
                    >= xs_local.shape[1]
                ):
                    break

        if not hyps_local:
            return []
        hyps_cat = torch.cat(hyps_local)
        decode = get_output_with_timestamps([hyps_cat], char_dict)[0]
        return [
            {"start": item["start"], "end": item["end"], "text_raw": item["decode"]}
            for item in decode
        ]

    # Optional VAD pre-segmentation
    # Lazy imports for config and utils to avoid early sys.path issues
    from omoai.config.schemas import get_config  # type: ignore
    from omoai.integrations.vad import (  # type: ignore
        apply_overlap,
        detect_speech,
        merge_chunks,
    )
    from omoai.pipeline.postprocess_core_utils import (
        _parse_time_to_seconds as _parse_time_to_seconds,  # type: ignore
    )
    cfg = get_config()
    use_vad = False
    windows: list[dict] = []
    try:
        vcfg = getattr(cfg, "vad", None)
        if vcfg and bool(getattr(vcfg, "enabled", False)):
            use_vad = True
            intervals = detect_speech(
                audio_path,
                method=getattr(vcfg, "method", "webrtc"),
                sample_rate=16000,
                vad_onset=float(getattr(vcfg, "vad_onset", 0.5)),
                vad_offset=float(getattr(vcfg, "vad_offset", 0.363)),
                min_speech_s=float(getattr(vcfg, "min_speech_s", 0.3)),
                min_silence_s=float(getattr(vcfg, "min_silence_s", 0.3)),
                chunk_size=float(getattr(vcfg, "chunk_size", 30.0)),
                # webrtc
                webrtc_mode=int(getattr(getattr(vcfg, "webrtc", None), "mode", 2)),
                frame_ms=int(getattr(getattr(vcfg, "webrtc", None), "frame_ms", 20)),
                # silero
                speech_pad_ms=int(getattr(getattr(vcfg, "silero", None), "speech_pad_ms", 30)),
                window_size_samples=int(getattr(getattr(vcfg, "silero", None), "window_size_samples", 512)),
                device=str(getattr(vcfg, "device", "cpu")),
            )
            if intervals:
                windows = merge_chunks(intervals, chunk_size=float(getattr(vcfg, "chunk_size", 30.0)))
                windows = apply_overlap(
                    windows,
                    overlap_s=float(getattr(vcfg, "overlap_s", 0.4)),
                    audio_duration=audio_duration_s,
                )
            else:
                use_vad = False
    except (ImportError, AttributeError, ValueError) as e:
        logger.debug(f"VAD setup failed: {e}")
        use_vad = False
        windows = []

    segments: list[dict[str, Any]] = []
    transcript_raw = ""

    if use_vad and windows:
        try:
            speech_seconds = sum(
                max(0.0, float(w.get("end", 0.0)) - float(w.get("start", 0.0))) for w in windows
            )
            ratio = speech_seconds / max(1e-6, float(audio_duration_s))
            logger.info(
                f"[VAD] enabled method={getattr(cfg.vad, 'method', 'unknown')} windows={len(windows)} "
                f"speech_ratio={ratio:.3f}"
            )
        except (ValueError, TypeError) as e:
            logger.debug(f"VAD ratio calculation failed: {e}")
        for w in windows:
            ws = float(w.get("start", 0.0) or 0.0)
            we = float(w.get("end", ws) or ws)
            s_idx = int(max(0.0, ws) * 16000.0)
            e_idx = int(min(audio_duration_s, we) * 16000.0)
            if e_idx <= s_idx:
                continue
            wf_slice = waveform_full[:, s_idx:e_idx]
            segs_local = _decode_waveform_slice(wf_slice)
            # Offset local times by window start; keep HH:MM:SS:MS format
            for seg in segs_local:
                start_s = _parse_time_to_seconds(seg.get("start")) or 0.0
                end_s = _parse_time_to_seconds(seg.get("end")) or start_s
                seg["start"] = _sec_to_hhmmssms(ws + start_s)
                seg["end"] = _sec_to_hhmmssms(ws + end_s)
            segments.extend(segs_local)
        transcript_raw = (
            " ".join(seg.get("text_raw", "").strip() for seg in segments if seg.get("text_raw"))
            .replace("  ", " ")
            .strip()
        )
    else:
        # Fallback to single-window full audio path
        segs_local = _decode_waveform_slice(waveform_full)
        segments = segs_local
        transcript_raw = (
            " ".join(seg.get("text_raw", "").strip() for seg in segments if seg.get("text_raw"))
            .replace("  ", " ")
            .strip()
        )

    output: dict[str, Any] = {
        "audio": {
            "sr": 16000,
            "path": str(audio_path.resolve()),
            "duration_s": audio_duration_s,
        },
        "segments": segments,
        "transcript_raw": transcript_raw,
        "metadata": {
            "asr_model": str(model_checkpoint),
            "params": {
                "total_batch_duration": total_batch_duration,
                "chunk_size": chunk_size,
                "left_context_size": left_context_size,
                "right_context_size": right_context_size,
                "autocast_dtype": autocast_dtype or "none",
            },
        },
    }
    # Add VAD metadata when used
    try:
        if use_vad:
            output.setdefault("metadata", {}).setdefault("vad", {})
            output["metadata"]["vad"] = {
                "enabled": True,
                "method": getattr(cfg.vad, "method", "webrtc"),
                "chunk_size": getattr(cfg.vad, "chunk_size", 30.0),
                "overlap_s": getattr(cfg.vad, "overlap_s", 0.4),
                "windows": len(windows or []),
                "speech_ratio": (
                    sum(max(0.0, float(w.get("end", 0.0)) - float(w.get("start", 0.0))) for w in windows)
                    / max(1e-6, float(audio_duration_s))
                ),
            }
    except (ValueError, TypeError) as e:
        logger.debug(f"VAD metadata calculation failed: {e}")

    # Alignment processing
    alignment_device = None
    try:
        if cfg.alignment.enabled:
            logger.info("Starting phonetic alignment processing")

            # Import alignment functions
            from omoai.integrations.alignment import (
                align_segments,
                load_alignment_model,
                merge_alignment_back,
                to_whisperx_segments,
            )

            # Determine alignment language
            alignment_language = cfg.alignment.language
            if alignment_language == "auto":
                # Default to Vietnamese for now, can be extended to auto-detect
                alignment_language = "vi"

            # Load alignment model
            alignment_device = cfg.alignment.device
            if alignment_device == "auto":
                alignment_device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(f"Loading alignment model for language: {alignment_language}, device: {alignment_device}")
            align_model, align_metadata = load_alignment_model(
                language=alignment_language,
                device=alignment_device,
                model_name=cfg.alignment.align_model,
            )

            # Convert segments to alignment format
            logger.info(f"[ALIGNMENT] Converting {len(segments)} segments to whisperx format.")
            wx_segments = to_whisperx_segments(segments)
            logger.info(f"[ALIGNMENT] Converted to {len(wx_segments)} wx_segments: {wx_segments}")
            if not wx_segments:
                logger.warning("No segments to align, skipping alignment")
            else:
                logger.info(f"Aligning {len(wx_segments)} segments")

                # Run alignment
                aligned_result = align_segments(
                    wx_segments=wx_segments,
                    audio_path_or_array=str(audio_path),
                    model=align_model,
                    metadata=align_metadata,
                    device=alignment_device,
                    return_char_alignments=cfg.alignment.return_char_alignments,
                    interpolate_method=cfg.alignment.interpolate_method,
                    print_progress=cfg.alignment.print_progress,
                )
                logger.info(f"[ALIGNMENT] align_segments result: {aligned_result}")

                # Merge alignment results back
                enriched_segments, word_segments = merge_alignment_back(segments, aligned_result)
                logger.info(f"[ALIGNMENT] merge_alignment_back enriched_segments: {enriched_segments}")
                logger.info(f"[ALIGNMENT] merge_alignment_back word_segments: {word_segments}")

                # Update output with enriched segments and word segments
                output["segments"] = enriched_segments
                output["word_segments"] = word_segments

                # Add alignment metadata
                output.setdefault("metadata", {}).setdefault("alignment", {})
                output["metadata"]["alignment"] = {
                    "enabled": True,
                    "language": alignment_language,
                    "model": cfg.alignment.align_model or "default",
                    "device": alignment_device,
                    "return_char_alignments": cfg.alignment.return_char_alignments,
                    "interpolate_method": cfg.alignment.interpolate_method,
                    "segments_aligned": len(enriched_segments),
                    "words_aligned": len(word_segments),
                }

                logger.info(f"Alignment completed: {len(enriched_segments)} segments, {len(word_segments)} words")

    except (ImportError, RuntimeError, ValueError, OSError) as e:
        logger.warning(f"Alignment failed: {e}", exc_info=True)
        # Add failure metadata
        output.setdefault("metadata", {}).setdefault("alignment", {})
        output["metadata"]["alignment"] = {
            "enabled": cfg.alignment.enabled,
            "error": str(e),
            "status": "failed",
        }
    finally:
        # Clean up GPU memory if CUDA was used
        if cfg.alignment.enabled and alignment_device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if DEBUG_EMPTY_CACHE:
                logger.info("Cleared CUDA cache after alignment")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ASR wrapper for ChunkFormer (JSON output)"
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
        "--model-dir",
        type=str,
        required=False,
        help="Override ChunkFormer checkpoint directory (defaults to config.paths.chunkformer_checkpoint)",
    )
    parser.add_argument("--out", type=str, required=True, help="Path to output JSON")
    parser.add_argument(
        "--auto-outdir",
        action="store_true",
        help="Create per-input folder under paths.out_dir/{stem-YYYYMMDD-HHMMSS}",
    )
    parser.add_argument(
        "--total-batch-duration",
        type=int,
        default=None,
        help="Total audio duration per batch in seconds (default from config.asr.total_batch_duration_s)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size (default from config.asr.chunk_size)",
    )
    parser.add_argument(
        "--left-context-size",
        type=int,
        default=None,
        help="Left context size (default from config.asr.left_context_size)",
    )
    parser.add_argument(
        "--right-context-size",
        type=int,
        default=None,
        help="Right context size (default from config.asr.right_context_size)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the model on (defaults to config.asr.device)",
    )
    parser.add_argument(
        "--autocast-dtype",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        default=None,
        help="Autocast dtype (default from config.asr.autocast_dtype)",
    )
    parser.add_argument(
        "--chunkformer-dir",
        type=str,
        default=None,
        help="Path to chunkformer source directory (defaults to config.paths.chunkformer_dir)",
    )

    args = parser.parse_args()

    # Load centralized configuration using the project's Pydantic schemas
    try:
        from omoai.config.schemas import get_config
    except ImportError:
        # If package imports fail (script execution from repo root), add src/ to sys.path and retry
        import sys as _sys
        from pathlib import Path as _Path

        _sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))
        from omoai.config.schemas import get_config

    cfg = get_config()

    # Resolve parameters with precedence: CLI flag -> centralized config (Pydantic defaults applied there)
    model_dir = args.model_dir or str(cfg.paths.chunkformer_checkpoint)
    if not model_dir:
        raise SystemExit(
            "Missing --model-dir and paths.chunkformer_checkpoint in configuration"
        )

    total_batch_duration = (
        args.total_batch_duration
        if args.total_batch_duration is not None
        else cfg.asr.total_batch_duration_s
    )
    chunk_size = (
        args.chunk_size if args.chunk_size is not None else cfg.asr.chunk_size
    )
    left_context_size = (
        args.left_context_size
        if args.left_context_size is not None
        else cfg.asr.left_context_size
    )
    right_context_size = (
        args.right_context_size
        if args.right_context_size is not None
        else cfg.asr.right_context_size
    )
    device_value = args.device if args.device is not None else cfg.asr.device
    if isinstance(device_value, str) and device_value.lower() == "auto":
        device_value = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = (
        args.autocast_dtype
        if args.autocast_dtype is not None
        else cfg.asr.autocast_dtype
    )
    chunkformer_dir = Path(
        args.chunkformer_dir
        if args.chunkformer_dir is not None
        else cfg.paths.chunkformer_dir
    )

    # Auto output dir per input file, if requested
    out_path = Path(args.out)
    if args.auto_outdir:
        from datetime import datetime

        stem = Path(args.audio).stem
        base_root = cfg.paths.out_dir
        # Timestamp-based folder name, UTC for stability
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        candidate = base_root / f"{stem}-{timestamp}"
        # Avoid rare collisions within the same second by suffixing a counter
        if candidate.exists():
            suffix = 1
            while (base_root / f"{stem}-{timestamp}-{suffix:02d}").exists():
                suffix += 1
            candidate = base_root / f"{stem}-{timestamp}-{suffix:02d}"
        candidate.mkdir(parents=True, exist_ok=True)
        base_dir = candidate
        out_path = base_dir / "asr.json"

    run_asr(
        audio_path=Path(args.audio),
        model_checkpoint=Path(model_dir),
        out_path=out_path,
        total_batch_duration=int(total_batch_duration),
        chunk_size=int(chunk_size),
        left_context_size=int(left_context_size),
        right_context_size=int(right_context_size),
        device_str=str(device_value),
        autocast_dtype=autocast_dtype,
        chunkformer_dir=chunkformer_dir,
    )


if __name__ == "__main__":
    main()

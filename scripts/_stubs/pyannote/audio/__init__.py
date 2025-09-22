"""Minimal pyannote.audio stub for WhisperX when pyannote is unavailable."""

from __future__ import annotations


class Model:  # pragma: no cover - stub will only be used when dependency missing
    """Placeholder that raises if instantiated; WhisperX shouldn't call it when using silero VAD."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        raise RuntimeError(
            "pyannote.audio is not installed. Install pyannote.audio or switch WhisperX VAD to a supported backend."
        )


__all__ = ["Model"]

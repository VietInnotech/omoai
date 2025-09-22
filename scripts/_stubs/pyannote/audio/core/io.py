"""Stub module for pyannote.audio.core.io to satisfy WhisperX import side effects."""

from __future__ import annotations


class AudioFile:  # pragma: no cover - stub only used when dependency missing
    def __init__(self, *args, **kwargs) -> None:  # noqa: D401
        raise RuntimeError(
            "pyannote.audio.core.io.AudioFile stub invoked. Install pyannote.audio "
            "or switch WhisperX VAD to silero/other supported backend."
        )


__all__ = ["AudioFile"]

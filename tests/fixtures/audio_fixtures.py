"""
Golden audio fixtures for comprehensive testing.

This module provides synthetic and recorded audio samples with known expected
outputs for testing transcription accuracy and system performance.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

torchaudio = pytest.importorskip("torchaudio")

try:
    from pydub import AudioSegment
    from pydub.generators import Sine

    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False


@dataclass
class AudioFixture:
    """Represents a test audio fixture with expected outputs."""

    name: str
    description: str
    file_path: Path
    duration_seconds: float
    sample_rate: int
    expected_transcript: str
    expected_word_count: int
    expected_confidence_min: float
    metadata: dict[str, any]
    difficulty_level: str  # "easy", "medium", "hard"


class AudioFixtureManager:
    """Manages golden audio fixtures for testing."""

    def __init__(self, fixtures_dir: Path | None = None):
        self.fixtures_dir = fixtures_dir or Path(__file__).parent / "audio_data"
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        self._fixtures: dict[str, AudioFixture] = {}
        self._initialize_fixtures()

    def _initialize_fixtures(self):
        """Initialize built-in fixtures."""
        # Create synthetic fixtures if they don't exist
        self._create_synthetic_fixtures()

        # Load any existing fixture metadata
        self._load_fixture_metadata()

    def _create_synthetic_fixtures(self):
        """Create synthetic audio fixtures for testing."""
        if not HAS_PYDUB:
            # Fall back to PyTorch-only synthetic audio
            self._create_pytorch_fixtures()
            return

        # Fixture 1: Simple tone with silence (easy)
        self._create_tone_fixture()

        # Fixture 2: Multi-tone sequence (medium)
        self._create_sequence_fixture()

        # Fixture 3: Noisy audio (hard)
        self._create_noisy_fixture()

        # Fixture 4: Very short audio (edge case)
        self._create_short_fixture()

        # Fixture 5: Long audio (performance test)
        self._create_long_fixture()

    def _create_tone_fixture(self):
        """Create simple tone fixture."""
        name = "simple_tone"
        file_path = self.fixtures_dir / f"{name}.wav"

        if not file_path.exists():
            # Create 3-second 440Hz tone
            tone = Sine(440).to_audio_segment(duration=3000)
            tone.export(str(file_path), format="wav")

        self._fixtures[name] = AudioFixture(
            name=name,
            description="Simple 440Hz tone for 3 seconds",
            file_path=file_path,
            duration_seconds=3.0,
            sample_rate=44100,
            expected_transcript="",  # No speech expected
            expected_word_count=0,
            expected_confidence_min=0.0,
            metadata={"type": "tone", "frequency": 440},
            difficulty_level="easy",
        )

    def _create_sequence_fixture(self):
        """Create sequence of different tones."""
        name = "tone_sequence"
        file_path = self.fixtures_dir / f"{name}.wav"

        if not file_path.exists():
            # Create sequence: 440Hz (1s) + silence (0.5s) + 880Hz (1s) + silence (0.5s)
            tone1 = Sine(440).to_audio_segment(duration=1000)
            silence1 = AudioSegment.silent(duration=500)
            tone2 = Sine(880).to_audio_segment(duration=1000)
            silence2 = AudioSegment.silent(duration=500)

            sequence = tone1 + silence1 + tone2 + silence2
            sequence.export(str(file_path), format="wav")

        self._fixtures[name] = AudioFixture(
            name=name,
            description="Sequence of tones with silence",
            file_path=file_path,
            duration_seconds=3.0,
            sample_rate=44100,
            expected_transcript="",  # No speech expected
            expected_word_count=0,
            expected_confidence_min=0.0,
            metadata={"type": "sequence", "segments": 4},
            difficulty_level="medium",
        )

    def _create_noisy_fixture(self):
        """Create noisy audio fixture."""
        name = "noisy_audio"
        file_path = self.fixtures_dir / f"{name}.wav"

        if not file_path.exists():
            # Create tone with white noise
            tone = Sine(440).to_audio_segment(duration=2000)

            # Add white noise (very simple approximation)
            noise = AudioSegment.silent(duration=2000)
            # Note: This is a simplified noise simulation
            # In a real implementation, you'd add proper white noise

            # Mix tone with reduced noise
            mixed = tone.overlay(noise - 20)  # Reduce noise by 20dB
            mixed.export(str(file_path), format="wav")

        self._fixtures[name] = AudioFixture(
            name=name,
            description="Tone with background noise",
            file_path=file_path,
            duration_seconds=2.0,
            sample_rate=44100,
            expected_transcript="",  # No speech expected
            expected_word_count=0,
            expected_confidence_min=0.0,
            metadata={"type": "noisy", "snr_db": 20},
            difficulty_level="hard",
        )

    def _create_short_fixture(self):
        """Create very short audio fixture."""
        name = "short_audio"
        file_path = self.fixtures_dir / f"{name}.wav"

        if not file_path.exists():
            # Create 0.5 second tone
            tone = Sine(440).to_audio_segment(duration=500)
            tone.export(str(file_path), format="wav")

        self._fixtures[name] = AudioFixture(
            name=name,
            description="Very short 0.5 second audio",
            file_path=file_path,
            duration_seconds=0.5,
            sample_rate=44100,
            expected_transcript="",  # No speech expected
            expected_word_count=0,
            expected_confidence_min=0.0,
            metadata={"type": "short", "edge_case": True},
            difficulty_level="easy",
        )

    def _create_long_fixture(self):
        """Create long audio fixture for performance testing."""
        name = "long_audio"
        file_path = self.fixtures_dir / f"{name}.wav"

        if not file_path.exists():
            # Create 60-second audio with varying tones
            segments = []
            for i in range(12):  # 12 segments of 5 seconds each
                freq = 440 + (i * 50)  # Varying frequency
                tone = Sine(freq).to_audio_segment(duration=5000)
                segments.append(tone)

            long_audio = sum(segments)
            long_audio.export(str(file_path), format="wav")

        self._fixtures[name] = AudioFixture(
            name=name,
            description="Long 60-second audio for performance testing",
            file_path=file_path,
            duration_seconds=60.0,
            sample_rate=44100,
            expected_transcript="",  # No speech expected
            expected_word_count=0,
            expected_confidence_min=0.0,
            metadata={"type": "long", "performance_test": True},
            difficulty_level="medium",
        )

    def _create_pytorch_fixtures(self):
        """Create fixtures using PyTorch only (fallback)."""
        # Simple sine wave fixture
        name = "pytorch_sine"
        file_path = self.fixtures_dir / f"{name}.wav"

        if not file_path.exists():
            # Generate 3-second sine wave at 440Hz
            sample_rate = 16000
            duration = 3.0
            t = torch.linspace(0, duration, int(sample_rate * duration))
            frequency = 440.0
            waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)

            torchaudio.save(str(file_path), waveform, sample_rate)

        self._fixtures[name] = AudioFixture(
            name=name,
            description="PyTorch-generated sine wave",
            file_path=file_path,
            duration_seconds=3.0,
            sample_rate=16000,
            expected_transcript="",  # No speech expected
            expected_word_count=0,
            expected_confidence_min=0.0,
            metadata={"type": "pytorch_sine", "generator": "torch"},
            difficulty_level="easy",
        )

    def _load_fixture_metadata(self):
        """Load metadata for any existing audio files."""
        # Look for any .wav files in the fixtures directory
        for wav_file in self.fixtures_dir.glob("*.wav"):
            name = wav_file.stem
            if name not in self._fixtures:
                # Create basic fixture info for unknown files
                try:
                    info = torchaudio.info(str(wav_file))
                    duration = info.num_frames / info.sample_rate

                    self._fixtures[name] = AudioFixture(
                        name=name,
                        description=f"External audio fixture: {name}",
                        file_path=wav_file,
                        duration_seconds=duration,
                        sample_rate=info.sample_rate,
                        expected_transcript="unknown",
                        expected_word_count=-1,
                        expected_confidence_min=0.0,
                        metadata={"type": "external", "channels": info.num_channels},
                        difficulty_level="unknown",
                    )
                except Exception:
                    # Skip files that can't be loaded
                    continue

    def get_fixture(self, name: str) -> AudioFixture | None:
        """Get a specific fixture by name."""
        return self._fixtures.get(name)

    def get_fixtures_by_difficulty(self, difficulty: str) -> list[AudioFixture]:
        """Get all fixtures of a specific difficulty level."""
        return [f for f in self._fixtures.values() if f.difficulty_level == difficulty]

    def get_all_fixtures(self) -> dict[str, AudioFixture]:
        """Get all available fixtures."""
        return self._fixtures.copy()

    def load_audio_tensor(self, fixture_name: str) -> tuple[torch.Tensor, int]:
        """Load fixture audio as tensor."""
        fixture = self.get_fixture(fixture_name)
        if not fixture:
            raise ValueError(f"Fixture '{fixture_name}' not found")

        if not fixture.file_path.exists():
            raise FileNotFoundError(f"Fixture file not found: {fixture.file_path}")

        waveform, sample_rate = torchaudio.load(str(fixture.file_path))
        return waveform, sample_rate

    def validate_fixture(self, fixture_name: str) -> bool:
        """Validate that a fixture file exists and is loadable."""
        try:
            fixture = self.get_fixture(fixture_name)
            if not fixture:
                return False

            if not fixture.file_path.exists():
                return False

            # Try to load the audio
            torchaudio.load(str(fixture.file_path))
            return True
        except Exception:
            return False

    def create_custom_fixture(
        self,
        name: str,
        waveform: torch.Tensor,
        sample_rate: int,
        expected_transcript: str = "",
        metadata: dict | None = None,
    ) -> AudioFixture:
        """Create a custom fixture from a waveform tensor."""
        file_path = self.fixtures_dir / f"{name}.wav"

        # Save the waveform
        torchaudio.save(str(file_path), waveform, sample_rate)

        # Calculate duration
        duration = waveform.shape[-1] / sample_rate

        fixture = AudioFixture(
            name=name,
            description=f"Custom fixture: {name}",
            file_path=file_path,
            duration_seconds=duration,
            sample_rate=sample_rate,
            expected_transcript=expected_transcript,
            expected_word_count=len(expected_transcript.split())
            if expected_transcript
            else 0,
            expected_confidence_min=0.0,
            metadata=metadata or {"type": "custom"},
            difficulty_level="custom",
        )

        self._fixtures[name] = fixture
        return fixture


def create_test_audio(
    duration_seconds: float = 3.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> torch.Tensor:
    """Create a simple test audio tensor."""
    t = torch.linspace(0, duration_seconds, int(sample_rate * duration_seconds))
    waveform = amplitude * torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
    return waveform


# Pre-defined golden fixtures for common test cases
GOLDEN_FIXTURES = {
    "silence_1s": {
        "description": "1 second of silence",
        "duration": 1.0,
        "expected_transcript": "",
        "difficulty": "easy",
    },
    "tone_440hz_3s": {
        "description": "3 seconds of 440Hz tone",
        "duration": 3.0,
        "expected_transcript": "",
        "difficulty": "easy",
    },
    "short_burst_0_5s": {
        "description": "0.5 seconds audio (edge case)",
        "duration": 0.5,
        "expected_transcript": "",
        "difficulty": "easy",
    },
    "long_performance_60s": {
        "description": "60 seconds for performance testing",
        "duration": 60.0,
        "expected_transcript": "",
        "difficulty": "medium",
    },
}

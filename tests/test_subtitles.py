import pytest

from scripts import post


def test_normalize_output_formats_aliases():
    formats = post.normalize_output_formats(["text", "SRT", "markdown", "all"])
    assert "txt" in formats
    assert "srt" in formats
    assert "md" in formats
    assert "json" in formats
    assert "vtt" in formats
    assert "tsv" in formats


def test_prepare_subtitle_units_and_rendering():
    segments = [
        {
            "start": 0.0,
            "end": 3.2,
            "text_punct": "Hello world again!",
            "sentences": [
                {
                    "text": "Hello world again!",
                    "start": 0.0,
                    "end": 3.2,
                    "sentence_index": 0,
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.8},
                        {"word": "world", "start": 0.8, "end": 1.5},
                        {"word": "again", "start": 1.5, "end": 2.6},
                    ],
                }
            ],
        }
    ]

    settings = {
        "max_chars_per_line": 12,
        "max_lines": 2,
        "complex_languages": ["zh", "ja", "ko"],
    }

    units = post.prepare_subtitle_units(segments, language="en", settings=settings)
    assert len(units) == 1
    unit = units[0]
    assert pytest.approx(unit["start"], rel=1e-3) == 0.0
    assert pytest.approx(unit["end"], rel=1e-3) == 3.2
    assert any(line["text"].startswith("Hello") for line in unit["lines"])

    srt = post.format_srt(units, highlight=True)
    assert "<b" in srt
    assert "-->" in srt

    vtt = post.format_vtt(units, highlight=False)
    assert vtt.startswith("WEBVTT")
    assert "Hello" in vtt

    tsv = post.format_tsv(units)
    assert "start\tend\ttext" in tsv
    assert "Hello" in tsv

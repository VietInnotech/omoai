#!/usr/bin/env python3
"""
Simple test for the self-contained alignment module (no pytest dependency)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.omoai.integrations.alignment import (
    SingleSegment,
    align,
    load_align_model,
    to_whisperx_segments,
)


def test_alignment_module():
    """Test basic functionality of the alignment module"""
    print("Testing self-contained alignment module...")

    # Test 1: Check imports work
    print("✓ Imports successful")

    # Test 2: Test model loading (using CPU for testing)
    try:
        device = "cpu"
        model, metadata = load_align_model("vi", device, model_dir="/home/cetech/omoai/models")
        print(f"✓ Vietnamese model loaded successfully: {type(model)}")
        print(f"✓ Metadata: {metadata['language']}, {metadata['type']}")
    except Exception as e:
        print(f"✗ Vietnamese model loading failed: {e}")
        return False

    # Test 3: Test segment conversion
    test_segments = [
        {"start": 0.0, "end": 2.0, "text": "xin chào thế giới"},
        {"start": 2.0, "end": 4.0, "text": "đây là một bài kiểm tra"}
    ]

    wx_segments = to_whisperx_segments(test_segments)
    print(f"✓ Segment conversion: {len(wx_segments)} segments")

    # Test 4: Test alignment with dummy audio
    try:
        # Create dummy audio (2 seconds of silence at 16kHz)
        dummy_audio = np.zeros(32000, dtype=np.float32)

        # Convert to proper format
        typed_segments: list[SingleSegment] = []
        for seg in wx_segments:
            typed_segments.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": seg["text"]
            })

        result = align(
            typed_segments,
            model,
            metadata,
            dummy_audio,
            device,
            return_char_alignments=False,
            print_progress=True
        )

        print("✓ Alignment completed successfully")
        print(f"✓ Result contains {len(result['segments'])} aligned segments")
        print(f"✓ Result contains {len(result['word_segments'])} word segments")

        # Show first segment details
        if result['segments']:
            first_seg = result['segments'][0]
            print(f"✓ First segment: '{first_seg['text']}' ({first_seg['start']:.2f}s - {first_seg['end']:.2f}s)")
            print(f"✓ Words in first segment: {len(first_seg['words'])}")

    except Exception as e:
        print(f"✗ Alignment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n🎉 All tests passed! The alignment module is working correctly.")
    return True


def test_english_model():
    """Test English model loading"""
    print("\nTesting English model...")
    try:
        device = "cpu"
        _model, metadata = load_align_model("en", device, model_dir="/home/cetech/omoai/models")
        print(f"✓ English model loaded: {metadata['type']}")
        return True
    except Exception as e:
        print(f"✗ English model test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_alignment_module()
    if success:
        success = test_english_model()

    print(f"\n{'='*50}")
    if success:
        print("🎉 ALL TESTS PASSED! Self-contained alignment module is ready.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

    sys.exit(0 if success else 1)

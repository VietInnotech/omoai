#!/usr/bin/env python3
"""
Test API include filtering for summary and timestamped_summary fields.

These tests verify that the 'include' parameter from config.yaml and query
parameters govern which fields are returned. The structured summary (and its
raw text) follow the include list unless summary=none explicitly disables it.
"""

from unittest.mock import Mock, patch

import pytest

from omoai.api.models import OutputFormatParams, PipelineRequest
from omoai.api.services import _run_full_pipeline_script
from omoai.config.schemas import get_config


class TestAPIIncludeFiltering:
    """Test API include filtering functionality."""

    @pytest.fixture
    def mock_pipeline_request(self):
        """Create a mock pipeline request."""
        mock_request = Mock(spec=PipelineRequest)
        mock_request.audio_file = Mock()
        mock_request.audio_file.read = Mock(return_value=b"fake audio data")
        return mock_request

    @pytest.fixture
    def mock_final_obj(self):
        """Create a mock final object with all fields."""
        return {
            "summary": {
                "title": "Test Title",
                "abstract": "Test abstract",
                "bullets": ["Point 1", "Point 2"],
                "raw": "Raw LLM summary text",
            },
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Segment 1"},
                {"start": 1.0, "end": 2.0, "text": "Segment 2"}
            ],
            "transcript_punct": "This is a punctuated transcript.",
            "transcript_raw": "this is raw transcript",
            "timestamped_summary": {
                "summary_text": "[00:00:30] Introduction\n[00:01:00] Main Topic",
                "timestamps": [
                    {"time": "00:00:30", "text": "Introduction"},
                    {"time": "00:01:00", "text": "Main Topic"}
                ],
                "raw": "Raw timestamped summary"
            },
        }

    @pytest.mark.asyncio
    async def test_include_summary_default(self, mock_pipeline_request, mock_final_obj):
        """Test that summary is included by default when not specified."""
        # Mock the file operations and subprocess calls
        with patch('builtins.open', create=True) as _, \
             patch('json.load', return_value=mock_final_obj), \
             patch('os.urandom', return_value=b'fake_random'), \
             patch('pathlib.Path.mkdir'), \
             patch('os.open'), \
             patch('os.write'), \
             patch('os.close'), \
             patch('shutil.rmtree'), \
             patch('asyncio.to_thread'), \
             patch('omoai.api.services.run_preprocess_script'), \
             patch('omoai.api.services._run_asr_script'), \
             patch('omoai.api.services._run_postprocess_script'):

            # Configure output_params with no include list (should include all by default)
            output_params = OutputFormatParams()

            result, _raw_transcript = await _run_full_pipeline_script(mock_pipeline_request, output_params)

            # Summary should be included by default
            assert result.summary == {
                "title": "Test Title",
                "abstract": "Test abstract",
                "bullets": ["Point 1", "Point 2"]
            }
            assert result.summary.get("raw") is None
            assert result.timestamped_summary == mock_final_obj["timestamped_summary"]

    @pytest.mark.asyncio
    async def test_exclude_summary_from_include_list(self, mock_pipeline_request, mock_final_obj):
        """Summary (and raw text) should be excluded when not in include list."""
        # Mock the file operations and subprocess calls
        with patch('builtins.open', create=True) as _, \
             patch('json.load', return_value=mock_final_obj), \
             patch('os.urandom', return_value=b'fake_random'), \
             patch('pathlib.Path.mkdir'), \
             patch('os.open'), \
             patch('os.write'), \
             patch('os.close'), \
             patch('shutil.rmtree'), \
             patch('asyncio.to_thread'), \
             patch('omoai.api.services.run_preprocess_script'), \
             patch('omoai.api.services._run_asr_script'), \
             patch('omoai.api.services._run_postprocess_script'):

            # Configure output_params to exclude summary
            output_params = OutputFormatParams(
                include=["transcript_punct", "segments", "timestamped_summary"]
            )

            result, _raw_transcript = await _run_full_pipeline_script(mock_pipeline_request, output_params)

            # Summary should be excluded and raw summary suppressed
            assert result.summary == {}
            assert result.summary.get("raw") is None
            # timestamped_summary should be included
            assert result.timestamped_summary == mock_final_obj["timestamped_summary"]

    @pytest.mark.asyncio
    async def test_exclude_timestamped_summary_from_include_list(self, mock_pipeline_request, mock_final_obj):
        """Test that timestamped_summary is excluded when not in include list."""
        # Mock the file operations and subprocess calls
        with patch('builtins.open', create=True) as _, \
             patch('json.load', return_value=mock_final_obj), \
             patch('os.urandom', return_value=b'fake_random'), \
             patch('pathlib.Path.mkdir'), \
             patch('os.open'), \
             patch('os.write'), \
             patch('os.close'), \
             patch('shutil.rmtree'), \
             patch('asyncio.to_thread'), \
             patch('omoai.api.services.run_preprocess_script'), \
             patch('omoai.api.services._run_asr_script'), \
             patch('omoai.api.services._run_postprocess_script'):

            # Configure output_params to exclude timestamped_summary
            output_params = OutputFormatParams(
                include=["transcript_punct", "segments", "summary"]
            )

            result, _raw_transcript = await _run_full_pipeline_script(mock_pipeline_request, output_params)

            # Summary should be included
            assert result.summary == {
                "title": "Test Title",
                "abstract": "Test abstract",
                "bullets": ["Point 1", "Point 2"]
            }
            assert result.summary.get("raw") is None
            # timestamped_summary should be excluded
            assert result.timestamped_summary is None

    @pytest.mark.asyncio
    async def test_summary_fields_filtering(self, mock_pipeline_request, mock_final_obj):
        """Summary fields list should control returned keys."""
        with patch('builtins.open', create=True) as _, \
             patch('json.load', return_value=mock_final_obj), \
             patch('os.urandom', return_value=b'fake_random'), \
             patch('pathlib.Path.mkdir'), \
             patch('os.open'), \
             patch('os.write'), \
             patch('os.close'), \
             patch('shutil.rmtree'), \
             patch('asyncio.to_thread'), \
             patch('omoai.api.services.run_preprocess_script'), \
             patch('omoai.api.services._run_asr_script'), \
             patch('omoai.api.services._run_postprocess_script'):

            output_params = OutputFormatParams(
                include=["summary"],
                return_summary_raw=True,
                summary_fields=["title", "raw"],
            )

            result, _ = await _run_full_pipeline_script(mock_pipeline_request, output_params)

            assert result.summary == {
                "title": "Test Title",
                "raw": "Raw LLM summary text"
            }

    @pytest.mark.asyncio
    async def test_summary_raw_text_omitted_without_summary(self, mock_pipeline_request, mock_final_obj):
        """Raw text should not be returned when summary is excluded."""
        with patch('builtins.open', create=True) as _, \
             patch('json.load', return_value=mock_final_obj), \
             patch('os.urandom', return_value=b'fake_random'), \
             patch('pathlib.Path.mkdir'), \
             patch('os.open'), \
             patch('os.write'), \
             patch('os.close'), \
             patch('shutil.rmtree'), \
             patch('asyncio.to_thread'), \
             patch('omoai.api.services.run_preprocess_script'), \
             patch('omoai.api.services._run_asr_script'), \
             patch('omoai.api.services._run_postprocess_script'):

            output_params = OutputFormatParams(
                include=["transcript_punct", "segments"],
                return_summary_raw=True
            )

            result, _ = await _run_full_pipeline_script(mock_pipeline_request, output_params)

            assert result.summary == {}
            assert result.summary.get("raw") is None

    @pytest.mark.asyncio
    async def test_summary_none_parameter_excludes_summary(self, mock_pipeline_request, mock_final_obj):
        """summary=none should suppress the structured summary even without include filters."""
        with patch('builtins.open', create=True) as _, \
             patch('json.load', return_value=mock_final_obj), \
             patch('os.urandom', return_value=b'fake_random'), \
             patch('pathlib.Path.mkdir'), \
             patch('os.open'), \
             patch('os.write'), \
             patch('os.close'), \
             patch('shutil.rmtree'), \
             patch('asyncio.to_thread'), \
             patch('omoai.api.services.run_preprocess_script'), \
             patch('omoai.api.services._run_asr_script'), \
             patch('omoai.api.services._run_postprocess_script'):

            output_params = OutputFormatParams(summary="none")

            result, _ = await _run_full_pipeline_script(mock_pipeline_request, output_params)

            assert result.summary == {}
            assert result.summary.get("raw") is None

    @pytest.mark.asyncio
    async def test_include_overrides_summary_none(self, mock_pipeline_request, mock_final_obj):
        """Explicit include should restore summary even when summary=none."""
        with patch('builtins.open', create=True) as _, \
             patch('json.load', return_value=mock_final_obj), \
             patch('os.urandom', return_value=b'fake_random'), \
             patch('pathlib.Path.mkdir'), \
             patch('os.open'), \
             patch('os.write'), \
             patch('os.close'), \
             patch('shutil.rmtree'), \
             patch('asyncio.to_thread'), \
             patch('omoai.api.services.run_preprocess_script'), \
             patch('omoai.api.services._run_asr_script'), \
             patch('omoai.api.services._run_postprocess_script'):

            output_params = OutputFormatParams(
                summary="none",
                include=["summary"],
                return_summary_raw=True,
            )

            result, _ = await _run_full_pipeline_script(mock_pipeline_request, output_params)

            assert result.summary == {
                "title": "Test Title",
                "abstract": "Test abstract",
                "bullets": ["Point 1", "Point 2"],
                "raw": "Raw LLM summary text"
            }

    @pytest.mark.asyncio
    async def test_exclude_both_summary_fields(self, mock_pipeline_request, mock_final_obj):
        """Test that both summary and timestamped_summary are excluded when not in include list."""
        # Mock the file operations and subprocess calls
        with patch('builtins.open', create=True) as _, \
             patch('json.load', return_value=mock_final_obj), \
             patch('os.urandom', return_value=b'fake_random'), \
             patch('pathlib.Path.mkdir'), \
             patch('os.open'), \
             patch('os.write'), \
             patch('os.close'), \
             patch('shutil.rmtree'), \
             patch('asyncio.to_thread'), \
             patch('omoai.api.services.run_preprocess_script'), \
             patch('omoai.api.services._run_asr_script'), \
             patch('omoai.api.services._run_postprocess_script'):

            # Configure output_params to exclude both summary fields
            output_params = OutputFormatParams(
                include=["transcript_punct", "segments"],
                summary="none"
            )

            result, _raw_transcript = await _run_full_pipeline_script(mock_pipeline_request, output_params)

            # Both summary and timestamped_summary should be excluded
            assert result.summary == {}
            assert result.summary.get("raw") is None
            assert result.timestamped_summary is None

    @pytest.mark.asyncio
    async def test_timestamped_summary_fields_filter(self, mock_pipeline_request, mock_final_obj):
        """Timestamped summary fields list should filter keys."""
        with patch('builtins.open', create=True) as _, \
             patch('json.load', return_value=mock_final_obj), \
             patch('os.urandom', return_value=b'fake_random'), \
             patch('pathlib.Path.mkdir'), \
             patch('os.open'), \
             patch('os.write'), \
             patch('os.close'), \
             patch('shutil.rmtree'), \
             patch('asyncio.to_thread'), \
             patch('omoai.api.services.run_preprocess_script'), \
             patch('omoai.api.services._run_asr_script'), \
             patch('omoai.api.services._run_postprocess_script'):

            output_params = OutputFormatParams(
                include=["timestamped_summary"],
                timestamped_summary_fields=["timestamps"],
            )

            result, _ = await _run_full_pipeline_script(mock_pipeline_request, output_params)

            assert result.timestamped_summary == {
                "timestamps": [
                    {"time": "00:00:30", "text": "Introduction"},
                    {"time": "00:01:00", "text": "Main Topic"}
                ]
            }

    def test_config_yaml_api_defaults(self):
        """Test that config.yaml api_defaults.include is respected."""
        config = get_config()
        api_defaults = getattr(config.output, "api_defaults", None)

        if api_defaults and hasattr(api_defaults, "include"):
            # Check that the default include list from config.yaml is as expected
            expected_includes = ["transcript_punct", "timestamped_summary"]
            assert api_defaults.include == expected_includes, f"Expected {expected_includes}, got {api_defaults.include}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

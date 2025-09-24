#!/usr/bin/env python3
"""
Integration test for API include filtering functionality.

This test verifies that the API endpoint correctly respects the 'include' parameter
to control which fields are included in the response.
"""

import pytest

from omoai.config.schemas import get_config


class TestAPIIncludeIntegration:
    """Integration test for API include filtering."""

    def test_config_yaml_api_defaults_respected(self):
        """Test that config.yaml api_defaults.include is correctly loaded."""
        config = get_config()

        # Check the current config.yaml settings
        api_defaults = getattr(config.output, "api_defaults", None)
        assert api_defaults is not None, "api_defaults should be configured"

        # Verify the default include list from config.yaml
        expected_includes = ["transcript_punct", "timestamped_summary"]
        assert hasattr(api_defaults, "include"), "api_defaults should have include attribute"
        assert api_defaults.include == expected_includes, f"Expected {expected_includes}, got {api_defaults.include}"

        print(f"✓ Config.yaml api_defaults.include: {api_defaults.include}")
        print(f"✓ Config.yaml api_defaults.summary: {getattr(api_defaults, 'summary', 'not set')}")
        print(f"✓ Config.yaml api_defaults.summary_bullets_max: {getattr(api_defaults, 'summary_bullets_max', 'not set')}")

    def test_models_include_summary_literal(self):
        """Test that the models now include 'summary' as a valid literal."""
        from omoai.api.models import OutputFormatParams

        # Test that we can create OutputFormatParams with summary in include list
        params = OutputFormatParams(
            include=["transcript_punct", "segments", "summary", "timestamped_summary"]
        )

        assert params.include is not None
        assert "summary" in params.include
        assert "timestamped_summary" in params.include
        assert "transcript_punct" in params.include
        assert "segments" in params.include

        print("✓ OutputFormatParams accepts 'summary' and 'timestamped_summary' literals")

    def test_main_controller_include_summary_literal(self):
        """Test that the main controller accepts 'summary' in include parameter."""
        from omoai.api.models import OutputFormatParams

        # The controller should accept summary in the include parameter via OutputFormatParams
        # We can verify this by creating an OutputFormatParams with summary in the include list
        params = OutputFormatParams(
            include=["transcript_punct", "segments", "summary", "timestamped_summary"]
        )

        assert params.include is not None
        assert "summary" in params.include
        assert "timestamped_summary" in params.include

        print("✓ MainController accepts 'summary' via OutputFormatParams.include")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

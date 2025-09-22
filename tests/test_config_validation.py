#!/usr/bin/env python3
"""
Test configuration validation with Pydantic schemas.
"""

import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

# Path setup is handled by tests/conftest.py
import torch
from pydantic import ValidationError

project_root = Path(__file__).parent.parent

from omoai.config import (
    APIConfig,
    ASRConfig,
    LLMConfig,
    OmoAIConfig,
    PathsConfig,
    get_config,
    reload_config,
)


class TestConfigValidation(unittest.TestCase):
    """Test configuration validation and schema enforcement."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_config_path = self.temp_dir / "test_config.yaml"

        # Create minimal valid config for testing
        self.valid_config = {
            "paths": {
                "chunkformer_dir": str(self.temp_dir / "chunkformer"),
                "chunkformer_checkpoint": str(self.temp_dir / "checkpoint"),
                "out_dir": str(self.temp_dir / "output"),
            },
            "llm": {
                "model_id": "test/model",
                "trust_remote_code": False,
            },
            "punctuation": {
                "llm": {
                    "model_id": "test/model",
                    "trust_remote_code": False,
                },
                "system_prompt": "Test prompt",
            },
            "summarization": {
                "llm": {
                    "model_id": "test/model",
                    "trust_remote_code": False,
                },
                "system_prompt": "Test prompt",
            },
        }

        # Create required directories for path validation
        (self.temp_dir / "chunkformer").mkdir()
        (self.temp_dir / "checkpoint").mkdir()

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

        # Clear cached config
        if hasattr(get_config, "_instance"):
            delattr(get_config, "_instance")

    def test_valid_config_loading(self):
        """Test loading a valid configuration."""
        import yaml

        with open(self.test_config_path, "w") as f:
            yaml.dump(self.valid_config, f)

        config = OmoAIConfig.load_from_yaml(self.test_config_path)
        self.assertEqual(config.llm.model_id, "test/model")
        self.assertFalse(config.llm.trust_remote_code)

    def test_security_defaults(self):
        """Test that security defaults are properly applied."""
        config = LLMConfig(model_id="test/model")

        # Should default to secure settings
        self.assertFalse(config.trust_remote_code)
        self.assertEqual(config.gpu_memory_utilization, 0.85)

        api_config = APIConfig()
        self.assertEqual(api_config.host, "127.0.0.1")  # localhost only
        self.assertFalse(api_config.enable_progress_output)  # no info leakage

    def test_trust_remote_code_warning(self):
        """Test that enabling trust_remote_code triggers warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            LLMConfig(model_id="test/model", trust_remote_code=True)

            # Should have triggered a warning
            self.assertTrue(len(w) > 0)
            self.assertIn("SECURITY RISK", str(w[0].message))

    def test_path_validation(self):
        """Test path validation and creation."""
        # Test non-existent required path
        with self.assertRaises(ValidationError) as ctx:
            PathsConfig(
                chunkformer_dir="/non/existent/path",
                chunkformer_checkpoint="/another/non/existent/path",
            )

        self.assertIn("does not exist", str(ctx.exception))

    def test_value_ranges(self):
        """Test that configuration values are within valid ranges."""
        # Test ASR config ranges
        asr_config = ASRConfig()
        self.assertGreaterEqual(asr_config.total_batch_duration_s, 60)
        self.assertLessEqual(asr_config.total_batch_duration_s, 7200)

        # Test invalid ranges
        with self.assertRaises(ValidationError):
            ASRConfig(total_batch_duration_s=30)  # Too low

        with self.assertRaises(ValidationError):
            ASRConfig(total_batch_duration_s=10000)  # Too high

    def test_device_auto_detection(self):
        """Test automatic device detection."""
        asr_config = ASRConfig(device="auto")

        # Should resolve to actual device
        self.assertIn(asr_config.device, ["cpu", "cuda"])

        # Should match torch availability
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(asr_config.device, expected)

    def test_whisperx_defaults(self):
        """Ensure WhisperX configuration has expected defaults."""
        cfg = ASRConfig(engine="whisperx")
        self.assertEqual(cfg.engine, "whisperx")
        self.assertEqual(cfg.whisperx.model_path, Path("models/whisperx/EraX"))
        self.assertEqual(cfg.whisperx.compute_type, "int8_float16")

    def test_environment_variable_override(self):
        """Test environment variable override functionality."""
        with patch.dict(
            os.environ,
            {
                "OMOAI_ASR__DEVICE": "cpu",
                "OMOAI_API__PORT": "9000",
            },
        ):
            # Create config with env vars
            config_data = self.valid_config.copy()
            config = OmoAIConfig(**config_data)

            # Environment variables should override
            self.assertEqual(config.asr.device, "cpu")
            self.assertEqual(config.api.port, 9000)

    def test_cross_field_validation(self):
        """Test cross-field validation logic."""
        config_data = self.valid_config.copy()

        # Remove punctuation llm model_id to test inheritance
        config_data["punctuation"]["llm"] = {"trust_remote_code": False}  # No model_id

        config = OmoAIConfig(**config_data)

        # Should inherit from base llm config
        self.assertEqual(config.punctuation.llm.model_id, config.llm.model_id)

    def test_yaml_export_import(self):
        """Test YAML export and import functionality."""
        import yaml

        with open(self.test_config_path, "w") as f:
            yaml.dump(self.valid_config, f)

        # Load config
        config = OmoAIConfig.load_from_yaml(self.test_config_path)

        # Export to YAML
        yaml_output = config.model_dump_yaml()

        # Should be valid YAML
        reimported = yaml.safe_load(yaml_output)
        self.assertIsInstance(reimported, dict)
        self.assertIn("paths", reimported)

    def test_invalid_config_rejection(self):
        """Test that invalid configs are properly rejected."""
        # Test missing required fields
        with self.assertRaises(ValidationError):
            OmoAIConfig(paths={"chunkformer_dir": "/tmp"})  # Missing llm config

        # Test extra fields (should be forbidden)
        config_data = self.valid_config.copy()
        config_data["unknown_field"] = "should_fail"

        with self.assertRaises(ValidationError):
            OmoAIConfig(**config_data)

    def test_secure_config_loading(self):
        """Test loading the secure configuration template."""
        secure_config_path = project_root / "config.secure.yaml"

        if secure_config_path.exists():
            # Should be able to load secure config (with mocked paths)
            import yaml

            with open(secure_config_path) as f:
                secure_data = yaml.safe_load(f)

            # Update paths to use temp directories for testing
            secure_data["paths"]["chunkformer_dir"] = str(self.temp_dir / "chunkformer")
            secure_data["paths"]["chunkformer_checkpoint"] = str(
                self.temp_dir / "checkpoint"
            )

            config = OmoAIConfig(**secure_data)

            # Verify security defaults
            self.assertFalse(config.llm.trust_remote_code)
            self.assertFalse(config.punctuation.llm.trust_remote_code)
            self.assertFalse(config.summarization.llm.trust_remote_code)
            self.assertEqual(config.api.host, "127.0.0.1")
            self.assertFalse(config.api.enable_progress_output)

    def test_global_config_singleton(self):
        """Test global configuration singleton pattern."""
        import yaml

        with open(self.test_config_path, "w") as f:
            yaml.dump(self.valid_config, f)

        # Mock the search path to find our test config
        with patch.dict(os.environ, {"OMOAI_CONFIG": str(self.test_config_path)}):
            config1 = get_config()
            config2 = get_config()

            # Should be the same instance
            self.assertIs(config1, config2)

            # Test reload
            new_config = reload_config(self.test_config_path)
            self.assertIsNot(config1, new_config)

    def test_model_consistency_validation(self):
        """Test model configuration consistency validation."""
        config_data = self.valid_config.copy()

        # Set different trust_remote_code values
        config_data["llm"]["trust_remote_code"] = True
        config_data["punctuation"]["llm"]["trust_remote_code"] = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # config = OmoAIConfig(**config_data)  # Unused variable removed

            # Should warn about mixed trust_remote_code settings
            warning_messages = [str(warning.message) for warning in w]
            self.assertTrue(
                any("trust_remote_code=True" in msg for msg in warning_messages)
            )

    def test_temp_directory_validation(self):
        """Test temporary directory validation and creation."""
        # Test valid temp directory
        temp_path = self.temp_dir / "api_temp"
        # api_config = APIConfig(temp_dir=temp_path)  # Unused variable removed

        # Should create the directory
        self.assertTrue(temp_path.exists())

        # Test inaccessible directory (mock permission error)
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("Access denied")):
            with self.assertRaises(ValidationError) as ctx:
                APIConfig(temp_dir="/root/inaccessible")

            self.assertIn("not accessible", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

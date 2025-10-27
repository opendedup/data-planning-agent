"""
Tests for MCP Configuration

Tests configuration loading and validation.
"""

import os
import pytest

from data_planning_agent.mcp.config import PlanningAgentConfig, load_config


def test_config_defaults(monkeypatch) -> None:
    """Test that configuration has sensible defaults."""
    # Clear any environment variable that might be set
    monkeypatch.delenv("MAX_CONVERSATION_TURNS", raising=False)
    
    config = PlanningAgentConfig()

    assert config.gemini_model == "gemini-2.5-pro"
    assert config.mcp_transport == "stdio"
    assert config.mcp_host == "0.0.0.0"
    assert config.mcp_port == 8080
    assert config.max_conversation_turns == 3
    assert config.log_level == "INFO"


def test_config_from_env(monkeypatch) -> None:
    """Test configuration loading from environment variables."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")
    monkeypatch.setenv("OUTPUT_DIR", "/test/output")
    monkeypatch.setenv("MCP_PORT", "9090")

    config = PlanningAgentConfig()

    assert config.gemini_api_key == "test-key-123"
    assert config.gemini_model == "gemini-test-model"
    assert config.output_dir == "/test/output"
    assert config.mcp_port == 9090


def test_config_validation_missing_api_key() -> None:
    """Test that validation fails when API key is missing."""
    config = PlanningAgentConfig(gemini_api_key="")

    with pytest.raises(ValueError, match="Required configuration missing.*gemini_api_key"):
        config.validate_required_fields()


def test_config_validation_success() -> None:
    """Test that validation succeeds with all required fields."""
    config = PlanningAgentConfig(gemini_api_key="test-key")

    # Should not raise
    config.validate_required_fields()


def test_load_config_with_api_key(monkeypatch) -> None:
    """Test load_config function with valid API key."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    config = load_config()

    assert config.gemini_api_key == "test-key"
    assert isinstance(config, PlanningAgentConfig)


def test_load_config_without_api_key(monkeypatch) -> None:
    """Test load_config function fails without API key."""
    monkeypatch.setenv("GEMINI_API_KEY", "")

    with pytest.raises(ValueError):
        load_config()


"""
Tests for MCP Tools

Tests MCP tool definitions and validation.
"""

import pytest

from data_planning_agent.mcp.tools import (
    CONTINUE_CONVERSATION_TOOL,
    GENERATE_DATA_PRP_TOOL,
    START_PLANNING_SESSION_TOOL,
    get_available_tools,
    validate_tool_params,
)


def test_get_available_tools() -> None:
    """Test that all tools are returned."""
    tools = get_available_tools()

    assert len(tools) == 3

    tool_names = [tool.name for tool in tools]
    assert START_PLANNING_SESSION_TOOL in tool_names
    assert CONTINUE_CONVERSATION_TOOL in tool_names
    assert GENERATE_DATA_PRP_TOOL in tool_names


def test_tool_schemas_complete() -> None:
    """Test that all tools have complete schemas."""
    tools = get_available_tools()

    for tool in tools:
        assert tool.name is not None
        assert tool.description is not None
        assert tool.inputSchema is not None
        assert "type" in tool.inputSchema
        assert "properties" in tool.inputSchema


def test_validate_start_planning_session_valid() -> None:
    """Test validation of valid start_planning_session params."""
    params = {"initial_intent": "Test business intent"}

    # Should not raise
    validate_tool_params(params, START_PLANNING_SESSION_TOOL)


def test_validate_start_planning_session_missing_intent() -> None:
    """Test validation fails when initial_intent is missing."""
    params = {}

    with pytest.raises(ValueError, match="Missing required parameter: initial_intent"):
        validate_tool_params(params, START_PLANNING_SESSION_TOOL)


def test_validate_start_planning_session_empty_intent() -> None:
    """Test validation fails when initial_intent is empty."""
    params = {"initial_intent": "   "}

    with pytest.raises(ValueError, match="initial_intent cannot be empty"):
        validate_tool_params(params, START_PLANNING_SESSION_TOOL)


def test_validate_continue_conversation_valid() -> None:
    """Test validation of valid continue_conversation params."""
    params = {"session_id": "test-session-id", "user_response": "My response"}

    # Should not raise
    validate_tool_params(params, CONTINUE_CONVERSATION_TOOL)


def test_validate_continue_conversation_missing_session_id() -> None:
    """Test validation fails when session_id is missing."""
    params = {"user_response": "My response"}

    with pytest.raises(ValueError, match="Missing required parameter: session_id"):
        validate_tool_params(params, CONTINUE_CONVERSATION_TOOL)


def test_validate_continue_conversation_missing_response() -> None:
    """Test validation fails when user_response is missing."""
    params = {"session_id": "test-session-id"}

    with pytest.raises(ValueError, match="Missing required parameter: user_response"):
        validate_tool_params(params, CONTINUE_CONVERSATION_TOOL)


def test_validate_generate_data_prp_valid() -> None:
    """Test validation of valid generate_data_prp params."""
    params = {"session_id": "test-session-id"}

    # Should not raise
    validate_tool_params(params, GENERATE_DATA_PRP_TOOL)


def test_validate_generate_data_prp_with_output_path() -> None:
    """Test validation with optional output_path."""
    params = {"session_id": "test-session-id", "output_path": "/test/path.md"}

    # Should not raise
    validate_tool_params(params, GENERATE_DATA_PRP_TOOL)


def test_validate_generate_data_prp_missing_session_id() -> None:
    """Test validation fails when session_id is missing."""
    params = {}

    with pytest.raises(ValueError, match="Missing required parameter: session_id"):
        validate_tool_params(params, GENERATE_DATA_PRP_TOOL)


def test_validate_unknown_tool() -> None:
    """Test validation fails for unknown tool."""
    params = {}

    with pytest.raises(ValueError, match="Unknown tool"):
        validate_tool_params(params, "unknown_tool")


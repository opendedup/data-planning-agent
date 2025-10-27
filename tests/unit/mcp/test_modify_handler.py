"""
Tests for modify_existing_prp Handler

Tests the PRP modification handler functionality.
"""

from typing import TYPE_CHECKING

import pytest

from data_planning_agent.mcp.handlers import MCPHandlers
from data_planning_agent.models.session import PlanningSession

if TYPE_CHECKING:
    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def mock_config(mocker: "MockerFixture") -> None:
    """Mock configuration for testing."""
    config = mocker.MagicMock()
    config.max_conversation_turns = 3
    return config


@pytest.fixture
def mock_gemini_client(mocker: "MockerFixture") -> None:
    """Mock Gemini client for testing."""
    return mocker.MagicMock()


@pytest.fixture
def mock_storage_client(mocker: "MockerFixture") -> None:
    """Mock storage client for testing."""
    return mocker.MagicMock()


@pytest.fixture
def handlers(
    mock_config: None, mock_gemini_client: None, mock_storage_client: None
) -> MCPHandlers:
    """Create handlers instance for testing."""
    return MCPHandlers(
        config=mock_config,
        gemini_client=mock_gemini_client,
        storage_client=mock_storage_client,
    )


@pytest.mark.asyncio
async def test_handle_modify_existing_prp_success(
    handlers: MCPHandlers, mocker: "MockerFixture"
) -> None:
    """Test successful PRP modification handler invocation."""
    # Mock the refiner's start_session method
    mock_session_id = "test-session-123"
    mock_questions = "Here are some questions about your changes..."

    mock_start_session = mocker.patch.object(
        handlers.refiner,
        "start_session",
        return_value=(mock_session_id, mock_questions),
    )

    # Mock the session retrieval
    mock_session = PlanningSession(initial_intent="test")
    mock_get_session = mocker.patch.object(
        handlers.refiner, "get_session", return_value=mock_session
    )

    # Mock conversation manager update
    mock_update = mocker.patch.object(handlers.conversation_manager, "update_session")

    # Call handler
    arguments = {
        "existing_prp": "# Original PRP\n\nSome content here",
        "requested_changes": "Add geographic breakdown by state",
    }

    result = await handlers.handle_modify_existing_prp(arguments)

    # Verify start_session was called with combined intent
    assert mock_start_session.call_count == 1
    called_intent = mock_start_session.call_args[0][0]
    assert "Original PRP" in called_intent
    assert "Add geographic breakdown by state" in called_intent

    # Verify session was retrieved and updated
    mock_get_session.assert_called_once_with(mock_session_id)
    mock_update.assert_called_once()

    # Verify source_prp was stored
    assert mock_session.source_prp == "# Original PRP\n\nSome content here"

    # Verify response format
    assert len(result) == 1
    assert result[0].type == "text"
    assert "PRP Modification Session Started" in result[0].text
    assert mock_session_id in result[0].text
    assert mock_questions in result[0].text


@pytest.mark.asyncio
async def test_handle_modify_existing_prp_combined_intent_format(
    handlers: MCPHandlers, mocker: "MockerFixture"
) -> None:
    """Test that combined intent is formatted correctly."""
    mock_session_id = "test-session-456"
    mock_questions = "Questions..."

    # Capture the intent passed to start_session
    captured_intent = None

    async def capture_start_session(intent: str) -> tuple[str, str]:
        nonlocal captured_intent
        captured_intent = intent
        return (mock_session_id, mock_questions)

    mocker.patch.object(handlers.refiner, "start_session", side_effect=capture_start_session)
    mocker.patch.object(handlers.refiner, "get_session", return_value=None)

    # Call handler
    arguments = {
        "existing_prp": "# Test PRP\n\n## Section 1\nContent",
        "requested_changes": "Add new metric: conversion rate",
    }

    await handlers.handle_modify_existing_prp(arguments)

    # Verify intent structure
    assert captured_intent is not None
    assert "existing Data Product Requirement Prompt" in captured_intent
    assert "Here is the existing PRP:" in captured_intent
    assert "# Test PRP" in captured_intent
    assert "Requested changes:" in captured_intent
    assert "Add new metric: conversion rate" in captured_intent


@pytest.mark.asyncio
async def test_handle_modify_existing_prp_error_handling(
    handlers: MCPHandlers, mocker: "MockerFixture"
) -> None:
    """Test error handling in modify handler."""
    # Mock start_session to raise an exception
    mocker.patch.object(
        handlers.refiner, "start_session", side_effect=Exception("Test error")
    )

    arguments = {
        "existing_prp": "# Test PRP",
        "requested_changes": "Add something",
    }

    result = await handlers.handle_modify_existing_prp(arguments)

    # Verify error response
    assert len(result) == 1
    assert result[0].type == "text"
    assert "❌ Error" in result[0].text
    assert "Test error" in result[0].text


@pytest.mark.asyncio
async def test_handle_modify_existing_prp_session_not_found(
    handlers: MCPHandlers, mocker: "MockerFixture"
) -> None:
    """Test when session cannot be retrieved after creation."""
    mock_session_id = "test-session-789"
    mock_questions = "Questions..."

    mocker.patch.object(
        handlers.refiner,
        "start_session",
        return_value=(mock_session_id, mock_questions),
    )

    # Session not found
    mocker.patch.object(handlers.refiner, "get_session", return_value=None)

    # Mock update should not be called
    mock_update = mocker.patch.object(handlers.conversation_manager, "update_session")

    arguments = {
        "existing_prp": "# Test PRP",
        "requested_changes": "Add something",
    }

    result = await handlers.handle_modify_existing_prp(arguments)

    # Should still succeed (source_prp storage is optional)
    assert len(result) == 1
    assert "PRP Modification Session Started" in result[0].text
    mock_update.assert_not_called()


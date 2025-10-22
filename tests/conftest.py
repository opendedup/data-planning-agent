"""
Test Configuration and Fixtures

Provides common test fixtures for Planning Agent tests.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from data_planning_agent.clients.gemini_client import GeminiClient
from data_planning_agent.clients.storage_client import StorageClient
from data_planning_agent.core.conversation import ConversationManager
from data_planning_agent.core.prp_generator import PRPGenerator
from data_planning_agent.core.refiner import RequirementRefiner
from data_planning_agent.mcp.config import PlanningAgentConfig
from data_planning_agent.models.session import PlanningSession


@pytest.fixture
def mock_config() -> PlanningAgentConfig:
    """
    Provide a mock configuration for testing.

    Returns:
        Mock PlanningAgentConfig
    """
    return PlanningAgentConfig(
        gemini_api_key="test-api-key",
        gemini_model="gemini-2.5-pro",
        output_dir="./test-output",
        mcp_server_name="test-planning-agent",
        mcp_server_version="0.1.0",
        mcp_transport="stdio",
        mcp_host="127.0.0.1",
        mcp_port=8080,
        max_conversation_turns=5,
        log_level="DEBUG",
    )


@pytest.fixture
def mock_gemini_client(mocker) -> Mock:
    """
    Provide a mock Gemini client.

    Returns:
        Mock GeminiClient
    """
    client = mocker.Mock(spec=GeminiClient)
    client.context = None  # No context by default
    client.generate_initial_questions = AsyncMock(
        return_value="1. What is your audience?\n   a) Executives\n   b) Analysts"
    )
    client.generate_follow_up_questions = AsyncMock(
        return_value=("2. What metrics do you need?", False)
    )
    client.generate_data_prp = AsyncMock(
        return_value="# Data Product Requirement Prompt\n\nTest content"
    )
    return client


@pytest.fixture
def mock_storage_client(mocker) -> Mock:
    """
    Provide a mock storage client.

    Returns:
        Mock StorageClient
    """
    client = mocker.Mock(spec=StorageClient)
    client.write_file = Mock(return_value="/test/path/data_prp.md")
    client.read_file = Mock(return_value="Test content")
    return client


@pytest.fixture
def conversation_manager() -> ConversationManager:
    """
    Provide a conversation manager instance.

    Returns:
        ConversationManager
    """
    return ConversationManager()


@pytest.fixture
def planning_session() -> PlanningSession:
    """
    Provide a sample planning session.

    Returns:
        PlanningSession
    """
    session = PlanningSession(
        initial_intent="Test intent: analyze sales trends in the region"
    )
    session.add_turn("user", "Test intent: analyze sales trends in the region")
    session.add_turn("assistant", "What is your target audience?")
    return session


@pytest.fixture
def requirement_refiner(
    mock_gemini_client, conversation_manager, mock_config
) -> RequirementRefiner:
    """
    Provide a requirement refiner instance.

    Returns:
        RequirementRefiner
    """
    return RequirementRefiner(
        gemini_client=mock_gemini_client,
        conversation_manager=conversation_manager,
        max_turns=mock_config.max_conversation_turns,
    )


@pytest.fixture
def prp_generator(mock_gemini_client, mock_storage_client) -> PRPGenerator:
    """
    Provide a PRP generator instance.

    Returns:
        PRPGenerator
    """
    return PRPGenerator(
        gemini_client=mock_gemini_client, storage_client=mock_storage_client
    )


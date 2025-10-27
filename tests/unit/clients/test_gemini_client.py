"""
Tests for Gemini Client

Tests Gemini client functionality including assumption tracking during PRP generation.
"""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from data_planning_agent.clients.gemini_client import GeminiClient
from data_planning_agent.models.session import PlanningSession

if TYPE_CHECKING:
    pass


@pytest.fixture
def gemini_client_with_vertex() -> GeminiClient:
    """Create a Gemini client with mocked vertex search."""
    # Create a mock vertex search client
    mock_vertex = Mock()
    mock_vertex.search = Mock(return_value=[])
    
    client = GeminiClient(
        api_key="test-api-key",
        model_name="gemini-2.5-pro",
        vertex_search_client=mock_vertex,
    )
    
    return client


@pytest.mark.asyncio
async def test_generate_data_prp_adds_no_results_assumption(
    gemini_client_with_vertex: GeminiClient,
    mocker,
) -> None:
    """Test that assumption is added when no datastore results are found."""
    session = PlanningSession(initial_intent="Test intent")
    session.add_turn("user", "Test intent")
    session.add_turn("assistant", "Questions")
    
    # Mock the internal methods
    mocker.patch.object(
        gemini_client_with_vertex,
        "_analyze_requirements",
        new=AsyncMock(return_value=Mock(target_views=[], data_gaps=[])),
    )
    mocker.patch.object(
        gemini_client_with_vertex,
        "_synthesize_search_query",
        new=AsyncMock(return_value="test query"),
    )
    mocker.patch.object(
        gemini_client_with_vertex,
        "_query_datastore",
        return_value=("no_match", "", []),  # Empty results
    )
    mocker.patch.object(
        gemini_client_with_vertex,
        "_build_detailed_data_section",
        return_value="Test data section",
    )
    mocker.patch.object(
        gemini_client_with_vertex,
        "_build_prompt_with_context",
        side_effect=lambda x: x,
    )
    
    # Mock the Gemini API response
    mock_response = Mock()
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].content.parts = [Mock()]
    mock_response.text = "# Data Product Requirement Prompt\n\nTest PRP"
    
    mocker.patch.object(
        gemini_client_with_vertex.model,
        "generate_content",
        return_value=mock_response,
    )
    
    # Generate PRP
    await gemini_client_with_vertex.generate_data_prp(session)
    
    # Check that assumption was added
    assert len(session.assumptions) > 0
    assumption_found = any(
        "No matching data assets found" in assumption
        for assumption in session.assumptions
    )
    assert assumption_found, "No datastore results assumption not found"


@pytest.mark.asyncio
async def test_generate_data_prp_adds_search_fallback_assumption(
    gemini_client_with_vertex: GeminiClient,
    mocker,
) -> None:
    """Test that assumption is added when search synthesis fails."""
    session = PlanningSession(initial_intent="Test intent")
    session.add_turn("user", "Test intent")
    session.add_turn("assistant", "Questions")
    
    # Mock the internal methods
    mocker.patch.object(
        gemini_client_with_vertex,
        "_analyze_requirements",
        new=AsyncMock(return_value=Mock(target_views=[], data_gaps=[])),
    )
    mocker.patch.object(
        gemini_client_with_vertex,
        "_synthesize_search_query",
        new=AsyncMock(return_value=""),  # Empty query triggers fallback
    )
    mocker.patch.object(
        gemini_client_with_vertex,
        "_query_datastore",
        return_value=("no_match", "", []),
    )
    mocker.patch.object(
        gemini_client_with_vertex,
        "_build_detailed_data_section",
        return_value="Test data section",
    )
    mocker.patch.object(
        gemini_client_with_vertex,
        "_build_prompt_with_context",
        side_effect=lambda x: x,
    )
    
    # Mock the Gemini API response
    mock_response = Mock()
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].content.parts = [Mock()]
    mock_response.text = "# Data Product Requirement Prompt\n\nTest PRP"
    
    mocker.patch.object(
        gemini_client_with_vertex.model,
        "generate_content",
        return_value=mock_response,
    )
    
    # Generate PRP
    await gemini_client_with_vertex.generate_data_prp(session)
    
    # Check that assumption was added
    assert len(session.assumptions) > 0
    assumption_found = any(
        "Used initial user intent for data search" in assumption
        for assumption in session.assumptions
    )
    assert assumption_found, "Search fallback assumption not found"


@pytest.mark.asyncio
async def test_generate_data_prp_includes_assumptions_in_prompt(
    gemini_client_with_vertex: GeminiClient,
    mocker,
) -> None:
    """Test that tracked assumptions are included in the generation prompt."""
    session = PlanningSession(initial_intent="Test intent")
    session.add_turn("user", "Test intent")
    session.add_turn("assistant", "Questions")
    
    # Add some assumptions to the session
    session.add_assumption("Test assumption 1")
    session.add_assumption("Test assumption 2")
    
    # Mock the internal methods
    mocker.patch.object(
        gemini_client_with_vertex,
        "_analyze_requirements",
        new=AsyncMock(return_value=Mock(target_views=[], data_gaps=[])),
    )
    mocker.patch.object(
        gemini_client_with_vertex,
        "_synthesize_search_query",
        new=AsyncMock(return_value="test query"),
    )
    mocker.patch.object(
        gemini_client_with_vertex,
        "_query_datastore",
        return_value=("match", "Test context", [{"title": "test"}]),
    )
    mocker.patch.object(
        gemini_client_with_vertex,
        "_build_detailed_data_section",
        return_value="Test data section",
    )
    mocker.patch.object(
        gemini_client_with_vertex,
        "_build_prompt_with_context",
        side_effect=lambda x: x,
    )
    
    # Mock the Gemini API response
    mock_response = Mock()
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].content.parts = [Mock()]
    mock_response.text = "# Data Product Requirement Prompt\n\nTest PRP"
    
    generate_content_mock = mocker.patch.object(
        gemini_client_with_vertex.model,
        "generate_content",
        return_value=mock_response,
    )
    
    # Generate PRP
    await gemini_client_with_vertex.generate_data_prp(session)
    
    # Check that generate_content was called with a prompt containing assumptions
    call_args = generate_content_mock.call_args
    prompt = call_args[0][0]
    
    assert "Tracked Assumptions from Planning Process:" in prompt
    assert "[ASSUMPTION-01]: Test assumption 1" in prompt
    assert "[ASSUMPTION-02]: Test assumption 2" in prompt
    assert "## 10. Assumptions & Defaults" in prompt


@pytest.mark.asyncio
async def test_generate_data_prp_no_assumptions_tracked(
    gemini_client_with_vertex: GeminiClient,
    mocker,
) -> None:
    """Test PRP generation when no assumptions have been tracked."""
    session = PlanningSession(initial_intent="Test intent")
    session.add_turn("user", "Test intent")
    session.add_turn("assistant", "Questions")
    
    # Don't add any assumptions
    
    # Mock the internal methods
    mocker.patch.object(
        gemini_client_with_vertex,
        "_analyze_requirements",
        new=AsyncMock(return_value=Mock(target_views=[], data_gaps=[])),
    )
    mocker.patch.object(
        gemini_client_with_vertex,
        "_synthesize_search_query",
        new=AsyncMock(return_value="test query"),
    )
    mocker.patch.object(
        gemini_client_with_vertex,
        "_query_datastore",
        return_value=("match", "Test context", [{"title": "test"}]),
    )
    mocker.patch.object(
        gemini_client_with_vertex,
        "_build_detailed_data_section",
        return_value="Test data section",
    )
    mocker.patch.object(
        gemini_client_with_vertex,
        "_build_prompt_with_context",
        side_effect=lambda x: x,
    )
    
    # Mock the Gemini API response
    mock_response = Mock()
    mock_response.candidates = [Mock()]
    mock_response.candidates[0].content.parts = [Mock()]
    mock_response.text = "# Data Product Requirement Prompt\n\nTest PRP"
    
    generate_content_mock = mocker.patch.object(
        gemini_client_with_vertex.model,
        "generate_content",
        return_value=mock_response,
    )
    
    # Generate PRP
    await gemini_client_with_vertex.generate_data_prp(session)
    
    # Check that prompt still includes Section 10 instruction
    call_args = generate_content_mock.call_args
    prompt = call_args[0][0]
    
    assert "## 10. Assumptions & Defaults" in prompt


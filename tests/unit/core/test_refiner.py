"""
Tests for Requirement Refiner

Tests the requirement refinement process including assumption tracking.
"""

from typing import TYPE_CHECKING

import pytest

from data_planning_agent.core.refiner import RequirementRefiner

if TYPE_CHECKING:
    from data_planning_agent.core.conversation import ConversationManager
    from unittest.mock import Mock


@pytest.mark.asyncio
async def test_circuit_breaker_adds_assumption(
    requirement_refiner: RequirementRefiner,
    conversation_manager: "ConversationManager",
    mock_gemini_client: "Mock",
) -> None:
    """Test that circuit breaker adds an assumption when max turns reached."""
    # Start a session
    session_id, _ = await requirement_refiner.start_session("Test intent")
    
    # Simulate reaching max turns (max_turns=5 from conftest)
    # We already have 2 turns from start_session, need 3 more to reach 5
    mock_gemini_client.generate_follow_up_questions.return_value = (
        "More questions?",
        False,
    )
    
    # Add turns until we hit the circuit breaker
    for i in range(2):
        await requirement_refiner.continue_conversation(
            session_id, f"User response {i}"
        )
    
    # This should trigger the circuit breaker
    response, is_complete = await requirement_refiner.continue_conversation(
        session_id, "Final response"
    )
    
    # Verify session is complete
    assert is_complete is True
    
    # Get session and check assumption was added
    session = requirement_refiner.get_session(session_id)
    assert session is not None
    assert len(session.assumptions) > 0
    
    # Check the assumption message
    assumption_found = any(
        "Maximum conversation turns reached" in assumption
        for assumption in session.assumptions
    )
    assert assumption_found, "Circuit breaker assumption not found in session"
    
    # Check the response message includes warning
    assert "Maximum conversation turns reached" in response
    assert "circuit breaker" in response.lower()


@pytest.mark.asyncio
async def test_assumption_has_correct_format(
    requirement_refiner: RequirementRefiner,
    conversation_manager: "ConversationManager",
    mock_gemini_client: "Mock",
) -> None:
    """Test that circuit breaker assumption has correct ID format."""
    # Start a session
    session_id, _ = await requirement_refiner.start_session("Test intent")
    
    # Set up to trigger circuit breaker
    mock_gemini_client.generate_follow_up_questions.return_value = (
        "More questions?",
        False,
    )
    
    # Add turns until circuit breaker (we need to reach turn count >= max_turns)
    # Start session creates 2 turns, so we need to check when it will trigger
    for i in range(2):  # Only 2 more turns to reach max_turns=5
        response, is_complete = await requirement_refiner.continue_conversation(
            session_id, f"User response {i}"
        )
        if is_complete:
            break
    
    # Get session and check assumption format
    session = requirement_refiner.get_session(session_id)
    assert session is not None
    assert len(session.assumptions) >= 1, "At least one assumption should be added"
    
    # Check format [ASSUMPTION-01]: ... (check the first assumption)
    assert session.assumptions[0].startswith("[ASSUMPTION-01]:")


@pytest.mark.asyncio
async def test_no_assumption_when_completed_normally(
    requirement_refiner: RequirementRefiner,
    conversation_manager: "ConversationManager",
    mock_gemini_client: "Mock",
) -> None:
    """Test that no circuit breaker assumption is added when completed normally."""
    # Start a session
    session_id, _ = await requirement_refiner.start_session("Test intent")
    
    # Mock normal completion (not circuit breaker)
    mock_gemini_client.generate_follow_up_questions.return_value = (
        "COMPLETE",
        True,
    )
    
    # Continue conversation - should complete normally
    response, is_complete = await requirement_refiner.continue_conversation(
        session_id, "User response"
    )
    
    # Verify session is complete
    assert is_complete is True
    
    # Get session and check NO assumption was added
    session = requirement_refiner.get_session(session_id)
    assert session is not None
    
    # Should not have circuit breaker assumption
    circuit_breaker_found = any(
        "Maximum conversation turns reached" in assumption
        for assumption in session.assumptions
    )
    assert not circuit_breaker_found, "Circuit breaker assumption should not be present"


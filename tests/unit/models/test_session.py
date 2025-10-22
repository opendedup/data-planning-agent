"""
Tests for Session Models

Tests session data models and operations.
"""

import pytest

from data_planning_agent.models.session import (
    ConversationTurn,
    ExtractedRequirement,
    PlanningSession,
)


def test_planning_session_creation() -> None:
    """Test creating a new planning session."""
    session = PlanningSession(initial_intent="Test business intent")

    assert session.session_id is not None
    assert session.initial_intent == "Test business intent"
    assert len(session.conversation_history) == 0
    assert len(session.extracted_requirements) == 0
    assert session.is_complete is False
    assert session.data_prp_generated is False


def test_add_turn() -> None:
    """Test adding a conversation turn."""
    session = PlanningSession(initial_intent="Test intent")

    turn = session.add_turn("user", "This is my response")

    assert len(session.conversation_history) == 1
    assert turn.speaker == "user"
    assert turn.content == "This is my response"
    assert turn.turn_number == 0


def test_add_multiple_turns() -> None:
    """Test adding multiple conversation turns."""
    session = PlanningSession(initial_intent="Test intent")

    session.add_turn("user", "User message 1")
    session.add_turn("assistant", "Assistant message 1")
    session.add_turn("user", "User message 2")

    assert len(session.conversation_history) == 3
    assert session.conversation_history[0].turn_number == 0
    assert session.conversation_history[1].turn_number == 1
    assert session.conversation_history[2].turn_number == 2


def test_add_requirement() -> None:
    """Test adding an extracted requirement."""
    session = PlanningSession(initial_intent="Test intent")
    session.add_turn("user", "Regional managers")

    req = session.add_requirement("audience", "Regional managers", source_turn=0)

    assert "audience" in session.extracted_requirements
    assert len(session.extracted_requirements["audience"]) == 1
    assert req.category == "audience"
    assert req.value == "Regional managers"
    assert req.source_turn == 0


def test_add_multiple_requirements_same_category() -> None:
    """Test adding multiple requirements in the same category."""
    session = PlanningSession(initial_intent="Test intent")

    session.add_requirement("metrics", "Sales volume", source_turn=0)
    session.add_requirement("metrics", "Revenue", source_turn=1)

    assert len(session.extracted_requirements["metrics"]) == 2


def test_get_conversation_text() -> None:
    """Test getting conversation as formatted text."""
    session = PlanningSession(initial_intent="Test intent")

    session.add_turn("user", "Hello")
    session.add_turn("assistant", "How can I help?")
    session.add_turn("user", "I need insights")

    text = session.get_conversation_text()

    assert "User: Hello" in text
    assert "Assistant: How can I help?" in text
    assert "User: I need insights" in text


def test_get_turn_count() -> None:
    """Test getting the count of conversation turns."""
    session = PlanningSession(initial_intent="Test intent")

    assert session.get_turn_count() == 0

    session.add_turn("user", "Message 1")
    assert session.get_turn_count() == 1

    session.add_turn("assistant", "Message 2")
    assert session.get_turn_count() == 2


def test_conversation_turn_defaults() -> None:
    """Test ConversationTurn default values."""
    turn = ConversationTurn(turn_number=0, speaker="user", content="Test")

    assert turn.timestamp is not None
    assert turn.speaker == "user"
    assert turn.content == "Test"


def test_extracted_requirement_creation() -> None:
    """Test creating an extracted requirement."""
    req = ExtractedRequirement(
        category="objective", value="Analyze sales trends", source_turn=1
    )

    assert req.category == "objective"
    assert req.value == "Analyze sales trends"
    assert req.source_turn == 1


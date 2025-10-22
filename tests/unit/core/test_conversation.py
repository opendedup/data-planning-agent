"""
Tests for Conversation Management

Tests conversation state tracking and session management.
"""

import pytest

from data_planning_agent.core.conversation import ConversationManager
from data_planning_agent.models.session import PlanningSession


def test_create_session() -> None:
    """Test creating a new planning session."""
    manager = ConversationManager()

    session = manager.create_session("Test business intent")

    assert session.initial_intent == "Test business intent"
    assert session.session_id in manager.sessions
    assert len(session.conversation_history) == 0
    assert not session.is_complete


def test_get_session() -> None:
    """Test retrieving a session by ID."""
    manager = ConversationManager()
    session = manager.create_session("Test intent")

    retrieved = manager.get_session(session.session_id)

    assert retrieved is not None
    assert retrieved.session_id == session.session_id
    assert retrieved.initial_intent == "Test intent"


def test_get_nonexistent_session() -> None:
    """Test retrieving a non-existent session returns None."""
    manager = ConversationManager()

    retrieved = manager.get_session("nonexistent-id")

    assert retrieved is None


def test_update_session() -> None:
    """Test updating a session."""
    manager = ConversationManager()
    session = manager.create_session("Test intent")

    session.add_turn("user", "User response")
    manager.update_session(session)

    retrieved = manager.get_session(session.session_id)
    assert len(retrieved.conversation_history) == 1
    assert retrieved.conversation_history[0].content == "User response"


def test_delete_session() -> None:
    """Test deleting a session."""
    manager = ConversationManager()
    session = manager.create_session("Test intent")

    result = manager.delete_session(session.session_id)

    assert result is True
    assert manager.get_session(session.session_id) is None


def test_delete_nonexistent_session() -> None:
    """Test deleting a non-existent session."""
    manager = ConversationManager()

    result = manager.delete_session("nonexistent-id")

    assert result is False


def test_list_sessions() -> None:
    """Test listing all active sessions."""
    manager = ConversationManager()

    session1 = manager.create_session("Intent 1")
    session2 = manager.create_session("Intent 2")

    session_ids = manager.list_sessions()

    assert len(session_ids) == 2
    assert session1.session_id in session_ids
    assert session2.session_id in session_ids


def test_get_session_count() -> None:
    """Test getting the count of active sessions."""
    manager = ConversationManager()

    assert manager.get_session_count() == 0

    manager.create_session("Intent 1")
    assert manager.get_session_count() == 1

    manager.create_session("Intent 2")
    assert manager.get_session_count() == 2


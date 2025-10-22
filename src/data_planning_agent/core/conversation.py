"""
Conversation Management

Manages conversation state and session tracking for planning sessions.
"""

import logging
from typing import Dict, Optional
from uuid import uuid4

from ..models.session import PlanningSession

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages planning conversation sessions.

    Provides session storage and retrieval (in-memory for MVP).
    """

    def __init__(self):
        """Initialize conversation manager."""
        self.sessions: Dict[str, PlanningSession] = {}
        logger.info("Initialized conversation manager")

    def create_session(self, initial_intent: str) -> PlanningSession:
        """
        Create a new planning session.

        Args:
            initial_intent: The initial business intent from the user

        Returns:
            New PlanningSession
        """
        session = PlanningSession(initial_intent=initial_intent)
        self.sessions[session.session_id] = session

        logger.info(f"Created session: {session.session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[PlanningSession]:
        """
        Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            PlanningSession if found, None otherwise
        """
        session = self.sessions.get(session_id)
        if session is None:
            logger.warning(f"Session not found: {session_id}")
        return session

    def update_session(self, session: PlanningSession) -> None:
        """
        Update a session.

        Args:
            session: Session to update
        """
        self.sessions[session.session_id] = session
        logger.debug(f"Updated session: {session.session_id}")

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False

    def list_sessions(self) -> list[str]:
        """
        List all active session IDs.

        Returns:
            List of session IDs
        """
        return list(self.sessions.keys())

    def get_session_count(self) -> int:
        """
        Get the number of active sessions.

        Returns:
            Number of sessions
        """
        return len(self.sessions)


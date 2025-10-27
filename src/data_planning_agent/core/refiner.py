"""
Requirement Refiner

Orchestrates the conversational requirement gathering process.
"""

import logging
from typing import Optional

from ..clients.gemini_client import GeminiClient
from ..models.session import PlanningSession
from .conversation import ConversationManager

logger = logging.getLogger(__name__)


class RequirementRefiner:
    """
    Orchestrates conversational requirement refinement.

    Uses Gemini to generate questions and determine when requirements are complete.
    """

    def __init__(
        self,
        gemini_client: GeminiClient,
        conversation_manager: ConversationManager,
        max_turns: int = 3,
    ):
        """
        Initialize requirement refiner.

        Args:
            gemini_client: Client for Gemini API
            conversation_manager: Manager for conversation sessions
            max_turns: Maximum conversation turns before forcing completion
        """
        self.gemini_client = gemini_client
        self.conversation_manager = conversation_manager
        self.max_turns = max_turns

        logger.info(f"Initialized requirement refiner (max_turns={max_turns})")

    async def start_session(self, initial_intent: str) -> tuple[str, str]:
        """
        Start a new planning session with initial intent.

        Args:
            initial_intent: The user's initial business intent

        Returns:
            Tuple of (session_id, initial_questions)
        """
        # Create session
        session = self.conversation_manager.create_session(initial_intent)

        # Add initial user turn
        session.add_turn("user", initial_intent)

        # Generate initial questions
        questions = await self.gemini_client.generate_initial_questions(initial_intent)

        # Add assistant turn
        session.add_turn("assistant", questions)

        # Update session
        self.conversation_manager.update_session(session)

        logger.info(f"Started session {session.session_id} with initial questions")
        return (session.session_id, questions)

    async def continue_conversation(
        self, session_id: str, user_response: str
    ) -> tuple[str, bool]:
        """
        Continue conversation with user response.

        Args:
            session_id: Session identifier
            user_response: User's response to previous questions

        Returns:
            Tuple of (next_questions, is_complete)
            - next_questions: Follow-up questions or completion message
            - is_complete: True if requirements are sufficient

        Raises:
            ValueError: If session not found
        """
        # Get session
        session = self.conversation_manager.get_session(session_id)
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        # Add user response
        session.add_turn("user", user_response)

        # Generate follow-up questions or determine completion
        # Agent's assessment takes priority over turn count
        questions, is_complete = await self.gemini_client.generate_follow_up_questions(session)

        # Add assistant turn
        session.add_turn("assistant", questions)

        # Update completion status
        if is_complete:
            session.is_complete = True
            logger.info(f"Session {session_id} requirements complete (agent-driven)")
        
        # Check if we've hit max turns as circuit breaker
        elif session.get_turn_count() >= self.max_turns:
            logger.warning(f"Session {session_id} reached max turns circuit breaker, forcing completion")
            session.is_complete = True
            is_complete = True
            
            # Log assumption about incomplete requirements
            session.add_assumption(
                "Maximum conversation turns reached. Proceeding with available information, "
                "which may be incomplete."
            )
            
            questions = (
                "⚠️ Maximum conversation turns reached (circuit breaker activated).\n\n"
                "I'll work with the information gathered so far to generate your Data PRP. "
                "You can refine it further after generation if needed.\n\n"
                f"{questions}"
            )

        # Update session
        self.conversation_manager.update_session(session)

        return (questions, is_complete)

    def get_session(self, session_id: str) -> Optional[PlanningSession]:
        """
        Get a session by ID.

        Args:
            session_id: Session identifier

        Returns:
            PlanningSession if found, None otherwise
        """
        return self.conversation_manager.get_session(session_id)


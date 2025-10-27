"""
Session Data Models

Models for tracking conversation state and session data.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ConversationTurn(BaseModel):
    """
    A single turn in the conversation.

    Represents one exchange: either a user response or an AI question set.
    """

    turn_number: int = Field(description="Turn number in the conversation (0-indexed)")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When this turn occurred"
    )
    speaker: str = Field(description="Who spoke: 'user' or 'assistant'")
    content: str = Field(description="The text content of this turn")


class ExtractedRequirement(BaseModel):
    """
    A structured requirement extracted from the conversation.

    Represents a piece of information gathered about the data product.
    """

    category: str = Field(
        description="Requirement category: objective, audience, metrics, dimensions, etc."
    )
    value: str = Field(description="The actual requirement value")
    source_turn: int = Field(description="Which conversation turn this came from")


class PlanningSession(BaseModel):
    """
    A complete planning session.

    Tracks the entire conversation and extracted requirements for generating
    a Data Product Requirement Prompt.
    """

    session_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique session identifier"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When the session was created"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When the session was last updated"
    )

    initial_intent: str = Field(description="The initial business intent provided by the user")

    conversation_history: List[ConversationTurn] = Field(
        default_factory=list, description="Complete conversation history"
    )

    extracted_requirements: Dict[str, List[ExtractedRequirement]] = Field(
        default_factory=dict, description="Requirements extracted from conversation, by category"
    )

    is_complete: bool = Field(default=False, description="Whether requirements gathering is complete")

    data_prp_generated: bool = Field(default=False, description="Whether Data PRP has been generated")

    data_prp_content: Optional[str] = Field(
        default=None, description="The generated Data PRP markdown content"
    )

    data_prp_path: Optional[str] = Field(
        default=None, description="Path where Data PRP was saved"
    )

    source_prp: Optional[str] = Field(
        default=None, description="Original PRP content if this is a modification session"
    )

    assumptions: List[str] = Field(
        default_factory=list, description="Assumptions and defaults made during planning"
    )

    def add_turn(self, speaker: str, content: str) -> ConversationTurn:
        """
        Add a turn to the conversation.

        Args:
            speaker: Who is speaking ('user' or 'assistant')
            content: The content of the turn

        Returns:
            The created ConversationTurn
        """
        turn = ConversationTurn(
            turn_number=len(self.conversation_history), speaker=speaker, content=content
        )
        self.conversation_history.append(turn)
        self.updated_at = datetime.now(timezone.utc)
        return turn

    def add_requirement(
        self, category: str, value: str, source_turn: Optional[int] = None
    ) -> ExtractedRequirement:
        """
        Add an extracted requirement.

        Args:
            category: Requirement category
            value: The requirement value
            source_turn: Which turn this came from (defaults to latest)

        Returns:
            The created ExtractedRequirement
        """
        if source_turn is None:
            source_turn = len(self.conversation_history) - 1

        requirement = ExtractedRequirement(category=category, value=value, source_turn=source_turn)

        if category not in self.extracted_requirements:
            self.extracted_requirements[category] = []

        self.extracted_requirements[category].append(requirement)
        self.updated_at = datetime.now(timezone.utc)
        return requirement

    def get_conversation_text(self) -> str:
        """
        Get the full conversation as text.

        Returns:
            Formatted conversation history
        """
        lines = []
        for turn in self.conversation_history:
            speaker_label = "User" if turn.speaker == "user" else "Assistant"
            lines.append(f"{speaker_label}: {turn.content}")
        return "\n\n".join(lines)

    def get_turn_count(self) -> int:
        """
        Get the number of conversation turns.

        Returns:
            Number of turns
        """
        return len(self.conversation_history)

    def add_assumption(self, description: str) -> str:
        """
        Add an assumption with auto-incrementing ID.

        Args:
            description: Description of the assumption

        Returns:
            The formatted assumption string with ID
        """
        assumption_id = len(self.assumptions) + 1
        formatted_assumption = f"[ASSUMPTION-{assumption_id:02d}]: {description}"
        self.assumptions.append(formatted_assumption)
        self.updated_at = datetime.now(timezone.utc)
        return formatted_assumption


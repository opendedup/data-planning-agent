"""
Data PRP Generator

Generates structured Data Product Requirement Prompts from conversation sessions.
"""

import logging
from typing import Optional

from ..clients.gemini_client import GeminiClient
from ..clients.storage_client import StorageClient
from ..models.session import PlanningSession

logger = logging.getLogger(__name__)


class PRPGenerator:
    """
    Generates Data Product Requirement Prompts.

    Converts conversation sessions into structured markdown documents.
    """

    def __init__(
        self,
        gemini_client: GeminiClient,
        storage_client: StorageClient,
    ):
        """
        Initialize PRP generator.

        Args:
            gemini_client: Client for Gemini API
            storage_client: Client for file storage
        """
        self.gemini_client = gemini_client
        self.storage_client = storage_client

        logger.info("Initialized PRP generator")

    async def generate_prp(
        self,
        session: PlanningSession,
        output_path: Optional[str] = None,
        save_to_file: bool = True,
    ) -> tuple[str, Optional[str]]:
        """
        Generate a Data PRP from a conversation session.

        Args:
            session: Planning session with conversation history
            output_path: Optional output path for saving
            save_to_file: Whether to save to file (default True)

        Returns:
            Tuple of (prp_content, file_path)
            - prp_content: The generated Data PRP markdown
            - file_path: Path where file was saved (None if not saved)

        Raises:
            ValueError: If session requirements are not complete
        """
        if not session.is_complete:
            logger.warning(
                f"Session {session.session_id} not marked complete, generating anyway"
            )

        # Generate Data PRP using Gemini
        logger.info(f"Generating Data PRP for session {session.session_id}")
        prp_content = await self.gemini_client.generate_data_prp(session)

        # Update session
        session.data_prp_content = prp_content
        session.data_prp_generated = True

        # Save to file if requested
        file_path = None
        if save_to_file:
            file_path = self.storage_client.write_file(prp_content, output_path)
            session.data_prp_path = file_path
            logger.info(f"Saved Data PRP to: {file_path}")

        return (prp_content, file_path)

    def format_prp_summary(
        self,
        prp_content: str,
        file_path: Optional[str] = None,
    ) -> str:
        """
        Format a summary of the generated PRP for display.

        Args:
            prp_content: The PRP content
            file_path: Optional file path

        Returns:
            Formatted summary text
        """
        lines = []
        lines.append("✅ Data Product Requirement Prompt Generated!")
        lines.append("")

        if file_path:
            lines.append(f"📄 Saved to: {file_path}")
            lines.append("")

        lines.append("📋 Preview:")
        lines.append("-" * 60)

        # Include first 500 characters of PRP
        preview = prp_content[:500]
        if len(prp_content) > 500:
            preview += "..."

        lines.append(preview)
        lines.append("-" * 60)

        return "\n".join(lines)


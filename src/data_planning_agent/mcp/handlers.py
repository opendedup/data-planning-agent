"""
MCP Request Handlers

Implements the business logic for handling MCP tool calls.
"""

import logging
from typing import Any, Dict, List

from mcp.types import TextContent

from ..clients.gemini_client import GeminiClient
from ..clients.storage_client import StorageClient
from ..core.conversation import ConversationManager
from ..core.prp_generator import PRPGenerator
from ..core.refiner import RequirementRefiner
from .config import PlanningAgentConfig

logger = logging.getLogger(__name__)


class MCPHandlers:
    """
    Handlers for MCP tool requests.

    Orchestrates the planning workflow using Gemini and storage clients.
    """

    def __init__(
        self,
        config: PlanningAgentConfig,
        gemini_client: GeminiClient,
        storage_client: StorageClient,
    ):
        """
        Initialize MCP handlers.

        Args:
            config: Configuration
            gemini_client: Gemini API client
            storage_client: Storage client
        """
        self.config = config
        self.gemini_client = gemini_client
        self.storage_client = storage_client

        # Initialize conversation manager
        self.conversation_manager = ConversationManager()

        # Initialize requirement refiner
        self.refiner = RequirementRefiner(
            gemini_client=gemini_client,
            conversation_manager=self.conversation_manager,
            max_turns=config.max_conversation_turns,
        )

        # Initialize PRP generator
        self.prp_generator = PRPGenerator(
            gemini_client=gemini_client, storage_client=storage_client
        )

        logger.info("Initialized MCP handlers")

    async def handle_start_planning_session(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """
        Handle start_planning_session tool call.

        Args:
            arguments: Tool arguments

        Returns:
            List of TextContent responses
        """
        try:
            initial_intent = arguments["initial_intent"]

            logger.info(f"Starting planning session with intent: {initial_intent[:100]}...")

            # Start session and get initial questions
            session_id, questions = await self.refiner.start_session(initial_intent)

            # Format response
            response = f"""✨ Planning Session Started!

**Session ID:** `{session_id}`

I'll help you refine your requirements through a series of questions. Please provide your responses, and I'll guide you through the process.

---

{questions}

---

**Next Step:** Use the `continue_conversation` tool with this session ID and your responses to continue."""

            return [TextContent(type="text", text=response)]

        except Exception as e:
            logger.error(f"Error in handle_start_planning_session: {e}", exc_info=True)
            return [TextContent(type="text", text=f"❌ Error: {str(e)}")]

    async def handle_continue_conversation(
        self, arguments: Dict[str, Any]
    ) -> List[TextContent]:
        """
        Handle continue_conversation tool call.

        Args:
            arguments: Tool arguments

        Returns:
            List of TextContent responses
        """
        try:
            session_id = arguments["session_id"]
            user_response = arguments["user_response"]

            logger.info(f"Continuing conversation for session: {session_id}")

            # Continue conversation
            next_questions, is_complete = await self.refiner.continue_conversation(
                session_id, user_response
            )

            # Format response based on completion status
            if is_complete:
                response = f"""✅ Requirements Gathering Complete!

{next_questions}

You now have enough information to generate a comprehensive Data Product Requirement Prompt.

**Next Step:** Use the `generate_data_prp` tool with session ID `{session_id}` to create the final document."""
            else:
                response = f"""📝 Let's continue refining the requirements...

{next_questions}

---

**Next Step:** Use the `continue_conversation` tool again with your responses."""

            return [TextContent(type="text", text=response)]

        except ValueError as e:
            logger.error(f"Validation error in handle_continue_conversation: {e}")
            return [TextContent(type="text", text=f"❌ Error: {str(e)}")]
        except Exception as e:
            logger.error(f"Error in handle_continue_conversation: {e}", exc_info=True)
            return [TextContent(type="text", text=f"❌ Error: {str(e)}")]

    async def handle_generate_data_prp(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """
        Handle generate_data_prp tool call.

        Args:
            arguments: Tool arguments

        Returns:
            List of TextContent responses
        """
        try:
            session_id = arguments["session_id"]
            output_path = arguments.get("output_path")
            save_to_file = arguments.get("save_to_file", True)

            logger.info(f"Generating Data PRP for session: {session_id}")

            # Get session
            session = self.refiner.get_session(session_id)
            if session is None:
                return [
                    TextContent(type="text", text=f"❌ Error: Session not found: {session_id}")
                ]

            # Generate PRP
            prp_content, file_path = await self.prp_generator.generate_prp(
                session=session, output_path=output_path, save_to_file=save_to_file
            )

            # Format response
            response_lines = ["✅ Data Product Requirement Prompt Generated!", ""]

            if file_path:
                response_lines.append(f"📄 **Saved to:** `{file_path}`")
                response_lines.append("")

            response_lines.append("📋 **Full Content:**")
            response_lines.append("")
            response_lines.append(prp_content)

            response = "\n".join(response_lines)

            return [TextContent(type="text", text=response)]

        except ValueError as e:
            logger.error(f"Validation error in handle_generate_data_prp: {e}")
            return [TextContent(type="text", text=f"❌ Error: {str(e)}")]
        except Exception as e:
            logger.error(f"Error in handle_generate_data_prp: {e}", exc_info=True)
            return [TextContent(type="text", text=f"❌ Error: {str(e)}")]


def format_tool_response(text: str) -> List[TextContent]:
    """
    Format a successful tool response.

    Args:
        text: Response text

    Returns:
        List with single TextContent
    """
    return [TextContent(type="text", text=text)]


def format_error_response(error_message: str, tool_name: str) -> List[TextContent]:
    """
    Format an error response.

    Args:
        error_message: Error message
        tool_name: Tool name that failed

    Returns:
        List with single TextContent
    """
    return [TextContent(type="text", text=f"❌ Error in {tool_name}: {error_message}")]


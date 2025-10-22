"""
MCP Tool Definitions

Defines the MCP tools exposed by the Planning Agent.
"""

from mcp.types import Tool

# Tool names
START_PLANNING_SESSION_TOOL = "start_planning_session"
CONTINUE_CONVERSATION_TOOL = "continue_conversation"
GENERATE_DATA_PRP_TOOL = "generate_data_prp"


def get_available_tools() -> list[Tool]:
    """
    Get list of available MCP tools.

    Returns:
        List of Tool definitions
    """
    return [
        Tool(
            name=START_PLANNING_SESSION_TOOL,
            description=(
                "Start a new planning session to gather requirements for a data product. "
                "Provide an initial business intent, and the agent will ask clarifying questions "
                "to refine the requirements. Returns a session ID and initial questions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "initial_intent": {
                        "type": "string",
                        "description": (
                            "The high-level business intent or goal. "
                            "Example: 'We want to provide the merchandising team at Costco "
                            "insights into trending items in region 7'"
                        ),
                    }
                },
                "required": ["initial_intent"],
            },
        ),
        Tool(
            name=CONTINUE_CONVERSATION_TOOL,
            description=(
                "Continue an existing planning conversation by providing responses to questions. "
                "The agent will ask follow-up questions or indicate when requirements are complete."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The session ID from start_planning_session",
                    },
                    "user_response": {
                        "type": "string",
                        "description": (
                            "Your responses to the agent's questions. For multiple choice questions, "
                            "include the letter (a, b, c, d) and any additional details if needed."
                        ),
                    },
                },
                "required": ["session_id", "user_response"],
            },
        ),
        Tool(
            name=GENERATE_DATA_PRP_TOOL,
            description=(
                "Generate a complete Data Product Requirement Prompt (Data PRP) from a session. "
                "Creates a structured markdown document with executive summary, business context, "
                "data requirements, and success criteria. Optionally saves to a file."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The session ID from start_planning_session",
                    },
                    "output_path": {
                        "type": "string",
                        "description": (
                            "Optional path where to save the Data PRP. "
                            "Supports GCS paths (gs://bucket/path/file.md) or local paths. "
                            "If not provided, uses default output directory with timestamp."
                        ),
                    },
                    "save_to_file": {
                        "type": "boolean",
                        "description": "Whether to save to file (default: true)",
                        "default": True,
                    },
                },
                "required": ["session_id"],
            },
        ),
    ]


def validate_tool_params(params: dict, tool_name: str) -> None:
    """
    Validate tool parameters.

    Args:
        params: Parameter dictionary
        tool_name: Tool name

    Raises:
        ValueError: If parameters are invalid
    """
    if tool_name == START_PLANNING_SESSION_TOOL:
        if "initial_intent" not in params:
            raise ValueError("Missing required parameter: initial_intent")
        if not isinstance(params["initial_intent"], str):
            raise ValueError("initial_intent must be a string")
        if not params["initial_intent"].strip():
            raise ValueError("initial_intent cannot be empty")

    elif tool_name == CONTINUE_CONVERSATION_TOOL:
        if "session_id" not in params:
            raise ValueError("Missing required parameter: session_id")
        if "user_response" not in params:
            raise ValueError("Missing required parameter: user_response")
        if not isinstance(params["session_id"], str):
            raise ValueError("session_id must be a string")
        if not isinstance(params["user_response"], str):
            raise ValueError("user_response must be a string")
        if not params["user_response"].strip():
            raise ValueError("user_response cannot be empty")

    elif tool_name == GENERATE_DATA_PRP_TOOL:
        if "session_id" not in params:
            raise ValueError("Missing required parameter: session_id")
        if not isinstance(params["session_id"], str):
            raise ValueError("session_id must be a string")
        if "output_path" in params and not isinstance(params["output_path"], str):
            raise ValueError("output_path must be a string")
        if "save_to_file" in params and not isinstance(params["save_to_file"], bool):
            raise ValueError("save_to_file must be a boolean")

    else:
        raise ValueError(f"Unknown tool: {tool_name}")


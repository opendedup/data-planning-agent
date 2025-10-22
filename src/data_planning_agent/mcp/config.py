"""
MCP Service Configuration

Configuration management for the MCP service using environment variables.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


class PlanningAgentConfig(BaseModel):
    """
    Configuration for Planning Agent MCP service.

    All configuration loaded from environment variables.
    Follows security best practices - no hardcoded credentials.
    """

    # Gemini Configuration
    gemini_api_key: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", ""),
        description="Gemini API key for conversational AI",
    )

    gemini_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-pro"),
        description="Gemini model name",
    )

    # Output Configuration
    output_dir: str = Field(
        default_factory=lambda: os.getenv("OUTPUT_DIR", "./output"),
        description="Default output directory for Data PRPs (supports GCS gs:// paths)",
    )

    # MCP Service Configuration
    mcp_server_name: str = Field(
        default_factory=lambda: os.getenv("MCP_SERVER_NAME", "data-planning-agent"),
        description="MCP server name",
    )

    mcp_server_version: str = Field(
        default_factory=lambda: os.getenv("MCP_SERVER_VERSION", "1.0.0"),
        description="MCP server version",
    )

    mcp_transport: str = Field(
        default_factory=lambda: os.getenv("MCP_TRANSPORT", "stdio"),
        description="MCP transport mode: 'stdio' for local/subprocess or 'http' for network",
    )

    mcp_host: str = Field(
        default_factory=lambda: os.getenv("MCP_HOST", "0.0.0.0"),
        description="Host address for HTTP server",
    )

    mcp_port: int = Field(
        default_factory=lambda: int(os.getenv("MCP_PORT", "8080")),
        description="Port for MCP HTTP service",
    )

    # Conversation Configuration
    max_conversation_turns: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CONVERSATION_TURNS", "10")),
        description="Maximum conversation turns before forcing completion",
    )

    # Context Configuration
    context_dir: Optional[str] = Field(
        default_factory=lambda: os.getenv("CONTEXT_DIR"),
        description="Optional directory containing organizational context markdown files (local or GCS)",
    )

    # Logging Configuration
    log_level: str = Field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"),
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    def validate_required_fields(self) -> None:
        """
        Validate that required configuration fields are set.

        Raises:
            ValueError: If required fields are missing
        """
        required_fields = {
            "gemini_api_key": self.gemini_api_key,
        }

        missing = [field for field, value in required_fields.items() if not value]

        if missing:
            raise ValueError(
                f"Required configuration missing: {', '.join(missing)}. "
                f"Please set the following environment variables: "
                f"{', '.join(f'{field.upper()}' for field in missing)}"
            )

    model_config = {
        "arbitrary_types_allowed": True,
    }


def load_config() -> PlanningAgentConfig:
    """
    Load and validate Planning Agent configuration from environment.

    Returns:
        PlanningAgentConfig instance

    Raises:
        ValueError: If required configuration is missing
    """
    config = PlanningAgentConfig()
    config.validate_required_fields()
    return config


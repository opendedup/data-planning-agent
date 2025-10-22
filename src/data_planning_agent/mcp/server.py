"""
MCP Server Implementation

Main MCP server that exposes planning tools via the Model Context Protocol.
"""

import asyncio
import logging
from typing import Any, Dict, Sequence

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from ..clients.gemini_client import GeminiClient
from ..clients.storage_client import StorageClient
from .config import PlanningAgentConfig, load_config
from .handlers import MCPHandlers
from .tools import (
    CONTINUE_CONVERSATION_TOOL,
    GENERATE_DATA_PRP_TOOL,
    START_PLANNING_SESSION_TOOL,
    get_available_tools,
    validate_tool_params,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_mcp_server(config: PlanningAgentConfig | None = None) -> Server:
    """
    Create and configure MCP server.

    Args:
        config: MCP configuration (loads from env if not provided)

    Returns:
        Configured MCP Server instance
    """
    # Load config if not provided
    if config is None:
        config = load_config()

    # Set logging level
    logging.getLogger().setLevel(config.log_level)

    # Load organizational context if configured
    logger.info("Loading organizational context...")
    from ..clients.context_loader import load_context_from_directory

    context = load_context_from_directory(config.context_dir)

    if context:
        logger.info(f"Loaded organizational context ({len(context)} chars)")
    else:
        logger.info("No organizational context loaded")

    # Initialize clients
    logger.info("Initializing Gemini client...")
    gemini_client = GeminiClient(
        api_key=config.gemini_api_key,
        model_name=config.gemini_model,
        temperature=0.7,
        context=context,
    )

    logger.info("Initializing storage client...")
    storage_client = StorageClient(default_output_dir=config.output_dir)

    # Initialize handlers
    logger.info("Initializing MCP handlers...")
    handlers = MCPHandlers(config=config, gemini_client=gemini_client, storage_client=storage_client)

    # Create MCP server
    server = Server(config.mcp_server_name)

    logger.info(
        f"MCP Server '{config.mcp_server_name}' v{config.mcp_server_version} initialized"
    )

    # Register list_tools handler
    @server.list_tools()
    async def handle_list_tools() -> list[Tool]:
        """
        List available MCP tools.

        Returns:
            List of available tools
        """
        logger.debug("Listing available tools")
        return get_available_tools()

    # Register call_tool handler
    @server.call_tool()
    async def handle_call_tool(
        name: str,
        arguments: Dict[str, Any] | None,
    ) -> Sequence[TextContent]:
        """
        Handle tool invocation.

        Args:
            name: Tool name
            arguments: Tool arguments

        Returns:
            Sequence of TextContent responses
        """
        logger.info(f"Tool called: {name}")

        # Default to empty dict if no arguments
        if arguments is None:
            arguments = {}

        try:
            # Validate parameters
            validate_tool_params(arguments, name)

            # Route to appropriate handler
            if name == START_PLANNING_SESSION_TOOL:
                return await handlers.handle_start_planning_session(arguments)

            elif name == CONTINUE_CONVERSATION_TOOL:
                return await handlers.handle_continue_conversation(arguments)

            elif name == GENERATE_DATA_PRP_TOOL:
                return await handlers.handle_generate_data_prp(arguments)

            else:
                error_msg = f"Unknown tool: {name}"
                logger.error(error_msg)
                return [TextContent(type="text", text=f"Error: {error_msg}")]

        except Exception as e:
            logger.error(f"Error handling tool {name}: {e}", exc_info=True)
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    return server


async def main() -> None:
    """
    Main entry point for MCP server (stdio mode only).

    Runs the server using stdio transport for local development
    and subprocess communication.
    """
    try:
        # Load configuration
        logger.info("Loading MCP configuration from environment...")
        config = load_config()

        logger.info(f"Starting MCP server: {config.mcp_server_name} v{config.mcp_server_version}")
        logger.info("Transport: stdio")
        logger.info(f"Gemini Model: {config.gemini_model}")
        logger.info(f"Output Directory: {config.output_dir}")

        # Create server
        server = create_mcp_server(config)

        # Run server with stdio transport
        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP server running on stdio...")

            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name=config.mcp_server_name,
                    server_version=config.mcp_server_version,
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )

    except KeyboardInterrupt:
        logger.info("Server interrupted by user")

    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise


def run_server() -> None:
    """
    Synchronous wrapper for running the server.

    Determines transport mode and runs appropriate server:
    - stdio: For local development (asyncio)
    - http: For containerized deployment (uvicorn)
    """
    # Load config to determine transport mode
    config = load_config()

    logger.info(f"Starting MCP server: {config.mcp_server_name} v{config.mcp_server_version}")
    logger.info(f"Transport: {config.mcp_transport}")
    logger.info(f"Gemini Model: {config.gemini_model}")
    logger.info(f"Output Directory: {config.output_dir}")

    if config.mcp_transport.lower() == "http":
        # HTTP transport - run uvicorn server
        from .http_server import run_http_server

        logger.info(f"Starting HTTP server on {config.mcp_host}:{config.mcp_port}")
        run_http_server(host=config.mcp_host, port=config.mcp_port)
    else:
        # stdio transport - run async server
        logger.info("Using stdio transport (for local/subprocess communication)")
        asyncio.run(main())


if __name__ == "__main__":
    run_server()


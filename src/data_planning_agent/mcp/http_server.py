"""
MCP HTTP Server Implementation

Network-based MCP server using FastAPI and SSE for remote client connections.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..clients.gemini_client import GeminiClient
from ..clients.storage_client import StorageClient
from .config import load_config
from .handlers import MCPHandlers
from .tools import (
    CONTINUE_CONVERSATION_TOOL,
    GENERATE_DATA_PRP_TOOL,
    START_PLANNING_SESSION_TOOL,
    validate_tool_params,
)

logger = logging.getLogger(__name__)

# Global instances
config_instance = None
handlers_instance = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    """
    FastAPI lifespan context manager for startup/shutdown.

    Args:
        app: FastAPI application instance

    Yields:
        Control to the application
    """
    global config_instance, handlers_instance

    # Startup
    logger.info("Starting MCP HTTP server...")
    config_instance = load_config()

    logger.info(f"Gemini Model: {config_instance.gemini_model}")
    logger.info(f"Output Directory: {config_instance.output_dir}")

    # Load organizational context if configured
    logger.info("Loading organizational context...")
    from ..clients.context_loader import load_context_from_directory

    context = load_context_from_directory(config_instance.context_dir)

    if context:
        logger.info(f"Loaded organizational context ({len(context)} chars)")
    else:
        logger.info("No organizational context loaded")

    # Initialize clients and handlers
    logger.info("Initializing Gemini client...")
    gemini_client = GeminiClient(
        api_key=config_instance.gemini_api_key,
        model_name=config_instance.gemini_model,
        temperature=0.7,
        context=context,
    )

    logger.info("Initializing storage client...")
    storage_client = StorageClient(default_output_dir=config_instance.output_dir)

    logger.info("Initializing MCP handlers...")
    handlers_instance = MCPHandlers(
        config=config_instance, gemini_client=gemini_client, storage_client=storage_client
    )

    logger.info("MCP HTTP server initialized")

    yield

    # Shutdown
    logger.info("Shutting down MCP HTTP server...")


def create_http_app() -> FastAPI:
    """
    Create FastAPI application for MCP HTTP server.

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Data Planning MCP Service",
        description="Model Context Protocol service for automated Data Product Requirement gathering",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Middleware to log incoming requests."""
        logger.debug(f"{request.method} {request.url.path}")
        response = await call_next(request)

        if response.status_code >= 400:
            logger.warning(f"{request.method} {request.url.path} -> {response.status_code}")
        else:
            logger.debug(f"{request.method} {request.url.path} -> {response.status_code}")

        return response

    @app.get("/health")
    async def health_check() -> Dict[str, str]:
        """Health check endpoint for container orchestration."""
        return {
            "status": "healthy",
            "service": "data-planning-mcp",
            "transport": "http",
        }

    @app.get("/")
    async def root(request: Request) -> Any:
        """Root endpoint with service information or SSE stream."""
        accept_header = request.headers.get("accept", "")
        if "text/event-stream" in accept_header:
            logger.debug("Opening SSE stream for MCP notifications")

            async def event_stream():
                """Generate SSE events for MCP notifications."""
                try:
                    yield ":"  # Empty comment to flush connection
                    while True:
                        await asyncio.sleep(15)
                        yield ": ping\n\n"
                except asyncio.CancelledError:
                    logger.debug("SSE stream closed")
                    raise

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        return {
            "service": "data-planning-mcp",
            "version": config_instance.mcp_server_version if config_instance else "unknown",
            "protocol": "MCP JSON-RPC 2.0",
            "transport": "HTTP",
            "endpoints": {
                "health": "/health",
                "jsonrpc": "/ (POST for JSON-RPC)",
                "sse": "/ (GET with Accept: text/event-stream)",
                "tools": "/mcp/tools (legacy REST)",
                "call_tool": "/mcp/call-tool (legacy REST)",
            },
        }

    @app.post("/")
    async def jsonrpc_handler(request: Request) -> Dict[str, Any]:
        """JSON-RPC 2.0 endpoint for MCP protocol."""
        if not handlers_instance:
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32603, "message": "MCP server not initialized"},
            }

        try:
            body = await request.json()
            rpc_id = body.get("id")
            method = body.get("method")
            params = body.get("params", {})

            logger.debug(f"JSON-RPC method: {method}")

            # Handle notifications
            if rpc_id is None:
                if method == "notifications/initialized":
                    logger.debug("Client sent initialized notification")
                    return {}
                else:
                    logger.warning(f"Unknown notification method: {method}")
                    return {}

            # Handle initialize
            if method == "initialize":
                logger.info("MCP client initializing connection")
                return {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {
                            "tools": {},
                            "prompts": {},
                            "resources": {},
                            "logging": {},
                        },
                        "serverInfo": {
                            "name": "data-planning-mcp",
                            "version": (
                                config_instance.mcp_server_version
                                if config_instance
                                else "1.0.0"
                            ),
                        },
                    },
                }

            # Handle tools/list
            elif method == "tools/list":
                logger.debug("Listing available tools")
                from .tools import get_available_tools

                tools = get_available_tools()

                return {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": {
                        "tools": [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "inputSchema": tool.inputSchema,
                            }
                            for tool in tools
                        ]
                    },
                }

            # Handle tools/call
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                if not tool_name:
                    return {
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "error": {"code": -32602, "message": "Missing 'name' parameter"},
                    }

                logger.info(f"Calling tool via JSON-RPC: {tool_name}")

                # Validate parameters
                try:
                    validate_tool_params(arguments, tool_name)
                except Exception as e:
                    return {
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "error": {"code": -32602, "message": f"Invalid parameters: {str(e)}"},
                    }

                # Route to appropriate handler
                if tool_name == START_PLANNING_SESSION_TOOL:
                    result = await handlers_instance.handle_start_planning_session(arguments)
                elif tool_name == CONTINUE_CONVERSATION_TOOL:
                    result = await handlers_instance.handle_continue_conversation(arguments)
                elif tool_name == GENERATE_DATA_PRP_TOOL:
                    result = await handlers_instance.handle_generate_data_prp(arguments)
                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                    }

                # Format response
                return {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "result": {
                        "content": [{"type": content.type, "text": content.text} for content in result]
                    },
                }

            else:
                return {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

        except json.JSONDecodeError:
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            }
        except Exception as e:
            logger.error(f"JSON-RPC error: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "id": rpc_id if "rpc_id" in locals() else None,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            }

    @app.get("/mcp/tools")
    async def list_tools() -> Dict[str, Any]:
        """List available MCP tools."""
        if not handlers_instance:
            raise HTTPException(status_code=503, detail="MCP server not initialized")

        try:
            from .tools import get_available_tools

            tools = get_available_tools()

            return {
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema,
                    }
                    for tool in tools
                ]
            }
        except Exception as e:
            logger.error(f"Error listing tools: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/mcp/call-tool")
    async def call_tool(request: Request) -> Dict[str, Any]:
        """Call an MCP tool."""
        if not handlers_instance:
            raise HTTPException(status_code=503, detail="MCP server not initialized")

        try:
            body = await request.json()
            tool_name = body.get("name")
            arguments = body.get("arguments", {})

            if not tool_name:
                raise HTTPException(status_code=400, detail="Missing 'name' in request")

            logger.info(f"Tool called: {tool_name}")

            # Validate parameters
            validate_tool_params(arguments, tool_name)

            # Route to appropriate handler
            if tool_name == START_PLANNING_SESSION_TOOL:
                result = await handlers_instance.handle_start_planning_session(arguments)
            elif tool_name == CONTINUE_CONVERSATION_TOOL:
                result = await handlers_instance.handle_continue_conversation(arguments)
            elif tool_name == GENERATE_DATA_PRP_TOOL:
                result = await handlers_instance.handle_generate_data_prp(arguments)
            else:
                raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")

            # Format response
            return {
                "result": [{"type": content.type, "text": content.text} for content in result]
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error calling tool: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return app


def run_http_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """
    Run the MCP HTTP server.

    Args:
        host: Host address to bind to
        port: Port to listen on
    """
    import uvicorn

    app = create_http_app()

    logger.info(f"Starting MCP HTTP server on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    config = load_config()
    logging.getLogger().setLevel(config.log_level)
    run_http_server(host=config.mcp_host, port=config.mcp_port)


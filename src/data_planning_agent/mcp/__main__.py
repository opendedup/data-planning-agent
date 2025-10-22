"""
MCP Module Entry Point

Enables running the MCP server via: python -m data_planning_agent.mcp
"""

from .server import run_server

if __name__ == "__main__":
    run_server()


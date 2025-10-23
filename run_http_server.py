#!/usr/bin/env python3
"""
Standalone script to launch the Data Planning MCP HTTP server.

Prerequisites:
    - Python 3.10 or higher
    - All dependencies installed (run: poetry install)

Usage:
    python run_http_server.py
    python run_http_server.py --port 9000
    python run_http_server.py -p 9000 --log-level DEBUG

For more options, run:
    python run_http_server.py --help

Alternative (using Poetry):
    poetry run planning-agent-http [options]
"""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the CLI
from data_planning_agent.mcp.cli import main

if __name__ == "__main__":
    sys.exit(main())


#!/bin/bash
#
# Convenience script to launch the Data Planning MCP HTTP server
#
# Usage:
#   ./scripts/start-http-server.sh              # Start on default port 8082
#   ./scripts/start-http-server.sh -p 9000      # Start on port 9000
#   ./scripts/start-http-server.sh --help       # Show help
#

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Run the CLI using Poetry
poetry run planning-agent-http "$@"


"""
Command-line interface for launching the Data Planning MCP HTTP server.

Provides a simple CLI to start the MCP server as an HTTP endpoint.
"""

import argparse
import logging
import os
import sys
from typing import Optional

from dotenv import load_dotenv


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging for the CLI.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Launch the Data Planning MCP server as an HTTP endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server on default port 8082
  %(prog)s

  # Start server on custom port
  %(prog)s --port 9000
  %(prog)s -p 9000

  # Start server on specific host
  %(prog)s --host 127.0.0.1 --port 8082

  # Enable debug logging
  %(prog)s --log-level DEBUG
        """,
    )

    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8082,
        help="TCP port for the HTTP server (default: 8082)",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def validate_port(port: int) -> None:
    """
    Validate that the port number is in valid range.

    Args:
        port: Port number to validate

    Raises:
        ValueError: If port is not in valid range (1-65535)
    """
    if not 1 <= port <= 65535:
        raise ValueError(f"Port must be between 1 and 65535, got {port}")


def main() -> int:
    """
    Main entry point for the CLI.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    # Load environment variables from .env file
    load_dotenv()

    # Parse command-line arguments
    args = parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Validate port
        validate_port(args.port)

        logger.info("Data Planning MCP HTTP Server")
        logger.info(f"Host: {args.host}")
        logger.info(f"Port: {args.port}")
        logger.info(f"Log Level: {args.log_level}")

        # Set environment variables for HTTP transport
        os.environ["MCP_TRANSPORT"] = "http"
        os.environ["MCP_HOST"] = args.host
        os.environ["MCP_PORT"] = str(args.port)
        os.environ["LOG_LEVEL"] = args.log_level

        # Import and run the HTTP server
        from .http_server import run_http_server

        logger.info("Starting MCP HTTP server...")
        run_http_server(host=args.host, port=args.port)

        return 0

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
        return 0

    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())


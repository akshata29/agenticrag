#!/usr/bin/env python3
"""
Launcher for MCP Server

This launcher script provides a unified way to start the MCP server
in different modes with proper configuration.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "mcp_server"))

def main():
    """Main launcher function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Server Launcher")
    parser.add_argument("--mode", choices=["stdio", "http"], default="stdio",
                       help="Server mode")
    parser.add_argument("--http", action="store_true", help="Run in HTTP mode")
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP mode")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Convert --http flag to mode
    if args.http:
        args.mode = "http"
    
    # Import and run the streaming server
    from mcp_server.streaming_mcp_server import main as server_main
    
    # Override sys.argv to pass arguments to the server
    sys.argv = [
        "streaming_mcp_server.py",
        "--mode", args.mode,
        "--host", args.host,
        "--port", str(args.port)
    ]
    
    if args.debug:
        sys.argv.append("--debug")
    
    server_main()

if __name__ == "__main__":
    main()

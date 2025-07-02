#!/usr/bin/env python3
"""
MCP Server Launcher

Simple script to launch the Financial RAG MCP server in different modes.
Handles environment setup and configuration automatically.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

def setup_environment():
    """Setup Python path and environment variables"""
    project_root = Path(__file__).parent.parent
    backend_path = project_root / "backend"
    
    # Add project root and backend to Python path
    env = os.environ.copy()
    python_path_parts = [str(project_root), str(backend_path)]
    
    if 'PYTHONPATH' in env:
        python_path_parts.append(env['PYTHONPATH'])
    
    env['PYTHONPATH'] = os.pathsep.join(python_path_parts)
    
    return env

def launch_stdio_server():
    """Launch MCP server in stdin/stdout mode (for Claude Desktop)"""
    print("🚀 Starting Financial RAG MCP Server (stdin/stdout mode)")
    print("   Compatible with: Claude Desktop, MCP CLI clients")
    print("   Protocol: JSON-RPC over stdin/stdout")
    print("-" * 50)
    
    server_path = Path(__file__).parent / "main.py"
    env = setup_environment()
    
    try:
        subprocess.run([sys.executable, str(server_path)], env=env)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed: {e}")
        sys.exit(1)

def launch_http_server(host="127.0.0.1", port=8001):
    """Launch MCP server in HTTP mode (for web apps, VS Code)"""
    print("🌐 Starting Financial RAG MCP Server (HTTP mode)")
    print(f"   Server URL: http://{host}:{port}")
    print("   Compatible with: Web apps, VS Code extensions, REST clients")
    print("   Protocols: HTTP JSON-RPC, WebSocket, Server-Sent Events")
    print("-" * 50)
    
    server_path = Path(__file__).parent / "streaming_mcp_server.py"
    env = setup_environment()
    
    try:
        subprocess.run([
            sys.executable, str(server_path),
            "--mode", "http",
            "--host", host,
            "--port", str(port)
        ], env=env)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed: {e}")
        sys.exit(1)

def run_tests():
    """Run the MCP server test suite"""
    print("🧪 Running Financial RAG MCP Server Tests")
    print("   Testing: stdio, HTTP, WebSocket, SSE protocols")
    print("-" * 50)
    
    test_path = Path(__file__).parent / "test_enhanced_mcp.py"
    env = setup_environment()
    
    try:
        result = subprocess.run([sys.executable, str(test_path)], env=env)
        sys.exit(result.returncode)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)

def show_info():
    """Show server information and available tools"""
    print("📋 Financial RAG MCP Server Information")
    print("=" * 50)
    
    print("\n🔧 Available Tools:")
    tools = [
        "answer_financial_question - Comprehensive Q&A with RAG (streaming)",
        "search_financial_documents - Knowledge base search (streaming)",
        "verify_source_credibility - Source reliability assessment",
        "coordinate_multi_agent_analysis - Multi-agent coordination (streaming)",
        "get_knowledge_base_stats - System health and statistics"
    ]
    
    for tool in tools:
        print(f"   • {tool}")
    
    print("\n📚 Available Resources:")
    resources = [
        "financial://knowledge-base/statistics - KB health metrics",
        "financial://agents/capabilities - Agent capabilities",
        "financial://documents/types - Available document types",
        "financial://system/status - System status"
    ]
    
    for resource in resources:
        print(f"   • {resource}")
    
    print("\n🌐 Supported Protocols:")
    protocols = [
        "stdin/stdout - For Claude Desktop and CLI clients",
        "HTTP JSON-RPC - For web applications and REST clients",
        "WebSocket - For real-time bidirectional communication",
        "Server-Sent Events - For streaming updates and progress"
    ]
    
    for protocol in protocols:
        print(f"   • {protocol}")
    
    print("\n🎯 Integration Examples:")
    print("   • Claude Desktop: Use --stdio mode")
    print("   • VS Code Extension: Use --http mode")
    print("   • Web Application: Use --http mode with SSE")
    print("   • Real-time App: Use --http mode with WebSocket")

def main():
    parser = argparse.ArgumentParser(
        description="Financial RAG MCP Server Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launcher.py --stdio              # For Claude Desktop
  python launcher.py --http               # For web apps (default port 8000)
  python launcher.py --http --port 9000   # Custom port
  python launcher.py --test               # Run test suite
  python launcher.py --info               # Show server information
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stdio", action="store_true",
                      help="Launch in stdin/stdout mode (for Claude Desktop)")
    group.add_argument("--http", action="store_true",
                      help="Launch in HTTP mode (for web apps, VS Code)")
    group.add_argument("--test", action="store_true",
                      help="Run the test suite")
    group.add_argument("--info", action="store_true",
                      help="Show server information")
    
    parser.add_argument("--host", default="127.0.0.1",
                       help="Host for HTTP mode (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port for HTTP mode (default: 8000)")
    
    args = parser.parse_args()
    
    if args.stdio:
        launch_stdio_server()
    elif args.http:
        launch_http_server(args.host, args.port)
    elif args.test:
        run_tests()
    elif args.info:
        show_info()

if __name__ == "__main__":
    main()

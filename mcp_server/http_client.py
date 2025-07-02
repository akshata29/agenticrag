"""
HTTP Client for Financial RAG MCP Server

This client demonstrates how external applications (like Claude, VS Code, etc.)
can interact with the MCP server using HTTP protocols instead of stdin/stdout.

Supports:
- JSON-RPC over HTTP
- Server-Sent Events (SSE) for streaming
- WebSocket for bidirectional communication
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator
import aiohttp
import websockets
from datetime import datetime

logger = logging.getLogger(__name__)

class MCPHTTPClient:
    """HTTP client for MCP server communication"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def close(self):
        """Close client connections"""
        if self.session:
            await self.session.close()
        if self.websocket:
            await self.websocket.close()
    
    async def call_rpc(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a JSON-RPC call to the MCP server"""
        if not self.session:
            raise ValueError("Client session not initialized")
            
        request_data = {
            "jsonrpc": "2.0",
            "id": f"req_{datetime.utcnow().timestamp()}",
            "method": method,
            "params": params or {}
        }
        
        async with self.session.post(
            f"{self.base_url}/mcp/rpc",
            json=request_data,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}: {await response.text()}")
            
            result = await response.json()
            
            if "error" in result:
                raise Exception(f"RPC Error: {result['error']}")
            
            return result.get("result", {})
    
    async def stream_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Make a streaming request using Server-Sent Events"""
        if not self.session:
            raise ValueError("Client session not initialized")
            
        request_data = {
            "jsonrpc": "2.0",
            "id": f"stream_{datetime.utcnow().timestamp()}",
            "method": method,
            "params": params or {}
        }
        
        async with self.session.post(
            f"{self.base_url}/mcp/stream",
            json=request_data,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream"
            }
        ) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}: {await response.text()}")
            
            async for line in response.content:
                if line:
                    try:
                        # Parse SSE data
                        line_str = line.decode('utf-8').strip()
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # Remove 'data: ' prefix
                            data = json.loads(data_str)
                            yield data
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.error(f"Error parsing stream data: {e}")
                        continue
    
    async def websocket_connect(self) -> None:
        """Connect to MCP server via WebSocket"""
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.websocket = await websockets.connect(f"{ws_url}/mcp/ws")
    
    async def websocket_call(self, method: str, params: Optional[Dict[str, Any]] = None, stream: bool = False) -> Any:
        """Make a call via WebSocket"""
        if not self.websocket:
            await self.websocket_connect()
            
        request_data = {
            "jsonrpc": "2.0",
            "id": f"ws_{datetime.utcnow().timestamp()}",
            "method": method,
            "params": params or {},
            "stream": stream
        }
        
        await self.websocket.send(json.dumps(request_data))
        
        if stream:
            # Return async generator for streaming responses
            return self._websocket_stream()
        else:
            # Return single response
            response = await self.websocket.recv()
            return json.loads(response)
    
    async def _websocket_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream responses from WebSocket"""
        try:
            while True:
                response = await self.websocket.recv()
                data = json.loads(response)
                yield data
                
                # Check if this is the final result
                if data.get("type") in ["result", "error"]:
                    break
        except websockets.exceptions.ConnectionClosed:
            pass
    
    # Convenience methods for common operations
    async def answer_financial_question(
        self,
        question: str,
        context: str = "",
        verification_level: str = "thorough",
        stream: bool = False
    ) -> Any:
        """Answer a financial question"""
        params = {
            "question": question,
            "context": context,
            "verification_level": verification_level
        }
        
        if stream:
            return self.stream_request("answer_financial_question", params)
        else:
            return await self.call_rpc("answer_financial_question", params)
    
    async def search_documents(
        self,
        query: str,
        document_types: Optional[List[str]] = None,
        top_k: int = 10,
        stream: bool = False
    ) -> Any:
        """Search financial documents"""
        params = {
            "query": query,
            "document_types": document_types or [],
            "top_k": top_k
        }
        
        if stream:
            return self.stream_request("search_financial_documents", params)
        else:
            return await self.call_rpc("search_financial_documents", params)
    
    async def coordinate_agents(
        self,
        request_type: str,
        content: str,
        requirements: Optional[Dict[str, Any]] = None,
        stream: bool = False
    ) -> Any:
        """Coordinate multi-agent analysis"""
        params = {
            "request_type": request_type,
            "content": content,
            "requirements": requirements or {}
        }
        
        if stream:
            return self.stream_request("coordinate_multi_agent_analysis", params)
        else:
            return await self.call_rpc("coordinate_multi_agent_analysis", params)
    
    async def get_server_info(self) -> Dict[str, Any]:
        """Get server information and capabilities"""
        if not self.session:
            raise ValueError("Client session not initialized")
            
        async with self.session.get(f"{self.base_url}/mcp/info") as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}: {await response.text()}")
            return await response.json()
    
    async def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        if not self.session:
            raise ValueError("Client session not initialized")
            
        async with self.session.get(f"{self.base_url}/health") as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}: {await response.text()}")
            return await response.json()


# Claude-style client interface
class ClaudeCompatibleMCPClient:
    """
    MCP client that mimics Claude's interface for external tool use
    This demonstrates how Claude or similar AI assistants can integrate with the MCP server
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.client = MCPHTTPClient(base_url)
        self.tools = []
        self.initialized = False
    
    async def initialize(self):
        """Initialize the client and discover available tools"""
        async with self.client:
            # Get server info and available tools
            info = await self.client.get_server_info()
            self.tools = info.get("tools", [])
            self.initialized = True
            
            logger.info(f"Initialized MCP client with {len(self.tools)} tools")
            for tool in self.tools:
                logger.info(f"  - {tool['name']}: {tool['description']}")
    
    async def use_tool(self, tool_name: str, parameters: Dict[str, Any], stream: bool = False) -> Any:
        """Use a specific tool (Claude-style interface)"""
        if not self.initialized:
            await self.initialize()
        
        # Check if tool exists
        tool_exists = any(tool["name"] == tool_name for tool in self.tools)
        if not tool_exists:
            raise ValueError(f"Tool '{tool_name}' not available")
        
        async with self.client:
            if stream:
                return self.client.stream_request(tool_name, parameters)
            else:
                return await self.client.call_rpc(tool_name, parameters)
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools"""
        if not self.initialized:
            await self.initialize()
        return self.tools


# Example usage and demo functions
async def demo_basic_usage():
    """Demonstrate basic MCP HTTP client usage"""
    print("🔧 Demo: Basic MCP HTTP Client Usage")
    print("=" * 50)
    
    async with MCPHTTPClient() as client:
        # Health check
        health = await client.health_check()
        print(f"✅ Server health: {health['status']}")
        
        # Get server info
        info = await client.get_server_info()
        print(f"📋 Server: {info['server_info']['name']} v{info['server_info']['version']}")
        print(f"🔧 Available tools: {len(info['tools'])}")
        
        # Ask a financial question
        print("\n❓ Asking financial question...")
        result = await client.answer_financial_question(
            question="What are Apple's main revenue streams?",
            verification_level="basic"
        )
        
        print(f"📝 Answer: {result.get('answer', 'No answer')[:200]}...")
        print(f"🎯 Confidence: {result.get('confidence', 0.0):.2f}")
        print(f"📚 Sources: {len(result.get('sources', []))}")

async def demo_streaming():
    """Demonstrate streaming capabilities"""
    print("\n🌊 Demo: Streaming MCP Client")
    print("=" * 50)
    
    async with MCPHTTPClient() as client:
        print("🔄 Starting streaming financial analysis...")
        
        stream = client.stream_request(
            "answer_financial_question",
            {
                "question": "Analyze Microsoft's financial performance over the last year",
                "verification_level": "thorough",
                "use_multi_agent": True
            }
        )
        
        async for chunk in stream:
            chunk_type = chunk.get("type", "unknown")
            step = chunk.get("step", "")
            message = chunk.get("message", "")
            timestamp = chunk.get("timestamp", "")
            
            if chunk_type == "progress":
                print(f"  📋 [{step}] {message}")
            elif chunk_type == "partial_result":
                data = chunk.get("data", {})
                progress = data.get("progress", 0)
                print(f"  📊 Progress: {progress:.1f}%")
            elif chunk_type == "result":
                result = chunk.get("data", {})
                print(f"\n✅ Final result received:")
                print(f"   Answer length: {len(result.get('answer', ''))}")
                print(f"   Confidence: {result.get('confidence', 0.0):.2f}")
                print(f"   Sources: {len(result.get('sources', []))}")
                break
            elif chunk_type == "error":
                error = chunk.get("error", {})
                print(f"❌ Error: {error.get('message', 'Unknown error')}")
                break

async def demo_claude_interface():
    """Demonstrate Claude-compatible interface"""
    print("\n🤖 Demo: Claude-Compatible Interface")
    print("=" * 50)
    
    claude_client = ClaudeCompatibleMCPClient()
    
    # Initialize and get tools
    await claude_client.initialize()
    tools = await claude_client.get_available_tools()
    
    print(f"🔧 Available tools for Claude:")
    for tool in tools[:3]:  # Show first 3 tools
        print(f"  - {tool['name']}: {tool['description']}")
    
    # Use a tool
    print(f"\n🎯 Using tool: answer_financial_question")
    result = await claude_client.use_tool(
        "answer_financial_question",
        {
            "question": "What is Amazon's primary business model?",
            "verification_level": "basic"
        }
    )
    
    print(f"✅ Tool result:")
    print(f"   Success: {result.get('success', False)}")
    print(f"   Answer length: {len(result.get('answer', ''))}")

async def main():
    """Run all demos"""
    try:
        await demo_basic_usage()
        await demo_streaming()
        await demo_claude_interface()
        
        print("\n🎉 All demos completed successfully!")
        print("\n💡 Integration notes:")
        print("  - Claude can use this HTTP interface for financial analysis")
        print("  - VS Code extensions can leverage the WebSocket API")
        print("  - Web apps can use SSE for real-time updates")
        print("  - The original stdin/stdout protocol is still supported")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logging.error(f"Demo error: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())

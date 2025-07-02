#!/usr/bin/env python3
"""
Enhanced Financial RAG MCP Server with Streaming Support

This server extends the basic MCP server to support:
- Server-Sent Events (SSE) for real-time streaming
- HTTP protocol for web-based clients
- WebSocket support for bidirectional communication
- Streaming responses for long-running operations

Protocols Supported:
- MCP over stdin/stdout (original)
- MCP over HTTP with JSON-RPC
- MCP over WebSocket
- SSE for streaming responses
"""

import asyncio
import json
import logging
import os
import sys
import uvicorn
import platform
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator
from datetime import datetime
from contextlib import asynccontextmanager

# Fix Windows asyncio issues - aiodns requires SelectorEventLoop on Windows
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, WebSocket, Request, Response, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette import EventSourceResponse
import websockets

# Add the parent directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "backend"))

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_environment_with_isolation():
    """Load environment variables with proper backend/MCP isolation."""
    import os
    from dotenv import load_dotenv, dotenv_values
    
    # Store original environment
    original_env = dict(os.environ)
    
    # First, load backend environment variables only
    backend_env_files = [
        project_root / "mcp_server" / ".env.backend",
        project_root / "backend" / ".env",
        project_root / ".env"
    ]
    
    backend_loaded = False
    for env_file in backend_env_files:
        if env_file.exists():
            # Load backend vars into a clean environment context
            backend_vars = dotenv_values(env_file)
            for key, value in backend_vars.items():
                if value is not None:
                    os.environ[key] = value
            backend_loaded = True
            print(f"✅ Loaded backend environment from: {env_file}")
            break
    
    if not backend_loaded:
        print("⚠️ No backend .env file found. Using system environment variables.")
    
    # Store the backend environment state
    backend_env_state = dict(os.environ)
    
    # Load MCP-specific configuration (this may add extra vars)
    mcp_env_file = project_root / "mcp_server" / ".env.mcp"
    mcp_vars = {}
    if mcp_env_file.exists():
        mcp_vars = dotenv_values(mcp_env_file)
        for key, value in mcp_vars.items():
            if value is not None:
                os.environ[key] = value
        print(f"✅ Loaded MCP configuration from: {mcp_env_file}")
    
    return original_env, backend_env_state, mcp_vars

def temporarily_isolate_backend_env(backend_env_state):
    """Context manager to temporarily set environment to only backend vars."""
    import os
    from contextlib import contextmanager
    
    @contextmanager
    def isolated_env():
        current_env = dict(os.environ)
        try:
            # Clear current environment and set only backend vars
            os.environ.clear()
            os.environ.update(backend_env_state)
            yield
        finally:
            # Restore full environment
            os.environ.clear()
            os.environ.update(current_env)
    
    return isolated_env()

# Load environment variables with isolation
original_env, backend_env_state, mcp_vars = load_environment_with_isolation()

# Validate critical environment variables
required_vars = [
    "AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET",
    "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"
]

missing_vars = []
for var in required_vars:
    if not os.getenv(var):
        missing_vars.append(var)

if missing_vars:
    print(f"❌ Missing required environment variables: {missing_vars}")
    print("💡 Please ensure your .env file contains all required Azure credentials")
    sys.exit(1)

print("✅ All required environment variables found")

# Import backend services with isolated environment
try:
    with temporarily_isolate_backend_env(backend_env_state):
        from backend.app.services.azure_services import AzureServiceManager
        from backend.app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
        from backend.app.services.multi_agent_orchestrator import MultiAgentOrchestrator, AgentType
        from backend.app.services.rag_pipeline import RAGPipeline
        from backend.app.core.config import settings
    print("✅ Successfully imported Azure services")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running from the project root or the backend is properly set up")
    print(f"💡 Project root: {project_root}")
    print(f"💡 Backend path: {project_root / 'backend'}")
    print("💡 Ensure all dependencies are installed with: pip install -r requirements.txt")
    sys.exit(1)

# Import the original MCP server
try:
    from main import FinancialRAGMCPServer, MCPRequest, MCPResponse
    print("✅ Successfully imported MCP server base classes")
except ImportError as e:
    print(f"❌ Failed to import MCP server base: {e}")
    print("💡 Make sure main.py exists in the mcp_server directory")
    sys.exit(1)

class StreamingMCPServer(FinancialRAGMCPServer):
    """
    Enhanced MCP Server with streaming capabilities
    """
    
    def __init__(self):
        super().__init__()
        self.active_streams: Dict[str, asyncio.Queue] = {}
        self.websocket_connections: Dict[str, WebSocket] = {}
        
    async def stream_response(self, request_id: str, method: str, params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Stream response for long-running operations
        """
        try:
            if method == "answer_financial_question":
                async for chunk in self._stream_financial_answer(request_id, params):
                    yield chunk
            elif method == "search_financial_documents":
                async for chunk in self._stream_document_search(request_id, params):
                    yield chunk
            elif method == "coordinate_multi_agent_analysis":
                async for chunk in self._stream_multi_agent_analysis(request_id, params):
                    yield chunk
            else:
                # For non-streaming methods, return single response
                result = await self.handle_tool_call(method, params)
                yield json.dumps({
                    "id": request_id,
                    "type": "result",
                    "data": result
                })
                
        except Exception as e:
            yield json.dumps({
                "id": request_id,
                "type": "error",
                "error": {"code": -32603, "message": str(e)}
            })
    
    async def _stream_financial_answer(self, request_id: str, params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream financial question answering with progress updates"""
        try:
            question = params.get("question", "")
            verification_level = params.get("verification_level", "thorough")
            use_multi_agent = params.get("use_multi_agent", True)
            
            # Send progress updates
            yield json.dumps({
                "id": request_id,
                "type": "progress",
                "step": "initializing",
                "message": "Starting financial analysis...",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Step 1: Document search
            yield json.dumps({
                "id": request_id,
                "type": "progress",
                "step": "searching",
                "message": f"Searching knowledge base for: {question[:100]}...",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Search for relevant documents
            logger.info(f"🔍 Starting search for query: '{question[:100]}...'")
            logger.info(f"🔍 Using kb_manager: {self.kb_manager is not None}")
            logger.info(f"🔍 KB manager type: {type(self.kb_manager)}")
            
            search_results = await self.kb_manager.search_knowledge_base(
                query=question,
                top_k=10 if verification_level == "basic" else 20
            )
            
            logger.info(f"🔍 Search completed. Results count: {len(search_results)}")
            if search_results:
                logger.info(f"🔍 First result keys: {list(search_results[0].keys())}")
                logger.info(f"🔍 First result title: {search_results[0].get('title', 'No title')}")
            else:
                logger.warning("🔍 No search results returned from knowledge base")
            
            yield json.dumps({
                "id": request_id,
                "type": "progress",
                "step": "found_documents",
                "message": f"Found {len(search_results)} relevant documents",
                "data": {"document_count": len(search_results)},
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Step 2: Multi-agent analysis if enabled
            if use_multi_agent and self.orchestrator:
                yield json.dumps({
                    "id": request_id,
                    "type": "progress", 
                    "step": "multi_agent_analysis",
                    "message": "Coordinating multi-agent analysis...",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Process with orchestrator with timeout
                agent_request = {
                    "agent_type": AgentType.QA_AGENT.value,
                    "capability": "answer_financial_question",
                    "question": question,
                    "context": params.get("context", ""),
                    "verification_level": verification_level,
                    "search_results": search_results
                }
                
                try:
                    # Add timeout for orchestrator processing - increased to 90 seconds
                    result = await asyncio.wait_for(
                        self.orchestrator.process_request(
                            agent_request, 
                            f"stream_session_{request_id}"
                        ),
                        timeout=90.0  # 90 second timeout for orchestrator
                    )
                except asyncio.TimeoutError:
                    logger.error(f"⏱️ Orchestrator processing timed out after 90s for question: {question[:100]}...")
                    result = {
                        "answer": f"Unable to process question '{question}' due to timeout. Please try a simpler question or check system resources.",
                        "confidence": 0.0,
                        "sources": [],
                        "search_results_count": len(search_results),
                        "success": True,
                        "error": "timeout"
                    }
                except Exception as e:
                    logger.error(f"❌ Orchestrator processing failed: {e}")
                    result = {
                        "answer": f"Unable to process question '{question}' due to error: {str(e)}",
                        "confidence": 0.0,
                        "sources": [],
                        "search_results_count": len(search_results),
                        "success": True,
                        "error": str(e)
                    }
            else:
                # Basic processing without orchestrator
                yield json.dumps({
                    "id": request_id,
                    "type": "progress",
                    "step": "processing",
                    "message": "Processing question with RAG pipeline...",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                result = await self.rag_pipeline.process_question(
                    question=question,
                    context=params.get("context", ""),
                    search_results=search_results
                )
            
            # Step 3: Finalize and return result
            yield json.dumps({
                "id": request_id,
                "type": "progress",
                "step": "finalizing", 
                "message": "Finalizing response...",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Send final result
            yield json.dumps({
                "id": request_id,
                "type": "result",
                "data": result,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error in streaming financial answer: {e}")
            yield json.dumps({
                "id": request_id,
                "type": "error",
                "error": {"code": -32603, "message": str(e)},
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _stream_document_search(self, request_id: str, params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream document search with incremental results"""
        try:
            query = params.get("query", "")
            document_types = params.get("document_types", [])
            top_k = params.get("top_k", 10)
            
            yield json.dumps({
                "id": request_id,
                "type": "progress",
                "step": "searching",
                "message": f"Searching for documents: {query}",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Perform search
            filters = {}
            if document_types:
                filters["document_type"] = document_types
            
            results = await self.kb_manager.search_knowledge_base(
                query=query,
                top_k=top_k,
                filters=filters
            )
            
            # Stream results incrementally
            batch_size = 3
            for i in range(0, len(results), batch_size):
                batch = results[i:i + batch_size]
                
                yield json.dumps({
                    "id": request_id,
                    "type": "partial_result",
                    "step": "results",
                    "data": {
                        "batch": batch,
                        "batch_number": i // batch_size + 1,
                        "total_batches": (len(results) + batch_size - 1) // batch_size,
                        "progress": min(100, ((i + batch_size) / len(results)) * 100)
                    },
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Small delay to allow client processing
                await asyncio.sleep(0.1)
            
            # Send final summary
            yield json.dumps({
                "id": request_id,
                "type": "result",
                "data": {
                    "results": results,
                    "total_found": len(results),
                    "query": query,
                    "success": True
                },
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error in streaming document search: {e}")
            yield json.dumps({
                "id": request_id,
                "type": "error",
                "error": {"code": -32603, "message": str(e)},
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _stream_multi_agent_analysis(self, request_id: str, params: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream multi-agent coordination with agent progress updates"""
        try:
            request_type = params.get("request_type", "")
            content = params.get("content", "")
            
            yield json.dumps({
                "id": request_id,
                "type": "progress",
                "step": "coordinating",
                "message": f"Coordinating agents for: {request_type}",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Simulate agent coordination phases
            agents = ["document_processor", "financial_analyzer", "qa_agent"]
            
            for i, agent in enumerate(agents):
                yield json.dumps({
                    "id": request_id,
                    "type": "progress",
                    "step": f"agent_{agent}",
                    "message": f"Processing with {agent.replace('_', ' ').title()}...",
                    "data": {
                        "current_agent": agent,
                        "progress": (i + 1) / len(agents) * 100
                    },
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Small delay to simulate processing
                await asyncio.sleep(0.5)
            
            # Process the request
            complex_request = {
                "type": request_type,
                "content": content,
                "requirements": params.get("requirements", {})
            }
            
            result = await self.orchestrator.coordinate_agents(
                complex_request,
                f"stream_session_{request_id}"
            )
            
            yield json.dumps({
                "id": request_id,
                "type": "result",
                "data": result,
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error in streaming multi-agent analysis: {e}")
            yield json.dumps({
                "id": request_id,
                "type": "error",
                "error": {"code": -32603, "message": str(e)},
                "timestamp": datetime.utcnow().isoformat()
            })


# FastAPI app for HTTP and WebSocket support
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager"""
    print("🚀 Starting MCP server initialization...")
    
    # Initialize the streaming MCP server
    server = StreamingMCPServer()
    
    try:
        await server.initialize()
        app.state.mcp_server = server
        print("✅ MCP server initialized successfully")
        print(f"🔗 Azure services initialized: {server.azure_manager is not None}")
        print(f"📚 Knowledge base initialized: {server.kb_manager is not None}")
        print(f"🤝 Multi-agent orchestrator: {server.orchestrator is not None}")
        print(f"🔍 RAG pipeline: {server.rag_pipeline is not None}")
        
    except Exception as e:
        print(f"❌ Failed to initialize MCP server: {e}")
        print("💡 Check your Azure credentials and network connectivity")
        print("💡 Ensure all required services are running")
        raise e
    
    yield
    
    # Cleanup
    print("🧹 Cleaning up MCP server...")
    try:
        # Close any active connections
        if hasattr(app.state.mcp_server, 'websocket_connections'):
            for conn in app.state.mcp_server.websocket_connections.values():
                await conn.close()
        print("✅ Cleanup completed")
    except Exception as e:
        print(f"⚠️ Error during cleanup: {e}")

app = FastAPI(
    title="Financial RAG MCP Server",
    description="Enhanced MCP Server with Streaming Support",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/mcp/rpc")
async def handle_mcp_request(request: Request):
    """Handle MCP JSON-RPC requests over HTTP"""
    start_time = datetime.utcnow()
    request_id = None
    
    try:
        logger.info(f"📨 Starting MCP RPC request handling at {start_time.isoformat()}")
        
        # Set a timeout for reading the request data
        data = await asyncio.wait_for(request.json(), timeout=5.0)
        request_id = data.get("id", "unknown")
        
        logger.info(f"🔍 Received MCP RPC request {request_id}: {json.dumps(data, indent=2)}")
        
        mcp_server = request.app.state.mcp_server
        logger.info(f"🏢 MCP Server status - Azure: {mcp_server.azure_manager is not None}, KB: {mcp_server.kb_manager is not None}, Orchestrator: {mcp_server.orchestrator is not None}")
        
        # Check if this is a direct tool call (not standard MCP protocol)
        method = data.get("method", "")
        available_tools = [tool["name"] for tool in mcp_server.get_available_tools()]
        
        logger.info(f"⚙️ Processing method '{method}' (available tools: {len(available_tools)})")
        
        if method in available_tools:
            # Convert direct tool call to MCP tools/call format
            logger.info(f"🔧 Converting direct tool call '{method}' to MCP format")
            mcp_data = {
                "jsonrpc": "2.0",
                "id": data.get("id"),
                "method": "tools/call",
                "params": {
                    "name": method,
                    "arguments": data.get("params", {})
                }
            }
            logger.info(f"🔄 About to call mcp_server.process_request for {request_id}")
            # Add timeout to prevent hanging - increased to 120 seconds for financial processing
            response = await asyncio.wait_for(mcp_server.process_request(mcp_data), timeout=120.0)
        else:
            # Standard MCP protocol call
            logger.info(f"🔄 About to call mcp_server.process_request (standard) for {request_id}")
            # Add timeout to prevent hanging - increased to 120 seconds for financial processing
            response = await asyncio.wait_for(mcp_server.process_request(data), timeout=120.0)
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.info(f"✅ MCP RPC response for {request_id} (took {processing_time:.2f}s): {json.dumps(response, indent=2) if isinstance(response, dict) else str(response)}")
        return response
        
    except asyncio.TimeoutError:
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.error(f"⏱️ MCP RPC request {request_id} timed out after {processing_time:.2f}s")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32603, "message": f"Request timed out after {processing_time:.2f}s"}
        }
    except Exception as e:
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        logger.error(f"❌ Error in MCP RPC request {request_id} after {processing_time:.2f}s: {e}", exc_info=True)
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32603, "message": str(e)}
        }

@app.post("/mcp/stream")
async def handle_streaming_request(request: Request):
    """Handle streaming MCP requests with SSE"""
    try:
        data = await request.json()
        mcp_server = request.app.state.mcp_server
        
        request_id = data.get("id", "")
        method = data.get("method", "")
        params = data.get("params", {})
        
        return EventSourceResponse(
            mcp_server.stream_response(request_id, method, params),
            media_type="text/plain"
        )
        
    except Exception as e:
        return {"error": str(e)}

@app.websocket("/mcp/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for bidirectional MCP communication"""
    await websocket.accept()
    connection_id = f"ws_{datetime.utcnow().timestamp()}"
    
    mcp_server = websocket.app.state.mcp_server
    mcp_server.websocket_connections[connection_id] = websocket
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Check if this is a streaming request
            if data.get("stream", False):
                # Handle streaming via WebSocket
                request_id = data.get("id", "")
                method = data.get("method", "")
                params = data.get("params", {})
                
                async for chunk in mcp_server.stream_response(request_id, method, params):
                    await websocket.send_text(chunk)
            else:
                # Handle regular request
                response = await mcp_server.process_request(data)
                await websocket.send_json(response)
                
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        # Clean up connection
        if connection_id in mcp_server.websocket_connections:
            del mcp_server.websocket_connections[connection_id]

@app.get("/mcp/info")
async def get_server_info():
    """Get MCP server information"""
    mcp_server = app.state.mcp_server
    return {
        "server_info": mcp_server.server_info,
        "tools": mcp_server.get_available_tools(),
        "resources": mcp_server.get_available_resources(),
        "prompts": mcp_server.get_available_prompts(),
        "protocols": ["stdio", "http", "websocket", "sse"],
        "streaming_support": True
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.post("/mcp/tools/call")
async def call_tool(request: dict):
    """Call a specific tool directly (REST endpoint)"""
    try:
        tool_name = request.get("name")
        arguments = request.get("arguments", {})
        
        if not tool_name:
            raise HTTPException(status_code=400, detail="Tool name is required")
        
        logger.info(f"🔧 Direct tool call - Tool: {tool_name}, Args: {json.dumps(arguments, indent=2)}")
        mcp_server = app.state.mcp_server
        
        logger.info(f"🏢 MCP Server status before tool call - Azure: {mcp_server.azure_manager is not None}, KB: {mcp_server.kb_manager is not None}, Orchestrator: {mcp_server.orchestrator is not None}")
        
        result = await mcp_server.handle_tool_call(tool_name, arguments)
        
        logger.info(f"✅ Tool call result: {json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)}")
        
        return {
            "success": True,
            "result": result,
            "tool": tool_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error calling tool {tool_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# CLI mode support (original stdin/stdout)
async def run_stdio_mode():
    """Run in stdio mode for MCP client compatibility"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Starting MCP server in stdio mode...", file=sys.stderr)
    
    server = StreamingMCPServer()
    
    try:
        await server.initialize()
        print("✅ MCP server initialized successfully", file=sys.stderr)
        print(f"🔗 Azure services: {'✅' if server.azure_manager else '❌'}", file=sys.stderr)
        print(f"📚 Knowledge base: {'✅' if server.kb_manager else '❌'}", file=sys.stderr)
        print(f"🤝 Multi-agent orchestrator: {'✅' if server.orchestrator else '❌'}", file=sys.stderr)
        print(f"🔍 RAG pipeline: {'✅' if server.rag_pipeline else '❌'}", file=sys.stderr)
        print("📡 Listening for MCP requests on stdin...", file=sys.stderr)
        
        # Handle stdin/stdout communication (MCP standard)
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                    
                request_data = json.loads(line.strip())
                response = await server.process_request(request_data)
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"}
                }
                print(json.dumps(error_response), flush=True)
                
            except KeyboardInterrupt:
                print("🛑 Received interrupt signal, shutting down...", file=sys.stderr)
                break
                
    except Exception as e:
        print(f"❌ Failed to start MCP server: {e}", file=sys.stderr)
        print("💡 Check your Azure credentials and configuration", file=sys.stderr)
        sys.exit(1)

def main():
    """Main entry point - support both HTTP server and stdio modes"""
    import argparse
    
    # Set Windows event loop policy if needed - aiodns requires SelectorEventLoop
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    parser = argparse.ArgumentParser(description="Financial RAG MCP Server")
    parser.add_argument("--mode", choices=["stdio", "http"], default="stdio",
                       help="Server mode: stdio for MCP clients, http for web")
    parser.add_argument("--host", default="127.0.0.1", help="Host for HTTP mode")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    
    print(f"🚀 Starting Financial RAG MCP Server in {args.mode} mode...")
    print(f"🖥️ Platform: {platform.system()}")
    print(f"🐍 Python: {platform.python_version()}")
    
    try:
        if args.mode == "stdio":
            asyncio.run(run_stdio_mode())
        else:
            # Run FastAPI server with streaming support
            print(f"🌐 Starting HTTP server on {args.host}:{args.port}")
            uvicorn.run(
                "streaming_mcp_server:app",
                host=args.host,
                port=args.port,
                log_level="info" if not args.debug else "debug",
                reload=False
            )
    except KeyboardInterrupt:
        print("🛑 Server shutdown by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

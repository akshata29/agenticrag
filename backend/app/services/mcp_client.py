"""
MCP Client for Financial RAG System

This client communicates with the Financial RAG MCP Server to handle
questions using the Model Context Protocol.
"""

import asyncio
import json
import logging
import subprocess
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime
import aiohttp
import platform
import time

# Fix Windows asyncio issues for HTTP client
if platform.system() == "Windows":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

logger = logging.getLogger(__name__)

class MCPClient:
    """Client for communicating with MCP servers"""
    
    def __init__(self, server_command: List[str], server_env: Optional[Dict[str, str]] = None):
        self.server_command = server_command
        self.server_env = server_env or {}
        self.process: Optional[subprocess.Popen] = None
        self.initialized = False
        self.server_capabilities = {}
        self.server_info = {}
        
    async def initialize(self):
        """Initialize connection to MCP server"""
        try:
            logger.info(f"Starting MCP server: {' '.join(self.server_command)}")
            
            # Start the MCP server process
            self.process = subprocess.Popen(
                self.server_command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**self.server_env}
            )
            
            # Send initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {
                            "listChanged": True
                        },
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "financial-rag-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            response = await self._send_request(init_request)
            
            if "error" in response:
                raise Exception(f"MCP server initialization failed: {response['error']}")
            
            result = response.get("result", {})
            self.server_capabilities = result.get("capabilities", {})
            self.server_info = result.get("serverInfo", {})
            self.initialized = True
            
            logger.info(f"MCP server initialized: {self.server_info.get('name', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            await self.close()
            raise
    
    async def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to MCP server and get response"""
        if not self.process:
            raise Exception("MCP server process not started")
        
        try:
            # Send request
            request_json = json.dumps(request) + "\n"
            self.process.stdin.write(request_json)
            self.process.stdin.flush()
            
            # Read response
            response_line = self.process.stdout.readline()
            if not response_line:
                raise Exception("No response from MCP server")
            
            response = json.loads(response_line.strip())
            return response
            
        except Exception as e:
            logger.error(f"Error communicating with MCP server: {e}")
            raise
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from MCP server"""
        request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/list"
        }
        
        response = await self._send_request(request)
        
        if "error" in response:
            raise Exception(f"Failed to list tools: {response['error']}")
        
        return response.get("result", {}).get("tools", [])
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server"""
        request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }
        
        response = await self._send_request(request)
        
        if "error" in response:
            raise Exception(f"Tool call failed: {response['error']}")
        
        result = response.get("result", {})
        content = result.get("content", [])
        
        # Extract text content from MCP response
        if content and isinstance(content, list) and len(content) > 0:
            text_content = content[0].get("text", "{}")
            try:
                return json.loads(text_content)
            except json.JSONDecodeError:
                return {"content": text_content}
        
        return result
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources from MCP server"""
        request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "resources/list"
        }
        
        response = await self._send_request(request)
        
        if "error" in response:
            raise Exception(f"Failed to list resources: {response['error']}")
        
        return response.get("result", {}).get("resources", [])
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource from the MCP server"""
        request = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "resources/read",
            "params": {
                "uri": uri
            }
        }
        
        response = await self._send_request(request)
        
        if "error" in response:
            raise Exception(f"Failed to read resource: {response['error']}")
        
        return response.get("result", {})
    
    async def answer_financial_question(
        self,
        question: str,
        context: str = "",
        verification_level: str = "thorough",
        use_multi_agent: bool = True
    ) -> Dict[str, Any]:
        """Answer a financial question using the MCP server"""
        return await self.call_tool("answer_financial_question", {
            "question": question,
            "context": context,
            "verification_level": verification_level,
            "use_multi_agent": use_multi_agent
        })
    
    async def search_documents(
        self,
        query: str,
        document_types: Optional[List[str]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Search financial documents using the MCP server"""
        return await self.call_tool("search_financial_documents", {
            "query": query,
            "document_types": document_types or [],
            "top_k": top_k
        })
    
    async def verify_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify source credibility using the MCP server"""
        return await self.call_tool("verify_source_credibility", {
            "sources": sources
        })
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics using the MCP server"""
        return await self.call_tool("get_knowledge_base_stats", {})
    
    async def close(self):
        """Close the MCP client and terminate server process"""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.sleep(1)
                if self.process.poll() is None:
                    self.process.kill()
                self.process = None
            except Exception as e:
                logger.error(f"Error closing MCP server: {e}")
        
        self.initialized = False


class HTTPMCPClient:
    """HTTP-based MCP Client for communicating with MCP servers running in HTTP mode"""
    
    def __init__(self, server_url: str, timeout: int = 120):  # Increased to 120 seconds for financial processing
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.initialized = False
        self.server_capabilities = {}
        self.server_info = {}
        
    async def initialize(self):
        """Initialize HTTP connection to MCP server"""
        try:
            if not self.session:
                # Use ThreadedResolver to avoid aiodns issues on Windows
                if platform.system() == "Windows":
                    resolver = aiohttp.ThreadedResolver()
                    connector = aiohttp.TCPConnector(resolver=resolver)
                else:
                    connector = aiohttp.TCPConnector()
                
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    connector=connector
                )
            
            # Test connection by getting server info
            async with self.session.get(f"{self.server_url}/mcp/info") as response:
                if response.status == 200:
                    server_info = await response.json()
                    self.server_info = server_info.get("server_info", {})
                    self.server_capabilities = {"tools": True, "resources": True}
                    self.initialized = True
                    logger.info(f"HTTP MCP server connected: {self.server_info.get('name', 'Unknown')}")
                else:
                    raise Exception(f"Failed to connect to MCP server: HTTP {response.status}")
                    
        except Exception as e:
            logger.error(f"Failed to initialize HTTP MCP client: {e}")
            await self.close()
            raise
    
    async def _send_http_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send HTTP request to MCP server"""
        if not self.session:
            raise Exception("HTTP session not initialized")
        
        request_start = time.time()
        try:
            request_data = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": method,
                "params": params or {}
            }
            
            logger.info(f"📤 Sending HTTP request to {self.server_url}/mcp/rpc: {method} (timeout: {self.timeout}s)")
            logger.debug(f"Request data: {request_data}")
            
            async with self.session.post(f"{self.server_url}/mcp/rpc", json=request_data) as response:
                response_received_time = time.time()
                logger.info(f"📥 HTTP response received after {response_received_time - request_start:.2f}s, status: {response.status}")
                
                if response.status == 200:
                    response_data = await response.json()
                    total_time = time.time() - request_start
                    logger.info(f"✅ HTTP request completed successfully in {total_time:.2f}s")
                    logger.debug(f"Response data: {response_data}")
                    return response_data
                else:
                    response_text = await response.text()
                    total_time = time.time() - request_start
                    logger.error(f"❌ HTTP request failed after {total_time:.2f}s: {response.status}, response: {response_text}")
                    raise Exception(f"HTTP request failed: {response.status}, response: {response_text}")
                    
        except asyncio.TimeoutError as e:
            elapsed_time = time.time() - request_start
            logger.error(f"⏱️ HTTP request timed out after {elapsed_time:.2f}s (timeout was {self.timeout}s)")
            raise
        except aiohttp.ClientError as e:
            elapsed_time = time.time() - request_start
            logger.error(f"❌ HTTP client error after {elapsed_time:.2f}s communicating with MCP server: {type(e).__name__}: {e}")
            raise
        except json.JSONDecodeError as e:
            elapsed_time = time.time() - request_start
            logger.error(f"❌ JSON decode error after {elapsed_time:.2f}s from MCP server response: {e}")
            raise
        except Exception as e:
            elapsed_time = time.time() - request_start
            logger.error(f"❌ Unexpected error after {elapsed_time:.2f}s communicating with HTTP MCP server: {type(e).__name__}: {e}")
            raise
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from HTTP MCP server"""
        response = await self._send_http_request("tools/list")
        
        if "error" in response:
            raise Exception(f"Failed to list tools: {response['error']}")
        
        return response.get("result", {}).get("tools", [])
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the HTTP MCP server"""
        response = await self._send_http_request("tools/call", {
            "name": name,
            "arguments": arguments
        })
        
        if "error" in response:
            raise Exception(f"Tool call failed: {response['error']}")
        
        result = response.get("result", {})
        content = result.get("content", [])
        
        # Extract text content from MCP response
        if content and isinstance(content, list) and len(content) > 0:
            text_content = content[0].get("text", "{}")
            try:
                return json.loads(text_content)
            except json.JSONDecodeError:
                return {"content": text_content}
        
        return result
    
    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources from HTTP MCP server"""
        response = await self._send_http_request("resources/list")
        
        if "error" in response:
            raise Exception(f"Failed to list resources: {response['error']}")
        
        return response.get("result", {}).get("resources", [])
    
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource from the HTTP MCP server"""
        response = await self._send_http_request("resources/read", {"uri": uri})
        
        if "error" in response:
            raise Exception(f"Failed to read resource: {response['error']}")
        
        return response.get("result", {})
    
    async def answer_financial_question(
        self,
        question: str,
        context: str = "",
        verification_level: str = "thorough",
        use_multi_agent: bool = True
    ) -> Dict[str, Any]:
        """Answer a financial question using the HTTP MCP server"""
        logger.info(f"HTTPMCPClient: Answering question: {question[:100]}...")
        try:
            result = await self.call_tool("answer_financial_question", {
                "question": question,
                "context": context,
                "verification_level": verification_level,
                "use_multi_agent": use_multi_agent
            })
            logger.info(f"HTTPMCPClient: Got result with keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            return result
        except Exception as e:
            logger.error(f"HTTPMCPClient: Error in answer_financial_question: {type(e).__name__}: {e}")
            raise
    
    async def search_documents(
        self,
        query: str,
        document_types: Optional[List[str]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Search financial documents using the HTTP MCP server"""
        return await self.call_tool("search_financial_documents", {
            "query": query,
            "document_types": document_types or [],
            "top_k": top_k
        })
    
    async def verify_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify source credibility using the HTTP MCP server"""
        return await self.call_tool("verify_source_credibility", {
            "sources": sources
        })
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics using the HTTP MCP server"""
        return await self.call_tool("get_knowledge_base_stats", {})
    
    async def close(self):
        """Close the HTTP MCP client session"""
        if self.session:
            await self.session.close()
            self.session = None
        self.initialized = False


class MCPService:
    """Service for managing MCP clients and routing requests"""
    
    def __init__(self):
        self.clients: Dict[str, Any] = {}  # Can hold both MCPClient and HTTPMCPClient
        
    async def initialize_client(self, name: str, command: List[str], env: Optional[Dict[str, str]] = None):
        """Initialize a stdio MCP client"""
        client = MCPClient(command, env)
        await client.initialize()
        self.clients[name] = client
        logger.info(f"MCP client '{name}' initialized successfully")
    
    async def initialize_http_client(self, name: str, server_url: str, timeout: int = 120):
        """Initialize an HTTP MCP client"""
        client = HTTPMCPClient(server_url, timeout)
        await client.initialize()
        self.clients[name] = client
        logger.info(f"HTTP MCP client '{name}' initialized successfully")
    
    def get_client(self, name: str) -> Optional[Any]:
        """Get an MCP client by name"""
        return self.clients.get(name)
    
    async def process_question_with_mcp(
        self,
        question: str,
        session_id: str,
        verification_level: str = "thorough",
        context: str = ""
    ) -> Dict[str, Any]:
        """Process a question using MCP"""
        try:
            logger.info(f"MCPService: Processing question with MCP: {question[:100]}...")
            
            # Try to get the financial RAG client
            client = self.get_client("financial-rag")
            if not client or not client.initialized:
                logger.info("MCPService: Initializing financial RAG client...")
                # Initialize client if not available
                await self.initialize_financial_rag_client()
                client = self.get_client("financial-rag")
            
            if not client:
                raise Exception("MCP client not available after initialization")
            
            logger.info(f"MCPService: Using client type: {type(client).__name__}")
            
            # Call the MCP server to answer the question
            result = await client.answer_financial_question(
                question=question,
                context=context,
                verification_level=verification_level,
                use_multi_agent=True
            )
            
            logger.info(f"MCPService: Got MCP result with keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            
            # Format response in the same structure as other RAG methods
            return {
                "answer": result.get("answer", "No answer provided"),
                "confidence": result.get("confidence", 0.0),
                "sources": result.get("sources", []),
                "citations": self._format_citations(result.get("sources", [])),
                "sub_questions": [],
                "verification_details": {
                    "overall_credibility_score": result.get("confidence", 0.0),
                    "verified_sources_count": len(result.get("sources", [])),
                    "total_sources_count": len(result.get("sources", [])),
                    "verification_summary": f"Verified through MCP with {verification_level} verification"
                },
                "metadata": {
                    "method": "MCP",
                    "session_id": session_id,
                    "verification_level": verification_level,
                    "mcp_method": result.get("method", "unknown"),
                    "success": result.get("success", False)
                }
            }
            
        except Exception as e:
            logger.error(f"MCPService: Error processing question with MCP: {type(e).__name__}: {e}")
            logger.exception("MCPService: Full exception details:")
            return {
                "answer": f"Error processing question with MCP: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "citations": [],
                "sub_questions": [],
                "verification_details": {
                    "overall_credibility_score": 0.0,
                    "verified_sources_count": 0,
                    "total_sources_count": 0,
                    "verification_summary": f"MCP processing failed: {str(e)}"
                },
                "metadata": {
                    "method": "MCP",
                    "session_id": session_id,
                    "error": str(e),
                    "success": False
                }
            }
    
    def _format_citations(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format MCP sources as citations"""
        citations = []
        for i, source in enumerate(sources):
            # Handle both numeric and string confidence values
            confidence_value = source.get("confidence", source.get("credibility_score", 0.0))
            if isinstance(confidence_value, str):
                # Already a string confidence level
                confidence_str = confidence_value
            else:
                # Convert numeric confidence to string
                confidence_str = "high" if confidence_value > 0.8 else "medium" if confidence_value > 0.5 else "low"
            
            citations.append({
                "id": f"mcp_citation_{i}",
                "content": source.get("content", ""),
                "source": source.get("source", "Unknown"),
                "documentId": source.get("document_id", f"mcp_doc_{i}"),
                "documentTitle": source.get("document_title", source.get("source", "MCP Document")),
                "pageNumber": source.get("page_number"),
                "sectionTitle": source.get("section_title"),
                "confidence": confidence_str,
                "url": source.get("url"),
                "credibilityScore": source.get("credibility_score", confidence_value if isinstance(confidence_value, (int, float)) else 0.0)
            })
        return citations
    
    async def initialize_financial_rag_client(self):
        """Initialize the financial RAG MCP client (HTTP or stdio mode)"""
        import os
        from app.core.config import settings
        
        # Check if MCP server is running in HTTP mode by checking environment variable or settings
        mcp_server_url = getattr(settings, 'MCP_SERVER_URL', None) or os.getenv('MCP_SERVER_URL')
        
        if mcp_server_url:
            # Use HTTP mode
            logger.info(f"Initializing MCP client in HTTP mode: {mcp_server_url}")
            await self.initialize_http_client("financial-rag", mcp_server_url)
        else:
            # Use stdio mode (legacy)
            logger.info("Initializing MCP client in stdio mode")
            
            # Path to the MCP server
            server_path = os.path.join(os.path.dirname(__file__), "..", "..", "mcp_server", "main.py")
            workspace_path = os.path.join(os.path.dirname(__file__), "..", "..")
            
            command = ["python", server_path]
            env = {
                "PYTHONPATH": workspace_path,
                **os.environ
            }
            
            await self.initialize_client("financial-rag", command, env)
    
    async def close_all_clients(self):
        """Close all MCP clients"""
        for name, client in self.clients.items():
            try:
                await client.close()
                logger.info(f"MCP client '{name}' closed")
            except Exception as e:
                logger.error(f"Error closing MCP client '{name}': {e}")
        
        self.clients.clear()


# Global MCP service instance
mcp_service = MCPService()

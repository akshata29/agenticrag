# Enhanced Financial RAG MCP Server

This directory contains an enhanced Model Context Protocol (MCP) server implementation that extends the AgenticRAG system with multiple communication protocols and streaming capabilities.

## 🚀 Features

### Multiple Protocol Support
- **Stdin/Stdout**: Original MCP protocol for Claude Desktop and similar clients
- **HTTP JSON-RPC**: REST API for web applications and services
- **Server-Sent Events (SSE)**: Real-time streaming for long-running operations
- **WebSocket**: Bidirectional communication for interactive applications

### Streaming Capabilities
- Real-time progress updates for financial analysis
- Incremental results for document searches
- Multi-agent coordination with step-by-step progress
- Partial result delivery for improved user experience

### Claude-Compatible Interface
- Tool discovery and usage patterns compatible with Claude
- Standardized response formats
- Error handling and retry logic
- Streaming support for external AI assistants

## 📁 Files

| File | Description |
|------|-------------|
| `main.py` | Original MCP server (stdin/stdout) |
| `streaming_mcp_server.py` | Enhanced server with HTTP/WebSocket/SSE support |
| `http_client.py` | HTTP client and Claude-compatible interface |
| `test_enhanced_mcp.py` | Comprehensive test suite for all protocols |
| `mcp_discovery_config.json` | Configuration for client discovery |
| `requirements.txt` | Dependencies for streaming server |

## 🛠 Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the backend system is configured and running.

## 🎯 Usage

### 1. Stdin/Stdout Mode (Original MCP)

For Claude Desktop or other MCP clients:

```bash
python main.py
```

**Claude Desktop Configuration:**
```json
{
  "mcpServers": {
    "financial-rag": {
      "command": "python",
      "args": ["path/to/agenticrag/mcp_server/main.py"],
      "env": {
        "PYTHONPATH": "path/to/agenticrag"
      }
    }
  }
}
```

### 2. HTTP Server Mode

For web applications and REST API access:

```bash
python streaming_mcp_server.py --mode http --port 8000
```

**API Endpoints:**
- `POST /mcp/rpc` - JSON-RPC requests
- `POST /mcp/stream` - Server-Sent Events streaming
- `WS /mcp/ws` - WebSocket connection
- `GET /mcp/info` - Server information and capabilities
- `GET /health` - Health check

### 3. Client Usage Examples

#### HTTP Client
```python
from http_client import MCPHTTPClient

async with MCPHTTPClient("http://localhost:8000") as client:
    # Basic question answering
    result = await client.answer_financial_question(
        question="What are Apple's main revenue streams?",
        verification_level="thorough"
    )
    
    # Streaming analysis
    async for update in client.stream_request("answer_financial_question", {
        "question": "Analyze Microsoft's financial performance",
        "use_multi_agent": True
    }):
        print(f"Progress: {update}")
```

#### Claude-Compatible Interface
```python
from http_client import ClaudeCompatibleMCPClient

claude_client = ClaudeCompatibleMCPClient()
await claude_client.initialize()

# Use tools like Claude would
result = await claude_client.use_tool(
    "answer_financial_question",
    {"question": "What is Amazon's business model?"}
)
```

## 🔧 Available Tools

### Core Financial Tools

1. **answer_financial_question**
   - Comprehensive financial Q&A with RAG
   - Multi-agent coordination
   - Source verification and credibility assessment
   - **Streaming Support**: ✅

2. **search_financial_documents**
   - Knowledge base document search
   - Document type filtering
   - Relevance ranking
   - **Streaming Support**: ✅

3. **verify_source_credibility**
   - Source reliability assessment
   - Cross-reference validation
   - Confidence scoring
   - **Streaming Support**: ❌

4. **coordinate_multi_agent_analysis**
   - Complex financial analysis coordination
   - Agent workflow orchestration
   - Distributed processing
   - **Streaming Support**: ✅

5. **get_knowledge_base_stats**
   - Knowledge base health metrics
   - Document statistics
   - System status
   - **Streaming Support**: ❌

## 📊 Resources

- `financial://knowledge-base/statistics` - KB statistics
- `financial://agents/capabilities` - Agent capabilities
- `financial://documents/types` - Available document types
- `financial://system/status` - System status

## 🎭 Prompts

- `financial_analysis` - Comprehensive financial analysis template
- `risk_assessment` - Financial risk assessment template
- `market_comparison` - Market comparison analysis template

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_enhanced_mcp.py
```

This tests:
- ✅ Stdin/stdout protocol
- ✅ HTTP JSON-RPC protocol
- ✅ SSE streaming protocol
- ✅ WebSocket protocol
- ✅ Claude-compatible interface

## 🌐 Integration Examples

### VS Code Extension

```typescript
// Connect to MCP server via HTTP
const response = await fetch('http://localhost:8000/mcp/rpc', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    jsonrpc: '2.0',
    id: 'vscode-req-1',
    method: 'answer_financial_question',
    params: {
      question: 'What are Tesla\'s competitive advantages?',
      verification_level: 'thorough'
    }
  })
});
```

### Web Application with Streaming

```javascript
// Server-Sent Events for real-time updates
const eventSource = new EventSource('/mcp/stream', {
  method: 'POST',
  body: JSON.stringify({
    method: 'answer_financial_question',
    params: { question: 'Analyze Apple vs Microsoft' }
  })
});

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'progress') {
    updateProgress(data.message);
  } else if (data.type === 'result') {
    displayResult(data.data);
  }
};
```

### Claude Integration

Claude can discover and use the MCP server automatically when configured in the desktop app. The server exposes financial analysis capabilities as tools that Claude can use to answer user questions about finance and markets.

## 🔒 Security Considerations

- The HTTP server binds to localhost by default
- No authentication is implemented (add as needed)
- CORS is enabled for development (configure for production)
- Consider rate limiting for production deployments

## 📈 Performance

### Streaming Benefits
- Reduced perceived latency for long operations
- Real-time feedback on analysis progress
- Ability to cancel long-running operations
- Better user experience for complex queries

### Scalability
- Multiple protocol support enables different client types
- WebSocket connections for high-frequency interactions
- HTTP for stateless requests
- Stdin/stdout for traditional MCP clients

## 🚀 Future Enhancements

1. **Authentication & Authorization**
   - API key authentication
   - Role-based access control
   - Rate limiting

2. **Advanced Streaming**
   - Cancellable operations
   - Resumable requests
   - Batch processing

3. **Monitoring & Observability**
   - Metrics collection
   - Distributed tracing
   - Performance monitoring

4. **Multi-Instance Support**
   - Load balancing
   - Service discovery
   - High availability

## 📚 Protocol Specifications

- [Model Context Protocol (MCP)](https://spec.modelcontextprotocol.io/)
- [JSON-RPC 2.0](https://www.jsonrpc.org/specification)
- [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [WebSocket Protocol](https://tools.ietf.org/html/rfc6455)

---

**Ready for production use with Claude, VS Code extensions, web applications, and any MCP-compatible client!**

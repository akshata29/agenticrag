# RAG Financial POC - Adaptive Knowledge Base Management

A comprehensive Retrieval Augmented Generation (RAG) system designed for financial document analysis, featuring adaptive knowledge base management, multi-agent orchestration, and advanced observability.

## Overview

This repo implements a complete RAG Financial POC with three interconnected use cases:

1. **Context-Aware Content Generation**: RAG-based content generation with source citation
2. **Agentic Question Answering**: Multi-source information retrieval with credibility verification
3. **Adaptive Knowledge Base Management**: Automated knowledge base updates and curation

## Key Features

### 🧠 Adaptive Knowledge Base Management
- **Information Acquisition**: Automated ingestion from multiple financial document sources
- **Relevance Assessment**: AI-powered content evaluation and credibility scoring
- **Knowledge Organization**: Hierarchical structuring with semantic relationships
- **Conflict Resolution**: Intelligent handling of contradictory financial information
- **Response Adaptation**: Dynamic updates based on new market data and reports

### 📊 Advanced Observability & Evaluation
- **Token Usage Tracking**: Real-time monitoring across all Azure OpenAI models
- **Evaluation Framework**: Custom metrics for relevance, groundedness, coherence, and fluency
- **Distributed Tracing**: OpenTelemetry integration for complete request tracking
- **Admin Dashboard**: Comprehensive metrics visualization with real-time updates
- **Cost Monitoring**: Detailed cost tracking and budget alerts

### 🤖 Multi-Agent Orchestration
- **Semantic Kernel Integration**: Coordinated agent workflows for document processing
- **MCP (Model Context Protocol)**: Standardized agent communication patterns
- **A2A (Agent-to-Agent)**: Inter-agent collaboration for complex financial analysis
- **Financial Document Specialists**: Dedicated agents for 10-K/10-Q report analysis

### 🌐 Model Context Protocol (MCP) Integration
- **Multiple Protocols**: Support for stdin/stdout, HTTP JSON-RPC, WebSocket, and Server-Sent Events
- **Claude Compatibility**: Native integration with Claude Desktop and other MCP clients
- **Streaming Support**: Real-time progress updates for long-running financial analysis
- **Tool Discovery**: Automatic tool and resource discovery for external AI assistants
- **External Client Support**: Easy integration with VS Code extensions, web apps, and custom clients

### 🔍 Hybrid Search & RAG Pipeline
- **Vector Search**: Semantic similarity using Azure AI Search
- **Keyword Search**: Traditional text matching for precise queries
- **Hybrid Search**: Combined vector and keyword search with semantic ranking
- **Citation Management**: Comprehensive source tracking and inline citations
- **Financial Context**: Industry-specific prompt engineering and chunking strategies

## Architecture

### Backend Services (Python 3.11)
- **FastAPI**: RESTful API with automatic OpenAPI documentation
- **Azure AI Search**: Vector store with hybrid search and semantic ranking
- **Azure OpenAI**: GPT-4, GPT-4-Turbo, and embedding models
- **Azure Cosmos DB**: Session history and evaluation results storage
- **Azure Document Intelligence**: Advanced document processing and extraction
- **Azure AI Foundry**: Project-based AI model deployment and management
- **Semantic Kernel**: Multi-agent orchestration framework

### Frontend Application
- **React + TypeScript**: Modern, responsive web interface
- **Vite**: Fast development server and optimized builds
- **ChatGPT-like UI**: Intuitive chat interface inspired by open-webui
- **Admin Dashboard**: Real-time observability metrics and system monitoring
- **Citation Preview**: Interactive document source navigation

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+ with npm/yarn/pnpm
- Azure subscription with AI services
- Docker (optional, for containerized deployment)

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Configure Azure services in .env file
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup
```bash
cd frontend/rag-financial-frontend
npm install
cp .env.example .env
# Configure API endpoints in .env file
npm run dev
```

### 3. Access the Application
- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Admin Dashboard**: http://localhost:5173 (Admin tab)

## Configuration

### Verification Level Features

The QA system supports three distinct verification levels with different performance and thoroughness characteristics:

| Feature | Basic | Thorough | Comprehensive |
|---------|-------|-----------|---------------|
| **Source Documents** | 5 | 10 | 15 |
| **Content Length** | 800 chars | 1200 chars | 1600 chars |
| **Response Time** | Fastest | Standard | Thorough |
| **Question Decomposition** | ❌ | ❌ | ✅ |
| **Source Verification** | Basic | ✅ | ✅ |
| **Cross-referencing** | ❌ | ✅ | ✅ |
| **Conflict Analysis** | ❌ | ✅ | ✅ |
| **Limitation Analysis** | ❌ | ❌ | ✅ |
| **Sub-question Analysis** | ❌ | ❌ | ✅ |
| **Use Case** | Quick answers | Standard analysis | Complex investigation |

**Basic Verification**: Speed-optimized for simple questions requiring quick answers with essential information and citations.

**Thorough Verification**: Balanced approach with comprehensive analysis, source verification, and conflict identification for standard financial research.

**Comprehensive Verification**: Exhaustive deep analysis with question decomposition, multi-angle investigation, and detailed limitation analysis for complex financial questions.

### Azure Services Setup
1. **Azure AI Foundry**: Create project with Foundry Project approach
2. **Azure OpenAI**: Deploy GPT-4, GPT-4-Turbo, and embedding models
3. **Azure AI Search**: Configure with vector search and semantic ranking
4. **Azure Cosmos DB**: Set up containers for sessions and evaluations
5. **Azure Document Intelligence**: Enable for financial document processing
6. **Azure Application Insights**: Configure for observability and tracing

### Environment Configuration
See detailed configuration in:
- `backend/.env.example` - Backend Azure services and API settings
- `frontend/.env.example` - Frontend configuration and feature flags
- `azure-deployment-config.json` - Complete Azure resource definitions

## API Documentation

The FastAPI backend provides comprehensive API documentation:
- **Interactive Swagger UI**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Financial Document Processing

### Supported Document Types
- **10-K Annual Reports**: Complete financial statements and analysis
- **10-Q Quarterly Reports**: Quarterly financial updates
- **8-K Current Reports**: Material events and corporate changes
- **Proxy Statements**: Governance and executive compensation
- **Earnings Transcripts**: Quarterly earnings call transcripts

### Chunking Strategy
- **Hierarchical Chunking**: Section-aware document segmentation
- **Financial Context**: Industry-specific chunk boundaries
- **Overlap Management**: Intelligent chunk overlap for context preservation
- **Metadata Enrichment**: Document type, section, and page number tracking

## Observability & Monitoring

### Metrics Tracked
- **Token Usage**: By model, user, and session
- **Response Times**: API endpoint performance
- **Evaluation Scores**: Relevance, groundedness, coherence
- **System Resources**: CPU, memory, and storage utilization
- **Error Rates**: Failed requests and error categorization

### Distributed Tracing
- **OpenTelemetry**: Complete request tracing across services
- **Azure Application Insights**: Centralized logging and monitoring
- **Custom Spans**: Financial document processing workflows
- **Performance Profiling**: Bottleneck identification and optimization

## Model Context Protocol (MCP) Server

The system includes a comprehensive MCP server that can be used by Claude, VS Code extensions, and other MCP-compatible clients.

### 🚀 MCP Server Features

- **Multiple Protocols**: Support for stdin/stdout, HTTP, WebSocket, and SSE
- **Streaming Responses**: Real-time progress updates for long-running operations
- **Tool Discovery**: Automatic discovery of available financial analysis tools
- **Claude Desktop Integration**: Native support for Claude Desktop app
- **External Client Support**: REST API for web applications and services

### Quick MCP Setup

#### For Claude Desktop
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

#### For HTTP Clients (Web Apps, VS Code)
```bash
cd mcp_server
python streaming_mcp_server.py --mode http --port 8000
```

### Available MCP Tools

1. **answer_financial_question** - Comprehensive Q&A with RAG (streaming supported)
2. **search_financial_documents** - Knowledge base search (streaming supported)
3. **verify_source_credibility** - Source reliability assessment
4. **coordinate_multi_agent_analysis** - Multi-agent coordination (streaming supported)
5. **get_knowledge_base_stats** - System health and statistics

See [mcp_server/README.md](./mcp_server/README.md) for complete documentation.

## Support & Documentation

- **Azure AI Foundry**: https://learn.microsoft.com/en-us/azure/ai-foundry/
- **Semantic Kernel**: https://learn.microsoft.com/en-us/semantic-kernel/
- **Azure OpenAI**: https://learn.microsoft.com/en-us/azure/ai-services/openai/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **React Documentation**: https://react.dev/

## License

This project is licensed under the MIT License - see the LICENSE file for details.
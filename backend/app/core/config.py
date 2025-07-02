from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class Settings(BaseSettings):
    AZURE_TENANT_ID: str = os.getenv("AZURE_TENANT_ID", "")
    AZURE_CLIENT_ID: str = os.getenv("AZURE_CLIENT_ID", "")
    AZURE_CLIENT_SECRET: str = os.getenv("AZURE_CLIENT_SECRET", "")
    
    AZURE_SEARCH_SERVICE_NAME: str = os.getenv("AZURE_SEARCH_SERVICE_NAME", "")
    AZURE_SEARCH_INDEX_NAME: str = os.getenv("AZURE_SEARCH_INDEX_NAME", "financial-documents")
    AZURE_SEARCH_API_VERSION: str = os.getenv("AZURE_SEARCH_API_VERSION", "2023-11-01")
    AZURE_SEARCH_API_KEY: str = os.getenv("AZURE_SEARCH_API_KEY", "")
    
    # Additional Azure AI Search configurations for Agentic RAG
    AZURE_SEARCH_AGENT_NAME: str = os.getenv("AZURE_SEARCH_AGENT_NAME", "financial-qa-agent")
    AZURE_OPENAI_CHAT_MODEL_NAME: str = os.getenv("AZURE_OPENAI_CHAT_MODEL_NAME", "gpt-4o-mini")
    
    @property
    def AZURE_AI_SEARCH_ENDPOINT(self) -> str:
        """Compute Azure AI Search endpoint from service name"""
        if self.AZURE_SEARCH_SERVICE_NAME:
            return f"https://{self.AZURE_SEARCH_SERVICE_NAME}.search.windows.net"
        return ""
    
    @property 
    def AZURE_AI_SEARCH_INDEX_NAME(self) -> str:
        """Alias for backward compatibility"""
        return self.AZURE_SEARCH_INDEX_NAME
    
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "chat4omini")
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-ada-002")
    AZURE_OPENAI_DEPLOYMENT_NAME: str = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "chat4omini")
    
    AVAILABLE_EMBEDDING_MODELS: List[str] = [
        "text-embedding-ada-002",
        "text-embedding-3-small", 
        "text-embedding-3-large"
    ]
    AVAILABLE_CHAT_MODELS: List[str] = [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o",  # Supported for agentic retrieval
        "gpt-4o-mini",  # Supported for agentic retrieval
        "gpt-4.1",  # Supported for agentic retrieval
        "gpt-4.1-nano",  # Supported for agentic retrieval
        "gpt-4.1-mini",  # Supported for agentic retrieval
        "gpt-35-turbo",
        "financial-llm",  # Industry specific
        "grok-beta",
        "deepseek-chat"
    ]
    
    AZURE_COSMOS_ENDPOINT: str = os.getenv("AZURE_COSMOS_ENDPOINT", "")
    AZURE_COSMOS_DATABASE_NAME: str = os.getenv("AZURE_COSMOS_DATABASE_NAME", "rag-financial-db")
    AZURE_COSMOS_CONTAINER_NAME: str = os.getenv("AZURE_COSMOS_CONTAINER_NAME", "chat-sessions")
    AZURE_COSMOS_EVALUATION_CONTAINER_NAME: str = os.getenv("AZURE_COSMOS_EVALUATION_CONTAINER_NAME", "evaluation-results")
    AZURE_COSMOS_TOKEN_USAGE_CONTAINER_NAME: str = os.getenv("AZURE_COSMOS_TOKEN_USAGE_CONTAINER_NAME", "token-usage")
    
    AZURE_FORM_RECOGNIZER_ENDPOINT: str = os.getenv("AZURE_FORM_RECOGNIZER_ENDPOINT", "")
    
    AZURE_AI_FOUNDRY_PROJECT_NAME: str = os.getenv("AZURE_AI_FOUNDRY_PROJECT_NAME", "")
    AZURE_AI_FOUNDRY_RESOURCE_GROUP: str = os.getenv("AZURE_AI_FOUNDRY_RESOURCE_GROUP", "")
    AZURE_SUBSCRIPTION_ID: str = os.getenv("AZURE_SUBSCRIPTION_ID", "")
    AZURE_AI_FOUNDRY_WORKSPACE_NAME: str = os.getenv("AZURE_AI_FOUNDRY_WORKSPACE_NAME", "")
    AZURE_AI_PROJECT_ENDPOINT: str = os.getenv("AZURE_AI_PROJECT_ENDPOINT", "")    # Azure Monitor and Application Insights Configuration
    azure_monitor_connection_string: str = os.getenv("AZURE_MONITOR_CONNECTION_STRING", "")
    azure_key_vault_url: str = os.getenv("AZURE_KEY_VAULT_URL", "")
    enable_telemetry: bool = os.getenv("ENABLE_TELEMETRY", "false").lower() == "true"
    
    mcp_enabled: bool = os.getenv("MCP_ENABLED", "true").lower() == "true"
    mcp_server_port: int = int(os.getenv("MCP_SERVER_PORT", "3001"))
    a2a_enabled: bool = os.getenv("A2A_ENABLED", "true").lower() == "true"
    a2a_discovery_endpoint: str = os.getenv("A2A_DISCOVERY_ENDPOINT", "https://your-a2a-discovery.azure.com/")
    
    max_document_size_mb: int = int(os.getenv("MAX_DOCUMENT_SIZE_MB", "50"))
    supported_document_types: str = os.getenv("SUPPORTED_DOCUMENT_TYPES", "pdf,docx,xlsx,txt")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    max_chunks_per_document: int = int(os.getenv("MAX_CHUNKS_PER_DOCUMENT", "500"))
    
    # Evaluation Configuration
    AZURE_EVALUATION_ENDPOINT: str = os.getenv("AZURE_EVALUATION_ENDPOINT", "")
    AZURE_EVALUATION_API_KEY: str = os.getenv("AZURE_EVALUATION_API_KEY", "")
    AZURE_EVALUATION_MODEL_DEPLOYMENT: str = os.getenv("AZURE_EVALUATION_MODEL_DEPLOYMENT", "gpt-4o-mini")
    AZURE_EVALUATION_MODEL_NAME: str = os.getenv("AZURE_EVALUATION_MODEL_NAME", "gpt-4o-mini")
    
    # Azure AI Foundry Configuration for Evaluation
    AZURE_AI_PROJECT_CONNECTION_STRING: str = os.getenv("AZURE_AI_PROJECT_CONNECTION_STRING", "")
    AZURE_AI_PROJECT_NAME: str = os.getenv("AZURE_AI_PROJECT_NAME", "")
    AZURE_AI_HUB_NAME: str = os.getenv("AZURE_AI_HUB_NAME", "")
    
    # Evaluation Settings
    EVALUATION_ENABLED: bool = os.getenv("EVALUATION_ENABLED", "true").lower() == "true"
    DEFAULT_EVALUATOR_TYPE: str = os.getenv("DEFAULT_EVALUATOR_TYPE", "custom")  # "foundry" or "custom"
    
    AVAILABLE_EVALUATOR_TYPES: List[str] = ["foundry", "custom"]
    AVAILABLE_EVALUATION_MODELS: List[str] = [
        "o3-mini",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4"
    ]
    
    rate_limit_requests_per_minute: int = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "100"))
    rate_limit_tokens_per_minute: int = int(os.getenv("RATE_LIMIT_TOKENS_PER_MINUTE", "50000"))
    
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "your-jwt-secret-key-here")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_expiration_hours: int = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    
    AZURE_STORAGE_ACCOUNT_NAME: Optional[str] = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    AZURE_STORAGE_ACCOUNT_KEY: Optional[str] = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    AZURE_STORAGE_CONTAINER_NAME: str = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "financial-documents")
    
    mock_azure_services: bool = os.getenv("MOCK_AZURE_SERVICES", "false").lower() == "true"
    enable_debug_logging: bool = os.getenv("ENABLE_DEBUG_LOGGING", "false").lower() == "true"
    enable_performance_profiling: bool = os.getenv("ENABLE_PERFORMANCE_PROFILING", "false").lower() == "true"
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    allowed_origins: str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173")
    
    # MCP (Model Context Protocol) Configuration
    MCP_SERVER_URL: Optional[str] = os.getenv("MCP_SERVER_URL", None)
    MCP_SERVER_TIMEOUT: int = int(os.getenv("MCP_SERVER_TIMEOUT", "30"))
    
    # Additional constants (not environment-configurable)
    MAX_TOKENS_PER_REQUEST: int = 4000
    TEMPERATURE: float = 0.1
    
    AUTO_UPDATE_ENABLED: bool = True
    UPDATE_FREQUENCY_HOURS: int = 24
    CREDIBILITY_THRESHOLD: float = 0.7
    
    class Config:
        env_file = ".env"
        extra = "allow"  # Allow extra fields to prevent validation errors

settings = Settings()

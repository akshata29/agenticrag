"""
Token Usage Tracking Service

This service provides comprehensive token usage tracking and analytics
for all AI operations across the application. It stores detailed usage
data in CosmosDB for analytics and reporting.
"""

import logging
import time
import uuid
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from app.core.config import settings
from app.services.azure_services import AzureServiceManager

logger = logging.getLogger(__name__)

class ServiceType(str, Enum):
    """Service types for token usage tracking"""
    QA_SERVICE = "qa_service"
    CHAT_SERVICE = "chat_service"
    SEC_DOCS = "sec_docs"
    KNOWLEDGE_BASE = "knowledge_base"
    CREDIBILITY_ASSESSMENT = "credibility_assessment"
    QUESTION_DECOMPOSITION = "question_decomposition"
    RELEVANCE_EXPLANATION = "relevance_explanation"
    AGENT_SERVICE = "azure_ai_agent_service"
    MCP = "mcp"

class OperationType(str, Enum):
    """Operation types for detailed tracking"""
    SEARCH_QUERY = "search_query"
    ANSWER_GENERATION = "answer_generation"
    DOCUMENT_ANALYSIS = "document_analysis"
    CREDIBILITY_CHECK = "credibility_check"
    SUB_QUESTION_GENERATION = "sub_question_generation"
    RELEVANCE_EXPLANATION = "relevance_explanation"
    CHAT_COMPLETION = "chat_completion"
    EMBEDDING_GENERATION = "embedding_generation"
    QUESTION_DECOMPOSITION = "question_decomposition"
    SOURCE_VERIFICATION = "source_verification"
    QA = "qa"
    MCP_TOOL_CALL = "mcp_tool_call"

@dataclass
class TokenUsageRecord:
    """Comprehensive token usage record"""
    # Unique identifiers
    record_id: str
    session_id: str
    user_id: Optional[str] = None
    
    # Timing information
    timestamp: datetime = None
    request_start_time: float = None
    request_end_time: float = None
    duration_ms: float = None
    
    # Service information
    service_type: ServiceType = None
    operation_type: OperationType = None
    endpoint: str = None
    
    # Model information
    model_name: str = None  # Display name (e.g., "GPT-4o")
    deployment_name: str = None  # Azure deployment name (e.g., "chat4o")
    model_version: str = None
    model_provider: str = "azure_openai"
    
    # Azure resource information
    azure_region: str = None
    resource_group: str = None
    azure_subscription_id: str = None
    
    # Token usage details
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    # Cost information
    prompt_cost: float = 0.0
    completion_cost: float = 0.0
    total_cost: float = 0.0
    cost_currency: str = "USD"
      # Request context
    request_text: str = None
    response_text: str = None
    request_size_chars: int = 0
    response_size_chars: int = 0
    temperature: float = None
    max_tokens: int = None
    
    # Business context
    verification_level: str = None
    credibility_check_enabled: bool = False
    decomposition_enabled: bool = False
    exercise_type: str = None
    
    # Result information
    success: bool = True
    error_message: str = None
    http_status_code: int = 200
      # Additional metadata
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.record_id is None:
            self.record_id = str(uuid.uuid4())
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        if self.total_cost == 0.0:
            self.total_cost = self.prompt_cost + self.completion_cost
        if self.metadata is None:
            self.metadata = {}
        if self.request_text and self.request_size_chars == 0:
            self.request_size_chars = len(self.request_text)
        if self.response_text and self.response_size_chars == 0:
            self.response_size_chars = len(self.response_text)

class TokenUsageTracker:
    """Token usage tracking service with CosmosDB storage"""
    
    def __init__(self, azure_manager: AzureServiceManager = None):
        self.azure_manager = azure_manager
        self._active_sessions: Dict[str, TokenUsageRecord] = {}
        
        # Token pricing (per 1K tokens) - these should be updated based on current Azure OpenAI pricing
        self.token_pricing = {
            "gpt-4o": {"prompt": 0.005, "completion": 0.015},
            "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-35-turbo": {"prompt": 0.0015, "completion": 0.002},
            "text-embedding-3-small": {"prompt": 0.00002, "completion": 0.0},
            "text-embedding-3-large": {"prompt": 0.00013, "completion": 0.0},
            "text-embedding-ada-002": {"prompt": 0.0001, "completion": 0.0}
        }
    
    async def initialize(self):
        """Initialize the token usage tracker"""
        if not self.azure_manager:
            from app.services.azure_services import AzureServiceManager
            self.azure_manager = AzureServiceManager()
            await self.azure_manager.initialize()
    
    def start_tracking(self, 
                      session_id: str,
                      service_type: ServiceType,
                      operation_type: OperationType,
                      endpoint: str = None,
                      user_id: str = None,
                      **kwargs) -> str:
        """Start tracking a new token usage session"""
        
        record = TokenUsageRecord(
            record_id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id,
            service_type=service_type,
            operation_type=operation_type,
            endpoint=endpoint,
            request_start_time=time.time(),
            **kwargs
        )
        
        self._active_sessions[record.record_id] = record
        return record.record_id
    
    def update_model_info(self,
                         record_id: str,
                         model_name: str = None,
                         deployment_name: str = None,
                         model_version: str = None,
                         temperature: float = None,
                         max_tokens: int = None):
        """Update model information for an active tracking session"""
        if record_id in self._active_sessions:
            record = self._active_sessions[record_id]
            if model_name:
                record.model_name = model_name
            if deployment_name:
                record.deployment_name = deployment_name
            if model_version:
                record.model_version = model_version
            if temperature is not None:
                record.temperature = temperature
            if max_tokens:
                record.max_tokens = max_tokens
    
    def update_request_context(self,
                              record_id: str,
                              request_size_chars: int = None,
                              verification_level: str = None,
                              credibility_check_enabled: bool = None,
                              decomposition_enabled: bool = None,
                              metadata: Dict[str, Any] = None):
        """Update request context information"""
        if record_id in self._active_sessions:
            record = self._active_sessions[record_id]
            if request_size_chars is not None:
                record.request_size_chars = request_size_chars
            if verification_level:
                record.verification_level = verification_level
            if credibility_check_enabled is not None:
                record.credibility_check_enabled = credibility_check_enabled
            if decomposition_enabled is not None:
                record.decomposition_enabled = decomposition_enabled
            if metadata:
                record.metadata.update(metadata)
    
    def record_token_usage(self,
                          record_id: str,
                          prompt_tokens: int,
                          completion_tokens: int,
                          response_size_chars: int = None,
                          success: bool = True,
                          error_message: str = None,
                          http_status_code: int = 200) -> TokenUsageRecord:
        """Record token usage and finalize the tracking session"""
        
        if record_id not in self._active_sessions:
            logger.warning(f"Token usage record {record_id} not found in active sessions")
            return None
        
        record = self._active_sessions[record_id]
        
        # Update token usage
        record.prompt_tokens = prompt_tokens
        record.completion_tokens = completion_tokens
        record.total_tokens = prompt_tokens + completion_tokens
        
        # Calculate costs
        model_key = self._get_model_key_for_pricing(record.model_name or record.deployment_name)
        if model_key in self.token_pricing:
            pricing = self.token_pricing[model_key]
            record.prompt_cost = (prompt_tokens / 1000) * pricing["prompt"]
            record.completion_cost = (completion_tokens / 1000) * pricing["completion"]
            record.total_cost = record.prompt_cost + record.completion_cost
        
        # Update timing and result information
        record.request_end_time = time.time()
        if record.request_start_time:
            record.duration_ms = (record.request_end_time - record.request_start_time) * 1000
        
        if response_size_chars is not None:
            record.response_size_chars = response_size_chars
        
        record.success = success
        record.error_message = error_message
        record.http_status_code = http_status_code
        
        # Add Azure resource information if available
        try:
            record.azure_region = getattr(settings, 'AZURE_REGION', None) or os.getenv('AZURE_REGION')
            record.azure_subscription_id = getattr(settings, 'AZURE_SUBSCRIPTION_ID', None) or os.getenv('AZURE_SUBSCRIPTION_ID')
        except Exception:
            pass
          # Remove from active sessions
        del self._active_sessions[record_id]
        
        return record

    async def store_token_usage(self, record: TokenUsageRecord):
        """Store token usage record in CosmosDB"""
        try:
            logger.info(f"🔍 Starting storage of token usage record {record.record_id}")
            
            if not self.azure_manager:
                logger.error("❌ Azure manager not available")
                return
                
            if not self.azure_manager.cosmos_client:
                logger.error("❌ CosmosDB client not available in Azure manager")
                return
            
            logger.info(f"✅ CosmosDB client available, attempting to store record {record.record_id}")
            
            # Get the token usage container
            database = self.azure_manager.cosmos_client.get_database_client(settings.AZURE_COSMOS_DATABASE_NAME)
            container = database.get_container_client(settings.AZURE_COSMOS_TOKEN_USAGE_CONTAINER_NAME)
            
            logger.info(f"✅ Got database and container clients")
            
            # Convert record to dict for storage
            record_dict = asdict(record)
            
            # Convert datetime to ISO format for JSON serialization
            if isinstance(record_dict['timestamp'], datetime):
                record_dict['timestamp'] = record_dict['timestamp'].isoformat()
            
            # Use record_id as the document id
            record_dict['id'] = record.record_id
            
            logger.info(f"📝 Prepared record dict with ID: {record.record_id}, timestamp: {record_dict['timestamp']}")
            logger.info(f"📊 Record details: model={record.model_name}, tokens={record.total_tokens}, operation={record.operation_type}")
            
            # Store in CosmosDB
            result = container.upsert_item(record_dict)
            
            logger.info(f"✅ Successfully stored token usage record {record.record_id} in CosmosDB")
            logger.info(f"📄 CosmosDB result: {result.get('id', 'no-id')} with etag: {result.get('_etag', 'no-etag')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to store token usage record in CosmosDB: {e}")
            logger.error(f"📋 Record details: {record.record_id}, operation: {record.operation_type}, tokens: {record.total_tokens}")
            import traceback
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            # Re-raise to make the failure visible
            raise
    
    def _get_model_key_for_pricing(self, model_identifier: str) -> str:
        """Map model identifier to pricing key"""
        if not model_identifier:
            return "gpt-4o-mini"  # Default fallback
        
        model_lower = model_identifier.lower()
        
        # Map deployment names and model names to pricing keys
        if "gpt-4o-mini" in model_lower or "chat4omini" in model_lower:
            return "gpt-4o-mini"
        elif "gpt-4o" in model_lower or "chat4o" in model_lower:
            return "gpt-4o"
        elif "gpt-4" in model_lower:
            return "gpt-4"
        elif "gpt-35-turbo" in model_lower or "gpt-3.5-turbo" in model_lower:
            return "gpt-35-turbo"
        elif "text-embedding-3-small" in model_lower:
            return "text-embedding-3-small"        
        elif "text-embedding-3-large" in model_lower:
            return "text-embedding-3-large"
        elif "text-embedding-ada-002" in model_lower:
            return "text-embedding-ada-002"
        
        return "gpt-4o-mini"  # Default fallback
    
    async def get_token_usage_analytics(self, 
                                       start_date: datetime = None,
                                       end_date: datetime = None,
                                       service_type: ServiceType = None,
                                       model_name: str = None) -> Dict[str, Any]:
        """Get token usage analytics from CosmosDB"""
        try:
            if not self.azure_manager or not self.azure_manager.cosmos_client:
                logger.warning("CosmosDB client not available")
                return {}
            
            database = self.azure_manager.cosmos_client.get_database_client(settings.AZURE_COSMOS_DATABASE_NAME)
            container = database.get_container_client(settings.AZURE_COSMOS_TOKEN_USAGE_CONTAINER_NAME)
            
            # Build query
            query = "SELECT * FROM c WHERE 1=1"
            parameters = []
            
            if start_date:
                query += " AND c.timestamp >= @start_date"
                parameters.append({"name": "@start_date", "value": start_date.isoformat()})
            
            if end_date:
                query += " AND c.timestamp <= @end_date"
                parameters.append({"name": "@end_date", "value": end_date.isoformat()})
            
            if service_type:
                query += " AND c.service_type = @service_type"
                parameters.append({"name": "@service_type", "value": service_type.value})
            
            if model_name:
                query += " AND (c.model_name = @model_name OR c.deployment_name = @model_name)"
                parameters.append({"name": "@model_name", "value": model_name})
            
            # Execute query
            items = list(container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            # Aggregate the results
            analytics = self._aggregate_token_usage(items)
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get token usage analytics: {e}")
            return {}
    
    def _aggregate_token_usage(self, items: List[Dict]) -> Dict[str, Any]:
        """Aggregate token usage data for analytics"""
        if not items:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "by_service": {},
                "by_model": {},
                "by_date": {},
                "average_tokens_per_request": 0,
                "success_rate": 0.0
            }
        
        total_requests = len(items)
        total_tokens = sum(item.get('total_tokens', 0) for item in items)
        total_cost = sum(item.get('total_cost', 0.0) for item in items)
        successful_requests = sum(1 for item in items if item.get('success', True))
        
        # Aggregate by service
        by_service = {}
        for item in items:
            service = item.get('service_type', 'unknown')
            if service not in by_service:
                by_service[service] = {"requests": 0, "tokens": 0, "cost": 0.0}
            by_service[service]["requests"] += 1
            by_service[service]["tokens"] += item.get('total_tokens', 0)
            by_service[service]["cost"] += item.get('total_cost', 0.0)
        
        # Aggregate by model
        by_model = {}
        for item in items:
            model = item.get('model_name') or item.get('deployment_name', 'unknown')
            if model not in by_model:
                by_model[model] = {"requests": 0, "tokens": 0, "cost": 0.0}
            by_model[model]["requests"] += 1
            by_model[model]["tokens"] += item.get('total_tokens', 0)
            by_model[model]["cost"] += item.get('total_cost', 0.0)
        
        # Aggregate by date
        by_date = {}
        for item in items:
            timestamp = item.get('timestamp', '')
            date_key = timestamp[:10] if timestamp else 'unknown'  # YYYY-MM-DD
            if date_key not in by_date:
                by_date[date_key] = {"requests": 0, "tokens": 0, "cost": 0.0}
            by_date[date_key]["requests"] += 1
            by_date[date_key]["tokens"] += item.get('total_tokens', 0)
            by_date[date_key]["cost"] += item.get('total_cost', 0.0)
        
        return {
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_cost": round(total_cost, 4),
            "by_service": by_service,
            "by_model": by_model,
            "by_date": by_date,
            "average_tokens_per_request": round(total_tokens / total_requests, 2) if total_requests > 0 else 0,
            "success_rate": round((successful_requests / total_requests) * 100, 2) if total_requests > 0 else 0.0
        }
    
    async def get_usage_analytics(self, 
                                days_back: int = 7,
                                service_type: ServiceType = None,
                                deployment_name: str = None) -> Dict[str, Any]:
        """Get comprehensive usage analytics for the specified period"""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        return await self.get_token_usage_analytics(
            start_date=start_date,
            end_date=end_date,
            service_type=service_type,
            model_name=deployment_name
        )
    
    async def get_usage_summary(self, hours_back: int = 24) -> Dict[str, Any]:
        """Get usage summary for the specified hours"""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(hours=hours_back)
        
        analytics = await self.get_token_usage_analytics(
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "period_hours": hours_back,
            "total_requests": analytics.get("total_requests", 0),
            "total_tokens": analytics.get("total_tokens", 0),
            "total_cost": analytics.get("total_cost", 0.0),
            "average_tokens_per_request": analytics.get("average_tokens_per_request", 0),
            "success_rate": analytics.get("success_rate", 0.0),
            "top_services": sorted(
                analytics.get("by_service", {}).items(),
                key=lambda x: x[1]["tokens"],
                reverse=True
            )[:5],
            "top_models": sorted(
                analytics.get("by_model", {}).items(),
                key=lambda x: x[1]["tokens"],
                reverse=True
            )[:5]
        }
    
    async def get_usage_trends(self, 
                             days_back: int = 30,
                             granularity: str = "daily") -> Dict[str, Any]:
        """Get usage trends over time"""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        analytics = await self.get_token_usage_analytics(
            start_date=start_date,
            end_date=end_date
        )
        
        by_date = analytics.get("by_date", {})
        
        # Convert to time series format
        trends = []
        current_date = start_date.date()
        while current_date <= end_date.date():
            date_key = current_date.strftime("%Y-%m-%d")
            date_data = by_date.get(date_key, {"requests": 0, "tokens": 0, "cost": 0.0})
            trends.append({
                "date": date_key,
                "requests": date_data["requests"],
                "tokens": date_data["tokens"],
                "cost": date_data["cost"]
            })
            current_date += timedelta(days=1)
        
        return {
            "trends": trends,
            "granularity": granularity,
            "period_days": days_back
        }
    
    async def get_cost_analytics(self, 
                               days_back: int = 30,
                               service_type: ServiceType = None) -> Dict[str, Any]:
        """Get cost analytics for the specified period"""
        analytics = await self.get_usage_analytics(
            days_back=days_back,
            service_type=service_type
        )
        
        by_service = analytics.get("by_service", {})
        by_model = analytics.get("by_model", {})
        
        # Calculate cost breakdowns
        service_costs = {k: v["cost"] for k, v in by_service.items()}
        model_costs = {k: v["cost"] for k, v in by_model.items()}
        
        return {
            "total_cost": analytics.get("total_cost", 0.0),
            "cost_by_service": service_costs,
            "cost_by_model": model_costs,
            "cost_per_token": round(
                analytics.get("total_cost", 0.0) / analytics.get("total_tokens", 1),
                6
            ),
            "period_days": days_back
        }
    
    async def get_deployment_usage(self, days_back: int = 7) -> Dict[str, Any]:
        """Get usage breakdown by deployment"""
        analytics = await self.get_usage_analytics(days_back=days_back)
        
        by_model = analytics.get("by_model", {})
        
        # Sort by token usage
        deployment_usage = sorted(
            [
                {
                    "deployment_name": model,
                    "requests": data["requests"],
                    "tokens": data["tokens"],
                    "cost": data["cost"],
                    "percentage": round((data["tokens"] / analytics.get("total_tokens", 1)) * 100, 2)
                }
                for model, data in by_model.items()
            ],
            key=lambda x: x["tokens"],
            reverse=True
        )
        
        return {
            "deployments": deployment_usage,
            "total_deployments": len(deployment_usage),
            "period_days": days_back
        }
    
    async def get_service_usage(self, days_back: int = 7) -> Dict[str, Any]:
        """Get usage breakdown by service type"""
        analytics = await self.get_usage_analytics(days_back=days_back)
        
        by_service = analytics.get("by_service", {})
        
        # Sort by token usage
        service_usage = sorted(
            [
                {
                    "service_type": service,
                    "requests": data["requests"],
                    "tokens": data["tokens"],
                    "cost": data["cost"],
                    "percentage": round((data["tokens"] / analytics.get("total_tokens", 1)) * 100, 2)
                }
                for service, data in by_service.items()
            ],
            key=lambda x: x["tokens"],
            reverse=True
        )
        
        return {
            "services": service_usage,
            "total_services": len(service_usage),
            "period_days": days_back        }
        
    async def update_usage(self,
                          tracking_id: str,
                          model_name: str = None,
                          deployment_name: str = None,
                          prompt_tokens: int = 0,
                          completion_tokens: int = 0,
                          response_text: str = None,
                          **kwargs):
        """Update token usage for an active tracking session - accumulates tokens"""
        if tracking_id in self._active_sessions:
            record = self._active_sessions[tracking_id]
            if model_name:
                record.model_name = model_name
            if deployment_name:
                record.deployment_name = deployment_name
            
            # ACCUMULATE token usage instead of overwriting
            if prompt_tokens:
                record.prompt_tokens += prompt_tokens
            if completion_tokens:
                record.completion_tokens += completion_tokens
            
            if response_text:
                # For response text, append if we already have some, otherwise set
                if record.response_text:
                    record.response_text += f"\n---\n{response_text}"
                else:
                    record.response_text = response_text
                record.response_size_chars = len(record.response_text)
            
            # Update total tokens
            record.total_tokens = record.prompt_tokens + record.completion_tokens
            
            # Calculate costs
            model_key = self._get_model_key_for_pricing(record.model_name or record.deployment_name)
            if model_key in self.token_pricing:
                pricing = self.token_pricing[model_key]
                record.prompt_cost = (record.prompt_tokens / 1000) * pricing["prompt"]
                record.completion_cost = (record.completion_tokens / 1000) * pricing["completion"]
                record.total_cost = record.prompt_cost + record.completion_cost
                
            logger.debug(f"Updated usage for {tracking_id}: prompt={record.prompt_tokens}, completion={record.completion_tokens}, total={record.total_tokens}")
        else:
            logger.warning(f"Tracking session {tracking_id} not found")

    async def finalize_tracking(self,
                               tracking_id: str,
                               success: bool = True,
                               http_status_code: int = 200,
                               error_message: str = None,
                               metadata: Dict[str, Any] = None):
        """Finalize and store a tracking session"""
        if tracking_id not in self._active_sessions:
            logger.warning(f"Tracking session {tracking_id} not found")
            return
            
        record = self._active_sessions[tracking_id]
        
        # Update final status
        record.success = success
        record.http_status_code = http_status_code
        record.error_message = error_message
        
        # Update timing
        record.request_end_time = time.time()
        if record.request_start_time:
            record.duration_ms = (record.request_end_time - record.request_start_time) * 1000
        
        # Add Azure resource information if available
        try:
            record.azure_region = getattr(settings, 'AZURE_REGION', None) or os.getenv('AZURE_REGION')
            record.azure_subscription_id = getattr(settings, 'AZURE_SUBSCRIPTION_ID', None) or os.getenv('AZURE_SUBSCRIPTION_ID')
        except Exception:
            pass
        
        # Merge metadata
        if metadata:
            if record.metadata:
                record.metadata.update(metadata)
            else:
                record.metadata = metadata
        
        # Store in CosmosDB
        await self.store_token_usage(record)
        
        # Remove from active sessions
        del self._active_sessions[tracking_id]
        
        logger.info(f"Finalized token tracking for session {tracking_id}")
    
    async def get_detailed_requests(
        self, 
        days_back: int = 7,
        service_type: Optional[ServiceType] = None,
        deployment_name: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get detailed request logs for admin dashboard"""
        try:
            if not self.azure_manager or not self.azure_manager.cosmos_client:
                return []
            
            start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
            
            # Get the container
            database = self.azure_manager.cosmos_client.get_database_client(settings.AZURE_COSMOS_DATABASE_NAME)
            container = database.get_container_client(settings.AZURE_COSMOS_TOKEN_USAGE_CONTAINER_NAME)
            
            # Build query
            query = "SELECT * FROM c WHERE c.timestamp >= @start_date"
            parameters = [{"name": "@start_date", "value": start_date.isoformat()}]
            
            if service_type:
                query += " AND c.service_type = @service_type"
                parameters.append({"name": "@service_type", "value": service_type.value})
            
            if deployment_name:
                query += " AND (c.model_name = @deployment_name OR c.deployment_name = @deployment_name)"
                parameters.append({"name": "@deployment_name", "value": deployment_name})
            
            # Add ordering and pagination
            query += " ORDER BY c.timestamp DESC"
            
            # Execute query
            items = list(container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            # Apply pagination
            paginated_items = items[offset:offset + limit]
            
            # Format the results for the admin dashboard
            detailed_requests = []
            for item in paginated_items:
                detailed_request = {
                    "record_id": item.get("record_id"),
                    "timestamp": item.get("timestamp"),
                    "session_id": item.get("session_id"),
                    "service_type": item.get("service_type"),
                    "operation_type": item.get("operation_type"),
                    "model_name": item.get("model_name"),
                    "deployment_name": item.get("deployment_name"),
                    "request_text": item.get("request_text", "")[:200] + "..." if item.get("request_text", "") and len(item.get("request_text", "")) > 200 else item.get("request_text", ""),
                    "response_text": item.get("response_text", "")[:200] + "..." if item.get("response_text", "") and len(item.get("response_text", "")) > 200 else item.get("response_text", ""),
                    "prompt_tokens": item.get("prompt_tokens", 0),
                    "completion_tokens": item.get("completion_tokens", 0),
                    "total_tokens": item.get("total_tokens", 0),
                    "total_cost": item.get("total_cost", 0.0),
                    "duration_ms": item.get("duration_ms", 0),
                    "success": item.get("success", True),
                    "verification_level": item.get("verification_level"),
                    "credibility_check_enabled": item.get("credibility_check_enabled", False),
                    "temperature": item.get("temperature"),
                    "max_tokens": item.get("max_tokens"),
                    "error_message": item.get("error_message")
                }
                detailed_requests.append(detailed_request)
            
            return detailed_requests
            
        except Exception as e:
            logger.error(f"Failed to get detailed requests: {e}")
            return []
        
# Global instance
token_tracker = TokenUsageTracker()

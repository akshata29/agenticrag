import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager, contextmanager
import os

# Temporarily disable OpenTelemetry imports due to version conflicts
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logging.warning("OpenTelemetry packages not available. Tracing features will be limited.")
    
    # Mock classes for when OpenTelemetry is not available
    class StatusCode:
        OK = "OK"
        ERROR = "ERROR"
    
    class Status:
        def __init__(self, code, description=""):
            self.code = code
            self.description = description

AZURE_MONITOR_AVAILABLE = False  # Temporarily disabled

try:
    if OPENTELEMETRY_AVAILABLE:
        from opentelemetry.instrumentation.openai import OpenAIInstrumentor
    OPENAI_INSTRUMENTATION_AVAILABLE = OPENTELEMETRY_AVAILABLE
except ImportError:
    OPENAI_INSTRUMENTATION_AVAILABLE = False
    logging.warning("OpenAI instrumentation not available. Install opentelemetry-instrumentation-openai-v2 for full tracing support.")

try:
    from azure.ai.projects import AIProjectClient
    # from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
    AZURE_AI_FOUNDRY_TRACING_AVAILABLE = True
except ImportError:
    AZURE_AI_FOUNDRY_TRACING_AVAILABLE = False
    logging.warning("Azure AI Foundry tracing packages not available. Some advanced tracing features may be limited.")

def setup_observability():
    """Setup Azure Monitor observability with OpenTelemetry and Azure AI Foundry tracing"""
    from app.core.config import settings
    
    # Check if telemetry is enabled
    if not settings.enable_telemetry:
        logging.info("Telemetry is disabled via ENABLE_TELEMETRY setting. Skipping Application Insights setup.")
        # Return basic tracer and meter without Azure Monitor
        if OPENTELEMETRY_AVAILABLE:
            tracer = trace.get_tracer(__name__)
            meter = metrics.get_meter(__name__)
        else:
            tracer = None
            meter = None
        return tracer, meter
    
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    azure_monitor_connection_string = os.getenv("AZURE_MONITOR_CONNECTION_STRING")
    
    if AZURE_MONITOR_AVAILABLE and (connection_string or azure_monitor_connection_string):
        try:
            # configure_azure_monitor(
            #     connection_string=connection_string or azure_monitor_connection_string
            # )
            logging.info("Azure Monitor configured with Application Insights")
        except Exception as e:
            logging.error(f"Failed to configure Azure Monitor: {e}")
    else:
        logging.warning("Azure Monitor temporarily disabled due to version conflicts. Some tracing features may be limited.")
    
    if os.getenv("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "false").lower() == "true":
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "true"
        logging.info("GenAI message content capture enabled")
    
    if os.getenv("AZURE_AI_FOUNDRY_TRACING_ENABLED", "true").lower() == "true":
        os.environ["AZURE_AI_FOUNDRY_TRACING_ENABLED"] = "true"
        logging.info("Azure AI Foundry agent tracing enabled")
    
    if OPENAI_INSTRUMENTATION_AVAILABLE:
        try:
            OpenAIInstrumentor().instrument()
            logging.info("OpenAI SDK instrumentation enabled")
        except Exception as e:
            logging.error(f"Failed to instrument OpenAI SDK: {e}")
    else:
        logging.warning("OpenAI instrumentation not available. Install opentelemetry-instrumentation-openai-v2")
    
    if AZURE_AI_FOUNDRY_TRACING_AVAILABLE:
        try:
            project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
            if project_endpoint:
                logging.info(f"Azure AI Foundry project tracing configured for endpoint: {project_endpoint}")
        except Exception as e:
            logging.error(f"Failed to configure Azure AI Foundry project tracing: {e}")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    tracer = trace.get_tracer(__name__)
    meter = metrics.get_meter(__name__)
    
    request_counter = meter.create_counter(
        "rag_requests_total",
        description="Total number of RAG requests"
    )
    
    token_usage_counter = meter.create_counter(
        "token_usage_total", 
        description="Total token usage"
    )
    
    knowledge_base_updates = meter.create_counter(
        "knowledge_base_updates_total",
        description="Total knowledge base updates"
    )
    
    agent_operations_counter = meter.create_counter(
        "agent_operations_total",
        description="Total Azure AI Agent Service operations"
    )
    
    return tracer, meter

class ObservabilityManager:
    def __init__(self):
        from app.core.config import settings
        
        self.telemetry_enabled = settings.enable_telemetry
        self.tracer, self.meter = setup_observability()
        
        if self.telemetry_enabled and self.meter:
            self.request_counter = self.meter.create_counter(
                "rag_requests_total",
                description="Total number of RAG requests"
            )
            self.token_counter = self.meter.create_counter(
                "token_usage_total",
                description="Total token usage"
            )
            self.kb_update_counter = self.meter.create_counter(
                "knowledge_base_updates_total", 
                description="Knowledge base updates"
            )
            self.error_counter = self.meter.create_counter(
                "rag_errors_total",
                description="Total number of errors"
            )
            
            self.response_time_histogram = self.meter.create_histogram(
                "rag_response_time_seconds",
                description="Response time in seconds"
            )
            self.evaluation_time_histogram = self.meter.create_histogram(
                "evaluation_time_seconds", 
                description="Evaluation time in seconds"
            )
        else:
            # Create null objects when telemetry is disabled
            self.request_counter = None
            self.token_counter = None
            self.kb_update_counter = None
            self.error_counter = None
            self.response_time_histogram = None
            self.evaluation_time_histogram = None
        
        self.active_sessions_gauge = self.meter.create_up_down_counter(
            "active_sessions_current",
            description="Current number of active sessions"
        )
        
        self.agent_operations_counter = self.meter.create_counter(
            "agent_operations_total",
            description="Total Azure AI Agent Service operations"
        )
        
        self.azure_ai_foundry_operations_counter = self.meter.create_counter(
            "azure_ai_foundry_operations_total",
            description="Total Azure AI Foundry operations"
        )
        
        self.metrics_storage = {
            "requests": [],
            "tokens": [],
            "evaluations": [],
            "errors": [],
            "response_times": [],
            "system_metrics": [],
            "agent_operations": [],
            "azure_ai_foundry_operations": []
        }
        
    def track_request(self, endpoint: str, user_id: str = None, session_id: str = None):
        """Track API request with enhanced metadata"""
        if not self.telemetry_enabled:
            return
            
        try:
            if self.request_counter is not None:
                attributes = {"endpoint": endpoint}
                if user_id:
                    attributes["user_id"] = user_id
                if session_id:
                    attributes["session_id"] = session_id
                    
                self.request_counter.add(1, attributes)
        except Exception as e:
            logging.warning(f"Failed to track request metrics: {e}")
        
        try:
            self.metrics_storage["requests"].append({
                "timestamp": datetime.utcnow(),
                "endpoint": endpoint,
                "user_id": user_id,
                "session_id": session_id
            })
        except Exception as e:
            logging.warning(f"Failed to store request metrics: {e}")
        
    def track_tokens(self, model: str, prompt_tokens: int, completion_tokens: int, 
                    session_id: str = None, cost: float = None):
        """Track token usage with cost calculation"""
        if not self.telemetry_enabled or not self.token_counter:
            return
            
        self.token_counter.add(prompt_tokens, {"model": model, "type": "prompt"})
        self.token_counter.add(completion_tokens, {"model": model, "type": "completion"})
        
        token_data = {
            "timestamp": datetime.utcnow(),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "session_id": session_id,
            "cost": cost or self._calculate_cost(model, prompt_tokens, completion_tokens)
        }
        self.metrics_storage["tokens"].append(token_data)
        
    def track_response_time(self, endpoint: str, duration: float, model: str = None, session_id: str = None):
        """Track response time"""
        if not self.telemetry_enabled or not self.response_time_histogram:
            return
            
        attributes = {"endpoint": endpoint}
        if model:
            attributes["model"] = model
        if session_id:
            attributes["session_id"] = session_id
            
        self.response_time_histogram.record(duration, attributes)
        
        self.metrics_storage["response_times"].append({
            "timestamp": datetime.utcnow(),
            "endpoint": endpoint,
            "duration": duration,
            "model": model,
            "session_id": session_id
        })
        
    def track_evaluation_metrics(self, session_id: str, evaluation_results: List[Dict[str, Any]], evaluator_framework: str = "custom"):
        """Track evaluation metrics from the evaluation framework with Azure AI Foundry support"""
        for result in evaluation_results:
            eval_data = {
                "timestamp": datetime.utcnow(),
                "session_id": session_id,
                "metric": result.get("metric"),
                "score": result.get("score", 0.0),
                "model": result.get("model_used"),
                "reasoning": result.get("reasoning", ""),
                "evaluator_framework": evaluator_framework,
                "evaluation_type": result.get("evaluation_type", "rag")
            }
            self.metrics_storage["evaluations"].append(eval_data)
            
            with self.tracer.start_as_current_span(f"evaluation.{result.get('metric', 'unknown')}") as span:
                span.set_attribute("ai.system", "rag_financial_assistant")
                span.set_attribute("ai.operation.type", "evaluation")
                span.set_attribute("evaluation.metric", result.get("metric", "unknown"))
                span.set_attribute("evaluation.score", result.get("score", 0.0))
                span.set_attribute("evaluation.framework", evaluator_framework)
                span.set_attribute("session_id", session_id)
                if result.get("model_used"):
                    span.set_attribute("ai.model.name", result.get("model_used"))
            
    def track_error(self, error_type: str, endpoint: str, error_message: str, 
                   session_id: str = None, user_id: str = None):
        """Track errors"""
        if not self.telemetry_enabled or not self.error_counter:
            return
            
        attributes = {
            "error_type": error_type,
            "endpoint": endpoint
        }
        if session_id:
            attributes["session_id"] = session_id
        if user_id:
            attributes["user_id"] = user_id
            
        self.error_counter.add(1, attributes)
        
        self.metrics_storage["errors"].append({
            "timestamp": datetime.utcnow(),
            "error_type": error_type,
            "endpoint": endpoint,
            "error_message": error_message,
            "session_id": session_id,
            "user_id": user_id
        })
        
    def track_kb_update(self, source: str, documents_added: int, documents_updated: int):
        """Track knowledge base updates"""
        if not self.telemetry_enabled or not self.kb_update_counter:
            return
            
        self.kb_update_counter.add(documents_added, {"source": source, "type": "added"})
        self.kb_update_counter.add(documents_updated, {"source": source, "type": "updated"})
        
    def track_document_processing_start(self, source: str, content_type: str):
        """Track the start of document processing"""
        self.metrics_storage["requests"].append({
            "timestamp": datetime.utcnow(),
            "endpoint": "document_processing",
            "source": source,
            "content_type": content_type,
            "status": "started"
        })
        
    def track_document_processing_error(self, source: str, error_message: str):
        """Track document processing errors"""
        self.error_counter.add(1, {"source": source, "type": "document_processing"})
        self.metrics_storage["errors"].append({
            "timestamp": datetime.utcnow(),
            "error_type": "document_processing",
            "endpoint": "document_processing",
            "error_message": error_message,
            "source": source
        })
        
    def track_document_processing_complete(self, source: str, chunks_created: int, processing_time: float):
        """Track successful completion of document processing"""
        self.kb_update_counter.add(chunks_created, {"source": source, "type": "chunks_created"})
        self.metrics_storage["requests"].append({
            "timestamp": datetime.utcnow(),
            "endpoint": "document_processing",
            "source": source,
            "status": "completed",
            "chunks_created": chunks_created,
            "processing_time": processing_time
        })
        
    def track_system_metrics(self, cpu_usage: float, memory_usage: float, disk_usage: float):
        """Track system performance metrics"""
        system_data = {
            "timestamp": datetime.utcnow(),
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "disk_usage": disk_usage
        }
        self.metrics_storage["system_metrics"].append(system_data)
        
    def _calculate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on model and token usage"""
        pricing = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
            "gpt-35-turbo": {"prompt": 0.0015, "completion": 0.002},
            "text-embedding-ada-002": {"prompt": 0.0001, "completion": 0.0001},
            "text-embedding-3-small": {"prompt": 0.00002, "completion": 0.00002},
            "text-embedding-3-large": {"prompt": 0.00013, "completion": 0.00013}
        }
        
        model_pricing = pricing.get(model, {"prompt": 0.001, "completion": 0.001})
        prompt_cost = (prompt_tokens / 1000) * model_pricing["prompt"]
        completion_cost = (completion_tokens / 1000) * model_pricing["completion"]
        
        return prompt_cost + completion_cost
        
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive metrics summary for admin dashboard"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_requests = [r for r in self.metrics_storage["requests"] if r["timestamp"] > cutoff_time]
        recent_tokens = [t for t in self.metrics_storage["tokens"] if t["timestamp"] > cutoff_time]
        recent_evaluations = [e for e in self.metrics_storage["evaluations"] if e["timestamp"] > cutoff_time]
        recent_errors = [e for e in self.metrics_storage["errors"] if e["timestamp"] > cutoff_time]
        recent_response_times = [r for r in self.metrics_storage["response_times"] if r["timestamp"] > cutoff_time]
        
        recent_agent_ops = [op for op in self.metrics_storage["agent_operations"] 
                           if op["timestamp"] > cutoff_time]
        agent_ops_by_type = {}
        for op in recent_agent_ops:
            agent_type = op.get("agent_type", "unknown")
            if agent_type not in agent_ops_by_type:
                agent_ops_by_type[agent_type] = {"count": 0, "avg_duration": 0}
            agent_ops_by_type[agent_type]["count"] += 1
            if op.get("duration"):
                current_avg = agent_ops_by_type[agent_type]["avg_duration"]
                count = agent_ops_by_type[agent_type]["count"]
                agent_ops_by_type[agent_type]["avg_duration"] = (current_avg * (count - 1) + op["duration"]) / count
        
        recent_foundry_ops = [op for op in self.metrics_storage["azure_ai_foundry_operations"] 
                             if op["timestamp"] > cutoff_time]
        foundry_ops_by_type = {}
        for op in recent_foundry_ops:
            op_type = op.get("operation_type", "unknown")
            if op_type not in foundry_ops_by_type:
                foundry_ops_by_type[op_type] = {"count": 0, "avg_duration": 0}
            foundry_ops_by_type[op_type]["count"] += 1
            if op.get("duration"):
                current_avg = foundry_ops_by_type[op_type]["avg_duration"]
                count = foundry_ops_by_type[op_type]["count"]
                foundry_ops_by_type[op_type]["avg_duration"] = (current_avg * (count - 1) + op["duration"]) / count
        
        total_tokens = sum(t["total_tokens"] for t in recent_tokens)
        total_cost = sum(t["cost"] for t in recent_tokens)
        total_requests = len(recent_requests)
        
        avg_response_time = 0
        if recent_response_times:
            avg_response_time = sum(r["duration"] for r in recent_response_times) / len(recent_response_times)
        
        token_by_model = {}
        for token_data in recent_tokens:
            model = token_data["model"]
            if model not in token_by_model:
                token_by_model[model] = {"prompt": 0, "completion": 0, "total": 0, "cost": 0}
            token_by_model[model]["prompt"] += token_data["prompt_tokens"]
            token_by_model[model]["completion"] += token_data["completion_tokens"]
            token_by_model[model]["total"] += token_data["total_tokens"]
            token_by_model[model]["cost"] += token_data["cost"]
        
        evaluation_summary = {}
        for eval_data in recent_evaluations:
            metric = eval_data["metric"]
            if metric not in evaluation_summary:
                evaluation_summary[metric] = {"scores": [], "count": 0}
            evaluation_summary[metric]["scores"].append(eval_data["score"])
            evaluation_summary[metric]["count"] += 1
        
        for metric, data in evaluation_summary.items():
            if data["scores"]:
                data["average"] = sum(data["scores"]) / len(data["scores"])
                data["min"] = min(data["scores"])
                data["max"] = max(data["scores"])
            else:
                data["average"] = 0
                data["min"] = 0
                data["max"] = 0
        
        system_health = "Healthy"
        if len(recent_errors) > 10:
            system_health = "Warning"
        if len(recent_errors) > 50:
            system_health = "Critical"
        
        return {
            "summary": {
                "total_tokens": total_tokens,
                "total_cost": round(total_cost, 4),
                "total_requests": total_requests,
                "total_errors": len(recent_errors),
                "avg_response_time": round(avg_response_time, 3),
                "system_health": system_health,
                "time_range_hours": hours,
                "total_agent_operations": len(recent_agent_ops),
                "total_foundry_operations": len(recent_foundry_ops)
            },
            "token_usage": {
                "by_model": token_by_model,
                "recent_usage": recent_tokens[-10:] if recent_tokens else []
            },
            "evaluation_metrics": evaluation_summary,
            "response_times": {
                "recent": recent_response_times[-20:] if recent_response_times else [],
                "average": avg_response_time
            },
            "errors": {
                "recent": recent_errors[-10:] if recent_errors else [],
                "count": len(recent_errors)
            },
            "requests": {
                "recent": recent_requests[-10:] if recent_requests else [],
                "count": total_requests
            },
            "agent_operations": {
                "recent": recent_agent_ops[-10:] if recent_agent_ops else [],
                "count": len(recent_agent_ops),
                "by_type": agent_ops_by_type
            },
            "azure_ai_foundry_operations": {
                "recent": recent_foundry_ops[-10:] if recent_foundry_ops else [],
                "count": len(recent_foundry_ops),
                "by_type": foundry_ops_by_type
            }
        }
        
    @asynccontextmanager
    async def trace_operation(self, operation_name: str, **attributes):
        """Context manager for distributed tracing with Azure AI Foundry compatibility"""
        with self.tracer.start_as_current_span(operation_name) as span:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
            
            span.set_attribute("ai.system", "rag_financial_assistant")
            span.set_attribute("ai.operation.name", operation_name)
            
            start_time = time.time()
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration_seconds", duration)
                span.set_attribute("ai.operation.duration", duration)
    
    @contextmanager
    def trace_operation_sync(self, operation_name: str, **attributes):
        """Synchronous context manager for distributed tracing with Azure AI Foundry compatibility"""
        with self.tracer.start_as_current_span(operation_name) as span:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
            
            span.set_attribute("ai.system", "rag_financial_assistant")
            span.set_attribute("ai.operation.name", operation_name)
            
            start_time = time.time()
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration_seconds", duration)
                span.set_attribute("ai.operation.duration", duration)

    @asynccontextmanager
    async def trace_rag_operation(self, operation_type: str, query: str = None, model: str = None, **attributes):
        """Specialized tracing for RAG operations with Azure AI Foundry semantics"""
        operation_name = f"rag.{operation_type}"
        
        with self.tracer.start_as_current_span(operation_name) as span:
            span.set_attribute("ai.system", "rag_financial_assistant")
            span.set_attribute("ai.operation.name", operation_name)
            span.set_attribute("ai.operation.type", operation_type)
            
            if query:
                span.set_attribute("ai.prompt", query[:500])  # Truncate long queries
            if model:
                span.set_attribute("ai.model.name", model)
            
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
            
            start_time = time.time()
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration_seconds", duration)
                span.set_attribute("ai.operation.duration", duration)
    
    @asynccontextmanager
    async def trace_evaluation_operation(self, evaluator_type: str, metric: str, **attributes):
        """Specialized tracing for evaluation operations"""
        operation_name = f"evaluation.{evaluator_type}"
        
        with self.tracer.start_as_current_span(operation_name) as span:
            span.set_attribute("ai.system", "rag_financial_assistant")
            span.set_attribute("ai.operation.name", operation_name)
            span.set_attribute("ai.operation.type", "evaluation")
            span.set_attribute("evaluation.type", evaluator_type)
            span.set_attribute("evaluation.metric", metric)
            
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
            
            start_time = time.time()
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration_seconds", duration)
                span.set_attribute("evaluation.duration", duration)
    
    @asynccontextmanager
    async def trace_agent_operation(self, agent_type: str, operation: str, agent_id: str = None, thread_id: str = None, **attributes):
        """Specialized tracing for Azure AI Agent Service operations with Azure AI Foundry compatibility"""
        operation_name = f"agent.{agent_type}.{operation}"
        
        with self.tracer.start_as_current_span(operation_name) as span:
            span.set_attribute("ai.system", "rag_financial_assistant")
            span.set_attribute("ai.operation.name", operation_name)
            span.set_attribute("ai.operation.type", "agent")
            span.set_attribute("ai.agent.type", agent_type)
            span.set_attribute("ai.agent.operation", operation)
            
            if agent_id:
                span.set_attribute("ai.agent.id", agent_id)
            if thread_id:
                span.set_attribute("ai.agent.thread_id", thread_id)
            
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
            
            start_time = time.time()
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
                
                self.agent_operations_counter.add(1, {
                    "agent_type": agent_type,
                    "operation": operation,
                    "status": "success"
                })
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                
                self.agent_operations_counter.add(1, {
                    "agent_type": agent_type,
                    "operation": operation,
                    "status": "error"
                })
                
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration_seconds", duration)
                span.set_attribute("ai.agent.duration", duration)
                
                self.metrics_storage["agent_operations"].append({
                    "timestamp": datetime.utcnow(),
                    "agent_type": agent_type,
                    "operation": operation,
                    "agent_id": agent_id,
                    "thread_id": thread_id,
                    "duration": duration,
                    "attributes": attributes
                })
    
    @asynccontextmanager
    async def trace_azure_ai_foundry_operation(self, operation_type: str, evaluator_type: str = None, **attributes):
        """Specialized tracing for Azure AI Foundry operations"""
        operation_name = f"azure_ai_foundry.{operation_type}"
        
        with self.tracer.start_as_current_span(operation_name) as span:
            span.set_attribute("ai.system", "rag_financial_assistant")
            span.set_attribute("ai.operation.name", operation_name)
            span.set_attribute("ai.operation.type", "azure_ai_foundry")
            span.set_attribute("ai.foundry.operation", operation_type)
            
            if evaluator_type:
                span.set_attribute("ai.foundry.evaluator_type", evaluator_type)
            
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
            
            start_time = time.time()
            try:
                yield span
                span.set_status(Status(StatusCode.OK))
                
                self.azure_ai_foundry_operations_counter.add(1, {
                    "operation_type": operation_type,
                    "evaluator_type": evaluator_type or "none",
                    "status": "success"
                })
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                
                self.azure_ai_foundry_operations_counter.add(1, {
                    "operation_type": operation_type,
                    "evaluator_type": evaluator_type or "none",
                    "status": "error"
                })
                
                raise
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration_seconds", duration)
                span.set_attribute("ai.foundry.duration", duration)
                
                self.metrics_storage["azure_ai_foundry_operations"].append({
                    "timestamp": datetime.utcnow(),
                    "operation_type": operation_type,
                    "evaluator_type": evaluator_type,
                    "duration": duration,
                    "attributes": attributes
                })
    
    def record_agent_operation(self, agent_type: str, operation: str, agent_id: str = None, thread_id: str = None, duration: float = None, status: str = "success", **metadata):
        """Record agent operation metrics for observability"""
        self.agent_operations_counter.add(1, {
            "agent_type": agent_type,
            "operation": operation,
            "status": status
        })
        
        self.metrics_storage["agent_operations"].append({
            "timestamp": datetime.utcnow(),
            "agent_type": agent_type,
            "operation": operation,
            "agent_id": agent_id,
            "thread_id": thread_id,
            "duration": duration,
            "status": status,
            "metadata": metadata
        })
    
    def record_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Record error with enhanced context for Azure AI Foundry tracing"""
        self.error_counter.add(1, {"error_type": error_type})
        
        error_data = {
            "timestamp": datetime.utcnow(),
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }
        
        self.metrics_storage["errors"].append(error_data)
        
        try:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                current_span.set_attribute("error.type", error_type)
                current_span.set_attribute("error.message", error_message)
                current_span.set_status(Status(StatusCode.ERROR, error_message))
                if context:
                    for key, value in context.items():
                        current_span.set_attribute(f"error.context.{key}", str(value))
        except Exception:
            pass  # Don't fail if tracing is not available

# Mock implementations when OpenTelemetry is not available
if not OPENTELEMETRY_AVAILABLE:
    class MockSpan:
        def set_attribute(self, key, value):
            pass
        
        def set_status(self, status):
            pass
            
        def record_exception(self, exception):
            pass
    
    class MockTracer:
        def start_as_current_span(self, name):
            from contextlib import contextmanager
            @contextmanager
            def mock_span():
                yield MockSpan()
            return mock_span()
    
    class MockMeter:
        def create_counter(self, name, description=None):
            class MockCounter:
                def add(self, amount, attributes=None):
                    pass
            return MockCounter()
        
        def create_histogram(self, name, description=None):
            class MockHistogram:
                def record(self, amount, attributes=None):
                    pass
            return MockHistogram()
        
        def create_up_down_counter(self, name, description=None):
            class MockUpDownCounter:
                def add(self, amount, attributes=None):
                    pass
            return MockUpDownCounter()
    
    class MockTrace:
        def get_tracer(self, name):
            return MockTracer()
    
    class MockMetrics:
        def get_meter(self, name):
            return MockMeter()
    
    trace = MockTrace()
    metrics = MockMetrics()

observability = ObservabilityManager()

def trace_operation(operation_name: str, **attributes):
    """Module-level trace_operation function"""
    return observability.trace_operation_sync(operation_name, **attributes)

def record_error(error_type: str, error_message: str, **attributes):
    """Module-level record_error function"""
    return observability.record_error(error_type, error_message, **attributes)

def setup_fastapi_instrumentation(app):
    """Setup FastAPI instrumentation for distributed tracing with Azure AI Foundry support"""
    try:
        if OPENTELEMETRY_AVAILABLE:
            FastAPIInstrumentor.instrument_app(app)
            HTTPXClientInstrumentor().instrument()
        
        if OPENAI_INSTRUMENTATION_AVAILABLE:
            # OpenAI instrumentation is already set up in setup_observability()
            logging.info("FastAPI and OpenAI instrumentation configured for Azure AI Foundry tracing")
        else:
            logging.info("FastAPI instrumentation disabled due to missing OpenTelemetry packages")
        
        if AZURE_AI_FOUNDRY_TRACING_AVAILABLE:
            project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
            if project_endpoint:
                logging.info(f"Azure AI Foundry tracing configured for project endpoint: {project_endpoint}")
        
        @app.middleware("http")
        async def azure_ai_foundry_trace_middleware(request, call_next):
            trace_id = request.headers.get("x-azure-ai-trace-id")
            operation_id = request.headers.get("x-azure-ai-operation-id")
            
            if trace_id or operation_id:
                current_span = trace.get_current_span()
                if current_span and current_span.is_recording():
                    if trace_id:
                        current_span.set_attribute("azure.ai.trace_id", trace_id)
                    if operation_id:
                        current_span.set_attribute("azure.ai.operation_id", operation_id)
            
            response = await call_next(request)
            
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                span_context = current_span.get_span_context()
                if span_context:
                    response.headers["x-azure-ai-trace-id"] = format(span_context.trace_id, '032x')
                    response.headers["x-azure-ai-span-id"] = format(span_context.span_id, '016x')
            
            return response
        
        logging.info("FastAPI instrumentation configured with Azure AI Foundry compatibility and trace correlation")
        
    except Exception as e:
        logging.error(f"Failed to configure FastAPI instrumentation: {e}")
        raise

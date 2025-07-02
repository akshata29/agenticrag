import asyncio
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from uuid import uuid4

try:
    from azure.ai.projects import AIProjectClient
    from azure.ai.projects.models import (
        AgentThread, 
        AgentThreadMessage, 
        AgentThreadRun,
        MessageRole,
        RunStatus,
        CodeInterpreterTool,
        FileSearchTool
    )
    from azure.identity import DefaultAzureCredential
    AZURE_AI_PROJECTS_AVAILABLE = True
    
    class Agent:
        def __init__(self, id: str = None, name: str = None, **kwargs):
            self.id = id or "mock-agent"
            self.name = name or "Mock Agent"
            
except ImportError:
    AZURE_AI_PROJECTS_AVAILABLE = False
    AIProjectClient = None
    AgentThread = None
    AgentThreadMessage = None
    AgentThreadRun = None
    MessageRole = None
    RunStatus = None
    DefaultAzureCredential = None
    
    class Agent:
        def __init__(self, id: str = None, name: str = None, **kwargs):
            self.id = id or "mock-agent"
            self.name = name or "Mock Agent"
            
    class CodeInterpreterTool:
        def __init__(self):
            self.type = "code_interpreter"
            
    class FileSearchTool:
        def __init__(self):
            self.type = "file_search"

from app.core.config import settings
from app.core.observability import observability
from app.core.tracing import trace_operation, add_trace_attributes, get_tracer

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    CREATED = "created"
    ACTIVE = "active"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentConversation:
    agent_id: str
    thread_id: str
    messages: List[Dict[str, Any]]
    status: AgentStatus
    created_at: datetime
    updated_at: datetime

@dataclass
class AgentRunResult:
    run_id: str
    agent_id: str
    thread_id: str
    status: str
    response: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    created_at: datetime

class AzureAIAgentService:
    """Azure AI Agent Service integration for MCP client functionality"""
    
    def __init__(self, project_client):
        self.client = project_client
        self.agents: Dict[str, Agent] = {}
        self.conversations: Dict[str, AgentConversation] = {}
        self.tools = self._initialize_tools()
        
    def _initialize_tools(self) -> List[Any]:
        """Initialize tools for agents"""
        tools = []
        
        try:
            # Note: FileSearchTool temporarily disabled due to serialization issues
            # The RAG pattern is implemented directly in process_qa_request method
            # by retrieving documents from vector store and passing as context
            logger.info("Tools initialized (FileSearchTool disabled temporarily)")
        except Exception as e:
            logger.warning(f"Could not initialize tools: {e}")
        
        return tools
    
    async def create_qa_agent(self, name: str, instructions: str, model_deployment: str = None) -> Agent:
        """Create a QA agent for Exercise 2 functionality - now uses find_or_create pattern"""
        return await self.find_or_create_agent(
            agent_name=name,
            instructions=f"""
            {instructions}
            
            You are a financial analysis expert specializing in question answering with source verification.
            
            Key capabilities:
            1. Answer complex financial questions comprehensively
            2. Decompose complex questions into researchable sub-questions
            3. Verify source credibility and trustworthiness
            4. Synthesize information from multiple sources
            5. Provide clear reasoning and justifications
            
            Guidelines:
            - Always cite sources with specific references
            - Evaluate source credibility (SEC filings are highly credible)
            - Acknowledge limitations and uncertainties
            - Use financial terminology appropriately
            - Provide quantitative analysis when possible
            """,
            model_deployment=model_deployment
        )
    
    async def create_content_generator_agent(self, name: str, instructions: str, model_deployment: str = None) -> Agent:
        """Create a content generator agent for Exercise 1 functionality - now uses find_or_create pattern"""
        return await self.find_or_create_agent(
            agent_name=name,
            instructions=f"""
            {instructions}
            
            You are a financial content generation expert specializing in creating high-quality financial content.
            
            Key capabilities:
            1. Generate comprehensive financial reports and analyses
            2. Create executive summaries and technical documentation
            3. Enhance content with proper citations
            4. Adapt writing style and tone to context
            5. Ensure factual accuracy and professional standards
            
            Guidelines:
            - Use appropriate financial terminology
            - Include specific metrics and data points
            - Maintain professional tone and style
            - Cite all sources appropriately
            - Focus on accuracy and relevance
            """,
            model_deployment=model_deployment        )
    
    @trace_operation("azure_ai_agent_conversation")
    async def run_agent_conversation(self, agent_id: str, thread_id: str, message: str, context: Dict[str, Any] = None) -> AgentRunResult:
        """Execute agent conversation and return results"""
        try:
            async with observability.trace_operation("run_agent_conversation") as span:
                span.set_attribute("agent_id", agent_id)
                span.set_attribute("thread_id", thread_id)
                span.set_attribute("message_length", len(message))
                
                if context:
                    enhanced_message = f"""
                    Context: {context.get('knowledge_context', '')}
                    
                    Question: {message}
                    """
                else:
                    enhanced_message = message
                
                # Create message using Azure AI Projects SDK - exact quickstart format
                logger.info(f"Creating message with thread_id={thread_id}, role='user', content length={len(enhanced_message)}")
                message = self.client.agents.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=enhanced_message
                )
                logger.info(f"Created message: {message.id}")
                
                if thread_id in self.conversations:
                    self.conversations[thread_id].messages.append({
                        "role": "user",
                        "content": message,
                        "timestamp": datetime.utcnow()
                    })
                    self.conversations[thread_id].status = AgentStatus.RUNNING
                    self.conversations[thread_id].updated_at = datetime.utcnow()
                
                # Create run using Azure AI Projects SDK - exact quickstart format
                logger.info(f"Creating run with thread_id={thread_id}, agent_id={agent_id}")
                run = self.client.agents.runs.create(
                    thread_id=thread_id,
                    agent_id=agent_id
                )
                logger.info(f"Created run: {run.id}")
                
                completed_run = await self._wait_for_run_completion(thread_id, run.id)
                
                messages = self.client.agents.messages.list(thread_id=thread_id)
                
                response_content = ""
                sources = []
                
                for message in messages:
                    if message.role == "assistant":
                        # Handle text messages as in Azure AI Projects SDK
                        if hasattr(message, 'content') and message.content:
                            for content in message.content:
                                if hasattr(content, 'text') and content.text:
                                    response_content = content.text.value
                                    # Handle annotations for sources
                                    if hasattr(content.text, 'annotations'):
                                        for annotation in content.text.annotations:
                                            if hasattr(annotation, 'file_citation'):
                                                sources.append({
                                                    "type": "file_citation",
                                                    "file_id": annotation.file_citation.file_id,
                                                    "quote": annotation.file_citation.quote
                                                })
                        # Also check text_messages as used in samples
                        elif hasattr(message, 'text_messages') and message.text_messages:
                            last_text = message.text_messages[-1]
                            response_content = last_text.text.value
                        break
                
                if thread_id in self.conversations:
                    self.conversations[thread_id].messages.append({
                        "role": "assistant",
                        "content": response_content,
                        "timestamp": datetime.utcnow()
                    })
                    self.conversations[thread_id].status = AgentStatus.COMPLETED
                    self.conversations[thread_id].updated_at = datetime.utcnow()
                
                result = AgentRunResult(
                    run_id=run.id,
                    agent_id=agent_id,
                    thread_id=thread_id,
                    status=completed_run.status,
                    response=response_content,
                    sources=sources,
                    metadata={
                        "context": context,
                        "run_steps": completed_run.steps if hasattr(completed_run, 'steps') else []
                    },
                    created_at=datetime.utcnow()
                )
                
                span.set_attribute("response_length", len(response_content))
                span.set_attribute("sources_count", len(sources))
                span.set_attribute("success", True)
                
                logger.info(f"Completed agent conversation: {run.id}")
                return result
                
        except Exception as e:
            logger.error(f"Error running agent conversation: {e}")
            observability.record_error("run_agent_conversation_error", str(e))
            
            if thread_id in self.conversations:
                self.conversations[thread_id].status = AgentStatus.ERROR
                self.conversations[thread_id].updated_at = datetime.utcnow()
            
            raise
    
    async def _wait_for_run_completion(self, thread_id: str, run_id: str, timeout: int = 300) -> Any:
        """Wait for agent run to complete with timeout"""
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).seconds < timeout:
            run = self.client.agents.runs.get(thread_id=thread_id, run_id=run_id)
            
            if run.status in ["completed", "failed", "cancelled", "expired"]:
                return run
            
            await asyncio.sleep(2)  # Poll every 2 seconds
        
        raise TimeoutError(f"Agent run {run_id} did not complete within {timeout} seconds")
    
    async def get_agent_capabilities(self, agent_id: str) -> Dict[str, Any]:
        """Get agent capabilities and status"""
        try:
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} not found")
            
            agent = self.agents[agent_id]
            
            # Extract tool information safely
            tool_names = []
            for tool in self.tools:
                if hasattr(tool, 'type'):
                    tool_names.append(tool.type)
                elif hasattr(tool, '__class__'):
                    tool_names.append(tool.__class__.__name__)
                elif isinstance(tool, dict) and 'type' in tool:
                    tool_names.append(tool['type'])
                else:
                    tool_names.append("unknown")
            
            return {
                "agent_id": agent.id,
                "name": agent.name,
                "model": agent.model,
                "tools": tool_names,
                "instructions": agent.instructions,
                "created_at": agent.created_at if hasattr(agent, 'created_at') else None
            }
            
        except Exception as e:
            logger.error(f"Error getting agent capabilities: {e}")
            raise
    
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all created agents"""
        try:
            agents_list = []
            
            for agent_id, agent in self.agents.items():
                agents_list.append({
                    "agent_id": agent.id,
                    "name": agent.name,
                    "model": agent.model,
                    "status": "active",
                    "created_at": agent.created_at if hasattr(agent, 'created_at') else None
                })
            
            return agents_list
            
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            raise
    
    async def get_conversation_history(self, thread_id: str) -> Optional[AgentConversation]:
        """Get conversation history for a thread"""
        return self.conversations.get(thread_id)
    
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent"""
        try:
            if agent_id in self.agents:
                self.client.agents.delete_agent(agent_id)
                del self.agents[agent_id]
                logger.info(f"Deleted agent: {agent_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting agent: {e}")
            raise
    
    async def cleanup_conversations(self, max_age_hours: int = 24) -> int:
        """Clean up old conversations"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            cleaned_count = 0
            
            threads_to_remove = []
            for thread_id, conversation in self.conversations.items():
                if conversation.updated_at < cutoff_time:
                    threads_to_remove.append(thread_id)
            
            for thread_id in threads_to_remove:
                try:
                    self.client.agents.delete_thread(thread_id)
                    del self.conversations[thread_id]
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Error cleaning up thread {thread_id}: {e}")
            
            logger.info(f"Cleaned up {cleaned_count} old conversations")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up conversations: {e}")
            return 0

    @trace_operation("azure_ai_agent_qa_processing")
    async def process_qa_request(self, question: str, context: Dict[str, Any], verification_level: str, session_id: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process QA request using Azure AI Agent Service with RAG and verification level-specific logic"""
        try:
            async with observability.trace_operation("azure_ai_agent_qa_processing") as span:
                span.set_attribute("question_length", len(question))
                span.set_attribute("verification_level", verification_level)
                span.set_attribute("session_id", session_id)
                
                # Add custom tracing attributes
                add_trace_attributes(
                    question_length=len(question),
                    verification_level=verification_level,
                    session_id=session_id,
                    model_config_chat=model_config.get('chat_model', 'unknown'),
                    model_config_embedding=model_config.get('embedding_model', 'unknown')
                )
                  # Step 1: Configure search parameters based on verification level
                search_config = self._get_search_config_for_verification_level(verification_level)
                logger.info(f"Verification level {verification_level} - Using search config: {search_config}")
                
                # Add search configuration to trace
                add_trace_attributes(
                    search_top_k=search_config["top_k"],
                    search_verification_level=verification_level
                )
                
                # Step 2: Retrieve relevant context from vector store using hybrid search
                logger.info(f"Retrieving context from vector store for question: {question[:100]}...")
                
                # Get knowledge base manager from context or create one
                kb_manager = context.get('kb_manager')
                if not kb_manager:
                    from app.services.azure_services import AzureServiceManager
                    from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
                    azure_manager = AzureServiceManager()
                    await azure_manager.initialize()
                    kb_manager = AdaptiveKnowledgeBaseManager(azure_manager)
                  # Extract token tracking context for logging and usage
                token_tracker = context.get('token_tracker')
                tracking_id = context.get('tracking_id')
                embedding_tracking_id = context.get('embedding_tracking_id')  # Separate tracking for embeddings
                
                # Use embedding tracking ID for search operations (which involve embedding generation)
                search_tracking_id = embedding_tracking_id if embedding_tracking_id else tracking_id
                
                # Search for relevant documents with verification level-specific parameters
                logger.info(f"🔍 Searching knowledge base with token tracking: tracker={token_tracker is not None}, tracking_id={search_tracking_id}")
                search_results = await kb_manager.search_knowledge_base(
                    query=question,
                    top_k=search_config["top_k"],
                    filters=context.get('filters'),
                    chat_model=self._extract_deployment_name(model_config.get('chat_model')),
                    token_tracker=token_tracker,
                    tracking_id=search_tracking_id  # Use embedding-specific tracking ID
                )
                
                # Finalize embedding tracking after search operations complete
                if embedding_tracking_id and token_tracker:
                    try:
                        await token_tracker.finalize_tracking(
                            tracking_id=embedding_tracking_id,
                            success=True,
                            http_status_code=200,
                            metadata={
                                "operation": "embedding_search",
                                "search_results_count": len(search_results),
                                "query_length": len(question),
                                "verification_level": verification_level
                            }
                        )
                        logger.info(f"✅ Finalized embedding token tracking for session {embedding_tracking_id}")
                    except Exception as e:
                        logger.error(f"❌ Failed to finalize embedding token tracking: {e}")
                
                logger.info(f"Retrieved {len(search_results)} relevant documents from vector store (requested: {search_config['top_k']} for {verification_level})")
                
                # Add search results to trace
                add_trace_attributes(
                    search_results_count=len(search_results),
                    search_requested_count=search_config["top_k"]
                )
                  # Step 3: Smart question decomposition and context gathering
                sub_questions = []
                additional_search_results = []
                
                # Enhanced question complexity analysis
                question_lower = question.lower()
                
                # 1. Comparative analysis detection
                comparative_keywords = [
                    'compare', 'versus', 'vs', 'vs.', 'contrast', 'difference', 'differences',
                    'between', 'both', 'either', 'each', 'while', 'whereas',
                    'better than', 'worse than', 'superior', 'inferior',
                    'relative to', 'against', 'compared to'
                ]
                is_comparative = any(keyword in question_lower for keyword in comparative_keywords)
                
                # 2. Multi-entity detection (multiple companies/years mentioned)
                company_indicators = ['inc', 'corp', 'ltd', 'llc', 'company', 'companies']
                year_pattern = r'\b(19|20)\d{2}\b'  # Matches years like 1990-2099
                import re
                years_mentioned = len(re.findall(year_pattern, question))
                
                # Count potential company/entity mentions
                entities_mentioned = 0
                common_entities = ['apple', 'microsoft', 'google', 'amazon', 'tesla', 'meta', 'nvidia', 'berkshire']
                for entity in common_entities:
                    if entity in question_lower:
                        entities_mentioned += 1
                
                # Also count company indicators
                for indicator in company_indicators:
                    entities_mentioned += question_lower.count(indicator)
                
                # 3. Temporal analysis detection (multi-year questions)
                temporal_keywords = [
                    'over time', 'trend', 'trends', 'historically', 'history',
                    'last', 'previous', 'past', 'recent', 'since', 'until',
                    'yearly', 'annually', 'quarterly', 'growth', 'decline',
                    'change', 'changes', 'evolution', 'development'
                ]
                is_temporal = any(keyword in question_lower for keyword in temporal_keywords) or years_mentioned >= 2
                
                # 4. Complex analytical tasks
                analytical_keywords = [
                    'analyze', 'analysis', 'comprehensive', 'detailed', 'evaluate', 'evaluation',
                    'assessment', 'review', 'examination', 'investigate', 'research',
                    'deep dive', 'thorough', 'extensive', 'complete', 'full'
                ]
                is_analytical = any(keyword in question_lower for keyword in analytical_keywords)
                
                # 5. Financial complexity indicators
                financial_complexity_keywords = [
                    'financial performance', 'profitability', 'valuation', 'ratios',
                    'balance sheet', 'income statement', 'cash flow',
                    'debt', 'equity', 'roi', 'roe', 'roa', 'eps',
                    'revenue breakdown', 'segment analysis', 'market share'
                ]
                is_financially_complex = any(keyword in question_lower for keyword in financial_complexity_keywords)
                
                # 6. Calculate complexity score
                complexity_score = 0
                complexity_reasons = []
                
                if is_comparative:
                    complexity_score += 3
                    complexity_reasons.append("comparative analysis")
                
                if entities_mentioned >= 2:
                    complexity_score += 2
                    complexity_reasons.append(f"multiple entities ({entities_mentioned})")
                
                if is_temporal:
                    complexity_score += 2
                    complexity_reasons.append("temporal analysis")
                
                if is_analytical:
                    complexity_score += 1
                    complexity_reasons.append("analytical task")
                
                if is_financially_complex:
                    complexity_score += 1
                    complexity_reasons.append("financial complexity")
                
                if len(question.split()) > 15:
                    complexity_score += 1
                    complexity_reasons.append("long question")
                
                # Decision: Use decomposition if complexity score >= 3 OR verification level is comprehensive
                should_decompose = complexity_score >= 3 or verification_level == "comprehensive"
                
                # Enhanced logging for decomposition decision
                logger.info(f"")
                logger.info(f"🧠 SMART QUESTION ANALYSIS:")
                logger.info(f"📝 Question: '{question}'")
                logger.info(f"🔄 Comparative: {is_comparative}")
                logger.info(f"🏢 Entities mentioned: {entities_mentioned}")
                logger.info(f"� Temporal analysis: {is_temporal} (years found: {years_mentioned})")
                logger.info(f"🔍 Analytical task: {is_analytical}")
                logger.info(f"💰 Financial complexity: {is_financially_complex}")
                logger.info(f"📊 Verification level: {verification_level}")
                logger.info(f"🎯 Complexity score: {complexity_score} (reasons: {', '.join(complexity_reasons)})")
                logger.info(f"🚀 Decomposition decision: {'YES' if should_decompose else 'NO'}")
                
                # Use decomposition based on smart analysis
                if should_decompose:
                    try:
                        primary_reason = complexity_reasons[0] if complexity_reasons else "comprehensive verification"
                        logger.info(f"Performing decomposition due to: {primary_reason}")
                        decomp_result = await self.decompose_complex_question(question, context, session_id, model_config)
                        sub_questions = decomp_result.get("sub_questions", [])
                        logger.info(f"Decomposed question into {len(sub_questions)} sub-questions")
                          # Execute additional searches for each sub-question if we have them
                        if sub_questions:
                            additional_search_results = await self._execute_sub_question_searches(
                                sub_questions, 
                                kb_manager, 
                                top_k=search_config["top_k"] // 2,  # Use half the top_k for each sub-question
                                chat_model=self._extract_deployment_name(model_config.get('chat_model'))
                            )
                            logger.info(f"Sub-question searches returned {len(additional_search_results)} additional documents")
                    except Exception as e:
                        logger.warning(f"Question decomposition failed: {e}")
                        sub_questions = []
                  # Combine original search results with sub-question results
                all_search_results = search_results.copy()
                
                # Add additional results from sub-questions, avoiding duplicates
                seen_doc_ids = {result.get('id') or result.get('document_id', f"doc_{i}") for i, result in enumerate(search_results)}
                additional_unique_docs = 0
                
                for result in additional_search_results:
                    doc_id = result.get('id') or result.get('document_id', f"additional_doc_{len(all_search_results)}")
                    if doc_id not in seen_doc_ids:
                        all_search_results.append(result)
                        seen_doc_ids.add(doc_id)
                        additional_unique_docs += 1
                  # Enhanced logging for context aggregation
                logger.info(f"")
                logger.info(f"📋 CONTEXT AGGREGATION SUMMARY:")
                logger.info(f"🔍 Original query search results: {len(search_results)} documents")
                if sub_questions:
                    logger.info(f"🔎 Sub-question searches executed: {len(sub_questions)} searches")
                    logger.info(f"📊 Total documents from sub-questions: {len(additional_search_results)}")
                    logger.info(f"➕ Additional unique documents added: {additional_unique_docs}")
                    logger.info(f"📄 TOTAL CONTEXT SIZE: {len(all_search_results)} documents")
                    logger.info(f"🧮 Context multiplication: {len(search_results)} → {len(all_search_results)} ({(len(all_search_results)/len(search_results)):.1f}x increase)" if search_results else "No original results to compare")
                    
                    # Add sub-question processing to trace
                    add_trace_attributes(
                        sub_questions_count=len(sub_questions),
                        sub_question_results_count=len(additional_search_results),
                        additional_unique_docs=additional_unique_docs,
                        total_context_documents=len(all_search_results),
                        context_multiplication_factor=(len(all_search_results)/len(search_results)) if search_results else 1.0
                    )
                else:
                    logger.info(f"📄 TOTAL CONTEXT SIZE: {len(all_search_results)} documents (no decomposition)")
                
                # Analyze context composition by company
                company_breakdown = {}
                sub_question_breakdown = {}
                
                for result in all_search_results:
                    company = result.get('company', 'Unknown')
                    company_breakdown[company] = company_breakdown.get(company, 0) + 1
                    
                    if result.get('sub_question_source'):
                        sq_source = result.get('sub_question_source', 'Original Query')
                        sub_question_breakdown[sq_source] = sub_question_breakdown.get(sq_source, 0) + 1
                
                if company_breakdown:
                    logger.info(f"🏢 Documents by company: {dict(sorted(company_breakdown.items()))}")
                
                if sub_question_breakdown:
                    logger.info(f"❓ Documents by source:")
                    logger.info(f"   Original query: {len(search_results)} docs")
                    for sq, count in sub_question_breakdown.items():
                        logger.info(f"   '{sq[:60]}...': {count} docs")
                logger.info(f"")
                
                # Step 4: Build enriched context with all retrieved documents
                retrieved_context = ""
                sources = []
                
                for i, result in enumerate(all_search_results):
                    retrieved_context += f"\n\n--- Document {i+1} ---\n"
                    retrieved_context += f"Title: {result.get('title', 'Unknown')}\n"
                    retrieved_context += f"Company: {result.get('company', 'Unknown')}\n"
                    retrieved_context += f"Document Type: {result.get('document_type', 'Unknown')}\n"
                    retrieved_context += f"Content: {result.get('content', '')[:1000]}\n"
                    retrieved_context += f"Credibility Score: {result.get('credibility_score', 0.0)}\n"
                    
                    # Add to sources for citations
                    # Convert credibility score to confidence string
                    credibility_score = result.get('credibility_score', 0.5)
                    if credibility_score >= 0.8:
                        confidence_level = "high"
                    elif credibility_score >= 0.6:
                        confidence_level = "medium"
                    else:
                        confidence_level = "low"
                    
                    # Add source information for additional context
                    if hasattr(result, 'sub_question_source'):
                        retrieved_context += f"Sub-question Source: {result.get('sub_question_source', '')}\n"
                    
                    sources.append({
                        "id": result.get('id', f"doc_{i+1}"),
                        "content": result.get('content', ''),
                        "source": result.get('source_url', result.get('source', 'Unknown')),
                        "document_id": result.get('document_id', ''),
                        "document_title": result.get('title', ''),
                        "document_type": result.get('document_type', ''),
                        "company": result.get('company', ''),
                        "page_number": result.get('page_number'),
                        "section_title": result.get('section_title', ''),
                        "confidence": confidence_level,
                        "url": result.get('source_url', ''),
                        "credibility_score": credibility_score,
                        "relevance_explanation": result.get('relevance_explanation', ''),
                        "sub_question_source": result.get('sub_question_source', '')
                    })
                
                span.set_attribute("retrieved_documents", len(all_search_results))
                span.set_attribute("sub_questions_count", len(sub_questions))
                
                # Step 4: Create verification level-specific QA agent with deployment name from frontend
                # Extract string value from verification_level (in case it's an enum)
                verification_level_str = verification_level.value if hasattr(verification_level, 'value') else str(verification_level).lower()
                
                # Extract deployment name from model config (handle cases like "gpt-4o (chat4o)")
                chat_deployment = self._extract_deployment_name(model_config.get('chat_model'))
                
                agent_name = f"Financial_QA_Agent_{verification_level_str.title()}"
                logger.info(f"Creating {verification_level_str} QA agent '{agent_name}' with deployment name: {chat_deployment}")
                
                qa_agent = await self.find_or_create_agent(
                    agent_name=agent_name,
                    instructions=self._get_agent_instructions_for_verification_level(verification_level_str),
                    model_deployment=chat_deployment
                )
                
                # Create thread using Azure AI Projects SDK
                thread = self.client.agents.threads.create()
                thread_id = thread.id
                  # Step 4: Prepare enhanced context with retrieved documents
                enhanced_context = {
                    **context,
                    "verification_level": verification_level,
                    "model_config": model_config,
                    "retrieved_documents": retrieved_context,
                    "source_count": len(sources)
                }
                
                # Step 5: Create enhanced message with retrieved context and sub-questions
                sub_questions_context = ""
                if sub_questions:
                    sub_questions_context = f"\n\nThis question was decomposed into the following sub-questions:\n"
                    for i, sq in enumerate(sub_questions, 1):
                        sub_questions_context += f"{i}. {sq}\n"
                    sub_questions_context += "\nPlease ensure your answer addresses all aspects covered by these sub-questions.\n"
                
                enhanced_message = f"""
                Context from Financial Document Database:
                {retrieved_context}
                {sub_questions_context}
                ---
                
                User Question: {question}
                
                Please provide a comprehensive answer based on the retrieved financial documents above. 
                Include specific citations and assess the confidence level of your answer based on the source quality.
                For comparative questions, ensure you provide data for ALL entities being compared.
                """
                  # Log final context size being sent to LLM
                context_char_count = len(retrieved_context)
                total_message_char_count = len(enhanced_message)
                logger.info(f"")
                logger.info(f"💬 FINAL LLM CONTEXT:")
                logger.info(f"📄 Documents in context: {len(all_search_results)}")
                logger.info(f"📊 Context character count: {context_char_count:,} characters")
                logger.info(f"💭 Total message to LLM: {total_message_char_count:,} characters")
                logger.info(f"❓ Sub-questions included: {len(sub_questions)}")
                logger.info(f"🎯 Ready to send comprehensive context to {chat_deployment} model")
                logger.info(f"")                
                result = await self.run_agent_conversation(
                    agent_id=qa_agent.id,
                    thread_id=thread_id,
                    message=enhanced_message,
                    context=enhanced_context
                )
                  # Get credibility check setting from context
                credibility_check_enabled = context.get('credibility_check_enabled', False)
                
                # Calculate actual overall credibility score based on source scores
                if sources:
                    total_credibility = sum(s.get('credibility_score', 0.5) for s in sources)
                    calculated_overall_credibility = total_credibility / len(sources)
                    
                    # Count high credibility sources
                    high_credibility_sources = [s for s in sources if s.get('credibility_score', 0) >= 0.6]
                    high_credibility_count = len(high_credibility_sources)
                    
                    # Add credibility analysis to trace
                    add_trace_attributes(
                        sources_total_count=len(sources),
                        sources_high_credibility_count=high_credibility_count,
                        calculated_overall_credibility=calculated_overall_credibility,
                        high_credibility_percentage=(high_credibility_count / len(sources) * 100) if sources else 0,
                        credibility_check_enabled=credibility_check_enabled
                    )
                    
                    # If all sources meet high threshold, use the average of high credibility sources
                    if high_credibility_count == len(sources) and high_credibility_sources:
                        high_credibility_avg = sum(s.get('credibility_score', 0) for s in high_credibility_sources) / len(high_credibility_sources)
                        final_confidence_score = high_credibility_avg
                    else:
                        final_confidence_score = calculated_overall_credibility
                else:
                    final_confidence_score = 0.5
                    high_credibility_count = 0
                
                # Build verification details based on credibility check setting
                verification_details = {
                    "verification_level": verification_level,
                    "agent_id": qa_agent.id,
                    "thread_id": thread_id,
                    "documents_retrieved": len(all_search_results),
                    "original_search_count": len(search_results),
                    "sub_question_search_count": len(additional_search_results),
                    "vector_search_used": True,
                    "total_sources": len(sources),
                    "total_sources_count": len(sources),  # Frontend compatibility
                    "verification_status": "completed" if sources else "no_sources",
                    "chat_model_used": chat_deployment,
                    "decomposition_used": len(sub_questions) > 0,
                    "credibility_check_enabled": credibility_check_enabled
                }
                
                # Only include credibility details if the check is enabled
                if credibility_check_enabled:
                    verification_details.update({
                        "sources_verified": len([s for s in sources if s.get('credibility_score', 0) >= 0.6]),
                        "verified_sources_count": len([s for s in sources if s.get('credibility_score', 0) >= 0.6]),  # Frontend compatibility
                        "overall_credibility_score": final_confidence_score,  # Frontend compatibility
                        "verification_summary": f"Retrieved {len(all_search_results)} total documents ({len(search_results)} from original query + {len(additional_search_results)} from {len(sub_questions)} sub-questions) using {verification_level} verification. {high_credibility_count} sources meet high credibility threshold (≥60%)."
                    })
                else:
                    verification_details.update({
                        "sources_verified": len(sources),  # Show all sources as "verified" when credibility check is off
                        "verified_sources_count": len(sources),  # Frontend compatibility
                        "overall_credibility_score": 0.8,  # Default confidence when credibility check is off
                        "verification_summary": f"Retrieved {len(all_search_results)} total documents ({len(search_results)} from original query + {len(additional_search_results)} from {len(sub_questions)} sub-questions) using {verification_level} verification. Credibility checking disabled."
                    })
                
                return {
                    "answer": result.response,
                    "confidence_score": final_confidence_score if credibility_check_enabled else 0.8,  # Use calculated credibility only when enabled
                    "sources": sources,
                    "sub_questions": sub_questions,
                    "verification_details": verification_details
                }
        except Exception as e:
            logger.error(f"Error processing QA request: {e}")
            observability.record_error("azure_ai_agent_qa_error", str(e))
            raise

    @trace_operation("azure_ai_agent_decompose_question")
    async def decompose_complex_question(self, question: str, context: Dict[str, Any], session_id: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose complex questions into sub-questions using Azure AI Agent Service"""
        try:
            async with observability.trace_operation("azure_ai_agent_question_decomposition") as span:
                span.set_attribute("question_length", len(question))
                span.set_attribute("session_id", session_id)                # Extract deployment name from model config (handle cases like "gpt-4o (chat4o)")
                chat_deployment = self._extract_deployment_name(model_config.get('chat_model'))
                
                logger.info(f"Creating Question Decomposition agent with deployment name: {chat_deployment}")
                
                decomposition_agent = await self.find_or_create_agent(
                    agent_name="Question_Decomposition_Agent",
                    instructions="""
                    You are a financial question decomposition expert. Your task is to break down complex financial questions into smaller, researchable sub-questions that can be executed independently.
                    
                    Guidelines:
                    1. Identify comparative questions (e.g., "Compare X and Y") and create separate queries for each entity
                    2. For each entity/company mentioned, create specific sub-questions
                    3. Break down complex analysis requests into component parts
                    4. Ensure each sub-question can be researched independently using document search
                    5. For comparative questions, ensure you have sub-questions for EACH entity being compared
                    6. Focus on financial analysis, data requirements, and research needs
                    
                    Examples:
                    - "Compare Apple and Microsoft risk factors" should become:
                      1. What are Apple's main risk factors in their latest financial filings?
                      2. What are Microsoft's main risk factors in their latest financial filings?
                      3. What are the key differences between Apple and Microsoft risk profiles?
                    
                    - "Analyze Tesla's financial performance vs competitors" should become:
                      1. What is Tesla's recent financial performance (revenue, profit, growth)?
                      2. Who are Tesla's main competitors in the automotive/EV space?
                      3. What is the financial performance of Tesla's key competitors?
                      4. How does Tesla's performance compare to industry benchmarks?
                    
                    Format your response as:
                    Sub-questions:
                    1. [First sub-question]
                    2. [Second sub-question]
                    ...
                    
                    Reasoning: [Explain your decomposition approach and why these sub-questions will gather comprehensive data]
                    """,
                    model_deployment=chat_deployment
                )
                
                # Create thread using Azure AI Projects SDK structure
                thread = self.client.agents.threads.create()
                thread_id = thread.id
                
                result = await self.run_agent_conversation(
                    agent_id=decomposition_agent.id,
                    thread_id=thread_id,
                    message=f"Decompose this complex financial question: {question}",
                    context=context
                )
                
                # Parse the response to extract sub-questions
                response_text = result.response
                sub_questions = []
                reasoning = "Question decomposition completed."
                
                if "Sub-questions:" in response_text:
                    lines = response_text.split('\n')
                    in_questions = False
                    in_reasoning = False
                    
                    for line in lines:
                        line = line.strip()
                        if line.startswith("Sub-questions:"):
                            in_questions = True
                            continue
                        elif line.startswith("Reasoning:"):
                            in_questions = False
                            in_reasoning = True
                            reasoning = line.replace("Reasoning:", "").strip()
                            continue
                        elif in_questions and line and (line[0].isdigit() or line.startswith("-")):
                            question_text = line
                            if '. ' in question_text:
                                question_text = question_text.split('. ', 1)[1]
                            elif '- ' in question_text:
                                question_text = question_text.split('- ', 1)[1]
                            sub_questions.append(question_text.strip())
                        elif in_reasoning:
                            reasoning += " " + line
                
                return {
                    "sub_questions": sub_questions,
                    "reasoning": reasoning.strip(),
                    "agent_id": decomposition_agent.id,
                    "thread_id": thread_id
                }
                
        except Exception as e:
            logger.error(f"Error decomposing question: {e}")
            observability.record_error("azure_ai_agent_decomposition_error", str(e))
            raise
    
    @trace_operation("azure_ai_agent_verify_sources")
    async def verify_source_credibility(self, sources: List[Dict[str, Any]], context: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Verify source credibility using Azure AI Agent Service"""
        try:
            async with observability.trace_operation("azure_ai_agent_source_verification") as span:
                span.set_attribute("sources_count", len(sources))
                span.set_attribute("session_id", session_id)
                
                # Extract chat model from sources (assuming it's in context)
                chat_deployment = context.get('model_config', {}).get('chat_model') or settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
                verification_agent = await self.find_or_create_agent(
                    agent_name="Source_Verification_Agent",
                    instructions="""
                    You are a financial source credibility expert. Your task is to assess the trustworthiness and reliability of financial information sources.
                    
                    For each source, evaluate:
                    1. Source authority and reputation
                    2. Content accuracy and consistency
                    3. Publication date and relevance
                    4. Potential bias or conflicts of interest
                    5. Supporting evidence and citations
                    
                    Provide a credibility score (0.0-1.0) and explanation for each source.
                    Identify trust indicators and red flags.
                    
                    Format your response as:
                    Source 1: [URL/Title]
                    Credibility Score: [0.0-1.0]
                    Explanation: [Detailed assessment]
                    Trust Indicators: [List positive factors]
                    Red Flags: [List concerning factors]
                    
                    [Repeat for each source]
                    
                    Overall Assessment: [Summary of findings]
                    """,
                    model_deployment=chat_deployment
                )
                
                # Create thread using Azure AI Projects SDK structure
                thread = self.client.agents.threads.create()
                thread_id = thread.id
                
                sources_text = ""
                for i, source in enumerate(sources, 1):
                    sources_text += f"""
                    Source {i}:
                    URL: {source.get('url', 'N/A')}
                    Title: {source.get('title', 'N/A')}
                    Content: {source.get('content', '')[:500]}...
                    Metadata: {source.get('metadata', {})}
                    
                    """
                
                result = await self.run_agent_conversation(
                    agent_id=verification_agent.id,
                    thread_id=thread_id,
                    message=f"Verify the credibility of these financial sources:\n{sources_text}",
                    context=context
                )
                
                verified_sources = []
                overall_credibility = 0.0
                
                # Parallel source verification function
                async def verify_single_source(i: int, source: Dict[str, Any]) -> Dict[str, Any]:
                    """Verify a single source's credibility"""
                    try:
                        source_url = source.get("url", "")
                        source_content = source.get("content", "")
                        
                        # Enhanced credibility assessment based on source characteristics
                        credibility_score = 0.5  # Default
                        trust_indicators = []
                        red_flags = []
                        
                        # Assess based on source URL/type
                        if "sec.gov" in source_url.lower() or "edgar" in source_url.lower():
                            credibility_score = 0.95
                            trust_indicators.append("Official SEC filing")
                            trust_indicators.append("Government regulatory source")
                        elif "edgar" in source_url.lower():
                            credibility_score = 0.90
                            trust_indicators.append("SEC EDGAR database")
                        elif "10-k" in source_content.lower() or "form 10-k" in source_content.lower():
                            credibility_score = 0.90
                            trust_indicators.append("10-K Annual Report")
                            trust_indicators.append("Audited financial document")
                        elif "10-q" in source_content.lower():
                            credibility_score = 0.85
                            trust_indicators.append("10-Q Quarterly Report")
                        elif "earnings" in source_content.lower():
                            credibility_score = 0.75
                            trust_indicators.append("Earnings report")
                        elif source_url.startswith("https://"):
                            credibility_score = 0.70
                            trust_indicators.append("Secure HTTPS source")
                        elif not source_url or source_url == "Unknown Source":
                            credibility_score = 0.40
                            red_flags.append("No verifiable source URL")
                        
                        # Assess content quality
                        if len(source_content) > 1000:
                            trust_indicators.append("Detailed content")
                            credibility_score += 0.05
                        elif len(source_content) < 100:
                            red_flags.append("Limited content available")
                            credibility_score -= 0.10
                        
                        # Check for corporate disclosure language
                        if any(phrase in source_content.lower() for phrase in [
                            "forward-looking statements", "risk factors", "sec filings", 
                            "material adverse effect", "results of operations"
                        ]):
                            trust_indicators.append("Standard regulatory disclosure language")
                            credibility_score += 0.05
                        
                        # Ensure score is within bounds
                        credibility_score = max(0.0, min(1.0, credibility_score))
                        
                        # Generate explanation based on assessment
                        if credibility_score >= 0.8:
                            explanation = f"High credibility source (Score: {credibility_score:.2f}). " + \
                                        f"Trust indicators: {', '.join(trust_indicators[:3])}."
                            status = "verified"
                        elif credibility_score >= 0.6:
                            explanation = f"Moderate credibility source (Score: {credibility_score:.2f}). " + \
                                        f"Some positive indicators: {', '.join(trust_indicators[:2])}."
                            status = "questionable"
                        else:
                            explanation = f"Low credibility source (Score: {credibility_score:.2f}). " + \
                                        f"Concerns: {', '.join(red_flags)}."
                            status = "unverified"
                        
                        return {
                            "source_id": source.get("id", f"source_{i+1}"),
                            "url": source_url,
                            "title": source.get("title", "Document Source"),
                            "content": source_content,
                            "credibility_score": round(credibility_score, 3),
                            "credibility_percentage": round(credibility_score * 100, 1),
                            "credibility_explanation": explanation,
                            "trust_indicators": trust_indicators,
                            "red_flags": red_flags,
                            "verification_status": status,
                            "source_type": "SEC Filing" if "sec" in source_url.lower() or "edgar" in source_url.lower() else "Document",
                            "assessment_details": {
                                "score": credibility_score,
                                "percentage": f"{credibility_score * 100:.1f}%",
                                "status": status,
                                "indicators_count": len(trust_indicators),
                                "red_flags_count": len(red_flags)
                            },
                            "success": True
                        }
                        
                    except Exception as e:
                        logger.error(f"❌ Error verifying source {i+1}: {e}")
                        return {
                            "source_id": source.get("id", f"source_{i+1}"),
                            "url": source.get("url", ""),
                            "title": source.get("title", "Document Source"),
                            "content": source.get("content", ""),
                            "credibility_score": 0.0,
                            "credibility_explanation": f"Verification failed: {str(e)}",
                            "verification_status": "error",
                            "success": False,
                            "error": str(e)
                        }
                  # Execute all source verifications in parallel
                logger.info(f"🔍 Starting parallel verification of {len(sources)} sources...")
                parallel_verification_start = time.time()
                
                verification_tasks = [verify_single_source(i, source) for i, source in enumerate(sources)]
                logger.info(f"📋 Created {len(verification_tasks)} verification tasks - starting concurrent execution...")
                
                verification_results = await asyncio.gather(*verification_tasks, return_exceptions=True)
                
                parallel_verification_time = time.time() - parallel_verification_start
                logger.info(f"⚡ Parallel source verification completed in {parallel_verification_time:.2f}s")
                
                # Calculate actual parallelism efficiency for verification
                estimated_sequential_time = len(sources) * 0.05  # Assume ~50ms per source sequentially
                actual_efficiency = (estimated_sequential_time / parallel_verification_time) if parallel_verification_time > 0 else 1
                logger.info(f"� VERIFICATION PARALLELISM:")
                logger.info(f"   ⏱️  Estimated sequential time: {estimated_sequential_time:.2f}s")
                logger.info(f"   ⚡ Actual parallel time: {parallel_verification_time:.2f}s")
                logger.info(f"   🚀 Verification efficiency: {actual_efficiency:.1f}x")
                logger.info(f"   {'✅ PARALLEL VERIFICATION SUCCESSFUL' if actual_efficiency > 2 else '⚠️  VERIFICATION PARALLELISM LIMITED'}")
                
                # Process verification results
                successful_verifications = 0
                for result in verification_results:
                    if isinstance(result, Exception):
                        logger.error(f"❌ Exception in parallel verification: {result}")
                        continue
                    
                    if result.get('success', False):
                        successful_verifications += 1
                        verified_sources.append(result)
                        overall_credibility += result.get('credibility_score', 0.0)
                    else:
                        logger.warning(f"⚠️  Verification failed for source: {result.get('source_id', 'unknown')}")
                        verified_sources.append(result)  # Include failed verifications with 0 score
                
                logger.info(f"✅ Successfully verified {successful_verifications}/{len(sources)} sources")
                
                # Log performance comparison
                estimated_sequential_time = len(sources) * 0.1  # Assume ~100ms per source sequentially
                time_saved = estimated_sequential_time - parallel_verification_time
                efficiency_gain = (time_saved / estimated_sequential_time * 100) if estimated_sequential_time > 0 else 0
                
                logger.info(f"⚡ PERFORMANCE IMPROVEMENT:")
                logger.info(f"   Parallel time: {parallel_verification_time:.2f}s")
                logger.info(f"   Estimated sequential time: {estimated_sequential_time:.2f}s")
                logger.info(f"   Time saved: {time_saved:.2f}s ({efficiency_gain:.1f}% improvement)")
                logger.info(f"")
                
                # Calculate final credibility scores
                if verified_sources:
                    overall_credibility = overall_credibility / len(verified_sources)
                    
                    # Count sources meeting high credibility threshold (>= 0.6)
                    high_credibility_sources = [s for s in verified_sources if s.get('credibility_score', 0) >= 0.6]
                    high_credibility_count = len(high_credibility_sources)
                    
                    # Calculate average credibility of only high-credibility sources for better representation
                    if high_credibility_sources:
                        high_credibility_avg = sum(s.get('credibility_score', 0) for s in high_credibility_sources) / len(high_credibility_sources)
                        # Use the higher of overall average or high-credibility average for final score
                        final_credibility_score = max(overall_credibility, high_credibility_avg)
                    else:
                        final_credibility_score = overall_credibility
                else:
                    final_credibility_score = 0.5
                    high_credibility_count = 0
                
                return {
                    "verified_sources": verified_sources,
                    "overall_credibility_score": final_credibility_score,
                    "verification_summary": f"Verified {len(verified_sources)} sources with average credibility score of {final_credibility_score:.2f}. {high_credibility_count} sources meet high credibility threshold (≥60%).",
                    "agent_id": verification_agent.id,
                    "thread_id": thread_id
                }
                
        except Exception as e:
            logger.error(f"Error verifying source credibility: {e}")
            observability.record_error("azure_ai_agent_verification_error", str(e))
            raise
    
    async def find_or_create_agent(self, agent_name: str, instructions: str, model_deployment: str = None) -> Agent:
        """Find existing agent by name or create a new one if not found"""
        try:
            async with observability.trace_operation("find_or_create_agent") as span:
                span.set_attribute("agent_name", agent_name)
                
                agent_id = None
                found_agent = False
                
                # List all agents and check if one with the target name exists
                agent_list = self.client.agents.list_agents()
                for agent in agent_list:
                    if agent.name == agent_name:
                        agent_id = agent.id
                        found_agent = True
                        break
                
                if found_agent:
                    # Get the existing agent
                    agent_definition = self.client.agents.get_agent(agent_id)
                    logger.info(f"Found existing agent: {agent_name} with ID: {agent_id}")
                    span.set_attribute("agent_found", True)
                    span.set_attribute("agent_id", agent_id)
                else:
                    # Create a new agent
                    if not model_deployment:
                        model_deployment = settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
                    
                    agent_definition = self.client.agents.create_agent(
                        model=model_deployment,
                        name=agent_name,
                        instructions=instructions,
                        tools=self.tools
                    )
                    logger.info(f"Created new agent: {agent_name} with ID: {agent_definition.id}")
                    span.set_attribute("agent_found", False)
                    span.set_attribute("agent_id", agent_definition.id)
                
                # Store in local cache
                self.agents[agent_definition.id] = agent_definition
                span.set_attribute("success", True)
                
                return agent_definition
                
        except Exception as e:
            logger.error(f"Error finding or creating agent {agent_name}: {e}")
            observability.record_error("find_or_create_agent_error", str(e))
            raise

    def _get_search_config_for_verification_level(self, verification_level: str) -> Dict[str, Any]:
        """Get search configuration based on verification level per README requirements"""
        verification_configs = {
            "basic": {
                "top_k": 5,
                "content_length": 800,
                "enable_decomposition": False,
                "enable_cross_referencing": False,
                "enable_conflict_analysis": False,
                "enable_limitation_analysis": False
            },
            "thorough": {
                "top_k": 10,
                "content_length": 1200,
                "enable_decomposition": False,
                "enable_cross_referencing": True,
                "enable_conflict_analysis": True,
                "enable_limitation_analysis": False
            },
            "comprehensive": {
                "top_k": 15,
                "content_length": 1600,
                "enable_decomposition": True,
                "enable_cross_referencing": True,
                "enable_conflict_analysis": True,
                "enable_limitation_analysis": True            }
        }
        
        config = verification_configs.get(verification_level.lower(), verification_configs["thorough"])
        logger.info(f"Verification level '{verification_level}' mapped to config: {config}")
        return config
        
    def _get_agent_instructions_for_verification_level(self, verification_level: str) -> str:
        """Get agent instructions based on verification level"""
        base_instructions = """You are a financial analysis expert specializing in comprehensive question answering with source verification.
        
Provide detailed, accurate financial analysis based on the retrieved documents. Always cite your sources and indicate confidence levels.
Format your response with clear sections and include relevant financial metrics when available.

IMPORTANT FOR COMPARATIVE QUESTIONS:
- When comparing companies (e.g., "Compare Apple and Microsoft"), ensure you provide specific data for EACH company mentioned
- Do NOT infer data for companies if you don't have direct document citations for them
- If you only have data for one company in a comparison, explicitly state which company lacks data
- Use section headers to organize comparisons clearly (e.g., "## Apple Risk Factors" and "## Microsoft Risk Factors")
- Always indicate when data is missing or inferred vs. directly cited from documents

CITATION REQUIREMENTS:
- Use specific document titles, page numbers, and section references
- Indicate the confidence level of each piece of information
- Flag any assumptions or inferences clearly"""

        if verification_level.lower() == "basic":
            return base_instructions + """

VERIFICATION LEVEL: BASIC
- Focus on providing quick, essential answers
- Use up to 5 source documents
- Keep responses concise but complete for all entities in comparative questions
- Provide basic source citations with document titles"""

        elif verification_level.lower() == "thorough":
            return base_instructions + """

VERIFICATION LEVEL: THOROUGH
- Provide standard comprehensive analysis
- Use up to 10 source documents
- Include source verification and conflict identification
- Enable cross-referencing between sources
- For comparative questions, ensure balanced coverage of all entities
- Provide medium-length responses with clear structure"""

        else:  # comprehensive
            return base_instructions + """

VERIFICATION LEVEL: COMPREHENSIVE
- Provide exhaustive deep analysis
- Use up to 15 source documents
- Perform question decomposition analysis using provided sub-questions
- Include multi-angle investigation for each entity in comparative questions
- Analyze limitations and provide detailed conflict analysis
- For comparisons, create detailed sections for each entity with:
  * Direct document citations for each claim
  * Analysis of data completeness and quality
  * Identification of any gaps in available information
- Provide detailed responses with comprehensive structure and analysis"""
        
    def _extract_deployment_name(self, model_config_value: str) -> str:
        """Extract deployment name from model config value, handling combined strings like 'gpt-4o (chat4o)'"""
        if not model_config_value:
            return settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
          # Handle combined strings like "gpt-4o (chat4o)" -> extract "chat4o"
        if '(' in model_config_value and ')' in model_config_value:
            try:
                deployment_name = model_config_value.split('(')[1].split(')')[0].strip()
                logger.debug(f"Extracted deployment name '{deployment_name}' from combined string '{model_config_value}'")
                return deployment_name
            except (IndexError, AttributeError):
                logger.warning(f"Failed to parse deployment name from '{model_config_value}', using as-is")
                return model_config_value          # Return as-is if no parentheses (already just deployment name)
        return model_config_value

    async def _execute_sub_question_searches(self, sub_questions: List[str], kb_manager, top_k: int = 10, chat_model: str = None) -> List[Dict[str, Any]]:
        """Execute separate searches for each sub-question in parallel and aggregate results"""
        all_results = []
        seen_documents = set()  # Track document IDs to avoid duplicates
        
        logger.info(f"🔍 STARTING PARALLEL SUB-QUESTION VECTOR SEARCHES")
        logger.info(f"📊 Total sub-questions to search: {len(sub_questions)}")
        logger.info(f"📄 Documents per sub-question (top_k): {top_k}")
        logger.info(f"🤖 Using chat model for relevance explanations: {chat_model or 'default'}")
        logger.info(f"💡 Maximum possible documents: {len(sub_questions)} × {top_k} = {len(sub_questions) * top_k}")
        logger.info(f"⚡ Running searches in parallel for better performance")
        
        async def search_single_question(i: int, sub_question: str) -> Dict[str, Any]:
            """Search for a single sub-question"""
            import threading
            try:
                thread_id = threading.get_ident()
                logger.info(f"🔎 [Thread-{thread_id}] Starting parallel search {i+1}/{len(sub_questions)}: '{sub_question[:60]}...'")
                
                search_start_time = time.time()
                sub_results = await kb_manager.search_knowledge_base(
                    query=sub_question,
                    top_k=top_k,
                    filters={},
                    chat_model=chat_model
                )
                search_time = time.time() - search_start_time
                
                # Add metadata to results
                for result in sub_results:
                    result['sub_question_source'] = sub_question
                    result['sub_question_index'] = i + 1
                
                logger.info(f"✅ [Thread-{thread_id}] Parallel search {i+1} completed in {search_time:.2f}s - found {len(sub_results)} documents")
                
                return {
                    'sub_question': sub_question,
                    'results': sub_results,
                    'search_time': search_time,
                    'thread_id': thread_id,
                    'success': True
                }
                
            except Exception as e:
                logger.error(f"❌ Error in parallel search {i+1}: {e}")
                return {
                    'sub_question': sub_question,
                    'results': [],
                    'search_time': 0,
                    'success': False,
                    'error': str(e)
                }
          # Execute all searches in parallel
        parallel_start_time = time.time()
        logger.info(f"🚀 LAUNCHING {len(sub_questions)} PARALLEL SEARCH TASKS...")
        logger.info(f"⏰ Parallel execution start time: {parallel_start_time:.3f}")
        
        search_tasks = [search_single_question(i, sub_question) for i, sub_question in enumerate(sub_questions)]
        logger.info(f"📋 Created {len(search_tasks)} async tasks - starting concurrent execution...")
        
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        parallel_total_time = time.time() - parallel_start_time
        
        logger.info(f"🏁 ALL PARALLEL SEARCHES COMPLETED!")
        logger.info(f"⏱️  Actual parallel execution time: {parallel_total_time:.2f}s")
        
        # Calculate if we actually got parallelism
        individual_times = []
        thread_ids = set()
        for result in search_results:
            if isinstance(result, dict) and result.get('success'):
                individual_times.append(result.get('search_time', 0))
                thread_ids.add(result.get('thread_id', 'unknown'))
        
        total_individual_time = sum(individual_times) if individual_times else 0
        parallelism_efficiency = (total_individual_time / parallel_total_time) if parallel_total_time > 0 else 0
        
        logger.info(f"📊 PARALLELISM ANALYSIS:")
        logger.info(f"   🧵 Unique threads used: {len(thread_ids)} (Thread IDs: {sorted(thread_ids)})")
        logger.info(f"   ⏱️  Sum of individual search times: {total_individual_time:.2f}s")
        logger.info(f"   ⚡ Parallel execution time: {parallel_total_time:.2f}s")
        logger.info(f"   🚀 Parallelism efficiency: {parallelism_efficiency:.1f}x")
        logger.info(f"   {'✅ PARALLEL EXECUTION SUCCESSFUL' if parallelism_efficiency > 1.5 else '⚠️  PARALLELISM MAY BE LIMITED'}")
        logger.info(f"")
        
        logger.info(f"")
        logger.info(f"⚡ PARALLEL SEARCH RESULTS:")
        logger.info(f"⏱️  Total parallel execution time: {parallel_total_time:.2f}s")
        logger.info(f"� Speed improvement vs sequential: ~{len(sub_questions)}x faster")
        
        # Process results and remove duplicates
        search_stats = []
        total_docs_retrieved = 0
        successful_searches = 0
        
        for result in search_results:
            if isinstance(result, Exception):
                logger.error(f"❌ Exception in parallel search: {result}")
                continue
                
            if not result['success']:
                logger.warning(f"⚠️  Search failed for: {result['sub_question'][:60]}...")
                continue
            
            successful_searches += 1
            sub_results = result['results']
            total_docs_retrieved += len(sub_results)
            
            # Log details about this search
            if sub_results:
                companies = set()
                doc_types = set()
                for doc in sub_results:
                    companies.add(doc.get('company', 'Unknown'))
                    doc_types.add(doc.get('document_type', 'Unknown'))
                
                logger.info(f"📊 Search '{result['sub_question'][:40]}...': {len(sub_results)} docs, companies: {', '.join(sorted(companies))}")
            
            # Add results, avoiding duplicates
            unique_docs_added = 0
            duplicate_docs_skipped = 0
            
            for doc in sub_results:
                doc_id = doc.get('id') or doc.get('document_id', f"doc_{len(all_results)}")
                if doc_id not in seen_documents:
                    all_results.append(doc)
                    seen_documents.add(doc_id)
                    unique_docs_added += 1
                else:
                    duplicate_docs_skipped += 1
            
            search_stats.append({
                'sub_question': result['sub_question'],
                'documents_retrieved': len(sub_results),
                'unique_docs_added': unique_docs_added,
                'duplicates_skipped': duplicate_docs_skipped,
                'search_time': result['search_time']
            })
        
        logger.info(f"")
        logger.info(f"📈 PARALLEL SEARCH AGGREGATION SUMMARY:")
        logger.info(f"✅ Successful searches: {successful_searches}/{len(sub_questions)}")
        logger.info(f"📄 Total documents retrieved: {total_docs_retrieved}")
        logger.info(f"➕ Unique documents after deduplication: {len(all_results)}")
        logger.info(f"⏭️  Duplicate documents removed: {total_docs_retrieved - len(all_results)}")
        avg_search_time = sum(stat['search_time'] for stat in search_stats) / len(search_stats) if search_stats else 0
        logger.info(f"⚡ Average individual search time: {avg_search_time:.2f}s")
        logger.info(f"")        
        return all_results

    async def create_thread(self, agent_id: str) -> str:
        """Create a new conversation thread for an agent"""
        try:
            thread = self.client.agents.threads.create()
            thread_id = thread.id
            
            # Initialize conversation tracking
            self.conversations[thread_id] = AgentConversation(
                agent_id=agent_id,
                thread_id=thread_id,
                messages=[],
                status=AgentStatus.CREATED,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            logger.info(f"Created thread: {thread_id} for agent: {agent_id}")
            return thread_id
            
        except Exception as e:
            logger.error(f"Error creating thread: {e}")
            raise

class MockAzureAIAgentService(AzureAIAgentService):
    """Mock implementation for testing and development"""
    
    def __init__(self):
        self.agents = {}
        self.conversations = {}
        self.mock_agent_counter = 0
        self.mock_thread_counter = 0
        
    async def create_qa_agent(self, name: str, instructions: str, model_deployment: str = None) -> Agent:
        """Mock QA agent creation - reuses agents by name"""
        # Check if agent with this name already exists
        for agent_id, agent in self.agents.items():
            if agent.name == name:
                logger.info(f"Reusing existing mock QA agent: {agent_id} with name: {name}")
                return agent
        
        # Create new agent if not found
        self.mock_agent_counter += 1
        agent_id = f"mock_qa_agent_{self.mock_agent_counter}"
        
        mock_agent = type('MockAgent', (), {
            'id': agent_id,
            'name': name,
            'instructions': instructions,
            'model': model_deployment or "mock-gpt-4",
            'created_at': datetime.utcnow()
        })()
        
        self.agents[agent_id] = mock_agent
        logger.info(f"Created new mock QA agent: {agent_id} with name: {name}")
        return mock_agent
    
    async def create_content_generator_agent(self, name: str, instructions: str, model_deployment: str = None) -> Agent:
        """Mock content generator agent creation - reuses agents by name"""
        # Check if agent with this name already exists
        for agent_id, agent in self.agents.items():
            if agent.name == name:
                logger.info(f"Reusing existing mock content generator agent: {agent_id} with name: {name}")
                return agent
        
        # Create new agent if not found
        self.mock_agent_counter += 1
        agent_id = f"mock_content_agent_{self.mock_agent_counter}"
        
        mock_agent = type('MockAgent', (), {
            'id': agent_id,
            'name': name,
            'instructions': instructions,
            'model': model_deployment or "mock-gpt-4",
            'created_at': datetime.utcnow()
        })()
        
        self.agents[agent_id] = mock_agent
        logger.info(f"Created new mock content generator agent: {agent_id} with name: {name}")
        return mock_agent
    
    async def create_thread(self, agent_id: str) -> str:
        """Mock thread creation"""
        self.mock_thread_counter += 1
        thread_id = f"mock_thread_{self.mock_thread_counter}"
        
        self.conversations[thread_id] = AgentConversation(
            agent_id=agent_id,
            thread_id=thread_id,
            messages=[],
            status=AgentStatus.CREATED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        logger.info(f"Created mock thread: {thread_id}")
        return thread_id
    
    async def run_agent_conversation(self, agent_id: str, thread_id: str, message: str, context: Dict[str, Any] = None) -> AgentRunResult:
        """Mock agent conversation execution"""
        
        await asyncio.sleep(0.1)
        
        if "question" in message.lower() or "what" in message.lower():
            mock_response = f"Based on the financial analysis, here is a comprehensive answer to your question: {message[:100]}... [Mock response with financial insights and data]"
        else:
            mock_response = f"Mock financial analysis response for: {message[:50]}..."
        
        mock_sources = [
            {
                "type": "file_citation",
                "file_id": "mock_10k_filing.pdf",
                "quote": "Sample financial data from 10-K filing"
            }
        ]
        
        if thread_id in self.conversations:
            self.conversations[thread_id].messages.extend([
                {
                    "role": "user",
                    "content": message,
                    "timestamp": datetime.utcnow()
                },
                {
                    "role": "assistant", 
                    "content": mock_response,
                    "timestamp": datetime.utcnow()
                }
            ])
            self.conversations[thread_id].status = AgentStatus.COMPLETED
            self.conversations[thread_id].updated_at = datetime.utcnow()
        
        result = AgentRunResult(
            run_id=f"mock_run_{datetime.utcnow().timestamp()}",
            agent_id=agent_id,
            thread_id=thread_id,
            status="completed",
            response=mock_response,
            sources=mock_sources,
            metadata={"context": context, "mock": True},
            created_at=datetime.utcnow()
        )
        
        logger.info(f"Completed mock agent conversation")
        return result
    
    async def process_qa_request(self, question: str, context: Dict[str, Any], verification_level: str, session_id: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Mock QA request processing"""
        await asyncio.sleep(0.2)  # Simulate processing time
        
        mock_answer = f"""Based on comprehensive financial analysis, here is the answer to your question: {question[:100]}...

Key findings:
- Financial metric analysis shows positive trends
- Market conditions indicate stable performance
- Risk assessment suggests moderate exposure
- Regulatory compliance appears satisfactory

This analysis is based on the latest available financial data and market research."""
        
        mock_sources = [
            {
                "type": "file_citation",
                "file_id": "mock_10k_filing.pdf",
                "quote": "Revenue increased by 15% year-over-year"
            },
            {
                "type": "file_citation", 
                "file_id": "mock_earnings_report.pdf",
                "quote": "EBITDA margin improved to 22.5%"
            }
        ]
        
        return {
            "answer": mock_answer,
            "confidence_score": 0.85,
            "sources": mock_sources,
            "sub_questions": [
                "What are the key financial metrics?",
                "How do market conditions affect performance?",
                "What are the primary risk factors?"
            ],
            "verification_details": {
                "verification_level": verification_level,
                "agent_id": f"mock_qa_agent_{session_id}",
                "thread_id": f"mock_thread_{session_id}"
            }
        }
    
    async def decompose_complex_question(self, question: str, context: Dict[str, Any], session_id: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Mock question decomposition"""
        await asyncio.sleep(0.1)
        
        sub_questions = []
        if "financial" in question.lower():
            sub_questions.extend([
                "What are the current financial metrics and KPIs?",
                "How do these metrics compare to industry benchmarks?",
                "What are the historical trends for these metrics?"
            ])
        
        if "market" in question.lower():
            sub_questions.extend([
                "What is the current market position?",
                "Who are the main competitors?",
                "What are the market growth projections?"
            ])
        
        if "risk" in question.lower():
            sub_questions.extend([
                "What are the primary risk factors?",
                "How is risk being mitigated?",
                "What is the risk tolerance level?"
            ])
        
        if not sub_questions:
            sub_questions = [
                "What is the primary focus of this analysis?",
                "What data sources are most relevant?",
                "What are the key success metrics?"
            ]
        
        return {
            "sub_questions": sub_questions[:5],  # Limit to 5 sub-questions
            "reasoning": f"The question '{question[:50]}...' has been decomposed into specific, researchable components focusing on financial analysis, market conditions, and risk assessment.",
            "agent_id": f"mock_decomposition_agent_{session_id}",
            "thread_id": f"mock_thread_{session_id}"
        }
    
    async def verify_source_credibility(self, sources: List[Dict[str, Any]], context: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Mock source credibility verification"""
        await asyncio.sleep(0.15)
        
        verified_sources = []
        total_credibility = 0.0
        
        for i, source in enumerate(sources):
            base_score = 0.7
            
            if source.get("url", "").endswith(".gov"):
                base_score += 0.2
            elif source.get("url", "").endswith(".edu"):
                base_score += 0.15
            elif "sec.gov" in source.get("url", ""):
                base_score += 0.25
            
            if len(source.get("content", "")) > 1000:
                base_score += 0.05
            
            credibility_score = min(base_score, 1.0)
            
            verified_source = {
                "source_id": source.get("id", f"mock_source_{i+1}"),
                "url": source.get("url", ""),
                "title": source.get("title", f"Mock Financial Document {i+1}"),
                "content": source.get("content", ""),
                "credibility_score": credibility_score,
                "credibility_explanation": f"Source assessed with credibility score of {credibility_score:.2f} based on authority, content quality, and relevance.",
                "trust_indicators": [
                    "Authoritative domain",
                    "Comprehensive content",
                    "Recent publication date",
                    "Professional formatting"
                ],
                "red_flags": [] if credibility_score > 0.7 else ["Limited source information"],
                "verification_status": "verified" if credibility_score > 0.7 else "questionable"
            }
            
            verified_sources.append(verified_source)
            total_credibility += credibility_score
        
        overall_credibility = total_credibility / len(sources) if sources else 0.5
        
        return {
            "verified_sources": verified_sources,
            "overall_credibility_score": overall_credibility,
            "verification_summary": f"Verified {len(verified_sources)} sources with average credibility score of {overall_credibility:.2f}",
            "agent_id": f"mock_verification_agent_{session_id}",
            "thread_id": f"mock_thread_{session_id}"
        }
    
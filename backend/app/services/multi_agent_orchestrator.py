import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import semantic_kernel as sk
from semantic_kernel.connectors.ai import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory

from app.services.azure_services import AzureServiceManager
from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
from app.services.document_processor import DocumentProcessor
from app.services.azure_ai_agent_service import AzureAIAgentService
from app.core.config import settings
from app.core.observability import observability

logger = logging.getLogger(__name__)

class AgentType(Enum):
    CONTENT_GENERATOR = "content_generator"
    QA_AGENT = "qa_agent"
    KNOWLEDGE_MANAGER = "knowledge_manager"

@dataclass
class AgentMessage:
    agent_id: str
    agent_type: AgentType
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    session_id: str

@dataclass
class AgentCapability:
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]

class FinancialAgent:
    """Base class for financial analysis agents"""
    
    def __init__(self, agent_type: AgentType, azure_manager: AzureServiceManager):
        self.agent_type = agent_type
        self.agent_id = f"{agent_type.value}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.azure_manager = azure_manager
        self.capabilities = []
        self.kernel = self._initialize_semantic_kernel()
        
    def _initialize_semantic_kernel(self) -> sk.Kernel:
        """Initialize Semantic Kernel for the agent"""
        kernel = sk.Kernel()
        
        
        return kernel
    
    async def process_request(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request using the agent's capabilities"""
        raise NotImplementedError("Subclasses must implement process_request")
    
    def get_capabilities(self) -> List[AgentCapability]:
        """Return list of agent capabilities"""
        return self.capabilities

class ContentGeneratorAgent(FinancialAgent):
    """Agent for Exercise 1: Context-Aware Content Generation"""
    
    def __init__(self, azure_manager: AzureServiceManager, kb_manager: AdaptiveKnowledgeBaseManager):
        super().__init__(AgentType.CONTENT_GENERATOR, azure_manager)
        self.kb_manager = kb_manager
        self._initialize_capabilities()
        
    def _initialize_capabilities(self):
        """Initialize content generation capabilities"""
        self.capabilities = [
            AgentCapability(
                name="generate_financial_content",
                description="Generate high-quality financial content based on prompts and knowledge base",
                input_schema={
                    "prompt": {"type": "string", "required": True},
                    "content_type": {"type": "string", "enum": ["report", "summary", "analysis"]},
                    "tone": {"type": "string", "enum": ["professional", "technical", "executive"]},
                    "max_length": {"type": "integer", "default": 2000}
                },
                output_schema={
                    "content": {"type": "string"},
                    "citations": {"type": "array"},
                    "confidence_score": {"type": "number"},
                    "sources_used": {"type": "array"}
                }
            ),
            AgentCapability(
                name="enhance_content_with_citations",
                description="Enhance existing content with proper citations and source verification",
                input_schema={
                    "content": {"type": "string", "required": True},
                    "citation_style": {"type": "string", "enum": ["apa", "mla", "chicago"], "default": "apa"}
                },
                output_schema={
                    "enhanced_content": {"type": "string"},
                    "citations": {"type": "array"},
                    "verification_status": {"type": "string"}
                }
            )
        ]
    
    async def process_request(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process content generation requests"""
        try:
            capability = request.get("capability")
            
            if capability == "generate_financial_content":
                return await self._generate_financial_content(request, context)
            elif capability == "enhance_content_with_citations":
                return await self._enhance_content_with_citations(request, context)
            else:
                raise ValueError(f"Unknown capability: {capability}")
                
        except Exception as e:
            logger.error(f"Error processing content generation request: {e}")
            return {"error": str(e), "success": False}
    
    async def _generate_financial_content(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate financial content based on prompt and knowledge base"""
        prompt = request["prompt"]
        content_type = request.get("content_type", "analysis")
        tone = request.get("tone", "professional")
        max_length = request.get("max_length", 2000)
        
        # Extract token tracking context if available
        token_tracker = context.get('token_tracker')
        tracking_id = context.get('tracking_id')
        
        relevant_chunks = await self.kb_manager.search_knowledge_base(
            query=prompt,
            top_k=10,
            filters={"document_type": ["10-K", "10-Q"]},
            token_tracker=token_tracker,
            tracking_id=tracking_id
        )
        
        knowledge_context = "\n\n".join([
            f"Source: {chunk.get('source', 'Unknown')}\n{chunk.get('content', '')}"
            for chunk in relevant_chunks
        ])
        
        system_prompt = f"""
        You are a financial content generation expert specializing in {content_type} creation.
        
        Guidelines:
        - Tone: {tone}
        - Maximum length: {max_length} words
        - Focus on accuracy and factual information
        - Include specific financial metrics and data points
        - Maintain professional financial industry standards
        - Cite sources appropriately
        
        Knowledge Base Context:
        {knowledge_context}
        
        User Request: {prompt}
        
        Generate comprehensive financial content that addresses the request while incorporating relevant information from the knowledge base.
        """
        
        response = await self.kernel.invoke_prompt(
            function_name="generate_content",
            prompt=system_prompt
        )
        
        generated_content = str(response)
        
        citations = [
            {
                "source": chunk.get('source', 'Unknown'),
                "document_type": chunk.get('document_type', 'Unknown'),
                "section": chunk.get('section_type', 'Unknown'),
                "confidence": chunk.get('score', 0.0)
            }
            for chunk in relevant_chunks
        ]
        
        return {
            "content": generated_content,
            "citations": citations,
            "confidence_score": self._calculate_content_confidence(relevant_chunks),
            "sources_used": len(relevant_chunks),
            "success": True        }
    
    async def _enhance_content_with_citations(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance content with proper citations"""
        content = request["content"]
        citation_style = request.get("citation_style", "apa")
        
        # Extract token tracking context if available
        token_tracker = context.get('token_tracker')
        tracking_id = context.get('tracking_id')
        
        relevant_chunks = await self.kb_manager.search_knowledge_base(
            query=content[:500],  # Use first 500 chars as query
            top_k=5,
            token_tracker=token_tracker,
            tracking_id=tracking_id
        )
        
        enhancement_prompt = f"""
        Enhance the following financial content by adding proper {citation_style} style citations:
        
        Original Content:
        {content}
        
        Available Sources:
        {chr(10).join([f"[{i+1}] {chunk.get('source', 'Unknown')} - {chunk.get('document_type', 'Unknown')}" for i, chunk in enumerate(relevant_chunks)])}
        
        Add inline citations where appropriate and create a reference list at the end.
        """
        
        try:
            response = f"Enhanced content with citations: {content[:100]}..."
        except Exception as e:
            logger.error(f"Error enhancing citations: {e}")
            response = content
        
        enhanced_content = str(response)
        
        citations = [
            {
                "id": i+1,
                "source": chunk.get('source', 'Unknown'),
                "document_type": chunk.get('document_type', 'Unknown'),
                "style": citation_style
            }
            for i, chunk in enumerate(relevant_chunks)
        ]
        
        return {
            "enhanced_content": enhanced_content,
            "citations": citations,
            "verification_status": "verified",
            "success": True
        }
    
    def _calculate_content_confidence(self, chunks: List[Dict]) -> float:
        """Calculate confidence score based on source quality"""
        if not chunks:
            return 0.0
        
        scores = [chunk.get('score', 0.0) for chunk in chunks]
        return sum(scores) / len(scores)

class QAAgent(FinancialAgent):
    """Agent for Exercise 2: Agentic Question Answering with Source Verification"""
    
    def __init__(self, azure_manager: AzureServiceManager, kb_manager: AdaptiveKnowledgeBaseManager):
        super().__init__(AgentType.QA_AGENT, azure_manager)
        self.kb_manager = kb_manager
        self.azure_ai_agent_service = self._initialize_azure_ai_agent_service(azure_manager)
        self.qa_agent_id = None
        self.current_thread_id = None
        self._initialize_capabilities()
        
    def _initialize_azure_ai_agent_service(self, azure_manager: AzureServiceManager) -> AzureAIAgentService:
        """Initialize Azure AI Agent Service for MCP client functionality"""
        try:
            # Use the correct initialization pattern with project client
            project_client = azure_manager.get_project_client()
            if not project_client:
                logger.error("Failed to get project client from Azure service manager")
                raise Exception("Azure AI Project client not available")
            
            return AzureAIAgentService(project_client)
            
        except Exception as e:
            logger.error(f"Error initializing Azure AI Agent Service: {e}")
            raise Exception(f"Failed to initialize Azure AI Agent Service: {e}")
    
    async def _ensure_qa_agent_initialized(self):
        """Ensure QA agent is created and ready"""
        if not self.qa_agent_id:
            try:
                agent = await self.azure_ai_agent_service.create_qa_agent(
                    name="Financial QA Agent",
                    instructions="You are a financial analysis expert specializing in comprehensive question answering with source verification.",
                    model_deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
                )
                self.qa_agent_id = agent.id
                logger.info(f"Created QA agent: {self.qa_agent_id}")
            except Exception as e:
                logger.error(f"Error creating QA agent: {e}")
                raise
        
        if not self.current_thread_id:
            try:
                self.current_thread_id = await self.azure_ai_agent_service.create_thread(self.qa_agent_id)
                logger.info(f"Created thread: {self.current_thread_id}")
            except Exception as e:
                logger.error(f"Error creating thread: {e}")
                raise
    
    def _initialize_capabilities(self):
        """Initialize Q&A capabilities"""
        self.capabilities = [
            AgentCapability(
                name="answer_financial_question",
                description="Answer complex financial questions with source verification",
                input_schema={
                    "question": {"type": "string", "required": True},
                    "context": {"type": "string"},
                    "verification_level": {"type": "string", "enum": ["basic", "thorough"], "default": "thorough"}
                },
                output_schema={
                    "answer": {"type": "string"},
                    "sources": {"type": "array"},
                    "confidence": {"type": "number"},
                    "verification_details": {"type": "object"}
                }
            ),
            AgentCapability(
                name="decompose_complex_question",
                description="Break down complex questions into researchable sub-questions",
                input_schema={
                    "question": {"type": "string", "required": True}
                },
                output_schema={
                    "sub_questions": {"type": "array"},
                    "research_strategy": {"type": "string"}
                }
            ),
            AgentCapability(
                name="verify_source_credibility",
                description="Evaluate the credibility and trustworthiness of sources",
                input_schema={
                    "sources": {"type": "array", "required": True}
                },
                output_schema={
                    "credibility_scores": {"type": "array"},
                    "verification_report": {"type": "object"}
                }
            )
        ]
    
    async def process_request(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process Q&A requests"""
        try:
            capability = request.get("capability")
            
            if capability == "answer_financial_question":
                return await self._answer_financial_question(request, context)
            elif capability == "decompose_complex_question":
                return await self._decompose_complex_question(request, context)
            elif capability == "verify_source_credibility":
                return await self._verify_source_credibility(request, context)
            else:
                raise ValueError(f"Unknown capability: {capability}")
                
        except Exception as e:
            logger.error(f"Error processing Q&A request: {e}")
            return {"error": str(e), "success": False}
    
    async def _answer_financial_question(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Answer financial questions with source verification using Azure AI Agent Service"""
        question = request["question"]
        verification_level = request.get("verification_level", "thorough")
        search_results = request.get("search_results", [])  # Check for pre-provided search results
        
        # Extract token tracking context if available
        token_tracker = context.get('token_tracker')
        tracking_id = context.get('tracking_id')
        
        try:
            # with observability.trace_operation("qa_agent_answer_question") as span:
            # span.set_attribute("question", question[:100])
            # span.set_attribute("verification_level", verification_level)
                
                await self._ensure_qa_agent_initialized()
                
                # Use pre-provided search results if available, otherwise search
                if search_results:
                    logger.info(f"🔍 Using pre-provided search results: {len(search_results)} results")
                    sorted_chunks = search_results[:10]  # Use the first 10 results
                else:
                    logger.info("🔍 No pre-provided search results, performing own search")
                    sub_questions = await self._decompose_question_internal(question)
                      # Search knowledge base for relevant information
                    all_chunks = []
                    for sub_q in sub_questions:
                        chunks = await self.kb_manager.search_knowledge_base(
                            query=sub_q,
                            top_k=5,
                            filters={"document_type": ["10-K", "10-Q", "8-K"]},
                            token_tracker=token_tracker,
                            tracking_id=tracking_id
                        )
                        all_chunks.extend(chunks)
                    
                    unique_chunks = {chunk['chunk_id']: chunk for chunk in all_chunks}.values()
                    sorted_chunks = sorted(unique_chunks, key=lambda x: x.get('score', 0), reverse=True)[:10]
                
                logger.info(f"📚 Using {len(sorted_chunks)} chunks for context")
                
                verification_details = await self._verify_sources_internal(sorted_chunks)
                
                knowledge_context = "\n\n".join([
                    f"Source: {chunk.get('source', 'Unknown')} (Credibility: {verification_details.get(chunk.get('chunk_id', 'unknown'), {}).get('score', 0.5):.2f})\n{chunk.get('content', '')}"
                    for chunk in sorted_chunks
                ])
                
                logger.info(f"📝 Knowledge context length: {len(knowledge_context)}")
                logger.info(f"📄 First chunk preview: {sorted_chunks[0]['content'][:200] if sorted_chunks else 'No chunks'}...")
                
                # Enhanced financial analysis prompt with search results context
                financial_prompt = f"""
                You are a financial analyst expert. Answer this question using the provided financial documents:
                
                Question: {question}
                
                Financial Documents Context:
                {knowledge_context}
                
                Please provide a comprehensive answer that:
                1. Directly addresses the question
                2. Uses specific data and quotes from the documents
                3. Cites the sources properly
                4. Provides confidence in your analysis
                
                Format your response as a detailed financial analysis.
                """
                
                logger.info(f"📤 Sending enhanced prompt to agent (length: {len(financial_prompt)})")
                
                try:
                    # Get response from Azure AI Agent Service
                    logger.info("🔄 Calling azure_ai_agent_service.run_agent_conversation...")
                    agent_result = await self.azure_ai_agent_service.run_agent_conversation(
                        agent_id=self.qa_agent_id,
                        thread_id=self.current_thread_id,
                        message=financial_prompt,
                        context={
                            "task": "financial_question_answering",
                            "search_results": sorted_chunks,
                            "verification_level": verification_level
                        }
                    )
                    logger.info("✅ Agent conversation completed successfully")
                    
                except Exception as agent_error:
                    logger.error(f"❌ Agent conversation failed: {agent_error}", exc_info=True)
                    answer_text = f"Agent processing error: {str(agent_error)}"
                    agent_result = type('obj', (object,), {'response': answer_text, 'sources': []})()
                
                logger.info(f"📥 Agent response type: {type(agent_result)}")
                logger.info(f"📥 Agent response attributes: {dir(agent_result) if hasattr(agent_result, '__dict__') else 'Not an object'}")
                
                if hasattr(agent_result, 'response'):
                    answer_text = agent_result.response
                    logger.info(f"📝 Agent answer length: {len(answer_text)}")
                    logger.info(f"📝 Agent answer preview: {answer_text[:300]}...")
                else:
                    logger.error(f"❌ Agent result missing 'response' attribute: {agent_result}")
                    answer_text = "Error: No response from agent"
                
                confidence = self._calculate_answer_confidence(sorted_chunks, verification_details)
                
                combined_sources = []
                for chunk in sorted_chunks:
                    credibility_score = verification_details.get(chunk.get('chunk_id', 'unknown'), {}).get('score', 0.5)
                    combined_sources.append({
                        "content": chunk.get('content', ''),
                        "source": chunk.get('source', 'Unknown'),
                        "document_type": chunk.get('document_type', 'Unknown'),
                        "document_id": chunk.get('document_id', chunk.get('chunk_id', 'unknown')),
                        "document_title": chunk.get('title', chunk.get('source', 'Unknown Document')),
                        "section": chunk.get('section_type', 'Unknown'),
                        "section_title": chunk.get('section_type', 'Unknown'),
                        "page_number": chunk.get('page_number'),
                        "url": chunk.get('document_url'),
                        "relevance_score": chunk.get('search_score', 0.0),
                        "credibility_score": credibility_score,
                        "confidence": "high" if credibility_score > 0.8 else "medium" if credibility_score > 0.5 else "low"
                    })
                
                if hasattr(agent_result, 'sources') and agent_result.sources:
                    for agent_source in agent_result.sources:
                        combined_sources.append({
                            "source": agent_source.get("file_id", "Azure AI Agent"),
                            "document_type": "Agent Citation",
                            "section": agent_source.get("quote", "")[:100],
                            "relevance_score": 1.0,
                            "credibility_score": 0.9
                        })
                
                logger.info(f"✅ Final result - Answer: {len(answer_text)} chars, Sources: {len(combined_sources)}, Confidence: {confidence}")
                
                return {
                    "answer": answer_text,
                    "sources": combined_sources,
                    "confidence": confidence,
                    "verification_details": verification_details,
                    "agent_metadata": {
                        "agent_id": self.qa_agent_id,
                        "thread_id": self.current_thread_id,
                        "run_id": getattr(agent_result, 'run_id', 'unknown')
                    },
                    "success": True
                }
                
        except Exception as e:
            logger.error(f"Error in QA agent answer generation: {e}")
            observability.record_error("qa_agent_answer_error", str(e))
            return {
                "error": str(e),
                "success": False,
                "fallback_answer": f"I encountered an error while processing your question: {question}. Please try again or rephrase your question."
            }
    
    async def _decompose_complex_question(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose complex questions into sub-questions using Azure AI Agent Service"""
        question = request["question"]
        
        try:
            # with observability.trace_operation("qa_agent_decompose_question") as span:
            # span.set_attribute("question", question[:100])
                
                await self._ensure_qa_agent_initialized()
                
                # Use Azure AI Agent Service for question decomposition
                decomposition_context = {
                    "task": "question_decomposition",
                    "instructions": "Break down this complex financial question into 3-5 specific, researchable sub-questions that can be answered with financial data and documents."
                }
                
                decomposition_message = f"""
                Please decompose this complex financial question into specific sub-questions:
                
                Question: {question}
                
                Requirements:
                - Generate 3-5 focused sub-questions
                - Each sub-question should be answerable with financial documents
                - Sub-questions should collectively address the main question
                - Focus on quantitative and qualitative aspects
                """
                
                # Run agent conversation for question decomposition
                agent_result = await self.azure_ai_agent_service.run_agent_conversation(
                    agent_id=self.qa_agent_id,
                    thread_id=self.current_thread_id,
                    message=decomposition_message,
                    context=decomposition_context
                )
                
                sub_questions_text = agent_result.response
                
                sub_questions = []
                for line in sub_questions_text.split('\n'):
                    line = line.strip()
                    if line and (line.startswith('-') or line.startswith('•') or line[0].isdigit()):
                        clean_question = line.lstrip('-•0123456789. ').strip()
                        if clean_question and '?' in clean_question:
                            sub_questions.append(clean_question)
                
                if not sub_questions:
                    sub_questions = await self._decompose_question_internal(question)
                
                strategy_message = f"""
                Given the main question: "{question}"
                And these sub-questions: {sub_questions}
                
                Describe the optimal research strategy to answer this question comprehensively, including:
                - Document types to prioritize (10-K, 10-Q, 8-K, etc.)
                - Key financial metrics to analyze
                - Verification approaches for sources
                - Analysis methodology
                """
                
                strategy_context = {
                    "task": "research_strategy",
                    "sub_questions": sub_questions
                }
                
                strategy_result = await self.azure_ai_agent_service.run_agent_conversation(
                    agent_id=self.qa_agent_id,
                    thread_id=self.current_thread_id,
                    message=strategy_message,
                    context=strategy_context
                )
                
                strategy_response = strategy_result.response
                
            # span.set_attribute("sub_questions_count", len(sub_questions))
            # span.set_attribute("strategy_length", len(strategy_response))
            # span.set_attribute("success", True)
                
                return {
                    "sub_questions": sub_questions,
                    "research_strategy": strategy_response,
                    "agent_metadata": {
                        "agent_id": self.qa_agent_id,
                        "thread_id": self.current_thread_id,
                        "decomposition_run_id": agent_result.run_id,
                        "strategy_run_id": strategy_result.run_id
                    },
                    "success": True
                }
                
        except Exception as e:
            logger.error(f"Error in QA agent question decomposition: {e}")
            observability.record_error("qa_agent_decompose_error", str(e))
            
            sub_questions = await self._decompose_question_internal(question)
            return {
                "sub_questions": sub_questions,
                "research_strategy": "Standard financial research approach with document analysis and source verification",
                "error": str(e),
                "success": False
            }
    
    async def _decompose_question_internal(self, question: str) -> List[str]:
        """Internal method to decompose questions"""
        decomposition_prompt = f"""
        Break down this complex financial question into 3-5 specific, researchable sub-questions:
        
        Question: {question}
        
        Each sub-question should:
        - Be specific and focused
        - Be answerable with financial data/documents
        - Contribute to answering the main question
        
        Return only the sub-questions, one per line, without numbering.
        """
        
        # Decompose question into sub-questions
        try:
            sub_questions = [
                f"What are the key financial metrics related to: {question}?",
                f"What are the trends and patterns in: {question}?",
                f"What are the risk factors associated with: {question}?"
            ]
            response = "\n".join(sub_questions)
        except Exception as e:
            logger.error(f"Error decomposing question: {e}")
            response = question
        
        sub_questions = [q.strip() for q in str(response).split('\n') if q.strip()]
        return sub_questions[:5]  # Limit to 5 sub-questions
    
    async def _verify_source_credibility(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify source credibility using Azure AI Agent Service"""
        sources = request["sources"]
        
        try:
            # with observability.trace_operation("qa_agent_verify_sources") as span:
            # span.set_attribute("sources_count", len(sources))
                
                await self._ensure_qa_agent_initialized()
                
                source_info = []
                for i, source in enumerate(sources):
                    source_info.append({
                        "id": source.get("chunk_id", f"source_{i}"),
                        "content": source.get("content", "")[:500],  # Limit content length
                        "metadata": source.get("metadata", {}),
                        "document_type": source.get("metadata", {}).get("document_type", "Unknown"),
                        "source_name": source.get("metadata", {}).get("source", "Unknown")
                    })
                
                verification_message = f"""
                Please evaluate the credibility and trustworthiness of these financial document sources:
                
                Sources to verify:
                {chr(10).join([f"Source {i+1}: {s['source_name']} ({s['document_type']})" for i, s in enumerate(source_info)])}
                
                For each source, consider:
                - Document type credibility (SEC filings are highly credible)
                - Source authority and reputation
                - Recency and relevance of information
                - Potential bias or conflicts of interest
                - Verification against known standards
                
                Provide a credibility assessment and reasoning for each source.
                """
                
                verification_context = {
                    "task": "source_verification",
                    "sources": source_info,
                    "verification_criteria": ["authority", "recency", "bias", "accuracy", "relevance"]
                }
                
                # Run agent conversation for source verification
                agent_result = await self.azure_ai_agent_service.run_agent_conversation(
                    agent_id=self.qa_agent_id,
                    thread_id=self.current_thread_id,
                    message=verification_message,
                    context=verification_context
                )
                
                verification_response = agent_result.response
                
                internal_verification = await self._verify_sources_internal(sources)
                
                credibility_scores = []
                for i, source in enumerate(sources):
                    source_id = source.get("chunk_id", f"source_{i}")
                    
                    internal_data = internal_verification.get(source_id, {})
                    internal_score = internal_data.get("score", 0.5)
                    internal_factors = internal_data.get("factors", [])
                    
                    credibility_scores.append({
                        "source_id": source_id,
                        "score": internal_score,
                        "factors": internal_factors + ["Azure AI Agent Analysis"],
                        "agent_analysis": verification_response[:200] + "..." if len(verification_response) > 200 else verification_response,
                        "verification_method": "hybrid_agent_internal"
                    })
                
                # Generate comprehensive verification report
                verification_report = internal_verification.copy()
                verification_report["agent_summary"] = verification_response
                verification_report["verification_method"] = "Azure AI Agent Service + Internal Analysis"
                verification_report["total_sources_analyzed"] = len(sources)
                
            # span.set_attribute("verification_completed", True)
            # span.set_attribute("agent_analysis_length", len(verification_response))
            # span.set_attribute("success", True)
                
                return {
                    "credibility_scores": credibility_scores,
                    "verification_report": verification_report,
                    "agent_metadata": {
                        "agent_id": self.qa_agent_id,
                        "thread_id": self.current_thread_id,
                        "run_id": agent_result.run_id
                    },
                    "success": True
                }
                
        except Exception as e:
            logger.error(f"Error in QA agent source verification: {e}")
            observability.record_error("qa_agent_verify_error", str(e))
            
            verification_details = await self._verify_sources_internal(sources)
            credibility_scores = [
                {
                    "source_id": source.get('chunk_id', 'unknown'),
                    "score": verification_details.get(source.get('chunk_id', 'unknown'), {}).get('score', 0.5),
                    "factors": verification_details.get(source.get('chunk_id', 'unknown'), {}).get('factors', []),
                    "verification_method": "internal_fallback"
                }
                for source in sources
            ]
            
            return {
                "credibility_scores": credibility_scores,
                "verification_report": verification_details,
                "error": str(e),
                "success": False
            }
    
    async def _verify_sources_internal(self, chunks: List[Dict]) -> Dict[str, Any]:
        """Internal method to verify source credibility"""
        verification_details = {}
        
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', 'unknown')
            
            score = 0.5  # Base score
            factors = []
            
            doc_type = chunk.get('document_type', 'unknown')
            if doc_type in ['10-K', '10-Q']:
                score += 0.3
                factors.append("SEC filing")
            elif doc_type == '8-K':
                score += 0.2
                factors.append("Current report")
            
            if chunk.get('company'):
                score += 0.1
                factors.append("Identified company")
            
            filing_date = chunk.get('filing_date')
            if filing_date:
                score += 0.1
                factors.append("Recent filing")
            
            score = min(1.0, max(0.0, score))
            
            verification_details[chunk_id] = {
                "score": score,
                "factors": factors,
                "document_type": doc_type,
                "source": chunk.get('source', 'Unknown')
            }
        
        return verification_details
    
    def _calculate_answer_confidence(self, chunks: List[Dict], verification_details: Dict) -> float:
        """Calculate confidence score for the answer"""
        if not chunks:
            return 0.0
        
        relevance_scores = [chunk.get('score', 0.0) for chunk in chunks]
        credibility_scores = [
            verification_details.get(chunk.get('chunk_id', ''), {}).get('score', 0.5)
            for chunk in chunks
        ]
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        avg_credibility = sum(credibility_scores) / len(credibility_scores)
        
        confidence = (avg_relevance * 0.6) + (avg_credibility * 0.4)
        return min(1.0, max(0.0, confidence))

class KnowledgeManagerAgent(FinancialAgent):
    """Agent for Exercise 3: Adaptive Knowledge Base Management"""
    
    def __init__(self, azure_manager: AzureServiceManager, kb_manager: AdaptiveKnowledgeBaseManager):
        super().__init__(AgentType.KNOWLEDGE_MANAGER, azure_manager)
        self.kb_manager = kb_manager
        self.azure_ai_agent_service = self._initialize_azure_ai_agent_service(azure_manager)
        self.kb_agent_id = None
        self.current_thread_id = None
        self._initialize_capabilities()
        
    def _initialize_azure_ai_agent_service(self, azure_manager: AzureServiceManager) -> AzureAIAgentService:
        """Initialize Azure AI Agent Service for MCP client functionality"""
        try:
            # Use the correct initialization pattern with project client
            project_client = azure_manager.get_project_client()
            if not project_client:
                logger.error("Failed to get project client from Azure service manager")
                raise Exception("Azure AI Project client not available")
            
            return AzureAIAgentService(project_client)
            
        except Exception as e:
            logger.error(f"Error initializing Azure AI Agent Service: {e}")
            raise Exception(f"Failed to initialize Azure AI Agent Service: {e}")
    
    async def _ensure_kb_agent_initialized(self):
        """Ensure Knowledge Base agent is created and ready"""
        if not self.kb_agent_id:
            try:
                agent = await self.azure_ai_agent_service.create_knowledge_agent(
                    name="Financial Knowledge Base Agent",
                    instructions="You are a financial knowledge base management expert specializing in document processing, conflict resolution, and knowledge base health assessment.",
                    model_deployment=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
                )
                self.kb_agent_id = agent.id
                logger.info(f"Created Knowledge Base agent: {self.kb_agent_id}")
            except Exception as e:
                logger.error(f"Error creating Knowledge Base agent: {e}")
                raise
        
        if not self.current_thread_id:
            try:
                self.current_thread_id = await self.azure_ai_agent_service.create_thread(self.kb_agent_id)
                logger.info(f"Created thread: {self.current_thread_id}")
            except Exception as e:
                logger.error(f"Error creating thread: {e}")
                raise
        
    def _initialize_capabilities(self):
        """Initialize knowledge management capabilities"""
        self.capabilities = [
            AgentCapability(
                name="update_knowledge_base",
                description="Update knowledge base with new information",
                input_schema={
                    "source_url": {"type": "string"},
                    "content": {"type": "string"},
                    "metadata": {"type": "object"}
                },
                output_schema={
                    "update_status": {"type": "string"},
                    "conflicts_resolved": {"type": "array"},
                    "new_chunks_added": {"type": "integer"}
                }
            ),
            AgentCapability(
                name="assess_knowledge_health",
                description="Assess the health and quality of the knowledge base",
                input_schema={
                    "assessment_type": {"type": "string", "enum": ["full", "incremental"]}
                },
                output_schema={
                    "health_score": {"type": "number"},
                    "recommendations": {"type": "array"},
                    "statistics": {"type": "object"}
                }
            ),
            AgentCapability(
                name="resolve_information_conflicts",
                description="Resolve conflicts between different information sources",
                input_schema={
                    "conflict_data": {"type": "object", "required": True}
                },
                output_schema={
                    "resolution": {"type": "string"},
                    "confidence": {"type": "number"},
                    "sources_used": {"type": "array"}
                }
            )
        ]
    
    async def process_request(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process knowledge management requests"""
        try:
            capability = request.get("capability")
            
            if capability == "update_knowledge_base":
                return await self._update_knowledge_base(request, context)
            elif capability == "assess_knowledge_health":
                return await self._assess_knowledge_health(request, context)
            elif capability == "resolve_information_conflicts":
                return await self._resolve_information_conflicts(request, context)
            else:
                raise ValueError(f"Unknown capability: {capability}")
                
        except Exception as e:
            logger.error(f"Error processing knowledge management request: {e}")
            return {"error": str(e), "success": False}
    
    async def _update_knowledge_base(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Update knowledge base with new information"""
        source_url = request.get("source_url")
        content = request.get("content")
        metadata = request.get("metadata", {})
        
        update_result = await self.kb_manager.process_knowledge_update(
            source_url=source_url,
            content=content,
            metadata=metadata
        )
        
        return {
            "update_status": "completed",
            "conflicts_resolved": update_result.get("conflicts_resolved", []),
            "new_chunks_added": update_result.get("new_chunks_added", 0),
            "success": True
        }
    
    async def _assess_knowledge_health(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess knowledge base health"""
        assessment_type = request.get("assessment_type", "incremental")
        
        stats = await self.kb_manager.get_knowledge_base_statistics()
        
        health_score = await self.kb_manager._calculate_kb_health_score(stats.get("documents", []))
        
        recommendations = await self._generate_health_recommendations(stats, health_score)
        
        return {
            "health_score": health_score,
            "recommendations": recommendations,
            "statistics": stats,
            "assessment_type": assessment_type,
            "success": True
        }
    
    async def _resolve_information_conflicts(self, request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve information conflicts"""
        conflict_data = request["conflict_data"]
        
        resolution_result = await self.kb_manager._resolve_document_conflicts(
            conflict_data.get("conflicting_documents", []),
            conflict_data.get("conflict_type", "content_mismatch")
        )
        
        return {
            "resolution": resolution_result.get("resolution_strategy", "unknown"),
            "confidence": resolution_result.get("confidence", 0.5),
            "sources_used": resolution_result.get("sources_used", []),
            "success": True
        }
    
    async def _generate_health_recommendations(self, stats: Dict, health_score: float) -> List[str]:
        """Generate recommendations for improving knowledge base health"""
        recommendations = []
        
        if health_score < 0.7:
            recommendations.append("Consider updating older documents with more recent filings")
        
        if stats.get("total_documents", 0) < 10:
            recommendations.append("Expand knowledge base with more diverse financial documents")
        
        if stats.get("average_credibility", 0) < 0.8:
            recommendations.append("Focus on higher-credibility sources like SEC filings")
        
        return recommendations

class MultiAgentOrchestrator:
    """Main orchestrator for coordinating multiple financial agents"""
    
    def __init__(self, azure_manager: AzureServiceManager, kb_manager: AdaptiveKnowledgeBaseManager):
        self.azure_manager = azure_manager
        self.kb_manager = kb_manager
        self.agents = {}
        self.session_contexts = {}
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize all financial agents"""
        self.agents[AgentType.CONTENT_GENERATOR] = ContentGeneratorAgent(
            self.azure_manager, self.kb_manager
        )
        self.agents[AgentType.QA_AGENT] = QAAgent(
            self.azure_manager, self.kb_manager
        )
        self.agents[AgentType.KNOWLEDGE_MANAGER] = KnowledgeManagerAgent(
            self.azure_manager, self.kb_manager
        )
        
        logger.info(f"Initialized {len(self.agents)} financial agents")
    
    async def process_request(self, request: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Process a request through the appropriate agent"""
        try:
            agent_type_str = request.get("agent_type")
            if not agent_type_str:
                return {"error": "agent_type is required", "success": False}
            
            agent_type = AgentType(agent_type_str)
            agent = self.agents.get(agent_type)
            
            if not agent:
                return {"error": f"Agent type {agent_type_str} not found", "success": False}
            
            context = self._get_session_context(session_id)
            
            result = await agent.process_request(request, context)
            
            self._update_session_context(session_id, request, result)
            
            # Track the agent interaction using existing observability method
            observability.track_request(
                endpoint=f"agent_{agent_type.value}_{request.get('capability', 'unknown')}",
                session_id=session_id
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in multi-agent orchestrator: {e}")
            return {"error": str(e), "success": False}
    
    async def coordinate_agents(self, complex_request: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Coordinate multiple agents for complex requests"""
        try:
            required_agents = self._analyze_request_requirements(complex_request)
            
            results = {}
            context = self._get_session_context(session_id)
            
            for agent_type in required_agents:
                agent_request = self._prepare_agent_request(complex_request, agent_type)
                agent_result = await self.agents[agent_type].process_request(agent_request, context)
                results[agent_type.value] = agent_result
                
                context["previous_results"] = results
            
            final_result = await self._synthesize_agent_results(results, complex_request)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error coordinating agents: {e}")
            return {"error": str(e), "success": False}
    
    def _analyze_request_requirements(self, request: Dict[str, Any]) -> List[AgentType]:
        """Analyze request to determine which agents are needed"""
        request_type = request.get("type", "").lower()
        
        if "generate" in request_type or "content" in request_type:
            return [AgentType.KNOWLEDGE_MANAGER, AgentType.CONTENT_GENERATOR]
        elif "question" in request_type or "answer" in request_type:
            return [AgentType.KNOWLEDGE_MANAGER, AgentType.QA_AGENT]
        elif "update" in request_type or "knowledge" in request_type:
            return [AgentType.KNOWLEDGE_MANAGER]
        else:
            return [AgentType.KNOWLEDGE_MANAGER, AgentType.QA_AGENT, AgentType.CONTENT_GENERATOR]
    
    def _prepare_agent_request(self, original_request: Dict[str, Any], agent_type: AgentType) -> Dict[str, Any]:
        """Prepare request for specific agent"""
        base_request = original_request.copy()
        
        if agent_type == AgentType.CONTENT_GENERATOR:
            base_request["capability"] = "generate_financial_content"
        elif agent_type == AgentType.QA_AGENT:
            base_request["capability"] = "answer_financial_question"
        elif agent_type == AgentType.KNOWLEDGE_MANAGER:
            base_request["capability"] = "assess_knowledge_health"
        
        return base_request
    
    async def _synthesize_agent_results(self, results: Dict[str, Any], original_request: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from multiple agents"""
        synthesized = {
            "success": all(result.get("success", False) for result in results.values()),
            "agent_results": results,
            "synthesis": {}
        }
        
        confidence_scores = [
            result.get("confidence", result.get("confidence_score", 0.5))
            for result in results.values()
            if "confidence" in result or "confidence_score" in result
        ]
        
        if confidence_scores:
            synthesized["synthesis"]["overall_confidence"] = sum(confidence_scores) / len(confidence_scores)
        
        all_sources = []
        for result in results.values():
            if "sources" in result:
                all_sources.extend(result["sources"])
            elif "sources_used" in result:
                all_sources.extend(result["sources_used"])
        
        synthesized["synthesis"]["total_sources"] = len(set(str(source) for source in all_sources))
        
        return synthesized
    
    def _get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get or create session context"""
        if session_id not in self.session_contexts:
            self.session_contexts[session_id] = {
                "session_id": session_id,
                "created_at": datetime.utcnow(),
                "interactions": [],
                "context_summary": ""
            }
        
        return self.session_contexts[session_id]
    
    def _update_session_context(self, session_id: str, request: Dict[str, Any], result: Dict[str, Any]):
        """Update session context with new interaction"""
        context = self.session_contexts[session_id]
        
        interaction = {
            "timestamp": datetime.utcnow(),
            "request": request,
            "result": result,
            "success": result.get("success", False)
        }
        
        context["interactions"].append(interaction)
        
        context["interactions"] = context["interactions"][-10:]
    
    def get_agent_capabilities(self) -> Dict[str, List[AgentCapability]]:
        """Get capabilities of all agents"""
        capabilities = {}
        for agent_type, agent in self.agents.items():
            capabilities[agent_type.value] = agent.get_capabilities()
        
        return capabilities
    
    async def _get_azure_ai_agent_service(self) -> AzureAIAgentService:
        """Get Azure AI Agent Service from QA Agent"""
        qa_agent = self.agents.get(AgentType.QA_AGENT)
        if not qa_agent:
            raise ValueError("QA Agent not initialized")
        
        if not hasattr(qa_agent, 'azure_ai_agent_service') or qa_agent.azure_ai_agent_service is None:
            await qa_agent._ensure_qa_agent_initialized()
        
        return qa_agent.azure_ai_agent_service

    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            "agents_active": len(self.agents),
            "active_sessions": len(self.session_contexts),
            "agent_types": [agent_type.value for agent_type in self.agents.keys()],
            "system_health": "operational"
        }

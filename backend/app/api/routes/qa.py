from fastapi import APIRouter, HTTPException, Depends, Header
from typing import List, Optional
import logging
import uuid
import time
from datetime import datetime

from app.models.schemas import (
    QARequest,
    QAResponse,
    QuestionDecompositionRequest,
    QuestionDecompositionResponse,
    SourceVerificationRequest,
    SourceVerificationResponse,
    Citation,
    PerformanceMetrics,
    EvaluationRequest,
    EvaluatorType
)
from app.core.observability import observability
# Temporarily disable evaluation due to package conflicts
# from app.core.evaluation import get_evaluation_framework
from app.services.multi_agent_orchestrator import MultiAgentOrchestrator, AgentType
from app.services.azure_ai_agent_service import AzureAIAgentService
from app.services.azure_services import AzureServiceManager
from app.services.token_usage_tracker import TokenUsageTracker, ServiceType, OperationType
from app.services.performance_tracker import performance_tracker
from app.services.traditional_rag_service import TraditionalRAGService
from app.services.agentic_vector_rag_service import AgenticVectorRAGService
from app.services.evaluation_service import evaluation_service
from app.services.mcp_client import mcp_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/ask", response_model=QAResponse)
async def ask_question(
    request: QARequest,
    x_session_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None)
):
    """Main QA endpoint for complex financial question answering with source verification"""
    start_time = time.time()
    session_id = x_session_id or request.session_id or str(uuid.uuid4())
    
    # Initialize token usage tracker
    token_tracker = TokenUsageTracker()
    await token_tracker.initialize()
    
    # Start tracking the main QA operation
    tracking_id = token_tracker.start_tracking(
        session_id=session_id,
        service_type=ServiceType.QA_SERVICE,
        operation_type=OperationType.ANSWER_GENERATION,
        endpoint="/qa/ask",
        user_id=x_user_id,
        request_text=request.question,
        temperature=request.temperature,
        verification_level=request.verification_level.value if request.verification_level else None,
        credibility_check_enabled=request.credibility_check_enabled
    )
    
    try:
        async with observability.trace_operation(
            "qa_processing",
            session_id=session_id,
            model=request.chat_model,
            verification_level=request.verification_level
        ) as span:
            observability.track_request("qa_ask", session_id=session_id)
            logger.info(f"QA request received for session {session_id}, question: {request.question[:100]}...")
            
            # Define helper function for extracting deployment names (handle combined strings like "gpt-4o (chat4o)")
            def extract_deployment_name(model_value: str) -> str:
                if not model_value:
                    return "unknown"
                if '(' in model_value and ')' in model_value:
                    try:
                        return model_value.split('(')[1].split(')')[0].strip()
                    except (IndexError, AttributeError):
                        return model_value
                return model_value
            
            logger.info("Initializing Azure services...")
            azure_manager = AzureServiceManager()
            await azure_manager.initialize()
            logger.info("Azure services initialized successfully")
            
            logger.info("Creating knowledge base manager...")
            from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
            kb_manager = AdaptiveKnowledgeBaseManager(azure_manager)
            
            logger.info("Creating multi-agent orchestrator...")
            from app.services.multi_agent_orchestrator import MultiAgentOrchestrator
            orchestrator = MultiAgentOrchestrator(azure_manager, kb_manager)
            
            logger.info("Getting Azure AI Agent Service...")
            azure_ai_agent_service = await orchestrator._get_azure_ai_agent_service()
            logger.info(f"Azure AI Agent Service type: {type(azure_ai_agent_service).__name__}")
            logger.info("Processing QA request with Azure AI Agent Service...")
            
            # Initialize performance tracking
            question_id = str(uuid.uuid4())
            reasoning_chain = performance_tracker.start_reasoning_chain(question_id, request.question, session_id)
            
            # Step 1: Question Analysis
            step_num = performance_tracker.add_reasoning_step(
                question_id, 
                "Analyzing question complexity and requirements",
                "analyze",
                confidence=0.9,
                output="Question analyzed and complexity assessed"
            )
            performance_tracker.complete_reasoning_step(question_id, step_num)
            
            # Step 2: Knowledge Base Search
            step_num = performance_tracker.add_reasoning_step(
                question_id,
                "Searching knowledge base for relevant documents",
                "search", 
                confidence=0.8,
                output="Relevant documents identified"
            )
            
            # Create separate tracking session for embedding operations
            embedding_tracking_id = token_tracker.start_tracking(
                session_id=session_id,
                service_type=ServiceType.QA_SERVICE,
                operation_type=OperationType.EMBEDDING_GENERATION,
                endpoint="/api/v1/qa/ask",
                user_id=x_user_id,
                model_name=request.embedding_model,
                deployment_name=extract_deployment_name(request.embedding_model)
            )
            logger.info(f"Started separate embedding tracking session: {embedding_tracking_id}")
            
            logger.info(f"Using RAG method: {request.rag_method}")
            
            # Route to appropriate RAG implementation
            if request.rag_method == "traditional":
                logger.info("Processing with Traditional RAG...")
                qa_result = await process_with_traditional_rag(
                    request=request,
                    azure_manager=azure_manager,
                    session_id=session_id,
                    token_tracker=token_tracker,
                    tracking_id=tracking_id,
                    embedding_tracking_id=embedding_tracking_id,
                    question_id=question_id
                )
            elif request.rag_method == "llamaindex":
                # Placeholder for LlamaIndex implementation
                logger.info("LlamaIndex RAG not implemented yet, falling back to Agent")
                qa_result = await process_with_agent_rag(
                    request=request,
                    azure_manager=azure_manager,
                    kb_manager=kb_manager,
                    orchestrator=orchestrator,
                    session_id=session_id,
                    token_tracker=token_tracker,
                    tracking_id=tracking_id,
                    embedding_tracking_id=embedding_tracking_id,
                    question_id=question_id
                )
            elif request.rag_method == "agentic-vector":
                # Process with Agentic Vector RAG implementation
                logger.info("Processing with Agentic Vector RAG...")
                qa_result = await process_with_agentic_vector_rag(
                    request=request,
                    azure_manager=azure_manager,
                    session_id=session_id,
                    token_tracker=token_tracker,
                    tracking_id=tracking_id,
                    embedding_tracking_id=embedding_tracking_id,
                    question_id=question_id
                )
            elif request.rag_method == "mcp":
                # Process with MCP (Model Context Protocol)
                logger.info("Processing with MCP...")
                qa_result = await process_with_mcp(
                    request=request,
                    session_id=session_id,
                    token_tracker=token_tracker,
                    tracking_id=tracking_id,
                    question_id=question_id
                )
            else:  # default to "agent"
                logger.info("Processing with Azure AI Agent Service...")
                qa_result = await process_with_agent_rag(
                    request=request,
                    azure_manager=azure_manager,
                    kb_manager=kb_manager,
                    orchestrator=orchestrator,
                    session_id=session_id,
                    token_tracker=token_tracker,
                    tracking_id=tracking_id,
                    embedding_tracking_id=embedding_tracking_id,
                    question_id=question_id
                )
            
            logger.info(f"QA result received: {len(qa_result.get('answer', ''))} characters")
            
            # Complete search step
            performance_tracker.complete_reasoning_step(
                question_id, step_num, 
                f"Found {len(qa_result.get('sources', []))} relevant sources",
                confidence=0.8
            )
            
            # Step 3: Answer Synthesis
            step_num = performance_tracker.add_reasoning_step(
                question_id,
                "Synthesizing comprehensive answer from sources",
                "synthesize",
                sources_consulted=[s.get('source', '') for s in qa_result.get('sources', [])],
                confidence=0.85
            )
            
            chat_deployment_used = extract_deployment_name(request.chat_model)
            embedding_deployment_used = extract_deployment_name(request.embedding_model)
            
            answer = qa_result.get("answer", "I apologize, but I couldn't generate a comprehensive answer to your question at this time.")
            confidence_score = qa_result.get("confidence_score", 0.5)
            sub_questions = qa_result.get("sub_questions", [])
            verification_details = qa_result.get("verification_details", {})
            
            citations = []
            # Handle citations differently based on RAG method
            if request.rag_method == "traditional":
                # Traditional RAG returns citations directly
                for citation_data in qa_result.get("citations", []):
                    citations.append(Citation(**citation_data))
            elif request.rag_method == "agentic-vector":
                # Agentic Vector RAG returns properly formatted citations directly
                logger.info(f"🔍 DEBUG: Processing {len(qa_result.get('citations', []))} citations from agentic-vector RAG")
                for citation_data in qa_result.get("citations", []):
                    #logger.info(f"🔍 DEBUG: Citation data before processing: {citation_data}")
                    citations.append(Citation(**citation_data))
            else:
                # Agent RAG returns sources that need to be converted to citations
                for source in qa_result.get("sources", []):
                    citations.append(Citation(
                        id=source.get("id", str(uuid.uuid4())),
                        content=source.get("content", ""),
                        source=source.get("source", "Unknown Source"),
                        document_id=source.get("document_id", ""),
                        document_title=source.get("document_title", ""),
                        page_number=source.get("page_number"),
                        section_title=source.get("section_title"),
                        confidence=source.get("confidence", "medium"),
                        url=source.get("url", ""),
                        credibility_score=source.get("credibility_score", 0.5)
                    ))
            
            # Handle token usage differently based on RAG method
            if request.rag_method == "traditional" and qa_result.get("metadata", {}).get("tokens_used"):
                # Traditional RAG provides actual token usage
                token_usage = qa_result["metadata"]["tokens_used"]
                total_tokens = token_usage.get("total_tokens", 0)
            else:
                # Estimate tokens for other methods (existing logic)
                prompt_tokens = len(request.question.split()) * 1.3
                completion_tokens = len(answer.split()) * 1.3
                total_tokens = int(prompt_tokens + completion_tokens)
                
                token_usage = {
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": total_tokens
                }
            
            # Update token tracking with actual usage
            await token_tracker.update_usage(
                tracking_id=tracking_id,
                model_name=request.chat_model,
                deployment_name=chat_deployment_used,
                prompt_tokens=token_usage.get("prompt_tokens", 0),
                completion_tokens=token_usage.get("completion_tokens", 0),
                response_text=answer
            )
            
            observability.track_tokens(
                model=chat_deployment_used,
                prompt_tokens=token_usage["prompt_tokens"],
                completion_tokens=token_usage["completion_tokens"],
                session_id=session_id
            )
            
            response_time = time.time() - start_time
            
            sources = []
            for citation in citations:
                sources.append({
                    "content": citation.content,
                    "source": citation.source,
                    "title": citation.document_title,
                    "page_number": citation.page_number,
                    "section_title": citation.section_title,
                    "credibility_score": getattr(citation, 'credibility_score', 0.5)
                })

            # Evaluation integration
            evaluation_results = []
            evaluation_id = None
            
            if request.evaluation_enabled:
                try:
                    logger.info(f"Starting evaluation for question_id: {question_id}")
                    
                    # Create evaluation request
                    evaluation_request = EvaluationRequest(
                        question_id=question_id,
                        session_id=session_id,
                        evaluator_type=request.evaluator_type,
                        rag_method=request.rag_method,
                        evaluation_model=request.evaluation_model,
                        question=request.question,
                        answer=answer,
                        context=[c.content for c in citations],
                        ground_truth=None  # For now, we don't have ground truth
                    )
                    
                    # Start async evaluation
                    evaluation_result = await evaluation_service.evaluate_answer(evaluation_request, azure_manager)
                    evaluation_id = evaluation_result.id  # Extract the ID from the result
                    logger.info(f"Evaluation completed with ID: {evaluation_id}")
                    
                    # Track evaluation metrics for observability
                    span.set_attribute("evaluation.enabled", True)
                    span.set_attribute("evaluation.id", evaluation_id)
                    span.set_attribute("evaluation.type", request.evaluator_type.value)
                    span.set_attribute("evaluation.model", request.evaluation_model or "unknown")
                    
                except Exception as e:
                    logger.error(f"Evaluation failed for question_id {question_id}: {e}")
                    span.set_attribute("evaluation.error", str(e))
            else:
                span.set_attribute("evaluation.enabled", False)
            
            # Complete synthesis step and finalize reasoning chain
            performance_tracker.complete_reasoning_step(
                question_id, step_num,
                f"Generated comprehensive answer with {len(citations)} citations",
                confidence=confidence_score
            )
            
            # Add verification step if credibility checking is enabled
            if request.credibility_check_enabled:
                verification_step = performance_tracker.add_reasoning_step(
                    question_id,
                    "Verifying source credibility and reliability",
                    "verify",
                    sources_consulted=[c.source for c in citations],
                    confidence=verification_details.get("overall_credibility_score", 0.5)
                )
                performance_tracker.complete_reasoning_step(
                    question_id, verification_step,
                    f"Verified {verification_details.get('verified_sources_count', 0)} of {verification_details.get('total_sources_count', 0)} sources",
                    confidence=verification_details.get("overall_credibility_score", 0.5)
                )
            
            # Finalize reasoning chain
            final_reasoning_chain = performance_tracker.finalize_reasoning_chain(question_id, confidence_score)
            logger.info(f"🔍 DEBUG: Finalized reasoning chain for question_id: {question_id}")
            logger.info(f"🔍 DEBUG: Reasoning chain object: {final_reasoning_chain}")
            logger.info(f"🔍 DEBUG: Reasoning chain question_id: {final_reasoning_chain.question_id if final_reasoning_chain else 'None'}")
            
            # Create performance benchmark
            performance_benchmark = performance_tracker.create_performance_benchmark(
                question_id=question_id,
                question=request.question,
                processing_time_seconds=response_time,
                source_count=len(citations),
                accuracy_score=sum(r.score for r in evaluation_results) / len(evaluation_results) if evaluation_results else 0.8,
                confidence_score=confidence_score,
                verification_level=request.verification_level,
                session_id=session_id
            )
            
            response = QAResponse(
                answer=answer,
                session_id=session_id,
                question_id=question_id,
                confidence_score=confidence_score,
                citations=citations,
                sub_questions=sub_questions,
                verification_details=verification_details,
                performance_benchmark=performance_benchmark,
                reasoning_chain=final_reasoning_chain,
                metadata={
                    "exercise_type": "qa",
                    "model_used": chat_deployment_used,
                    "embedding_model": embedding_deployment_used,
                    "temperature": request.temperature,
                    "verification_level": request.verification_level,
                    "evaluation_enabled": request.evaluation_enabled,
                    "evaluation_id": evaluation_id,
                    "evaluator_type": request.evaluator_type.value if request.evaluation_enabled else None,
                    "evaluation_model": request.evaluation_model if request.evaluation_enabled else None,
                    "evaluation_count": len(evaluation_results),
                    "avg_evaluation_score": sum(r.score for r in evaluation_results) / len(evaluation_results) if evaluation_results else 0,
                    "response_time": response_time,
                    "rag_method": request.rag_method  # Add RAG method to metadata
                },
                token_usage=token_usage)
            
            #logger.info(f"🔍 DEBUG: Response reasoning_chain question_id: {response.reasoning_chain.question_id if response.reasoning_chain else 'None'}")
            #logger.info(f"🔍 DEBUG: Response reasoning_chain available: {response.reasoning_chain is not None}")
            
            span.set_attribute("response.tokens", total_tokens)
            span.set_attribute("response.citations", len(citations))
            span.set_attribute("response.confidence", confidence_score)
            span.set_attribute("response.sub_questions", len(sub_questions))
            
            # Finalize token tracking 
            await token_tracker.finalize_tracking(
                tracking_id=tracking_id,
                success=True,
                http_status_code=200,
                metadata={
                    "sources_count": len(citations),
                    "confidence_score": confidence_score,
                    "sub_questions_count": len(sub_questions),
                    "verification_details": verification_details
                }            )
            
            return response
        
    except Exception as e:
        # Finalize token tracking with error
        try:
            await token_tracker.finalize_tracking(
                tracking_id=tracking_id,
                success=False,
                http_status_code=500,
                error_message=str(e)
            )
        except Exception as tracker_error:
            logger.warning(f"Failed to finalize token tracking: {tracker_error}")
            
        observability.track_error(
            error_type=type(e).__name__,
            endpoint="/api/v1/qa/ask",
            error_message=str(e),
            session_id=session_id,
            user_id=x_user_id
        )
        
        logger.error(f"QA processing failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process QA request: {str(e)}")

@router.post("/decompose", response_model=QuestionDecompositionResponse)
async def decompose_question(
    request: QuestionDecompositionRequest,
    x_session_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None)
):
    """Decompose complex questions into researchable sub-questions"""
    start_time = time.time()
    session_id = x_session_id or str(uuid.uuid4())
    
    # Initialize token usage tracker
    token_tracker = TokenUsageTracker()
    await token_tracker.initialize()
    
    # Start tracking the decomposition operation
    tracking_id = token_tracker.start_tracking(
        session_id=session_id,
        service_type=ServiceType.QA_SERVICE,
        operation_type=OperationType.QUESTION_DECOMPOSITION,
        endpoint="/qa/decompose",
        user_id=x_user_id,
        request_text=request.question
    )
    
    try:
        async with observability.trace_operation(
            "question_decomposition",
            session_id=session_id,
            model=request.chat_model        ) as span:
            
            # Define helper function for extracting deployment names
            def extract_deployment_name(model_value: str) -> str:
                if not model_value:
                    return "unknown"
                if '(' in model_value and ')' in model_value:
                    try:
                        return model_value.split('(')[1].split(')')[0].strip()
                    except (IndexError, AttributeError):
                        return model_value
                return model_value
            
            observability.track_request("qa_decompose", session_id=session_id)
            logger.info(f"Question decomposition request for session {session_id}")
            
            azure_manager = AzureServiceManager()
            await azure_manager.initialize()
            
            from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
            kb_manager = AdaptiveKnowledgeBaseManager(azure_manager)
            
            from app.services.multi_agent_orchestrator import MultiAgentOrchestrator
            orchestrator = MultiAgentOrchestrator(azure_manager, kb_manager)
            azure_ai_agent_service = await orchestrator._get_azure_ai_agent_service()
              # Use Azure AI Agent Service for question decomposition
            decomposition_result = await azure_ai_agent_service.decompose_complex_question(
                question=request.question,
                context=request.context or {},
                session_id=session_id,
                model_config={
                    "chat_model": request.chat_model
                }            )
            
            sub_questions = decomposition_result.get("sub_questions", [])
            reasoning = decomposition_result.get("reasoning", "Question decomposition completed.")
            
            response_time = time.time() - start_time
            
            chat_deployment_used = extract_deployment_name(request.chat_model)
            
            # Estimate token usage for decomposition operation
            prompt_tokens = len(request.question.split()) * 1.3
            completion_tokens = len(reasoning.split()) * 1.3
            
            # Update token tracking with usage
            await token_tracker.update_usage(
                tracking_id=tracking_id,
                model_name=request.chat_model,
                deployment_name=chat_deployment_used,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                response_text=reasoning
            )
            
            response = QuestionDecompositionResponse(
                original_question=request.question,
                sub_questions=sub_questions,
                reasoning=reasoning,
                session_id=session_id,
                metadata={
                    "model_used": chat_deployment_used,
                    "response_time": response_time
                }
            )
            
            span.set_attribute("response.sub_questions_count", len(sub_questions))
            
            # Finalize token tracking
            await token_tracker.finalize_tracking(
                tracking_id=tracking_id,
                success=True,
                http_status_code=200,
                metadata={
                    "sub_questions_count": len(sub_questions),
                    "operation_type": "question_decomposition"
                }
            )
            
            return response
        
    except Exception as e:
        # Finalize token tracking with error
        try:
            await token_tracker.finalize_tracking(
                tracking_id=tracking_id,
                success=False,
                http_status_code=500,
                error_message=str(e)
            )
        except Exception as tracker_error:
            logger.warning(f"Failed to finalize token tracking: {tracker_error}")
            
        observability.track_error(
            error_type=type(e).__name__,
            endpoint="/api/v1/qa/decompose",
            error_message=str(e),
            session_id=session_id,
            user_id=x_user_id
        )
        
        logger.error(f"Question decomposition failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to decompose question: {str(e)}")

@router.post("/verify-sources")
async def verify_sources_flexible(
    request_data: dict,
    x_session_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None)
):
    """Flexible verify sources endpoint that handles different payload formats"""
    start_time = time.time()
    session_id = request_data.get("session_id") or x_session_id or str(uuid.uuid4())
    
    # Initialize token usage tracker
    token_tracker = TokenUsageTracker()
    await token_tracker.initialize()
    
    # Start tracking the source verification operation
    tracking_id = token_tracker.start_tracking(
        session_id=session_id,
        service_type=ServiceType.QA_SERVICE,
        operation_type=OperationType.SOURCE_VERIFICATION,
        endpoint="/qa/verify-sources",
        user_id=x_user_id,
        request_text=str(request_data)
    )
    
    try:
        async with observability.trace_operation(
            "flexible_source_verification",
            session_id=session_id
        ) as span:
            
            observability.track_request("qa_verify_sources_flexible", session_id=session_id)
            logger.info(f"Flexible source verification request for session {session_id}")
            logger.info(f"Request payload keys: {list(request_data.keys())}")
            
            azure_manager = AzureServiceManager()
            await azure_manager.initialize()
            
            from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
            kb_manager = AdaptiveKnowledgeBaseManager(azure_manager)
            
            from app.services.multi_agent_orchestrator import MultiAgentOrchestrator
            orchestrator = MultiAgentOrchestrator(azure_manager, kb_manager)
            azure_ai_agent_service = await orchestrator._get_azure_ai_agent_service()
            
            # Handle different payload formats
            sources_dict = []
            
            # Check if it's the single source format (from frontend)
            if "source_url" in request_data and "content" in request_data:
                logger.info("Processing single source format")
                source_dict = {
                    "id": str(uuid.uuid4()),
                    "url": request_data.get("source_url", ""),
                    "title": request_data.get("title", "Document Source"),
                    "content": request_data.get("content", ""),
                    "metadata": {
                        "source": request_data.get("source_url", ""),
                        "verification_timestamp": time.time()
                    }
                }
                sources_dict = [source_dict]
                
            # Check if it's the expected SourceVerificationRequest format
            elif "sources" in request_data:
                logger.info("Processing sources array format")
                for source in request_data["sources"]:
                    source_dict = {
                        "id": source.get("id", str(uuid.uuid4())),
                        "url": source.get("url", ""),
                        "title": source.get("title", ""),
                        "content": source.get("content", ""),
                        "metadata": source.get("metadata", {})
                    }
                    sources_dict.append(source_dict)
            else:
                raise HTTPException(status_code=422, detail="Invalid payload format. Expected either 'source_url' and 'content' or 'sources' array.")
            
            verification_result = await azure_ai_agent_service.verify_source_credibility(
                sources=sources_dict,
                context=request_data.get("context", {}),
                session_id=session_id
            )
            
            verified_sources = verification_result.get("verified_sources", [])
            overall_credibility = verification_result.get("overall_credibility_score", 0.5)
            
            response_time = time.time() - start_time
              # Ensure all credibility scores are proper numbers
            for source in verified_sources:
                score = source.get("credibility_score", 0.0)
                if score is None or str(score).lower() == 'nan':
                    source["credibility_score"] = 0.0
                    source["credibility_percentage"] = 0.0
                    source["verification_status"] = "unverified"
                
            overall_credibility = overall_credibility or 0.0
            if str(overall_credibility).lower() == 'nan':
                overall_credibility = 0.0
            
            # Estimate token usage for source verification
            content_length = sum(len(s.get("content", "")) for s in sources_dict)
            prompt_tokens = content_length // 3  # Rough estimation 
            completion_tokens = len(str(verified_sources)) // 3
            
            # Update token tracking with usage
            await token_tracker.update_usage(
                tracking_id=tracking_id,
                model_name="gpt-4o",  # Default model for verification
                deployment_name="chat4o",
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                response_text=str(verified_sources)
            )
            
            response = {
                "verified_sources": verified_sources,
                "overall_credibility_score": round(overall_credibility, 3),  # Ensure it's a proper number
                "verification_summary": f"Verified {len(verified_sources)} source(s) with average credibility score of {overall_credibility:.1%}",
                "session_id": session_id,
                "metadata": {
                    "sources_count": len(sources_dict),
                    "verified_count": len([s for s in verified_sources if s["verification_status"] == "verified"]),
                    "questionable_count": len([s for s in verified_sources if s["verification_status"] == "questionable"]),
                    "unverified_count": len([s for s in verified_sources if s["verification_status"] == "unverified"]),
                    "response_time": round(response_time, 3),
                    "average_credibility_percentage": round(overall_credibility * 100, 1)
                }
            }
            
            span.set_attribute("response.sources_verified", len(verified_sources))
            span.set_attribute("response.overall_credibility", overall_credibility)
            
            # Finalize token tracking
            await token_tracker.finalize_tracking(
                tracking_id=tracking_id,
                success=True,
                http_status_code=200,
                metadata={
                    "sources_verified": len(verified_sources),
                    "overall_credibility": overall_credibility,
                    "operation_type": "source_verification"
                }
            )
            
            return response
        
    except Exception as e:
        # Finalize token tracking with error
        try:
            await token_tracker.finalize_tracking(
                tracking_id=tracking_id,
                success=False,
                http_status_code=500,
                error_message=str(e)
            )
        except Exception as tracker_error:
            logger.warning(f"Failed to finalize token tracking: {tracker_error}")
            
        observability.track_error(
            error_type=type(e).__name__,
            endpoint="/api/v1/qa/verify-sources",
            error_message=str(e),
            session_id=session_id,
            user_id=x_user_id
        )
        
        logger.error(f"Flexible source verification failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to verify sources: {str(e)}")

@router.post("/verify-citations", response_model=SourceVerificationResponse)
async def verify_citations(
    citations: List[Citation],
    x_session_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None)
):
    """Verify the credibility of Citation objects (alternative endpoint for frontend compatibility)"""
    start_time = time.time()
    session_id = x_session_id or str(uuid.uuid4())
    
    # Initialize token usage tracker
    token_tracker = TokenUsageTracker()
    await token_tracker.initialize()
    
    # Start tracking the citation verification operation
    tracking_id = token_tracker.start_tracking(
        session_id=session_id,
        service_type=ServiceType.QA_SERVICE,
        operation_type=OperationType.SOURCE_VERIFICATION,
        endpoint="/qa/verify-citations",
        user_id=x_user_id,
        request_text=f"Verifying {len(citations)} citations"
    )
    
    try:
        async with observability.trace_operation(
            "citation_verification",
            session_id=session_id
        ) as span:
            
            observability.track_request("qa_verify_citations", session_id=session_id)
            logger.info(f"Citation verification request for session {session_id}, {len(citations)} citations")
            
            azure_manager = AzureServiceManager()
            await azure_manager.initialize()
            
            from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
            kb_manager = AdaptiveKnowledgeBaseManager(azure_manager)
            
            from app.services.multi_agent_orchestrator import MultiAgentOrchestrator
            orchestrator = MultiAgentOrchestrator(azure_manager, kb_manager)
            azure_ai_agent_service = await orchestrator._get_azure_ai_agent_service()
            
            # Convert Citation objects to dictionaries for processing
            sources_dict = []
            for citation in citations:
                source_dict = {
                    "id": citation.id,
                    "url": citation.url or "",
                    "title": citation.document_title,
                    "content": citation.content,
                    "metadata": {
                        "document_id": citation.document_id,                        "page_number": citation.page_number,
                        "section_title": citation.section_title,
                        "credibility_score": citation.credibility_score
                    }
                }
                sources_dict.append(source_dict)
            
            verification_result = await azure_ai_agent_service.verify_source_credibility(
                sources=sources_dict,
                context={},
                session_id=session_id
            )
            
            verified_sources = verification_result.get("verified_sources", [])
            overall_credibility = verification_result.get("overall_credibility_score", 0.5)
            
            response_time = time.time() - start_time
            
            # Estimate token usage for citation verification
            content_length = sum(len(citation.content or "") for citation in citations)
            prompt_tokens = content_length // 3  # Rough estimation 
            completion_tokens = len(str(verified_sources)) // 3
            
            # Update token tracking with usage
            await token_tracker.update_usage(
                tracking_id=tracking_id,
                model_name="gpt-4o",  # Default model for verification
                deployment_name="chat4o",
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                response_text=str(verified_sources)
            )
            
            response = SourceVerificationResponse(
                verified_sources=verified_sources,
                overall_credibility_score=overall_credibility,
                verification_summary=f"Verified {len(verified_sources)} citations with average credibility score of {overall_credibility:.2f}",
                session_id=session_id,
                metadata={
                    "sources_count": len(citations),
                    "verified_count": len([s for s in verified_sources if s["verification_status"] == "verified"]),
                    "response_time": response_time
                }
            )
            
            span.set_attribute("response.sources_verified", len(verified_sources))
            span.set_attribute("response.overall_credibility", overall_credibility)
            
            # Finalize token tracking
            await token_tracker.finalize_tracking(
                tracking_id=tracking_id,
                success=True,
                http_status_code=200,
                metadata={
                    "citations_verified": len(verified_sources),
                    "overall_credibility": overall_credibility,
                    "operation_type": "citation_verification"
                }
            )
            
            return response
        
    except Exception as e:
        # Finalize token tracking with error
        try:
            await token_tracker.finalize_tracking(
                tracking_id=tracking_id,
                success=False,
                http_status_code=500,
                error_message=str(e)
            )
        except Exception as tracker_error:
            logger.warning(f"Failed to finalize token tracking: {tracker_error}")
            
        observability.track_error(
            error_type=type(e).__name__,
            endpoint="/api/v1/qa/verify-citations",
            error_message=str(e),
            session_id=session_id,
            user_id=x_user_id
        )
        
        logger.error(f"Citation verification failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to verify citations: {str(e)}")

@router.post("/verify-source", response_model=SourceVerificationResponse)
async def verify_single_source(
    source_url: str,
    content: str,
    session_id: str,
    title: Optional[str] = None,
    x_session_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None)
):
    """Verify a single source (matches the frontend payload format)"""
    start_time = time.time()
    session_id = session_id or x_session_id or str(uuid.uuid4())
    
    # Initialize token usage tracker
    token_tracker = TokenUsageTracker()
    await token_tracker.initialize()
    
    # Start tracking the single source verification operation
    tracking_id = token_tracker.start_tracking(
        session_id=session_id,
        service_type=ServiceType.QA_SERVICE,
        operation_type=OperationType.SOURCE_VERIFICATION,
        endpoint="/qa/verify-source",
        user_id=x_user_id,
        request_text=f"Verifying source: {source_url}"
    )
    
    try:
        async with observability.trace_operation(
            "single_source_verification",
            session_id=session_id
        ) as span:
            
            observability.track_request("qa_verify_single_source", session_id=session_id)
            logger.info(f"Single source verification request for session {session_id}")
            
            azure_manager = AzureServiceManager()
            await azure_manager.initialize()
            
            from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
            kb_manager = AdaptiveKnowledgeBaseManager(azure_manager)
            
            from app.services.multi_agent_orchestrator import MultiAgentOrchestrator
            orchestrator = MultiAgentOrchestrator(azure_manager, kb_manager)
            azure_ai_agent_service = await orchestrator._get_azure_ai_agent_service()
            
            # Convert single source to the expected format
            source_dict = {
                "id": str(uuid.uuid4()),
                "url": source_url,
                "title": title or "Document Source",                "content": content,
                "metadata": {
                    "source": source_url,
                    "verification_timestamp": time.time()
                }
            }
            
            verification_result = await azure_ai_agent_service.verify_source_credibility(
                sources=[source_dict],
                context={},
                session_id=session_id
            )
            
            verified_sources = verification_result.get("verified_sources", [])
            overall_credibility = verification_result.get("overall_credibility_score", 0.5)
            
            response_time = time.time() - start_time
            
            # Estimate token usage for single source verification
            prompt_tokens = len(content) // 3  # Rough estimation 
            completion_tokens = len(str(verified_sources)) // 3
            
            # Update token tracking with usage
            await token_tracker.update_usage(
                tracking_id=tracking_id,
                model_name="gpt-4o",  # Default model for verification
                deployment_name="chat4o",
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                response_text=str(verified_sources)
            )
            
            response = SourceVerificationResponse(
                verified_sources=verified_sources,
                overall_credibility_score=overall_credibility,
                verification_summary=f"Verified source with credibility score of {overall_credibility:.2f}",
                session_id=session_id,
                metadata={
                    "sources_count": 1,
                    "verified_count": len([s for s in verified_sources if s["verification_status"] == "verified"]),
                    "response_time": response_time,
                    "source_type": "single_source"
                }
            )
            
            span.set_attribute("response.sources_verified", len(verified_sources))
            span.set_attribute("response.overall_credibility", overall_credibility)
            
            # Finalize token tracking
            await token_tracker.finalize_tracking(
                tracking_id=tracking_id,
                success=True,
                http_status_code=200,
                metadata={
                    "source_verified": True,
                    "overall_credibility": overall_credibility,
                    "operation_type": "single_source_verification"
                }
            )
            
            return response
        
    except Exception as e:
        # Finalize token tracking with error
        try:
            await token_tracker.finalize_tracking(
                tracking_id=tracking_id,
                success=False,
                http_status_code=500,
                error_message=str(e)
            )
        except Exception as tracker_error:
            logger.warning(f"Failed to finalize token tracking: {tracker_error}")
            
        observability.track_error(
            error_type=type(e).__name__,
            endpoint="/api/v1/qa/verify-source",
            error_message=str(e),
            session_id=session_id,
            user_id=x_user_id
        )
        
        logger.error(f"Single source verification failed for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to verify source: {str(e)}")

@router.get("/capabilities")
async def get_qa_capabilities():
    """Get available QA agent capabilities"""
    try:
        observability.track_request("qa_capabilities")
        
        azure_manager = AzureServiceManager()
        await azure_manager.initialize()
        
        from app.services.knowledge_base_manager import AdaptiveKnowledgeBaseManager
        kb_manager = AdaptiveKnowledgeBaseManager(azure_manager)
        
        orchestrator = MultiAgentOrchestrator(azure_manager, kb_manager)
        all_capabilities = orchestrator.get_agent_capabilities()
        capabilities = all_capabilities.get("qa_agent", [])
        
        return {
            "capabilities": capabilities,
            "verification_levels": ["basic", "thorough", "comprehensive"],
            "supported_question_types": [
                "financial_analysis",
                "market_research", 
                "regulatory_compliance",
                "risk_assessment",
                "performance_evaluation"
            ]
        }
    except Exception as e:
        logger.error(f"Error getting QA capabilities: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve QA capabilities")

@router.get("/performance-metrics/{session_id}", response_model=PerformanceMetrics)
async def get_performance_metrics(
    session_id: str,
    x_user_id: Optional[str] = Header(None)
):
    """Get performance metrics for a QA session"""
    try:
        metrics = performance_tracker.get_session_metrics(session_id)
        logger.info(f"Retrieved performance metrics for session {session_id}: "
                   f"{metrics.total_questions} questions, "
                   f"{metrics.average_efficiency_gain:.1f}% avg efficiency gain")
        return metrics
    except Exception as e:
        logger.error(f"Error retrieving performance metrics for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance metrics: {str(e)}")

@router.get("/performance-metrics/question/{question_id}")
async def get_question_performance_metrics(
    question_id: str,
    x_user_id: Optional[str] = Header(None)
):
    """Get performance metrics for a specific question"""
    try:
        # Get the performance benchmark for this specific question
        benchmark = performance_tracker.get_performance_benchmark(question_id)
        if not benchmark:
            raise HTTPException(status_code=404, detail="Performance metrics not found for this question")
        
        logger.info(f"Retrieved performance metrics for question {question_id}")
        return benchmark
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving performance metrics for question {question_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance metrics: {str(e)}")

@router.get("/reasoning-chain/{question_id}")
async def get_reasoning_chain(
    question_id: str,
    x_user_id: Optional[str] = Header(None)
):
    """Get the reasoning chain for a specific question"""
    try:
        logger.info(f"Getting reasoning chain for question_id: {question_id}")
        reasoning_chain = performance_tracker.get_reasoning_chain(question_id)
        
        if not reasoning_chain:
            logger.warning(f"Reasoning chain not found for question_id: {question_id}")
            logger.info(f"Available reasoning chains: {list(performance_tracker.reasoning_chains.keys())}")
            raise HTTPException(status_code=404, detail="Reasoning chain not found")
        
        logger.info(f"Retrieved reasoning chain for question {question_id}: {len(reasoning_chain.reasoning_steps)} steps")
        logger.info(f"Reasoning chain data: question_id={reasoning_chain.question_id}, steps={len(reasoning_chain.reasoning_steps)}")
        
        # Return the reasoning chain directly to match frontend expectations
        return reasoning_chain
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving reasoning chain for question {question_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve reasoning chain: {str(e)}")

@router.get("/traditional-rag/diagnose")
async def diagnose_traditional_rag():
    """Diagnostic endpoint for Traditional RAG service"""
    try:
        logger.info("Running Traditional RAG diagnostics...")
        
        # Initialize services
        azure_manager = AzureServiceManager()
        await azure_manager.initialize()
        
        traditional_rag = TraditionalRAGService(azure_manager)
        await traditional_rag.initialize()
        
        # Run diagnostics
        results = await traditional_rag.diagnose_knowledge_base()
        
        return {
            "status": "success",
            "diagnostics": results,
            "message": "Traditional RAG diagnostic completed"
        }
        
    except Exception as e:
        logger.error(f"Error in Traditional RAG diagnostics: {e}")
        raise HTTPException(status_code=500, detail=f"Diagnostic failed: {str(e)}")

@router.get("/agentic-vector-rag/diagnose")
async def diagnose_agentic_vector_rag():
    """Diagnostic endpoint for Agentic Vector RAG service"""
    try:
        logger.info("Running Agentic Vector RAG diagnostics...")
        
        # Initialize services
        azure_manager = AzureServiceManager()
        await azure_manager.initialize()
        
        agentic_vector_rag = AgenticVectorRAGService(azure_manager)
        await agentic_vector_rag.initialize()
        
        # Run diagnostics
        results = await agentic_vector_rag.get_diagnostics()
        
        return {
            "status": "success",
            "diagnostics": results,
            "message": "Agentic Vector RAG diagnostic completed"
        }
        
    except Exception as e:
        logger.error(f"Error in Agentic Vector RAG diagnostics: {e}")
        raise HTTPException(status_code=500, detail=f"Diagnostic failed: {str(e)}")

@router.get("/diagnostic/knowledge-base")
async def diagnostic_knowledge_base():
    """Diagnostic endpoint to check knowledge base contents"""
    try:
        logger.info("Running knowledge base diagnostic...")
        
        # Initialize Azure manager
        azure_manager = AzureServiceManager()
        await azure_manager.initialize()
        
        # Try a simple wildcard search to see what's in the index
        logger.info("Testing wildcard search...")
        results = await azure_manager.hybrid_search(
            query="*",
            top_k=5,
            min_score=0.0
        )
        
        # Try searching for common financial terms
        test_queries = ["Apple", "financial", "revenue", "SEC", "10-K"]
        search_results = {}
        
        for query in test_queries:
            logger.info(f"Testing search for: {query}")
            search_res = await azure_manager.hybrid_search(
                query=query,
                top_k=3,
                min_score=0.0
            )
            search_results[query] = {
                "count": len(search_res),
                "scores": [r.get('search_score', 0) for r in search_res] if search_res else []
            }
        
        return {
            "status": "success",
            "total_documents_sample": len(results),
            "sample_documents": [
                {
                    "document_id": r.get("document_id", "unknown"),
                    "company": r.get("company", "unknown"),
                    "form_type": r.get("form_type", "unknown"),
                    "content_preview": r.get("content", "")[:200] + "..." if r.get("content") else "No content",
                    "search_score": r.get("search_score", 0)
                }
                for r in results[:3]
            ],
            "test_search_results": search_results,
            "message": f"Found {len(results)} documents in knowledge base"
        }
        
    except Exception as e:
        logger.error(f"Diagnostic error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Failed to run knowledge base diagnostic"
        }

async def process_with_traditional_rag(
    request: QARequest,
    azure_manager: AzureServiceManager,
    session_id: str,
    token_tracker: TokenUsageTracker,
    tracking_id: str,
    embedding_tracking_id: str,
    question_id: str
) -> dict:
    """Process QA request using Traditional RAG approach"""
    logger.info("Initializing Traditional RAG service...")
    
    # Add reasoning step for initialization
    step_num = performance_tracker.add_reasoning_step(
        question_id,
        "Initializing Traditional RAG service",
        "initialize",
        confidence=0.9,
        output="Traditional RAG service initialized"
    )
    
    # Initialize Traditional RAG service
    traditional_rag = TraditionalRAGService(azure_manager)
    await traditional_rag.initialize()
    
    performance_tracker.complete_reasoning_step(question_id, step_num)
    
    # Add reasoning step for search
    step_num = performance_tracker.add_reasoning_step(
        question_id,
        "Performing traditional vector search and retrieval",
        "search",
        confidence=0.8,
        output="Vector search initiated"
    )
    
    # Process the question
    result = await traditional_rag.process_question(
        question=request.question,
        session_id=session_id,
        model_config={
            "chat_model": request.chat_model,
            "embedding_model": request.embedding_model,
            "temperature": request.temperature
        },
        verification_level=request.verification_level,
        token_tracker=token_tracker,
        tracking_id=tracking_id,
        credibility_check_enabled=request.credibility_check_enabled
    )
    
    # Complete the search step
    sources_found = len(result.get("sources", []))
    citations_found = len(result.get("citations", []))
    performance_tracker.complete_reasoning_step(
        question_id, 
        step_num,
        f"Traditional search completed: {sources_found} sources, {citations_found} citations found",
        confidence=result.get("confidence_score", 0.8)
    )
    
    # Add reasoning step for answer generation if we have results
    if result.get("answer"):
        step_num = performance_tracker.add_reasoning_step(
            question_id,
            "Generating answer from traditional RAG results",
            "generate",
            sources_consulted=[f"Traditional source {i+1}" for i in range(sources_found)],
            confidence=result.get("confidence_score", 0.8),
            output="Answer generated from traditional RAG results"
        )
        performance_tracker.complete_reasoning_step(question_id, step_num)
    
    return result

async def process_with_agent_rag(
    request: QARequest,
    azure_manager: AzureServiceManager,
    kb_manager,
    orchestrator,
    session_id: str,
    token_tracker: TokenUsageTracker,
    tracking_id: str,
    embedding_tracking_id: str,
    question_id: str
) -> dict:
    """Process QA request using Azure AI Agent Service (existing implementation)"""
    logger.info("Getting Azure AI Agent Service...")
    azure_ai_agent_service = await orchestrator._get_azure_ai_agent_service()
    
    logger.info("Processing QA request with Azure AI Agent Service...")
    result = await azure_ai_agent_service.process_qa_request(
        question=request.question,
        context={
            **(request.context or {}),
            'kb_manager': kb_manager,
            'credibility_check_enabled': request.credibility_check_enabled,
            'token_tracker': token_tracker,
            'tracking_id': tracking_id,
            'embedding_tracking_id': embedding_tracking_id,
            'performance_tracker': performance_tracker,
            'question_id': question_id
        },
        verification_level=request.verification_level,
        session_id=session_id,
        model_config={
            "chat_model": request.chat_model,
            "embedding_model": request.embedding_model,
            "temperature": request.temperature
        }
    )
    
    return result

async def process_with_agentic_vector_rag(
    request: QARequest,
    azure_manager: AzureServiceManager,
    session_id: str,
    token_tracker: TokenUsageTracker,
    tracking_id: str,
    embedding_tracking_id: str,
    question_id: str
) -> dict:
    """Process QA request using Agentic Vector RAG implementation"""
    logger.info("Initializing Agentic Vector RAG service...")
    
    # Add reasoning step for initialization
    step_num = performance_tracker.add_reasoning_step(
        question_id,
        "Initializing Agentic Vector RAG service and knowledge agent",
        "initialize",
        confidence=0.9,
        output="Agentic Vector RAG service initialized"
    )
    
    # Initialize the Agentic Vector RAG service
    agentic_vector_service = AgenticVectorRAGService(azure_manager)
    await agentic_vector_service.initialize()
    
    performance_tracker.complete_reasoning_step(question_id, step_num)
    
    # Add reasoning step for query planning
    step_num = performance_tracker.add_reasoning_step(
        question_id,
        "Performing intelligent query planning and agentic retrieval",
        "agentic_retrieval",
        confidence=0.85,
        output="Query planning and parallel subquery execution initiated"
    )
    
    # Build context including conversation history and performance tracking
    context = {
        **(request.context or {}),
        'session_id': session_id,
        'token_tracker': token_tracker,
        'tracking_id': tracking_id,
        'embedding_tracking_id': embedding_tracking_id,
        'performance_tracker': performance_tracker,
        'question_id': question_id,
        'conversation_history': request.context.get('conversation_history', []) if request.context else []
    }
    
    logger.info("Processing QA request with Agentic Vector RAG service...")
    result = await agentic_vector_service.process_question(
        question=request.question,
        session_id=session_id,
        model_config={
            "chat_model": request.chat_model,
            "embedding_model": request.embedding_model,
            "temperature": request.temperature
        },
        verification_level=request.verification_level,
        token_tracker=token_tracker,
        tracking_id=tracking_id,
        context=context,
        credibility_check_enabled=request.credibility_check_enabled
    )
    
    # Complete the agentic retrieval step
    sources_found = len(result.get("sources", []))
    citations_found = len(result.get("citations", []))
    performance_tracker.complete_reasoning_step(
        question_id, 
        step_num,
        f"Agentic retrieval completed: {sources_found} sources, {citations_found} citations found",
        confidence=result.get("confidence_score", 0.8)
    )
    
    # Add reasoning step for answer synthesis if we have results
    if result.get("answer"):
        step_num = performance_tracker.add_reasoning_step(
            question_id,
            "Synthesizing comprehensive answer from agentic retrieval results",
            "synthesize",
            sources_consulted=[f"Agentic source {i+1}" for i in range(sources_found)],
            confidence=result.get("confidence_score", 0.8),
            output="Answer synthesized from agentic retrieval results"
        )
        performance_tracker.complete_reasoning_step(question_id, step_num)
    
    return result

async def process_with_mcp(
    request: QARequest,
    session_id: str,
    token_tracker: TokenUsageTracker,
    tracking_id: str,
    question_id: str
) -> dict:
    """Process question using MCP (Model Context Protocol)"""
    try:
        # Add reasoning step for MCP processing
        step_num = performance_tracker.add_reasoning_step(
            question_id,
            "Processing question using MCP (Model Context Protocol)",
            "mcp_query",
            sources_consulted=["MCP Server"],
            confidence=0.9,
            output="Initiating MCP request"
        )
        
        # Process question through MCP service
        context = request.context or ""
        result = await mcp_service.process_question_with_mcp(
            question=request.question,
            session_id=session_id,
            verification_level=request.verification_level,
            context=context
        )
        
        # Complete the MCP processing step
        mcp_success = result.get("metadata", {}).get("success", False)
        sources_found = len(result.get("sources", []))
        
        performance_tracker.complete_reasoning_step(
            question_id,
            step_num,
            f"MCP processing {'completed successfully' if mcp_success else 'failed'}: {sources_found} sources found",
            confidence=result.get("confidence", 0.5)
        )
        
        # Track token usage if available
        if result.get("metadata", {}).get("tokens_used"):
            await token_tracker.track_usage(
                tracking_id,
                service_type=ServiceType.MCP,
                operation_type=OperationType.QA,
                input_tokens=result["metadata"]["tokens_used"].get("input", 0),
                output_tokens=result["metadata"]["tokens_used"].get("output", 0),
                total_tokens=result["metadata"]["tokens_used"].get("total", 0),
                cost=result["metadata"]["tokens_used"].get("cost", 0.0)
            )
        
        # Add reasoning step for answer synthesis if we have results
        if result.get("answer") and mcp_success:
            step_num = performance_tracker.add_reasoning_step(
                question_id,
                "Synthesizing answer from MCP server response", 
                "synthesize",
                sources_consulted=[f"MCP source {i+1}" for i in range(sources_found)],
                confidence=result.get("confidence", 0.8),
                output="Answer synthesized from MCP server response"
            )
            performance_tracker.complete_reasoning_step(question_id, step_num)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in MCP processing: {e}")
        # Add error step
        step_num = performance_tracker.add_reasoning_step(
            question_id,
            f"MCP processing failed: {str(e)}",
            "error",
            sources_consulted=["MCP Server"],
            confidence=0.0,
            output=f"Error: {str(e)}"
        )
        performance_tracker.complete_reasoning_step(question_id, step_num)
        
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

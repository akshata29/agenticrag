"""
Agentic Vector RAG Service Implementation
Based on Azure AI Search Agentic Retrieval concept
https://learn.microsoft.com/en-us/azure/search/search-agentic-retrieval-concept
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional
from azure.search.documents.indexes.models import (
    KnowledgeAgent, 
    KnowledgeAgentAzureOpenAIModel, 
    KnowledgeAgentTargetIndex, 
    KnowledgeAgentRequestLimits, 
    AzureOpenAIVectorizerParameters
)
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.search.documents.agent.models import (
    KnowledgeAgentRetrievalRequest, 
    KnowledgeAgentMessage, 
    KnowledgeAgentMessageTextContent, 
    KnowledgeAgentIndexParams
)
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

from app.services.azure_services import AzureServiceManager
from app.services.token_usage_tracker import TokenUsageTracker
from app.services.credibility_assessor import CredibilityAssessor
from app.models.schemas import VerificationLevel
from app.core.config import settings

logger = logging.getLogger(__name__)

class AgenticVectorRAGService:
    """
    Agentic Vector RAG implementation following Azure AI Search best practices.
    Uses Knowledge Agents for intelligent query planning and parallel subquery execution.
    """
    
    def __init__(self, azure_manager: AzureServiceManager):
        self.azure_manager = azure_manager
        self.agent_name = getattr(settings, 'AZURE_SEARCH_AGENT_NAME', 'financial-qa-agent')
        self.knowledge_agent_client = None
        self.index_client = None
        self.credibility_assessor = CredibilityAssessor(azure_manager)
        
    async def initialize(self):
        """Initialize the Agentic Vector RAG service"""
        try:
            # Use proper authentication - prefer API key if available, fall back to managed identity
            if settings.AZURE_SEARCH_API_KEY:
                credential = AzureKeyCredential(settings.AZURE_SEARCH_API_KEY)
                logger.info("Using Azure Search API key for authentication")
            else:
                credential = DefaultAzureCredential()
                logger.info("Using DefaultAzureCredential for Azure Search authentication")
            
            # Initialize Azure Search Index Client with the latest API version for agentic features
            self.index_client = SearchIndexClient(
                endpoint=settings.AZURE_AI_SEARCH_ENDPOINT,
                credential=credential,
                api_version="2025-05-01-preview"  # Correct API version for agentic features
            )
            
            # Try to create or update the knowledge agent
            try:
                await self._create_or_update_knowledge_agent()
                
                # Initialize the Knowledge Agent Retrieval Client
                self.knowledge_agent_client = KnowledgeAgentRetrievalClient(
                    endpoint=settings.AZURE_AI_SEARCH_ENDPOINT,
                    agent_name=self.agent_name,
                    credential=credential,
                    api_version="2025-05-01-preview"  # Correct API version for agentic features
                )
                
                logger.info("Agentic Vector RAG service initialized successfully with full agentic capabilities")
                
            except Exception as agent_error:
                logger.warning(f"Failed to initialize knowledge agent: {agent_error}")
                logger.warning("Agentic Vector RAG will operate in fallback mode without knowledge agent")
                # Set knowledge_agent_client to None to indicate fallback mode
                self.knowledge_agent_client = None
            
        except Exception as e:
            logger.error(f"Failed to initialize Agentic Vector RAG service: {e}")
            raise

    async def _create_or_update_knowledge_agent(self):
        """Create or update the Knowledge Agent in Azure AI Search"""
        try:
            # Extract deployment name from model config
            chat_deployment = self._extract_deployment_name(settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME)
            
            logger.info(f"Creating/updating knowledge agent '{self.agent_name}' with:")
            logger.info(f"  - OpenAI endpoint: {settings.AZURE_OPENAI_ENDPOINT}")
            logger.info(f"  - Chat deployment: {chat_deployment}")
            logger.info(f"  - Model name: {settings.AZURE_OPENAI_CHAT_MODEL_NAME}")
            logger.info(f"  - Target index: {settings.AZURE_AI_SEARCH_INDEX_NAME}")
            
            agent = KnowledgeAgent(
                name=self.agent_name,
                models=[
                    KnowledgeAgentAzureOpenAIModel(
                        azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                            resource_url=settings.AZURE_OPENAI_ENDPOINT,
                            deployment_name=chat_deployment,
                            model_name=settings.AZURE_OPENAI_CHAT_MODEL_NAME or "gpt-4o-mini",
                            api_key=settings.AZURE_OPENAI_API_KEY  # Use API key for authentication
                        )
                    )
                ],
                target_indexes=[
                    KnowledgeAgentTargetIndex(
                        index_name=settings.AZURE_AI_SEARCH_INDEX_NAME,
                        default_reranker_threshold=2.0  # Lower threshold for financial data
                    )
                ],
                request_limits=KnowledgeAgentRequestLimits(
                    max_output_size=15000  # Larger output for comprehensive financial answers
                )
            )
            
            self.index_client.create_or_update_agent(agent)
            logger.info(f"Knowledge agent '{self.agent_name}' created or updated successfully")
            
        except Exception as e:
            logger.error(f"Failed to create/update knowledge agent: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            if hasattr(e, 'status_code'):
                logger.error(f"HTTP status code: {e.status_code}")
            if hasattr(e, 'response'):
                logger.error(f"Response content: {e.response}")
            raise

    async def process_question(
        self,
        question: str,
        session_id: str,
        model_config: Dict[str, Any],
        verification_level: VerificationLevel = VerificationLevel.BASIC,
        token_tracker: Optional[TokenUsageTracker] = None,
        tracking_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        credibility_check_enabled: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a question using Agentic Vector RAG pattern:
        1. Use LLM for intelligent query planning
        2. Break down complex questions into subqueries
        3. Execute subqueries in parallel
        4. Semantically rank and merge results
        5. Generate comprehensive response with citations
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting Agentic Vector RAG processing for question: {question[:100]}...")
            
            if not self.knowledge_agent_client and not self.index_client:
                await self.initialize()
            
            # Check if we have the knowledge agent available
            if not self.knowledge_agent_client:
                logger.warning("Knowledge agent not available, using fallback implementation")
                return await self._fallback_process_question(
                    question, session_id, model_config, verification_level, 
                    token_tracker, tracking_id, context, **kwargs
                )
            
            # Build conversation history for context-aware query planning
            messages = self._build_conversation_messages(question, context)
            
            logger.info(f"🤖 Agentic Vector RAG: Step 1 - Intelligent query planning and retrieval for: '{question[:100]}...'")
            
            # Use the Knowledge Agent for agentic retrieval
            retrieval_result = await self._perform_agentic_retrieval(
                messages=messages,
                verification_level=verification_level,
                token_tracker=token_tracker,
                tracking_id=tracking_id
            )
            
            logger.info(f"🤖 Agentic Vector RAG: Retrieval completed")
            
            if not retrieval_result or not retrieval_result.response:
                logger.warning("🤖 Agentic Vector RAG: No response from agentic retrieval")
                return {
                    "answer": "I couldn't find relevant information in the knowledge base to answer your question using agentic retrieval.",
                    "confidence_score": 0.1,
                    "sources": [],
                    "citations": [],
                    "metadata": {
                        "rag_method": "agentic-vector",
                        "sources_found": 0,
                        "processing_time": time.time() - start_time,
                        "query_plan_activities": []
                    }
                }

            # Extract the unified response and activity metadata
            unified_response = retrieval_result.response[0].content[0].text
            activity_data = [activity.as_dict() for activity in retrieval_result.activity] if retrieval_result.activity else []
            references_data = [ref.as_dict() for ref in retrieval_result.references] if retrieval_result.references else []
            
            logger.info(f"🤖 Agentic Vector RAG: Processing unified response and {len(references_data)} references")
            logger.info(f"🔍 DEBUG: Unified response preview: {unified_response[:200]}...")
            logger.info(f"🔍 DEBUG: References data sample: {references_data[:2] if references_data else 'No references'}")
            
            # Parse citations from the response text if it contains JSON data
            citations_from_response = self._parse_citations_from_response(unified_response)
            logger.info(f"🔍 DEBUG: Citations from response: {len(citations_from_response)} found")
            if citations_from_response:
                logger.info(f"🔍 DEBUG: First citation from response: {citations_from_response[0]}")
            
            # Format citations from references (the traditional way) 
            citations_from_references = await self._format_citations_from_references(
                references_data, credibility_check_enabled, token_tracker, tracking_id
            )
            logger.info(f"🔍 DEBUG: Citations from references: {len(citations_from_references)} found")
            if citations_from_references:
                logger.info(f"🔍 DEBUG: First citation from references: {citations_from_references[0]}")
            
            # Use citations from response if available, otherwise use references
            citations = citations_from_response if citations_from_response else citations_from_references
            #logger.info(f"🔍 DEBUG: Final citations count: {len(citations)}")
            if citations:
                logger.info(f"🔍 DEBUG: Final citation structure: {citations[0]}")
            else:
                logger.warning("🔍 DEBUG: No citations found from either source!")
            
            # If the response contains JSON citation data, we need to generate an answer from the citations
            if citations_from_response:
                actual_answer = self._generate_answer_from_citations(citations_from_response, question)
                logger.info("Generated answer from citation data since response was only JSON")
            else:
                actual_answer = unified_response
            
            # Extract token usage from activity data
            token_usage = self._extract_token_usage_from_activity(activity_data)
            
            # Update token tracking if provided
            if token_tracker and tracking_id and token_usage:
                await token_tracker.update_usage(
                    tracking_id=tracking_id,
                    prompt_tokens=token_usage.get("prompt_tokens", 0),
                    completion_tokens=token_usage.get("completion_tokens", 0),
                    model_name=self._extract_deployment_name(model_config.get("chat_model", "unknown"))
                )
            
            # Calculate confidence based on retrieval quality
            confidence_score = self._calculate_confidence_score(actual_answer, references_data, activity_data)
            
            # Calculate verification details if credibility checking is enabled
            verification_details = self._calculate_verification_details(citations, credibility_check_enabled, verification_level)
            
            processing_time = time.time() - start_time
            logger.info(f"Agentic Vector RAG processing completed in {processing_time:.2f}s")
            
            return {
                "answer": actual_answer,
                "confidence_score": confidence_score,
                "sources": references_data,  # Raw reference data
                "citations": citations,
                "verification_details": verification_details,
                "metadata": {
                    "rag_method": "agentic-vector",
                    "sources_found": len(references_data),
                    "processing_time": processing_time,
                    "model_used": model_config.get("chat_model", "unknown"),
                    "tokens_used": token_usage,
                    "query_plan_activities": activity_data,
                    "subqueries_executed": len([a for a in activity_data if a.get("type") == "AzureSearchQuery"]),
                    "semantic_ranking_performed": any(a.get("type") == "AzureSearchSemanticRanker" for a in activity_data),
                    "debug_citations_count": len(citations),  # Debug info
                    "debug_references_count": len(references_data)  # Debug info
                }
            }
            
        except Exception as e:
            logger.error(f"Error in Agentic Vector RAG processing: {e}")
            raise

    def _build_conversation_messages(
        self, 
        question: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Build conversation messages for context-aware query planning.
        The LLM uses this history to understand the information need.
        """
        messages = []
        
        # Add system message with financial domain context
        messages.append({
            "role": "system",
            "content": """You are a financial analyst assistant. Analyze questions about financial data, 
            SEC documents, company performance, and market analysis. Break down complex questions into 
            focused subqueries that can effectively search financial documents and data."""
        })
        
        # Add conversation history if available in context
        if context and "conversation_history" in context:
            for msg in context["conversation_history"][-5:]:  # Last 5 messages for context
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(msg)
        
        # Add current user question
        messages.append({
            "role": "user", 
            "content": question
        })
        
        return messages

    async def _perform_agentic_retrieval(
        self,
        messages: List[Dict[str, str]],
        verification_level: VerificationLevel,
        token_tracker: Optional[TokenUsageTracker] = None,
        tracking_id: Optional[str] = None
    ):
        """
        Perform agentic retrieval using the Knowledge Agent.
        This includes query planning, parallel subquery execution, and result merging.
        """
        try:
            # Convert messages to KnowledgeAgentMessage format
            agent_messages = []
            for msg in messages:
                if msg["role"] != "system":  # Skip system messages for retrieval
                    agent_messages.append(
                        KnowledgeAgentMessage(
                            role=msg["role"],
                            content=[KnowledgeAgentMessageTextContent(text=msg["content"])]
                        )
                    )
            
            # Set reranker threshold based on verification level
            reranker_threshold = {
                VerificationLevel.BASIC: 1.5,
                VerificationLevel.THOROUGH: 2.0,
                VerificationLevel.COMPREHENSIVE: 2.5
            }.get(verification_level, 2.0)
            
            # Create retrieval request
            retrieval_request = KnowledgeAgentRetrievalRequest(
                messages=agent_messages,
                target_index_params=[
                    KnowledgeAgentIndexParams(
                        index_name=settings.AZURE_AI_SEARCH_INDEX_NAME,
                        reranker_threshold=reranker_threshold
                    )
                ]
            )
            
            # Perform agentic retrieval
            retrieval_result = self.knowledge_agent_client.retrieve(
                retrieval_request=retrieval_request
            )
            
            return retrieval_result
            
        except Exception as e:
            logger.error(f"Error in agentic retrieval: {e}")
            raise

    async def _format_citations_from_references(self, references_data: List[Dict[str, Any]], 
                                              credibility_check_enabled: bool = False,
                                              token_tracker: Optional[TokenUsageTracker] = None,
                                              tracking_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Format citations from agentic retrieval references.
        References contain metadata about source documents that contributed to the response.
        Expected format: {"ref_id": 0, "title": "Apple Inc. - 10-K", "terms": "...", "content": "..."}
        """
        logger.info(f"🔍 DEBUG: _format_citations_from_references called with {len(references_data)} references")
        if references_data:
            logger.info(f"🔍 DEBUG: First reference structure: {references_data[0]}")
        
        citations = []
        
        for i, ref in enumerate(references_data, 1):
            #logger.info(f"🔍 DEBUG: Processing reference {i}: {ref}")
            # Extract information from the new reference format
            ref_id = ref.get("ref_id", i)
            title = ref.get("title", f"Document {i}")
            content = ref.get("content", "")
            terms = ref.get("terms", "")
            
            # Parse terms to extract meaningful metadata
            page_number = None
            section_info = None
            document_type = None
            company_name = None
            
            if terms:
                # Extract company name and document type from terms
                term_parts = terms.split()
                if "10-K" in terms:
                    document_type = "10-K Filing"
                elif "10-Q" in terms:
                    document_type = "10-Q Filing"
                elif "8-K" in terms:
                    document_type = "8-K Filing"
                
                # Look for company indicators
                for term in term_parts:
                    if term in ["AAPL", "Apple", "Inc."]:
                        company_name = "Apple Inc."
                        break
                    elif "Corp" in term or "Inc" in term or "LLC" in term:
                        company_name = term
                        break
            
            # Try to extract page number from content if available
            if "page" in content.lower():
                import re
                page_match = re.search(r'page\s+(\d+)', content.lower())
                if page_match:
                    page_number = int(page_match.group(1))
            
            # Create a more descriptive document ID
            doc_id = f"ref_{ref_id}"
            if company_name and document_type:
                doc_id = f"{company_name}_{document_type}_{ref_id}"
            
            # Assess credibility if enabled
            if credibility_check_enabled:
                try:
                    # Create a mock document for credibility assessment
                    mock_doc = {
                        'content': content,
                        'company': company_name,
                        'form_type': document_type,
                        'document_title': title,
                        'source': title or f"Document {ref_id}"
                    }
                    assessed_credibility = await self.credibility_assessor.assess_credibility(
                        processed_doc=mock_doc,
                        source=title or f"Document {ref_id}",
                        token_tracker=token_tracker,
                        tracking_id=tracking_id
                    )
                    final_credibility = assessed_credibility
                    logger.info(f"Agentic Vector RAG: Assessed credibility for reference {i}: {final_credibility:.3f}")
                except Exception as e:
                    logger.warning(f"Agentic Vector RAG: Credibility assessment failed for reference {i}: {e}")
                    final_credibility = 0.9  # Default high credibility for agentic ranking
            else:
                final_credibility = 0.9  # Higher credibility due to agentic ranking
            
            citation = {
                "id": f"agentic_citation_{i}",
                "content": content,  # Truncate long content
                "source": title or f"Document {ref_id}",
                "document_id": doc_id,  # snake_case for Citation schema
                "document_title": title,  # snake_case for Citation schema
                "page_number": page_number,  # snake_case for Citation schema
                "section_title": document_type or section_info,  # snake_case for Citation schema
                "confidence": "high",  # Agentic retrieval uses semantic ranking
                "url": "",
                "credibility_score": final_credibility,
                "company": company_name,
                "document_type": document_type,
                "terms": terms,
                "ref_id": ref_id
            }
            citations.append(citation)
            #logger.info(f"🔍 DEBUG: Created citation from reference: {citation}")
        
        logger.info(f"🔍 DEBUG: Returning {len(citations)} citations from references")
        return citations

    def _parse_citations_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse citations from the response text if it contains JSON citation data.
        Expected format: Response starts with JSON array like [{"ref_id":0,"title":"...","content":"..."}]
        """
        logger.info(f"🔍 DEBUG: _parse_citations_from_response called with response length: {len(response_text)}")
        logger.info(f"🔍 DEBUG: Response starts with: '{response_text[:100]}...'")
        
        try:
            # Check if the response starts with JSON array
            response_stripped = response_text.strip()
            if response_stripped.startswith('['):
                logger.info("🔍 DEBUG: Response starts with '[' - attempting JSON parsing")
                # Try to parse the JSON, handling potentially truncated JSON
                json_text = response_stripped
                
                # If it doesn't end with ], try to complete it
                if not json_text.endswith(']'):
                    logger.info("🔍 DEBUG: JSON doesn't end with ']' - attempting to complete it")
                    # Find the last complete object
                    last_brace = json_text.rfind('}')
                    if last_brace > 0:
                        json_text = json_text[:last_brace + 1] + ']'
                        logger.info(f"🔍 DEBUG: Completed JSON to: '{json_text[:200]}...'")
                
                try:
                    citation_data = json.loads(json_text)
                    logger.info(f"🔍 DEBUG: Successfully parsed JSON with {len(citation_data) if isinstance(citation_data, list) else 'non-list'} items")
                except json.JSONDecodeError as e:
                    # If still failing, try to find complete JSON objects
                    logger.info(f"🔍 DEBUG: JSON parsing failed: {e} - trying to extract complete objects")
                    return self._extract_complete_citation_objects(response_stripped)
                
                if isinstance(citation_data, list):
                    logger.info(f"🔍 DEBUG: Processing {len(citation_data)} citation items from JSON")
                    citations = []
                    for i, item in enumerate(citation_data, 1):
                        #logger.info(f"🔍 DEBUG: Processing citation item {i}: {item}")
                        citation = {
                            "id": f"agentic_citation_{i}",
                            "content": item.get("content", "")[:500] + "..." if len(item.get("content", "")) > 500 else item.get("content", ""),
                            "source": item.get("title", f"Document {i}"),
                            "document_id": f"ref_{item.get('ref_id', i)}",  # snake_case for Citation schema
                            "document_title": item.get("title", ""),  # snake_case for Citation schema
                            "page_number": None,  # Not available in this format
                            "section_title": self._extract_document_type_from_title(item.get("title", "")),  # snake_case for Citation schema
                            "confidence": "high",
                            "url": "",
                            "credibility_score": 0.9,  # snake_case for Citation schema
                            "terms": item.get("terms", ""),
                            "ref_id": item.get("ref_id", i)
                        }
                        citations.append(citation)
                        #logger.info(f"🔍 DEBUG: Created citation: {citation}")
                    
                    logger.info(f"🔍 DEBUG: Parsed {len(citations)} citations from response JSON")
                    return citations
            else:
                logger.info("🔍 DEBUG: Response does not start with '[' - no JSON citations found")
            
            return []  # No JSON citations found
            
        except Exception as e:
            logger.error(f"🔍 DEBUG: Exception in _parse_citations_from_response: {e}")
            return []

    def _extract_complete_citation_objects(self, json_text: str) -> List[Dict[str, Any]]:
        """Extract complete citation objects from potentially truncated JSON"""
        try:
            # Use regex to find complete JSON objects
            import re
            pattern = r'\{"ref_id":\d+,"title":"[^"]*","terms":"[^"]*","content":"[^"]*"\}'
            matches = re.findall(pattern, json_text)
            
            citations = []
            for i, match in enumerate(matches, 1):
                try:
                    obj = json.loads(match)
                    citation = {
                        "id": f"agentic_citation_{i}",
                        "content": obj.get("content", "")[:500] + "..." if len(obj.get("content", "")) > 500 else obj.get("content", ""),
                        "source": obj.get("title", f"Document {i}"),
                        "document_id": f"ref_{obj.get('ref_id', i)}",  # snake_case for Citation schema
                        "document_title": obj.get("title", ""),  # snake_case for Citation schema
                        "page_number": None,  # snake_case for Citation schema
                        "section_title": self._extract_document_type_from_title(obj.get("title", "")),  # snake_case for Citation schema
                        "confidence": "high",
                        "url": "",
                        "credibility_score": 0.9,  # snake_case for Citation schema
                        "terms": obj.get("terms", ""),
                        "ref_id": obj.get("ref_id", i)
                    }
                    citations.append(citation)
                except json.JSONDecodeError:
                    continue
            
            if citations:
                logger.info(f"Extracted {len(citations)} complete citation objects from truncated JSON")
            return citations
            
        except Exception as e:
            logger.debug(f"Could not extract citation objects: {e}")
            return []

    def _generate_answer_from_citations(self, citations: List[Dict[str, Any]], question: str) -> str:
        """
        Generate a comprehensive answer from citation data when the response is only JSON.
        """
        if not citations:
            return "I found relevant documents but couldn't extract specific information to answer your question."
        
        # Extract key information from citations
        companies = set()
        document_types = set()
        content_snippets = []
        
        for citation in citations:
            # Extract company information
            terms = citation.get("terms", "")
            if "Apple" in terms or "AAPL" in terms:
                companies.add("Apple Inc.")
            elif "MSFT" in terms or "Microsoft" in terms:
                companies.add("Microsoft")
            
            # Extract document types
            if citation.get("section_title"):
                document_types.add(citation["section_title"])
            
            # Get full content snippets (no truncation for comprehensive answer)
            content = citation.get("content", "")
            if content and len(content) > 50:
                content_snippets.append(content)  # Keep full content for comprehensive answer
        
        # Use Azure OpenAI to generate a comprehensive answer from the citation content
        try:
            from openai import AsyncAzureOpenAI
            from app.core.config import settings
            import asyncio
            
            # Prepare context from all citations
            context = "\n\n".join([f"Source {i+1}: {snippet}" for i, snippet in enumerate(content_snippets)])
            
            system_prompt = f"""You are a financial analyst AI assistant. Based on the provided financial document excerpts, generate a comprehensive answer to the user's question.

Question: {question}
Companies mentioned: {', '.join(companies) if companies else 'Various'}
Document types: {', '.join(document_types) if document_types else 'Financial documents'}

Context from financial documents:
{context}

Instructions:
1. Provide a detailed, comprehensive answer (500-1000 words when appropriate)
2. Use specific information from the provided context
3. Structure your response clearly with headings when relevant
4. Include specific financial details and metrics mentioned in the documents
5. Maintain a professional, analytical tone
6. Cite specific information from the sources when relevant

Generate a detailed financial analysis response:"""

            async def generate_response():
                client = AsyncAzureOpenAI(
                    api_key=settings.AZURE_OPENAI_API_KEY,
                    api_version=settings.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT
                )
                
                response = await client.chat.completions.create(
                    model=settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question}
                    ],
                    max_tokens=2000,
                    temperature=0.3,
                )
                
                if response.choices and response.choices[0].message.content:
                    return response.choices[0].message.content.strip()
                return None
            
            # Run the async function
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we need to use a different approach
                # Fall back to template generation
                pass
            else:
                comprehensive_answer = loop.run_until_complete(generate_response())
                if comprehensive_answer:
                    return comprehensive_answer
        
        except Exception as e:
            logger.warning(f"Could not generate comprehensive answer with Azure OpenAI: {e}")
            # Fall back to template generation
        
        # Fallback: Generate a comprehensive template answer
        answer_parts = []
        
        if "risk" in question.lower():
            answer_parts.append("Based on the financial documents, here are the key financial risks identified:")
        elif "financial" in question.lower():
            answer_parts.append("Based on the financial filings, here are the key financial details:")
        else:
            answer_parts.append("Based on the available documents, here is the relevant information:")
        
        # Add ALL content from citations (not just 3 snippets)
        for i, snippet in enumerate(content_snippets, 1):
            answer_parts.append(f"\n\n{i}. {snippet}")  # Include full content
        
        # Add document source information
        if document_types:
            doc_list = ", ".join(document_types)
            answer_parts.append(f"\n\nThis information is sourced from {doc_list}")
            if companies:
                company_list = ", ".join(companies)
                answer_parts.append(f" for {company_list}.")
        
        return "".join(answer_parts)

    def _extract_answer_from_response(self, response_text: str) -> str:
        """
        Extract the actual answer from response text that may contain JSON citation data.
        If response starts with JSON, extract the text after the JSON array.
        """
        try:
            # Check if the response starts with JSON array
            if response_text.strip().startswith('['):
                # Find the end of the JSON array
                bracket_count = 0
                json_end = 0
                for i, char in enumerate(response_text):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > 0 and json_end < len(response_text):
                    # Extract text after the JSON
                    answer_part = response_text[json_end:].strip()
                    if answer_part:
                        logger.info("Extracted answer text from response after JSON citations")
                        return answer_part
            
            # If no JSON found or no text after JSON, return the full response
            return response_text
            
        except Exception as e:
            logger.debug(f"Could not extract answer from response: {e}")
            return response_text

    def _extract_document_type_from_title(self, title: str) -> str:
        """Extract document type from title"""
        if "10-K" in title:
            return "10-K Filing"
        elif "10-Q" in title:
            return "10-Q Filing"
        elif "8-K" in title:
            return "8-K Filing"
        elif "Proxy" in title:
            return "Proxy Statement"
        else:
            return "SEC Document"

    def _extract_token_usage_from_activity(self, activity_data: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Extract token usage information from agentic retrieval activity data.
        """
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }
        
        for activity in activity_data:
            activity_type = activity.get("type", "")
            
            if activity_type == "ModelQueryPlanning":
                # Query planning tokens
                token_usage["prompt_tokens"] += activity.get("input_tokens", 0)
                token_usage["completion_tokens"] += activity.get("output_tokens", 0)
            elif activity_type == "AzureSearchSemanticRanker":
                # Semantic ranking input tokens
                token_usage["prompt_tokens"] += activity.get("input_tokens", 0)
        
        token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]
        
        return token_usage

    def _calculate_confidence_score(
        self, 
        unified_response: str, 
        references_data: List[Dict[str, Any]], 
        activity_data: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate confidence score based on agentic retrieval quality indicators.
        """
        base_confidence = 0.5
        
        # Boost confidence based on response quality
        if unified_response and len(unified_response) > 100:
            base_confidence += 0.2
        
        # Boost confidence based on number of references
        if len(references_data) > 0:
            base_confidence += min(0.2, len(references_data) * 0.05)
        
        # Boost confidence if semantic ranking was performed
        if any(a.get("type") == "AzureSearchSemanticRanker" for a in activity_data):
            base_confidence += 0.1
        
        # Boost confidence based on successful subqueries
        successful_queries = len([a for a in activity_data if a.get("type") == "AzureSearchQuery" and a.get("count", 0) > 0])
        if successful_queries > 0:
            base_confidence += min(0.2, successful_queries * 0.1)
        
        return min(0.95, base_confidence)  # Cap at 95%

    def _extract_deployment_name(self, model_config_value: str) -> str:
        """Extract deployment name from model config value, handling combined strings like 'gpt-4o (chat4o)'"""
        if not model_config_value:
            return settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME or "chat4o"
        
        # Handle combined strings like "gpt-4o (chat4o)" -> extract "chat4o"
        if '(' in model_config_value and ')' in model_config_value:
            try:
                deployment_name = model_config_value.split('(')[1].split(')')[0].strip()
                logger.debug(f"Extracted deployment name '{deployment_name}' from combined string '{model_config_value}'")
                return deployment_name
            except (IndexError, AttributeError):
                logger.warning(f"Failed to parse deployment name from '{model_config_value}', using as-is")
                return model_config_value
        
        # Return as-is if no parentheses (already just deployment name)
        return model_config_value

    async def diagnose_knowledge_agent(self) -> Dict[str, Any]:
        """
        Diagnostic method to check the state of the knowledge agent and its capabilities.
        """
        try:
            if not self.knowledge_agent_client and not self.index_client:
                await self.initialize()
            
            # Check if knowledge agent is available
            if not self.knowledge_agent_client:
                return {
                    "agent_name": self.agent_name,
                    "agent_status": "not_available",
                    "fallback_mode": True,
                    "message": "Knowledge agent not available, service will use traditional RAG fallback",
                    "capabilities": ["traditional_search", "citation_extraction", "confidence_scoring"]
                }
            
            # Test with a simple query
            test_messages = [
                KnowledgeAgentMessage(
                    role="user",
                    content=[KnowledgeAgentMessageTextContent(text="Test query for financial data")]
                )
            ]
            
            test_request = KnowledgeAgentRetrievalRequest(
                messages=test_messages,
                target_index_params=[
                    KnowledgeAgentIndexParams(
                        index_name=settings.AZURE_AI_SEARCH_INDEX_NAME,
                        reranker_threshold=1.0
                    )
                ]
            )
            
            test_result = self.knowledge_agent_client.retrieve(retrieval_request=test_request)
            
            return {
                "agent_name": self.agent_name,
                "agent_status": "operational",
                "fallback_mode": False,
                "test_query_successful": test_result is not None,
                "activity_types_available": [a.type for a in test_result.activity] if test_result and test_result.activity else [],
                "references_returned": len(test_result.references) if test_result and test_result.references else 0,
                "response_available": test_result is not None and test_result.response is not None,
                "capabilities": [
                    "intelligent_query_planning",
                    "parallel_subquery_execution", 
                    "semantic_ranking",
                    "result_merging",
                    "citation_extraction",
                    "confidence_scoring"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error diagnosing knowledge agent: {e}")
            return {
                "agent_name": self.agent_name,
                "agent_status": "error",
                "error": str(e),
                "fallback_available": True
            }
    
    async def _fallback_process_question(
        self,
        question: str,
        session_id: str,
        model_config: Dict[str, Any],
        verification_level: VerificationLevel,
        token_tracker: Optional[TokenUsageTracker] = None,
        tracking_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fallback processing when knowledge agent is not available.
        Uses traditional search with enhanced processing to simulate agentic behavior.
        """
        start_time = time.time()
        
        try:
            logger.info("Processing question using Agentic Vector RAG fallback mode")
            
            # Use the traditional RAG service as fallback
            from app.services.traditional_rag_service import TraditionalRAGService
            
            # Initialize traditional RAG service
            traditional_rag = TraditionalRAGService(self.azure_manager)
            await traditional_rag.initialize()
            
            # Process using traditional RAG
            result = await traditional_rag.process_question(
                question=question,
                session_id=session_id,
                model_config=model_config,
                verification_level=verification_level,
                token_tracker=token_tracker,
                tracking_id=tracking_id,
                context=context
            )
            
            # Enhance the result to indicate it came from agentic-vector (fallback)
            result["metadata"]["rag_method"] = "agentic-vector"
            result["metadata"]["fallback_mode"] = True
            result["metadata"]["fallback_reason"] = "Knowledge agent not available"
            
            processing_time = time.time() - start_time
            result["metadata"]["processing_time"] = processing_time
            
            logger.info(f"Agentic Vector RAG fallback processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in Agentic Vector RAG fallback processing: {e}")
            processing_time = time.time() - start_time
            
            return {
                "answer": "I apologize, but I encountered an error while processing your question using the agentic vector retrieval method.",
                "confidence_score": 0.1,
                "sources": [],
                "citations": [],
                "metadata": {
                    "rag_method": "agentic-vector",
                    "fallback_mode": True,
                    "processing_time": processing_time,
                    "error": str(e),
                    "tokens_used": {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0}
                }
            }
    
    async def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information about the Agentic Vector RAG service"""
        try:
            # Get knowledge agent diagnostics
            agent_diagnostics = await self.diagnose_knowledge_agent()
            
            diagnostics = {
                "service_name": "Agentic Vector RAG",
                "status": "operational",
                "agent_name": self.agent_name,
                "index_name": settings.AZURE_AI_SEARCH_INDEX_NAME,
                "endpoint": settings.AZURE_AI_SEARCH_ENDPOINT,
                "model_config": {
                    "deployment_name": settings.AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
                    "model_name": settings.AZURE_OPENAI_CHAT_MODEL_NAME
                },
                "agent_details": agent_diagnostics,
                "authentication": {
                    "type": "API Key" if settings.AZURE_SEARCH_API_KEY else "DefaultAzureCredential",
                    "api_version": "2025-05-01-preview"
                }
            }
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Error getting diagnostics: {e}")
            return {
                "service_name": "Agentic Vector RAG",
                "status": "error",
                "error": str(e)
            }
    
    def _calculate_verification_details(self, citations: List[Dict[str, Any]], 
                                       credibility_check_enabled: bool, 
                                       verification_level: VerificationLevel) -> Dict[str, Any]:
        """Calculate verification details based on citations and credibility scores"""
        
        if not credibility_check_enabled or not citations:
            return {
                "overall_credibility_score": 0.5,
                "verified_sources_count": 0,
                "total_sources_count": len(citations),
                "verification_summary": "Credibility checking disabled or no sources available",
                "verification_level": verification_level.value if hasattr(verification_level, 'value') else str(verification_level)
            }
        
        # Calculate overall credibility score
        credibility_scores = [citation.get('credibility_score', 0.5) for citation in citations]
        overall_credibility = sum(credibility_scores) / len(credibility_scores) if credibility_scores else 0.5
        
        # Count verified sources (credibility >= 0.6)
        verified_sources = [c for c in citations if c.get('credibility_score', 0) >= 0.6]
        verified_count = len(verified_sources)
        total_count = len(citations)
        
        # Generate verification summary
        if verified_count == total_count:
            verification_summary = f"All {total_count} sources verified as credible (agentic ranking)"
        elif verified_count > 0:
            verification_summary = f"{verified_count} of {total_count} sources verified as credible (agentic ranking)"
        else:
            verification_summary = f"None of {total_count} sources met credibility threshold"
        
        logger.info(f"Agentic Vector RAG verification: {verified_count}/{total_count} sources verified, "
                   f"overall credibility: {overall_credibility:.3f}")
        
        return {
            "overall_credibility_score": overall_credibility,
            "verified_sources_count": verified_count,
            "total_sources_count": total_count,
            "verification_summary": verification_summary,
            "verification_level": verification_level.value if hasattr(verification_level, 'value') else str(verification_level),
            "credibility_method": "agentic_vector_rag"
        }

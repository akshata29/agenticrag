import React, { useState, useEffect, useRef } from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from '@/components/ui/resizable';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { QuestionInput } from './QuestionInput';
import { AnswerDisplay } from './AnswerDisplay';
import { SourceVerification } from './SourceVerificationModal';
import { QuestionDecomposition } from './QuestionDecomposition';
import { AgentServiceStatus } from './AgentServiceStatus';
import { PerformanceDashboard } from './PerformanceDashboard';
import { ReasoningChainDisplay } from './ReasoningChainDisplay';
import { EvaluationModal } from './EvaluationModal';
import { ModelSettings } from '../shared/ModelConfiguration';
import { apiService } from '../../services/api';

export interface QAQuestion {
  id: string;
  question: string;
  timestamp: Date;
  verificationLevel: 'basic' | 'thorough' | 'comprehensive';
  backendQuestionId?: string; // Store the backend's question_id once we get the response
}

export interface QAAnswer {
  id: string;
  questionId: string;
  answer: string;
  confidenceScore: number;
  timestamp: Date;
  citations: QACitation[];
  subQuestions: string[];
  verificationDetails: {
    overallCredibilityScore: number;
    verifiedSourcesCount: number;
    totalSourcesCount: number;
    verificationSummary: string;
  };
  performanceBenchmark?: {
    question_id: string;
    question: string;
    complexity_score: number;
    estimated_manual_time: number;
    ai_processing_time: number;
    efficiency_gain: number;
    source_count: number;
    accuracy_score: number;
    confidence_score: number;
    verification_level: string;
    session_id: string;
    timestamp: string;
  };
  reasoningChain?: {
    question_id: string;
    question: string;
    reasoning_steps: Array<{
      step_number: number;
      description: string;
      action_type: string;
      sources_consulted: string[];
      confidence: number;
      duration_ms: number;
      output: string;
      metadata: { [key: string]: any };
    }>;
    total_duration_ms: number;
    final_confidence: number;
    session_id: string;
    timestamp: string;
  };
  metadata: {
    model?: string;
    tokens?: number;
    responseTime?: number;
    verificationLevel?: string;
    agentServiceUsed?: boolean;
    agentId?: string;
    threadId?: string;
    rag_method?: string;
    evaluation_enabled?: boolean;
    evaluation_id?: string;
    evaluator_type?: string;
    evaluation_model?: string;
  };
}

export interface QACitation {
  id: string;
  content: string;
  source: string;
  documentId: string;
  documentTitle: string;
  pageNumber?: number;
  sectionTitle?: string;
  confidence: 'high' | 'medium' | 'low';
  url?: string;
  credibilityScore: number;
}

export interface VerifiedSource {
  sourceId: string;
  url: string;
  title: string;
  content: string;
  credibilityScore: number;
  credibilityExplanation: string;
  trustIndicators: string[];
  redFlags: string[];
  verificationStatus: 'verified' | 'questionable' | 'unverified';
}

interface QAContainerProps {
  modelSettings: ModelSettings;
}

export const QAContainer: React.FC<QAContainerProps> = ({ modelSettings }) => {
  const [questions, setQuestions] = useState<QAQuestion[]>([]);
  const [answers, setAnswers] = useState<QAAnswer[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isVerifyingSourcesForAnswer, setIsVerifyingSourcesForAnswer] = useState<string | null>(null);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const [showSourceVerificationModal, setShowSourceVerificationModal] = useState(false);
  const [showQuestionDecomposition, setShowQuestionDecomposition] = useState(false);
  const [verifiedSources, setVerifiedSources] = useState<VerifiedSource[]>([]);
  const [decomposedQuestions, setDecomposedQuestions] = useState<{
    originalQuestion: string;
    subQuestions: string[];
    reasoning: string;
  } | null>(null);
  const [showAgentStatus, setShowAgentStatus] = useState(false);
  const [agentServiceConnected, setAgentServiceConnected] = useState(false);
  const [credibilityCheckEnabled, setCredibilityCheckEnabled] = useState(false); // Off by default
  const [ragMethod, setRagMethod] = useState<'agent' | 'traditional' | 'llamaindex' | 'agentic-vector' | 'mcp'>('agent'); // Default to current implementation
  const [evaluationEnabled, setEvaluationEnabled] = useState(false); // Off by default
  const [evaluatorType, setEvaluatorType] = useState<'foundry' | 'custom'>('custom'); // Default to custom
  const [evaluationModel, setEvaluationModel] = useState<string>('o3-mini'); // Default evaluation model
  const [showEvaluationModal, setShowEvaluationModal] = useState(false);
  const [selectedEvaluationId, setSelectedEvaluationId] = useState<string | null>(null);
  const [evaluationResults, setEvaluationResults] = useState<{[key: string]: any}>({});
  const [availableEvaluationModels, setAvailableEvaluationModels] = useState<Array<{id: string, name: string, provider: string, model_name?: string}>>([
    { id: 'o3-mini', name: 'o3-mini', provider: 'Custom' },
    { id: 'gpt-4o', name: 'gpt-4o', provider: 'Custom' },
    { id: 'gpt-4o-mini', name: 'gpt-4o-mini', provider: 'Custom' },
    { id: 'gpt-4-turbo', name: 'gpt-4-turbo', provider: 'Custom' },
    { id: 'gpt-4', name: 'gpt-4', provider: 'Custom' }
  ]);
  const [isLoadingEvaluationModels, setIsLoadingEvaluationModels] = useState(false);
  const [availableEvaluators, setAvailableEvaluators] = useState<{foundry_available: boolean, custom_available: boolean} | null>(null);
  const [showReasoningChain, setShowReasoningChain] = useState(false);
  const [sessionMetrics, setSessionMetrics] = useState<any>(null);
  const [selectedReasoningChain, setSelectedReasoningChain] = useState<any>(null);
  const [showOverallPerformance, setShowOverallPerformance] = useState(false);
  const [selectedQuestionPerformance, setSelectedQuestionPerformance] = useState<any>(null);
  const [showQuestionPerformance, setShowQuestionPerformance] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [questions, answers]);

  useEffect(() => {
    const sessionId = localStorage.getItem('qaSessionId') || generateSessionId();
    setCurrentSessionId(sessionId);
    localStorage.setItem('qaSessionId', sessionId);
    
    checkAgentServiceStatus();
    checkAvailableEvaluators();
  }, []);

  useEffect(() => {
    // Fetch evaluation models when evaluator type changes to foundry
    if (evaluatorType === 'foundry') {
      fetchEvaluationModels();
    } else {
      // Reset to default custom models
      setAvailableEvaluationModels([
        { id: 'o3-mini', name: 'o3-mini', provider: 'Custom' },
        { id: 'gpt-4o', name: 'gpt-4o', provider: 'Custom' },
        { id: 'gpt-4o-mini', name: 'gpt-4o-mini', provider: 'Custom' },
        { id: 'gpt-4-turbo', name: 'gpt-4-turbo', provider: 'Custom' },
        { id: 'gpt-4', name: 'gpt-4', provider: 'Custom' }
      ]);
      // Set default model for custom
      if (!['o3-mini', 'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4'].includes(evaluationModel)) {
        setEvaluationModel('o3-mini');
      }
    }
  }, [evaluatorType]);

  const fetchEvaluationModels = async () => {
    setIsLoadingEvaluationModels(true);
    try {
      if (evaluatorType === 'foundry') {
        // Check if foundry is available
        if (!availableEvaluators || !availableEvaluators.foundry_available) {
          console.warn('Foundry evaluator not available, switching to custom');
          setEvaluatorType('custom');
          setAvailableEvaluationModels([
            { id: 'o3-mini', name: 'o3-mini', provider: 'Custom', model_name: 'o3-mini' },
            { id: 'gpt-4o', name: 'gpt-4o', provider: 'Custom', model_name: 'gpt-4o' },
            { id: 'gpt-4o-mini', name: 'gpt-4o-mini', provider: 'Custom', model_name: 'gpt-4o-mini' },
            { id: 'gpt-4-turbo', name: 'gpt-4-turbo', provider: 'Custom', model_name: 'gpt-4-turbo' },
            { id: 'gpt-4', name: 'gpt-4', provider: 'Custom', model_name: 'gpt-4' }
          ]);
          setEvaluationModel('o3-mini');
          return;
        }
        
        // Fetch foundry models (deployment names)
        const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || '/api/v1';
        const response = await fetch(`${apiBaseUrl}/admin/foundry/models`);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch models: ${response.statusText}`);
        }
        
        const data = await response.json();
        const foundryModels = data.models || [];
        
        const evaluationModelsList: Array<{id: string, name: string, provider: string}> = [];
        
        foundryModels.forEach((model: any) => {
          if (model.type === 'chat') {
            evaluationModelsList.push({
              id: model.deployment_name || model.id, // Use deployment name for API calls
              name: model.name, // Display name with deployment name
              provider: 'Azure AI Foundry',
              model_name: model.model_name // Store model name for evaluation
            });
          }
        });
        
        if (evaluationModelsList.length > 0) {
          setAvailableEvaluationModels(evaluationModelsList);
          // Set default to first available foundry model
          const preferredModel = evaluationModelsList.find(m => 
            m.id.includes('o3') || m.id.includes('gpt-4o') || m.id.includes('gpt-4')
          );
          setEvaluationModel(preferredModel ? preferredModel.id : evaluationModelsList[0].id);
        } else {
          // Fallback to custom models if no foundry models available
          setAvailableEvaluationModels([
            { id: 'o3-mini', name: 'o3-mini (Fallback)', provider: 'Custom', model_name: 'o3-mini' },
            { id: 'gpt-4o', name: 'gpt-4o (Fallback)', provider: 'Custom', model_name: 'gpt-4o' }
          ]);
          setEvaluationModel('o3-mini');
        }
      }
      
    } catch (error) {
      console.error('Error fetching evaluation models:', error);
      // Fallback to custom models on error
      setEvaluatorType('custom');
      setAvailableEvaluationModels([
        { id: 'o3-mini', name: 'o3-mini (Fallback)', provider: 'Custom', model_name: 'o3-mini' },
        { id: 'gpt-4o', name: 'gpt-4o (Fallback)', provider: 'Custom', model_name: 'gpt-4o' }
      ]);
      setEvaluationModel('o3-mini');
    } finally {
      setIsLoadingEvaluationModels(false);
    }
  };

  const checkAgentServiceStatus = async () => {
    try {
      await apiService.getQACapabilities();
      setAgentServiceConnected(true);
    } catch (error) {
      console.warn('Agent service check failed:', error);
      setAgentServiceConnected(false);
    }
  };

  const checkAvailableEvaluators = async () => {
    try {
      const evaluators = await apiService.getAvailableEvaluators();
      console.log('Available evaluators response:', evaluators);
      
      // Handle both possible response formats
      const foundryAvailable = (evaluators.available_evaluators as any).foundry_available || 
                              (evaluators.available_evaluators as any).foundry?.available || 
                              false;
      const customAvailable = (evaluators.available_evaluators as any).custom_available || 
                             (evaluators.available_evaluators as any).custom?.available || 
                             false;
      
      setAvailableEvaluators({
        foundry_available: foundryAvailable,
        custom_available: customAvailable
      });
      
      // If user had foundry selected but it's not available, switch to custom
      if (evaluatorType === 'foundry' && !foundryAvailable) {
        console.warn('Foundry evaluator not available, switching to custom');
        setEvaluatorType('custom');
      }
    } catch (error) {
      console.warn('Failed to check available evaluators:', error);
      setAvailableEvaluators({
        foundry_available: false,
        custom_available: true // Assume custom is always available
      });
    }
  };

  const generateSessionId = (): string => {
    return `qa_session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  };

  const handleAskQuestion = async (
    question: string, 
    verificationLevel: 'basic' | 'thorough' | 'comprehensive'
  ) => {
    const qaQuestion: QAQuestion = {
      id: Date.now().toString(),
      question,
      timestamp: new Date(),
      verificationLevel,
    };

    setQuestions(prev => [...prev, qaQuestion]);
    setIsLoading(true);    try {      const selectedModel = availableEvaluationModels.find(m => m.id === evaluationModel);
      // For foundry evaluators, use deployment name (id), for custom use model name
      const modelForEvaluation = evaluatorType === 'foundry' 
        ? evaluationModel  // This is already the deployment name/id
        : (selectedModel?.model_name || evaluationModel);

      const data = await apiService.askQuestion({
        question,
        session_id: currentSessionId,
        verification_level: verificationLevel,
        chat_model: modelSettings.selectedModel,
        embedding_model: modelSettings.embeddingModel,
        temperature: modelSettings.temperature,
        credibility_check_enabled: credibilityCheckEnabled, // Pass the toggle state
        rag_method: ragMethod, // Pass the selected RAG method
        evaluation_enabled: evaluationEnabled, // Pass evaluation toggle state
        evaluator_type: evaluatorType, // Pass evaluator type
        evaluation_model: modelForEvaluation, // Pass evaluation model (model_name for foundry, deployment name for custom)
      });
        const qaAnswer: QAAnswer = {
        id: (Date.now() + 1).toString(),
        questionId: data.question_id, // Use the backend's question_id instead of qaQuestion.id
        answer: data.answer,
        confidenceScore: data.confidence_score || 0.8,
        timestamp: new Date(),
        citations: data.citations || [],
        subQuestions: data.sub_questions || [],
        verificationDetails: {
          overallCredibilityScore: data.verification_details?.overall_credibility_score || 0.5,
          verifiedSourcesCount: data.verification_details?.verified_sources_count || 0,
          totalSourcesCount: data.verification_details?.total_sources_count || 0,
          verificationSummary: data.verification_details?.verification_summary || 'No verification details available',
        },
        performanceBenchmark: data.performance_benchmark,
        reasoningChain: data.reasoning_chain,
        metadata: {
          model: data.verification_details?.chat_model_used || data.metadata?.model_used || modelSettings.selectedModel,
          tokens: data.token_usage?.total_tokens,
          responseTime: data.metadata?.response_time,
          verificationLevel: verificationLevel,
          agentServiceUsed: data.metadata?.agent_service_used || agentServiceConnected,
          agentId: data.verification_details?.agent_id,
          threadId: data.verification_details?.thread_id,
          rag_method: ragMethod, // Add the RAG method used for this answer
          evaluation_enabled: evaluationEnabled,
          evaluation_id: data.metadata?.evaluation_id,
          evaluator_type: data.metadata?.evaluator_type,
          evaluation_model: data.metadata?.evaluation_model,
        },
      };

      setAnswers(prev => [...prev, qaAnswer]);
      
      // Update the question with the backend's question_id for proper matching
      setQuestions(prev => prev.map(q => 
        q.id === qaQuestion.id 
          ? { ...q, backendQuestionId: data.question_id }
          : q
      ));
    } catch (error) {
      console.error('Error processing QA request:', error);
      const errorAnswer: QAAnswer = {
        id: (Date.now() + 1).toString(),
        questionId: qaQuestion.id,
        answer: 'Sorry, I encountered an error processing your question. Please try again.',
        confidenceScore: 0,
        timestamp: new Date(),
        citations: [],
        subQuestions: [],
        verificationDetails: {
          overallCredibilityScore: 0,
          verifiedSourcesCount: 0,
          totalSourcesCount: 0,
          verificationSummary: 'Error occurred during processing',
        },
        metadata: {},
      };
      setAnswers(prev => [...prev, errorAnswer]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDecomposeQuestion = async (question: string) => {
    try {
      const data = await apiService.decomposeQuestion({
        question,
        session_id: currentSessionId,
      });
      setDecomposedQuestions({
        originalQuestion: question,
        subQuestions: data.sub_questions,
        reasoning: data.metadata?.reasoning || 'Question decomposed successfully',
      });
      setShowQuestionDecomposition(true);
    } catch (error) {
      console.error('Error decomposing question:', error);
    }
  };

  const handleVerifySources = async (answer: QAAnswer) => {
    setIsVerifyingSourcesForAnswer(answer.id);
    try {
      const verifiedSources: VerifiedSource[] = [];
      
      for (const citation of answer.citations) {
        const data = await apiService.verifySource({
          source_url: citation.url || citation.source,
          content: citation.content,
          session_id: currentSessionId,
        });
        
        verifiedSources.push({
          sourceId: citation.id,
          url: citation.url || citation.source,
          title: citation.documentTitle,
          content: citation.content,
          credibilityScore: data.credibility_score,
          credibilityExplanation: data.verification_details?.explanation || 'No explanation provided',
          trustIndicators: data.verification_details?.trust_indicators || [],
          redFlags: data.verification_details?.red_flags || [],
          verificationStatus: data.credibility_score > 0.7 ? 'verified' : 
                            data.credibility_score > 0.4 ? 'questionable' : 'unverified',
        });
      }
      
      setVerifiedSources(verifiedSources);
      setShowSourceVerificationModal(true);
    } catch (error) {
      console.error('Error verifying sources:', error);
    } finally {
      setIsVerifyingSourcesForAnswer(null);
    }
  };
  const handleNewSession = () => {
    const newSessionId = generateSessionId();
    setCurrentSessionId(newSessionId);
    localStorage.setItem('qaSessionId', newSessionId);
    setQuestions([]);
    setAnswers([]);
    setShowSourceVerificationModal(false);
    setShowQuestionDecomposition(false);
    setShowAgentStatus(false);
    setShowOverallPerformance(false);
    setShowQuestionPerformance(false);
    setShowReasoningChain(false);
    setVerifiedSources([]);
    setDecomposedQuestions(null);
    setSessionMetrics(null);
    setSelectedReasoningChain(null);
    setSelectedQuestionPerformance(null);
    setIsVerifyingSourcesForAnswer(null);
  };

  const handleToggleAgentStatus = () => {
    setShowAgentStatus(!showAgentStatus);
  };

  const handleRefreshAgentStatus = () => {
    checkAgentServiceStatus();
  };

  const handleShowOverallPerformance = async () => {
    setShowOverallPerformance(!showOverallPerformance);
    setShowQuestionPerformance(false); // Close question-specific if open
    if (!showOverallPerformance) {
      try {
        const metrics = await apiService.getPerformanceMetrics(currentSessionId);
        setSessionMetrics(metrics);
      } catch (error) {
        console.error('Error fetching overall performance metrics:', error);
      }
    }
  };

  const handleShowQuestionPerformance = async (questionId: string) => {
    try {
      const questionMetrics = await apiService.getQuestionPerformanceMetrics(questionId);
      setSelectedQuestionPerformance(questionMetrics);
      setShowQuestionPerformance(true);
      setShowOverallPerformance(false); // Close overall if open
    } catch (error) {
      console.error('Error fetching question performance metrics:', error);
    }
  };

  const handleCloseQuestionPerformance = () => {
    setShowQuestionPerformance(false);
    setSelectedQuestionPerformance(null);
  };

  const handleShowReasoningChain = async (questionId: string) => {
    try {
      console.log('Fetching reasoning chain for question_id:', questionId);
      const reasoningChain = await apiService.getReasoningChain(questionId);
      console.log('Received reasoning chain:', reasoningChain);
      console.log('Reasoning steps:', reasoningChain?.reasoning_steps);
      setSelectedReasoningChain(reasoningChain);
      setShowReasoningChain(true);
    } catch (error) {
      console.error('Error fetching reasoning chain:', error);
    }
  };

  // Evaluation handlers
  const handleShowEvaluation = async (evaluationId: string) => {
    try {
      console.log('handleShowEvaluation called with:', evaluationId, 'type:', typeof evaluationId);
      const result = await apiService.getEvaluationResult(evaluationId);
      setEvaluationResults(prev => ({ ...prev, [evaluationId]: result }));
      setSelectedEvaluationId(evaluationId);
      setShowEvaluationModal(true);
    } catch (error) {
      console.error('Error fetching evaluation result:', error);
    }
  };

  return (
    <div className="flex h-screen bg-background">
      <ResizablePanelGroup direction="horizontal" className="min-h-screen">
        <ResizablePanel defaultSize={75} minSize={50}>
          <div className="flex flex-col h-full">
            <ScrollArea className="flex-1 p-4" ref={scrollAreaRef}>
              <div className="space-y-6">
                {questions.map((question, index) => {
                  // Match answers using either the backend question_id or fallback to frontend id
                  const answer = answers.find(a => 
                    a.questionId === (question.backendQuestionId || question.id)
                  );
                  return (
                    <div key={question.id} className="space-y-4">
                      <div className="bg-muted/50 p-4 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-muted-foreground">
                            Question #{index + 1} • {question.verificationLevel} verification
                          </span>
                          <span className="text-xs text-muted-foreground">
                            {question.timestamp.toLocaleTimeString()}
                          </span>                        </div>
                        <p className="text-foreground text-left">{question.question}</p>
                        <div className="flex gap-2 mt-2">
                          <button
                            onClick={() => handleDecomposeQuestion(question.question)}
                            className="px-2 py-1 text-xs bg-secondary text-secondary-foreground rounded hover:bg-secondary/80"
                          >
                            Decompose
                          </button>
                        </div>
                      </div>                        {answer && (
                        <AnswerDisplay
                          answer={answer}
                          onVerifySources={() => handleVerifySources(answer)}
                          onShowReasoningChain={answer.reasoningChain ? () => handleShowReasoningChain(answer.reasoningChain?.question_id || '') : undefined}
                          onShowPerformance={answer.questionId ? () => handleShowQuestionPerformance(answer.questionId) : undefined}
                          onShowEvaluation={answer.metadata.evaluation_id ? () => handleShowEvaluation(String(answer.metadata.evaluation_id || '')) : undefined}
                          isVerifyingSources={isVerifyingSourcesForAnswer === answer.id}
                          credibilityCheckEnabled={credibilityCheckEnabled}
                          evaluationEnabled={evaluationEnabled}
                          evaluationId={answer.metadata.evaluation_id}
                        />
                      )}
                    </div>
                  );
                })}
                
                {isLoading && (
                  <div className="bg-gradient-to-r from-muted/50 to-muted/30 p-6 rounded-lg border-l-4 border-primary shadow-sm">
                    <div className="flex items-center space-x-4">
                      <div className="animate-spin rounded-full h-6 w-6 border-3 border-primary border-t-transparent"></div>
                      <div className="flex-1">
                        <p className="text-base font-medium text-foreground">Processing your question...</p>
                        <p className="text-sm text-muted-foreground mt-1">
                          Searching through documents, analyzing context, and generating a comprehensive response
                        </p>
                      </div>
                    </div>
                    <div className="mt-4 w-full bg-muted rounded-full h-2">
                      <div className="bg-primary h-2 rounded-full animate-pulse transition-all duration-300" style={{width: '70%'}}></div>
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>
              <div className="border-t p-4">
              <div className="flex items-center gap-2 mb-2">
                <button
                  onClick={handleNewSession}
                  className="px-3 py-1 text-sm bg-primary text-primary-foreground rounded-md hover:bg-primary/90"
                >
                  New QA Session
                </button>                <button
                  onClick={handleToggleAgentStatus}
                  className={`px-3 py-1 text-sm rounded-md hover:opacity-80 ${
                    agentServiceConnected 
                      ? 'bg-green-100 text-green-800 border border-green-200' 
                      : 'bg-red-100 text-red-800 border border-red-200'
                  }`}
                >
                  Agent Status
                </button>
                
                {/* RAG Method Selection */}
                <div className="flex items-center space-x-2">
                  <Label htmlFor="rag-method" className="text-xs text-muted-foreground">
                    RAG Method:
                  </Label>
                  <Select value={ragMethod} onValueChange={(value: 'agent' | 'traditional' | 'llamaindex' | 'agentic-vector' | 'mcp') => setRagMethod(value)}>
                    <SelectTrigger className="w-32 h-7 text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="agent">Agent</SelectItem>
                      <SelectItem value="traditional">Traditional RAG</SelectItem>
                      <SelectItem value="llamaindex">LlamaIndex</SelectItem>
                      <SelectItem value="agentic-vector">Agentic Vector</SelectItem>
                      <SelectItem value="mcp">MCP</SelectItem>
                    </SelectContent>
                  </Select>
                  
                  {/* Overall Performance Button */}
                  <button
                    onClick={handleShowOverallPerformance}
                    className={`px-3 py-1 text-sm rounded-md hover:opacity-80 ${
                      showOverallPerformance 
                        ? 'bg-blue-100 text-blue-800 border border-blue-200' 
                        : 'bg-gray-100 text-gray-800 border border-gray-200'
                    }`}
                    title="Show overall session performance metrics"
                  >
                    📊 Session Performance
                  </button>
                </div>
                
                {/* Credibility Check Toggle */}
                <div className="flex items-center space-x-2 ml-4">
                  <Switch
                    id="credibility-check"
                    checked={credibilityCheckEnabled}
                    onCheckedChange={setCredibilityCheckEnabled}
                  />
                  <Label htmlFor="credibility-check" className="text-xs text-muted-foreground">
                    Credibility Check
                  </Label>
                </div>
                
                {/* Evaluation Toggle */}
                <div className="flex items-center space-x-2 ml-4">
                  <Switch
                    id="evaluation-enabled"
                    checked={evaluationEnabled}
                    onCheckedChange={setEvaluationEnabled}
                  />
                  <Label htmlFor="evaluation-enabled" className="text-xs text-muted-foreground">
                    Evaluation
                  </Label>
                </div>
                
                {/* Evaluator Type Selection */}
                {evaluationEnabled && (
                  <div className="flex items-center space-x-2 ml-4">
                    <Label className="text-xs text-muted-foreground">Type:</Label>
                    <Select value={evaluatorType} onValueChange={(value) => setEvaluatorType(value as 'foundry' | 'custom')}>
                      <SelectTrigger className="w-20 h-6 text-xs border-none shadow-none p-0">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="custom">Custom</SelectItem>
                        {availableEvaluators?.foundry_available && (
                          <SelectItem value="foundry">Foundry</SelectItem>
                        )}
                        {!availableEvaluators?.foundry_available && availableEvaluators !== null && (
                          <SelectItem value="foundry" disabled>
                            <div className="flex items-center space-x-1">
                              <span>Foundry</span>
                              <span className="text-xs text-muted-foreground">(Unavailable)</span>
                            </div>
                          </SelectItem>
                        )}
                      </SelectContent>
                    </Select>
                  </div>
                )}
                
                {/* Evaluation Model Selection */}
                {evaluationEnabled && (
                  <div className="flex items-center space-x-2 ml-4">
                    <Label className="text-xs text-muted-foreground">Model:</Label>
                    <Select value={evaluationModel} onValueChange={setEvaluationModel}>
                      <SelectTrigger className="w-32 h-6 text-xs border-none shadow-none p-0">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {isLoadingEvaluationModels ? (
                          <SelectItem value="loading" disabled>Loading models...</SelectItem>
                        ) : (
                          availableEvaluationModels.map((model) => (
                            <SelectItem key={model.id} value={model.id}>
                              <div className="flex flex-col">
                                <span>{model.name}</span>
                                <span className="text-xs text-muted-foreground">{model.provider}</span>
                              </div>
                            </SelectItem>
                          ))
                        )}
                      </SelectContent>
                    </Select>
                  </div>
                )}
                
                <span className="text-xs text-muted-foreground">{/* separator */}</span>
                <span className="text-xs text-muted-foreground">
                  Session: {currentSessionId.split('_')[2]}
                </span>
                {agentServiceConnected && (
                  <span className="text-xs text-green-600 font-medium">
                    ✓ Azure AI Agent Service
                  </span>
                )}
              </div>
              <QuestionInput
                onAskQuestion={handleAskQuestion}
                disabled={isLoading}
              />
            </div>
          </div>
        </ResizablePanel>
          {(showQuestionDecomposition || showAgentStatus || showOverallPerformance || showQuestionPerformance || showReasoningChain) && (
          <>
            <ResizableHandle />
            <ResizablePanel defaultSize={25} minSize={20} maxSize={40}>
              {showQuestionDecomposition && decomposedQuestions && (
                <QuestionDecomposition
                  decomposition={decomposedQuestions}
                  onClose={() => setShowQuestionDecomposition(false)}
                />
              )}
              
              {showAgentStatus && (
                <div className="p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-medium">Agent Service Status</h3>
                    <button
                      onClick={() => setShowAgentStatus(false)}
                      className="text-xs text-muted-foreground hover:text-foreground"
                    >
                      ✕ Close
                    </button>
                  </div>
                  <AgentServiceStatus 
                    onRefresh={handleRefreshAgentStatus} 
                    isVisible={showAgentStatus}
                  />
                </div>
              )}
              
              {/* Overall Performance Modal */}
              {showOverallPerformance && (
                <div className="p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-medium">📊 Session Performance Analytics</h3>
                    <button
                      onClick={() => setShowOverallPerformance(false)}
                      className="text-xs text-muted-foreground hover:text-foreground"
                    >
                      ✕ Close
                    </button>
                  </div>
                  <PerformanceDashboard
                    sessionMetrics={sessionMetrics}
                    isVisible={showOverallPerformance}
                  />
                </div>
              )}
              
              {/* Question-Specific Performance Modal */}
              {showQuestionPerformance && selectedQuestionPerformance && (
                <div className="p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-medium">📈 Question Performance</h3>
                    <button
                      onClick={handleCloseQuestionPerformance}
                      className="text-xs text-muted-foreground hover:text-foreground"
                    >
                      ✕ Close
                    </button>
                  </div>
                  <PerformanceDashboard
                    benchmark={selectedQuestionPerformance}
                    isVisible={showQuestionPerformance}
                  />
                </div>
              )}
              
              {showReasoningChain && selectedReasoningChain && (
                <div className="p-4">
                  <ReasoningChainDisplay
                    reasoningChain={selectedReasoningChain}
                    isVisible={showReasoningChain}
                    onClose={() => setShowReasoningChain(false)}
                  />
                </div>
              )}
              
              {/* Evaluation Result Modal */}
              {showEvaluationModal && selectedEvaluationId && (
                <div className="p-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-medium">📋 Evaluation Result</h3>
                    <button
                      onClick={() => setShowEvaluationModal(false)}
                      className="text-xs text-muted-foreground hover:text-foreground"
                    >
                      ✕ Close
                    </button>
                  </div>
                  <div className="text-sm text-muted-foreground mb-4">
                    Evaluation ID: <span className="text-foreground">{selectedEvaluationId}</span>
                  </div>
                  <pre className="bg-muted p-4 rounded-md text-xs overflow-auto">
                    {JSON.stringify(evaluationResults[selectedEvaluationId], null, 2)}
                  </pre>
                </div>
              )}
            </ResizablePanel>
          </>
        )}
      </ResizablePanelGroup>
      
      {/* Source Verification Modal */}
      <SourceVerification
        verifiedSources={verifiedSources}
        isOpen={showSourceVerificationModal}
        onClose={() => setShowSourceVerificationModal(false)}
      />
      
      {/* Evaluation Result Modal */}
      <EvaluationModal
        isOpen={showEvaluationModal}
        onClose={() => setShowEvaluationModal(false)}
        evaluationResult={selectedEvaluationId ? evaluationResults[selectedEvaluationId] : null}
      />
    </div>
  );
};

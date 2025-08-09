"""
APG RAG Generation Engine

Advanced generation system with intelligent model selection, comprehensive source attribution,
and quality validation using qwen3/deepseek-r1 via Ollama integration.
"""

import asyncio
import logging
import time
import json
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from uuid_extensions import uuid7str
from collections import defaultdict

# Database imports
import asyncpg
from asyncpg import Pool

# APG imports
from .models import (
	GenerationRequest, GeneratedResponse, ConversationTurn, RetrievalResult,
	ValidationStatus, APGBaseModel
)
from .retrieval_engine import IntelligentRetrievalEngine, RankedResult
from .ollama_integration import AdvancedOllamaIntegration, GenerationQueueRequest, RequestPriority

class GenerationTask(str, Enum):
	"""Types of generation tasks for model selection"""
	GENERAL_QA = "general_qa"
	TECHNICAL_DOCS = "technical_docs"
	CODE_GENERATION = "code_generation"
	ANALYSIS = "analysis"
	SUMMARIZATION = "summarization"
	CREATIVE_WRITING = "creative_writing"
	REASONING = "reasoning"
	EXPLANATION = "explanation"

class SourceAttributionLevel(str, Enum):
	"""Levels of source attribution detail"""
	MINIMAL = "minimal"      # Just source IDs
	STANDARD = "standard"    # Source IDs + titles
	DETAILED = "detailed"    # Full source metadata
	GRANULAR = "granular"    # Sentence-level attribution

@dataclass
class GenerationConfig:
	"""Configuration for generation operations"""
	# Model selection
	default_model: str = "qwen3"
	model_routing_enabled: bool = True
	
	# Generation parameters
	default_max_tokens: int = 2048
	default_temperature: float = 0.7
	default_top_p: float = 0.9
	
	# Context management
	max_context_tokens: int = 8000  # bge-m3 context length
	context_overlap_tokens: int = 200
	source_context_ratio: float = 0.6  # 60% of context for sources
	
	# Source attribution
	attribution_level: SourceAttributionLevel = SourceAttributionLevel.STANDARD
	min_attribution_confidence: float = 0.8
	max_sources_per_response: int = 10
	
	# Quality control
	enable_fact_checking: bool = True
	enable_consistency_check: bool = True
	enable_citation_validation: bool = True
	
	# Performance
	generation_timeout_seconds: float = 60.0
	enable_streaming: bool = True

@dataclass
class SourceAttribution:
	"""Detailed source attribution information"""
	chunk_id: str
	document_id: str
	document_title: str
	chunk_content: str
	similarity_score: float
	confidence_score: float
	citation_text: str
	text_spans: List[Tuple[int, int]]  # Start, end positions in generated text
	metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GenerationContext:
	"""Context for generation including sources and conversation history"""
	retrieved_sources: List[RankedResult]
	conversation_history: List[ConversationTurn] = field(default_factory=list)
	user_context: Dict[str, Any] = field(default_factory=dict)
	system_context: Dict[str, Any] = field(default_factory=dict)
	constraints: List[str] = field(default_factory=list)

@dataclass
class GenerationResult:
	"""Complete generation result with attribution and quality metrics"""
	response_text: str
	sources_used: List[SourceAttribution]
	generation_model: str
	generation_time_ms: float
	token_count: int
	confidence_score: float
	factual_accuracy_score: float
	citation_coverage: float
	consistency_score: float
	metadata: Dict[str, Any] = field(default_factory=dict)

class ModelSelector:
	"""Intelligent model selection based on task and content analysis"""
	
	def __init__(self, config: GenerationConfig):
		self.config = config
		self.logger = logging.getLogger(__name__)
		
		# Model capabilities mapping
		self.model_capabilities = {
			"qwen3": {
				"strengths": [GenerationTask.GENERAL_QA, GenerationTask.ANALYSIS, 
				            GenerationTask.SUMMARIZATION, GenerationTask.EXPLANATION],
				"max_tokens": 4096,
				"temperature_range": (0.1, 1.0),
				"good_for_reasoning": True,
				"good_for_factual": True
			},
			"deepseek-r1": {
				"strengths": [GenerationTask.CODE_GENERATION, GenerationTask.TECHNICAL_DOCS,
				            GenerationTask.REASONING, GenerationTask.ANALYSIS],
				"max_tokens": 8192,
				"temperature_range": (0.0, 0.8),
				"good_for_reasoning": True,
				"good_for_code": True
			}
		}
	
	def select_model(self, task: GenerationTask, content_analysis: Dict[str, Any]) -> str:
		"""Select optimal model based on task and content analysis"""
		if not self.config.model_routing_enabled:
			return self.config.default_model
		
		# Score each model for this task
		model_scores = {}
		
		for model, capabilities in self.model_capabilities.items():
			score = 0.0
			
			# Task-specific scoring
			if task in capabilities["strengths"]:
				score += 0.4
			
			# Content-specific scoring
			if content_analysis.get("has_code", False) and capabilities.get("good_for_code", False):
				score += 0.3
			
			if content_analysis.get("requires_reasoning", False) and capabilities.get("good_for_reasoning", False):
				score += 0.2
			
			if content_analysis.get("factual_query", False) and capabilities.get("good_for_factual", False):
				score += 0.1
			
			model_scores[model] = score
		
		# Select highest scoring model
		best_model = max(model_scores, key=model_scores.get)
		
		self.logger.debug(f"Model selection: {best_model} (scores: {model_scores})")
		return best_model
	
	def get_optimal_parameters(self, model: str, task: GenerationTask) -> Dict[str, Any]:
		"""Get optimal generation parameters for model and task"""
		if model not in self.model_capabilities:
			model = self.config.default_model
		
		capabilities = self.model_capabilities[model]
		
		# Task-specific parameter adjustment
		temperature = self.config.default_temperature
		top_p = self.config.default_top_p
		
		if task in [GenerationTask.CODE_GENERATION, GenerationTask.TECHNICAL_DOCS]:
			temperature = 0.3  # More deterministic for technical content
		elif task == GenerationTask.CREATIVE_WRITING:
			temperature = 0.9  # More creative
		elif task == GenerationTask.REASONING:
			temperature = 0.5  # Balanced for reasoning
		
		# Ensure temperature is within model's range
		temp_min, temp_max = capabilities["temperature_range"]
		temperature = max(temp_min, min(temperature, temp_max))
		
		return {
			"temperature": temperature,
			"top_p": top_p,
			"max_tokens": min(self.config.default_max_tokens, capabilities["max_tokens"])
		}

class SourceAttributor:
	"""Handles comprehensive source attribution and citation"""
	
	def __init__(self, config: GenerationConfig):
		self.config = config
		self.logger = logging.getLogger(__name__)
	
	async def create_attributions(self,
	                            generated_text: str,
	                            sources: List[RankedResult],
	                            generation_context: GenerationContext) -> List[SourceAttribution]:
		"""Create detailed source attributions for generated text"""
		
		attributions = []
		
		if self.config.attribution_level == SourceAttributionLevel.MINIMAL:
			# Just basic source info
			for i, source in enumerate(sources[:self.config.max_sources_per_response]):
				attribution = SourceAttribution(
					chunk_id=source.chunk_id,
					document_id=source.document_id,
					document_title=source.metadata.get('document_title', ''),
					chunk_content=source.content[:200] + "..." if len(source.content) > 200 else source.content,
					similarity_score=source.similarity_score,
					confidence_score=source.final_score,
					citation_text=f"[{i+1}]",
					text_spans=[]
				)
				attributions.append(attribution)
		
		elif self.config.attribution_level == SourceAttributionLevel.GRANULAR:
			# Detailed sentence-level attribution
			attributions = await self._granular_attribution(generated_text, sources)
		
		else:
			# Standard or detailed attribution
			attributions = await self._standard_attribution(generated_text, sources)
		
		return attributions
	
	async def _standard_attribution(self, generated_text: str, sources: List[RankedResult]) -> List[SourceAttribution]:
		"""Create standard source attributions"""
		attributions = []
		
		for i, source in enumerate(sources[:self.config.max_sources_per_response]):
			# Calculate confidence based on content overlap
			confidence = self._calculate_attribution_confidence(generated_text, source.content)
			
			if confidence >= self.config.min_attribution_confidence:
				attribution = SourceAttribution(
					chunk_id=source.chunk_id,
					document_id=source.document_id,
					document_title=source.metadata.get('document_title', ''),
					chunk_content=source.content,
					similarity_score=source.similarity_score,
					confidence_score=confidence,
					citation_text=f"[{i+1}]",
					text_spans=self._find_text_spans(generated_text, source.content),
					metadata=source.metadata
				)
				attributions.append(attribution)
		
		return attributions
	
	async def _granular_attribution(self, generated_text: str, sources: List[RankedResult]) -> List[SourceAttribution]:
		"""Create granular sentence-level attribution"""
		# Split generated text into sentences
		sentences = self._split_into_sentences(generated_text)
		attributions = []
		
		for source_idx, source in enumerate(sources[:self.config.max_sources_per_response]):
			attributed_sentences = []
			
			for sent_idx, sentence in enumerate(sentences):
				confidence = self._calculate_attribution_confidence(sentence, source.content)
				if confidence >= self.config.min_attribution_confidence:
					attributed_sentences.append((sent_idx, sentence))
			
			if attributed_sentences:
				# Find text spans for attributed sentences
				text_spans = []
				for sent_idx, sentence in attributed_sentences:
					start_pos = generated_text.find(sentence)
					if start_pos != -1:
						end_pos = start_pos + len(sentence)
						text_spans.append((start_pos, end_pos))
				
				attribution = SourceAttribution(
					chunk_id=source.chunk_id,
					document_id=source.document_id,
					document_title=source.metadata.get('document_title', ''),
					chunk_content=source.content,
					similarity_score=source.similarity_score,
					confidence_score=sum(conf for _, _, conf in [(0, 0, self._calculate_attribution_confidence(s[1], source.content)) for s in attributed_sentences]) / len(attributed_sentences),
					citation_text=f"[{source_idx+1}]",
					text_spans=text_spans,
					metadata=source.metadata
				)
				attributions.append(attribution)
		
		return attributions
	
	def _calculate_attribution_confidence(self, generated_text: str, source_content: str) -> float:
		"""Calculate confidence that generated text is attributed to source"""
		# Simple word overlap confidence
		gen_words = set(generated_text.lower().split())
		source_words = set(source_content.lower().split())
		
		if not gen_words or not source_words:
			return 0.0
		
		overlap = len(gen_words.intersection(source_words))
		return overlap / len(gen_words)
	
	def _find_text_spans(self, generated_text: str, source_content: str) -> List[Tuple[int, int]]:
		"""Find text spans in generated text that correspond to source"""
		spans = []
		
		# Find common phrases (3+ words)
		source_words = source_content.lower().split()
		gen_text_lower = generated_text.lower()
		
		for i in range(len(source_words) - 2):
			phrase = " ".join(source_words[i:i+3])
			start_pos = gen_text_lower.find(phrase)
			if start_pos != -1:
				end_pos = start_pos + len(phrase)
				spans.append((start_pos, end_pos))
		
		# Merge overlapping spans
		if spans:
			spans.sort()
			merged = [spans[0]]
			for start, end in spans[1:]:
				if start <= merged[-1][1]:
					merged[-1] = (merged[-1][0], max(merged[-1][1], end))
				else:
					merged.append((start, end))
			spans = merged
		
		return spans
	
	def _split_into_sentences(self, text: str) -> List[str]:
		"""Split text into sentences"""
		# Simple sentence splitting
		sentences = re.split(r'[.!?]+', text)
		return [s.strip() for s in sentences if s.strip()]

class QualityValidator:
	"""Validates quality of generated responses"""
	
	def __init__(self, config: GenerationConfig):
		self.config = config
		self.logger = logging.getLogger(__name__)
	
	async def validate_response(self,
	                          generated_text: str,
	                          sources: List[SourceAttribution],
	                          generation_context: GenerationContext) -> Dict[str, float]:
		"""Comprehensive quality validation"""
		
		scores = {}
		
		# Factual accuracy check
		if self.config.enable_fact_checking:
			scores['factual_accuracy'] = await self._check_factual_accuracy(generated_text, sources)
		
		# Consistency check
		if self.config.enable_consistency_check:
			scores['consistency'] = self._check_consistency(generated_text, generation_context)
		
		# Citation validation
		if self.config.enable_citation_validation:
			scores['citation_coverage'] = self._validate_citations(generated_text, sources)
		
		# Overall confidence
		scores['confidence'] = self._calculate_overall_confidence(scores, generated_text, sources)
		
		return scores
	
	async def _check_factual_accuracy(self, generated_text: str, sources: List[SourceAttribution]) -> float:
		"""Check factual accuracy against sources"""
		if not sources:
			return 0.5  # Neutral score when no sources
		
		# Simple factual accuracy based on source attribution confidence
		attribution_scores = [source.confidence_score for source in sources]
		
		if attribution_scores:
			return sum(attribution_scores) / len(attribution_scores)
		
		return 0.5
	
	def _check_consistency(self, generated_text: str, generation_context: GenerationContext) -> float:
		"""Check consistency with conversation history"""
		if not generation_context.conversation_history:
			return 1.0  # Perfect consistency when no history
		
		# Simple consistency check based on topic coherence
		recent_turns = generation_context.conversation_history[-3:]  # Last 3 turns
		
		consistency_scores = []
		for turn in recent_turns:
			if turn.turn_type == "assistant":
				consistency = self._calculate_text_similarity(generated_text, turn.content)
				consistency_scores.append(consistency)
		
		if consistency_scores:
			return sum(consistency_scores) / len(consistency_scores)
		
		return 1.0
	
	def _validate_citations(self, generated_text: str, sources: List[SourceAttribution]) -> float:
		"""Validate citation coverage"""
		if not sources:
			return 0.0
		
		# Count how much of the generated text is covered by citations
		total_chars = len(generated_text)
		covered_chars = 0
		
		for source in sources:
			for start, end in source.text_spans:
				covered_chars += (end - start)
		
		return min(covered_chars / max(total_chars, 1), 1.0)
	
	def _calculate_overall_confidence(self, scores: Dict[str, float], generated_text: str, sources: List[SourceAttribution]) -> float:
		"""Calculate overall confidence score"""
		if not scores:
			return 0.5
		
		# Weighted average of quality scores
		weights = {
			'factual_accuracy': 0.4,
			'consistency': 0.3,
			'citation_coverage': 0.3
		}
		
		weighted_score = 0.0
		total_weight = 0.0
		
		for metric, score in scores.items():
			weight = weights.get(metric, 0.1)
			weighted_score += score * weight
			total_weight += weight
		
		return weighted_score / max(total_weight, 1.0)
	
	def _calculate_text_similarity(self, text1: str, text2: str) -> float:
		"""Calculate similarity between two texts"""
		words1 = set(text1.lower().split())
		words2 = set(text2.lower().split())
		
		if not words1 or not words2:
			return 0.0
		
		intersection = len(words1.intersection(words2))
		union = len(words1.union(words2))
		
		return intersection / union if union > 0 else 0.0

class ContextManager:
	"""Manages context for generation including sources and conversation history"""
	
	def __init__(self, config: GenerationConfig):
		self.config = config
		self.logger = logging.getLogger(__name__)
	
	async def prepare_context(self,
	                        query: str,
	                        retrieved_sources: List[RankedResult],
	                        conversation_history: List[ConversationTurn] = None,
	                        user_context: Dict[str, Any] = None) -> GenerationContext:
		"""Prepare comprehensive generation context"""
		
		# Limit sources to fit in context window
		limited_sources = self._limit_sources_by_tokens(retrieved_sources)
		
		# Prepare conversation history
		history = conversation_history or []
		limited_history = self._limit_history_by_tokens(history)
		
		return GenerationContext(
			retrieved_sources=limited_sources,
			conversation_history=limited_history,
			user_context=user_context or {},
			system_context={
				"timestamp": datetime.now().isoformat(),
				"query": query
			}
		)
	
	def build_prompt(self, query: str, context: GenerationContext, task: GenerationTask) -> str:
		"""Build comprehensive prompt for generation"""
		
		prompt_parts = []
		
		# System instruction
		prompt_parts.append(self._get_system_instruction(task))
		
		# Context sources
		if context.retrieved_sources:
			prompt_parts.append("\n--- Retrieved Information ---")
			for i, source in enumerate(context.retrieved_sources, 1):
				source_text = f"[{i}] {source.content}"
				if len(source_text) > 500:
					source_text = source_text[:500] + "..."
				prompt_parts.append(source_text)
		
		# Conversation history
		if context.conversation_history:
			prompt_parts.append("\n--- Previous Conversation ---")
			for turn in context.conversation_history[-3:]:  # Last 3 turns
				role = "Human" if turn.turn_type == "user" else "Assistant"
				content = turn.content
				if len(content) > 200:
					content = content[:200] + "..."
				prompt_parts.append(f"{role}: {content}")
		
		# Current query
		prompt_parts.append(f"\n--- Current Question ---\n{query}")
		
		# Instructions
		prompt_parts.append(self._get_generation_instructions(task, context))
		
		return "\n\n".join(prompt_parts)
	
	def _get_system_instruction(self, task: GenerationTask) -> str:
		"""Get system instruction based on task"""
		instructions = {
			GenerationTask.GENERAL_QA: "You are a helpful assistant that answers questions using provided information. Always cite your sources.",
			GenerationTask.TECHNICAL_DOCS: "You are a technical documentation expert. Provide accurate, detailed technical information with proper citations.",
			GenerationTask.CODE_GENERATION: "You are a code generation expert. Provide working code with explanations and cite any reference materials.",
			GenerationTask.ANALYSIS: "You are an analytical expert. Provide thorough analysis based on the provided information with proper citations.",
			GenerationTask.SUMMARIZATION: "You are a summarization expert. Create concise, accurate summaries while citing source materials.",
			GenerationTask.REASONING: "You are a reasoning expert. Provide step-by-step logical reasoning based on the provided information."
		}
		
		return instructions.get(task, instructions[GenerationTask.GENERAL_QA])
	
	def _get_generation_instructions(self, task: GenerationTask, context: GenerationContext) -> str:
		"""Get specific generation instructions"""
		base_instructions = [
			"Please provide a comprehensive response based on the retrieved information.",
			"Include citations using [1], [2], etc. format for source references.",
			"If the retrieved information doesn't fully answer the question, acknowledge this limitation."
		]
		
		if task == GenerationTask.TECHNICAL_DOCS:
			base_instructions.append("Include technical details and examples where appropriate.")
		elif task == GenerationTask.CODE_GENERATION:
			base_instructions.append("Provide working code with clear explanations.")
		elif task == GenerationTask.ANALYSIS:
			base_instructions.append("Structure your analysis with clear sections and reasoning.")
		
		return "\n".join(base_instructions)
	
	def _limit_sources_by_tokens(self, sources: List[RankedResult]) -> List[RankedResult]:
		"""Limit sources to fit within context window"""
		max_source_tokens = int(self.config.max_context_tokens * self.config.source_context_ratio)
		
		limited_sources = []
		current_tokens = 0
		
		for source in sources:
			# Rough token estimation (4 chars per token)
			source_tokens = len(source.content) // 4
			
			if current_tokens + source_tokens <= max_source_tokens:
				limited_sources.append(source)
				current_tokens += source_tokens
			else:
				break
		
		return limited_sources
	
	def _limit_history_by_tokens(self, history: List[ConversationTurn]) -> List[ConversationTurn]:
		"""Limit conversation history to fit within context window"""
		max_history_tokens = int(self.config.max_context_tokens * (1 - self.config.source_context_ratio))
		
		# Start from most recent and work backwards
		limited_history = []
		current_tokens = 0
		
		for turn in reversed(history):
			turn_tokens = len(turn.content) // 4
			
			if current_tokens + turn_tokens <= max_history_tokens:
				limited_history.insert(0, turn)
				current_tokens += turn_tokens
			else:
				break
		
		return limited_history

class RAGGenerationEngine:
	"""Main RAG generation engine with intelligent model selection and source attribution"""
	
	def __init__(self,
	             config: GenerationConfig,
	             ollama_integration: AdvancedOllamaIntegration,
	             retrieval_engine: IntelligentRetrievalEngine,
	             db_pool: Pool,
	             tenant_id: str,
	             capability_id: str = "rag"):
		
		self.config = config
		self.ollama_integration = ollama_integration
		self.retrieval_engine = retrieval_engine
		self.db_pool = db_pool
		self.tenant_id = tenant_id
		self.capability_id = capability_id
		
		# Core components
		self.model_selector = ModelSelector(config)
		self.source_attributor = SourceAttributor(config)
		self.quality_validator = QualityValidator(config)
		self.context_manager = ContextManager(config)
		
		# Statistics
		self.stats = {
			'total_generations': 0,
			'successful_generations': 0,
			'failed_generations': 0,
			'average_generation_time_ms': 0.0,
			'model_usage': defaultdict(int),
			'task_distribution': defaultdict(int)
		}
		
		self.logger = logging.getLogger(__name__)
	
	async def generate_response(self,
	                          request: GenerationRequest,
	                          retrieval_result: Optional[RetrievalResult] = None,
	                          conversation_history: List[ConversationTurn] = None) -> GenerationResult:
		"""Generate RAG response with comprehensive source attribution"""
		start_time = time.time()
		request_id = uuid7str()
		
		try:
			self.logger.info(f"[{request_id}] Starting RAG generation for: {request.prompt[:100]}...")
			
			# Analyze content and determine task
			content_analysis = await self._analyze_content(request.prompt)
			task = self._determine_task_type(request.prompt, content_analysis)
			
			# Select optimal model
			selected_model = self.model_selector.select_model(task, content_analysis)
			model_params = self.model_selector.get_optimal_parameters(selected_model, task)
			
			# Get retrieval sources if not provided
			if retrieval_result and hasattr(retrieval_result, 'retrieved_chunk_ids'):
				sources = await self._get_ranked_sources_from_retrieval(retrieval_result)
			else:
				sources = []
			
			# Prepare generation context
			generation_context = await self.context_manager.prepare_context(
				request.prompt, sources, conversation_history
			)
			
			# Build prompt
			full_prompt = self.context_manager.build_prompt(request.prompt, generation_context, task)
			
			# Generate response
			generated_response = await self._generate_with_model(
				full_prompt, selected_model, model_params, request_id
			)
			
			# Create source attributions
			attributions = await self.source_attributor.create_attributions(
				generated_response, sources, generation_context
			)
			
			# Validate quality
			quality_scores = await self.quality_validator.validate_response(
				generated_response, attributions, generation_context
			)
			
			# Calculate final metrics
			generation_time_ms = (time.time() - start_time) * 1000
			token_count = len(generated_response.split())  # Rough estimation
			
			# Update statistics
			self._update_stats(generation_time_ms, selected_model, task, True)
			
			result = GenerationResult(
				response_text=generated_response,
				sources_used=attributions,
				generation_model=selected_model,
				generation_time_ms=generation_time_ms,
				token_count=token_count,
				confidence_score=quality_scores.get('confidence', 0.5),
				factual_accuracy_score=quality_scores.get('factual_accuracy', 0.5),
				citation_coverage=quality_scores.get('citation_coverage', 0.0),
				consistency_score=quality_scores.get('consistency', 1.0),
				metadata={
					'task_type': task.value,
					'model_params': model_params,
					'quality_scores': quality_scores,
					'sources_count': len(sources),
					'context_tokens': len(full_prompt.split()) * 4  # Rough estimation
				}
			)
			
			self.logger.info(f"[{request_id}] Generation completed: {len(generated_response)} chars, {len(attributions)} sources, {generation_time_ms:.1f}ms")
			return result
		
		except Exception as e:
			generation_time_ms = (time.time() - start_time) * 1000
			self._update_stats(generation_time_ms, self.config.default_model, GenerationTask.GENERAL_QA, False)
			
			self.logger.error(f"[{request_id}] Generation failed: {str(e)}")
			
			# Return minimal error result
			return GenerationResult(
				response_text=f"I apologize, but I encountered an error while generating a response: {str(e)}",
				sources_used=[],
				generation_model=self.config.default_model,
				generation_time_ms=generation_time_ms,
				token_count=0,
				confidence_score=0.0,
				factual_accuracy_score=0.0,
				citation_coverage=0.0,
				consistency_score=0.0,
				metadata={'error': str(e)}
			)
	
	async def _analyze_content(self, prompt: str) -> Dict[str, Any]:
		"""Analyze content to determine characteristics"""
		prompt_lower = prompt.lower()
		
		analysis = {
			'has_code': any(keyword in prompt_lower for keyword in ['code', 'function', 'class', 'method', 'algorithm']),
			'requires_reasoning': any(keyword in prompt_lower for keyword in ['why', 'how', 'analyze', 'compare', 'evaluate']),
			'factual_query': any(keyword in prompt_lower for keyword in ['what', 'when', 'where', 'who', 'define']),
			'creative_request': any(keyword in prompt_lower for keyword in ['create', 'write', 'generate', 'compose']),
			'technical_content': any(keyword in prompt_lower for keyword in ['api', 'database', 'system', 'architecture']),
			'length': len(prompt.split()),
			'complexity': self._calculate_query_complexity(prompt)
		}
		
		return analysis
	
	def _determine_task_type(self, prompt: str, content_analysis: Dict[str, Any]) -> GenerationTask:
		"""Determine the type of generation task"""
		prompt_lower = prompt.lower()
		
		# Code generation indicators
		if content_analysis['has_code'] or any(keyword in prompt_lower for keyword in ['implement', 'code', 'program']):
			return GenerationTask.CODE_GENERATION
		
		# Technical documentation
		if content_analysis['technical_content'] or 'document' in prompt_lower:
			return GenerationTask.TECHNICAL_DOCS
		
		# Analysis tasks
		if content_analysis['requires_reasoning'] or any(keyword in prompt_lower for keyword in ['analyze', 'compare', 'evaluate']):
			return GenerationTask.ANALYSIS
		
		# Summarization
		if any(keyword in prompt_lower for keyword in ['summariz', 'summary', 'brief']):
			return GenerationTask.SUMMARIZATION
		
		# Creative writing
		if content_analysis['creative_request']:
			return GenerationTask.CREATIVE_WRITING
		
		# Reasoning tasks
		if any(keyword in prompt_lower for keyword in ['why', 'how', 'explain', 'reason']):
			return GenerationTask.REASONING
		
		# Default to general QA
		return GenerationTask.GENERAL_QA
	
	def _calculate_query_complexity(self, prompt: str) -> float:
		"""Calculate query complexity score"""
		factors = []
		
		# Length factor
		words = prompt.split()
		length_score = min(len(words) / 50.0, 1.0)
		factors.append(length_score)
		
		# Question complexity
		question_count = prompt.count('?')
		question_score = min(question_count / 3.0, 1.0)
		factors.append(question_score)
		
		# Technical terms
		technical_terms = ['algorithm', 'architecture', 'implementation', 'optimization', 'analysis']
		tech_count = sum(1 for term in technical_terms if term in prompt.lower())
		tech_score = min(tech_count / len(technical_terms), 1.0)
		factors.append(tech_score)
		
		return sum(factors) / len(factors)
	
	async def _get_ranked_sources_from_retrieval(self, retrieval_result: RetrievalResult) -> List[RankedResult]:
		"""Convert retrieval result to ranked sources"""
		sources = []
		
		for i, chunk_id in enumerate(retrieval_result.retrieved_chunk_ids):
			chunk_data = await self._get_chunk_data(chunk_id)
			if chunk_data:
				similarity_score = retrieval_result.similarity_scores[i] if i < len(retrieval_result.similarity_scores) else 0.0
				
				ranked_result = RankedResult(
					chunk_id=chunk_id,
					document_id=chunk_data['document_id'],
					content=chunk_data['content'],
					similarity_score=similarity_score,
					relevance_score=similarity_score,
					authority_score=0.5,
					recency_score=0.5,
					diversity_score=0.5,
					final_score=similarity_score,
					metadata=chunk_data
				)
				sources.append(ranked_result)
		
		return sources
	
	async def _get_chunk_data(self, chunk_id: str) -> Optional[Dict[str, Any]]:
		"""Get chunk data from database"""
		try:
			async with self.db_pool.acquire() as conn:
				result = await conn.fetchrow("""
					SELECT 
						c.*,
						d.title as document_title,
						d.filename as document_filename
					FROM apg_rag_document_chunks c
					JOIN apg_rag_documents d ON c.document_id = d.id
					WHERE c.id = $1 AND c.tenant_id = $2
				""", chunk_id, self.tenant_id)
				
				return dict(result) if result else None
		except Exception as e:
			self.logger.error(f"Failed to get chunk data for {chunk_id}: {str(e)}")
			return None
	
	async def _generate_with_model(self,
	                             prompt: str,
	                             model: str,
	                             model_params: Dict[str, Any],
	                             request_id: str) -> str:
		"""Generate response using selected model"""
		
		generation_result = {}
		
		def generation_callback(result):
			if result['success']:
				generation_result['text'] = result['text']
			else:
				generation_result['error'] = result.get('error', 'Generation failed')
		
		# Request generation
		await self.ollama_integration.generate_text_async(
			prompt=prompt,
			model=model,
			tenant_id=self.tenant_id,
			capability_id=self.capability_id,
			priority=RequestPriority.HIGH,
			callback=generation_callback,
			**model_params
		)
		
		# Wait for result
		max_wait = self.config.generation_timeout_seconds
		wait_start = time.time()
		
		while 'text' not in generation_result and 'error' not in generation_result and (time.time() - wait_start) < max_wait:
			await asyncio.sleep(0.1)
		
		if 'text' in generation_result:
			return generation_result['text']
		elif 'error' in generation_result:
			raise Exception(generation_result['error'])
		else:
			raise TimeoutError(f"Generation timed out after {max_wait}s")
	
	def _update_stats(self, generation_time_ms: float, model: str, task: GenerationTask, success: bool) -> None:
		"""Update generation statistics"""
		self.stats['total_generations'] += 1
		
		if success:
			self.stats['successful_generations'] += 1
		else:
			self.stats['failed_generations'] += 1
		
		# Update average generation time
		current_avg = self.stats['average_generation_time_ms']
		total_gens = self.stats['total_generations']
		self.stats['average_generation_time_ms'] = (
			(current_avg * (total_gens - 1) + generation_time_ms) / total_gens
		)
		
		# Update model and task usage
		self.stats['model_usage'][model] += 1
		self.stats['task_distribution'][task.value] += 1
	
	def get_statistics(self) -> Dict[str, Any]:
		"""Get comprehensive generation statistics"""
		success_rate = self.stats['successful_generations'] / max(1, self.stats['total_generations'])
		
		return {
			**self.stats,
			'success_rate': success_rate,
			'model_usage_dict': dict(self.stats['model_usage']),
			'task_distribution_dict': dict(self.stats['task_distribution'])
		}
	
	async def health_check(self) -> Dict[str, Any]:
		"""Comprehensive health check"""
		health_info = {
			'generation_engine_healthy': True,
			'ollama_integration': False,
			'database_connection': False,
			'model_availability': {},
			'timestamp': datetime.now().isoformat()
		}
		
		try:
			# Test Ollama integration
			ollama_status = await self.ollama_integration.get_system_status()
			health_info['ollama_integration'] = ollama_status['service_status'] == 'running'
			
			# Check model availability
			for model in ['qwen3', 'deepseek-r1']:
				model_healthy = False
				for endpoint in ollama_status.get('endpoints', []):
					if endpoint.get('healthy', False):
						model_healthy = True
						break
				health_info['model_availability'][model] = model_healthy
			
			# Test database connection
			async with self.db_pool.acquire() as conn:
				await conn.fetchval("SELECT 1")
				health_info['database_connection'] = True
		
		except Exception as e:
			health_info['generation_engine_healthy'] = False
			health_info['error'] = str(e)
		
		return health_info

# Factory function for APG integration
async def create_generation_engine(
	tenant_id: str,
	capability_id: str,
	ollama_integration: AdvancedOllamaIntegration,
	retrieval_engine: IntelligentRetrievalEngine,
	db_pool: Pool,
	config: GenerationConfig = None
) -> RAGGenerationEngine:
	"""Create RAG generation engine"""
	if config is None:
		config = GenerationConfig()
	
	engine = RAGGenerationEngine(
		config, ollama_integration, retrieval_engine, db_pool, tenant_id, capability_id
	)
	
	return engine
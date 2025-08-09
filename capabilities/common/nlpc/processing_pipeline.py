"""
APG NLP Advanced Processing Pipeline

Comprehensive text processing pipeline with multi-language support, streaming capabilities,
batch processing, and domain-specific templates.

Features:
- Text preprocessing and normalization pipeline
- Multi-language processing with automatic detection
- Batch processing for high throughput
- Real-time streaming with WebSocket support
- Domain-specific processing templates
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, Callable, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import hashlib
from contextlib import asynccontextmanager
from uuid_extensions import uuid7str

from models import (
	ProcessingRequest, ProcessingResult, StreamingChunk, StreamingSession,
	NLPTaskType, ModelProvider, QualityLevel, LanguageCode, ProcessingStatus
)

# Configure logging
logger = logging.getLogger(__name__)

class PipelineStage(str, Enum):
	"""Pipeline processing stages"""
	PREPROCESSING = "preprocessing"
	LANGUAGE_DETECTION = "language_detection"
	MODEL_PROCESSING = "model_processing"
	POSTPROCESSING = "postprocessing"
	VALIDATION = "validation"
	AGGREGATION = "aggregation"

class ProcessingMode(str, Enum):
	"""Processing execution modes"""
	SINGLE = "single"
	BATCH = "batch"
	STREAMING = "streaming"
	ENSEMBLE = "ensemble"

@dataclass
class PipelineConfig:
	"""Pipeline configuration for different processing scenarios"""
	name: str
	stages: List[str] = field(default_factory=list)
	preprocessing_steps: List[str] = field(default_factory=list)
	postprocessing_steps: List[str] = field(default_factory=list)
	validation_rules: List[str] = field(default_factory=list)
	language_detection_enabled: bool = True
	batch_size: int = 50
	streaming_chunk_size: int = 800
	streaming_overlap: int = 80
	timeout_seconds: int = 300
	quality_threshold: float = 0.7
	ensemble_enabled: bool = False
	fallback_enabled: bool = True
	caching_enabled: bool = True

@dataclass
class ProcessingContext:
	"""Context for pipeline processing"""
	request_id: str
	tenant_id: str
	mode: ProcessingMode
	config: PipelineConfig
	start_time: datetime = field(default_factory=datetime.utcnow)
	current_stage: Optional[PipelineStage] = None
	metadata: Dict[str, Any] = field(default_factory=dict)
	intermediate_results: Dict[str, Any] = field(default_factory=dict)
	performance_metrics: Dict[str, float] = field(default_factory=dict)
	error_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class BatchRequest:
	"""Batch processing request container"""
	batch_id: str = field(default_factory=uuid7str)
	requests: List[ProcessingRequest] = field(default_factory=list)
	batch_config: Optional[Dict[str, Any]] = None
	priority: str = "normal"
	created_at: datetime = field(default_factory=datetime.utcnow)
	
	@property
	def total_requests(self) -> int:
		return len(self.requests)
	
	@property
	def estimated_processing_time(self) -> float:
		"""Estimate total processing time for the batch"""
		base_time = sum(len(req.text_content or "") * 0.001 for req in self.requests)
		return max(base_time, 1.0)

class TextPreprocessor:
	"""Advanced text preprocessing with normalization pipeline"""
	
	def __init__(self, config: Dict[str, Any] = None):
		self.config = config or {}
		self._setup_patterns()
	
	def _setup_patterns(self) -> None:
		"""Setup regex patterns for text processing"""
		# URL pattern
		self.url_pattern = re.compile(
			r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
		)
		
		# Email pattern
		self.email_pattern = re.compile(
			r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
		)
		
		# Phone pattern
		self.phone_pattern = re.compile(
			r'(\+?1[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
		)
		
		# Whitespace normalization
		self.whitespace_pattern = re.compile(r'\s+')
		
		# Special character handling
		self.punct_pattern = re.compile(r'[^\w\s]')
		
		# Negation handling
		self.negation_pattern = re.compile(
			r'\b(not|no|never|nothing|nobody|nowhere|neither|nor|cannot|can\'t|won\'t|shouldn\'t|wouldn\'t|couldn\'t|doesn\'t|don\'t|isn\'t|aren\'t|wasn\'t|weren\'t|hasn\'t|haven\'t|hadn\'t)\b',
			re.IGNORECASE
		)
	
	def normalize_text(self, text: str, steps: List[str] = None) -> str:
		"""Apply normalization steps to text"""
		if not text or not text.strip():
			return ""
		
		steps = steps or ["whitespace", "case", "punctuation"]
		normalized = text
		
		for step in steps:
			if step == "whitespace":
				normalized = self._normalize_whitespace(normalized)
			elif step == "case":
				normalized = self._normalize_case(normalized)
			elif step == "punctuation":
				normalized = self._normalize_punctuation(normalized)
			elif step == "urls":
				normalized = self._handle_urls(normalized)
			elif step == "emails":
				normalized = self._handle_emails(normalized)
			elif step == "phones":
				normalized = self._handle_phones(normalized)
			elif step == "negations":
				normalized = self._handle_negations(normalized)
		
		return normalized.strip()
	
	def _normalize_whitespace(self, text: str) -> str:
		"""Normalize whitespace characters"""
		return self.whitespace_pattern.sub(' ', text)
	
	def _normalize_case(self, text: str) -> str:
		"""Normalize text case (optional, depends on task)"""
		if self.config.get("lowercase", False):
			return text.lower()
		return text
	
	def _normalize_punctuation(self, text: str) -> str:
		"""Handle punctuation normalization"""
		if self.config.get("remove_punctuation", False):
			return self.punct_pattern.sub(' ', text)
		return text
	
	def _handle_urls(self, text: str) -> str:
		"""Handle URL detection and replacement"""
		if self.config.get("replace_urls", True):
			return self.url_pattern.sub('<URL>', text)
		return text
	
	def _handle_emails(self, text: str) -> str:
		"""Handle email detection and replacement"""
		if self.config.get("replace_emails", True):
			return self.email_pattern.sub('<EMAIL>', text)
		return text
	
	def _handle_phones(self, text: str) -> str:
		"""Handle phone number detection and replacement"""
		if self.config.get("replace_phones", True):
			return self.phone_pattern.sub('<PHONE>', text)
		return text
	
	def _handle_negations(self, text: str) -> str:
		"""Handle negation detection and marking"""
		if self.config.get("mark_negations", True):
			return self.negation_pattern.sub(r'NOT_\1', text)
		return text
	
	def extract_features(self, text: str) -> Dict[str, Any]:
		"""Extract preprocessing features from text"""
		return {
			"original_length": len(text),
			"word_count": len(text.split()),
			"sentence_count": len(re.split(r'[.!?]+', text)),
			"url_count": len(self.url_pattern.findall(text)),
			"email_count": len(self.email_pattern.findall(text)),
			"phone_count": len(self.phone_pattern.findall(text)),
			"negation_count": len(self.negation_pattern.findall(text)),
			"special_char_ratio": len(self.punct_pattern.findall(text)) / max(len(text), 1)
		}

class LanguageDetector:
	"""Multi-language detection with confidence scoring"""
	
	def __init__(self):
		self._language_patterns = self._setup_language_patterns()
	
	def _setup_language_patterns(self) -> Dict[str, Dict[str, Any]]:
		"""Setup basic language detection patterns"""
		return {
			"en": {
				"common_words": {"the", "and", "is", "in", "to", "of", "a", "that", "it", "with"},
				"char_patterns": [r'[a-zA-Z]'],
				"stop_words_ratio": 0.3
			},
			"es": {
				"common_words": {"el", "la", "de", "que", "y", "a", "en", "un", "es", "se"},
				"char_patterns": [r'[a-zA-ZñáéíóúüÑÁÉÍÓÚÜ]'],
				"stop_words_ratio": 0.25
			},
			"fr": {
				"common_words": {"le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir"},
				"char_patterns": [r'[a-zA-ZàâäéèêëïîôöùûüÿçÀÂÄÉÈÊËÏÎÔÖÙÛÜŸÇ]'],
				"stop_words_ratio": 0.3
			},
			"de": {
				"common_words": {"der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich"},
				"char_patterns": [r'[a-zA-ZäöüÄÖÜß]'],
				"stop_words_ratio": 0.25
			}
		}
	
	async def detect_language(self, text: str) -> Dict[str, Any]:
		"""Detect language with confidence scores"""
		if not text or len(text.strip()) < 10:
			return {
				"detected_language": LanguageCode.AUTO,
				"confidence": 0.0,
				"scores": {},
				"fallback_reason": "text_too_short"
			}
		
		text_lower = text.lower()
		words = re.findall(r'\b\w+\b', text_lower)
		
		if len(words) < 3:
			return {
				"detected_language": LanguageCode.AUTO,
				"confidence": 0.0,
				"scores": {},
				"fallback_reason": "insufficient_words"
			}
		
		scores = {}
		
		for lang_code, patterns in self._language_patterns.items():
			score = 0.0
			
			# Check common words
			common_count = sum(1 for word in words if word in patterns["common_words"])
			common_ratio = common_count / len(words)
			score += common_ratio * 0.6
			
			# Check character patterns
			char_matches = sum(len(re.findall(pattern, text)) for pattern in patterns["char_patterns"])
			char_ratio = char_matches / max(len(text), 1)
			score += min(char_ratio, 1.0) * 0.4
			
			scores[lang_code] = score
		
		# Find best match
		if not scores:
			return {
				"detected_language": LanguageCode.AUTO,
				"confidence": 0.0,
				"scores": {},
				"fallback_reason": "no_patterns_matched"
			}
		
		best_lang = max(scores, key=scores.get)
		best_score = scores[best_lang]
		
		# Convert to LanguageCode enum
		try:
			detected_lang = LanguageCode(best_lang)
		except ValueError:
			detected_lang = LanguageCode.AUTO
		
		return {
			"detected_language": detected_lang,
			"confidence": best_score,
			"scores": scores,
			"word_count": len(words)
		}

class DomainProcessor:
	"""Domain-specific processing templates"""
	
	def __init__(self):
		self.domain_configs = self._setup_domain_configs()
	
	def _setup_domain_configs(self) -> Dict[str, PipelineConfig]:
		"""Setup domain-specific processing configurations"""
		return {
			"social_media": PipelineConfig(
				name="Social Media Processing",
				preprocessing_steps=["whitespace", "urls", "hashtags", "mentions", "emojis"],
				postprocessing_steps=["sentiment_normalization", "confidence_boost"],
				validation_rules=["informal_language_ok", "emoji_sentiment_check"],
				batch_size=100,
				streaming_chunk_size=280,  # Tweet-like size
				quality_threshold=0.6  # Lower threshold for informal text
			),
			"academic": PipelineConfig(
				name="Academic Text Processing",
				preprocessing_steps=["whitespace", "citations", "technical_terms"],
				postprocessing_steps=["technical_validation", "citation_analysis"],
				validation_rules=["formal_language_check", "technical_coherence"],
				batch_size=20,  # Larger documents
				streaming_chunk_size=1500,
				quality_threshold=0.85  # Higher threshold for academic text
			),
			"customer_service": PipelineConfig(
				name="Customer Service Processing",
				preprocessing_steps=["whitespace", "politeness_markers", "urgency_indicators"],
				postprocessing_steps=["sentiment_calibration", "urgency_scoring"],
				validation_rules=["sentiment_consistency", "urgency_alignment"],
				batch_size=75,
				streaming_chunk_size=600,
				quality_threshold=0.75
			),
			"legal": PipelineConfig(
				name="Legal Document Processing",
				preprocessing_steps=["whitespace", "legal_citations", "section_markers"],
				postprocessing_steps=["legal_validation", "citation_verification"],
				validation_rules=["legal_terminology_check", "structure_validation"],
				batch_size=10,  # Large, complex documents
				streaming_chunk_size=2000,
				quality_threshold=0.9  # Highest threshold
			),
			"medical": PipelineConfig(
				name="Medical Text Processing",
				preprocessing_steps=["whitespace", "medical_abbreviations", "dosage_normalization"],
				postprocessing_steps=["medical_validation", "safety_checks"],
				validation_rules=["medical_terminology_check", "safety_validation"],
				batch_size=25,
				streaming_chunk_size=1200,
				quality_threshold=0.88
			)
		}
	
	def get_domain_config(self, domain: str) -> PipelineConfig:
		"""Get configuration for specific domain"""
		return self.domain_configs.get(domain, PipelineConfig(name="Generic"))
	
	def detect_domain(self, text: str) -> Dict[str, Any]:
		"""Detect likely domain based on text characteristics"""
		text_lower = text.lower()
		
		domain_indicators = {
			"social_media": ["@", "#", "lol", "omg", "😊", "👍", "rt:", "dm"],
			"academic": ["abstract", "methodology", "conclusion", "references", "et al", "journal"],
			"customer_service": ["dear", "thank you", "issue", "problem", "help", "support"],
			"legal": ["whereas", "therefore", "party", "agreement", "contract", "clause"],
			"medical": ["patient", "diagnosis", "treatment", "symptoms", "medical", "clinical"]
		}
		
		domain_scores = {}
		words = text_lower.split()
		
		for domain, indicators in domain_indicators.items():
			score = sum(1 for word in words if any(indicator in word for indicator in indicators))
			score += sum(1 for indicator in indicators if indicator in text_lower)
			domain_scores[domain] = score / max(len(words), 1)
		
		if not domain_scores or max(domain_scores.values()) == 0:
			return {
				"detected_domain": "generic",
				"confidence": 0.0,
				"scores": domain_scores
			}
		
		best_domain = max(domain_scores, key=domain_scores.get)
		confidence = domain_scores[best_domain]
		
		return {
			"detected_domain": best_domain,
			"confidence": confidence,
			"scores": domain_scores
		}

class StreamingProcessor:
	"""Real-time streaming text processing with WebSocket support"""
	
	def __init__(self, tenant_id: str, config: Dict[str, Any] = None):
		assert tenant_id, "Tenant ID is required for streaming processor"
		
		self.tenant_id = tenant_id
		self.config = config or {}
		self.active_sessions: Dict[str, StreamingSession] = {}
		self.chunk_queues: Dict[str, asyncio.Queue] = {}
		self.result_queues: Dict[str, asyncio.Queue] = {}
		self.processing_tasks: Dict[str, asyncio.Task] = {}
		
		self._setup_streaming_config()
	
	def _setup_streaming_config(self) -> None:
		"""Setup streaming configuration"""
		self.chunk_size = self.config.get("chunk_size", 800)
		self.overlap_size = self.config.get("overlap_size", 80)
		self.buffer_size = self.config.get("buffer_size", 100)
		self.processing_timeout = self.config.get("processing_timeout", 30)
		self.max_concurrent_sessions = self.config.get("max_concurrent_sessions", 50)
	
	async def create_session(self, user_id: str, task_type: NLPTaskType, 
							 config: Dict[str, Any] = None) -> StreamingSession:
		"""Create new streaming session"""
		if len(self.active_sessions) >= self.max_concurrent_sessions:
			raise RuntimeError(f"Maximum concurrent sessions ({self.max_concurrent_sessions}) exceeded")
		
		# Extract only valid StreamingSession fields from config
		config = config or {}
		valid_fields = {
			"model_id": config.get("model_id"),
			"language": config.get("language"),
			"chunk_size": config.get("chunk_size", self.chunk_size),
			"overlap_size": config.get("overlap_size", self.overlap_size),
			"aggregation_window_ms": config.get("aggregation_window_ms"),
			"connection_id": config.get("connection_id")
		}
		# Filter out None values
		session_config = {k: v for k, v in valid_fields.items() if v is not None}
		
		session = StreamingSession(
			tenant_id=self.tenant_id,
			user_id=user_id,
			task_type=task_type,
			**session_config
		)
		
		# Initialize queues
		self.chunk_queues[session.id] = asyncio.Queue(maxsize=self.buffer_size)
		self.result_queues[session.id] = asyncio.Queue(maxsize=self.buffer_size)
		
		# Start processing task
		self.processing_tasks[session.id] = asyncio.create_task(
			self._process_session_chunks(session)
		)
		
		self.active_sessions[session.id] = session
		self._log_session_created(session.id)
		
		return session
	
	def _log_session_created(self, session_id: str) -> None:
		"""Log streaming session creation"""
		logger.info(f"Streaming session created: {session_id}")
	
	async def close_session(self, session_id: str) -> bool:
		"""Close streaming session and cleanup resources"""
		if session_id not in self.active_sessions:
			return False
		
		session = self.active_sessions[session_id]
		session.status = "stopped"  # Use valid status from model
		
		# Cancel processing task
		if session_id in self.processing_tasks:
			task = self.processing_tasks[session_id]
			if not task.done():
				task.cancel()
				try:
					await task
				except asyncio.CancelledError:
					pass
			del self.processing_tasks[session_id]
		
		# Cleanup queues
		if session_id in self.chunk_queues:
			del self.chunk_queues[session_id]
		if session_id in self.result_queues:
			del self.result_queues[session_id]
		
		del self.active_sessions[session_id]
		self._log_session_closed(session_id)
		
		return True
	
	def _log_session_closed(self, session_id: str) -> None:
		"""Log streaming session closure"""
		logger.info(f"Streaming session closed: {session_id}")
	
	async def add_chunk(self, session_id: str, text_content: str) -> bool:
		"""Add text chunk to processing queue"""
		if session_id not in self.active_sessions:
			return False
		
		session = self.active_sessions[session_id]
		
		if session.status != "active":
			return False
		
		chunk = StreamingChunk(
			session_id=session_id,
			sequence_number=session.chunks_processed + 1,
			text_content=text_content,
			start_position=session.total_characters,
			end_position=session.total_characters + len(text_content)
		)
		
		try:
			await self.chunk_queues[session_id].put(chunk)
			session.chunks_processed += 1
			session.total_characters += len(text_content)
			session.last_activity = datetime.utcnow()
			return True
		except asyncio.QueueFull:
			logger.warning(f"Chunk queue full for session: {session_id}")
			return False
	
	async def get_result(self, session_id: str, timeout: float = None) -> Optional[ProcessingResult]:
		"""Get next processing result from session"""
		if session_id not in self.result_queues:
			return None
		
		try:
			timeout = timeout or self.processing_timeout
			result = await asyncio.wait_for(
				self.result_queues[session_id].get(),
				timeout=timeout
			)
			return result
		except asyncio.TimeoutError:
			return None
		except asyncio.QueueEmpty:
			return None
	
	async def _process_session_chunks(self, session: StreamingSession) -> None:
		"""Process chunks for a streaming session"""
		chunk_queue = self.chunk_queues[session.id]
		result_queue = self.result_queues[session.id]
		
		while session.status == "active":
			try:
				chunk = await asyncio.wait_for(
					chunk_queue.get(),
					timeout=self.processing_timeout
				)
				
				# Mock processing (in real implementation, would use actual NLP models)
				start_time = time.time()
				
				# Simulate processing delay
				await asyncio.sleep(0.01)
				
				# Create mock result
				result = ProcessingResult(
					request_id=uuid7str(),
					tenant_id=session.tenant_id,
					task_type=session.task_type,
					model_used="streaming_model",
					provider_used=ModelProvider.TRANSFORMERS,
					processing_time_ms=(time.time() - start_time) * 1000,
					total_time_ms=(time.time() - start_time) * 1000,
					results={
						"chunk_id": chunk.id,
						"sequence_number": chunk.sequence_number,
						"processed": True,
						"confidence": 0.85
					},
					confidence_score=0.85,
					status="completed"
				)
				
				chunk.status = ProcessingStatus.COMPLETED
				chunk.processing_time_ms = result.processing_time_ms
				chunk.processed_at = datetime.utcnow()
				
				await result_queue.put(result)
				
			except asyncio.TimeoutError:
				# No chunks to process, continue waiting
				continue
			except asyncio.CancelledError:
				break
			except Exception as e:
				logger.error(f"Error processing chunk in session {session.id}: {e}")
				continue
	
	def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
		"""Get streaming session statistics"""
		if session_id not in self.active_sessions:
			return None
		
		session = self.active_sessions[session_id]
		
		return {
			"session_id": session_id,
			"status": session.status,
			"chunks_processed": session.chunks_processed,
			"queue_sizes": {
				"chunks": self.chunk_queues[session_id].qsize(),
				"results": self.result_queues[session_id].qsize()
			},
			"session_duration": (
				datetime.utcnow() - session.created_at
			).total_seconds(),
			"last_activity": session.last_activity
		}

class BatchProcessor:
	"""High-throughput batch processing system"""
	
	def __init__(self, tenant_id: str, config: Dict[str, Any] = None):
		assert tenant_id, "Tenant ID is required for batch processor"
		
		self.tenant_id = tenant_id
		self.config = config or {}
		self.active_batches: Dict[str, BatchRequest] = {}
		self.batch_queues: Dict[str, asyncio.Queue] = defaultdict(lambda: asyncio.Queue())
		self.processing_semaphore = asyncio.Semaphore(
			self.config.get("max_concurrent_batches", 10)
		)
		
		self._setup_batch_config()
	
	def _setup_batch_config(self) -> None:
		"""Setup batch processing configuration"""
		self.default_batch_size = self.config.get("batch_size", 50)
		self.max_batch_size = self.config.get("max_batch_size", 200)
		self.batch_timeout = self.config.get("batch_timeout", 300)
		self.priority_levels = ["low", "normal", "high", "urgent"]
	
	async def create_batch(self, requests: List[ProcessingRequest], 
						   config: Dict[str, Any] = None) -> BatchRequest:
		"""Create new batch processing request"""
		if len(requests) > self.max_batch_size:
			raise ValueError(f"Batch size ({len(requests)}) exceeds maximum ({self.max_batch_size})")
		
		batch = BatchRequest(
			requests=requests,
			batch_config=config or {},
			priority=config.get("priority", "normal") if config else "normal"
		)
		
		self.active_batches[batch.batch_id] = batch
		
		# Add to appropriate priority queue
		await self.batch_queues[batch.priority].put(batch)
		
		self._log_batch_created(batch.batch_id, len(requests))
		
		return batch
	
	def _log_batch_created(self, batch_id: str, request_count: int) -> None:
		"""Log batch creation"""
		logger.info(f"Batch created: {batch_id} with {request_count} requests")
	
	async def process_batch(self, batch: BatchRequest) -> List[ProcessingResult]:
		"""Process a batch of requests"""
		async with self.processing_semaphore:
			start_time = time.time()
			results = []
			
			try:
				# Group requests by task type for optimal processing
				grouped_requests = defaultdict(list)
				for request in batch.requests:
					grouped_requests[request.task_type].append(request)
				
				# Process each group
				for task_type, task_requests in grouped_requests.items():
					task_results = await self._process_task_group(task_type, task_requests)
					results.extend(task_results)
				
				processing_time = (time.time() - start_time) * 1000
				self._log_batch_completed(batch.batch_id, len(results), processing_time)
				
			except Exception as e:
				logger.error(f"Batch processing error for {batch.batch_id}: {e}")
				# Create error results for failed requests
				for request in batch.requests:
					error_result = ProcessingResult(
						request_id=request.id,
						tenant_id=request.tenant_id,
						task_type=request.task_type,
						model_used="batch_processor",
						provider_used=ModelProvider.CUSTOM,
						processing_time_ms=0.0,
						total_time_ms=(time.time() - start_time) * 1000,
						results={"error": str(e)},
						status="failed",
						error_message=str(e)
					)
					results.append(error_result)
			
			finally:
				# Cleanup batch
				if batch.batch_id in self.active_batches:
					del self.active_batches[batch.batch_id]
			
			return results
	
	def _log_batch_completed(self, batch_id: str, result_count: int, processing_time: float) -> None:
		"""Log batch completion"""
		logger.info(f"Batch completed: {batch_id} - {result_count} results in {processing_time:.2f}ms")
	
	async def _process_task_group(self, task_type: NLPTaskType, 
								  requests: List[ProcessingRequest]) -> List[ProcessingResult]:
		"""Process a group of requests with the same task type"""
		results = []
		
		# Mock batch processing (in real implementation, would use optimized model calls)
		for request in requests:
			start_time = time.time()
			
			# Simulate processing
			await asyncio.sleep(0.005)  # 5ms per request in batch
			
			result = ProcessingResult(
				request_id=request.id,
				tenant_id=request.tenant_id,
				task_type=task_type,
				model_used="batch_model",
				provider_used=ModelProvider.TRANSFORMERS,
				processing_time_ms=(time.time() - start_time) * 1000,
				total_time_ms=(time.time() - start_time) * 1000,
				results={
					"processed_in_batch": True,
					"batch_optimization": True,
					"confidence": 0.82
				},
				confidence_score=0.82,
				status="completed"
			)
			
			results.append(result)
		
		return results
	
	def get_batch_stats(self) -> Dict[str, Any]:
		"""Get batch processing statistics"""
		queue_sizes = {
			priority: queue.qsize() 
			for priority, queue in self.batch_queues.items()
		}
		
		return {
			"active_batches": len(self.active_batches),
			"queue_sizes": queue_sizes,
			"total_queued": sum(queue_sizes.values()),
			"processing_capacity": self.processing_semaphore._value,
			"max_concurrent_batches": self.config.get("max_concurrent_batches", 10)
		}

class AdvancedProcessingPipeline:
	"""Main advanced processing pipeline orchestrator"""
	
	def __init__(self, tenant_id: str, config: Dict[str, Any] = None):
		assert tenant_id, "Tenant ID is required for processing pipeline"
		
		self.tenant_id = tenant_id
		self.config = config or {}
		
		# Initialize components
		self.preprocessor = TextPreprocessor(self.config.get("preprocessing", {}))
		self.language_detector = LanguageDetector()
		self.domain_processor = DomainProcessor()
		self.streaming_processor = StreamingProcessor(tenant_id, self.config.get("streaming", {}))
		self.batch_processor = BatchProcessor(tenant_id, self.config.get("batch", {}))
		
		# Pipeline state
		self.active_contexts: Dict[str, ProcessingContext] = {}
		self.performance_history: deque = deque(maxlen=1000)
		
		self._log_pipeline_initialized()
	
	def _log_pipeline_initialized(self) -> None:
		"""Log pipeline initialization"""
		logger.info(f"Advanced processing pipeline initialized for tenant: {self.tenant_id}")
	
	async def process_single(self, request: ProcessingRequest, 
							 config: PipelineConfig = None) -> ProcessingResult:
		"""Process single request through pipeline"""
		context = ProcessingContext(
			request_id=request.id,
			tenant_id=request.tenant_id,
			mode=ProcessingMode.SINGLE,
			config=config or PipelineConfig(name="default")
		)
		
		self.active_contexts[request.id] = context
		
		try:
			result = await self._execute_pipeline(request, context)
			self._update_performance_history(context, result)
			return result
		finally:
			if request.id in self.active_contexts:
				del self.active_contexts[request.id]
	
	async def process_batch_async(self, requests: List[ProcessingRequest], 
								  config: Dict[str, Any] = None) -> List[ProcessingResult]:
		"""Process batch of requests asynchronously"""
		batch = await self.batch_processor.create_batch(requests, config)
		return await self.batch_processor.process_batch(batch)
	
	async def create_streaming_session(self, user_id: str, task_type: NLPTaskType,
									   config: Dict[str, Any] = None) -> StreamingSession:
		"""Create streaming processing session"""
		return await self.streaming_processor.create_session(user_id, task_type, config)
	
	async def _execute_pipeline(self, request: ProcessingRequest, 
								context: ProcessingContext) -> ProcessingResult:
		"""Execute processing pipeline for request"""
		start_time = time.time()
		
		try:
			# Stage 1: Preprocessing
			context.current_stage = PipelineStage.PREPROCESSING
			preprocessed_text = await self._preprocess_stage(request, context)
			context.intermediate_results["preprocessed_text"] = preprocessed_text
			
			# Stage 2: Language Detection
			if context.config.language_detection_enabled:
				context.current_stage = PipelineStage.LANGUAGE_DETECTION
				language_info = await self._language_detection_stage(preprocessed_text, context)
				context.intermediate_results["language_info"] = language_info
			
			# Stage 3: Model Processing (mock)
			context.current_stage = PipelineStage.MODEL_PROCESSING
			model_results = await self._model_processing_stage(request, preprocessed_text, context)
			context.intermediate_results["model_results"] = model_results
			
			# Stage 4: Postprocessing
			context.current_stage = PipelineStage.POSTPROCESSING
			processed_results = await self._postprocessing_stage(model_results, context)
			
			# Stage 5: Validation
			context.current_stage = PipelineStage.VALIDATION
			validated_results = await self._validation_stage(processed_results, context)
			
			# Create final result
			total_time = (time.time() - start_time) * 1000
			
			result = ProcessingResult(
				request_id=request.id,
				tenant_id=request.tenant_id,
				task_type=request.task_type,
				model_used="advanced_pipeline",
				provider_used=ModelProvider.CUSTOM,
				processing_time_ms=total_time * 0.8,  # Actual processing time
				total_time_ms=total_time,
				results=validated_results,
				confidence_score=validated_results.get("confidence", 0.8),
				quality_score=validated_results.get("quality", 0.8),
				status="completed"
			)
			
			return result
			
		except Exception as e:
			# Handle pipeline errors
			total_time = (time.time() - start_time) * 1000
			context.error_history.append({
				"stage": context.current_stage.value if context.current_stage else "unknown",
				"error": str(e),
				"timestamp": datetime.utcnow()
			})
			
			error_result = ProcessingResult(
				request_id=request.id,
				tenant_id=request.tenant_id,
				task_type=request.task_type,
				model_used="advanced_pipeline",
				provider_used=ModelProvider.CUSTOM,
				processing_time_ms=0.0,
				total_time_ms=total_time,
				results={"error": str(e), "stage": context.current_stage.value if context.current_stage else "unknown"},
				status="failed",
				error_message=str(e)
			)
			
			return error_result
	
	async def _preprocess_stage(self, request: ProcessingRequest, 
								context: ProcessingContext) -> str:
		"""Execute preprocessing stage"""
		text = request.text_content or ""
		
		# Apply preprocessing steps
		preprocessed = self.preprocessor.normalize_text(
			text, 
			steps=context.config.preprocessing_steps
		)
		
		# Extract preprocessing features
		features = self.preprocessor.extract_features(text)
		context.metadata["preprocessing_features"] = features
		
		return preprocessed
	
	async def _language_detection_stage(self, text: str, 
										context: ProcessingContext) -> Dict[str, Any]:
		"""Execute language detection stage"""
		language_info = await self.language_detector.detect_language(text)
		
		# Store in context
		context.metadata["detected_language"] = language_info["detected_language"]
		context.metadata["language_confidence"] = language_info["confidence"]
		
		return language_info
	
	async def _model_processing_stage(self, request: ProcessingRequest, 
									  text: str, context: ProcessingContext) -> Dict[str, Any]:
		"""Execute model processing stage (mock implementation)"""
		# Mock model processing based on task type
		await asyncio.sleep(0.01)  # Simulate processing time
		
		if request.task_type == NLPTaskType.SENTIMENT_ANALYSIS:
			return {
				"sentiment": "positive",
				"confidence": 0.85,
				"scores": {"positive": 0.85, "negative": 0.10, "neutral": 0.05}
			}
		elif request.task_type == NLPTaskType.NAMED_ENTITY_RECOGNITION:
			return {
				"entities": [
					{"text": "test", "label": "MISC", "start": 0, "end": 4, "confidence": 0.9}
				]
			}
		else:
			return {
				"result": "processed",
				"confidence": 0.8
			}
	
	async def _postprocessing_stage(self, results: Dict[str, Any], 
									context: ProcessingContext) -> Dict[str, Any]:
		"""Execute postprocessing stage"""
		# Apply postprocessing steps
		processed = results.copy()
		
		for step in context.config.postprocessing_steps:
			if step == "confidence_calibration":
				processed = self._calibrate_confidence(processed)
			elif step == "result_validation":
				processed = self._validate_results(processed)
		
		return processed
	
	async def _validation_stage(self, results: Dict[str, Any], 
								context: ProcessingContext) -> Dict[str, Any]:
		"""Execute validation stage"""
		# Apply validation rules
		validated = results.copy()
		validation_passed = True
		
		for rule in context.config.validation_rules:
			if rule.startswith("confidence_threshold_"):
				threshold = float(rule.split("_")[-1])
				if results.get("confidence", 0) < threshold:
					validation_passed = False
					break
		
		validated["validation_passed"] = validation_passed
		validated["quality"] = 0.85 if validation_passed else 0.6
		
		return validated
	
	def _calibrate_confidence(self, results: Dict[str, Any]) -> Dict[str, Any]:
		"""Apply confidence calibration"""
		calibrated = results.copy()
		original_confidence = results.get("confidence", 0.5)
		
		# Simple calibration: slightly reduce overconfident predictions
		if original_confidence > 0.9:
			calibrated["confidence"] = original_confidence * 0.95
		elif original_confidence < 0.3:
			calibrated["confidence"] = original_confidence * 1.1
		
		return calibrated
	
	def _validate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
		"""Apply result validation"""
		validated = results.copy()
		validated["validation_timestamp"] = datetime.utcnow().isoformat()
		return validated
	
	def _update_performance_history(self, context: ProcessingContext, 
									result: ProcessingResult) -> None:
		"""Update performance history"""
		self.performance_history.append({
			"timestamp": datetime.utcnow(),
			"request_id": context.request_id,
			"mode": context.mode.value,
			"total_time_ms": result.total_time_ms,
			"success": result.is_successful,
			"confidence": result.confidence_score
		})
	
	def get_pipeline_stats(self) -> Dict[str, Any]:
		"""Get comprehensive pipeline statistics"""
		recent_performance = list(self.performance_history)[-100:]  # Last 100 requests
		
		if recent_performance:
			avg_time = sum(p["total_time_ms"] for p in recent_performance) / len(recent_performance)
			success_rate = sum(1 for p in recent_performance if p["success"]) / len(recent_performance)
			avg_confidence = sum(p.get("confidence", 0) for p in recent_performance) / len(recent_performance)
		else:
			avg_time = 0.0
			success_rate = 1.0
			avg_confidence = 0.8
		
		return {
			"active_contexts": len(self.active_contexts),
			"total_processed": len(self.performance_history),
			"recent_performance": {
				"avg_time_ms": round(avg_time, 2),
				"success_rate": round(success_rate * 100, 2),
				"avg_confidence": round(avg_confidence, 3)
			},
			"streaming": self.streaming_processor.get_batch_stats() if hasattr(self.streaming_processor, 'get_batch_stats') else {},
			"batch": self.batch_processor.get_batch_stats()
		}
	
	async def cleanup(self) -> None:
		"""Cleanup pipeline resources"""
		# Close all streaming sessions
		for session_id in list(self.streaming_processor.active_sessions.keys()):
			await self.streaming_processor.close_session(session_id)
		
		# Clear active contexts
		self.active_contexts.clear()
		
		logger.info(f"Advanced processing pipeline cleanup completed for tenant: {self.tenant_id}")

# Export main classes
__all__ = [
	"AdvancedProcessingPipeline", "TextPreprocessor", "LanguageDetector", 
	"DomainProcessor", "StreamingProcessor", "BatchProcessor",
	"PipelineConfig", "ProcessingContext", "BatchRequest",
	"PipelineStage", "ProcessingMode"
]
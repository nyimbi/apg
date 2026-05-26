"""
NLPC Data Models - Natural Language Processing Core

Copyright © 2025 Datacraft
Author: Nyimbi Odero
Email: nyimbi@gmail.com
Website: www.datacraft.co.ke

This module defines core data models for the Natural Language Processing Core (NLPC) capability.
All models follow APG standards with Pydantic v2, async support, and multi-tenancy.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from enum import Enum
from typing import Any, Annotated, Optional
from uuid_extensions import uuid7str

from pydantic import BaseModel, Field, ConfigDict, AfterValidator, field_validator, model_validator


class NLPTask(str, Enum):
	"""Enumeration of supported NLP tasks."""
	TOKENIZATION = "tokenization"
	SENTENCE_SEGMENTATION = "sentence_segmentation"
	LANGUAGE_DETECTION = "language_detection"
	POS_TAGGING = "pos_tagging"
	PART_OF_SPEECH_TAGGING = "part_of_speech_tagging"
	NER = "named_entity_recognition"
	NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
	DEPENDENCY_PARSING = "dependency_parsing"
	CONSTITUENCY_PARSING = "constituency_parsing"
	SENTIMENT_ANALYSIS = "sentiment_analysis"
	ENTITY_EXTRACTION = "entity_extraction"
	EMOTION_DETECTION = "emotion_detection"
	INTENT_CLASSIFICATION = "intent_classification"
	TOPIC_MODELING = "topic_modeling"
	SEMANTIC_SIMILARITY = "semantic_similarity"
	TEXT_SIMILARITY = "text_similarity"
	TEXT_SUMMARIZATION = "text_summarization"
	RELATION_EXTRACTION = "relation_extraction"
	COREFERENCE_RESOLUTION = "coreference_resolution"
	TEMPORAL_EXTRACTION = "temporal_extraction"
	EVENT_EXTRACTION = "event_extraction"
	QUESTION_ANSWERING = "question_answering"
	TEXT_GENERATION = "text_generation"
	TEXT_TRANSLATION = "text_translation"
	PII_DETECTION = "pii_detection"
	TEXT_CLASSIFICATION = "text_classification"
	KEYWORD_EXTRACTION = "keyword_extraction"
	ENTITY_LINKING = "entity_linking"
	TEXT_NORMALIZATION = "text_normalization"
	TEXT_CLUSTERING = "text_clustering"


class NLPTaskType(str, Enum):
	"""Legacy NLP task names retained for older APG tests and callers."""
	SENTIMENT_ANALYSIS = "sentiment_analysis"
	ENTITY_EXTRACTION = "entity_extraction"
	TEXT_CLASSIFICATION = "text_classification"
	TEXT_SUMMARIZATION = "text_summarization"
	LANGUAGE_DETECTION = "language_detection"
	TEXT_SIMILARITY = "text_similarity"
	QUESTION_ANSWERING = "question_answering"
	TEXT_GENERATION = "text_generation"
	NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
	PART_OF_SPEECH_TAGGING = "part_of_speech_tagging"
	DEPENDENCY_PARSING = "dependency_parsing"
	TOPIC_MODELING = "topic_modeling"
	KEYWORD_EXTRACTION = "keyword_extraction"
	TEXT_CLUSTERING = "text_clustering"


class ModelProvider(str, Enum):
	"""Legacy on-device model providers."""
	OLLAMA = "ollama"
	TRANSFORMERS = "transformers"
	SPACY = "spacy"
	NLTK = "nltk"
	CUSTOM = "custom"


class QualityLevel(str, Enum):
	"""Quality versus latency preference for legacy callers."""
	FAST = "fast"
	BALANCED = "balanced"
	ACCURATE = "accurate"
	BEST = "best"


class DocumentType(str, Enum):
	"""Legacy document content types."""
	PLAIN_TEXT = "plain_text"
	HTML = "html"
	MARKDOWN = "markdown"
	JSON = "json"
	XML = "xml"
	PDF = "pdf"
	DOCX = "docx"


class ProcessingStatus(str, Enum):
	"""Enumeration of processing status values."""
	PENDING = "pending"
	PROCESSING = "processing"
	IN_PROGRESS = "in_progress"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"
	RETRY = "retry"


class ModelType(str, Enum):
	"""Enumeration of supported model types."""
	SPACY = "spacy"
	NLTK = "nltk"
	TEXTBLOB = "textblob"
	GENSIM = "gensim"
	TRANSFORMERS = "transformers"
	SKLEARN = "sklearn"
	CUSTOM = "custom"
	ENSEMBLE = "ensemble"


class LanguageCode(str, Enum):
	"""Common language codes (ISO 639-1)."""
	AUTO = "auto"
	ENGLISH = "en"
	EN = "en"
	SPANISH = "es"
	ES = "es"
	FRENCH = "fr"
	FR = "fr"
	GERMAN = "de"
	DE = "de"
	ITALIAN = "it"
	IT = "it"
	PORTUGUESE = "pt"
	PT = "pt"
	RUSSIAN = "ru"
	RU = "ru"
	CHINESE = "zh"
	ZH = "zh"
	JAPANESE = "ja"
	JA = "ja"
	KOREAN = "ko"
	KO = "ko"
	ARABIC = "ar"
	AR = "ar"
	HINDI = "hi"
	HI = "hi"
	AUTO_DETECT = "auto"
	MULTILINGUAL = "multi"
	AFRIKAANS = "af"
	AFAR = "aa"
	AKAN = "ak"
	AMHARIC = "am"
	BAMBARA = "bm"
	EWE = "ee"
	FULAH = "ff"
	HAUSA = "ha"
	IGBO = "ig"
	KANURI = "kr"
	KIKUYU = "ki"
	KINYARWANDA = "rw"
	KIRUNDI = "rn"
	KONGO = "kg"
	LINGALA = "ln"
	LUGANDA = "lg"
	MALAGASY = "mg"
	NYANJA = "ny"
	OROMO = "om"
	SANGO = "sg"
	SHONA = "sn"
	SOMALI = "so"
	SOUTHERN_SOTHO = "st"
	SWAHILI = "sw"
	SWATI = "ss"
	TIGRINYA = "ti"
	TSONGA = "ts"
	TSWANA = "tn"
	TWI = "tw"
	VENDA = "ve"
	WOLOF = "wo"
	XHOSA = "xh"
	YORUBA = "yo"
	ZULU = "zu"
	KABYLE = "kab"
	KAMBA = "kam"
	LUO = "luo"
	MAASAI = "mas"
	MERU = "mer"
	MOORE = "mos"
	NUER = "nus"
	SUKUMA = "suk"
	TAMAZIGHT = "tzm"
	TIGRE = "tig"
	UMBUNDU = "umb"


class PriorityLevel(str, Enum):
	"""Processing priority levels."""
	LOW = "low"
	NORMAL = "medium"
	MEDIUM = "medium"
	HIGH = "high"
	URGENT = "high"
	CRITICAL = "critical"


def _validate_confidence_score(value: float) -> float:
	"""Validate confidence score is between 0 and 1."""
	if not 0.0 <= value <= 1.0:
		raise ValueError("Confidence score must be between 0.0 and 1.0")
	return value


def _validate_positive_float(value: float) -> float:
	"""Validate float is positive."""
	if value < 0:
		raise ValueError("Value must be positive")
	return value


def _validate_non_empty_string(value: str) -> str:
	"""Validate string is not empty."""
	if not value or not value.strip():
		raise ValueError("String cannot be empty")
	return value.strip()


class BaseNLPCModel(BaseModel):
	"""Base model for all NLPC entities with common fields."""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		use_enum_values=True
	)

	id: str = Field(default_factory=uuid7str, description="Unique identifier")
	tenant_id: str = Field(description="Tenant identifier for multi-tenancy")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	version: int = Field(default=1, ge=1, description="Version number for optimistic locking")


class TextDocument(BaseModel):
	"""Legacy rich text document with metadata and processing hints."""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		use_enum_values=True
	)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	content: str
	title: Optional[str] = Field(default=None, max_length=500)
	language: Optional[LanguageCode] = None
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)

	@field_validator('content')
	@classmethod
	def validate_content(cls, value: str) -> str:
		if not value or not value.strip():
			raise ValueError("Content cannot be empty")
		if len(value) > 10_000_000:
			raise ValueError("Content exceeds maximum length")
		return value.strip()

	@property
	def estimated_processing_time(self) -> float:
		return 0.1 + (len(self.content) / 1_000_000)


class NLPDocument(BaseNLPCModel):
	"""Model representing a document to be processed."""
	
	document_id: str = Field(default_factory=uuid7str, description="Unique document identifier")
	content: str = Field(
		description="Text content to be processed",
		max_length=10_000_000  # 10MB text limit
	)
	language: Optional[LanguageCode] = Field(
		default=None,
		description="Document language (auto-detected if not specified)"
	)
	metadata: dict[str, Any] = Field(
		default_factory=dict,
		description="Additional document metadata"
	)
	source: Optional[str] = Field(default=None, description="Document source system")
	source_id: Optional[str] = Field(default=None, description="Original document ID in source system")
	processing_history: list['ProcessingRecord'] = Field(
		default_factory=list,
		description="History of processing operations"
	)
	content_hash: Optional[str] = Field(default=None, description="Content hash for deduplication")
	content_type: str = Field(default="text/plain", description="MIME type of content")
	word_count: Optional[int] = Field(default=None, ge=0, description="Approximate word count")
	char_count: Optional[int] = Field(default=None, ge=0, description="Character count")
	is_sensitive: bool = Field(default=False, description="Contains sensitive/PII data")
	retention_days: Optional[int] = Field(default=None, ge=1, description="Data retention period in days")

	@field_validator('content')
	@classmethod
	def validate_document_content(cls, value: str) -> str:
		if not value or not value.strip():
			raise ValueError("content cannot be empty")
		return value.strip()


class ProcessingRequest(BaseNLPCModel):
	"""Model for NLP processing requests."""
	
	request_id: str = Field(default_factory=uuid7str, description="Unique request identifier")
	document_id: Optional[str] = Field(default=None, description="Document to process")
	tasks: list[NLPTask] = Field(default_factory=list, description="List of NLP tasks to perform")
	priority: PriorityLevel = Field(default=PriorityLevel.NORMAL, description="Processing priority")
	model_preferences: dict[str, str] = Field(
		default_factory=dict,
		description="Preferred models for specific tasks"
	)
	parameters: dict[str, Any] = Field(
		default_factory=dict,
		description="Task-specific parameters"
	)
	callback_url: Optional[str] = Field(default=None, description="Webhook URL for completion notification")
	max_processing_time: Optional[int] = Field(
		default=None,
		ge=1,
		description="Maximum processing time in seconds"
	)
	require_explanation: bool = Field(default=False, description="Include model explanations")
	batch_id: Optional[str] = Field(default=None, description="Batch identifier for grouped processing")
	user_id: Optional[str] = Field(default=None, description="User requesting processing")
	task_type: Optional[Any] = Field(default=None, description="Legacy single-task request type")
	text_content: Optional[str] = Field(default=None, description="Legacy inline text content")
	preferred_model: Optional[str] = Field(default=None, description="Legacy preferred model identifier")
	quality_level: Optional[Any] = Field(default=None, description="Legacy quality preference")
	options: dict[str, Any] = Field(default_factory=dict, description="Legacy task options")
	performance_requirements: dict[str, Any] = Field(default_factory=dict, description="Performance requirements")
	fallback_enabled: bool = Field(default=True, description="Whether fallback processing is allowed")

	@model_validator(mode='after')
	def validate_processing_target(self) -> 'ProcessingRequest':
		if not self.tasks and self.task_type is None:
			raise ValueError("at least one task must be specified")
		if self.task_type is not None and not self.text_content and not self.document_id:
			raise ValueError("Either text_content or document_id must be provided")
		if self.text_content is not None and len(self.text_content) > 1_000_000:
			raise ValueError("Direct text content exceeds 1MB limit")
		if self.document_id is None and self.text_content is not None:
			self.document_id = uuid7str()
		return self


class ProcessingResult(BaseNLPCModel):
	"""Model representing the result of an NLP processing operation."""
	
	result_id: str = Field(default_factory=uuid7str, description="Unique result identifier")
	request_id: str = Field(default_factory=uuid7str, description="Reference to processing request")
	document_id: str = Field(default="", description="Reference to processed document")
	task: Optional[Any] = Field(default=None, description="Legacy task field")
	task_type: Any = Field(default=NLPTask.TEXT_CLASSIFICATION, description="The NLP task that was performed")
	status: ProcessingStatus = Field(default=ProcessingStatus.COMPLETED, description="Processing status")
	confidence_score: Optional[float] = Field(
		default=None,
		description="Confidence score for the result",
		ge=0.0,
		le=1.0
	)
	processing_time: Annotated[float, AfterValidator(_validate_positive_float)] = Field(
		default=0.0,
		description="Processing time in seconds",
		ge=0.0
	)
	result_data: dict[str, Any] = Field(default_factory=dict, description="Actual processing results")
	model_version: str = Field(default="1.0", description="Version of the model used")
	model_type: ModelType = Field(default=ModelType.CUSTOM, description="Type of model used")
	error_message: Optional[str] = Field(default=None, description="Error message if processing failed")
	explanation: Optional[dict[str, Any]] = Field(
		default=None,
		description="Model explanation if requested"
	)
	performance_metrics: dict[str, float] = Field(
		default_factory=dict,
		description="Performance metrics for this operation"
	)
	cache_key: Optional[str] = Field(default=None, description="Cache key for result reuse")
	model_used: Optional[str] = Field(default=None, description="Legacy model identifier")
	provider_used: Optional[Any] = Field(default=None, description="Legacy provider identifier")
	processing_time_ms: float = Field(default=0.0, ge=0.0, description="Legacy processing time in ms")
	total_time_ms: float = Field(default=0.0, ge=0.0, description="Legacy total time in ms")
	results: dict[str, Any] = Field(default_factory=dict, description="Legacy result payload")
	context_used: bool = Field(default=False, description="Whether context was applied")
	security_applied: bool = Field(default=False, description="Whether security controls were applied")
	encryption_applied: bool = Field(default=False, description="Whether encryption controls were applied")
	cache_used: bool = Field(default=False, description="Whether a cached result was used")
	optimization_applied: bool = Field(default=False, description="Whether performance optimization was applied")

	@property
	def is_successful(self) -> bool:
		"""Whether the processing request completed successfully."""
		return self.status == ProcessingStatus.COMPLETED or self.status == ProcessingStatus.COMPLETED.value

	@property
	def performance_rating(self) -> str:
		if self.processing_time_ms < 50:
			return "excellent"
		if self.processing_time_ms < 150:
			return "good"
		if self.processing_time_ms < 500:
			return "acceptable"
		return "poor"

	@model_validator(mode='after')
	def sync_task_fields(self) -> 'ProcessingResult':
		if self.task is not None:
			self.task_type = self.task
		else:
			self.task = self.task_type
		if not self.result_data and self.results:
			self.result_data = self.results
		if not self.results and self.result_data:
			self.results = self.result_data
		return self


class ProcessingRecord(BaseModel):
	"""Record of a processing operation for audit trail."""
	model_config = ConfigDict(extra='allow', validate_by_name=True, validate_by_alias=True, use_enum_values=True)
	
	record_id: str = Field(default_factory=uuid7str, description="Unique record identifier")
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Processing timestamp")
	task: Optional[NLPTask] = Field(default=None, description="Legacy task field")
	task_type: Optional[NLPTask] = Field(default=None, description="Type of NLP task performed")
	status: ProcessingStatus = Field(default=ProcessingStatus.COMPLETED, description="Processing status")
	model_used: str = Field(default="", description="Model identifier used for processing")
	results: dict[str, Any] = Field(default_factory=dict, description="Legacy processing results")
	processing_time: float = Field(default=0.0, ge=0.0, description="Processing time in seconds")
	input_size: int = Field(default=0, ge=0, description="Size of input in characters")
	result_id: Optional[str] = Field(default=None, description="Reference to processing result")
	error_code: Optional[str] = Field(default=None, description="Error code if failed")
	user_id: Optional[str] = Field(default=None, description="User who initiated processing")

	@model_validator(mode='after')
	def sync_task_fields(self) -> 'ProcessingRecord':
		if self.task is not None and self.task_type is None:
			self.task_type = self.task
		if self.task is None and self.task_type is not None:
			self.task = self.task_type
		return self


class ModelConfiguration(BaseNLPCModel):
	"""Configuration for NLP models."""
	
	config_id: str = Field(default_factory=uuid7str, description="Unique configuration identifier")
	name: Annotated[str, AfterValidator(_validate_non_empty_string)] = Field(
		default="default",
		description="Configuration name",
		min_length=1,
		max_length=100
	)
	model_type: ModelType = Field(description="Type of model")
	model_name: str = Field(default="default", description="Specific model name/path")
	language: Optional[LanguageCode] = Field(default=None, description="Primary model language")
	supported_tasks: list[NLPTask] = Field(default_factory=lambda: [NLPTask.TEXT_CLASSIFICATION], description="Tasks this model supports", min_items=1)
	supported_languages: list[LanguageCode] = Field(
		default_factory=lambda: [LanguageCode.EN],
		description="Languages this model supports",
		min_items=1
	)
	configuration: dict[str, Any] = Field(
		default_factory=dict,
		description="Model-specific configuration parameters"
	)
	performance_metrics: dict[str, float] = Field(
		default_factory=dict,
		description="Model performance benchmarks"
	)
	memory_requirements: Optional[int] = Field(
		default=None,
		ge=0,
		description="Memory requirements in MB"
	)
	gpu_required: bool = Field(default=False, description="Whether GPU is required")
	is_active: bool = Field(default=True, description="Whether this configuration is active")
	load_priority: int = Field(default=50, ge=1, le=100, description="Model loading priority")


class ModelConfig(BaseModel):
	"""Legacy service-level model configuration."""
	model_config = ConfigDict(extra='allow', validate_by_name=True, validate_by_alias=True)

	enable_ollama: bool = True
	enable_transformers: bool = True
	enable_spacy: bool = True
	default_quality_level: QualityLevel = QualityLevel.BALANCED
	max_concurrent_requests: int = 10
	model_cache_size: int = 5


class NLPModel(BaseModel):
	"""Legacy NLP model metadata used by migrated tests and callers."""
	model_config = ConfigDict(extra='allow', validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str
	model_key: str
	provider: ModelProvider
	provider_model_name: str
	version: str = "1.0.0"
	supported_tasks: list[NLPTaskType] = Field(default_factory=list)
	supported_languages: list[LanguageCode] = Field(default_factory=list)
	max_input_length: Optional[int] = None
	context_window: Optional[int] = None
	is_active: bool = True
	is_loaded: bool = True
	health_status: str = "unknown"
	successful_requests: int = 0
	failed_requests: int = 0

	@property
	def is_available(self) -> bool:
		return self.is_active and self.is_loaded and self.health_status in {"healthy", "unknown"}

	@property
	def success_rate(self) -> float:
		total = self.successful_requests + self.failed_requests
		return (self.successful_requests / total * 100) if total else 0.0


class StreamingSession(BaseModel):
	"""Legacy streaming session state."""
	model_config = ConfigDict(extra='allow', validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	user_id: str
	task_type: NLPTaskType
	chunk_size: int = Field(default=1000, ge=100, le=10000)
	overlap_size: int = 0
	status: str = "active"
	chunks_processed: int = 0
	total_characters: int = 0
	average_latency_ms: float = 0.0
	created_at: datetime = Field(default_factory=datetime.utcnow)

	@property
	def is_connected(self) -> bool:
		return self.status == "active"


class StreamingChunk(BaseModel):
	"""Legacy streaming text chunk."""
	model_config = ConfigDict(extra='allow', validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	session_id: str
	sequence_number: int
	text_content: Annotated[str, AfterValidator(_validate_non_empty_string)]
	start_position: int
	end_position: int
	status: ProcessingStatus = ProcessingStatus.PENDING
	results: Optional[dict[str, Any]] = None


class SystemHealth(BaseModel):
	"""Legacy aggregate service health model."""
	model_config = ConfigDict(extra='allow', validate_by_name=True, validate_by_alias=True)

	tenant_id: str
	overall_status: str
	component_status: dict[str, Any] = Field(default_factory=dict)
	components: dict[str, Any] = Field(default_factory=dict)
	average_response_time_ms: float = 0.0
	requests_per_minute: int = 0
	active_sessions: int = 0
	queue_depth: int = 0
	cpu_usage_percent: float = 0.0
	memory_usage_percent: float = 0.0
	disk_usage_percent: float = 0.0
	total_models: int = 0
	active_models: int = 0
	loaded_models: int = 0
	failed_models: int = 0
	error_messages: list[str] = Field(default_factory=list)
	last_check: datetime = Field(default_factory=datetime.utcnow)
	timestamp: datetime = Field(default_factory=datetime.utcnow)

	@property
	def model_availability_percent(self) -> float:
		return (self.loaded_models / self.total_models * 100) if self.total_models else 0.0

	@property
	def performance_rating(self) -> str:
		if (
			self.average_response_time_ms < 100
			and self.cpu_usage_percent < 70
			and self.memory_usage_percent < 80
			and self.queue_depth <= 5
		):
			return "excellent"
		if self.average_response_time_ms < 250 and self.cpu_usage_percent < 85 and self.memory_usage_percent < 90:
			return "good"
		if self.average_response_time_ms < 500 and self.cpu_usage_percent < 95 and self.memory_usage_percent < 95:
			return "acceptable"
		return "poor"


class ContextSession(BaseNLPCModel):
	"""Session for maintaining context across multiple processing requests."""
	
	session_id: str = Field(default_factory=uuid7str, description="Unique session identifier")
	name: Optional[str] = Field(default=None, description="Human-readable session name")
	context_type: str = Field(default="conversation", description="Type of context being maintained")
	context_data: list[dict[str, Any]] = Field(
		default_factory=list,
		description="Context information"
	)
	max_context_length: int = Field(default=10000, gt=0, description="Maximum context length")
	ttl_seconds: int = Field(default=3600, ge=60, description="Session TTL in seconds")
	last_accessed: datetime = Field(default_factory=datetime.utcnow, description="Last access time")
	is_active: bool = Field(default=True, description="Whether session is active")
	document_ids: list[str] = Field(
		default_factory=list,
		description="Documents associated with this session"
	)
	user_id: Optional[str] = Field(default=None, description="Legacy user owner")
	session_name: Optional[str] = Field(default=None, description="Legacy session name")
	context_window_size: int = Field(default=10, ge=1, description="Legacy context window size")
	enable_learning: bool = Field(default=True, description="Whether context learning is enabled")
	learning_rate: float = Field(default=0.1, ge=0.0, description="Context learning rate")
	context_decay_rate: float = Field(default=0.05, ge=0.0, description="Context decay rate")
	max_context_age_hours: int = Field(default=24, ge=1, description="Maximum retained context age")
	memory_retention_hours: int = Field(default=24, gt=0, description="Legacy memory retention")
	session_metadata: dict[str, Any] = Field(default_factory=dict, description="Legacy session metadata")
	context_history: list[dict[str, Any]] = Field(default_factory=list, description="Context history")
	performance_history: list[dict[str, Any]] = Field(default_factory=list, description="Context performance history")


class TextAnnotation(BaseModel):
	"""Annotation attached to a text document."""
	model_config = ConfigDict(extra='allow', validate_by_name=True, validate_by_alias=True, use_enum_values=True)

	id: str = Field(default_factory=uuid7str)
	document_id: Optional[str] = None
	annotator_id: Optional[str] = None
	annotation_type: Optional[Any] = None
	label: Optional[str] = None
	start_position: Optional[int] = None
	end_position: Optional[int] = None
	confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
	metadata: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=datetime.utcnow)


class AnnotationProject(BaseNLPCModel):
	"""Human annotation project for supervised NLP workflows."""

	name: Annotated[str, AfterValidator(_validate_non_empty_string)]
	description: Optional[str] = None
	annotation_type: Any
	team_members: list[str] = Field(default_factory=list)
	project_manager: str
	annotation_schema: dict[str, Any] = Field(default_factory=dict)
	status: str = "planning"
	consensus_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
	document_count: int = Field(default=0, ge=0)
	completed_annotations: int = Field(default=0, ge=0)

	@property
	def completion_percentage(self) -> float:
		return (self.completed_annotations / self.document_count * 100) if self.document_count else 0.0


class ModelTrainingConfig(BaseModel):
	"""Configuration for training or fine-tuning NLP models."""
	model_config = ConfigDict(extra='allow', validate_by_name=True, validate_by_alias=True, use_enum_values=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	name: str = "training-config"
	base_model: Optional[str] = None
	task_type: Optional[Any] = None
	training_parameters: dict[str, Any] = Field(default_factory=dict)
	dataset_ids: list[str] = Field(default_factory=list)
	created_at: datetime = Field(default_factory=datetime.utcnow)


class TextAnalytics(BaseModel):
	"""Aggregated analytics for a text corpus."""
	model_config = ConfigDict(extra='allow', validate_by_name=True, validate_by_alias=True)

	id: str = Field(default_factory=uuid7str)
	tenant_id: str
	document_count: int = Field(default=0, ge=0)
	token_count: int = Field(default=0, ge=0)
	language_distribution: dict[str, int] = Field(default_factory=dict)
	task_metrics: dict[str, Any] = Field(default_factory=dict)
	created_at: datetime = Field(default_factory=datetime.utcnow)


class BatchProcessingJob(BaseNLPCModel):
	"""Model for batch processing jobs."""
	
	job_id: str = Field(default_factory=uuid7str, description="Unique job identifier")
	name: Annotated[str, AfterValidator(_validate_non_empty_string)] = Field(
		description="Job name",
		min_length=1,
		max_length=200
	)
	description: Optional[str] = Field(default=None, description="Job description")
	status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Job status")
	document_ids: list[str] = Field(description="Documents to process", min_items=1)
	tasks: list[NLPTask] = Field(description="NLP tasks to perform", min_items=1)
	priority: PriorityLevel = Field(default=PriorityLevel.NORMAL, description="Processing priority")
	progress: float = Field(default=0.0, ge=0.0, le=100.0, description="Progress percentage")
	total_documents: int = Field(ge=1, description="Total number of documents")
	processed_documents: int = Field(default=0, ge=0, description="Number of processed documents")
	failed_documents: int = Field(default=0, ge=0, description="Number of failed documents")
	started_at: Optional[datetime] = Field(default=None, description="Job start time")
	completed_at: Optional[datetime] = Field(default=None, description="Job completion time")
	estimated_completion: Optional[datetime] = Field(default=None, description="Estimated completion time")
	results_location: Optional[str] = Field(default=None, description="Location of results")
	error_summary: Optional[str] = Field(default=None, description="Summary of errors")
	configuration: dict[str, Any] = Field(
		default_factory=dict,
		description="Job configuration parameters"
	)


# Model relationships and forward references
NLPDocument.model_rebuild()
ProcessingRequest.model_rebuild()
ProcessingResult.model_rebuild()
BatchProcessingJob.model_rebuild()


def _log_model_validation_error(model_name: str, error: Exception) -> str:
	"""Log model validation errors for debugging."""
	error_msg = f"Model validation error in {model_name}: {str(error)}"
	print(f"[NLPC Models] {error_msg}")
	return error_msg


async def validate_models_async() -> dict[str, bool]:
	"""
	Async validation of all model classes.
	
	Returns:
		Dictionary mapping model names to validation status
	"""
	models_to_test = [
		('NLPDocument', NLPDocument),
		('ProcessingRequest', ProcessingRequest),
		('ProcessingResult', ProcessingResult),
		('ProcessingRecord', ProcessingRecord),
		('ModelConfiguration', ModelConfiguration),
		('ContextSession', ContextSession),
		('BatchProcessingJob', BatchProcessingJob)
	]
	
	validation_results = {}
	
	for model_name, model_class in models_to_test:
		try:
			# Test basic instantiation
			if model_name == 'NLPDocument':
				test_instance = model_class(
					tenant_id="test_tenant",
					content="Test document content"
				)
			elif model_name == 'ProcessingRequest':
				test_instance = model_class(
					tenant_id="test_tenant",
					document_id="test_doc_id",
					tasks=[NLPTask.TOKENIZATION]
				)
			elif model_name == 'ProcessingResult':
				test_instance = model_class(
					tenant_id="test_tenant",
					request_id="test_request",
					document_id="test_doc",
					task_type=NLPTask.TOKENIZATION,
					status=ProcessingStatus.COMPLETED,
					confidence_score=0.95,
					processing_time=0.1,
					result_data={"tokens": ["test"]},
					model_version="1.0",
					model_type=ModelType.SPACY
				)
			elif model_name == 'BatchProcessingJob':
				test_instance = model_class(
					tenant_id="test_tenant",
					name="Test Job",
					document_ids=["doc1"],
					tasks=[NLPTask.TOKENIZATION],
					total_documents=1
				)
			else:
				# For other models, try with minimal required fields
				test_instance = model_class(tenant_id="test_tenant")
			
			# Test serialization/deserialization
			dict_data = test_instance.model_dump()
			restored_instance = model_class.model_validate(dict_data)
			
			validation_results[model_name] = True
			
		except Exception as e:
			_log_model_validation_error(model_name, e)
			validation_results[model_name] = False
	
	return validation_results


def _log_successful_model_load() -> None:
	"""Log successful model loading."""
	print("[NLPC Models] All NLP data models loaded successfully")


# Initialize models on module load
_log_successful_model_load()

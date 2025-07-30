"""
APG Natural Language Processing Models

Pydantic v2 models for enterprise NLP platform with multi-model orchestration,
real-time streaming, collaborative annotation, and domain adaptation.

All models follow APG coding standards with async patterns, modern typing,
and comprehensive validation.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Literal
from enum import Enum
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, field_validator, computed_field
from pydantic.types import Json
from typing import Annotated, get_type_hints

# APG Model Configuration
MODEL_CONFIG = ConfigDict(
	extra='forbid',
	validate_by_name=True,
	validate_by_alias=True,
	str_strip_whitespace=True,
	validate_default=True,
	use_enum_values=True
)

# Enumerations for Type Safety

class NLPTaskType(str, Enum):
	"""NLP task types for processing classification"""
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
	"""On-device model providers"""
	OLLAMA = "ollama"
	TRANSFORMERS = "transformers"
	SPACY = "spacy" 
	NLTK = "nltk"
	CUSTOM = "custom"

class ProcessingStatus(str, Enum):
	"""Processing status for async operations"""
	PENDING = "pending"
	PROCESSING = "processing"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"

class QualityLevel(str, Enum):
	"""Quality vs speed tradeoff levels"""
	FAST = "fast"
	BALANCED = "balanced"
	ACCURATE = "accurate"
	BEST = "best"

class DocumentType(str, Enum):
	"""Document content types"""
	PLAIN_TEXT = "plain_text"
	HTML = "html"
	MARKDOWN = "markdown"
	JSON = "json"
	XML = "xml"
	PDF = "pdf"
	DOCX = "docx"

class LanguageCode(str, Enum):
	"""Supported language codes"""
	AUTO = "auto"  # Automatic detection
	EN = "en"      # English
	ES = "es"      # Spanish
	FR = "fr"      # French
	DE = "de"      # German
	IT = "it"      # Italian
	PT = "pt"      # Portuguese
	RU = "ru"      # Russian
	ZH = "zh"      # Chinese
	JA = "ja"      # Japanese
	KO = "ko"      # Korean
	AR = "ar"      # Arabic
	HI = "hi"      # Hindi

# Core Data Models

class TextDocument(BaseModel):
	"""Rich text document with metadata and processing history"""
	model_config = MODEL_CONFIG
	
	# Identity and tenant isolation
	id: str = Field(default_factory=uuid7str, description="Unique document identifier")
	tenant_id: str = Field(..., description="Tenant ID for multi-tenancy")
	
	# Document content
	content: str = Field(..., min_length=1, description="Document text content")
	title: Optional[str] = Field(None, max_length=500, description="Document title")
	language: Optional[LanguageCode] = Field(None, description="Document language")
	detected_language: Optional[LanguageCode] = Field(None, description="Auto-detected language")
	content_type: DocumentType = Field(DocumentType.PLAIN_TEXT, description="Content format type")
	
	# Document metadata
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata")
	source_url: Optional[str] = Field(None, description="Source URL if applicable")
	author: Optional[str] = Field(None, description="Document author")
	created_date: Optional[datetime] = Field(None, description="Original creation date")
	
	# Processing information
	processing_history: List[str] = Field(default_factory=list, description="Processing step IDs")
	quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Content quality score")
	word_count: Optional[int] = Field(None, ge=0, description="Word count")
	character_count: Optional[int] = Field(None, ge=0, description="Character count")
	
	# APG audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: Optional[str] = Field(None, description="User ID who created document") 
	updated_by: Optional[str] = Field(None, description="User ID who last updated document")
	
	@field_validator('content')
	@classmethod
	def validate_content_length(cls, v: str) -> str:
		"""Validate content is not empty and within reasonable limits"""
		if len(v.strip()) == 0:
			raise ValueError("Content cannot be empty")
		if len(v) > 10_000_000:  # 10MB limit
			raise ValueError("Content exceeds maximum length of 10MB")
		return v
	
	@computed_field
	@property
	def estimated_processing_time(self) -> float:
		"""Estimate processing time in seconds based on content length"""
		base_time = 0.1
		length_factor = len(self.content) / 1000  # 1ms per 1000 characters
		return base_time + (length_factor * 0.001)
	
	def _log_document_created(self) -> None:
		"""Log document creation for audit trail"""
		print(f"Document created: {self.id} ({len(self.content)} chars, {self.language})")

class NLPModel(BaseModel):
	"""On-device NLP model configuration and metadata"""
	model_config = MODEL_CONFIG
	
	# Model identity
	id: str = Field(default_factory=uuid7str, description="Unique model identifier")
	tenant_id: str = Field(..., description="Tenant ID for multi-tenancy")
	name: str = Field(..., min_length=1, max_length=200, description="Model display name")
	model_key: str = Field(..., description="Unique model key for identification")
	version: str = Field("1.0.0", description="Model version")
	
	# Model provider and configuration
	provider: ModelProvider = Field(..., description="Model provider type")
	provider_model_name: str = Field(..., description="Provider-specific model name")
	model_path: Optional[str] = Field(None, description="Local model path")
	config_params: Dict[str, Any] = Field(default_factory=dict, description="Model configuration")
	
	# Model capabilities
	supported_tasks: List[NLPTaskType] = Field(..., description="Supported NLP tasks")
	supported_languages: List[LanguageCode] = Field(..., description="Supported languages")
	max_input_length: Optional[int] = Field(None, ge=1, description="Maximum input length")
	context_window: Optional[int] = Field(None, ge=1, description="Context window size")
	
	# Performance characteristics
	average_latency_ms: float = Field(0.0, ge=0.0, description="Average processing latency")
	throughput_docs_per_minute: int = Field(0, ge=0, description="Processing throughput")
	memory_usage_mb: float = Field(0.0, ge=0.0, description="Memory usage in MB")
	accuracy_score: float = Field(0.0, ge=0.0, le=1.0, description="Model accuracy score")
	
	# Model status and health
	is_active: bool = Field(True, description="Is model active and available")
	is_loaded: bool = Field(False, description="Is model loaded in memory")
	health_status: Literal["healthy", "degraded", "unhealthy", "unknown"] = Field("unknown")
	last_health_check: Optional[datetime] = Field(None, description="Last health check time")
	
	# Usage statistics
	total_requests: int = Field(0, ge=0, description="Total processing requests")
	successful_requests: int = Field(0, ge=0, description="Successful requests")
	failed_requests: int = Field(0, ge=0, description="Failed requests")
	
	# APG audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: Optional[str] = Field(None)
	updated_by: Optional[str] = Field(None)
	
	@computed_field
	@property
	def success_rate(self) -> float:
		"""Calculate model success rate percentage"""
		total = self.successful_requests + self.failed_requests
		if total == 0:
			return 0.0
		return (self.successful_requests / total) * 100
	
	@computed_field
	@property
	def is_available(self) -> bool:
		"""Check if model is available for processing"""
		return (self.is_active and 
				self.is_loaded and 
				self.health_status in ["healthy", "degraded"])
	
	def _log_model_status_change(self, old_status: str, new_status: str) -> None:
		"""Log model status changes for monitoring"""
		print(f"Model {self.name} status changed: {old_status} -> {new_status}")

class ProcessingRequest(BaseModel):
	"""Request for NLP processing with configuration options"""
	model_config = MODEL_CONFIG
	
	# Request identity
	id: str = Field(default_factory=uuid7str, description="Unique request identifier")
	tenant_id: str = Field(..., description="Tenant ID for multi-tenancy")
	user_id: Optional[str] = Field(None, description="User ID who made request")
	session_id: Optional[str] = Field(None, description="Session identifier")
	
	# Processing configuration
	task_type: NLPTaskType = Field(..., description="Type of NLP task to perform")
	document_id: Optional[str] = Field(None, description="Document ID if processing existing document")
	text_content: Optional[str] = Field(None, description="Direct text content to process")
	language: Optional[LanguageCode] = Field(None, description="Text language hint")
	quality_level: QualityLevel = Field(QualityLevel.BALANCED, description="Quality vs speed preference")
	
	# Model selection preferences
	preferred_model: Optional[str] = Field(None, description="Preferred model ID")
	preferred_provider: Optional[ModelProvider] = Field(None, description="Preferred provider")
	fallback_enabled: bool = Field(True, description="Enable model fallback on failure")
	
	# Processing options
	parameters: Dict[str, Any] = Field(default_factory=dict, description="Task-specific parameters")
	timeout_seconds: int = Field(300, ge=1, le=3600, description="Processing timeout")
	priority: Literal["low", "normal", "high", "urgent"] = Field("normal", description="Request priority")
	
	# Output options
	include_confidence: bool = Field(True, description="Include confidence scores")
	include_explanations: bool = Field(False, description="Include processing explanations")
	output_format: Literal["json", "xml", "text"] = Field("json", description="Output format")
	
	# APG audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: Optional[str] = Field(None)
	
	@field_validator('text_content')
	@classmethod
	def validate_text_or_document(cls, v: Optional[str], info) -> Optional[str]:
		"""Ensure either text_content or document_id is provided"""
		document_id = info.data.get('document_id')
		if not v and not document_id:
			raise ValueError("Either text_content or document_id must be provided")
		if v and len(v) > 1_000_000:  # 1MB limit for direct text
			raise ValueError("Direct text content exceeds 1MB limit")
		return v
	
	def _log_request_created(self) -> None:
		"""Log request creation for audit trail"""
		print(f"Processing request created: {self.id} ({self.task_type}, {self.quality_level})")

class ProcessingResult(BaseModel):
	"""Comprehensive NLP processing result with metadata"""
	model_config = MODEL_CONFIG
	
	# Result identity
	id: str = Field(default_factory=uuid7str, description="Unique result identifier")
	request_id: str = Field(..., description="Originating request ID")
	tenant_id: str = Field(..., description="Tenant ID for multi-tenancy")
	
	# Processing metadata
	task_type: NLPTaskType = Field(..., description="Task type that was performed")
	model_used: str = Field(..., description="Model ID that processed the request")
	provider_used: ModelProvider = Field(..., description="Provider that was used")
	language_detected: Optional[LanguageCode] = Field(None, description="Detected language")
	
	# Processing performance
	processing_time_ms: float = Field(..., ge=0.0, description="Processing time in milliseconds")
	queue_time_ms: float = Field(0.0, ge=0.0, description="Time spent in queue")
	total_time_ms: float = Field(..., ge=0.0, description="Total request time")
	confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall confidence")
	
	# Processing results
	results: Dict[str, Any] = Field(..., description="Task-specific results")
	metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
	explanations: Optional[List[str]] = Field(None, description="Processing explanations")
	warnings: List[str] = Field(default_factory=list, description="Processing warnings")
	
	# Quality metrics
	quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Result quality score")
	completeness_score: float = Field(0.0, ge=0.0, le=1.0, description="Completeness score")
	
	# Status and error handling
	status: ProcessingStatus = Field(ProcessingStatus.COMPLETED, description="Processing status")
	error_message: Optional[str] = Field(None, description="Error message if failed")
	error_code: Optional[str] = Field(None, description="Error code if failed")
	
	# APG audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	processed_by: Optional[str] = Field(None, description="User context for processing")
	
	@computed_field
	@property
	def is_successful(self) -> bool:
		"""Check if processing was successful"""
		return self.status == ProcessingStatus.COMPLETED and self.error_message is None
	
	@computed_field
	@property
	def performance_rating(self) -> Literal["excellent", "good", "acceptable", "poor"]:
		"""Performance rating based on processing time"""
		if self.processing_time_ms < 50:
			return "excellent"
		elif self.processing_time_ms < 100:
			return "good"
		elif self.processing_time_ms < 500:
			return "acceptable"
		else:
			return "poor"
	
	def _log_processing_completed(self) -> None:
		"""Log processing completion for monitoring"""
		print(f"Processing completed: {self.id} ({self.processing_time_ms}ms, {self.confidence_score:.2f})")

# Specialized Result Models

class SentimentResult(BaseModel):
	"""Sentiment analysis specific results"""
	model_config = MODEL_CONFIG
	
	sentiment: Literal["positive", "negative", "neutral"] = Field(..., description="Overall sentiment")
	confidence: float = Field(..., ge=0.0, le=1.0, description="Sentiment confidence")
	scores: Dict[str, float] = Field(..., description="Detailed sentiment scores")
	aspects: Optional[List[Dict[str, Any]]] = Field(None, description="Aspect-based sentiment")
	emotions: Optional[Dict[str, float]] = Field(None, description="Emotion detection scores")

class EntityResult(BaseModel):
	"""Named entity recognition results"""
	model_config = MODEL_CONFIG
	
	entities: List[Dict[str, Any]] = Field(..., description="Extracted entities")
	entity_types_found: List[str] = Field(..., description="Types of entities found")
	total_entities: int = Field(..., ge=0, description="Total number of entities")
	confidence_distribution: Dict[str, int] = Field(..., description="Confidence score distribution")

class ClassificationResult(BaseModel):
	"""Text classification results"""
	model_config = MODEL_CONFIG
	
	predicted_class: str = Field(..., description="Predicted class label")
	confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
	class_probabilities: Dict[str, float] = Field(..., description="All class probabilities")
	top_k_classes: List[Dict[str, float]] = Field(..., description="Top K most likely classes")

# Streaming and Real-time Models

class StreamingSession(BaseModel):
	"""Real-time streaming processing session"""
	model_config = MODEL_CONFIG
	
	# Session identity
	id: str = Field(default_factory=uuid7str, description="Unique session identifier")
	tenant_id: str = Field(..., description="Tenant ID for multi-tenancy")
	user_id: str = Field(..., description="User ID who created session")
	
	# Session configuration
	task_type: NLPTaskType = Field(..., description="Task type for streaming")
	model_id: Optional[str] = Field(None, description="Preferred model for processing")
	language: Optional[LanguageCode] = Field(None, description="Expected language")
	
	# Streaming parameters
	chunk_size: int = Field(1000, ge=100, le=10000, description="Text chunk size")
	overlap_size: int = Field(100, ge=0, le=1000, description="Chunk overlap size")
	aggregation_window_ms: int = Field(5000, ge=1000, le=60000, description="Result aggregation window")
	
	# Session status
	status: Literal["active", "paused", "stopped", "error"] = Field("active", description="Session status")
	is_connected: bool = Field(True, description="WebSocket connection status")
	connection_id: Optional[str] = Field(None, description="WebSocket connection ID")
	
	# Processing metrics
	chunks_processed: int = Field(0, ge=0, description="Number of chunks processed")
	total_characters: int = Field(0, ge=0, description="Total characters processed")
	average_latency_ms: float = Field(0.0, ge=0.0, description="Average processing latency")
	
	# APG audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	last_activity: datetime = Field(default_factory=datetime.utcnow)
	
	def _log_session_activity(self, activity: str) -> None:
		"""Log session activity for monitoring"""
		print(f"Streaming session {self.id}: {activity}")

class StreamingChunk(BaseModel):
	"""Individual chunk in streaming processing"""
	model_config = MODEL_CONFIG
	
	# Chunk identity
	id: str = Field(default_factory=uuid7str, description="Unique chunk identifier")
	session_id: str = Field(..., description="Parent streaming session ID")
	sequence_number: int = Field(..., ge=0, description="Chunk sequence number")
	
	# Chunk content
	text_content: str = Field(..., min_length=1, description="Chunk text content")
	start_position: int = Field(..., ge=0, description="Start position in original text")
	end_position: int = Field(..., ge=0, description="End position in original text")
	
	# Processing metadata
	processing_time_ms: Optional[float] = Field(None, ge=0.0, description="Chunk processing time")
	confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Processing confidence")
	
	# Chunk results
	results: Optional[Dict[str, Any]] = Field(None, description="Processing results")
	status: ProcessingStatus = Field(ProcessingStatus.PENDING, description="Processing status")
	
	# APG audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	processed_at: Optional[datetime] = Field(None, description="Processing completion time")

# Collaborative Annotation Models

class AnnotationProject(BaseModel):
	"""Collaborative annotation project configuration"""
	model_config = MODEL_CONFIG
	
	# Project identity
	id: str = Field(default_factory=uuid7str, description="Unique project identifier")
	tenant_id: str = Field(..., description="Tenant ID for multi-tenancy")
	name: str = Field(..., min_length=1, max_length=200, description="Project name")
	description: Optional[str] = Field(None, max_length=1000, description="Project description")
	
	# Project configuration
	annotation_type: NLPTaskType = Field(..., description="Type of annotation task")
	annotation_schema: Dict[str, Any] = Field(..., description="Annotation schema definition")
	guidelines: Optional[str] = Field(None, description="Annotation guidelines")
	
	# Team and collaboration
	team_members: List[str] = Field(..., description="List of annotator user IDs")
	project_manager: str = Field(..., description="Project manager user ID")
	consensus_threshold: float = Field(0.8, ge=0.5, le=1.0, description="Consensus threshold")
	
	# Project status
	status: Literal["planning", "active", "review", "completed", "archived"] = Field("planning")
	is_training_enabled: bool = Field(False, description="Enable model training from annotations")
	
	# Document management
	document_count: int = Field(0, ge=0, description="Number of documents to annotate")
	completed_annotations: int = Field(0, ge=0, description="Number of completed annotations")
	
	# Quality metrics
	inter_annotator_agreement: Optional[float] = Field(None, ge=0.0, le=1.0)
	average_annotation_time: Optional[float] = Field(None, ge=0.0)
	quality_score: float = Field(0.0, ge=0.0, le=1.0)
	
	# APG audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = Field(..., description="User ID who created project")
	
	@computed_field
	@property
	def completion_percentage(self) -> float:
		"""Calculate project completion percentage"""
		if self.document_count == 0:
			return 0.0
		return min(100.0, (self.completed_annotations / self.document_count) * 100)
	
	def _log_project_milestone(self, milestone: str) -> None:
		"""Log project milestones for tracking"""
		print(f"Annotation project {self.name}: {milestone}")

class TextAnnotation(BaseModel):
	"""Individual text annotation with consensus tracking"""
	model_config = MODEL_CONFIG
	
	# Annotation identity
	id: str = Field(default_factory=uuid7str, description="Unique annotation identifier")
	project_id: str = Field(..., description="Parent project ID")
	document_id: str = Field(..., description="Document being annotated")
	annotator_id: str = Field(..., description="Annotator user ID")
	
	# Annotation content
	start_position: int = Field(..., ge=0, description="Start position of annotation")
	end_position: int = Field(..., ge=0, description="End position of annotation")
	annotated_text: str = Field(..., description="Text that was annotated")
	annotation_value: Union[str, Dict[str, Any]] = Field(..., description="Annotation value")
	
	# Annotation metadata
	confidence: float = Field(1.0, ge=0.0, le=1.0, description="Annotator confidence")
	notes: Optional[str] = Field(None, description="Annotator notes")
	time_spent_seconds: Optional[float] = Field(None, ge=0.0, description="Time spent on annotation")
	
	# Consensus and quality
	consensus_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Consensus with other annotators")
	quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Annotation quality score")
	is_gold_standard: bool = Field(False, description="Is this a gold standard annotation")
	
	# Validation and review
	is_validated: bool = Field(False, description="Has annotation been validated")
	validation_feedback: Optional[str] = Field(None, description="Validation feedback")
	validated_by: Optional[str] = Field(None, description="User who validated annotation")
	validated_at: Optional[datetime] = Field(None, description="Validation timestamp")
	
	# APG audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)
	
	def _log_annotation_created(self) -> None:
		"""Log annotation creation for quality tracking"""
		print(f"Annotation created: {self.id} by {self.annotator_id} (confidence: {self.confidence})")

# Analytics and Intelligence Models

class TextAnalytics(BaseModel):
	"""Advanced text analytics with business intelligence"""
	model_config = MODEL_CONFIG
	
	# Analytics identity
	id: str = Field(default_factory=uuid7str, description="Unique analytics identifier")
	tenant_id: str = Field(..., description="Tenant ID for multi-tenancy")
	name: str = Field(..., description="Analytics session name")
	
	# Analytics configuration
	analysis_type: Literal["sentiment_trends", "entity_analysis", "topic_modeling", "custom"] = Field(...)
	time_period_start: datetime = Field(..., description="Analysis start time")
	time_period_end: datetime = Field(..., description="Analysis end time")
	
	# Data sources
	document_ids: List[str] = Field(..., description="Documents included in analysis")
	filter_criteria: Dict[str, Any] = Field(default_factory=dict, description="Analysis filters")
	
	# Analysis results
	insights: List[Dict[str, Any]] = Field(default_factory=list, description="Generated insights")
	trends: List[Dict[str, Any]] = Field(default_factory=list, description="Detected trends")
	predictions: List[Dict[str, Any]] = Field(default_factory=list, description="Predictive insights")
	
	# Quality and confidence
	confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Overall confidence")
	data_quality_score: float = Field(0.0, ge=0.0, le=1.0, description="Input data quality")
	statistical_significance: Optional[float] = Field(None, ge=0.0, le=1.0)
	
	# Processing metadata
	processing_time_seconds: float = Field(..., ge=0.0, description="Analysis processing time")
	model_versions: Dict[str, str] = Field(default_factory=dict, description="Models used")
	
	# APG audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	created_by: str = Field(..., description="User who requested analytics")
	
	def _log_analytics_completed(self) -> None:
		"""Log analytics completion for monitoring"""
		print(f"Analytics completed: {self.name} ({len(self.insights)} insights, {self.confidence_score:.2f})")

# Model Training and Adaptation

class ModelTrainingConfig(BaseModel):
	"""Configuration for custom model training and domain adaptation"""
	model_config = MODEL_CONFIG
	
	# Training identity
	id: str = Field(default_factory=uuid7str, description="Unique training configuration ID")
	tenant_id: str = Field(..., description="Tenant ID for multi-tenancy")
	name: str = Field(..., description="Training configuration name")
	
	# Base model configuration
	base_model_id: str = Field(..., description="Base model to adapt")
	target_task: NLPTaskType = Field(..., description="Target task type")
	domain: Optional[str] = Field(None, description="Target domain")
	
	# Training data
	training_data_source: Literal["annotations", "documents", "external"] = Field(...)
	annotation_project_id: Optional[str] = Field(None, description="Source annotation project")
	training_document_ids: List[str] = Field(default_factory=list, description="Training documents")
	validation_split: float = Field(0.2, ge=0.1, le=0.5, description="Validation data split")
	
	# Training parameters
	learning_rate: float = Field(0.001, gt=0.0, le=1.0, description="Learning rate")
	batch_size: int = Field(32, ge=1, le=512, description="Training batch size")
	max_epochs: int = Field(10, ge=1, le=1000, description="Maximum training epochs")
	early_stopping_patience: int = Field(5, ge=1, le=100, description="Early stopping patience")
	
	# Resource configuration
	use_gpu: bool = Field(True, description="Use GPU for training if available")
	max_memory_gb: Optional[float] = Field(None, gt=0.0, description="Maximum memory usage")
	parallel_workers: int = Field(1, ge=1, le=16, description="Number of parallel workers")
	
	# Training status
	status: Literal["pending", "preparing", "training", "evaluating", "completed", "failed"] = Field("pending")
	progress_percentage: float = Field(0.0, ge=0.0, le=100.0, description="Training progress")
	
	# Results and metrics
	final_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Final model accuracy")
	training_loss: Optional[float] = Field(None, ge=0.0, description="Final training loss")
	validation_loss: Optional[float] = Field(None, ge=0.0, description="Final validation loss")
	training_time_seconds: Optional[float] = Field(None, ge=0.0, description="Total training time")
	
	# APG audit fields
	created_at: datetime = Field(default_factory=datetime.utcnow)
	started_at: Optional[datetime] = Field(None, description="Training start time")
	completed_at: Optional[datetime] = Field(None, description="Training completion time")
	created_by: str = Field(..., description="User who initiated training")
	
	@computed_field
	@property 
	def estimated_completion_time(self) -> Optional[datetime]:
		"""Estimate training completion time based on progress"""
		if not self.started_at or self.progress_percentage <= 0:
			return None
		
		elapsed = (datetime.utcnow() - self.started_at).total_seconds()
		estimated_total = elapsed / (self.progress_percentage / 100)
		remaining = estimated_total - elapsed
		
		return datetime.utcnow().replace(microsecond=0) + timedelta(seconds=remaining)
	
	def _log_training_progress(self, current_epoch: int, current_loss: float) -> None:
		"""Log training progress for monitoring"""
		print(f"Training {self.name} - Epoch {current_epoch}: loss={current_loss:.4f}")

# System Health and Monitoring

class SystemHealth(BaseModel):
	"""System health and performance monitoring"""
	model_config = MODEL_CONFIG
	
	# Health check identity
	id: str = Field(default_factory=uuid7str, description="Unique health check ID")
	tenant_id: Optional[str] = Field(None, description="Tenant ID (None for system-wide)")
	
	# System status
	overall_status: Literal["healthy", "degraded", "unhealthy", "maintenance"] = Field(...)
	component_status: Dict[str, str] = Field(..., description="Individual component status")
	
	# Performance metrics
	average_response_time_ms: float = Field(..., ge=0.0, description="Average API response time")
	requests_per_minute: int = Field(..., ge=0, description="Current request rate")
	active_sessions: int = Field(..., ge=0, description="Active streaming sessions")
	queue_depth: int = Field(..., ge=0, description="Processing queue depth")
	
	# Resource utilization
	cpu_usage_percent: float = Field(..., ge=0.0, le=100.0, description="CPU utilization")
	memory_usage_percent: float = Field(..., ge=0.0, le=100.0, description="Memory utilization")
	gpu_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0, description="GPU utilization")
	disk_usage_percent: float = Field(..., ge=0.0, le=100.0, description="Disk utilization")
	
	# Model status summary
	total_models: int = Field(..., ge=0, description="Total configured models")
	active_models: int = Field(..., ge=0, description="Currently active models")
	loaded_models: int = Field(..., ge=0, description="Currently loaded models")
	failed_models: int = Field(..., ge=0, description="Models in failed state")
	
	# Error and alert information
	recent_errors: List[Dict[str, Any]] = Field(default_factory=list, description="Recent error summary")
	active_alerts: List[str] = Field(default_factory=list, description="Active system alerts")
	
	# APG audit fields
	timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
	
	@computed_field
	@property
	def model_availability_percent(self) -> float:
		"""Calculate percentage of models that are available"""
		if self.total_models == 0:
			return 0.0
		return (self.active_models / self.total_models) * 100
	
	@computed_field
	@property
	def performance_rating(self) -> Literal["excellent", "good", "acceptable", "poor"]:
		"""Overall performance rating based on metrics"""
		if (self.average_response_time_ms < 100 and 
			self.cpu_usage_percent < 70 and 
			self.memory_usage_percent < 80):
			return "excellent"
		elif (self.average_response_time_ms < 200 and 
			  self.cpu_usage_percent < 85 and 
			  self.memory_usage_percent < 90):
			return "good"
		elif (self.average_response_time_ms < 500 and 
			  self.cpu_usage_percent < 95 and 
			  self.memory_usage_percent < 95):
			return "acceptable"
		else:
			return "poor"
	
	def _log_health_status(self) -> None:
		"""Log system health status for monitoring"""
		print(f"System health: {self.overall_status} ({self.performance_rating} performance)")

# Export all models for API usage
__all__ = [
	# Enums
	"NLPTaskType", "ModelProvider", "ProcessingStatus", "QualityLevel", 
	"DocumentType", "LanguageCode",
	
	# Core models
	"TextDocument", "NLPModel", "ProcessingRequest", "ProcessingResult",
	
	# Specialized results
	"SentimentResult", "EntityResult", "ClassificationResult",
	
	# Streaming models
	"StreamingSession", "StreamingChunk",
	
	# Collaboration models
	"AnnotationProject", "TextAnnotation",
	
	# Analytics models
	"TextAnalytics", "ModelTrainingConfig",
	
	# System models
	"SystemHealth"
]
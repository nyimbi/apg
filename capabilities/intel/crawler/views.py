"""
APG Crawler Capability - Pydantic v2 Data Models and Views
==========================================================

Comprehensive data models for enterprise web intelligence platform with:
- Modern Python 3.12+ typing patterns
- Multi-tenant architecture with APG integration  
- Advanced validation and business logic
- Real-time collaboration and validation support

Copyright © 2025 Datacraft (nyimbi@gmail.com)
"""

from typing import Any, Dict, List, Optional, Union, Literal, Annotated
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
import json
import hashlib

from pydantic import (
	BaseModel, 
	Field, 
	ConfigDict, 
	AfterValidator, 
	BeforeValidator,
	field_validator,
	model_validator,
	computed_field
)
from uuid_extensions import uuid7str


# =====================================================
# CONFIGURATION AND VALIDATION
# =====================================================

# Pydantic v2 configuration for all models
model_config = ConfigDict(
	extra='forbid',
	validate_assignment=True,
	validate_by_name=True,
	validate_by_alias=True,
	str_strip_whitespace=True,
	frozen=False,
	populate_by_name=True
)


# =====================================================
# ENUMS AND CONSTANTS
# =====================================================

class TargetType(str, Enum):
	"""Types of crawl targets"""
	WEB_CRAWL = "web_crawl"
	API_SCRAPE = "api_scrape"
	SOCIAL_MEDIA = "social_media"
	NEWS_AGGREGATION = "news_aggregation"
	DATABASE_EXPORT = "database_export"
	REAL_TIME_STREAM = "real_time_stream"
	HYBRID_MULTI_SOURCE = "hybrid_multi_source"
	RAG_KNOWLEDGE_BASE = "rag_knowledge_base"
	GRAPHRAG_CORPUS = "graphrag_corpus"


class ExtractionMethod(str, Enum):
	"""Methods for data extraction"""
	HTML_PARSING = "html_parsing"
	JAVASCRIPT_RENDERING = "javascript_rendering"
	API_CONSUMPTION = "api_consumption"
	STEALTH_CRAWLING = "stealth_crawling"
	MULTI_SOURCE_FUSION = "multi_source_fusion"
	AI_POWERED_EXTRACTION = "ai_powered_extraction"


class ValidationStatus(str, Enum):
	"""Status of data validation"""
	PENDING = "pending"
	IN_PROGRESS = "in_progress"
	APPROVED = "approved"
	REJECTED = "rejected"
	REQUIRES_REVIEW = "requires_review"
	CONSENSUS_REACHED = "consensus_reached"
	CONFLICTED = "conflicted"


class SessionStatus(str, Enum):
	"""Status of validation or processing sessions"""
	DRAFT = "draft"
	ACTIVE = "active"
	PAUSED = "paused"
	COMPLETED = "completed"
	CANCELLED = "cancelled"
	ARCHIVED = "archived"


class StealthStrategy(str, Enum):
	"""Available stealth strategies"""
	CLOUDSCRAPER = "cloudscraper"
	PLAYWRIGHT = "playwright"
	SELENIUM = "selenium"
	HTTP_STEALTH = "http_stealth"
	PROXY_ROTATION = "proxy_rotation"
	BEHAVIORAL_MIMICRY = "behavioral_mimicry"
	ADAPTIVE_LEARNING = "adaptive_learning"


class ProtectionType(str, Enum):
	"""Types of web protection mechanisms"""
	NONE = "none"
	CLOUDFLARE = "cloudflare"
	AKAMAI = "akamai"
	INCAPSULA = "incapsula"
	RECAPTCHA = "recaptcha"
	HCAPTCHA = "hcaptcha"
	CUSTOM_JAVASCRIPT = "custom_javascript"
	WAF_GENERIC = "waf_generic"


class InsightType(str, Enum):
	"""Types of analytics insights"""
	TREND = "trend"
	ANOMALY = "anomaly"
	PATTERN = "pattern"
	PREDICTION = "prediction"
	OPPORTUNITY = "opportunity"
	RISK = "risk"
	RECOMMENDATION = "recommendation"


class ProcessingStatus(str, Enum):
	"""Status of data processing operations"""
	QUEUED = "queued"
	PROCESSING = "processing"
	COMPLETED = "completed"
	FAILED = "failed"
	RETRYING = "retrying"
	CANCELLED = "cancelled"


class ContentProcessingStage(str, Enum):
	"""Stages of content processing for RAG/GraphRAG integration"""
	RAW_EXTRACTED = "raw_extracted"
	CLEANED = "cleaned"
	MARKDOWN_CONVERTED = "markdown_converted"
	FINGERPRINTED = "fingerprinted"
	RAG_PROCESSED = "rag_processed"
	GRAPHRAG_PROCESSED = "graphrag_processed"
	KNOWLEDGE_GRAPH_INTEGRATED = "knowledge_graph_integrated"


class RAGIntegrationType(str, Enum):
	"""Types of RAG integration"""
	VECTOR_SEARCH = "vector_search"
	SEMANTIC_CHUNKS = "semantic_chunks"
	HIERARCHICAL_SUMMARIES = "hierarchical_summaries"
	ENTITY_EXTRACTION = "entity_extraction"
	RELATION_MAPPING = "relation_mapping"
	CONTEXT_ENRICHMENT = "context_enrichment"


# =====================================================
# VALIDATION FUNCTIONS
# =====================================================

def validate_url_list(urls: List[str]) -> List[str]:
	"""Validate list of URLs"""
	if not urls:
		raise ValueError("URL list cannot be empty")
	
	from urllib.parse import urlparse
	validated_urls = []
	
	for url in urls:
		parsed = urlparse(url.strip())
		if not parsed.scheme or not parsed.netloc:
			raise ValueError(f"Invalid URL format: {url}")
		validated_urls.append(url.strip())
	
	return validated_urls


def validate_tenant_id(tenant_id: str) -> str:
	"""Validate tenant ID format"""
	if not tenant_id or len(tenant_id) < 3:
		raise ValueError("Tenant ID must be at least 3 characters")
	if not tenant_id.replace("_", "").replace("-", "").isalnum():
		raise ValueError("Tenant ID must contain only alphanumeric characters, hyphens, and underscores")
	return tenant_id.lower()


def validate_json_dict(value: Any) -> Dict[str, Any]:
	"""Validate and normalize JSON dictionary"""
	if value is None:
		return {}
	if isinstance(value, str):
		try:
			return json.loads(value)
		except json.JSONDecodeError as e:
			raise ValueError(f"Invalid JSON string: {e}")
	if isinstance(value, dict):
		return value
	raise ValueError("Value must be a dictionary or JSON string")


def validate_confidence_score(score: float) -> float:
	"""Validate confidence score is between 0.0 and 1.0"""
	if not 0.0 <= score <= 1.0:
		raise ValueError("Confidence score must be between 0.0 and 1.0")
	return round(score, 4)


def validate_quality_rating(rating: int) -> int:
	"""Validate quality rating is between 1 and 5"""
	if not 1 <= rating <= 5:
		raise ValueError("Quality rating must be between 1 and 5")
	return rating


def generate_content_fingerprint(content: str) -> str:
	"""Generate SHA-256 fingerprint for content"""
	if not content:
		return ""
	content_bytes = content.encode('utf-8')
	return hashlib.sha256(content_bytes).hexdigest()


def validate_markdown_content(content: str) -> str:
	"""Validate and clean markdown content"""
	if not content:
		return ""
	# Remove excessive whitespace while preserving markdown structure
	lines = content.split('\n')
	cleaned_lines = []
	for line in lines:
		cleaned_line = line.rstrip()
		cleaned_lines.append(cleaned_line)
	# Remove excessive empty lines (max 2 consecutive)
	result_lines = []
	empty_count = 0
	for line in cleaned_lines:
		if line.strip() == "":
			empty_count += 1
			if empty_count <= 2:
				result_lines.append(line)
		else:
			empty_count = 0
			result_lines.append(line)
	return '\n'.join(result_lines).strip()


def validate_rag_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
	"""Validate RAG-specific metadata"""
	if not isinstance(metadata, dict):
		raise ValueError("RAG metadata must be a dictionary")
	
	# Ensure required RAG fields exist
	rag_metadata = metadata.copy()
	rag_metadata.setdefault('chunk_size', 1000)
	rag_metadata.setdefault('overlap_size', 200)
	rag_metadata.setdefault('vector_dimensions', 1536)
	rag_metadata.setdefault('embedding_model', 'text-embedding-ada-002')
	rag_metadata.setdefault('indexing_strategy', 'semantic_chunks')
	
	return rag_metadata


# =====================================================
# CORE CRAWLER MODELS
# =====================================================

class BusinessContext(BaseModel):
	"""Business context for intelligent crawling"""
	model_config = model_config
	
	domain: str = Field(..., min_length=1, max_length=100, description="Business domain")
	industry: Optional[str] = Field(None, max_length=100, description="Industry sector")
	use_case: Optional[str] = Field(None, max_length=200, description="Specific use case")
	priority_entities: List[str] = Field(default_factory=list, description="Priority business entities")
	context_keywords: List[str] = Field(default_factory=list, description="Context-specific keywords")
	quality_criteria: Dict[str, Any] = Field(default_factory=dict, description="Quality criteria")
	success_metrics: Dict[str, Any] = Field(default_factory=dict, description="Success metrics")


class StealthRequirements(BaseModel):
	"""Stealth and protection bypass requirements"""
	model_config = model_config
	
	enable_stealth: bool = Field(True, description="Enable stealth capabilities")
	preferred_strategies: List[StealthStrategy] = Field(default_factory=list, description="Preferred stealth strategies")
	protection_tolerance: Literal["low", "medium", "high", "maximum"] = Field("medium", description="Protection bypass tolerance")
	success_rate_threshold: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.8, description="Minimum success rate")
	max_retry_attempts: int = Field(3, ge=1, le=10, description="Maximum retry attempts")
	adaptive_learning: bool = Field(True, description="Enable adaptive learning")
	cost_optimization: bool = Field(True, description="Enable cost optimization")


class RAGProcessingConfig(BaseModel):
	"""Configuration for RAG and GraphRAG processing"""
	model_config = model_config
	
	enable_rag_processing: bool = Field(True, description="Enable RAG processing")
	enable_graphrag_processing: bool = Field(False, description="Enable GraphRAG processing")
	chunk_size: int = Field(1000, ge=100, le=8000, description="Text chunk size for RAG")
	overlap_size: int = Field(200, ge=0, le=1000, description="Chunk overlap size")
	vector_dimensions: int = Field(1536, ge=512, le=4096, description="Vector embedding dimensions")
	embedding_model: str = Field("text-embedding-ada-002", description="Embedding model name")
	indexing_strategy: RAGIntegrationType = Field(RAGIntegrationType.SEMANTIC_CHUNKS, description="RAG indexing strategy")
	graph_extraction_depth: int = Field(2, ge=1, le=5, description="GraphRAG extraction depth")
	entity_resolution_threshold: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.8, description="Entity resolution threshold")
	relation_confidence_threshold: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.7, description="Relation confidence threshold")
	knowledge_graph_integration: bool = Field(True, description="Enable knowledge graph integration")
	rag_metadata: Annotated[Dict[str, Any], AfterValidator(validate_rag_metadata)] = Field(default_factory=dict, description="RAG-specific metadata")


class ContentCleaningConfig(BaseModel):
	"""Configuration for content cleaning and markdown conversion"""
	model_config = model_config
	
	remove_ads: bool = Field(True, description="Remove advertisements")
	remove_navigation: bool = Field(True, description="Remove navigation elements")
	remove_comments: bool = Field(True, description="Remove comment sections")
	remove_social_widgets: bool = Field(True, description="Remove social media widgets")
	preserve_links: bool = Field(True, description="Preserve links in markdown")
	preserve_images: bool = Field(True, description="Preserve image references")
	preserve_tables: bool = Field(True, description="Preserve table structures")
	preserve_code_blocks: bool = Field(True, description="Preserve code blocks")
	min_content_length: int = Field(100, ge=10, description="Minimum content length")
	max_content_length: int = Field(1000000, ge=1000, description="Maximum content length")
	language_detection: bool = Field(True, description="Enable language detection")
	custom_cleaning_rules: List[str] = Field(default_factory=list, description="Custom cleaning rules")
	markdown_formatting: bool = Field(True, description="Apply markdown formatting")


class QualityRequirements(BaseModel):
	"""Data quality requirements and thresholds"""
	model_config = model_config
	
	minimum_confidence: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.7, description="Minimum confidence threshold")
	require_validation: bool = Field(False, description="Require human validation")
	consensus_threshold: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.8, description="Consensus threshold for validation")
	accuracy_requirements: Dict[str, float] = Field(default_factory=dict, description="Accuracy requirements by data type")
	completeness_requirements: Dict[str, float] = Field(default_factory=dict, description="Completeness requirements")
	freshness_requirements: Optional[timedelta] = Field(None, description="Data freshness requirements")
	duplicate_tolerance: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.05, description="Duplicate tolerance")
	content_cleaning_config: ContentCleaningConfig = Field(default_factory=ContentCleaningConfig, description="Content cleaning configuration")
	rag_processing_config: RAGProcessingConfig = Field(default_factory=RAGProcessingConfig, description="RAG processing configuration")


class SchedulingConfig(BaseModel):
	"""Scheduling configuration for crawl operations"""
	model_config = model_config
	
	schedule_type: Literal["one_time", "recurring", "continuous", "triggered"] = Field("one_time", description="Schedule type")
	cron_expression: Optional[str] = Field(None, description="Cron expression for recurring schedules")
	interval_minutes: Optional[int] = Field(None, ge=1, description="Interval in minutes for recurring schedules")
	start_time: Optional[datetime] = Field(None, description="Schedule start time")
	end_time: Optional[datetime] = Field(None, description="Schedule end time")
	timezone: str = Field("UTC", description="Timezone for scheduling")
	max_concurrent: int = Field(5, ge=1, le=100, description="Maximum concurrent operations")
	priority_level: int = Field(5, ge=1, le=10, description="Priority level (1=highest, 10=lowest)")


class CollaborationConfig(BaseModel):
	"""Collaboration configuration for team-based validation"""
	model_config = model_config
	
	enable_collaboration: bool = Field(False, description="Enable collaborative validation")
	required_validators: int = Field(2, ge=1, le=10, description="Required number of validators")
	validator_roles: List[str] = Field(default_factory=list, description="Required validator roles")
	consensus_strategy: Literal["majority", "unanimous", "weighted", "expert"] = Field("majority", description="Consensus strategy")
	conflict_resolution: Literal["expert_override", "discussion", "majority_rule", "quality_score"] = Field("majority_rule", description="Conflict resolution method")
	auto_approve_threshold: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.95, description="Auto-approval threshold")
	escalation_rules: Dict[str, Any] = Field(default_factory=dict, description="Escalation rules")


class CrawlTarget(BaseModel):
	"""Comprehensive crawl target with business intelligence and RAG/GraphRAG integration"""
	model_config = model_config
	
	id: str = Field(default_factory=uuid7str, description="Unique target identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	name: str = Field(..., min_length=1, max_length=500, description="Target name")
	description: Optional[str] = Field(None, max_length=2000, description="Target description")
	target_urls: Annotated[List[str], AfterValidator(validate_url_list)] = Field(..., description="Target URLs")
	target_type: TargetType = Field(TargetType.WEB_CRAWL, description="Type of crawl target")
	data_schema: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Expected data schema")
	business_context: BusinessContext = Field(..., description="Business context")
	stealth_requirements: StealthRequirements = Field(default_factory=StealthRequirements, description="Stealth requirements")
	quality_requirements: QualityRequirements = Field(default_factory=QualityRequirements, description="Quality requirements")
	scheduling_config: SchedulingConfig = Field(default_factory=SchedulingConfig, description="Scheduling configuration")
	collaboration_config: CollaborationConfig = Field(default_factory=CollaborationConfig, description="Collaboration configuration")
	# RAG/GraphRAG Integration Fields
	rag_integration_enabled: bool = Field(True, description="Enable RAG integration")
	graphrag_integration_enabled: bool = Field(False, description="Enable GraphRAG integration")
	knowledge_graph_target: Optional[str] = Field(None, description="Target knowledge graph identifier")
	content_fingerprinting: bool = Field(True, description="Enable content fingerprinting")
	markdown_storage: bool = Field(True, description="Store cleaned content as markdown")
	status: Literal["draft", "active", "paused", "completed", "archived"] = Field("draft", description="Target status")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: Optional[str] = Field(None, description="Creator user ID")
	updated_by: Optional[str] = Field(None, description="Last updater user ID")
	
	@field_validator('target_urls')
	@classmethod
	def validate_urls(cls, v: List[str]) -> List[str]:
		"""Additional URL validation"""
		if len(v) > 1000:
			raise ValueError("Maximum 1000 URLs allowed per target")
		return v
	
	@model_validator(mode='after')
	def validate_scheduling(self) -> 'CrawlTarget':
		"""Validate scheduling configuration consistency"""
		if self.scheduling_config.schedule_type == "recurring":
			if not self.scheduling_config.cron_expression and not self.scheduling_config.interval_minutes:
				raise ValueError("Recurring schedule requires either cron_expression or interval_minutes")
		
		if self.scheduling_config.start_time and self.scheduling_config.end_time:
			if self.scheduling_config.start_time >= self.scheduling_config.end_time:
				raise ValueError("start_time must be before end_time")
		
		return self


class DataSource(BaseModel):
	"""Data source configuration and capabilities"""
	model_config = model_config
	
	id: str = Field(default_factory=uuid7str, description="Unique source identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	name: str = Field(..., min_length=1, max_length=500, description="Source name")
	source_type: str = Field(..., min_length=1, max_length=100, description="Source type")
	connection_config: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Connection configuration")
	authentication_config: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Authentication configuration")
	rate_limits: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Rate limiting configuration")
	stealth_config: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Stealth configuration")
	capabilities: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Source capabilities")
	health_status: Literal["healthy", "degraded", "unhealthy", "unknown"] = Field("unknown", description="Health status")
	last_health_check: Optional[datetime] = Field(None, description="Last health check timestamp")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: Optional[str] = Field(None, description="Creator user ID")
	updated_by: Optional[str] = Field(None, description="Last updater user ID")


# =====================================================
# DATA EXTRACTION MODELS
# =====================================================

class QualityMetrics(BaseModel):
	"""Comprehensive quality metrics for datasets"""
	model_config = model_config
	
	completeness_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Completeness score")
	accuracy_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Accuracy score")
	consistency_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Consistency score")
	freshness_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Freshness score")
	uniqueness_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Uniqueness score")
	relevance_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Relevance score")
	duplicate_ratio: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Duplicate ratio")
	missing_data_ratio: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Missing data ratio")
	validation_coverage: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Validation coverage")
	overall_quality_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Overall quality score")
	
	@computed_field
	@property
	def quality_grade(self) -> str:
		"""Compute quality grade based on overall score"""
		if self.overall_quality_score >= 0.9:
			return "A"
		elif self.overall_quality_score >= 0.8:
			return "B"
		elif self.overall_quality_score >= 0.7:
			return "C"
		elif self.overall_quality_score >= 0.6:
			return "D"
		else:
			return "F"


class ExtractedDataset(BaseModel):
	"""Rich extracted dataset with comprehensive metadata"""
	model_config = model_config
	
	id: str = Field(default_factory=uuid7str, description="Unique dataset identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	crawl_target_id: str = Field(..., description="Source crawl target ID")
	pipeline_id: Optional[str] = Field(None, description="Processing pipeline ID")
	dataset_name: str = Field(..., min_length=1, max_length=500, description="Dataset name")
	extraction_method: ExtractionMethod = Field(..., description="Extraction method used")
	source_urls: Annotated[List[str], AfterValidator(validate_url_list)] = Field(..., description="Source URLs")
	record_count: int = Field(0, ge=0, description="Number of records")
	quality_metrics: QualityMetrics = Field(default_factory=QualityMetrics, description="Quality metrics")
	validation_status: ValidationStatus = Field(ValidationStatus.PENDING, description="Validation status")
	consensus_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Validation consensus score")
	data_schema: Optional[Dict[str, Any]] = Field(None, description="Actual data schema")
	metadata: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Additional metadata")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: Optional[str] = Field(None, description="Creator user ID")
	updated_by: Optional[str] = Field(None, description="Last updater user ID")
	
	@computed_field
	@property
	def is_high_quality(self) -> bool:
		"""Check if dataset meets high quality standards"""
		return (
			self.quality_metrics.overall_quality_score >= 0.8 and
			self.consensus_score >= 0.8 and
			self.validation_status in [ValidationStatus.APPROVED, ValidationStatus.CONSENSUS_REACHED]
		)


class BusinessEntity(BaseModel):
	"""Business entity extracted from content"""
	model_config = model_config
	
	id: str = Field(default_factory=uuid7str, description="Unique entity identifier")
	entity_type: str = Field(..., min_length=1, max_length=100, description="Entity type")
	entity_name: str = Field(..., min_length=1, max_length=1000, description="Entity name")
	entity_value: str = Field(..., min_length=1, description="Entity value")
	confidence_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(..., description="Confidence score")
	context_window: Optional[str] = Field(None, max_length=2000, description="Context window")
	start_position: Optional[int] = Field(None, ge=0, description="Start position in text")
	end_position: Optional[int] = Field(None, ge=0, description="End position in text")
	semantic_properties: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Semantic properties")
	business_relevance: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Business relevance score")
	validation_status: ValidationStatus = Field(ValidationStatus.PENDING, description="Validation status")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	
	@model_validator(mode='after')
	def validate_positions(self) -> 'BusinessEntity':
		"""Validate start and end positions"""
		if self.start_position is not None and self.end_position is not None:
			if self.start_position >= self.end_position:
				raise ValueError("start_position must be less than end_position")
		return self


class DataRecord(BaseModel):
	"""Individual data record with comprehensive metadata and RAG/GraphRAG integration"""
	model_config = model_config
	
	id: str = Field(default_factory=uuid7str, description="Unique record identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	dataset_id: str = Field(..., description="Parent dataset ID")
	record_index: int = Field(..., ge=0, description="Record index within dataset")
	source_url: str = Field(..., min_length=1, description="Source URL")
	extracted_data: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Extracted structured data")
	raw_content: Optional[str] = Field(None, description="Raw content")
	processed_content: Optional[str] = Field(None, description="Processed content")
	# RAG/GraphRAG Content Fields
	cleaned_content: Optional[str] = Field(None, description="Cleaned and processed content")
	markdown_content: Annotated[Optional[str], AfterValidator(validate_markdown_content)] = Field(None, description="Content converted to clean markdown")
	content_fingerprint: str = Field(default="", description="SHA-256 fingerprint of content")
	content_processing_stage: ContentProcessingStage = Field(ContentProcessingStage.RAW_EXTRACTED, description="Current processing stage")
	vector_embeddings: Optional[List[float]] = Field(None, description="Vector embeddings for RAG")
	rag_chunk_ids: List[str] = Field(default_factory=list, description="Associated RAG chunk identifiers")
	graphrag_node_id: Optional[str] = Field(None, description="Associated GraphRAG node identifier")
	knowledge_graph_entities: List[str] = Field(default_factory=list, description="Knowledge graph entity IDs")
	content_type: Optional[str] = Field(None, max_length=100, description="Content type")
	language: Optional[str] = Field(None, max_length=10, description="Detected language")
	quality_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Quality score")
	confidence_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Confidence score")
	validation_status: ValidationStatus = Field(ValidationStatus.PENDING, description="Validation status")
	business_entities: List[BusinessEntity] = Field(default_factory=list, description="Extracted business entities")
	semantic_tags: List[str] = Field(default_factory=list, description="Semantic tags")
	extraction_metadata: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Extraction metadata")
	rag_metadata: Annotated[Dict[str, Any], AfterValidator(validate_rag_metadata)] = Field(default_factory=dict, description="RAG-specific metadata")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	
	@model_validator(mode='after')
	def generate_fingerprint(self) -> 'DataRecord':
		"""Generate content fingerprint after validation"""
		if self.markdown_content and not self.content_fingerprint:
			self.content_fingerprint = generate_content_fingerprint(self.markdown_content)
		elif self.cleaned_content and not self.content_fingerprint:
			self.content_fingerprint = generate_content_fingerprint(self.cleaned_content)
		elif self.processed_content and not self.content_fingerprint:
			self.content_fingerprint = generate_content_fingerprint(self.processed_content)
		return self
	
	@computed_field
	@property
	def entity_count(self) -> int:
		"""Count of extracted business entities"""
		return len(self.business_entities)
	
	@computed_field
	@property
	def high_confidence_entities(self) -> List[BusinessEntity]:
		"""High confidence business entities (>0.8)"""
		return [entity for entity in self.business_entities if entity.confidence_score > 0.8]
	
	@computed_field
	@property
	def is_rag_ready(self) -> bool:
		"""Check if record is ready for RAG processing"""
		return (
			self.content_processing_stage in [ContentProcessingStage.CLEANED, ContentProcessingStage.MARKDOWN_CONVERTED, ContentProcessingStage.FINGERPRINTED] and
			self.markdown_content is not None and
			len(self.markdown_content.strip()) > 0 and
			self.content_fingerprint != ""
		)
	
	@computed_field
	@property
	def is_graphrag_ready(self) -> bool:
		"""Check if record is ready for GraphRAG processing"""
		return (
			self.is_rag_ready and
			len(self.business_entities) > 0 and
			self.quality_score > 0.7
		)


# =====================================================
# COLLABORATIVE VALIDATION MODELS
# =====================================================

class ValidationSchema(BaseModel):
	"""Schema for validation requirements"""
	model_config = model_config
	
	required_fields: List[str] = Field(default_factory=list, description="Required fields for validation")
	validation_rules: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Validation rules")
	quality_criteria: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Quality criteria")
	scoring_weights: Annotated[Dict[str, float], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Scoring weights")
	approval_thresholds: Annotated[Dict[str, float], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Approval thresholds")


class ValidatorProfile(BaseModel):
	"""Validator profile and expertise"""
	model_config = model_config
	
	id: str = Field(default_factory=uuid7str, description="Unique validator identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	user_id: str = Field(..., min_length=1, description="User identifier")
	validator_name: str = Field(..., min_length=1, max_length=500, description="Validator name")
	validator_role: str = Field(..., min_length=1, max_length=100, description="Validator role")
	expertise_areas: List[str] = Field(default_factory=list, description="Expertise areas")
	validation_permissions: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Validation permissions")
	assignment_status: Literal["active", "inactive", "pending", "suspended"] = Field("active", description="Assignment status")
	validation_stats: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Validation statistics")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class ValidationFeedback(BaseModel):
	"""Validation feedback from validators"""
	model_config = model_config
	
	id: str = Field(default_factory=uuid7str, description="Unique feedback identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	session_id: str = Field(..., description="Validation session ID")
	validator_id: str = Field(..., description="Validator ID")
	record_id: str = Field(..., description="Data record ID")
	feedback_type: Literal["approve", "reject", "modify", "comment"] = Field(..., description="Feedback type")
	quality_rating: Annotated[int, AfterValidator(validate_quality_rating)] = Field(..., description="Quality rating (1-5)")
	accuracy_rating: Annotated[int, AfterValidator(validate_quality_rating)] = Field(..., description="Accuracy rating (1-5)")
	completeness_rating: Annotated[int, AfterValidator(validate_quality_rating)] = Field(..., description="Completeness rating (1-5)")
	suggested_changes: Optional[Dict[str, Any]] = Field(None, description="Suggested changes")
	comments: Optional[str] = Field(None, max_length=5000, description="Validator comments")
	validation_tags: List[str] = Field(default_factory=list, description="Validation tags")
	confidence_level: Annotated[float, AfterValidator(validate_confidence_score)] = Field(1.0, description="Validator confidence level")
	processing_time_seconds: Optional[int] = Field(None, ge=0, description="Processing time in seconds")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	
	@computed_field
	@property
	def overall_rating(self) -> float:
		"""Compute overall rating from individual ratings"""
		return (self.quality_rating + self.accuracy_rating + self.completeness_rating) / 3.0


class ConsensusMetrics(BaseModel):
	"""Consensus metrics for validation sessions"""
	model_config = model_config
	
	total_records: int = Field(0, ge=0, description="Total records")
	validated_records: int = Field(0, ge=0, description="Validated records")
	consensus_reached: int = Field(0, ge=0, description="Records with consensus")
	conflicts_pending: int = Field(0, ge=0, description="Records with conflicts")
	approval_rate: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Approval rate")
	average_quality_rating: float = Field(0.0, ge=1.0, le=5.0, description="Average quality rating")
	validator_agreement: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Validator agreement rate")
	completion_percentage: float = Field(0.0, ge=0.0, le=100.0, description="Completion percentage")
	estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


class ValidationSession(BaseModel):
	"""Collaborative validation session"""
	model_config = model_config
	
	id: str = Field(default_factory=uuid7str, description="Unique session identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	dataset_id: str = Field(..., description="Dataset ID")
	session_name: str = Field(..., min_length=1, max_length=500, description="Session name")
	description: Optional[str] = Field(None, max_length=2000, description="Session description")
	validation_schema: ValidationSchema = Field(..., description="Validation schema")
	session_status: SessionStatus = Field(SessionStatus.DRAFT, description="Session status")
	consensus_threshold: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.8, description="Consensus threshold")
	quality_threshold: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.7, description="Quality threshold")
	validator_count: int = Field(0, ge=0, description="Number of validators")
	completion_percentage: float = Field(0.0, ge=0.0, le=100.0, description="Completion percentage")
	consensus_metrics: ConsensusMetrics = Field(default_factory=ConsensusMetrics, description="Consensus metrics")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_by: Optional[str] = Field(None, description="Creator user ID")
	updated_by: Optional[str] = Field(None, description="Last updater user ID")
	
	@computed_field
	@property
	def is_complete(self) -> bool:
		"""Check if validation session is complete"""
		return (
			self.session_status == SessionStatus.COMPLETED or
			self.completion_percentage >= 100.0
		)


# =====================================================
# STEALTH AND PROCESSING MODELS
# =====================================================

class StealthStrategy(BaseModel):
	"""Stealth strategy configuration"""
	model_config = model_config
	
	id: str = Field(default_factory=uuid7str, description="Unique strategy identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	strategy_name: str = Field(..., min_length=1, max_length=500, description="Strategy name")
	strategy_type: str = Field(..., min_length=1, max_length=100, description="Strategy type")
	configuration: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Strategy configuration")
	capabilities: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Strategy capabilities")
	success_rate: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Historical success rate")
	performance_metrics: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Performance metrics")
	cost_score: float = Field(0.0, ge=0.0, description="Cost score (lower is better)")
	detection_resistance: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Detection resistance score")
	resource_usage: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Resource usage metrics")
	status: Literal["active", "inactive", "testing", "deprecated"] = Field("active", description="Strategy status")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	
	@computed_field
	@property
	def effectiveness_score(self) -> float:
		"""Compute overall effectiveness score"""
		# Combine success rate and detection resistance, penalize high cost
		cost_penalty = max(0, (self.cost_score - 1.0) * 0.1)
		return max(0.0, (self.success_rate * 0.6 + self.detection_resistance * 0.4) - cost_penalty)


class ProtectionProfile(BaseModel):
	"""Website protection mechanism profile"""
	model_config = model_config
	
	id: str = Field(default_factory=uuid7str, description="Unique profile identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	domain: str = Field(..., min_length=1, max_length=500, description="Domain name")
	protection_types: List[ProtectionType] = Field(default_factory=list, description="Detected protection types")
	detection_confidence: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Detection confidence")
	protection_characteristics: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Protection characteristics")
	recommended_strategies: List[str] = Field(default_factory=list, description="Recommended stealth strategies")
	last_analyzed: datetime = Field(default_factory=datetime.utcnow, description="Last analysis timestamp")
	success_history: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Success history by strategy")
	adaptation_notes: Optional[str] = Field(None, max_length=2000, description="Adaptation notes")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	
	@computed_field
	@property
	def difficulty_level(self) -> str:
		"""Assess difficulty level based on protection types"""
		if not self.protection_types:
			return "easy"
		
		advanced_protection = [ProtectionType.CLOUDFLARE, ProtectionType.AKAMAI, ProtectionType.INCAPSULA]
		if any(ptype in advanced_protection for ptype in self.protection_types):
			return "hard"
		elif len(self.protection_types) > 2:
			return "medium"
		else:
			return "easy"


# =====================================================
# RAG AND GRAPHRAG INTEGRATION MODELS
# =====================================================

class RAGChunk(BaseModel):
	"""RAG chunk with vector embeddings and metadata"""
	model_config = model_config
	
	id: str = Field(default_factory=uuid7str, description="Unique chunk identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	record_id: str = Field(..., description="Source data record ID")
	chunk_index: int = Field(..., ge=0, description="Chunk index within record")
	chunk_text: str = Field(..., min_length=1, description="Chunk text content")
	chunk_markdown: Annotated[str, AfterValidator(validate_markdown_content)] = Field(..., description="Chunk in markdown format")
	chunk_fingerprint: str = Field(default="", description="SHA-256 fingerprint of chunk")
	vector_embeddings: Optional[List[float]] = Field(None, description="Vector embeddings")
	embedding_model: str = Field("text-embedding-ada-002", description="Embedding model used")
	vector_dimensions: int = Field(1536, ge=512, le=4096, description="Vector dimensions")
	semantic_similarity_threshold: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.8, description="Semantic similarity threshold")
	chunk_overlap_start: int = Field(0, ge=0, description="Overlap start position")
	chunk_overlap_end: int = Field(0, ge=0, description="Overlap end position")
	entities_extracted: List[str] = Field(default_factory=list, description="Extracted entity IDs")
	related_chunks: List[str] = Field(default_factory=list, description="Related chunk IDs")
	contextual_metadata: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Contextual metadata")
	indexing_status: Literal["pending", "indexed", "failed", "stale"] = Field("pending", description="Indexing status")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	
	@model_validator(mode='after')
	def generate_chunk_fingerprint(self) -> 'RAGChunk':
		"""Generate chunk fingerprint after validation"""
		if self.chunk_markdown and not self.chunk_fingerprint:
			self.chunk_fingerprint = generate_content_fingerprint(self.chunk_markdown)
		return self
	
	@computed_field
	@property
	def chunk_size(self) -> int:
		"""Calculate chunk size in characters"""
		return len(self.chunk_text)
	
	@computed_field
	@property
	def is_embedded(self) -> bool:
		"""Check if chunk has vector embeddings"""
		return self.vector_embeddings is not None and len(self.vector_embeddings) == self.vector_dimensions


class GraphRAGNode(BaseModel):
	"""GraphRAG knowledge graph node"""
	model_config = model_config
	
	id: str = Field(default_factory=uuid7str, description="Unique node identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	record_id: str = Field(..., description="Source data record ID")
	node_type: str = Field(..., min_length=1, max_length=100, description="Node type (entity, concept, etc.)")
	node_name: str = Field(..., min_length=1, max_length=1000, description="Node name")
	node_description: Optional[str] = Field(None, description="Node description")
	node_properties: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Node properties")
	entity_type: Optional[str] = Field(None, max_length=100, description="Entity type if node represents entity")
	confidence_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Node confidence score")
	salience_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Node salience score")
	vector_embeddings: Optional[List[float]] = Field(None, description="Node vector embeddings")
	related_chunks: List[str] = Field(default_factory=list, description="Related RAG chunk IDs")
	knowledge_graph_id: Optional[str] = Field(None, description="Parent knowledge graph ID")
	node_status: Literal["active", "merged", "deprecated", "pending_review"] = Field("active", description="Node status")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	
	@computed_field
	@property
	def is_high_confidence(self) -> bool:
		"""Check if node has high confidence"""
		return self.confidence_score >= 0.8 and self.salience_score >= 0.7


class GraphRAGRelation(BaseModel):
	"""GraphRAG knowledge graph relation"""
	model_config = model_config
	
	id: str = Field(default_factory=uuid7str, description="Unique relation identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	source_node_id: str = Field(..., description="Source node ID")
	target_node_id: str = Field(..., description="Target node ID")
	relation_type: str = Field(..., min_length=1, max_length=100, description="Relation type")
	relation_label: str = Field(..., min_length=1, max_length=500, description="Relation label")
	relation_properties: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Relation properties")
	confidence_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Relation confidence score")
	strength_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Relation strength score")
	evidence_chunks: List[str] = Field(default_factory=list, description="Evidence chunk IDs")
	context_window: Optional[str] = Field(None, max_length=2000, description="Context window where relation was extracted")
	knowledge_graph_id: Optional[str] = Field(None, description="Parent knowledge graph ID")
	relation_status: Literal["active", "verified", "disputed", "deprecated"] = Field("active", description="Relation status")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	
	@computed_field
	@property
	def is_verified(self) -> bool:
		"""Check if relation is verified"""
		return (
			self.confidence_score >= 0.8 and
			self.strength_score >= 0.7 and
			len(self.evidence_chunks) >= 2
		)


class KnowledgeGraph(BaseModel):
	"""Knowledge graph container for GraphRAG"""
	model_config = model_config
	
	id: str = Field(default_factory=uuid7str, description="Unique knowledge graph identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	graph_name: str = Field(..., min_length=1, max_length=500, description="Knowledge graph name")
	description: Optional[str] = Field(None, max_length=2000, description="Knowledge graph description")
	domain: str = Field(..., min_length=1, max_length=100, description="Domain or topic area")
	node_count: int = Field(0, ge=0, description="Total number of nodes")
	relation_count: int = Field(0, ge=0, description="Total number of relations")
	entity_types: List[str] = Field(default_factory=list, description="Entity types in graph")
	relation_types: List[str] = Field(default_factory=list, description="Relation types in graph")
	graph_statistics: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Graph statistics")
	graph_schema: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Graph schema definition")
	indexing_config: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Indexing configuration")
	graph_status: Literal["building", "active", "updating", "archived"] = Field("building", description="Graph status")
	last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	created_by: Optional[str] = Field(None, description="Creator user ID")
	
	@computed_field
	@property
	def graph_density(self) -> float:
		"""Calculate graph density"""
		if self.node_count <= 1:
			return 0.0
		max_relations = self.node_count * (self.node_count - 1)
		if max_relations == 0:
			return 0.0
		return self.relation_count / max_relations
	
	@computed_field
	@property
	def is_substantial(self) -> bool:
		"""Check if graph is substantial for analysis"""
		return self.node_count >= 10 and self.relation_count >= 5


class ContentFingerprint(BaseModel):
	"""Content fingerprint for duplicate detection and versioning"""
	model_config = model_config
	
	id: str = Field(default_factory=uuid7str, description="Unique fingerprint identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	fingerprint_hash: str = Field(..., min_length=64, max_length=64, description="SHA-256 content hash")
	content_type: str = Field(..., min_length=1, max_length=100, description="Content type")
	content_length: int = Field(..., ge=0, description="Content length in characters")
	source_url: str = Field(..., min_length=1, description="Source URL")
	related_records: List[str] = Field(default_factory=list, description="Related data record IDs")
	first_seen: datetime = Field(default_factory=datetime.utcnow, description="First time fingerprint was seen")
	last_seen: datetime = Field(default_factory=datetime.utcnow, description="Last time fingerprint was seen")
	occurrence_count: int = Field(1, ge=1, description="Number of times this fingerprint occurred")
	duplicate_cluster_id: Optional[str] = Field(None, description="Duplicate cluster identifier")
	content_similarity_scores: Annotated[Dict[str, float], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Similarity scores with other content")
	fingerprint_metadata: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Fingerprint metadata")
	status: Literal["unique", "duplicate", "near_duplicate", "canonical"] = Field("unique", description="Fingerprint status")
	
	@computed_field
	@property
	def is_frequent_duplicate(self) -> bool:
		"""Check if content is a frequent duplicate"""
		return self.occurrence_count >= 5 and self.status in ["duplicate", "near_duplicate"]


# =====================================================
# REAL-TIME ANALYTICS MODELS
# =====================================================

class AnalyticsInsight(BaseModel):
	"""Real-time analytics insight"""
	model_config = model_config
	
	id: str = Field(default_factory=uuid7str, description="Unique insight identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	insight_type: InsightType = Field(..., description="Type of insight")
	data_source: str = Field(..., min_length=1, max_length=100, description="Data source")
	insight_title: str = Field(..., min_length=1, max_length=1000, description="Insight title")
	insight_description: str = Field(..., min_length=1, description="Insight description")
	insight_data: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Insight data")
	confidence_score: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Confidence score")
	business_impact: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.0, description="Business impact score")
	actionable_recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="Actionable recommendations")
	related_entities: List[str] = Field(default_factory=list, description="Related entities")
	time_window_start: Optional[datetime] = Field(None, description="Time window start")
	time_window_end: Optional[datetime] = Field(None, description="Time window end")
	status: Literal["active", "resolved", "dismissed", "archived"] = Field("active", description="Insight status")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	
	@computed_field
	@property
	def priority_score(self) -> float:
		"""Compute priority score based on confidence and business impact"""
		return (self.confidence_score * 0.3 + self.business_impact * 0.7)


class StreamSession(BaseModel):
	"""Real-time data stream session"""
	model_config = model_config
	
	id: str = Field(default_factory=uuid7str, description="Unique session identifier")
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	session_name: str = Field(..., min_length=1, max_length=500, description="Session name")
	target_id: str = Field(..., description="Crawl target ID")
	pipeline_id: Optional[str] = Field(None, description="Pipeline ID")
	stream_config: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Stream configuration")
	processing_config: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Processing configuration")
	session_status: SessionStatus = Field(SessionStatus.DRAFT, description="Session status")
	start_time: datetime = Field(default_factory=datetime.utcnow, description="Session start time")
	end_time: Optional[datetime] = Field(None, description="Session end time")
	records_processed: int = Field(0, ge=0, description="Records processed")
	records_per_second: float = Field(0.0, ge=0.0, description="Processing rate")
	error_count: int = Field(0, ge=0, description="Error count")
	quality_metrics: QualityMetrics = Field(default_factory=QualityMetrics, description="Quality metrics")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
	
	@computed_field
	@property
	def duration_seconds(self) -> int:
		"""Calculate session duration in seconds"""
		end = self.end_time or datetime.utcnow()
		return int((end - self.start_time).total_seconds())
	
	@computed_field
	@property
	def error_rate(self) -> float:
		"""Calculate error rate"""
		if self.records_processed == 0:
			return 0.0
		return self.error_count / self.records_processed


# =====================================================
# RAG/GRAPHRAG REQUEST/RESPONSE MODELS
# =====================================================

class RAGProcessingRequest(BaseModel):
	"""Request model for RAG processing"""
	model_config = model_config
	
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	record_ids: List[str] = Field(..., min_items=1, description="Data record IDs to process")
	rag_config: RAGProcessingConfig = Field(..., description="RAG processing configuration")
	force_reprocessing: bool = Field(False, description="Force reprocessing of existing chunks")
	priority_level: int = Field(5, ge=1, le=10, description="Processing priority (1=highest)")


class GraphRAGProcessingRequest(BaseModel):
	"""Request model for GraphRAG processing"""
	model_config = model_config
	
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	rag_chunk_ids: List[str] = Field(..., min_items=1, description="RAG chunk IDs to process")
	knowledge_graph_id: Optional[str] = Field(None, description="Target knowledge graph ID")
	entity_extraction_config: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Entity extraction configuration")
	relation_extraction_config: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Relation extraction configuration")
	merge_similar_entities: bool = Field(True, description="Merge similar entities")
	priority_level: int = Field(5, ge=1, le=10, description="Processing priority (1=highest)")


class ContentCleaningRequest(BaseModel):
	"""Request model for content cleaning and markdown conversion"""
	model_config = model_config
	
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	record_ids: List[str] = Field(..., min_items=1, description="Data record IDs to clean")
	cleaning_config: ContentCleaningConfig = Field(..., description="Content cleaning configuration")
	generate_fingerprints: bool = Field(True, description="Generate content fingerprints")
	detect_duplicates: bool = Field(True, description="Detect duplicate content")
	priority_level: int = Field(5, ge=1, le=10, description="Processing priority (1=highest)")


class RAGProcessingResult(BaseModel):
	"""Result of RAG processing operations"""
	model_config = model_config
	
	operation_id: str = Field(default_factory=uuid7str, description="Operation identifier")
	status: ProcessingStatus = Field(..., description="Processing status")
	records_processed: int = Field(0, ge=0, description="Records processed")
	chunks_created: int = Field(0, ge=0, description="RAG chunks created")
	chunks_embedded: int = Field(0, ge=0, description="Chunks with embeddings")
	chunks_indexed: int = Field(0, ge=0, description="Chunks indexed")
	processing_time_ms: float = Field(0.0, ge=0.0, description="Processing time in milliseconds")
	created_chunk_ids: List[str] = Field(default_factory=list, description="Created chunk IDs")
	vector_index_status: Literal["pending", "indexing", "indexed", "failed"] = Field("pending", description="Vector index status")
	error_count: int = Field(0, ge=0, description="Number of errors")
	errors: List[str] = Field(default_factory=list, description="Error messages")
	warnings: List[str] = Field(default_factory=list, description="Warning messages")
	metadata: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Additional metadata")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	
	@computed_field
	@property
	chunking_success_rate(self) -> float:
		"""Calculate chunking success rate"""
		if self.records_processed == 0:
			return 0.0
		return self.chunks_created / self.records_processed
	
	@computed_field
	@property
	embedding_success_rate(self) -> float:
		"""Calculate embedding success rate"""
		if self.chunks_created == 0:
			return 0.0
		return self.chunks_embedded / self.chunks_created


class GraphRAGProcessingResult(BaseModel):
	"""Result of GraphRAG processing operations"""
	model_config = model_config
	
	operation_id: str = Field(default_factory=uuid7str, description="Operation identifier")
	status: ProcessingStatus = Field(..., description="Processing status")
	chunks_processed: int = Field(0, ge=0, description="Chunks processed")
	nodes_created: int = Field(0, ge=0, description="Graph nodes created")
	relations_created: int = Field(0, ge=0, description="Graph relations created")
	entities_merged: int = Field(0, ge=0, description="Entities merged")
	processing_time_ms: float = Field(0.0, ge=0.0, description="Processing time in milliseconds")
	created_node_ids: List[str] = Field(default_factory=list, description="Created node IDs")
	created_relation_ids: List[str] = Field(default_factory=list, description="Created relation IDs")
	knowledge_graph_id: Optional[str] = Field(None, description="Updated knowledge graph ID")
	graph_statistics: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Updated graph statistics")
	error_count: int = Field(0, ge=0, description="Number of errors")
	errors: List[str] = Field(default_factory=list, description="Error messages")
	warnings: List[str] = Field(default_factory=list, description="Warning messages")
	metadata: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Additional metadata")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	
	@computed_field
	@property
	entity_extraction_rate(self) -> float:
		"""Calculate entity extraction rate"""
		if self.chunks_processed == 0:
			return 0.0
		return self.nodes_created / self.chunks_processed
	
	@computed_field
	@property
	relation_extraction_rate(self) -> float:
		"""Calculate relation extraction rate"""
		if self.nodes_created == 0:
			return 0.0
		return self.relations_created / self.nodes_created


# =====================================================
# REQUEST/RESPONSE MODELS
# =====================================================

class CrawlTargetRequest(BaseModel):
	"""Request model for creating crawl targets"""
	model_config = model_config
	
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	name: str = Field(..., min_length=1, max_length=500, description="Target name")
	description: Optional[str] = Field(None, max_length=2000, description="Target description")
	target_urls: Annotated[List[str], AfterValidator(validate_url_list)] = Field(..., description="Target URLs")
	target_type: TargetType = Field(TargetType.WEB_CRAWL, description="Type of crawl target")
	business_context: BusinessContext = Field(..., description="Business context")
	stealth_requirements: Optional[StealthRequirements] = Field(None, description="Stealth requirements")
	quality_requirements: Optional[QualityRequirements] = Field(None, description="Quality requirements")
	scheduling_config: Optional[SchedulingConfig] = Field(None, description="Scheduling configuration")
	collaboration_config: Optional[CollaborationConfig] = Field(None, description="Collaboration configuration")


class ValidationSessionRequest(BaseModel):
	"""Request model for creating validation sessions"""
	model_config = model_config
	
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant identifier")
	dataset_id: str = Field(..., description="Dataset ID")
	session_name: str = Field(..., min_length=1, max_length=500, description="Session name")
	description: Optional[str] = Field(None, max_length=2000, description="Session description")
	validation_schema: ValidationSchema = Field(..., description="Validation schema")
	consensus_threshold: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.8, description="Consensus threshold")
	quality_threshold: Annotated[float, AfterValidator(validate_confidence_score)] = Field(0.7, description="Quality threshold")
	validator_profiles: List[ValidatorProfile] = Field(..., min_items=1, description="Validator profiles")


class ProcessingResult(BaseModel):
	"""Result of data processing operations"""
	model_config = model_config
	
	operation_id: str = Field(default_factory=uuid7str, description="Operation identifier")
	status: ProcessingStatus = Field(..., description="Processing status")
	records_processed: int = Field(0, ge=0, description="Records processed")
	records_successful: int = Field(0, ge=0, description="Successful records")
	records_failed: int = Field(0, ge=0, description="Failed records")
	processing_time_ms: float = Field(0.0, ge=0.0, description="Processing time in milliseconds")
	quality_metrics: QualityMetrics = Field(default_factory=QualityMetrics, description="Quality metrics")
	insights_generated: List[AnalyticsInsight] = Field(default_factory=list, description="Generated insights")
	errors: List[str] = Field(default_factory=list, description="Error messages")
	warnings: List[str] = Field(default_factory=list, description="Warning messages")
	metadata: Annotated[Dict[str, Any], AfterValidator(validate_json_dict)] = Field(default_factory=dict, description="Additional metadata")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	
	@computed_field
	@property
	def success_rate(self) -> float:
		"""Calculate success rate"""
		if self.records_processed == 0:
			return 0.0
		return self.records_successful / self.records_processed
	
	@computed_field
	@property
	def records_per_second(self) -> float:
		"""Calculate processing rate"""
		if self.processing_time_ms == 0:
			return 0.0
		return self.records_processed / (self.processing_time_ms / 1000.0)


# =====================================================
# EXPORT ALL MODELS
# =====================================================

__all__ = [
	# Enums
	"TargetType",
	"ExtractionMethod", 
	"ValidationStatus",
	"SessionStatus",
	"StealthStrategy",
	"ProtectionType",
	"InsightType",
	"ProcessingStatus",
	
	# Core Models
	"BusinessContext",
	"StealthRequirements",
	"QualityRequirements",
	"SchedulingConfig",
	"CollaborationConfig",
	"CrawlTarget",
	"DataSource",
	
	# Data Models
	"QualityMetrics",
	"ExtractedDataset",
	"BusinessEntity",
	"DataRecord",
	
	# Validation Models
	"ValidationSchema",
	"ValidatorProfile",
	"ValidationFeedback", 
	"ConsensusMetrics",
	"ValidationSession",
	
	# Processing Models
	"StealthStrategy",
	"ProtectionProfile",
	"AnalyticsInsight",
	"StreamSession",
	
	# RAG/GraphRAG Models
	"RAGChunk",
	"GraphRAGNode",
	"GraphRAGRelation",
	"KnowledgeGraph",
	"ContentFingerprint",
	"RAGProcessingConfig",
	"ContentCleaningConfig",
	"ContentProcessingStage",
	"RAGIntegrationType",
	
	# Request/Response Models
	"CrawlTargetRequest",
	"ValidationSessionRequest",
	"ProcessingResult",
	"RAGProcessingRequest",
	"GraphRAGProcessingRequest",
	"ContentCleaningRequest",
	"RAGProcessingResult",
	"GraphRAGProcessingResult",
	
	# Validation Functions
	"validate_url_list",
	"validate_tenant_id",
	"validate_json_dict",
	"validate_confidence_score",
	"validate_quality_rating",
]
"""
APG Document Service Data Models

Comprehensive data models for intelligent document management with APG integration.
Uses Pydantic v2 with modern typing patterns following APG standards.

Author: Nyimbi Odero <nyimbi@gmail.com>
Copyright: © 2025 Datacraft
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Union, Annotated
from enum import Enum
from uuid_extensions import uuid7str
from pydantic import BaseModel, Field, ConfigDict, AfterValidator, field_validator
from sqlalchemy import Column, String, Text, DateTime, Integer, Float, Boolean, JSON, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.orm import declarative_base

# SQLAlchemy Base
Base = declarative_base()


def validate_non_empty_string(v: str) -> str:
	"""Validator for non-empty strings"""
	if not v or not v.strip():
		raise ValueError("String cannot be empty")
	return v.strip()


def validate_positive_int(v: int) -> int:
	"""Validator for positive integers"""
	if v <= 0:
		raise ValueError("Value must be positive")
	return v


def validate_file_path(v: str) -> str:
	"""Validator for file paths"""
	if not v:
		return v
	# Basic path validation
	if len(v) > 1000:
		raise ValueError("File path too long")
	return v


# Enums for document service
class DocumentStatus(str, Enum):
	"""Document lifecycle status"""
	DRAFT = "draft"
	PROCESSING = "processing" 
	PUBLISHED = "published"
	ARCHIVED = "archived"
	DELETED = "deleted"
	QUARANTINED = "quarantined"
	# Legacy support
	GENERATING = "generating"
	READY = "ready"
	EXPIRED = "expired"
	ERROR = "error"


class DocumentType(str, Enum):
	"""Document content types"""
	TEXT = "text"
	PDF = "pdf"
	IMAGE = "image"
	SPREADSHEET = "spreadsheet"
	PRESENTATION = "presentation"
	AUDIO = "audio"
	VIDEO = "video"
	ARCHIVE = "archive"
	OTHER = "other"
	# Legacy support
	REPORT = "report"
	DASHBOARD = "dashboard"
	AUDIT_LOG = "audit_log"
	WORKFLOW_SUMMARY = "workflow_summary"
	METRICS_REPORT = "metrics_report"
	COMPLIANCE_REPORT = "compliance_report"
	PERFORMANCE_REPORT = "performance_report"
	USER_GUIDE = "user_guide"
	API_DOCUMENTATION = "api_documentation"
	CUSTOM = "custom"


class DocumentFormat(str, Enum):
	"""Document format enumeration"""
	PDF = "pdf"
	DOCX = "docx"
	XLSX = "xlsx"
	HTML = "html"
	JSON = "json"
	CSV = "csv"
	XML = "xml"
	MARKDOWN = "markdown"
	TXT = "txt"
	RTF = "rtf"
	ODT = "odt"
	JPEG = "jpeg"
	JPG = "jpg"
	PNG = "png"
	TIFF = "tiff"
	GIF = "gif"
	BMP = "bmp"
	WEBP = "webp"


class ProcessingStatus(str, Enum):
	"""Document processing status"""
	PENDING = "pending"
	RUNNING = "running"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"


class ClassificationLevel(str, Enum):
	"""Document security classification levels"""
	PUBLIC = "public"
	INTERNAL = "internal"
	CONFIDENTIAL = "confidential"
	RESTRICTED = "restricted"
	TOP_SECRET = "top_secret"


class WorkflowStatus(str, Enum):
	"""Workflow execution status"""
	CREATED = "created"
	RUNNING = "running"
	PAUSED = "paused"
	COMPLETED = "completed"
	FAILED = "failed"
	CANCELLED = "cancelled"


class TemplateCategory(str, Enum):
	"""Template categories"""
	GENERAL = "general"
	LEGAL = "legal"
	BUSINESS = "business"
	TECHNICAL = "technical"
	MARKETING = "marketing"
	FINANCIAL = "financial"
	HR = "hr"
	COMPLIANCE = "compliance"


class MetricType(str, Enum):
	"""Metric type enumeration"""
	COUNTER = "counter"
	GAUGE = "gauge"
	HISTOGRAM = "histogram"
	SUMMARY = "summary"
	TIMER = "timer"


# Core Pydantic Models (placed in models.py per APG standards)

class DSDocument(BaseModel):
	"""
	Enhanced document model with APG integration patterns.
	
	Supports intelligent processing, multi-tenant isolation, real-time collaboration,
	and comprehensive audit trails following APG coding standards.
	"""
	
	# APG Standard Fields
	document_id: str = Field(default_factory=uuid7str, description="Unique document identifier")
	tenant_id: Annotated[str, AfterValidator(validate_non_empty_string)] = Field(description="Tenant identifier for multi-tenancy")
	created_by: Annotated[str, AfterValidator(validate_non_empty_string)] = Field(description="User who created the document")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	modified_by: Optional[str] = Field(None, description="User who last modified the document")
	modified_at: Optional[datetime] = Field(None, description="Last modification timestamp")
	
	# Document Core Properties
	title: Annotated[str, AfterValidator(validate_non_empty_string)] = Field(description="Document title")
	description: Optional[str] = Field(None, description="Document description")
	content: Optional[str] = Field(None, description="Document text content")
	file_path: Optional[Annotated[str, AfterValidator(validate_file_path)]] = Field(None, description="File system path")
	file_size: Optional[Annotated[int, AfterValidator(validate_positive_int)]] = Field(None, description="File size in bytes")
	mime_type: Optional[str] = Field(None, description="MIME type of the document")
	file_hash: Optional[str] = Field(None, description="SHA256 hash for integrity verification")
	
	# Document Classification and Metadata
	document_type: DocumentType = Field(default=DocumentType.TEXT, description="Document type classification")
	classification: ClassificationLevel = Field(default=ClassificationLevel.INTERNAL, description="Security classification")
	tags: List[str] = Field(default_factory=list, description="Document tags for organization")
	custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")
	
	# AI Processing Results
	extracted_text: Optional[str] = Field(None, description="OCR extracted text content")
	extracted_entities: List[Dict[str, Any]] = Field(default_factory=list, description="NLP extracted entities")
	content_summary: Optional[str] = Field(None, description="AI generated summary")
	topics: List[Dict[str, Any]] = Field(default_factory=list, description="Identified content topics")
	sentiment_analysis: Optional[Dict[str, Any]] = Field(None, description="Content sentiment analysis")
	language_detection: Optional[str] = Field(None, description="Detected content language")
	confidence_scores: Dict[str, float] = Field(default_factory=dict, description="AI confidence scores")
	
	# Processing Status
	status: DocumentStatus = Field(default=DocumentStatus.DRAFT, description="Document lifecycle status")
	processing_status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="AI processing status")
	processing_started_at: Optional[datetime] = Field(None, description="Processing start timestamp")
	processing_completed_at: Optional[datetime] = Field(None, description="Processing completion timestamp")
	processing_error: Optional[str] = Field(None, description="Processing error message if failed")
	
	# Collaboration and Workflow
	version_number: int = Field(default=1, description="Document version number")
	parent_document_id: Optional[str] = Field(None, description="Parent document for versioning")
	workflow_id: Optional[str] = Field(None, description="Associated workflow identifier")
	approval_status: Optional[str] = Field(None, description="Approval workflow status")
	collaborators: List[str] = Field(default_factory=list, description="List of collaborator user IDs")
	current_editors: List[str] = Field(default_factory=list, description="Currently active editors")
	
	# Access Control and Compliance
	access_permissions: Dict[str, List[str]] = Field(default_factory=dict, description="Role-based access permissions")
	sharing_settings: Dict[str, Any] = Field(default_factory=dict, description="Document sharing configuration")
	retention_date: Optional[datetime] = Field(None, description="Document retention/deletion date")
	compliance_tags: List[str] = Field(default_factory=list, description="Compliance requirement tags")
	
	# Performance and Analytics
	view_count: int = Field(default=0, description="Number of times document was viewed")
	download_count: int = Field(default=0, description="Number of times document was downloaded")
	last_accessed_at: Optional[datetime] = Field(None, description="Last access timestamp")
	last_accessed_by: Optional[str] = Field(None, description="User who last accessed document")
	
	# APG Configuration
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		use_enum_values=True
	)
	
	@field_validator('file_size')
	@classmethod
	def validate_file_size(cls, v):
		if v is not None and v > 100 * 1024 * 1024:  # 100MB limit
			raise ValueError('File size cannot exceed 100MB')
		return v
	
	@field_validator('tags')
	@classmethod
	def validate_tags(cls, v):
		if len(v) > 50:
			raise ValueError('Cannot have more than 50 tags')
		return [tag.strip().lower() for tag in v if tag.strip()]
	
	def is_processing_complete(self) -> bool:
		"""Check if document processing is complete"""
		return self.processing_status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CANCELLED]
	
	def get_processing_duration(self) -> Optional[float]:
		"""Get processing duration in seconds"""
		if not self.processing_started_at:
			return None
		end_time = self.processing_completed_at or datetime.utcnow()
		return (end_time - self.processing_started_at).total_seconds()


class DSTemplate(BaseModel):
	"""Document template model for automated document generation"""
	
	# APG Standard Fields
	template_id: str = Field(default_factory=uuid7str, description="Unique template identifier")
	tenant_id: Annotated[str, AfterValidator(validate_non_empty_string)] = Field(description="Tenant identifier")
	created_by: Annotated[str, AfterValidator(validate_non_empty_string)] = Field(description="Template creator")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	modified_by: Optional[str] = Field(None, description="Last modifier")
	modified_at: Optional[datetime] = Field(None, description="Last modification timestamp")
	
	# Template Properties
	name: Annotated[str, AfterValidator(validate_non_empty_string)] = Field(description="Template name")
	description: Optional[str] = Field(None, description="Template description")
	category: TemplateCategory = Field(default=TemplateCategory.GENERAL, description="Template category")
	template_content: str = Field(description="Template content with variables")
	template_variables: Dict[str, str] = Field(default_factory=dict, description="Variable definitions")
	
	# Template Configuration
	default_classification: ClassificationLevel = Field(default=ClassificationLevel.INTERNAL, description="Default security classification")
	default_tags: List[str] = Field(default_factory=list, description="Default tags for generated documents")
	output_format: DocumentType = Field(default=DocumentType.TEXT, description="Output document format")
	
	# Usage and Analytics
	usage_count: int = Field(default=0, description="Number of times template was used")
	last_used_at: Optional[datetime] = Field(None, description="Last usage timestamp")
	last_used_by: Optional[str] = Field(None, description="User who last used template")
	
	# Template Status
	is_active: bool = Field(default=True, description="Whether template is active")
	version: str = Field(default="1.0.0", description="Template version")
	
	# APG Configuration
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		use_enum_values=True
	)


class DSWorkflow(BaseModel):
	"""Document workflow model for process automation"""
	
	# APG Standard Fields
	workflow_id: str = Field(default_factory=uuid7str, description="Unique workflow identifier")
	tenant_id: Annotated[str, AfterValidator(validate_non_empty_string)] = Field(description="Tenant identifier")
	created_by: Annotated[str, AfterValidator(validate_non_empty_string)] = Field(description="Workflow creator")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	
	# Workflow Definition
	name: Annotated[str, AfterValidator(validate_non_empty_string)] = Field(description="Workflow name")
	description: Optional[str] = Field(None, description="Workflow description")
	workflow_type: str = Field(description="Type of workflow (approval, processing, etc.)")
	steps: List[Dict[str, Any]] = Field(default_factory=list, description="Workflow step definitions")
	
	# Execution Status
	status: WorkflowStatus = Field(default=WorkflowStatus.CREATED, description="Workflow execution status")
	current_step: int = Field(default=0, description="Current step index")
	started_at: Optional[datetime] = Field(None, description="Execution start timestamp")
	completed_at: Optional[datetime] = Field(None, description="Execution completion timestamp")
	
	# Associated Documents
	document_ids: List[str] = Field(default_factory=list, description="Associated document identifiers")
	
	# Results and Metrics
	execution_results: Dict[str, Any] = Field(default_factory=dict, description="Workflow execution results")
	error_messages: List[str] = Field(default_factory=list, description="Error messages if failed")
	performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
	
	# APG Configuration
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		use_enum_values=True
	)
	
	def is_complete(self) -> bool:
		"""Check if workflow is complete"""
		return self.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]
	
	def get_execution_duration(self) -> Optional[float]:
		"""Get workflow execution duration in seconds"""
		if not self.started_at:
			return None
		end_time = self.completed_at or datetime.utcnow()
		return (end_time - self.started_at).total_seconds()


class DSProcessingJob(BaseModel):
	"""Document processing job model for async operations"""
	
	# APG Standard Fields
	job_id: str = Field(default_factory=uuid7str, description="Unique job identifier")
	tenant_id: Annotated[str, AfterValidator(validate_non_empty_string)] = Field(description="Tenant identifier")
	created_by: Annotated[str, AfterValidator(validate_non_empty_string)] = Field(description="Job creator")
	created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
	
	# Job Definition
	job_name: Annotated[str, AfterValidator(validate_non_empty_string)] = Field(description="Job name")
	processing_type: str = Field(description="Type of processing (ocr, nlp, classification, etc.)")
	input_file_path: Annotated[str, AfterValidator(validate_file_path)] = Field(description="Input file path")
	document_id: Optional[str] = Field(None, description="Associated document ID")
	
	# Processing Configuration
	processing_parameters: Dict[str, Any] = Field(default_factory=dict, description="Processing configuration parameters")
	priority: int = Field(default=5, description="Job priority (1-10, higher is more urgent)")
	retry_count: int = Field(default=0, description="Number of retry attempts")
	max_retries: int = Field(default=3, description="Maximum retry attempts")
	
	# Job Status
	status: ProcessingStatus = Field(default=ProcessingStatus.PENDING, description="Job processing status")
	started_at: Optional[datetime] = Field(None, description="Processing start timestamp")
	completed_at: Optional[datetime] = Field(None, description="Processing completion timestamp")
	progress_percentage: float = Field(default=0.0, description="Processing progress percentage")
	
	# Results
	output_data: Dict[str, Any] = Field(default_factory=dict, description="Processing output data")
	error_message: Optional[str] = Field(None, description="Error message if failed")
	processing_metrics: Dict[str, float] = Field(default_factory=dict, description="Processing performance metrics")
	
	# Resource Usage
	cpu_time_seconds: Optional[float] = Field(None, description="CPU time used in seconds")
	memory_usage_mb: Optional[float] = Field(None, description="Peak memory usage in MB")
	
	# APG Configuration
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		use_enum_values=True
	)
	
	@field_validator('priority')
	@classmethod
	def validate_priority(cls, v):
		if not 1 <= v <= 10:
			raise ValueError('Priority must be between 1 and 10')
		return v
	
	@field_validator('progress_percentage')
	@classmethod
	def validate_progress(cls, v):
		if not 0.0 <= v <= 100.0:
			raise ValueError('Progress percentage must be between 0 and 100')
		return v
	
	def is_completed(self) -> bool:
		"""Check if processing job is completed"""
		return self.status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED, ProcessingStatus.CANCELLED]
	
	def can_retry(self) -> bool:
		"""Check if job can be retried"""
		return self.status == ProcessingStatus.FAILED and self.retry_count < self.max_retries
	
	def get_duration_seconds(self) -> Optional[float]:
		"""Get job duration in seconds"""
		if not self.started_at:
			return None
		end_time = self.completed_at or datetime.utcnow()
		return (end_time - self.started_at).total_seconds()


# SQLAlchemy Models for Database

class DocumentTemplate(Base):
	"""Document template SQLAlchemy model"""
	__tablename__ = "ds_document_templates"
	
	template_id = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id = Column(String(100), nullable=False)
	name = Column(String(255), nullable=False)
	description = Column(Text, nullable=True)
	category = Column(String(50), nullable=False, default=TemplateCategory.GENERAL.value)
	document_type = Column(String(50), nullable=False)
	template_content = Column(Text, nullable=False)
	template_variables = Column(JSON, nullable=True)
	default_classification = Column(String(20), nullable=False, default=ClassificationLevel.INTERNAL.value)
	default_tags = Column(JSON, nullable=True)
	output_format = Column(String(20), nullable=False, default=DocumentType.TEXT.value)
	
	# Usage tracking
	usage_count = Column(Integer, default=0)
	last_used_at = Column(DateTime(timezone=True), nullable=True)
	last_used_by = Column(String(100), nullable=True)
	
	# Template status
	is_active = Column(Boolean, default=True)
	version = Column(String(20), default="1.0.0")
	
	# APG standard fields
	created_by = Column(String(100), nullable=False)
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	modified_by = Column(String(100), nullable=True)
	modified_at = Column(DateTime(timezone=True), nullable=True)
	
	# Indexes
	__table_args__ = (
		Index('idx_template_tenant', 'tenant_id'),
		Index('idx_template_category', 'category'),
		Index('idx_template_type', 'document_type'),
		Index('idx_template_active', 'is_active'),
	)


class Document(Base):
	"""Document SQLAlchemy model"""
	__tablename__ = "ds_documents"
	
	document_id = Column(String(50), primary_key=True, default=uuid7str)
	tenant_id = Column(String(100), nullable=False)
	title = Column(String(500), nullable=False)
	description = Column(Text, nullable=True)
	content = Column(Text, nullable=True)
	file_path = Column(String(1000), nullable=True)
	file_size = Column(Integer, nullable=True)
	mime_type = Column(String(100), nullable=True)
	file_hash = Column(String(64), nullable=True)
	
	# Classification and metadata
	document_type = Column(String(50), nullable=False, default=DocumentType.TEXT.value)
	classification = Column(String(20), nullable=False, default=ClassificationLevel.INTERNAL.value)
	tags = Column(JSON, nullable=True)
	custom_metadata = Column(JSON, nullable=True)
	
	# AI processing results
	extracted_text = Column(Text, nullable=True)
	extracted_entities = Column(JSON, nullable=True)
	content_summary = Column(Text, nullable=True)
	topics = Column(JSON, nullable=True)
	sentiment_analysis = Column(JSON, nullable=True)
	language_detection = Column(String(10), nullable=True)
	confidence_scores = Column(JSON, nullable=True)
	
	# Processing status
	status = Column(String(20), default=DocumentStatus.DRAFT.value)
	processing_status = Column(String(20), default=ProcessingStatus.PENDING.value)
	processing_started_at = Column(DateTime(timezone=True), nullable=True)
	processing_completed_at = Column(DateTime(timezone=True), nullable=True)
	processing_error = Column(Text, nullable=True)
	
	# Collaboration and workflow
	version_number = Column(Integer, default=1)
	parent_document_id = Column(String(50), nullable=True)
	workflow_id = Column(String(50), nullable=True)
	approval_status = Column(String(50), nullable=True)
	collaborators = Column(JSON, nullable=True)
	current_editors = Column(JSON, nullable=True)
	
	# Access control and compliance
	access_permissions = Column(JSON, nullable=True)
	sharing_settings = Column(JSON, nullable=True)
	retention_date = Column(DateTime(timezone=True), nullable=True)
	compliance_tags = Column(JSON, nullable=True)
	
	# Performance and analytics
	view_count = Column(Integer, default=0)
	download_count = Column(Integer, default=0)
	last_accessed_at = Column(DateTime(timezone=True), nullable=True)
	last_accessed_by = Column(String(100), nullable=True)
	
	# APG standard fields
	created_by = Column(String(100), nullable=False)
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	modified_by = Column(String(100), nullable=True)
	modified_at = Column(DateTime(timezone=True), nullable=True)
	
	# Indexes
	__table_args__ = (
		Index('idx_document_tenant', 'tenant_id'),
		Index('idx_document_type', 'document_type'),
		Index('idx_document_status', 'status'),
		Index('idx_document_classification', 'classification'),
		Index('idx_document_created_by', 'created_by'),
		Index('idx_document_created_at', 'created_at'),
		Index('idx_document_processing_status', 'processing_status'),
	)


class Metric(Base):
	"""Metric data point SQLAlchemy model"""
	__tablename__ = "ds_metrics"
	
	metric_id = Column(String(50), primary_key=True, default=uuid7str)
	metric_name = Column(String(255), nullable=False)
	metric_type = Column(String(20), nullable=False)
	value = Column(Float, nullable=False)
	string_value = Column(String(1000), nullable=True)
	tags = Column(JSON, nullable=True)
	timestamp = Column(DateTime(timezone=True), nullable=False)
	source = Column(String(100), nullable=False)
	tenant_id = Column(String(100), nullable=True)
	partition_date = Column(String(10), nullable=False)
	
	# Indexes
	__table_args__ = (
		Index('idx_metric_name_timestamp', 'metric_name', 'timestamp'),
		Index('idx_metric_source', 'source'),
		Index('idx_metric_type', 'metric_type'),
		Index('idx_metric_partition', 'partition_date'),
		Index('idx_metric_tenant', 'tenant_id'),
		Index('idx_metric_timestamp', 'timestamp'),
	)


class MetricSummary(Base):
	"""Pre-aggregated metric summaries SQLAlchemy model"""
	__tablename__ = "ds_metric_summaries"
	
	summary_id = Column(String(50), primary_key=True, default=uuid7str)
	metric_name = Column(String(255), nullable=False)
	summary_type = Column(String(20), nullable=False)
	start_time = Column(DateTime(timezone=True), nullable=False)
	end_time = Column(DateTime(timezone=True), nullable=False)
	
	# Statistical aggregations
	count = Column(Integer, nullable=False)
	sum_value = Column(Float, nullable=True)
	min_value = Column(Float, nullable=True)
	max_value = Column(Float, nullable=True)
	avg_value = Column(Float, nullable=True)
	median_value = Column(Float, nullable=True)
	std_dev = Column(Float, nullable=True)
	percentile_95 = Column(Float, nullable=True)
	percentile_99 = Column(Float, nullable=True)
	
	unique_sources = Column(JSON, nullable=True)
	unique_tags = Column(JSON, nullable=True)
	tenant_id = Column(String(100), nullable=True)
	created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	
	# Indexes
	__table_args__ = (
		Index('idx_summary_metric_type_time', 'metric_name', 'summary_type', 'start_time'),
		Index('idx_summary_tenant', 'tenant_id'),
	)


class DocumentAccess(Base):
	"""Document access log SQLAlchemy model"""
	__tablename__ = "ds_document_access"
	
	access_id = Column(String(50), primary_key=True, default=uuid7str)
	document_id = Column(String(50), nullable=False)
	accessed_by = Column(String(100), nullable=False)
	access_type = Column(String(20), nullable=False)
	ip_address = Column(String(45), nullable=True)
	user_agent = Column(String(500), nullable=True)
	referer = Column(String(1000), nullable=True)
	accessed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
	tenant_id = Column(String(100), nullable=True)
	
	# Indexes
	__table_args__ = (
		Index('idx_access_document', 'document_id'),
		Index('idx_access_user', 'accessed_by'),
		Index('idx_access_time', 'accessed_at'),
		Index('idx_access_tenant', 'tenant_id'),
	)


# API Request/Response Models

class DocumentCreateRequest(BaseModel):
	"""Request model for creating documents"""
	title: Annotated[str, AfterValidator(validate_non_empty_string)]
	description: Optional[str] = None
	content: Optional[str] = None
	classification: ClassificationLevel = ClassificationLevel.INTERNAL
	tags: List[str] = Field(default_factory=list)
	template_id: Optional[str] = None
	custom_metadata: Dict[str, Any] = Field(default_factory=dict)
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)


class DocumentUpdateRequest(BaseModel):
	"""Request model for updating documents"""
	title: Optional[str] = None
	description: Optional[str] = None  
	content: Optional[str] = None
	classification: Optional[ClassificationLevel] = None
	tags: Optional[List[str]] = None
	custom_metadata: Optional[Dict[str, Any]] = None
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)


class DocumentResponse(BaseModel):
	"""Response model for document operations"""
	document_id: str
	title: str
	status: DocumentStatus
	processing_status: ProcessingStatus
	created_at: datetime
	modified_at: Optional[datetime]
	file_size: Optional[int]
	processing_progress: Optional[float] = None
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)


class DocumentSearchRequest(BaseModel):
	"""Request model for document search"""
	query: str
	filters: Dict[str, Any] = Field(default_factory=dict)
	sort_by: str = Field(default="relevance")
	limit: int = Field(default=20, ge=1, le=100)
	offset: int = Field(default=0, ge=0)
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)


class DocumentSearchResponse(BaseModel):
	"""Response model for document search results"""
	total_count: int
	results: List[DocumentResponse]
	facets: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
	search_time_ms: float
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)


class ProcessingJobResponse(BaseModel):
	"""Response model for processing jobs"""
	job_id: str
	status: ProcessingStatus
	progress_percentage: float
	started_at: Optional[datetime]
	estimated_completion: Optional[datetime]
	error_message: Optional[str]
	
	model_config = ConfigDict(extra='forbid', validate_by_name=True)


# Legacy Support Models for backward compatibility

class DocumentGenerationRequest(BaseModel):
	"""Legacy request model for generating documents"""
	document_type: DocumentType
	format: DocumentFormat
	title: str = Field(..., min_length=1, max_length=500)
	data: Dict[str, Any] = Field(..., min_length=1)
	template_id: Optional[str] = None
	expires_hours: Optional[int] = Field(default=24, ge=1, le=8760)
	metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
	
	model_config = ConfigDict(extra='forbid', validate_assignment=True)


class MetricRequest(BaseModel):
	"""Request model for recording metrics"""
	metric_name: str = Field(..., min_length=1, max_length=255)
	value: float
	metric_type: MetricType = MetricType.GAUGE
	tags: Optional[Dict[str, str]] = Field(default_factory=dict)
	timestamp: Optional[datetime] = None
	source: str = Field(default="api", max_length=100)
	
	model_config = ConfigDict(extra='forbid', validate_assignment=True)


class MetricResponse(BaseModel):
	"""Response model for metrics"""
	metric_id: str
	metric_name: str
	metric_type: MetricType
	value: float
	tags: Dict[str, str]
	timestamp: datetime
	source: str
	
	model_config = ConfigDict(extra='forbid')
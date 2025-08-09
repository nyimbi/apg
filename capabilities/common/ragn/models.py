"""
APG RAG Capability Data Models

Comprehensive Pydantic v2 models following APG standards for RAG operations
with PostgreSQL + pgvector + pgai support and multi-tenant isolation.
"""

from typing import Dict, Any, List, Optional, Union, Literal, Annotated
from datetime import datetime
from enum import Enum
import json
from pydantic import BaseModel, Field, ConfigDict, AfterValidator, field_validator, model_validator
from uuid_extensions import uuid7str

# ============================================================================
# ENUMS AND TYPES
# ============================================================================

class DocumentStatus(str, Enum):
	"""Document processing status"""
	PENDING = "pending"
	PROCESSING = "processing" 
	COMPLETED = "completed"
	FAILED = "failed"

class KnowledgeBaseStatus(str, Enum):
	"""Knowledge base status"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	INDEXING = "indexing"
	ERROR = "error"

class ConversationStatus(str, Enum):
	"""Conversation status"""
	ACTIVE = "active"
	COMPLETED = "completed"
	ARCHIVED = "archived"

class TurnType(str, Enum):
	"""Conversation turn type"""
	USER = "user"
	ASSISTANT = "assistant"
	SYSTEM = "system"

class ValidationStatus(str, Enum):
	"""Response validation status"""
	PENDING = "pending"
	APPROVED = "approved"
	REJECTED = "rejected"
	NEEDS_REVIEW = "needs_review"

class EntityType(str, Enum):
	"""Knowledge graph entity types"""
	PERSON = "PERSON"
	ORGANIZATION = "ORG"
	LOCATION = "GPE"
	MONEY = "MONEY"
	DATE = "DATE"
	EMAIL = "EMAIL"
	PHONE = "PHONE"
	PRODUCT = "PRODUCT"
	CONCEPT = "CONCEPT"
	OTHER = "OTHER"

class RetrievalMethod(str, Enum):
	"""Retrieval methods for RAG"""
	VECTOR_SIMILARITY = "vector_similarity"
	HYBRID_SEARCH = "hybrid_search"
	KNOWLEDGE_GRAPH = "knowledge_graph"
	FULL_TEXT = "full_text"

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_tenant_id(v: str) -> str:
	"""Validate tenant ID format"""
	if not v or not v.strip():
		raise ValueError("Tenant ID cannot be empty")
	if len(v) > 255:
		raise ValueError("Tenant ID too long")
	return v.strip()

def validate_capability_id(v: str) -> str:
	"""Validate capability ID format"""
	if not v or not v.strip():
		raise ValueError("Capability ID cannot be empty")
	if len(v) > 100:
		raise ValueError("Capability ID too long")
	return v.strip()

def validate_embedding_vector(v: Optional[List[float]]) -> Optional[List[float]]:
	"""Validate embedding vector dimensions"""
	if v is None:
		return v
	if len(v) != 1024:  # bge-m3 standard dimension
		raise ValueError("Embedding vector must have 1024 dimensions for bge-m3")
	if not all(isinstance(x, (int, float)) for x in v):
		raise ValueError("Embedding vector must contain only numbers")
	return v

def validate_similarity_threshold(v: float) -> float:
	"""Validate similarity threshold range"""
	if not 0.0 <= v <= 1.0:
		raise ValueError("Similarity threshold must be between 0.0 and 1.0")
	return v

def validate_chunk_size(v: int) -> int:
	"""Validate chunk size limits"""
	if not 100 <= v <= 8192:  # Reasonable limits
		raise ValueError("Chunk size must be between 100 and 8192")
	return v

def validate_json_data(v: Union[Dict[str, Any], str]) -> Dict[str, Any]:
	"""Validate and parse JSON data"""
	if isinstance(v, dict):
		return v
	if isinstance(v, str):
		try:
			return json.loads(v)
		except json.JSONDecodeError:
			raise ValueError("Invalid JSON format")
	raise ValueError("JSON data must be dict or valid JSON string")

# ============================================================================
# BASE MODELS
# ============================================================================

class APGBaseModel(BaseModel):
	"""Base model with APG standards"""
	model_config = ConfigDict(
		extra='forbid',
		validate_assignment=True,
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		use_enum_values=True
	)

class TimestampedModel(APGBaseModel):
	"""Base model with timestamp fields"""
	created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
	updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")
	created_by: Optional[str] = Field(default=None, max_length=255, description="Created by user")
	updated_by: Optional[str] = Field(default=None, max_length=255, description="Updated by user")

class TenantIsolatedModel(TimestampedModel):
	"""Base model with tenant isolation"""
	tenant_id: Annotated[str, AfterValidator(validate_tenant_id)] = Field(..., description="Tenant ID for isolation")

# ============================================================================
# KNOWLEDGE BASE MODELS
# ============================================================================

class KnowledgeBaseCreate(APGBaseModel):
	"""Create knowledge base request"""
	name: str = Field(..., min_length=1, max_length=255, description="Knowledge base name")
	description: Optional[str] = Field(default=None, description="Knowledge base description")
	capability_id: Annotated[str, AfterValidator(validate_capability_id)] = Field(..., description="APG capability ID")
	
	# Configuration
	embedding_model: str = Field(default="bge-m3", max_length=100, description="Embedding model")
	generation_model: str = Field(default="qwen3", max_length=100, description="Generation model")
	chunk_size: Annotated[int, AfterValidator(validate_chunk_size)] = Field(default=1000, description="Text chunk size")
	chunk_overlap: int = Field(default=100, ge=0, le=500, description="Chunk overlap size")
	
	# Vector configuration
	vector_dimensions: int = Field(default=1024, gt=0, description="Vector dimensions")
	similarity_threshold: Annotated[float, AfterValidator(validate_similarity_threshold)] = Field(default=0.7, description="Similarity threshold")
	
	# APG integration
	apg_context: Annotated[Dict[str, Any], AfterValidator(validate_json_data)] = Field(default_factory=dict, description="APG context data")
	sharing_permissions: Annotated[Dict[str, Any], AfterValidator(validate_json_data)] = Field(default_factory=dict, description="Cross-capability sharing")

class KnowledgeBase(TenantIsolatedModel):
	"""Knowledge base model"""
	id: str = Field(default_factory=uuid7str, description="Knowledge base ID")
	name: str = Field(..., min_length=1, max_length=255, description="Knowledge base name")
	description: Optional[str] = Field(default=None, description="Knowledge base description")
	capability_id: Annotated[str, AfterValidator(validate_capability_id)] = Field(..., description="APG capability ID")
	
	# Configuration
	embedding_model: str = Field(default="bge-m3", max_length=100, description="Embedding model")
	generation_model: str = Field(default="qwen3", max_length=100, description="Generation model")
	chunk_size: Annotated[int, AfterValidator(validate_chunk_size)] = Field(default=1000, description="Text chunk size")
	chunk_overlap: int = Field(default=100, ge=0, le=500, description="Chunk overlap size")
	
	# Vector configuration
	vector_dimensions: int = Field(default=1024, gt=0, description="Vector dimensions")
	similarity_threshold: Annotated[float, AfterValidator(validate_similarity_threshold)] = Field(default=0.7, description="Similarity threshold")
	
	# Status and metadata
	status: KnowledgeBaseStatus = Field(default=KnowledgeBaseStatus.ACTIVE, description="Knowledge base status")
	document_count: int = Field(default=0, ge=0, description="Number of documents")
	total_chunks: int = Field(default=0, ge=0, description="Total number of chunks")
	last_indexed_at: Optional[datetime] = Field(default=None, description="Last indexing timestamp")
	
	# APG integration
	apg_context: Annotated[Dict[str, Any], AfterValidator(validate_json_data)] = Field(default_factory=dict, description="APG context data")
	sharing_permissions: Annotated[Dict[str, Any], AfterValidator(validate_json_data)] = Field(default_factory=dict, description="Cross-capability sharing")

class KnowledgeBaseUpdate(APGBaseModel):
	"""Update knowledge base request"""
	name: Optional[str] = Field(default=None, min_length=1, max_length=255, description="Knowledge base name")
	description: Optional[str] = Field(default=None, description="Knowledge base description")
	status: Optional[KnowledgeBaseStatus] = Field(default=None, description="Knowledge base status")
	apg_context: Optional[Annotated[Dict[str, Any], AfterValidator(validate_json_data)]] = Field(default=None, description="APG context data")
	sharing_permissions: Optional[Annotated[Dict[str, Any], AfterValidator(validate_json_data)]] = Field(default=None, description="Cross-capability sharing")

# ============================================================================
# DOCUMENT MODELS
# ============================================================================

class DocumentCreate(APGBaseModel):
	"""Create document request"""
	knowledge_base_id: str = Field(..., description="Knowledge base ID")
	source_path: str = Field(..., min_length=1, description="Source file path")
	filename: str = Field(..., min_length=1, max_length=500, description="Original filename")
	content_type: str = Field(..., max_length=100, description="MIME content type")
	content: str = Field(..., min_length=1, description="Document content")
	title: Optional[str] = Field(default=None, description="Document title")
	metadata: Annotated[Dict[str, Any], AfterValidator(validate_json_data)] = Field(default_factory=dict, description="Document metadata")
	language: str = Field(default="en", max_length=10, description="Document language")

class Document(TenantIsolatedModel):
	"""Document model"""
	id: str = Field(default_factory=uuid7str, description="Document ID")
	knowledge_base_id: str = Field(..., description="Knowledge base ID")
	
	# Document identification
	source_path: str = Field(..., min_length=1, description="Source file path")
	filename: str = Field(..., min_length=1, max_length=500, description="Original filename")
	file_hash: str = Field(..., min_length=1, max_length=64, description="File SHA-256 hash")
	content_type: str = Field(..., max_length=100, description="MIME content type")
	file_size: int = Field(..., gt=0, description="File size in bytes")
	
	# Content and metadata
	title: Optional[str] = Field(default=None, description="Document title")
	content: str = Field(..., min_length=1, description="Document content")
	metadata: Annotated[Dict[str, Any], AfterValidator(validate_json_data)] = Field(default_factory=dict, description="Document metadata")
	language: str = Field(default="en", max_length=10, description="Document language")
	
	# Processing status
	processing_status: DocumentStatus = Field(default=DocumentStatus.PENDING, description="Processing status")
	processing_error: Optional[str] = Field(default=None, description="Processing error message")
	chunk_count: int = Field(default=0, ge=0, description="Number of chunks created")
	
	# Version control
	version: int = Field(default=1, gt=0, description="Document version")
	parent_document_id: Optional[str] = Field(default=None, description="Parent document ID")

class DocumentUpdate(APGBaseModel):
	"""Update document request"""
	title: Optional[str] = Field(default=None, description="Document title")
	metadata: Optional[Annotated[Dict[str, Any], AfterValidator(validate_json_data)]] = Field(default=None, description="Document metadata")
	processing_status: Optional[DocumentStatus] = Field(default=None, description="Processing status")
	processing_error: Optional[str] = Field(default=None, description="Processing error message")

# ============================================================================
# DOCUMENT CHUNK MODELS
# ============================================================================

class DocumentChunkCreate(APGBaseModel):
	"""Create document chunk request"""
	document_id: str = Field(..., description="Document ID")
	knowledge_base_id: str = Field(..., description="Knowledge base ID")
	chunk_index: int = Field(..., ge=0, description="Chunk index in document")
	content: str = Field(..., min_length=1, description="Chunk content")
	embedding: Annotated[List[float], AfterValidator(validate_embedding_vector)] = Field(..., description="Embedding vector")
	
	# Optional positioning and metadata
	start_position: Optional[int] = Field(default=None, ge=0, description="Start position in document")
	end_position: Optional[int] = Field(default=None, ge=0, description="End position in document")
	token_count: Optional[int] = Field(default=None, gt=0, description="Number of tokens")
	section_title: Optional[str] = Field(default=None, description="Section title")
	section_level: int = Field(default=0, ge=0, description="Section hierarchy level")

class DocumentChunk(TenantIsolatedModel):
	"""Document chunk model"""
	id: str = Field(default_factory=uuid7str, description="Chunk ID")
	document_id: str = Field(..., description="Document ID")
	knowledge_base_id: str = Field(..., description="Knowledge base ID")
	
	# Chunk identification and content
	chunk_index: int = Field(..., ge=0, description="Chunk index in document")
	content: str = Field(..., min_length=1, description="Chunk content")
	content_hash: str = Field(..., min_length=1, max_length=64, description="Content hash for deduplication")
	
	# Vector embeddings
	embedding: Annotated[List[float], AfterValidator(validate_embedding_vector)] = Field(..., description="Embedding vector")
	
	# Chunk metadata and positioning
	start_position: Optional[int] = Field(default=None, ge=0, description="Start position in document")
	end_position: Optional[int] = Field(default=None, ge=0, description="End position in document")
	token_count: Optional[int] = Field(default=None, gt=0, description="Number of tokens")
	character_count: int = Field(..., gt=0, description="Number of characters")
	
	# Hierarchical relationships
	parent_chunk_id: Optional[str] = Field(default=None, description="Parent chunk ID")
	section_title: Optional[str] = Field(default=None, description="Section title")
	section_level: int = Field(default=0, ge=0, description="Section hierarchy level")
	
	# Quality and confidence metrics
	embedding_confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Embedding confidence")
	content_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Content quality score")
	
	# Processing metadata
	processed_at: datetime = Field(default_factory=datetime.now, description="Processing timestamp")
	embedding_model: str = Field(default="bge-m3", max_length=100, description="Embedding model used")
	
	@field_validator('end_position')
	@classmethod
	def validate_positions(cls, v, info):
		"""Validate end position is after start position"""
		if v is not None and info.data.get('start_position') is not None:
			if v <= info.data['start_position']:
				raise ValueError("End position must be greater than start position")
		return v

# ============================================================================
# CONVERSATION MODELS
# ============================================================================

class ConversationCreate(APGBaseModel):
	"""Create conversation request"""
	knowledge_base_id: Optional[str] = Field(default=None, description="Associated knowledge base ID")
	title: Optional[str] = Field(default=None, max_length=500, description="Conversation title")
	description: Optional[str] = Field(default=None, description="Conversation description")
	generation_model: str = Field(default="qwen3", max_length=100, description="Generation model")
	max_context_tokens: int = Field(default=4096, gt=0, description="Maximum context tokens")
	temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
	user_id: Optional[str] = Field(default=None, max_length=255, description="User ID")
	session_id: Optional[str] = Field(default=None, max_length=255, description="Session ID")

class Conversation(TenantIsolatedModel):
	"""Conversation model"""
	id: str = Field(default_factory=uuid7str, description="Conversation ID")
	knowledge_base_id: Optional[str] = Field(default=None, description="Associated knowledge base ID")
	
	# Conversation metadata
	title: Optional[str] = Field(default=None, max_length=500, description="Conversation title")
	description: Optional[str] = Field(default=None, description="Conversation description")
	context_summary: Optional[str] = Field(default=None, description="Context summary")
	
	# Configuration
	generation_model: str = Field(default="qwen3", max_length=100, description="Generation model")
	max_context_tokens: int = Field(default=4096, gt=0, description="Maximum context tokens")
	temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
	
	# State management
	status: ConversationStatus = Field(default=ConversationStatus.ACTIVE, description="Conversation status")
	turn_count: int = Field(default=0, ge=0, description="Number of turns")
	total_tokens_used: int = Field(default=0, ge=0, description="Total tokens used")
	
	# User and session info
	user_id: Optional[str] = Field(default=None, max_length=255, description="User ID")
	session_id: Optional[str] = Field(default=None, max_length=255, description="Session ID")

class ConversationUpdate(APGBaseModel):
	"""Update conversation request"""
	title: Optional[str] = Field(default=None, max_length=500, description="Conversation title")
	description: Optional[str] = Field(default=None, description="Conversation description")
	status: Optional[ConversationStatus] = Field(default=None, description="Conversation status")
	context_summary: Optional[str] = Field(default=None, description="Context summary")

# ============================================================================
# CONVERSATION TURN MODELS
# ============================================================================

class ConversationTurnCreate(APGBaseModel):
	"""Create conversation turn request"""
	conversation_id: str = Field(..., description="Conversation ID")
	turn_type: TurnType = Field(..., description="Turn type")
	content: str = Field(..., min_length=1, description="Turn content")
	query_embedding: Optional[Annotated[List[float], AfterValidator(validate_embedding_vector)]] = Field(default=None, description="Query embedding")
	retrieved_chunks: List[str] = Field(default_factory=list, description="Retrieved chunk IDs")
	retrieval_scores: List[float] = Field(default_factory=list, description="Retrieval scores")
	model_used: Optional[str] = Field(default=None, max_length=100, description="Model used")
	context_used: Optional[str] = Field(default=None, description="Context used")

class ConversationTurn(TenantIsolatedModel):
	"""Conversation turn model"""
	id: str = Field(default_factory=uuid7str, description="Turn ID")
	conversation_id: str = Field(..., description="Conversation ID")
	
	# Turn identification
	turn_number: int = Field(..., ge=0, description="Turn number in conversation")
	turn_type: TurnType = Field(..., description="Turn type")
	
	# Content
	content: str = Field(..., min_length=1, description="Turn content")
	content_tokens: Optional[int] = Field(default=None, gt=0, description="Content token count")
	
	# RAG-specific data
	query_embedding: Optional[Annotated[List[float], AfterValidator(validate_embedding_vector)]] = Field(default=None, description="Query embedding")
	retrieved_chunks: List[str] = Field(default_factory=list, description="Retrieved chunk IDs")
	retrieval_scores: List[float] = Field(default_factory=list, description="Retrieval scores")
	
	# Generation metadata
	model_used: Optional[str] = Field(default=None, max_length=100, description="Model used")
	generation_time_ms: Optional[int] = Field(default=None, ge=0, description="Generation time in milliseconds")
	generation_tokens: Optional[int] = Field(default=None, ge=0, description="Generation token count")
	confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
	
	# Context and memory
	context_used: Optional[str] = Field(default=None, description="Context used")
	memory_summary: Optional[str] = Field(default=None, description="Memory summary")

# ============================================================================
# RETRIEVAL MODELS
# ============================================================================

class RetrievalRequest(APGBaseModel):
	"""Retrieval request"""
	query_text: str = Field(..., min_length=1, description="Query text")
	knowledge_base_id: str = Field(..., description="Knowledge base ID")
	k_retrievals: int = Field(default=10, gt=0, le=100, description="Number of retrievals")
	similarity_threshold: Annotated[float, AfterValidator(validate_similarity_threshold)] = Field(default=0.7, description="Similarity threshold")
	retrieval_method: RetrievalMethod = Field(default=RetrievalMethod.VECTOR_SIMILARITY, description="Retrieval method")
	filters: Annotated[Dict[str, Any], AfterValidator(validate_json_data)] = Field(default_factory=dict, description="Retrieval filters")

class RetrievalResult(TenantIsolatedModel):
	"""Retrieval result model"""
	id: str = Field(default_factory=uuid7str, description="Retrieval result ID")
	conversation_turn_id: Optional[str] = Field(default=None, description="Associated conversation turn ID")
	
	# Query information
	query_text: str = Field(..., min_length=1, description="Query text")
	query_embedding: Annotated[List[float], AfterValidator(validate_embedding_vector)] = Field(..., description="Query embedding")
	query_hash: str = Field(..., min_length=1, max_length=64, description="Query hash")
	
	# Retrieval configuration
	knowledge_base_id: str = Field(..., description="Knowledge base ID")
	k_retrievals: int = Field(default=10, gt=0, description="Number of retrievals requested")
	similarity_threshold: Annotated[float, AfterValidator(validate_similarity_threshold)] = Field(default=0.7, description="Similarity threshold")
	
	# Results
	retrieved_chunk_ids: List[str] = Field(..., description="Retrieved chunk IDs")
	similarity_scores: List[float] = Field(..., description="Similarity scores")
	retrieval_method: RetrievalMethod = Field(default=RetrievalMethod.VECTOR_SIMILARITY, description="Retrieval method")
	
	# Performance metrics
	retrieval_time_ms: int = Field(..., ge=0, description="Retrieval time in milliseconds")
	total_candidates: int = Field(default=0, ge=0, description="Total candidates considered")
	
	# Quality metrics
	result_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Result quality score")
	diversity_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Result diversity score")

# ============================================================================
# GENERATION MODELS
# ============================================================================

class GenerationRequest(APGBaseModel):
	"""Generation request"""
	prompt: str = Field(..., min_length=1, description="Generation prompt")
	conversation_id: Optional[str] = Field(default=None, description="Associated conversation ID")
	retrieval_result_id: Optional[str] = Field(default=None, description="Associated retrieval result ID")
	model: str = Field(default="qwen3", max_length=100, description="Generation model")
	max_tokens: int = Field(default=2048, gt=0, description="Maximum tokens to generate")
	temperature: float = Field( default=0.7, ge=0.0, le=2.0, description="Generation temperature")
	context_used: Optional[str] = Field(default=None, description="Context used")
	source_chunks: List[str] = Field(default_factory=list, description="Source chunk IDs")

class GeneratedResponse(TenantIsolatedModel):
	"""Generated response model"""
	id: str = Field(default_factory=uuid7str, description="Generated response ID")
	conversation_turn_id: str = Field(..., description="Associated conversation turn ID")
	retrieval_result_id: Optional[str] = Field(default=None, description="Associated retrieval result ID")
	
	# Generation input
	prompt: str = Field(..., min_length=1, description="Generation prompt")
	context_used: Optional[str] = Field(default=None, description="Context used")
	source_chunks: List[str] = Field(default_factory=list, description="Source chunk IDs")
	
	# Generated content
	response_text: str = Field(..., min_length=1, description="Generated response text")
	response_tokens: Optional[int] = Field(default=None, gt=0, description="Response token count")
	
	# Model and generation metadata
	generation_model: str = Field(..., max_length=100, description="Generation model used")
	model_parameters: Annotated[Dict[str, Any], AfterValidator(validate_json_data)] = Field(default_factory=dict, description="Model parameters")
	generation_time_ms: int = Field(..., ge=0, description="Generation time in milliseconds")
	
	# Quality and attribution
	confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
	factual_accuracy_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Factual accuracy score")
	source_attribution: Annotated[Dict[str, Any], AfterValidator(validate_json_data)] = Field(default_factory=dict, description="Source attribution mapping")
	
	# Validation status
	validation_status: ValidationStatus = Field(default=ValidationStatus.PENDING, description="Validation status")
	validation_feedback: Optional[str] = Field(default=None, description="Validation feedback")

# ============================================================================
# KNOWLEDGE GRAPH MODELS
# ============================================================================

class EntityCreate(APGBaseModel):
	"""Create entity request"""
	knowledge_base_id: str = Field(..., description="Knowledge base ID")
	name: str = Field(..., min_length=1, max_length=500, description="Entity name")
	entity_type: EntityType = Field(..., description="Entity type")
	description: Optional[str] = Field(default=None, description="Entity description")
	properties: Annotated[Dict[str, Any], AfterValidator(validate_json_data)] = Field(default_factory=dict, description="Entity properties")
	embedding: Optional[Annotated[List[float], AfterValidator(validate_embedding_vector)]] = Field(default=None, description="Entity embedding")
	source_documents: List[str] = Field(default_factory=list, description="Source document IDs")

class Entity(TenantIsolatedModel):
	"""Knowledge graph entity model"""
	id: str = Field(default_factory=uuid7str, description="Entity ID")
	knowledge_base_id: str = Field(..., description="Knowledge base ID")
	
	# Entity identification
	name: str = Field(..., min_length=1, max_length=500, description="Entity name")
	entity_type: EntityType = Field(..., description="Entity type")
	normalized_name: Optional[str] = Field(default=None, max_length=500, description="Normalized name")
	
	# Entity embedding and similarity
	embedding: Optional[Annotated[List[float], AfterValidator(validate_embedding_vector)]] = Field(default=None, description="Entity embedding")
	
	# Entity metadata
	description: Optional[str] = Field(default=None, description="Entity description")
	properties: Annotated[Dict[str, Any], AfterValidator(validate_json_data)] = Field(default_factory=dict, description="Entity properties")
	confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
	
	# Frequency and importance
	mention_count: int = Field(default=1, gt=0, description="Number of mentions")
	importance_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Importance score")
	
	# Source tracking
	source_documents: List[str] = Field(default_factory=list, description="Source document IDs")
	first_mentioned_at: datetime = Field(default_factory=datetime.now, description="First mention timestamp")

class RelationshipCreate(APGBaseModel):
	"""Create relationship request"""
	knowledge_base_id: str = Field(..., description="Knowledge base ID")
	source_entity_id: str = Field(..., description="Source entity ID")
	target_entity_id: str = Field(..., description="Target entity ID")
	relationship_type: str = Field(..., min_length=1, max_length=100, description="Relationship type")
	description: Optional[str] = Field(default=None, description="Relationship description")
	properties: Annotated[Dict[str, Any], AfterValidator(validate_json_data)] = Field(default_factory=dict, description="Relationship properties")
	source_chunks: List[str] = Field(default_factory=list, description="Source chunk IDs")
	evidence_text: Optional[str] = Field(default=None, description="Supporting evidence text")

class Relationship(TenantIsolatedModel):
	"""Knowledge graph relationship model"""
	id: str = Field(default_factory=uuid7str, description="Relationship ID")
	knowledge_base_id: str = Field(..., description="Knowledge base ID")
	
	# Relationship definition
	source_entity_id: str = Field(..., description="Source entity ID")
	target_entity_id: str = Field(..., description="Target entity ID")
	relationship_type: str = Field(..., min_length=1, max_length=100, description="Relationship type")
	
	# Relationship metadata
	description: Optional[str] = Field(default=None, description="Relationship description")
	properties: Annotated[Dict[str, Any], AfterValidator(validate_json_data)] = Field(default_factory=dict, description="Relationship properties")
	confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score")
	
	# Frequency and strength
	mention_count: int = Field(default=1, gt=0, description="Number of mentions")
	strength_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Relationship strength")
	
	# Source tracking
	source_chunks: List[str] = Field(default_factory=list, description="Source chunk IDs")
	evidence_text: Optional[str] = Field(default=None, description="Supporting evidence text")
	
	@model_validator(mode='after')
	def validate_not_self_referencing(self):
		"""Ensure relationship is not self-referencing"""
		if self.source_entity_id == self.target_entity_id:
			raise ValueError("Relationship cannot be self-referencing")
		return self

# ============================================================================
# ANALYTICS AND METRICS MODELS
# ============================================================================

class QueryAnalytics(TenantIsolatedModel):
	"""Query analytics model"""
	id: str = Field(default_factory=uuid7str, description="Analytics ID")
	
	# Query information
	query_text: str = Field(..., min_length=1, description="Query text")
	query_hash: str = Field(..., min_length=1, max_length=64, description="Query hash")
	query_type: str = Field(..., max_length=50, description="Query type")
	
	# Performance metrics
	total_time_ms: int = Field(..., ge=0, description="Total processing time")
	retrieval_time_ms: int = Field(default=0, ge=0, description="Retrieval time")
	generation_time_ms: int = Field(default=0, ge=0, description="Generation time")
	
	# Results quality
	results_count: int = Field(default=0, ge=0, description="Number of results")
	user_satisfaction: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="User satisfaction score")
	click_through_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Click-through rate")
	
	# Context
	knowledge_base_id: Optional[str] = Field(default=None, description="Knowledge base ID")
	user_id: Optional[str] = Field(default=None, max_length=255, description="User ID")
	session_id: Optional[str] = Field(default=None, max_length=255, description="Session ID")

class SystemMetrics(TenantIsolatedModel):
	"""System metrics model"""
	id: str = Field(default_factory=uuid7str, description="Metric ID")
	
	# Metric identification
	metric_name: str = Field(..., min_length=1, max_length=100, description="Metric name")
	metric_type: Literal["counter", "gauge", "histogram", "summary"] = Field(..., description="Metric type")
	
	# Metric values
	value_numeric: Optional[float] = Field(default=None, description="Numeric value")
	value_text: Optional[str] = Field(default=None, description="Text value")
	
	# Context and labels
	labels: Annotated[Dict[str, Any], AfterValidator(validate_json_data)] = Field(default_factory=dict, description="Metric labels")
	component: Optional[str] = Field(default=None, max_length=100, description="System component")
	
	# Timestamp
	recorded_at: datetime = Field(default_factory=datetime.now, description="Recording timestamp")

# ============================================================================
# RESPONSE MODELS
# ============================================================================

class PaginatedResponse(APGBaseModel):
	"""Paginated response wrapper"""
	items: List[Any] = Field(..., description="Response items")
	total: int = Field(..., ge=0, description="Total items")
	page: int = Field(..., gt=0, description="Current page")
	per_page: int = Field(..., gt=0, description="Items per page")
	pages: int = Field(..., gt=0, description="Total pages")

class APIResponse(APGBaseModel):
	"""Standard API response"""
	success: bool = Field(..., description="Request success status")
	message: Optional[str] = Field(default=None, description="Response message")
	data: Optional[Any] = Field(default=None, description="Response data")
	errors: List[str] = Field(default_factory=list, description="Error messages")
	request_id: str = Field(default_factory=uuid7str, description="Request ID for tracking")
	timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class HealthCheck(APGBaseModel):
	"""Health check response"""
	status: Literal["healthy", "degraded", "unhealthy"] = Field(..., description="Overall health status")
	components: Dict[str, Dict[str, Any]] = Field(..., description="Component health details")
	timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
	version: str = Field(default="1.0.0", description="Service version")

# ============================================================================
# EXPORT ALL MODELS
# ============================================================================

__all__ = [
	# Enums
	"DocumentStatus", "KnowledgeBaseStatus", "ConversationStatus", 
	"TurnType", "ValidationStatus", "EntityType", "RetrievalMethod",
	
	# Base models
	"APGBaseModel", "TimestampedModel", "TenantIsolatedModel",
	
	# Knowledge base models
	"KnowledgeBaseCreate", "KnowledgeBase", "KnowledgeBaseUpdate",
	
	# Document models
	"DocumentCreate", "Document", "DocumentUpdate",
	
	# Document chunk models
	"DocumentChunkCreate", "DocumentChunk",
	
	# Conversation models
	"ConversationCreate", "Conversation", "ConversationUpdate",
	"ConversationTurnCreate", "ConversationTurn",
	
	# Retrieval models
	"RetrievalRequest", "RetrievalResult",
	
	# Generation models
	"GenerationRequest", "GeneratedResponse",
	
	# Knowledge graph models
	"EntityCreate", "Entity", "RelationshipCreate", "Relationship",
	
	# Analytics models
	"QueryAnalytics", "SystemMetrics",
	
	# Response models
	"PaginatedResponse", "APIResponse", "HealthCheck"
]
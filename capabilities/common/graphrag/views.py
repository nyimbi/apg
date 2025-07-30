"""
APG GraphRAG Capability - Pydantic v2 Data Models

Revolutionary graph-based retrieval-augmented generation with Apache AGE integration.
Comprehensive data models for knowledge graphs, entities, relationships, and reasoning.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
"""

from __future__ import annotations
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict, validator, AfterValidator
from uuid_extensions import uuid7str


# ============================================================================
# CONFIGURATION AND ENUMS
# ============================================================================

class EntityType(str, Enum):
	"""Knowledge graph entity types"""
	PERSON = "person"
	ORGANIZATION = "organization"
	LOCATION = "location"
	CONCEPT = "concept"
	EVENT = "event"
	DOCUMENT = "document"
	PRODUCT = "product"
	TECHNOLOGY = "technology"
	PROCESS = "process"
	CUSTOM = "custom"


class RelationshipType(str, Enum):
	"""Knowledge graph relationship types"""
	RELATED_TO = "related_to"
	PART_OF = "part_of"
	LOCATED_IN = "located_in"
	WORKS_FOR = "works_for"
	CREATED_BY = "created_by"
	INFLUENCES = "influences"
	DEPENDS_ON = "depends_on"
	SIMILAR_TO = "similar_to"
	OPPOSITE_OF = "opposite_of"
	TEMPORAL_BEFORE = "temporal_before"
	TEMPORAL_AFTER = "temporal_after"
	CAUSAL_RELATION = "causal_relation"
	CUSTOM = "custom"


class QueryType(str, Enum):
	"""GraphRAG query processing types"""
	QUESTION_ANSWERING = "question_answering"
	SEMANTIC_SEARCH = "semantic_search"
	GRAPH_EXPLORATION = "graph_exploration"
	KNOWLEDGE_SYNTHESIS = "knowledge_synthesis"
	REASONING_CHAIN = "reasoning_chain"
	ENTITY_DISCOVERY = "entity_discovery"
	RELATIONSHIP_ANALYSIS = "relationship_analysis"


class ExplanationLevel(str, Enum):
	"""Explanation detail levels for GraphRAG responses"""
	MINIMAL = "minimal"
	STANDARD = "standard"
	DETAILED = "detailed"
	COMPREHENSIVE = "comprehensive"


class GraphStatus(str, Enum):
	"""Knowledge graph status values"""
	ACTIVE = "active"
	INACTIVE = "inactive"
	BUILDING = "building"
	ERROR = "error"


class ProcessingStatus(str, Enum):
	"""Processing status for queries and operations"""
	PENDING = "pending"
	PROCESSING = "processing"
	COMPLETED = "completed"
	FAILED = "failed"
	CACHED = "cached"


# ============================================================================
# BASE MODELS AND MIXINS
# ============================================================================

class BaseGraphRAGModel(BaseModel):
	"""Base model for all GraphRAG entities with common fields"""
	model_config = ConfigDict(
		extra='forbid',
		validate_by_name=True,
		validate_by_alias=True,
		str_strip_whitespace=True,
		validate_assignment=True
	)


class TimestampMixin(BaseModel):
	"""Mixin for timestamp fields"""
	created_at: datetime = Field(default_factory=datetime.utcnow)
	updated_at: datetime = Field(default_factory=datetime.utcnow)


class TenantMixin(BaseModel):
	"""Mixin for multi-tenant support"""
	tenant_id: str = Field(min_length=1, max_length=255)


# ============================================================================
# TEMPORAL AND CONTEXT MODELS
# ============================================================================

class TemporalRange(BaseGraphRAGModel):
	"""Temporal validity range for relationships"""
	start_time: Optional[datetime] = None
	end_time: Optional[datetime] = None
	timezone: str = "UTC"
	confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class QueryContext(BaseGraphRAGModel):
	"""Context information for GraphRAG queries"""
	user_id: Optional[str] = None
	session_id: Optional[str] = None
	conversation_history: List[str] = Field(default_factory=list)
	domain_context: Dict[str, Any] = Field(default_factory=dict)
	business_context: Dict[str, Any] = Field(default_factory=dict)
	temporal_context: Optional[TemporalRange] = None
	geographic_context: Dict[str, Any] = Field(default_factory=dict)


class RetrievalConfig(BaseGraphRAGModel):
	"""Configuration for graph retrieval operations"""
	max_entities: int = Field(default=50, ge=1, le=1000)
	max_relationships: int = Field(default=100, ge=1, le=2000)
	similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
	diversity_factor: float = Field(default=0.3, ge=0.0, le=1.0)
	include_communities: bool = True
	temporal_weighting: bool = True
	relationship_types: List[RelationshipType] = Field(default_factory=list)


class ReasoningConfig(BaseGraphRAGModel):
	"""Configuration for reasoning operations"""
	reasoning_depth: int = Field(default=3, ge=1, le=10)
	confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
	explanation_detail: ExplanationLevel = ExplanationLevel.STANDARD
	include_uncertainty: bool = True
	validate_consistency: bool = True
	use_symbolic_reasoning: bool = False


# ============================================================================
# CORE GRAPH MODELS
# ============================================================================

class GraphMetadata(BaseGraphRAGModel):
	"""Metadata for knowledge graphs"""
	total_entities: int = 0
	total_relationships: int = 0
	total_communities: int = 0
	last_updated: datetime = Field(default_factory=datetime.utcnow)
	schema_evolution: List[Dict[str, Any]] = Field(default_factory=list)
	data_sources: List[str] = Field(default_factory=list)
	update_frequency: str = "real_time"
	consistency_score: float = Field(default=1.0, ge=0.0, le=1.0)


class GraphQualityMetrics(BaseGraphRAGModel):
	"""Quality metrics for knowledge graphs"""
	completeness_score: float = Field(ge=0.0, le=1.0)
	accuracy_score: float = Field(ge=0.0, le=1.0)
	consistency_score: float = Field(ge=0.0, le=1.0)
	freshness_score: float = Field(ge=0.0, le=1.0)
	coverage_score: float = Field(ge=0.0, le=1.0)
	reliability_score: float = Field(ge=0.0, le=1.0)
	last_evaluated: datetime = Field(default_factory=datetime.utcnow)
	evaluation_metrics: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraph(BaseGraphRAGModel, TenantMixin, TimestampMixin):
	"""Comprehensive knowledge graph representation"""
	graph_id: str = Field(default_factory=uuid7str)
	name: str = Field(min_length=1, max_length=500)
	description: Optional[str] = None
	schema_version: str = "1.0.0"
	graph_type: str = "knowledge_graph"
	metadata: GraphMetadata = Field(default_factory=GraphMetadata)
	quality_metrics: GraphQualityMetrics
	status: GraphStatus = GraphStatus.ACTIVE


class GraphEntity(BaseGraphRAGModel, TenantMixin, TimestampMixin):
	"""Knowledge graph entity with rich properties"""
	entity_id: str = Field(default_factory=uuid7str)
	knowledge_graph_id: str
	canonical_entity_id: str = Field(min_length=1, max_length=500)
	entity_type: EntityType
	canonical_name: str = Field(min_length=1, max_length=1000)
	aliases: List[str] = Field(default_factory=list)
	properties: Dict[str, Any] = Field(default_factory=dict)
	embeddings: Optional[List[float]] = None
	confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
	evidence_sources: List[str] = Field(default_factory=list)
	provenance: Dict[str, Any] = Field(default_factory=dict)
	quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
	status: str = Field(default="active")

	@validator('embeddings')
	def validate_embeddings(cls, v):
		if v is not None and len(v) != 1024:
			raise ValueError('Embeddings must be 1024 dimensions for bge-m3')
		return v


class GraphRelationship(BaseGraphRAGModel, TenantMixin, TimestampMixin):
	"""Knowledge graph relationship with contextual information"""
	relationship_id: str = Field(default_factory=uuid7str)
	knowledge_graph_id: str
	canonical_relationship_id: str = Field(min_length=1, max_length=500)
	source_entity_id: str
	target_entity_id: str
	relationship_type: RelationshipType
	strength: float = Field(default=0.0, ge=0.0, le=1.0)
	context: Dict[str, Any] = Field(default_factory=dict)
	properties: Dict[str, Any] = Field(default_factory=dict)
	evidence_sources: List[str] = Field(default_factory=list)
	provenance: Dict[str, Any] = Field(default_factory=dict)
	temporal_validity: Optional[TemporalRange] = None
	confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
	quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
	status: str = Field(default="active")

	@validator('source_entity_id', 'target_entity_id')
	def validate_no_self_loops(cls, v, values):
		if 'source_entity_id' in values and v == values['source_entity_id']:
			raise ValueError('Self-loops not allowed in relationships')
		return v


class GraphCommunity(BaseGraphRAGModel, TenantMixin, TimestampMixin):
	"""Graph community detection results"""
	community_id: str = Field(default_factory=uuid7str)
	knowledge_graph_id: str
	canonical_community_id: str = Field(min_length=1, max_length=500)
	name: Optional[str] = None
	description: Optional[str] = None
	algorithm: str = Field(min_length=1, max_length=100)
	members: List[str] = Field(min_items=1)  # Entity IDs
	centrality_metrics: Dict[str, Any] = Field(default_factory=dict)
	cohesion_score: Optional[float] = Field(None, ge=0.0, le=1.0)
	size_metrics: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# QUERY AND RESPONSE MODELS
# ============================================================================

class GraphRAGQuery(BaseGraphRAGModel, TenantMixin):
	"""GraphRAG query with context and preferences"""
	query_id: str = Field(default_factory=uuid7str)
	knowledge_graph_id: str
	query_text: str = Field(min_length=1)
	query_type: QueryType = QueryType.QUESTION_ANSWERING
	query_embedding: Optional[List[float]] = None
	context: Optional[QueryContext] = None
	retrieval_config: RetrievalConfig = Field(default_factory=RetrievalConfig)
	reasoning_config: ReasoningConfig = Field(default_factory=ReasoningConfig)
	explanation_level: ExplanationLevel = ExplanationLevel.STANDARD
	max_hops: int = Field(default=3, ge=1, le=10)
	status: ProcessingStatus = ProcessingStatus.PENDING
	processing_time_ms: Optional[int] = None
	created_at: datetime = Field(default_factory=datetime.utcnow)
	completed_at: Optional[datetime] = None

	@validator('query_embedding')
	def validate_query_embeddings(cls, v):
		if v is not None and len(v) != 1024:
			raise ValueError('Query embeddings must be 1024 dimensions for bge-m3')
		return v


class Evidence(BaseGraphRAGModel):
	"""Supporting evidence for GraphRAG responses"""
	evidence_id: str = Field(default_factory=uuid7str)
	source_type: str  # entity, relationship, document, etc.
	source_id: str
	content: str
	confidence: float = Field(ge=0.0, le=1.0)
	relevance_score: float = Field(ge=0.0, le=1.0)
	provenance: Dict[str, Any] = Field(default_factory=dict)


class GraphPath(BaseGraphRAGModel):
	"""Path through the knowledge graph"""
	path_id: str = Field(default_factory=uuid7str)
	entities: List[str]  # Entity IDs in path order
	relationships: List[str]  # Relationship IDs connecting entities
	path_length: int = Field(ge=1)
	path_strength: float = Field(ge=0.0, le=1.0)
	semantic_coherence: float = Field(ge=0.0, le=1.0)
	confidence: float = Field(ge=0.0, le=1.0)


class EntityMention(BaseGraphRAGModel):
	"""Entity mention in GraphRAG response"""
	mention_id: str = Field(default_factory=uuid7str)
	entity_id: str
	mention_text: str
	position_start: int = Field(ge=0)
	position_end: int = Field(ge=0)
	confidence: float = Field(ge=0.0, le=1.0)

	@validator('position_end')
	def validate_position_order(cls, v, values):
		if 'position_start' in values and v <= values['position_start']:
			raise ValueError('Position end must be greater than position start')
		return v


class SourceAttribution(BaseGraphRAGModel):
	"""Source attribution for generated content"""
	source_id: str
	source_type: str  # entity, relationship, document
	contribution_weight: float = Field(ge=0.0, le=1.0)
	citation_text: str
	confidence: float = Field(ge=0.0, le=1.0)


class ReasoningStep(BaseGraphRAGModel):
	"""Individual step in reasoning chain"""
	step_id: str = Field(default_factory=uuid7str)
	step_number: int = Field(ge=1)
	operation: str  # retrieve, reason, synthesize, validate
	description: str
	inputs: Dict[str, Any] = Field(default_factory=dict)
	outputs: Dict[str, Any] = Field(default_factory=dict)
	confidence: float = Field(ge=0.0, le=1.0)
	execution_time_ms: int = Field(ge=0)


class ReasoningChain(BaseGraphRAGModel):
	"""Complete reasoning process chain"""
	chain_id: str = Field(default_factory=uuid7str)
	steps: List[ReasoningStep]
	total_steps: int = Field(ge=1)
	overall_confidence: float = Field(ge=0.0, le=1.0)
	reasoning_type: str = "multi_hop_graph"
	validation_results: Dict[str, Any] = Field(default_factory=dict)


class QualityIndicators(BaseGraphRAGModel):
	"""Quality indicators for GraphRAG responses"""
	factual_accuracy: float = Field(ge=0.0, le=1.0)
	completeness: float = Field(ge=0.0, le=1.0)
	relevance: float = Field(ge=0.0, le=1.0)
	coherence: float = Field(ge=0.0, le=1.0)
	clarity: float = Field(ge=0.0, le=1.0)
	confidence: float = Field(ge=0.0, le=1.0)
	source_reliability: float = Field(ge=0.0, le=1.0)


class GraphRAGResponse(BaseGraphRAGModel, TenantMixin):
	"""Comprehensive GraphRAG response with reasoning chains"""
	response_id: str = Field(default_factory=uuid7str)
	query_id: str
	answer: str = Field(min_length=1)
	confidence_score: float = Field(ge=0.0, le=1.0)
	reasoning_chain: ReasoningChain
	supporting_evidence: List[Evidence] = Field(default_factory=list)
	graph_paths: List[GraphPath] = Field(default_factory=list)
	entity_mentions: List[EntityMention] = Field(default_factory=list)
	source_attribution: List[SourceAttribution] = Field(default_factory=list)
	quality_indicators: QualityIndicators
	processing_metrics: Dict[str, Any] = Field(default_factory=dict)
	model_used: Optional[str] = None
	created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# CURATION AND COLLABORATION MODELS
# ============================================================================

class ExpertParticipant(BaseGraphRAGModel):
	"""Expert participant in curation workflows"""
	user_id: str
	role: str  # reviewer, approver, domain_expert
	expertise_areas: List[str] = Field(default_factory=list)
	weight: float = Field(default=1.0, ge=0.0, le=5.0)
	performance_metrics: Dict[str, Any] = Field(default_factory=dict)


class CurationWorkflow(BaseGraphRAGModel, TenantMixin, TimestampMixin):
	"""Collaborative knowledge curation workflow"""
	workflow_id: str = Field(default_factory=uuid7str)
	knowledge_graph_id: str
	name: str = Field(min_length=1, max_length=500)
	description: Optional[str] = None
	workflow_type: str = Field(min_length=1, max_length=100)
	participants: List[ExpertParticipant] = Field(min_items=1)
	consensus_threshold: float = Field(default=0.80, ge=0.50, le=1.00)
	status: str = Field(default="active")
	metrics: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeEdit(BaseGraphRAGModel, TenantMixin):
	"""Proposed changes to knowledge graph"""
	edit_id: str = Field(default_factory=uuid7str)
	workflow_id: str
	knowledge_graph_id: str
	editor_id: str
	edit_type: str  # create, update, delete, merge, split
	target_type: str  # entity, relationship, community, graph
	target_id: str
	proposed_changes: Dict[str, Any] = Field(min_items=1)
	justification: Optional[str] = None
	evidence: List[str] = Field(default_factory=list)
	status: str = Field(default="pending")
	reviews: List[Dict[str, Any]] = Field(default_factory=list)
	consensus_score: Optional[float] = Field(None, ge=0.0, le=1.0)
	applied_at: Optional[datetime] = None
	created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# ANALYTICS AND PERFORMANCE MODELS
# ============================================================================

class PerformanceMetrics(BaseGraphRAGModel):
	"""Performance metrics for GraphRAG operations"""
	retrieval_time_ms: int = Field(ge=0)
	reasoning_time_ms: int = Field(ge=0)
	generation_time_ms: int = Field(ge=0)
	total_time_ms: int = Field(ge=0)
	entities_retrieved: int = Field(ge=0)
	relationships_traversed: int = Field(ge=0)
	graph_hops: int = Field(ge=0)
	memory_usage_mb: int = Field(ge=0)
	cache_hits: int = Field(ge=0)
	cache_misses: int = Field(ge=0)
	model_tokens: int = Field(ge=0)


class GraphAnalytics(BaseGraphRAGModel, TenantMixin):
	"""Analytics data for knowledge graphs"""
	analytics_id: str = Field(default_factory=uuid7str)
	knowledge_graph_id: str
	metric_type: str  # performance, usage, quality, accuracy, efficiency
	metric_name: str = Field(min_length=1, max_length=200)
	metric_value: Optional[Decimal] = None
	metric_data: Dict[str, Any] = Field(default_factory=dict)
	time_period: Optional[str] = None  # hour, day, week, month
	created_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# REQUEST AND RESPONSE MODELS FOR API
# ============================================================================

class CreateKnowledgeGraphRequest(BaseGraphRAGModel, TenantMixin):
	"""Request to create a new knowledge graph"""
	name: str = Field(min_length=1, max_length=500)
	description: Optional[str] = None
	initial_documents: List[str] = Field(default_factory=list)
	configuration: Dict[str, Any] = Field(default_factory=dict)


class UpdateKnowledgeGraphRequest(BaseGraphRAGModel):
	"""Request to update knowledge graph"""
	name: Optional[str] = None
	description: Optional[str] = None
	configuration: Optional[Dict[str, Any]] = None
	status: Optional[GraphStatus] = None


class GraphQueryRequest(BaseGraphRAGModel, TenantMixin):
	"""Request for GraphRAG query processing"""
	knowledge_graph_id: str
	query_text: str = Field(min_length=1)
	query_type: QueryType = QueryType.QUESTION_ANSWERING
	context: Optional[QueryContext] = None
	retrieval_config: Optional[RetrievalConfig] = None
	reasoning_config: Optional[ReasoningConfig] = None
	explanation_level: ExplanationLevel = ExplanationLevel.STANDARD
	max_hops: int = Field(default=3, ge=1, le=10)


class BatchQueryRequest(BaseGraphRAGModel, TenantMixin):
	"""Request for batch GraphRAG query processing"""
	knowledge_graph_id: str
	queries: List[str] = Field(min_items=1, max_items=100)
	query_type: QueryType = QueryType.QUESTION_ANSWERING
	shared_context: Optional[QueryContext] = None
	shared_config: Optional[RetrievalConfig] = None


class GraphExplorationRequest(BaseGraphRAGModel, TenantMixin):
	"""Request for interactive graph exploration"""
	knowledge_graph_id: str
	start_entities: List[str] = Field(min_items=1)
	exploration_depth: int = Field(default=2, ge=1, le=5)
	relationship_filters: List[RelationshipType] = Field(default_factory=list)
	include_properties: bool = True


class GraphAnalyticsRequest(BaseGraphRAGModel, TenantMixin):
	"""Request for graph analytics"""
	knowledge_graph_id: str
	analytics_type: str  # centrality, communities, paths, statistics
	parameters: Dict[str, Any] = Field(default_factory=dict)
	time_range: Optional[TemporalRange] = None


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class APIResponse(BaseGraphRAGModel):
	"""Base API response model"""
	success: bool
	message: str
	timestamp: datetime = Field(default_factory=datetime.utcnow)


class GraphRAGQueryResponse(APIResponse):
	"""Response for GraphRAG queries"""
	query_id: str
	response: GraphRAGResponse
	performance_metrics: PerformanceMetrics


class BatchQueryResponse(APIResponse):
	"""Response for batch queries"""
	batch_id: str = Field(default_factory=uuid7str)
	results: List[GraphRAGQueryResponse]
	total_queries: int
	successful_queries: int
	failed_queries: int
	batch_performance: PerformanceMetrics


class GraphStatisticsResponse(APIResponse):
	"""Response with graph statistics"""
	knowledge_graph_id: str
	statistics: Dict[str, Any]
	quality_metrics: GraphQualityMetrics
	last_updated: datetime


class ErrorResponse(APIResponse):
	"""Error response model"""
	error_code: str
	error_details: Optional[Dict[str, Any]] = None
	suggestion: Optional[str] = None


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_confidence_score(v: float) -> float:
	"""Validate confidence scores are between 0.0 and 1.0"""
	if not 0.0 <= v <= 1.0:
		raise ValueError('Confidence score must be between 0.0 and 1.0')
	return v


def validate_entity_id(v: str) -> str:
	"""Validate entity ID format"""
	if not v.strip():
		raise ValueError('Entity ID cannot be empty')
	return v.strip()


def validate_non_empty_string(v: str) -> str:
	"""Validate string is not empty after stripping"""
	if not v.strip():
		raise ValueError('Value cannot be empty')
	return v.strip()


# Apply validators
ConfidenceScore = Annotated[float, AfterValidator(validate_confidence_score)]
EntityId = Annotated[str, AfterValidator(validate_entity_id)]
NonEmptyString = Annotated[str, AfterValidator(validate_non_empty_string)]


# ============================================================================
# MODEL REGISTRY
# ============================================================================

GRAPHRAG_MODELS = {
	# Core Models
	'KnowledgeGraph': KnowledgeGraph,
	'GraphEntity': GraphEntity,
	'GraphRelationship': GraphRelationship,
	'GraphCommunity': GraphCommunity,
	
	# Query Models
	'GraphRAGQuery': GraphRAGQuery,
	'GraphRAGResponse': GraphRAGResponse,
	'ReasoningChain': ReasoningChain,
	
	# Curation Models
	'CurationWorkflow': CurationWorkflow,
	'KnowledgeEdit': KnowledgeEdit,
	
	# Analytics Models
	'GraphAnalytics': GraphAnalytics,
	'PerformanceMetrics': PerformanceMetrics,
	
	# Request/Response Models
	'GraphQueryRequest': GraphQueryRequest,
	'GraphRAGQueryResponse': GraphRAGQueryResponse,
	'BatchQueryResponse': BatchQueryResponse,
	'ErrorResponse': ErrorResponse,
}


__all__ = [
	# Enums
	'EntityType', 'RelationshipType', 'QueryType', 'ExplanationLevel',
	'GraphStatus', 'ProcessingStatus',
	
	# Core Models
	'KnowledgeGraph', 'GraphEntity', 'GraphRelationship', 'GraphCommunity',
	
	# Query Models
	'GraphRAGQuery', 'GraphRAGResponse', 'ReasoningChain', 'Evidence',
	'GraphPath', 'EntityMention', 'SourceAttribution',
	
	# Context Models
	'QueryContext', 'RetrievalConfig', 'ReasoningConfig', 'TemporalRange',
	
	# Curation Models
	'CurationWorkflow', 'KnowledgeEdit', 'ExpertParticipant',
	
	# Analytics Models
	'GraphAnalytics', 'PerformanceMetrics', 'QualityIndicators',
	
	# Request Models
	'CreateKnowledgeGraphRequest', 'UpdateKnowledgeGraphRequest',
	'GraphQueryRequest', 'BatchQueryRequest', 'GraphExplorationRequest',
	
	# Response Models
	'GraphRAGQueryResponse', 'BatchQueryResponse', 'GraphStatisticsResponse',
	'ErrorResponse', 'APIResponse',
	
	# Validators
	'validate_confidence_score', 'validate_entity_id', 'validate_non_empty_string',
	
	# Registry
	'GRAPHRAG_MODELS',
]
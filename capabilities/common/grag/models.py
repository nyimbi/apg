"""
APG GraphRAG Capability - SQLAlchemy Database Models

Revolutionary graph-based retrieval-augmented generation with Apache AGE integration.
Complete database models for knowledge graphs, entities, relationships, and analytics.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
"""

from __future__ import annotations
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from sqlalchemy import (
	Column, String, Text, Integer, BigInteger, Boolean, DateTime, 
	Numeric, JSON, ARRAY, ForeignKey, UniqueConstraint, CheckConstraint,
	Index, func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import expression
from uuid_extensions import uuid7str
import uuid


# Base model with common functionality
Base = declarative_base()


class TimestampMixin:
	"""Mixin for automatic timestamp management"""
	created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
	updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)


class TenantMixin:
	"""Mixin for multi-tenant support"""
	tenant_id = Column(String(255), nullable=False, index=True)


# ============================================================================
# CORE GRAPHRAG MODELS
# ============================================================================

class GrKnowledgeGraph(Base, TenantMixin, TimestampMixin):
	"""Knowledge Graphs - Top-level knowledge graph containers"""
	__tablename__ = 'gr_knowledge_graphs'
	
	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	name = Column(String(500), nullable=False)
	description = Column(Text)
	schema_version = Column(String(20), default='1.0.0', nullable=False)
	graph_type = Column(String(100), default='knowledge_graph', nullable=False)
	metadata = Column(JSONB, default=dict, nullable=False)
	quality_metrics = Column(JSONB, default=dict, nullable=False)
	status = Column(String(50), default='active', nullable=False)
	
	# Relationships
	entities = relationship("GrGraphEntity", back_populates="knowledge_graph", cascade="all, delete-orphan")
	relationships = relationship("GrGraphRelationship", back_populates="knowledge_graph", cascade="all, delete-orphan")
	communities = relationship("GrGraphCommunity", back_populates="knowledge_graph", cascade="all, delete-orphan")
	queries = relationship("GrQuery", back_populates="knowledge_graph")
	workflows = relationship("GrCurationWorkflow", back_populates="knowledge_graph")
	analytics = relationship("GrAnalytics", back_populates="knowledge_graph")
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('tenant_id', 'name', name='uq_gr_kg_tenant_name'),
		CheckConstraint("status IN ('active', 'inactive', 'building', 'error')", name='ck_gr_kg_status'),
		Index('idx_gr_kg_tenant', 'tenant_id'),
		Index('idx_gr_kg_status', 'status'),
		Index('idx_gr_kg_updated', 'updated_at'),
	)
	
	@validates('name')
	def _validate_name(self, key, value):
		if not value or not value.strip():
			raise ValueError("Knowledge graph name cannot be empty")
		return value.strip()
	
	def __repr__(self):
		return f"<GrKnowledgeGraph(id={self.id}, name='{self.name}', tenant='{self.tenant_id}')>"


class GrGraphEntity(Base, TenantMixin, TimestampMixin):
	"""Graph Entities - Nodes in the knowledge graph"""
	__tablename__ = 'gr_graph_entities'
	
	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	knowledge_graph_id = Column(UUID(as_uuid=True), ForeignKey('gr_knowledge_graphs.id', ondelete='CASCADE'), nullable=False)
	entity_id = Column(String(500), nullable=False)  # Canonical entity identifier
	entity_type = Column(String(100), nullable=False)
	canonical_name = Column(String(1000), nullable=False)
	aliases = Column(ARRAY(Text), default=list, nullable=False)
	properties = Column(JSONB, default=dict, nullable=False)
	embeddings = Column(ARRAY(Numeric), nullable=True)  # bge-m3 embeddings (1024 dims)
	confidence_score = Column(Numeric(5, 4), default=0.0000, nullable=False)
	evidence_sources = Column(ARRAY(Text), default=list, nullable=False)
	provenance = Column(JSONB, default=dict, nullable=False)
	quality_score = Column(Numeric(5, 4), default=0.0000, nullable=False)
	status = Column(String(50), default='active', nullable=False)
	
	# Relationships
	knowledge_graph = relationship("GrKnowledgeGraph", back_populates="entities")
	source_relationships = relationship("GrGraphRelationship", foreign_keys="GrGraphRelationship.source_entity_id", back_populates="source_entity")
	target_relationships = relationship("GrGraphRelationship", foreign_keys="GrGraphRelationship.target_entity_id", back_populates="target_entity")
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('knowledge_graph_id', 'entity_id', name='uq_gr_entities_kg_entity'),
		CheckConstraint('confidence_score >= 0.0 AND confidence_score <= 1.0', name='ck_gr_entities_confidence'),
		CheckConstraint('quality_score >= 0.0 AND quality_score <= 1.0', name='ck_gr_entities_quality'),
		CheckConstraint("status IN ('active', 'inactive', 'pending', 'merged')", name='ck_gr_entities_status'),
		Index('idx_gr_entities_kg', 'knowledge_graph_id'),
		Index('idx_gr_entities_tenant', 'tenant_id'),
		Index('idx_gr_entities_type', 'entity_type'),
		Index('idx_gr_entities_name', 'canonical_name'),
		Index('idx_gr_entities_confidence', 'confidence_score'),
		Index('idx_gr_entities_status', 'status'),
	)
	
	@validates('canonical_name')
	def _validate_canonical_name(self, key, value):
		if not value or not value.strip():
			raise ValueError("Entity canonical name cannot be empty")
		return value.strip()
	
	@validates('embeddings')
	def _validate_embeddings(self, key, value):
		if value is not None and len(value) != 1024:
			raise ValueError("Embeddings must be 1024 dimensions for bge-m3")
		return value
	
	def __repr__(self):
		return f"<GrGraphEntity(id={self.id}, name='{self.canonical_name}', type='{self.entity_type}')>"


class GrGraphRelationship(Base, TenantMixin, TimestampMixin):
	"""Graph Relationships - Edges in the knowledge graph"""
	__tablename__ = 'gr_graph_relationships'
	
	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	knowledge_graph_id = Column(UUID(as_uuid=True), ForeignKey('gr_knowledge_graphs.id', ondelete='CASCADE'), nullable=False)
	relationship_id = Column(String(500), nullable=False)  # Canonical relationship identifier
	source_entity_id = Column(UUID(as_uuid=True), ForeignKey('gr_graph_entities.id', ondelete='CASCADE'), nullable=False)
	target_entity_id = Column(UUID(as_uuid=True), ForeignKey('gr_graph_entities.id', ondelete='CASCADE'), nullable=False)
	relationship_type = Column(String(100), nullable=False)
	strength = Column(Numeric(5, 4), default=0.0000, nullable=False)
	context = Column(JSONB, default=dict, nullable=False)
	properties = Column(JSONB, default=dict, nullable=False)
	evidence_sources = Column(ARRAY(Text), default=list, nullable=False)
	provenance = Column(JSONB, default=dict, nullable=False)
	temporal_validity = Column(JSONB, nullable=True)  # Start/end timestamps
	confidence_score = Column(Numeric(5, 4), default=0.0000, nullable=False)
	quality_score = Column(Numeric(5, 4), default=0.0000, nullable=False)
	status = Column(String(50), default='active', nullable=False)
	
	# Relationships
	knowledge_graph = relationship("GrKnowledgeGraph", back_populates="relationships")
	source_entity = relationship("GrGraphEntity", foreign_keys=[source_entity_id], back_populates="source_relationships")
	target_entity = relationship("GrGraphEntity", foreign_keys=[target_entity_id], back_populates="target_relationships")
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('knowledge_graph_id', 'relationship_id', name='uq_gr_rel_kg_rel'),
		CheckConstraint('strength >= 0.0 AND strength <= 1.0', name='ck_gr_rel_strength'),
		CheckConstraint('confidence_score >= 0.0 AND confidence_score <= 1.0', name='ck_gr_rel_confidence'),
		CheckConstraint('quality_score >= 0.0 AND quality_score <= 1.0', name='ck_gr_rel_quality'),
		CheckConstraint("status IN ('active', 'inactive', 'pending', 'deprecated')", name='ck_gr_rel_status'),
		CheckConstraint('source_entity_id != target_entity_id', name='ck_gr_rel_no_self_loop'),
		Index('idx_gr_rel_kg', 'knowledge_graph_id'),
		Index('idx_gr_rel_tenant', 'tenant_id'),
		Index('idx_gr_rel_source', 'source_entity_id'),
		Index('idx_gr_rel_target', 'target_entity_id'),
		Index('idx_gr_rel_type', 'relationship_type'),
		Index('idx_gr_rel_strength', 'strength'),
		Index('idx_gr_rel_status', 'status'),
		Index('idx_gr_rel_traversal', 'source_entity_id', 'relationship_type', 'status'),
	)
	
	@validates('relationship_type')
	def _validate_relationship_type(self, key, value):
		if not value or not value.strip():
			raise ValueError("Relationship type cannot be empty")
		return value.strip()
	
	def __repr__(self):
		return f"<GrGraphRelationship(id={self.id}, type='{self.relationship_type}', strength={self.strength})>"


class GrGraphCommunity(Base, TenantMixin, TimestampMixin):
	"""Graph Communities - Community detection results"""
	__tablename__ = 'gr_graph_communities'
	
	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	knowledge_graph_id = Column(UUID(as_uuid=True), ForeignKey('gr_knowledge_graphs.id', ondelete='CASCADE'), nullable=False)
	community_id = Column(String(500), nullable=False)
	name = Column(String(500), nullable=True)
	description = Column(Text, nullable=True)
	algorithm = Column(String(100), nullable=False)  # louvain, leiden, etc.
	members = Column(JSONB, nullable=False)  # Array of entity IDs
	centrality_metrics = Column(JSONB, default=dict, nullable=False)
	cohesion_score = Column(Numeric(5, 4), nullable=True)
	size_metrics = Column(JSONB, default=dict, nullable=False)
	
	# Relationships
	knowledge_graph = relationship("GrKnowledgeGraph", back_populates="communities")
	
	# Constraints
	__table_args__ = (
		UniqueConstraint('knowledge_graph_id', 'community_id', name='uq_gr_comm_kg_comm'),
		CheckConstraint('cohesion_score IS NULL OR (cohesion_score >= 0.0 AND cohesion_score <= 1.0)', name='ck_gr_comm_cohesion'),
		Index('idx_gr_comm_kg', 'knowledge_graph_id'),
		Index('idx_gr_comm_tenant', 'tenant_id'),
		Index('idx_gr_comm_algorithm', 'algorithm'),
	)
	
	@validates('algorithm')
	def _validate_algorithm(self, key, value):
		if not value or not value.strip():
			raise ValueError("Algorithm cannot be empty")
		return value.strip()
	
	def __repr__(self):
		return f"<GrGraphCommunity(id={self.id}, name='{self.name}', algorithm='{self.algorithm}')>"


# ============================================================================
# QUERY AND RESPONSE MODELS
# ============================================================================

class GrQuery(Base, TenantMixin):
	"""GraphRAG Queries - Query processing and caching"""
	__tablename__ = 'gr_queries'
	
	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	knowledge_graph_id = Column(UUID(as_uuid=True), ForeignKey('gr_knowledge_graphs.id', ondelete='CASCADE'), nullable=False)
	query_text = Column(Text, nullable=False)
	query_type = Column(String(100), default='question_answering', nullable=False)
	query_embedding = Column(ARRAY(Numeric), nullable=True)  # bge-m3 query embedding
	context = Column(JSONB, default=dict, nullable=False)
	retrieval_config = Column(JSONB, default=dict, nullable=False)
	reasoning_config = Column(JSONB, default=dict, nullable=False)
	explanation_level = Column(String(50), default='standard', nullable=False)
	max_hops = Column(Integer, default=3, nullable=False)
	status = Column(String(50), default='pending', nullable=False)
	processing_time_ms = Column(BigInteger, nullable=True)
	created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
	completed_at = Column(DateTime(timezone=True), nullable=True)
	
	# Relationships
	knowledge_graph = relationship("GrKnowledgeGraph", back_populates="queries")
	responses = relationship("GrResponse", back_populates="query", cascade="all, delete-orphan")
	performance_logs = relationship("GrQueryPerformance", back_populates="query", cascade="all, delete-orphan")
	
	# Constraints
	__table_args__ = (
		CheckConstraint('max_hops >= 1 AND max_hops <= 10', name='ck_gr_queries_max_hops'),
		CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed', 'cached')", name='ck_gr_queries_status'),
		CheckConstraint("explanation_level IN ('minimal', 'standard', 'detailed', 'comprehensive')", name='ck_gr_queries_explanation'),
		Index('idx_gr_queries_kg', 'knowledge_graph_id'),
		Index('idx_gr_queries_tenant', 'tenant_id'),
		Index('idx_gr_queries_type', 'query_type'),
		Index('idx_gr_queries_status', 'status'),
		Index('idx_gr_queries_created', 'created_at'),
	)
	
	@validates('query_text')
	def _validate_query_text(self, key, value):
		if not value or not value.strip():
			raise ValueError("Query text cannot be empty")
		return value.strip()
	
	@validates('query_embedding')
	def _validate_query_embedding(self, key, value):
		if value is not None and len(value) != 1024:
			raise ValueError("Query embeddings must be 1024 dimensions for bge-m3")
		return value
	
	def __repr__(self):
		return f"<GrQuery(id={self.id}, type='{self.query_type}', status='{self.status}')>"


class GrResponse(Base, TenantMixin):
	"""GraphRAG Responses - Generated responses with reasoning"""
	__tablename__ = 'gr_responses'
	
	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	query_id = Column(UUID(as_uuid=True), ForeignKey('gr_queries.id', ondelete='CASCADE'), nullable=False)
	answer = Column(Text, nullable=False)
	confidence_score = Column(Numeric(5, 4), nullable=False)
	reasoning_chain = Column(JSONB, nullable=False)  # Complete reasoning process
	supporting_evidence = Column(JSONB, default=list, nullable=False)
	graph_paths = Column(JSONB, default=list, nullable=False)  # Paths through the graph
	entity_mentions = Column(JSONB, default=list, nullable=False)
	source_attribution = Column(JSONB, default=list, nullable=False)
	quality_indicators = Column(JSONB, default=dict, nullable=False)
	processing_metrics = Column(JSONB, default=dict, nullable=False)
	model_used = Column(String(200), nullable=True)
	created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
	
	# Relationships
	query = relationship("GrQuery", back_populates="responses")
	
	# Constraints
	__table_args__ = (
		CheckConstraint('confidence_score >= 0.0 AND confidence_score <= 1.0', name='ck_gr_resp_confidence'),
		Index('idx_gr_resp_query', 'query_id'),
		Index('idx_gr_resp_tenant', 'tenant_id'),
		Index('idx_gr_resp_confidence', 'confidence_score'),
		Index('idx_gr_resp_created', 'created_at'),
	)
	
	@validates('answer')
	def _validate_answer(self, key, value):
		if not value or not value.strip():
			raise ValueError("Answer cannot be empty")
		return value.strip()
	
	def __repr__(self):
		return f"<GrResponse(id={self.id}, confidence={self.confidence_score})>"


# ============================================================================
# KNOWLEDGE CURATION MODELS
# ============================================================================

class GrCurationWorkflow(Base, TenantMixin, TimestampMixin):
	"""Curation Workflows - Collaborative knowledge improvement"""
	__tablename__ = 'gr_curation_workflows'
	
	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	knowledge_graph_id = Column(UUID(as_uuid=True), ForeignKey('gr_knowledge_graphs.id', ondelete='CASCADE'), nullable=False)
	name = Column(String(500), nullable=False)
	description = Column(Text, nullable=True)
	workflow_type = Column(String(100), nullable=False)
	participants = Column(JSONB, nullable=False)  # Expert users and roles
	consensus_threshold = Column(Numeric(3, 2), default=0.80, nullable=False)
	status = Column(String(50), default='active', nullable=False)
	metrics = Column(JSONB, default=dict, nullable=False)
	
	# Relationships
	knowledge_graph = relationship("GrKnowledgeGraph", back_populates="workflows")
	edits = relationship("GrKnowledgeEdit", back_populates="workflow", cascade="all, delete-orphan")
	
	# Constraints
	__table_args__ = (
		CheckConstraint('consensus_threshold >= 0.50 AND consensus_threshold <= 1.00', name='ck_gr_curation_consensus'),
		CheckConstraint("status IN ('active', 'paused', 'completed', 'archived')", name='ck_gr_curation_status'),
		Index('idx_gr_curation_kg', 'knowledge_graph_id'),
		Index('idx_gr_curation_tenant', 'tenant_id'),
		Index('idx_gr_curation_status', 'status'),
	)
	
	@validates('name')
	def _validate_name(self, key, value):
		if not value or not value.strip():
			raise ValueError("Workflow name cannot be empty")
		return value.strip()
	
	def __repr__(self):
		return f"<GrCurationWorkflow(id={self.id}, name='{self.name}', status='{self.status}')>"


class GrKnowledgeEdit(Base, TenantMixin):
	"""Knowledge Edits - Proposed changes to knowledge"""
	__tablename__ = 'gr_knowledge_edits'
	
	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	workflow_id = Column(UUID(as_uuid=True), ForeignKey('gr_curation_workflows.id', ondelete='CASCADE'), nullable=False)
	knowledge_graph_id = Column(UUID(as_uuid=True), ForeignKey('gr_knowledge_graphs.id', ondelete='CASCADE'), nullable=False)
	editor_id = Column(String(255), nullable=False)  # User making the edit
	edit_type = Column(String(100), nullable=False)
	target_type = Column(String(100), nullable=False)  # entity, relationship, graph
	target_id = Column(UUID(as_uuid=True), nullable=False)
	proposed_changes = Column(JSONB, nullable=False)
	justification = Column(Text, nullable=True)
	evidence = Column(JSONB, default=list, nullable=False)
	status = Column(String(50), default='pending', nullable=False)
	reviews = Column(JSONB, default=list, nullable=False)
	consensus_score = Column(Numeric(5, 4), nullable=True)
	applied_at = Column(DateTime(timezone=True), nullable=True)
	created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
	
	# Relationships
	workflow = relationship("GrCurationWorkflow", back_populates="edits")
	knowledge_graph = relationship("GrKnowledgeGraph")
	
	# Constraints
	__table_args__ = (
		CheckConstraint("edit_type IN ('create', 'update', 'delete', 'merge', 'split')", name='ck_gr_edits_type'),
		CheckConstraint("target_type IN ('entity', 'relationship', 'community', 'graph')", name='ck_gr_edits_target'),
		CheckConstraint("status IN ('pending', 'reviewing', 'approved', 'rejected', 'applied')", name='ck_gr_edits_status'),
		CheckConstraint('consensus_score IS NULL OR (consensus_score >= 0.0 AND consensus_score <= 1.0)', name='ck_gr_edits_consensus'),
		Index('idx_gr_edits_workflow', 'workflow_id'),
		Index('idx_gr_edits_kg', 'knowledge_graph_id'),
		Index('idx_gr_edits_editor', 'editor_id'),
		Index('idx_gr_edits_status', 'status'),
		Index('idx_gr_edits_created', 'created_at'),
	)
	
	@validates('editor_id')
	def _validate_editor_id(self, key, value):
		if not value or not value.strip():
			raise ValueError("Editor ID cannot be empty")
		return value.strip()
	
	def __repr__(self):
		return f"<GrKnowledgeEdit(id={self.id}, type='{self.edit_type}', status='{self.status}')>"


# ============================================================================
# ANALYTICS AND PERFORMANCE MODELS
# ============================================================================

class GrAnalytics(Base, TenantMixin):
	"""Graph Analytics - Performance and usage metrics"""
	__tablename__ = 'gr_analytics'
	
	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	knowledge_graph_id = Column(UUID(as_uuid=True), ForeignKey('gr_knowledge_graphs.id', ondelete='CASCADE'), nullable=False)
	metric_type = Column(String(100), nullable=False)
	metric_name = Column(String(200), nullable=False)
	metric_value = Column(Numeric(10, 4), nullable=True)
	metric_data = Column(JSONB, default=dict, nullable=False)
	time_period = Column(String(50), nullable=True)  # hour, day, week, month
	created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
	
	# Relationships
	knowledge_graph = relationship("GrKnowledgeGraph", back_populates="analytics")
	
	# Constraints
	__table_args__ = (
		CheckConstraint("metric_type IN ('performance', 'usage', 'quality', 'accuracy', 'efficiency')", name='ck_gr_analytics_type'),
		Index('idx_gr_analytics_kg', 'knowledge_graph_id'),
		Index('idx_gr_analytics_tenant', 'tenant_id'),
		Index('idx_gr_analytics_type', 'metric_type'),
		Index('idx_gr_analytics_created', 'created_at'),
	)
	
	@validates('metric_name')
	def _validate_metric_name(self, key, value):
		if not value or not value.strip():
			raise ValueError("Metric name cannot be empty")
		return value.strip()
	
	def __repr__(self):
		return f"<GrAnalytics(id={self.id}, metric='{self.metric_name}', value={self.metric_value})>"


class GrQueryPerformance(Base, TenantMixin):
	"""Query Performance Logs - Detailed query performance tracking"""
	__tablename__ = 'gr_query_performance'
	
	id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	query_id = Column(UUID(as_uuid=True), ForeignKey('gr_queries.id', ondelete='CASCADE'), nullable=False)
	retrieval_time_ms = Column(BigInteger, nullable=True)
	reasoning_time_ms = Column(BigInteger, nullable=True)
	generation_time_ms = Column(BigInteger, nullable=True)
	total_time_ms = Column(BigInteger, nullable=True)
	entities_retrieved = Column(Integer, nullable=True)
	relationships_traversed = Column(Integer, nullable=True)
	graph_hops = Column(Integer, nullable=True)
	memory_usage_mb = Column(Integer, nullable=True)
	cache_hits = Column(Integer, nullable=True)
	cache_misses = Column(Integer, nullable=True)
	model_tokens = Column(Integer, nullable=True)
	created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
	
	# Relationships
	query = relationship("GrQuery", back_populates="performance_logs")
	
	# Constraints
	__table_args__ = (
		Index('idx_gr_perf_query', 'query_id'),
		Index('idx_gr_perf_tenant', 'tenant_id'),
		Index('idx_gr_perf_total_time', 'total_time_ms'),
		Index('idx_gr_perf_created', 'created_at'),
	)
	
	def __repr__(self):
		return f"<GrQueryPerformance(id={self.id}, total_time={self.total_time_ms}ms)>"


# ============================================================================
# MODEL REGISTRY AND UTILITIES
# ============================================================================

# All GraphRAG models for easy import and registration
GRAPHRAG_MODELS = {
	'GrKnowledgeGraph': GrKnowledgeGraph,
	'GrGraphEntity': GrGraphEntity,
	'GrGraphRelationship': GrGraphRelationship,
	'GrGraphCommunity': GrGraphCommunity,
	'GrQuery': GrQuery,
	'GrResponse': GrResponse,
	'GrCurationWorkflow': GrCurationWorkflow,
	'GrKnowledgeEdit': GrKnowledgeEdit,
	'GrAnalytics': GrAnalytics,
	'GrQueryPerformance': GrQueryPerformance,
}


def create_all_tables(engine):
	"""Create all GraphRAG tables in the database"""
	Base.metadata.create_all(engine)


def drop_all_tables(engine):
	"""Drop all GraphRAG tables from the database"""
	Base.metadata.drop_all(engine)


def get_table_names():
	"""Get all GraphRAG table names"""
	return [table.name for table in Base.metadata.tables.values()]


def get_model_by_table_name(table_name: str):
	"""Get model class by table name"""
	for model_class in GRAPHRAG_MODELS.values():
		if hasattr(model_class, '__tablename__') and model_class.__tablename__ == table_name:
			return model_class
	return None


# Export all models and utilities
__all__ = [
	# Core Models
	'GrKnowledgeGraph', 'GrGraphEntity', 'GrGraphRelationship', 'GrGraphCommunity',
	
	# Query Models
	'GrQuery', 'GrResponse',
	
	# Curation Models
	'GrCurationWorkflow', 'GrKnowledgeEdit',
	
	# Analytics Models
	'GrAnalytics', 'GrQueryPerformance',
	
	# Mixins
	'TimestampMixin', 'TenantMixin',
	
	# Base and Registry
	'Base', 'GRAPHRAG_MODELS',
	
	# Utilities
	'create_all_tables', 'drop_all_tables', 'get_table_names', 'get_model_by_table_name',
]
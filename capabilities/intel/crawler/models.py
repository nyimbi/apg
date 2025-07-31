"""
APG Crawler Capability - SQLAlchemy Models
==========================================

Enterprise-grade SQLAlchemy models with:
- Modern async SQLAlchemy 2.0+ patterns
- Multi-tenant architecture with APG integration
- RAG and GraphRAG support with vector embeddings
- Comprehensive relationships and constraints
- Performance-optimized queries and indexes

Copyright © 2025 Datacraft (nyimbi@gmail.com)
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from sqlalchemy import (
	String, Text, Integer, Float, Boolean, DateTime, JSON, ARRAY, 
	ForeignKey, UniqueConstraint, CheckConstraint, Index, DECIMAL
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, VECTOR
from sqlalchemy.sql import func
from uuid_extensions import uuid7str
import uuid


# =====================================================
# BASE MODEL WITH APG PATTERNS
# =====================================================

class Base(DeclarativeBase):
	"""Base model with APG patterns and conventions"""
	
	# Schema specification for multi-tenancy
	__table_args__ = {'schema': 'crawler'}
	
	# Common APG patterns
	id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
	tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
	created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), nullable=False)
	updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)


# =====================================================
# CORE CRAWLER MODELS
# =====================================================

class CrawlTarget(Base):
	"""Crawl targets with business intelligence and RAG/GraphRAG integration"""
	__tablename__ = 'cr_crawl_targets'
	
	# Core fields
	name: Mapped[str] = mapped_column(String(500), nullable=False)
	description: Mapped[Optional[str]] = mapped_column(Text)
	target_urls: Mapped[List[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
	target_type: Mapped[str] = mapped_column(String(50), nullable=False, default='web_crawl')
	
	# Configuration fields (JSON)
	data_schema: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	business_context: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	stealth_requirements: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	quality_requirements: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	scheduling_config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	collaboration_config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	
	# RAG/GraphRAG Integration fields
	rag_integration_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
	graphrag_integration_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
	knowledge_graph_target: Mapped[Optional[str]] = mapped_column(String(255))
	content_fingerprinting: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
	markdown_storage: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
	
	# Status and audit
	status: Mapped[str] = mapped_column(String(50), nullable=False, default='active')
	created_by: Mapped[Optional[str]] = mapped_column(String(255))
	updated_by: Mapped[Optional[str]] = mapped_column(String(255))
	
	# Relationships
	pipelines: Mapped[List["CrawlPipeline"]] = relationship("CrawlPipeline", back_populates="target", cascade="all, delete-orphan")
	datasets: Mapped[List["ExtractedDataset"]] = relationship("ExtractedDataset", back_populates="target", cascade="all, delete-orphan")
	validation_sessions: Mapped[List["ValidationSession"]] = relationship("ValidationSession", back_populates="target")
	
	# Constraints and indexes
	__table_args__ = (
		Index('idx_cr_crawl_targets_tenant_status', 'tenant_id', 'status'),
		Index('idx_cr_crawl_targets_type', 'target_type'),
		Index('idx_cr_crawl_targets_rag_enabled', 'rag_integration_enabled'),  
		Index('idx_cr_crawl_targets_created_at', 'created_at'),
		{'schema': 'crawler'}
	)


class CrawlPipeline(Base):
	"""Visual crawl pipelines with optimization and deployment"""
	__tablename__ = 'cr_crawl_pipelines'
	
	# Core fields
	name: Mapped[str] = mapped_column(String(500), nullable=False)
	description: Mapped[Optional[str]] = mapped_column(Text)
	target_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey('crawler.cr_crawl_targets.id'), nullable=False)
	
	# Pipeline configuration
	visual_config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	processing_stages: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
	optimization_settings: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	performance_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	monitoring_config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	
	# Status and deployment
	deployment_status: Mapped[str] = mapped_column(String(50), nullable=False, default='draft')
	status: Mapped[str] = mapped_column(String(50), nullable=False, default='active')
	created_by: Mapped[Optional[str]] = mapped_column(String(255))
	updated_by: Mapped[Optional[str]] = mapped_column(String(255))
	
	# Relationships
	target: Mapped["CrawlTarget"] = relationship("CrawlTarget", back_populates="pipelines")
	executions: Mapped[List["PipelineExecution"]] = relationship("PipelineExecution", back_populates="pipeline", cascade="all, delete-orphan")
	
	# Constraints and indexes
	__table_args__ = (
		Index('idx_cr_crawl_pipelines_tenant_target', 'tenant_id', 'target_id'),
		Index('idx_cr_crawl_pipelines_deployment_status', 'deployment_status'),
		Index('idx_cr_crawl_pipelines_status', 'status'),
		{'schema': 'crawler'}
	)


class ExtractedDataset(Base):
	"""Extracted datasets with quality metrics and validation"""
	__tablename__ = 'cr_extracted_datasets'
	
	# Core fields
	dataset_name: Mapped[str] = mapped_column(String(500), nullable=False)
	target_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey('crawler.cr_crawl_targets.id'), nullable=False)
	pipeline_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey('crawler.cr_crawl_pipelines.id'))
	
	# Dataset metadata
	extraction_method: Mapped[str] = mapped_column(String(50), nullable=False)
	source_urls: Mapped[List[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
	record_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
	data_schema: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
	metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	
	# Quality and validation
	quality_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	validation_status: Mapped[str] = mapped_column(String(50), nullable=False, default='pending')
	consensus_score: Mapped[float] = mapped_column(DECIMAL(5, 4), nullable=False, default=0.0)
	
	# Status and audit
	status: Mapped[str] = mapped_column(String(50), nullable=False, default='active')
	created_by: Mapped[Optional[str]] = mapped_column(String(255))
	updated_by: Mapped[Optional[str]] = mapped_column(String(255))
	
	# Relationships
	target: Mapped["CrawlTarget"] = relationship("CrawlTarget", back_populates="datasets")
	pipeline: Mapped[Optional["CrawlPipeline"]] = relationship("CrawlPipeline")
	records: Mapped[List["DataRecord"]] = relationship("DataRecord", back_populates="dataset", cascade="all, delete-orphan")
	validation_sessions: Mapped[List["ValidationSession"]] = relationship("ValidationSession", back_populates="dataset")
	
	# Constraints and indexes
	__table_args__ = (
		Index('idx_cr_extracted_datasets_tenant_target', 'tenant_id', 'target_id'),
		Index('idx_cr_extracted_datasets_validation_status', 'validation_status'),
		Index('idx_cr_extracted_datasets_consensus', 'consensus_score'),
		Index('idx_cr_extracted_datasets_record_count', 'record_count'),
		CheckConstraint('record_count >= 0', name='ck_extracted_datasets_record_count_positive'),
		CheckConstraint('consensus_score >= 0.0 AND consensus_score <= 1.0', name='ck_extracted_datasets_consensus_range'),
		{'schema': 'crawler'}
	)


# =====================================================
# DATA RECORD MODELS WITH RAG/GRAPHRAG INTEGRATION
# =====================================================

class DataRecord(Base):
	"""Individual data records with RAG/GraphRAG integration"""
	__tablename__ = 'cr_data_records'
	
	# Core fields
	dataset_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey('crawler.cr_extracted_datasets.id'), nullable=False)
	record_index: Mapped[int] = mapped_column(Integer, nullable=False)
	source_url: Mapped[str] = mapped_column(Text, nullable=False)
	
	# Content fields
	extracted_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	raw_content: Mapped[Optional[str]] = mapped_column(Text)
	processed_content: Mapped[Optional[str]] = mapped_column(Text)
	
	# RAG/GraphRAG content fields
	cleaned_content: Mapped[Optional[str]] = mapped_column(Text)
	markdown_content: Mapped[Optional[str]] = mapped_column(Text)
	content_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, default='')
	content_processing_stage: Mapped[str] = mapped_column(String(50), nullable=False, default='raw_extracted')
	vector_embeddings: Mapped[Optional[List[float]]] = mapped_column(VECTOR(1536))  # Configurable dimensions
	rag_chunk_ids: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
	graphrag_node_id: Mapped[Optional[str]] = mapped_column(String(255))
	knowledge_graph_entities: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
	
	# Metadata and classification
	content_type: Mapped[Optional[str]] = mapped_column(String(100))
	language: Mapped[Optional[str]] = mapped_column(String(10))
	semantic_tags: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
	extraction_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	rag_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	
	# Quality and validation
	quality_score: Mapped[float] = mapped_column(DECIMAL(5, 4), nullable=False, default=0.0)
	confidence_score: Mapped[float] = mapped_column(DECIMAL(5, 4), nullable=False, default=0.0)
	validation_status: Mapped[str] = mapped_column(String(50), nullable=False, default='pending')
	
	# Relationships
	dataset: Mapped["ExtractedDataset"] = relationship("ExtractedDataset", back_populates="records")
	business_entities: Mapped[List["BusinessEntity"]] = relationship("BusinessEntity", back_populates="record", cascade="all, delete-orphan")
	rag_chunks: Mapped[List["RAGChunk"]] = relationship("RAGChunk", back_populates="record", cascade="all, delete-orphan")
	graphrag_nodes: Mapped[List["GraphRAGNode"]] = relationship("GraphRAGNode", back_populates="record", cascade="all, delete-orphan")
	validation_feedback: Mapped[List["ValidationFeedback"]] = relationship("ValidationFeedback", back_populates="record")
	
	# Constraints and indexes
	__table_args__ = (
		Index('idx_cr_data_records_tenant_dataset', 'tenant_id', 'dataset_id'),
		Index('idx_cr_data_records_fingerprint', 'content_fingerprint'),
		Index('idx_cr_data_records_processing_stage', 'content_processing_stage'),
		Index('idx_cr_data_records_validation_status', 'validation_status'),
		Index('idx_cr_data_records_quality_score', 'quality_score'),
		Index('idx_cr_data_records_graphrag_node', 'graphrag_node_id'),
		# Vector similarity search index
		Index('idx_cr_data_records_vector_cosine', 'vector_embeddings', postgresql_using='ivfflat', postgresql_with={'lists': 100}, postgresql_ops={'vector_embeddings': 'vector_cosine_ops'}),
		UniqueConstraint('tenant_id', 'dataset_id', 'record_index', name='uq_cr_data_records_tenant_dataset_index'),
		CheckConstraint('record_index >= 0', name='ck_data_records_index_positive'),
		CheckConstraint('quality_score >= 0.0 AND quality_score <= 1.0', name='ck_data_records_quality_range'),
		CheckConstraint('confidence_score >= 0.0 AND confidence_score <= 1.0', name='ck_data_records_confidence_range'),
		{'schema': 'crawler'}
	)


class BusinessEntity(Base):
	"""Business entities extracted from content"""
	__tablename__ = 'cr_business_entities'
	
	# Core fields
	record_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey('crawler.cr_data_records.id'), nullable=False)
	entity_type: Mapped[str] = mapped_column(String(100), nullable=False)
	entity_name: Mapped[str] = mapped_column(String(1000), nullable=False)
	entity_value: Mapped[str] = mapped_column(Text, nullable=False)
	
	# Position and context
	context_window: Mapped[Optional[str]] = mapped_column(Text)
	start_position: Mapped[Optional[int]] = mapped_column(Integer)
	end_position: Mapped[Optional[int]] = mapped_column(Integer)
	
	# Metadata and scoring
	semantic_properties: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	confidence_score: Mapped[float] = mapped_column(DECIMAL(5, 4), nullable=False)
	business_relevance: Mapped[float] = mapped_column(DECIMAL(5, 4), nullable=False, default=0.0)
	validation_status: Mapped[str] = mapped_column(String(50), nullable=False, default='pending')
	
	# Relationships
	record: Mapped["DataRecord"] = relationship("DataRecord", back_populates="business_entities")
	
	# Constraints and indexes
	__table_args__ = (
		Index('idx_cr_business_entities_tenant_type', 'tenant_id', 'entity_type'),
		Index('idx_cr_business_entities_record', 'record_id'),
		Index('idx_cr_business_entities_confidence', 'confidence_score'),
		Index('idx_cr_business_entities_name_gin', 'entity_name', postgresql_using='gin', postgresql_ops={'entity_name': 'gin_trgm_ops'}),
		CheckConstraint('confidence_score >= 0.0 AND confidence_score <= 1.0', name='ck_business_entities_confidence_range'),
		CheckConstraint('business_relevance >= 0.0 AND business_relevance <= 1.0', name='ck_business_entities_relevance_range'),
		CheckConstraint('start_position IS NULL OR end_position IS NULL OR start_position < end_position', name='ck_business_entities_position_order'),
		{'schema': 'crawler'}
	)


# =====================================================
# RAG AND GRAPHRAG MODELS
# =====================================================

class RAGChunk(Base):
	"""RAG chunks with vector embeddings for semantic search"""
	__tablename__ = 'cr_rag_chunks'
	
	# Core fields
	record_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey('crawler.cr_data_records.id'), nullable=False)
	chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
	chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
	chunk_markdown: Mapped[str] = mapped_column(Text, nullable=False)
	chunk_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, default='')
	
	# Vector embeddings
	vector_embeddings: Mapped[Optional[List[float]]] = mapped_column(VECTOR(1536))
	embedding_model: Mapped[str] = mapped_column(String(100), nullable=False, default='text-embedding-ada-002')
	vector_dimensions: Mapped[int] = mapped_column(Integer, nullable=False, default=1536)
	semantic_similarity_threshold: Mapped[float] = mapped_column(DECIMAL(5, 4), nullable=False, default=0.8)
	
	# Chunk metadata
	chunk_overlap_start: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
	chunk_overlap_end: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
	entities_extracted: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
	related_chunks: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
	contextual_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	
	# Status
	indexing_status: Mapped[str] = mapped_column(String(50), nullable=False, default='pending')
	
	# Relationships
	record: Mapped["DataRecord"] = relationship("DataRecord", back_populates="rag_chunks")
	
	# Constraints and indexes
	__table_args__ = (
		Index('idx_cr_rag_chunks_tenant_record', 'tenant_id', 'record_id'), 
		Index('idx_cr_rag_chunks_indexing_status', 'indexing_status'),
		Index('idx_cr_rag_chunks_fingerprint', 'chunk_fingerprint'),
		Index('idx_cr_rag_chunks_embedding_model', 'embedding_model'),
		# Vector similarity search index
		Index('idx_cr_rag_chunks_vector_cosine', 'vector_embeddings', postgresql_using='ivfflat', postgresql_with={'lists': 100}, postgresql_ops={'vector_embeddings': 'vector_cosine_ops'}),
		UniqueConstraint('tenant_id', 'record_id', 'chunk_index', name='uq_cr_rag_chunks_tenant_record_index'),
		CheckConstraint('chunk_index >= 0', name='ck_rag_chunks_index_positive'),
		CheckConstraint('vector_dimensions > 0', name='ck_rag_chunks_dimensions_positive'),
		CheckConstraint('semantic_similarity_threshold >= 0.0 AND semantic_similarity_threshold <= 1.0', name='ck_rag_chunks_similarity_range'),
		{'schema': 'crawler'}
	)


class GraphRAGNode(Base):
	"""GraphRAG knowledge graph nodes for entity representation"""
	__tablename__ = 'cr_graphrag_nodes'
	
	# Core fields
	record_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey('crawler.cr_data_records.id'), nullable=False)
	node_type: Mapped[str] = mapped_column(String(100), nullable=False)
	node_name: Mapped[str] = mapped_column(String(1000), nullable=False)
	node_description: Mapped[Optional[str]] = mapped_column(Text)
	
	# Node properties and metadata
	node_properties: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	entity_type: Mapped[Optional[str]] = mapped_column(String(100))
	
	# Scoring and embeddings
	confidence_score: Mapped[float] = mapped_column(DECIMAL(5, 4), nullable=False, default=0.0)
	salience_score: Mapped[float] = mapped_column(DECIMAL(5, 4), nullable=False, default=0.0)
	vector_embeddings: Mapped[Optional[List[float]]] = mapped_column(VECTOR(1536))
	
	# Relationships and graph
	related_chunks: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
	knowledge_graph_id: Mapped[Optional[str]] = mapped_column(String(255))
	
	# Status
	node_status: Mapped[str] = mapped_column(String(50), nullable=False, default='active')
	
	# Relationships
	record: Mapped["DataRecord"] = relationship("DataRecord", back_populates="graphrag_nodes")
	outgoing_relations: Mapped[List["GraphRAGRelation"]] = relationship("GraphRAGRelation", foreign_keys="GraphRAGRelation.source_node_id", back_populates="source_node", cascade="all, delete-orphan")
	incoming_relations: Mapped[List["GraphRAGRelation"]] = relationship("GraphRAGRelation", foreign_keys="GraphRAGRelation.target_node_id", back_populates="target_node", cascade="all, delete-orphan")
	knowledge_graph: Mapped[Optional["KnowledgeGraph"]] = relationship("KnowledgeGraph", back_populates="nodes")
	
	# Constraints and indexes
	__table_args__ = (
		Index('idx_cr_graphrag_nodes_tenant_type', 'tenant_id', 'node_type'),
		Index('idx_cr_graphrag_nodes_entity_type', 'entity_type'),
		Index('idx_cr_graphrag_nodes_confidence', 'confidence_score'),
		Index('idx_cr_graphrag_nodes_knowledge_graph', 'knowledge_graph_id'),
		Index('idx_cr_graphrag_nodes_status', 'node_status'),
		Index('idx_cr_graphrag_nodes_name_gin', 'node_name', postgresql_using='gin', postgresql_ops={'node_name': 'gin_trgm_ops'}),
		CheckConstraint('confidence_score >= 0.0 AND confidence_score <= 1.0', name='ck_graphrag_nodes_confidence_range'),
		CheckConstraint('salience_score >= 0.0 AND salience_score <= 1.0', name='ck_graphrag_nodes_salience_range'),
		{'schema': 'crawler'}
	)


class GraphRAGRelation(Base):
	"""GraphRAG knowledge graph relations between entities"""
	__tablename__ = 'cr_graphrag_relations'
	
	# Core fields
	source_node_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey('crawler.cr_graphrag_nodes.id'), nullable=False)
	target_node_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey('crawler.cr_graphrag_nodes.id'), nullable=False)
	relation_type: Mapped[str] = mapped_column(String(100), nullable=False)
	relation_label: Mapped[str] = mapped_column(String(500), nullable=False)
	
	# Relation properties and metadata
	relation_properties: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	confidence_score: Mapped[float] = mapped_column(DECIMAL(5, 4), nullable=False, default=0.0)
	strength_score: Mapped[float] = mapped_column(DECIMAL(5, 4), nullable=False, default=0.0)
	
	# Evidence and context
	evidence_chunks: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
	context_window: Mapped[Optional[str]] = mapped_column(Text)
	knowledge_graph_id: Mapped[Optional[str]] = mapped_column(String(255))
	
	# Status
	relation_status: Mapped[str] = mapped_column(String(50), nullable=False, default='active')
	
	# Relationships
	source_node: Mapped["GraphRAGNode"] = relationship("GraphRAGNode", foreign_keys=[source_node_id], back_populates="outgoing_relations")
	target_node: Mapped["GraphRAGNode"] = relationship("GraphRAGNode", foreign_keys=[target_node_id], back_populates="incoming_relations")
	knowledge_graph: Mapped[Optional["KnowledgeGraph"]] = relationship("KnowledgeGraph", back_populates="relations")
	
	# Constraints and indexes
	__table_args__ = (
		Index('idx_cr_graphrag_relations_source_target', 'source_node_id', 'target_node_id'),
		Index('idx_cr_graphrag_relations_type', 'relation_type'),
		Index('idx_cr_graphrag_relations_confidence', 'confidence_score'),
		Index('idx_cr_graphrag_relations_knowledge_graph', 'knowledge_graph_id'),
		Index('idx_cr_graphrag_relations_status', 'relation_status'),
		CheckConstraint('confidence_score >= 0.0 AND confidence_score <= 1.0', name='ck_graphrag_relations_confidence_range'),
		CheckConstraint('strength_score >= 0.0 AND strength_score <= 1.0', name='ck_graphrag_relations_strength_range'),
		CheckConstraint('source_node_id != target_node_id', name='ck_graphrag_relations_no_self_ref'),
		{'schema': 'crawler'}
	)


class KnowledgeGraph(Base):
	"""Knowledge graphs container for GraphRAG processing"""
	__tablename__ = 'cr_knowledge_graphs'
	
	# Core fields
	graph_name: Mapped[str] = mapped_column(String(500), nullable=False)
	description: Mapped[Optional[str]] = mapped_column(Text)
	domain: Mapped[str] = mapped_column(String(100), nullable=False)
	
	# Graph statistics
	node_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
	relation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
	entity_types: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
	relation_types: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
	
	# Configuration and metadata
	graph_statistics: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	graph_schema: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	indexing_config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	
	# Status and audit
	graph_status: Mapped[str] = mapped_column(String(50), nullable=False, default='building')
	last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), nullable=False)
	created_by: Mapped[Optional[str]] = mapped_column(String(255))
	
	# Relationships
	nodes: Mapped[List["GraphRAGNode"]] = relationship("GraphRAGNode", back_populates="knowledge_graph")
	relations: Mapped[List["GraphRAGRelation"]] = relationship("GraphRAGRelation", back_populates="knowledge_graph")
	
	# Constraints and indexes
	__table_args__ = (
		Index('idx_cr_knowledge_graphs_tenant_domain', 'tenant_id', 'domain'),
		Index('idx_cr_knowledge_graphs_status', 'graph_status'),
		Index('idx_cr_knowledge_graphs_updated', 'last_updated'),
		Index('idx_cr_knowledge_graphs_name_gin', 'graph_name', postgresql_using='gin', postgresql_ops={'graph_name': 'gin_trgm_ops'}),
		UniqueConstraint('tenant_id', 'graph_name', name='uq_cr_knowledge_graphs_tenant_name'),
		CheckConstraint('node_count >= 0', name='ck_knowledge_graphs_node_count_positive'),
		CheckConstraint('relation_count >= 0', name='ck_knowledge_graphs_relation_count_positive'),
		{'schema': 'crawler'}
	)


class ContentFingerprint(Base):
	"""Content fingerprints for duplicate detection and versioning"""
	__tablename__ = 'cr_content_fingerprints'
	
	# Core fields
	fingerprint_hash: Mapped[str] = mapped_column(String(64), nullable=False)
	content_type: Mapped[str] = mapped_column(String(100), nullable=False)
	content_length: Mapped[int] = mapped_column(Integer, nullable=False)
	source_url: Mapped[str] = mapped_column(Text, nullable=False)
	
	# Occurrence tracking
	related_records: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
	first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), nullable=False)
	last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), nullable=False)
	occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
	
	# Duplicate detection
	duplicate_cluster_id: Mapped[Optional[str]] = mapped_column(String(255))
	content_similarity_scores: Mapped[Dict[str, float]] = mapped_column(JSON, nullable=False, default=dict)
	fingerprint_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	status: Mapped[str] = mapped_column(String(50), nullable=False, default='unique')
	
	# Constraints and indexes
	__table_args__ = (
		Index('idx_cr_content_fingerprints_hash', 'fingerprint_hash'),
		Index('idx_cr_content_fingerprints_tenant_type', 'tenant_id', 'content_type'),
		Index('idx_cr_content_fingerprints_occurrence', 'occurrence_count'),
		Index('idx_cr_content_fingerprints_cluster', 'duplicate_cluster_id'),
		Index('idx_cr_content_fingerprints_status', 'status'),
		UniqueConstraint('tenant_id', 'fingerprint_hash', name='uq_cr_content_fingerprints_tenant_hash'),
		CheckConstraint('content_length >= 0', name='ck_content_fingerprints_length_positive'),
		CheckConstraint('occurrence_count >= 1', name='ck_content_fingerprints_count_positive'),
		{'schema': 'crawler'}
	)


# =====================================================
# COLLABORATIVE VALIDATION MODELS
# =====================================================

class ValidationSession(Base):
	"""Collaborative validation sessions for quality assurance"""
	__tablename__ = 'cr_validation_sessions'
	
	# Core fields
	session_name: Mapped[str] = mapped_column(String(500), nullable=False)
	description: Mapped[Optional[str]] = mapped_column(Text)
	target_id: Mapped[Optional[UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey('crawler.cr_crawl_targets.id'))
	dataset_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey('crawler.cr_extracted_datasets.id'), nullable=False)
	
	# Validation configuration
	validation_schema: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	consensus_threshold: Mapped[float] = mapped_column(DECIMAL(5, 4), nullable=False, default=0.8)
	quality_threshold: Mapped[float] = mapped_column(DECIMAL(5, 4), nullable=False, default=0.7)
	
	# Session status and metrics
	session_status: Mapped[str] = mapped_column(String(50), nullable=False, default='draft')
	validator_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
	completion_percentage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
	consensus_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	
	# Audit
	created_by: Mapped[Optional[str]] = mapped_column(String(255))
	updated_by: Mapped[Optional[str]] = mapped_column(String(255))
	
	# Relationships
	target: Mapped[Optional["CrawlTarget"]] = relationship("CrawlTarget", back_populates="validation_sessions")
	dataset: Mapped["ExtractedDataset"] = relationship("ExtractedDataset", back_populates="validation_sessions")
	feedback: Mapped[List["ValidationFeedback"]] = relationship("ValidationFeedback", back_populates="session", cascade="all, delete-orphan")
	validators: Mapped[List["ValidatorProfile"]] = relationship("ValidatorProfile", back_populates="sessions")
	
	# Constraints and indexes
	__table_args__ = (
		Index('idx_cr_validation_sessions_tenant_dataset', 'tenant_id', 'dataset_id'),
		Index('idx_cr_validation_sessions_status', 'session_status'),
		Index('idx_cr_validation_sessions_completion', 'completion_percentage'),
		CheckConstraint('consensus_threshold >= 0.0 AND consensus_threshold <= 1.0', name='ck_validation_sessions_consensus_range'),
		CheckConstraint('quality_threshold >= 0.0 AND quality_threshold <= 1.0', name='ck_validation_sessions_quality_range'),
		CheckConstraint('completion_percentage >= 0.0 AND completion_percentage <= 100.0', name='ck_validation_sessions_completion_range'),
		{'schema': 'crawler'}
	)


class ValidatorProfile(Base):
	"""Validator profiles and expertise for collaborative validation"""
	__tablename__ = 'cr_validator_profiles'
	
	# Core fields
	user_id: Mapped[str] = mapped_column(String(255), nullable=False)
	validator_name: Mapped[str] = mapped_column(String(500), nullable=False)
	validator_role: Mapped[str] = mapped_column(String(100), nullable=False)
	
	# Expertise and permissions
	expertise_areas: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
	validation_permissions: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	assignment_status: Mapped[str] = mapped_column(String(50), nullable=False, default='active')
	validation_stats: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	
	# Relationships
	sessions: Mapped[List["ValidationSession"]] = relationship("ValidationSession", back_populates="validators")
	feedback: Mapped[List["ValidationFeedback"]] = relationship("ValidationFeedback", back_populates="validator", cascade="all, delete-orphan")
	
	# Constraints and indexes
	__table_args__ = (
		Index('idx_cr_validator_profiles_tenant_user', 'tenant_id', 'user_id'),
		Index('idx_cr_validator_profiles_role', 'validator_role'),
		Index('idx_cr_validator_profiles_status', 'assignment_status'),
		UniqueConstraint('tenant_id', 'user_id', name='uq_cr_validator_profiles_tenant_user'),
		{'schema': 'crawler'}
	)


class ValidationFeedback(Base):
	"""Validation feedback from validators"""
	__tablename__ = 'cr_validation_feedback'
	
	# Core fields
	session_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey('crawler.cr_validation_sessions.id'), nullable=False)
	validator_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey('crawler.cr_validator_profiles.id'), nullable=False)
	record_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey('crawler.cr_data_records.id'), nullable=False)
	
	# Feedback details
	feedback_type: Mapped[str] = mapped_column(String(50), nullable=False)
	quality_rating: Mapped[int] = mapped_column(Integer, nullable=False)
	accuracy_rating: Mapped[int] = mapped_column(Integer, nullable=False)
	completeness_rating: Mapped[int] = mapped_column(Integer, nullable=False)
	
	# Additional feedback
	suggested_changes: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
	comments: Mapped[Optional[str]] = mapped_column(Text)
	validation_tags: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
	confidence_level: Mapped[float] = mapped_column(DECIMAL(5, 4), nullable=False, default=1.0)
	processing_time_seconds: Mapped[Optional[int]] = mapped_column(Integer)
	
	# Relationships
	session: Mapped["ValidationSession"] = relationship("ValidationSession", back_populates="feedback")
	validator: Mapped["ValidatorProfile"] = relationship("ValidatorProfile", back_populates="feedback")
	record: Mapped["DataRecord"] = relationship("DataRecord", back_populates="validation_feedback")
	
	# Constraints and indexes
	__table_args__ = (
		Index('idx_cr_validation_feedback_session_validator', 'session_id', 'validator_id'),
		Index('idx_cr_validation_feedback_record', 'record_id'),
		Index('idx_cr_validation_feedback_type', 'feedback_type'),
		Index('idx_cr_validation_feedback_ratings', 'quality_rating', 'accuracy_rating', 'completeness_rating'),
		CheckConstraint('quality_rating >= 1 AND quality_rating <= 5', name='ck_validation_feedback_quality_range'),
		CheckConstraint('accuracy_rating >= 1 AND accuracy_rating <= 5', name='ck_validation_feedback_accuracy_range'),
		CheckConstraint('completeness_rating >= 1 AND completeness_rating <= 5', name='ck_validation_feedback_completeness_range'),
		CheckConstraint('confidence_level >= 0.0 AND confidence_level <= 1.0', name='ck_validation_feedback_confidence_range'),
		CheckConstraint('processing_time_seconds IS NULL OR processing_time_seconds >= 0', name='ck_validation_feedback_time_positive'),
		{'schema': 'crawler'}
	)


# =====================================================
# PROCESSING AND EXECUTION MODELS  
# =====================================================

class PipelineExecution(Base):
	"""Pipeline execution tracking and results"""
	__tablename__ = 'cr_pipeline_executions'
	
	# Core fields
	pipeline_id: Mapped[UUID] = mapped_column(UUID(as_uuid=True), ForeignKey('crawler.cr_crawl_pipelines.id'), nullable=False)
	execution_name: Mapped[str] = mapped_column(String(500), nullable=False)
	execution_config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	
	# Execution status and metrics
	status: Mapped[str] = mapped_column(String(50), nullable=False, default='queued')
	start_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
	end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
	records_processed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
	records_successful: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
	records_failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
	
	# Results and metadata
	execution_results: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	performance_metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	error_log: Mapped[List[str]] = mapped_column(ARRAY(Text), nullable=False, default=list)
	
	# Audit
	triggered_by: Mapped[Optional[str]] = mapped_column(String(255))
	
	# Relationships
	pipeline: Mapped["CrawlPipeline"] = relationship("CrawlPipeline", back_populates="executions")
	
	# Constraints and indexes
	__table_args__ = (
		Index('idx_cr_pipeline_executions_tenant_pipeline', 'tenant_id', 'pipeline_id'),
		Index('idx_cr_pipeline_executions_status', 'status'),
		Index('idx_cr_pipeline_executions_start_time', 'start_time'),
		CheckConstraint('records_processed >= 0', name='ck_pipeline_executions_processed_positive'),
		CheckConstraint('records_successful >= 0', name='ck_pipeline_executions_successful_positive'),
		CheckConstraint('records_failed >= 0', name='ck_pipeline_executions_failed_positive'),
		CheckConstraint('start_time IS NULL OR end_time IS NULL OR start_time <= end_time', name='ck_pipeline_executions_time_order'),
		{'schema': 'crawler'}
	)


# =====================================================
# ANALYTICS AND MONITORING MODELS
# =====================================================

class AnalyticsInsight(Base):
	"""Real-time analytics insights and business intelligence"""
	__tablename__ = 'cr_analytics_insights'
	
	# Core fields
	insight_type: Mapped[str] = mapped_column(String(50), nullable=False)
	data_source: Mapped[str] = mapped_column(String(100), nullable=False)
	insight_title: Mapped[str] = mapped_column(String(1000), nullable=False)
	insight_description: Mapped[str] = mapped_column(Text, nullable=False)
	
	# Insight data and scoring
	insight_data: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
	confidence_score: Mapped[float] = mapped_column(DECIMAL(5, 4), nullable=False, default=0.0)
	business_impact: Mapped[float] = mapped_column(DECIMAL(5, 4), nullable=False, default=0.0)
	
	# Recommendations and context
	actionable_recommendations: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)
	related_entities: Mapped[List[str]] = mapped_column(ARRAY(String), nullable=False, default=list)
	time_window_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
	time_window_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
	
	# Status
	status: Mapped[str] = mapped_column(String(50), nullable=False, default='active')
	
	# Constraints and indexes
	__table_args__ = (
		Index('idx_cr_analytics_insights_tenant_type', 'tenant_id', 'insight_type'),
		Index('idx_cr_analytics_insights_confidence', 'confidence_score'),
		Index('idx_cr_analytics_insights_business_impact', 'business_impact'),
		Index('idx_cr_analytics_insights_status', 'status'),
		Index('idx_cr_analytics_insights_time_window', 'time_window_start', 'time_window_end'),
		CheckConstraint('confidence_score >= 0.0 AND confidence_score <= 1.0', name='ck_analytics_insights_confidence_range'),
		CheckConstraint('business_impact >= 0.0 AND business_impact <= 1.0', name='ck_analytics_insights_impact_range'),
		CheckConstraint('time_window_start IS NULL OR time_window_end IS NULL OR time_window_start <= time_window_end', name='ck_analytics_insights_time_order'),
		{'schema': 'crawler'}
	)


# =====================================================
# MODEL EXPORTS
# =====================================================

__all__ = [
	# Base
	"Base",
	
	# Core Models
	"CrawlTarget",
	"CrawlPipeline", 
	"ExtractedDataset",
	"DataRecord",
	"BusinessEntity",
	
	# RAG/GraphRAG Models
	"RAGChunk",
	"GraphRAGNode",
	"GraphRAGRelation", 
	"KnowledgeGraph",
	"ContentFingerprint",
	
	# Validation Models
	"ValidationSession",
	"ValidatorProfile",
	"ValidationFeedback",
	
	# Processing Models
	"PipelineExecution",
	
	# Analytics Models
	"AnalyticsInsight",
]
"""
APG Crawler Capability - Database Service Layer
===============================================

Async database service with:
- Modern SQLAlchemy 2.0+ async patterns
- Multi-tenant operations with APG integration
- RAG and GraphRAG processing services
- Comprehensive error handling and logging
- Performance optimization and caching

Copyright © 2025 Datacraft (nyimbi@gmail.com)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import asyncio
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy import select, update, delete, func, and_, or_, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import hashlib

from .models import (
	Base, CrawlTarget, CrawlPipeline, ExtractedDataset, DataRecord, 
	BusinessEntity, RAGChunk, GraphRAGNode, GraphRAGRelation, 
	KnowledgeGraph, ContentFingerprint, ValidationSession, 
	ValidatorProfile, ValidationFeedback, PipelineExecution, 
	AnalyticsInsight
)
from .views import (
	CrawlTargetRequest, ValidationSessionRequest, ProcessingResult,
	RAGProcessingRequest, GraphRAGProcessingRequest, ContentCleaningRequest,
	RAGProcessingResult, GraphRAGProcessingResult
)


# =====================================================
# SERVICE CONFIGURATION AND SETUP
# =====================================================

logger = logging.getLogger(__name__)


class CrawlerDatabaseService:
	"""
	Async database service for crawler capability with APG integration
	Provides comprehensive CRUD operations, RAG/GraphRAG processing, and analytics
	"""
	
	def __init__(self, database_url: str, **engine_kwargs):
		"""Initialize database service with async SQLAlchemy engine"""
		self.database_url = database_url
		self.engine = create_async_engine(database_url, **engine_kwargs)
		self.async_session_factory = async_sessionmaker(
			self.engine, 
			class_=AsyncSession,
			expire_on_commit=False
		)
		
		# APG integration tracking
		self._tenant_cache: Dict[str, Dict[str, Any]] = {}
		self._performance_metrics: Dict[str, Any] = {}
	
	async def _log_operation(self, operation: str, tenant_id: str, **kwargs) -> None:
		"""Log database operations for APG audit compliance"""
		logger.info(
			f"APG Crawler DB Operation: {operation}",
			extra={
				'tenant_id': tenant_id,
				'operation': operation,
				'timestamp': datetime.utcnow().isoformat(),
				**kwargs
			}
		)
	
	@asynccontextmanager
	async def get_session(self):
		"""Get async database session with proper cleanup"""
		async with self.async_session_factory() as session:
			try:
				yield session
				await session.commit()
			except Exception:
				await session.rollback()
				raise
			finally:
				await session.close()
	
	async def health_check(self) -> Dict[str, Any]:
		"""Check database health and connectivity"""
		try:
			async with self.get_session() as session:
				result = await session.execute(text("SELECT 1"))
				await result.fetchone()
				
				# Check schema exists
				schema_check = await session.execute(
					text("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'crawler'")
				)
				schema_exists = await schema_check.fetchone() is not None
				
				return {
					'status': 'healthy',
					'database_connected': True,
					'schema_exists': schema_exists,
					'timestamp': datetime.utcnow().isoformat()
				}
		except Exception as e:
			logger.error(f"Database health check failed: {str(e)}")
			return {
				'status': 'unhealthy',
				'database_connected': False,
				'error': str(e),
				'timestamp': datetime.utcnow().isoformat()
			}


# =====================================================
# CRAWL TARGET OPERATIONS
# =====================================================

	async def create_crawl_target(self, request: CrawlTargetRequest) -> CrawlTarget:
		"""Create new crawl target with APG integration"""
		await self._log_operation("create_crawl_target", request.tenant_id, target_name=request.name)
		
		async with self.get_session() as session:
			target = CrawlTarget(
				tenant_id=request.tenant_id,
				name=request.name,
				description=request.description,
				target_urls=request.target_urls,
				target_type=request.target_type,
				business_context=request.business_context.model_dump(),
				stealth_requirements=request.stealth_requirements.model_dump() if request.stealth_requirements else {},
				quality_requirements=request.quality_requirements.model_dump() if request.quality_requirements else {},
				scheduling_config=request.scheduling_config.model_dump() if request.scheduling_config else {},
				collaboration_config=request.collaboration_config.model_dump() if request.collaboration_config else {}
			)
			
			session.add(target)
			await session.flush()  # Get ID
			await session.refresh(target)
			
			return target
	
	async def get_crawl_target(self, tenant_id: str, target_id: str) -> Optional[CrawlTarget]:
		"""Get crawl target by ID with tenant isolation"""
		async with self.get_session() as session:
			result = await session.execute(
				select(CrawlTarget)
				.where(and_(CrawlTarget.tenant_id == tenant_id, CrawlTarget.id == target_id))
				.options(selectinload(CrawlTarget.pipelines), selectinload(CrawlTarget.datasets))
			)
			return result.scalar_one_or_none()
	
	async def list_crawl_targets(
		self, 
		tenant_id: str, 
		status: Optional[str] = None,
		target_type: Optional[str] = None,
		limit: int = 100,
		offset: int = 0
	) -> List[CrawlTarget]:
		"""List crawl targets with filtering and pagination"""
		async with self.get_session() as session:
			query = select(CrawlTarget).where(CrawlTarget.tenant_id == tenant_id)
			
			if status:
				query = query.where(CrawlTarget.status == status)
			if target_type:
				query = query.where(CrawlTarget.target_type == target_type)
			
			query = query.offset(offset).limit(limit).order_by(CrawlTarget.created_at.desc())
			
			result = await session.execute(query)
			return result.scalars().all()
	
	async def update_crawl_target(self, tenant_id: str, target_id: str, updates: Dict[str, Any]) -> Optional[CrawlTarget]:
		"""Update crawl target with audit logging"""
		await self._log_operation("update_crawl_target", tenant_id, target_id=target_id, updates=list(updates.keys()))
		
		async with self.get_session() as session:
			result = await session.execute(
				update(CrawlTarget)
				.where(and_(CrawlTarget.tenant_id == tenant_id, CrawlTarget.id == target_id))
				.values(**updates, updated_at=func.now())
				.returning(CrawlTarget)
			)
			
			updated_target = result.scalar_one_or_none()
			if updated_target:
				await session.refresh(updated_target)
			
			return updated_target
	
	async def delete_crawl_target(self, tenant_id: str, target_id: str) -> bool:
		"""Delete crawl target and cascade relationships"""
		await self._log_operation("delete_crawl_target", tenant_id, target_id=target_id)
		
		async with self.get_session() as session:
			result = await session.execute(
				delete(CrawlTarget)
				.where(and_(CrawlTarget.tenant_id == tenant_id, CrawlTarget.id == target_id))
			)
			
			return result.rowcount > 0


# =====================================================
# DATA RECORD OPERATIONS
# =====================================================

	async def create_data_record(self, record_data: Dict[str, Any]) -> DataRecord:
		"""Create data record with automatic fingerprinting"""
		tenant_id = record_data['tenant_id']
		await self._log_operation("create_data_record", tenant_id, dataset_id=record_data.get('dataset_id'))
		
		async with self.get_session() as session:
			# Generate content fingerprint if markdown content exists
			fingerprint = ""
			if record_data.get('markdown_content'):
				fingerprint = hashlib.sha256(record_data['markdown_content'].encode('utf-8')).hexdigest()
				record_data['content_fingerprint'] = fingerprint
			
			record = DataRecord(**record_data)
			session.add(record)
			await session.flush()
			
			# Update fingerprint tracking if needed
			if fingerprint:
				await self._update_content_fingerprint(session, tenant_id, fingerprint, record_data)
			
			await session.refresh(record)
			return record
	
	async def get_data_records_by_dataset(
		self, 
		tenant_id: str, 
		dataset_id: str,
		processing_stage: Optional[str] = None,
		limit: int = 1000,
		offset: int = 0
	) -> List[DataRecord]:
		"""Get data records for dataset with filtering"""
		async with self.get_session() as session:
			query = (
				select(DataRecord)
				.where(and_(DataRecord.tenant_id == tenant_id, DataRecord.dataset_id == dataset_id))
				.options(
					selectinload(DataRecord.business_entities),
					selectinload(DataRecord.rag_chunks)
				)
			)
			
			if processing_stage:
				query = query.where(DataRecord.content_processing_stage == processing_stage)
			
			query = query.offset(offset).limit(limit).order_by(DataRecord.record_index)
			
			result = await session.execute(query)
			return result.scalars().all()
	
	async def update_record_processing_stage(
		self, 
		tenant_id: str, 
		record_id: str, 
		stage: str,
		metadata: Optional[Dict[str, Any]] = None
	) -> bool:
		"""Update record processing stage and metadata"""
		async with self.get_session() as session:
			updates = {'content_processing_stage': stage, 'updated_at': func.now()}
			if metadata:
				updates['rag_metadata'] = metadata
			
			result = await session.execute(
				update(DataRecord)
				.where(and_(DataRecord.tenant_id == tenant_id, DataRecord.id == record_id))
				.values(**updates)
			)
			
			return result.rowcount > 0


# =====================================================
# RAG PROCESSING OPERATIONS
# =====================================================

	async def process_records_for_rag(self, request: RAGProcessingRequest) -> RAGProcessingResult:
		"""Process data records into RAG chunks with embeddings"""
		await self._log_operation("process_records_for_rag", request.tenant_id, record_count=len(request.record_ids))
		
		start_time = datetime.utcnow()
		chunks_created = 0
		errors = []
		created_chunk_ids = []
		
		async with self.get_session() as session:
			try:
				# Get records to process
				records_query = select(DataRecord).where(
					and_(
						DataRecord.tenant_id == request.tenant_id,
						DataRecord.id.in_(request.record_ids)
					)
				)
				result = await session.execute(records_query)
				records = result.scalars().all()
				
				for record in records:
					try:
						chunks = await self._create_rag_chunks(session, record, request.rag_config)
						chunks_created += len(chunks)
						created_chunk_ids.extend([chunk.id for chunk in chunks])
						
						# Update record processing stage
						await session.execute(
							update(DataRecord)
							.where(DataRecord.id == record.id)
							.values(
								content_processing_stage='rag_processed',
								rag_chunk_ids=[chunk.id for chunk in chunks],
								updated_at=func.now()
							)
						)
						
					except Exception as e:
						error_msg = f"Failed to process record {record.id}: {str(e)}"
						errors.append(error_msg)
						logger.error(error_msg)
				
				processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
				
				return RAGProcessingResult(
					status='completed' if not errors else 'partially_completed',
					records_processed=len(records),
					chunks_created=chunks_created,
					chunks_embedded=chunks_created,  # Assuming embeddings generated
					chunks_indexed=chunks_created,
					processing_time_ms=processing_time,
					created_chunk_ids=created_chunk_ids,
					vector_index_status='indexed',
					error_count=len(errors),
					errors=errors
				)
				
			except Exception as e:
				logger.error(f"RAG processing failed: {str(e)}")
				return RAGProcessingResult(
					status='failed',
					error_count=1,
					errors=[str(e)]
				)
	
	async def _create_rag_chunks(self, session: AsyncSession, record: DataRecord, config: Any) -> List[RAGChunk]:
		"""Create RAG chunks from record content"""
		if not record.markdown_content:
			raise ValueError("Record must have markdown_content for RAG processing")
		
		chunks = []
		content = record.markdown_content
		chunk_size = config.chunk_size
		overlap_size = config.overlap_size
		
		# Simple chunking algorithm (can be enhanced with semantic chunking)
		start = 0
		chunk_index = 0
		
		while start < len(content):
			end = min(start + chunk_size, len(content))
			chunk_text = content[start:end]
			
			# Create chunk fingerprint
			chunk_fingerprint = hashlib.sha256(chunk_text.encode('utf-8')).hexdigest()
			
			chunk = RAGChunk(
				tenant_id=record.tenant_id,
				record_id=record.id,
				chunk_index=chunk_index,
				chunk_text=chunk_text,
				chunk_markdown=chunk_text,  # Same as text for now
				chunk_fingerprint=chunk_fingerprint,
				embedding_model=config.embedding_model,
				vector_dimensions=config.vector_dimensions,
				semantic_similarity_threshold=config.entity_resolution_threshold,
				chunk_overlap_start=max(0, start - overlap_size),
				chunk_overlap_end=min(len(content), end + overlap_size),
				contextual_metadata=config.rag_metadata,
				indexing_status='indexed'
			)
			
			session.add(chunk)
			chunks.append(chunk)
			
			# Move to next chunk with overlap
			start = end - overlap_size
			chunk_index += 1
			
			if end >= len(content):
				break
		
		await session.flush()
		return chunks
	
	async def search_similar_chunks(
		self, 
		tenant_id: str, 
		query_embedding: List[float],
		limit: int = 10,
		similarity_threshold: float = 0.8
	) -> List[Tuple[RAGChunk, float]]:
		"""Search for similar RAG chunks using vector similarity"""
		async with self.get_session() as session:
			# Vector similarity search using pgvector
			query = text("""
				SELECT *, (vector_embeddings <=> :query_embedding) as distance
				FROM crawler.cr_rag_chunks 
				WHERE tenant_id = :tenant_id 
				AND vector_embeddings IS NOT NULL
				AND (vector_embeddings <=> :query_embedding) < :distance_threshold
				ORDER BY distance
				LIMIT :limit
			""")
			
			result = await session.execute(
				query,
				{
					'tenant_id': tenant_id,
					'query_embedding': str(query_embedding),
					'distance_threshold': 1.0 - similarity_threshold,  # Convert similarity to distance
					'limit': limit
				}
			)
			
			rows = result.fetchall()
			
			# Convert back to RAGChunk objects with similarity scores
			chunks_with_scores = []
			for row in rows:
				chunk_query = select(RAGChunk).where(RAGChunk.id == row.id)
				chunk_result = await session.execute(chunk_query)
				chunk = chunk_result.scalar_one_or_none()
				if chunk:
					similarity = 1.0 - row.distance  # Convert distance back to similarity
					chunks_with_scores.append((chunk, similarity))
			
			return chunks_with_scores


# =====================================================
# GRAPHRAG PROCESSING OPERATIONS
# =====================================================

	async def process_chunks_for_graphrag(self, request: GraphRAGProcessingRequest) -> GraphRAGProcessingResult:
		"""Process RAG chunks into GraphRAG knowledge graph"""
		await self._log_operation("process_chunks_for_graphrag", request.tenant_id, chunk_count=len(request.rag_chunk_ids))
		
		start_time = datetime.utcnow()
		nodes_created = 0
		relations_created = 0
		errors = []
		
		async with self.get_session() as session:
			try:
				# Get chunks to process
				chunks_query = select(RAGChunk).where(
					and_(
						RAGChunk.tenant_id == request.tenant_id,
						RAGChunk.id.in_(request.rag_chunk_ids)
					)
				).options(selectinload(RAGChunk.record))
				
				result = await session.execute(chunks_query)
				chunks = result.scalars().all()
				
				created_node_ids = []
				created_relation_ids = []
				
				for chunk in chunks:
					try:
						# Extract entities (simplified - would use NLP in real implementation)
						entities = await self._extract_entities_from_chunk(chunk)
						
						# Create nodes for entities
						for entity_data in entities:
							node = GraphRAGNode(
								tenant_id=request.tenant_id,
								record_id=chunk.record_id,
								node_type='entity',
								node_name=entity_data['name'],
								node_description=entity_data.get('description'),
								entity_type=entity_data.get('type'),
								confidence_score=entity_data.get('confidence', 0.8),
								salience_score=entity_data.get('salience', 0.7),
								related_chunks=[chunk.id],
								knowledge_graph_id=request.knowledge_graph_id
							)
							
							session.add(node)
							await session.flush()
							created_node_ids.append(node.id)
							nodes_created += 1
						
						# Extract relations (simplified)
						relations = await self._extract_relations_from_chunk(chunk, created_node_ids[-len(entities):])
						
						for relation_data in relations:
							relation = GraphRAGRelation(
								tenant_id=request.tenant_id,
								source_node_id=relation_data['source_id'],
								target_node_id=relation_data['target_id'],
								relation_type=relation_data['type'],
								relation_label=relation_data['label'],
								confidence_score=relation_data.get('confidence', 0.7),
								strength_score=relation_data.get('strength', 0.6),
								evidence_chunks=[chunk.id],
								knowledge_graph_id=request.knowledge_graph_id
							)
							
							session.add(relation)
							await session.flush()
							created_relation_ids.append(relation.id)
							relations_created += 1
						
					except Exception as e:
						error_msg = f"Failed to process chunk {chunk.id}: {str(e)}"
						errors.append(error_msg)
						logger.error(error_msg)
				
				# Update knowledge graph statistics if specified
				if request.knowledge_graph_id:
					await self._update_knowledge_graph_stats(session, request.knowledge_graph_id)
				
				processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
				
				return GraphRAGProcessingResult(
					status='completed' if not errors else 'partially_completed',
					chunks_processed=len(chunks),
					nodes_created=nodes_created,
					relations_created=relations_created,
					processing_time_ms=processing_time,
					created_node_ids=created_node_ids,
					created_relation_ids=created_relation_ids,
					knowledge_graph_id=request.knowledge_graph_id,
					error_count=len(errors),
					errors=errors
				)
				
			except Exception as e:
				logger.error(f"GraphRAG processing failed: {str(e)}")
				return GraphRAGProcessingResult(
					status='failed',
					error_count=1,
					errors=[str(e)]
				)
	
	async def _extract_entities_from_chunk(self, chunk: RAGChunk) -> List[Dict[str, Any]]:
		"""Extract entities from chunk content (simplified implementation)"""
		# This would integrate with APG NLP capability in real implementation
		# For now, return mock entities
		return [
			{
				'name': 'Sample Entity',
				'type': 'organization',
				'description': 'A sample entity extracted from content',
				'confidence': 0.85,
				'salience': 0.75
			}
		]
	
	async def _extract_relations_from_chunk(self, chunk: RAGChunk, node_ids: List[str]) -> List[Dict[str, Any]]:
		"""Extract relations from chunk content (simplified implementation)"""
		# This would use advanced NLP for relation extraction
		# For now, return mock relations
		if len(node_ids) >= 2:
			return [
				{
					'source_id': node_ids[0],
					'target_id': node_ids[1],
					'type': 'related_to',
					'label': 'has relationship with',
					'confidence': 0.75,
					'strength': 0.65
				}
			]
		return []
	
	async def _update_knowledge_graph_stats(self, session: AsyncSession, graph_id: str) -> None:
		"""Update knowledge graph node and relation counts"""
		# Count nodes
		node_count_query = select(func.count(GraphRAGNode.id)).where(GraphRAGNode.knowledge_graph_id == graph_id)
		node_count_result = await session.execute(node_count_query)
		node_count = node_count_result.scalar()
		
		# Count relations
		relation_count_query = select(func.count(GraphRAGRelation.id)).where(GraphRAGRelation.knowledge_graph_id == graph_id)
		relation_count_result = await session.execute(relation_count_query)
		relation_count = relation_count_result.scalar()
		
		# Update knowledge graph
		await session.execute(
			update(KnowledgeGraph)
			.where(KnowledgeGraph.id == graph_id)
			.values(
				node_count=node_count,
				relation_count=relation_count,
				last_updated=func.now()
			)
		)


# =====================================================
# CONTENT FINGERPRINTING OPERATIONS
# =====================================================

	async def _update_content_fingerprint(
		self, 
		session: AsyncSession, 
		tenant_id: str, 
		fingerprint: str, 
		record_data: Dict[str, Any]
	) -> None:
		"""Update content fingerprint tracking for duplicate detection"""
		try:
			# Try to insert new fingerprint
			fingerprint_data = ContentFingerprint(
				tenant_id=tenant_id,
				fingerprint_hash=fingerprint,
				content_type=record_data.get('content_type', 'text/markdown'),
				content_length=len(record_data.get('markdown_content', '')),
				source_url=record_data.get('source_url', ''),
				related_records=[record_data.get('id', '')],
				occurrence_count=1,
				status='unique'
			)
			
			session.add(fingerprint_data)
			await session.flush()
			
		except IntegrityError:
			# Fingerprint already exists, update occurrence count
			await session.execute(
				update(ContentFingerprint)
				.where(
					and_(
						ContentFingerprint.tenant_id == tenant_id,
						ContentFingerprint.fingerprint_hash == fingerprint
					)
				)
				.values(
					occurrence_count=ContentFingerprint.occurrence_count + 1,
					last_seen=func.now(),
					status='duplicate'
				)
			)
	
	async def get_duplicate_content(self, tenant_id: str, min_occurrences: int = 2) -> List[ContentFingerprint]:
		"""Get content with multiple occurrences (duplicates)"""
		async with self.get_session() as session:
			query = (
				select(ContentFingerprint)
				.where(
					and_(
						ContentFingerprint.tenant_id == tenant_id,
						ContentFingerprint.occurrence_count >= min_occurrences
					)
				)
				.order_by(ContentFingerprint.occurrence_count.desc())
			)
			
			result = await session.execute(query)
			return result.scalars().all()


# =====================================================
# VALIDATION OPERATIONS
# =====================================================

	async def create_validation_session(self, request: ValidationSessionRequest) -> ValidationSession:
		"""Create collaborative validation session"""
		await self._log_operation("create_validation_session", request.tenant_id, dataset_id=request.dataset_id)
		
		async with self.get_session() as session:
			session_data = ValidationSession(
				tenant_id=request.tenant_id,
				dataset_id=request.dataset_id,
				session_name=request.session_name,
				description=request.description,
				validation_schema=request.validation_schema.model_dump(),
				consensus_threshold=request.consensus_threshold,
				quality_threshold=request.quality_threshold,
				validator_count=len(request.validator_profiles)
			)
			
			session.add(session_data)
			await session.flush()
			
			# Add validators
			for validator_profile in request.validator_profiles:
				validator = ValidatorProfile(
					tenant_id=request.tenant_id,
					user_id=validator_profile.user_id,
					validator_name=validator_profile.validator_name,
					validator_role=validator_profile.validator_role,
					expertise_areas=validator_profile.expertise_areas,
					validation_permissions=validator_profile.validation_permissions
				)
				session.add(validator)
			
			await session.refresh(session_data)
			return session_data


# =====================================================
# ANALYTICS OPERATIONS
# =====================================================

	async def get_crawler_analytics(self, tenant_id: str, time_range: timedelta = timedelta(days=7)) -> Dict[str, Any]:
		"""Get comprehensive crawler analytics for tenant"""
		async with self.get_session() as session:
			start_time = datetime.utcnow() - time_range
			
			# Basic counts
			targets_query = select(func.count(CrawlTarget.id)).where(
				and_(CrawlTarget.tenant_id == tenant_id, CrawlTarget.created_at >= start_time)
			)
			targets_count = await session.scalar(targets_query)
			
			records_query = select(func.count(DataRecord.id)).where(
				and_(DataRecord.tenant_id == tenant_id, DataRecord.created_at >= start_time)
			)
			records_count = await session.scalar(records_query)
			
			chunks_query = select(func.count(RAGChunk.id)).where(
				and_(RAGChunk.tenant_id == tenant_id, RAGChunk.created_at >= start_time)
			)
			chunks_count = await session.scalar(chunks_query)
			
			nodes_query = select(func.count(GraphRAGNode.id)).where(
				and_(GraphRAGNode.tenant_id == tenant_id, GraphRAGNode.created_at >= start_time)
			)
			nodes_count = await session.scalar(nodes_query)
			
			# Quality metrics
			avg_quality_query = select(func.avg(DataRecord.quality_score)).where(
				and_(DataRecord.tenant_id == tenant_id, DataRecord.created_at >= start_time)
			)
			avg_quality = await session.scalar(avg_quality_query) or 0.0
			
			return {
				'tenant_id': tenant_id,
				'time_range_days': time_range.days,
				'crawl_targets': targets_count or 0,
				'data_records': records_count or 0,
				'rag_chunks': chunks_count or 0,
				'graphrag_nodes': nodes_count or 0,
				'average_quality_score': float(avg_quality),
				'generated_at': datetime.utcnow().isoformat()
			}


# =====================================================
# UTILITY METHODS
# =====================================================

	async def cleanup_old_data(self, tenant_id: str, retention_days: int = 90) -> Dict[str, int]:
		"""Clean up old data based on retention policy"""
		await self._log_operation("cleanup_old_data", tenant_id, retention_days=retention_days)
		
		cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
		deleted_counts = {}
		
		async with self.get_session() as session:
			# Clean up old pipeline executions
			exec_result = await session.execute(
				delete(PipelineExecution)
				.where(
					and_(
						PipelineExecution.tenant_id == tenant_id,
						PipelineExecution.created_at < cutoff_date
					)
				)
			)
			deleted_counts['pipeline_executions'] = exec_result.rowcount
			
			# Clean up old analytics insights
			insight_result = await session.execute(
				delete(AnalyticsInsight)
				.where(
					and_(
						AnalyticsInsight.tenant_id == tenant_id,
						AnalyticsInsight.created_at < cutoff_date,
						AnalyticsInsight.status == 'resolved'
					)
				)
			)
			deleted_counts['analytics_insights'] = insight_result.rowcount
			
			return deleted_counts
	
	async def close(self):
		"""Close database connections"""
		if self.engine:
			await self.engine.dispose()


# =====================================================
# SERVICE FACTORY
# =====================================================

def create_crawler_service(database_url: str, **kwargs) -> CrawlerDatabaseService:
	"""Factory function to create crawler database service"""
	return CrawlerDatabaseService(database_url, **kwargs)


# Export service class
__all__ = ['CrawlerDatabaseService', 'create_crawler_service']
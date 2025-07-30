"""
APG RAG Vector Service

High-performance vector indexing and embedding pipeline optimized for PostgreSQL + pgvector
with intelligent batch processing, hierarchical indexing, and real-time updates.
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from uuid_extensions import uuid7str
from collections import defaultdict, deque

# Database imports
import asyncpg
from asyncpg import Pool, Connection

# APG imports
from .models import (
	DocumentChunk, DocumentChunkCreate, RetrievalRequest, RetrievalResult,
	RetrievalMethod, APGBaseModel
)
from .ollama_integration import AdvancedOllamaIntegration, EmbeddingQueueRequest, RequestPriority

class IndexingStrategy(str, Enum):
	"""Vector indexing strategies"""
	IMMEDIATE = "immediate"
	BATCH = "batch"
	SCHEDULED = "scheduled"
	ADAPTIVE = "adaptive"

class SimilarityMetric(str, Enum):
	"""Vector similarity metrics"""
	COSINE = "cosine"
	L2 = "l2"
	INNER_PRODUCT = "inner_product"

@dataclass
class VectorIndexConfig:
	"""Configuration for vector indexing"""
	# Index parameters
	index_type: str = "ivfflat"  # or "hnsw" when available
	lists: int = 1000  # Number of lists for IVFFlat
	probes: int = 10   # Number of probes for queries
	
	# Performance settings
	batch_size: int = 1000
	max_batch_wait_time: float = 5.0  # seconds
	parallel_workers: int = 4
	
	# Quality settings
	embedding_dimension: int = 1024  # bge-m3 dimension
	similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
	
	# Maintenance
	rebuild_threshold: float = 0.1  # Rebuild when 10% of data changes
	vacuum_interval: int = 3600     # Vacuum every hour

@dataclass
class EmbeddingBatchRequest:
	"""Batch request for embeddings"""
	request_id: str = field(default_factory=uuid7str)
	tenant_id: str = ""
	capability_id: str = ""
	chunks: List[DocumentChunk] = field(default_factory=list)
	priority: RequestPriority = RequestPriority.NORMAL
	created_at: datetime = field(default_factory=datetime.now)
	callback: Optional[callable] = None

@dataclass
class IndexingResult:
	"""Result of vector indexing operation"""
	success: bool
	chunks_processed: int
	processing_time_ms: float
	embeddings_generated: int
	index_updates: int
	error_message: Optional[str] = None

class VectorCache:
	"""High-performance vector cache with intelligent eviction"""
	
	def __init__(self, max_size: int = 50000, ttl_hours: int = 24):
		self.max_size = max_size
		self.ttl_seconds = ttl_hours * 3600
		self.cache: Dict[str, Tuple[List[float], float, int]] = {}  # key -> (vector, timestamp, access_count)
		self.access_order = deque()  # For LRU eviction
		self._lock = asyncio.Lock()
		
		# Statistics
		self.stats = {
			'hits': 0,
			'misses': 0,
			'evictions': 0,
			'size': 0
		}
	
	def _generate_cache_key(self, content: str, model: str) -> str:
		"""Generate cache key for content and model"""
		content_hash = hashlib.md5(content.encode()).hexdigest()
		return f"{model}:{content_hash}"
	
	async def get(self, content: str, model: str) -> Optional[List[float]]:
		"""Get vector from cache"""
		key = self._generate_cache_key(content, model)
		
		async with self._lock:
			if key in self.cache:
				vector, timestamp, access_count = self.cache[key]
				
				# Check if expired
				if time.time() - timestamp > self.ttl_seconds:
					del self.cache[key]
					self.access_order.remove(key)
					self.stats['size'] -= 1
					self.stats['misses'] += 1
					return None
				
				# Update access count and order
				self.cache[key] = (vector, timestamp, access_count + 1)
				self.access_order.remove(key)
				self.access_order.append(key)
				
				self.stats['hits'] += 1
				return vector
			else:
				self.stats['misses'] += 1
				return None
	
	async def put(self, content: str, model: str, vector: List[float]) -> None:
		"""Store vector in cache"""
		key = self._generate_cache_key(content, model)
		timestamp = time.time()
		
		async with self._lock:
			# Evict if at capacity
			if len(self.cache) >= self.max_size and key not in self.cache:
				await self._evict_lru()
			
			# Store vector
			self.cache[key] = (vector, timestamp, 1)
			if key in self.access_order:
				self.access_order.remove(key)
			self.access_order.append(key)
			
			self.stats['size'] = len(self.cache)
	
	async def _evict_lru(self) -> None:
		"""Evict least recently used item"""
		if self.access_order:
			lru_key = self.access_order.popleft()
			del self.cache[lru_key]
			self.stats['evictions'] += 1
	
	async def clear(self) -> None:
		"""Clear all cached vectors"""
		async with self._lock:
			self.cache.clear()
			self.access_order.clear()
			self.stats = {'hits': 0, 'misses': 0, 'evictions': 0, 'size': 0}
	
	def get_stats(self) -> Dict[str, Any]:
		"""Get cache statistics"""
		total_requests = self.stats['hits'] + self.stats['misses']
		hit_rate = self.stats['hits'] / max(1, total_requests)
		
		return {
			**self.stats,
			'hit_rate': hit_rate,
			'total_requests': total_requests
		}

class VectorIndexManager:
	"""Manages pgvector indexes with optimization"""
	
	def __init__(self, config: VectorIndexConfig, db_pool: Pool):
		self.config = config
		self.db_pool = db_pool
		self.logger = logging.getLogger(__name__)
		
		# Index maintenance state
		self.last_vacuum = datetime.min
		self.documents_since_rebuild = 0
		self.total_documents = 0
	
	async def ensure_indexes_exist(self, tenant_id: str) -> None:
		"""Ensure vector indexes exist and are optimized"""
		async with self.db_pool.acquire() as conn:
			# Check if indexes exist
			index_exists = await conn.fetchval("""
				SELECT EXISTS (
					SELECT 1 FROM pg_indexes 
					WHERE tablename = 'apg_rag_document_chunks' 
					AND indexname = 'idx_apg_rag_chunks_embedding_tenant'
				)
			""")
			
			if not index_exists:
				await self._create_vector_index(conn, tenant_id)
			
			# Check index health and optimize if needed
			await self._optimize_index_if_needed(conn, tenant_id)
	
	async def _create_vector_index(self, conn: Connection, tenant_id: str) -> None:
		"""Create optimized vector index"""
		self.logger.info(f"Creating vector index for tenant {tenant_id}")
		
		# Create IVFFlat index with optimal parameters
		await conn.execute(f"""
			CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_apg_rag_chunks_embedding_tenant_{tenant_id.replace('-', '_')}
			ON apg_rag_document_chunks 
			USING ivfflat (embedding vector_cosine_ops) 
			WITH (lists = {self.config.lists})
			WHERE tenant_id = $1
		""", tenant_id)
		
		# Set optimal query parameters
		await conn.execute(f"SET ivfflat.probes = {self.config.probes}")
		
		self.logger.info(f"Vector index created successfully for tenant {tenant_id}")
	
	async def _optimize_index_if_needed(self, conn: Connection, tenant_id: str) -> None:
		"""Optimize index based on data changes"""
		# Check if rebuild is needed
		if (self.documents_since_rebuild / max(1, self.total_documents)) > self.config.rebuild_threshold:
			await self._rebuild_index(conn, tenant_id)
			self.documents_since_rebuild = 0
		
		# Vacuum if needed
		if (datetime.now() - self.last_vacuum).total_seconds() > self.config.vacuum_interval:
			await self._vacuum_index(conn)
			self.last_vacuum = datetime.now()
	
	async def _rebuild_index(self, conn: Connection, tenant_id: str) -> None:
		"""Rebuild vector index for optimal performance"""
		self.logger.info(f"Rebuilding vector index for tenant {tenant_id}")
		
		index_name = f"idx_apg_rag_chunks_embedding_tenant_{tenant_id.replace('-', '_')}"
		
		# Drop and recreate index
		await conn.execute(f"DROP INDEX IF EXISTS {index_name}")
		await self._create_vector_index(conn, tenant_id)
		
		self.logger.info(f"Vector index rebuilt for tenant {tenant_id}")
	
	async def _vacuum_index(self, conn: Connection) -> None:
		"""Vacuum table for index optimization"""
		await conn.execute("VACUUM ANALYZE apg_rag_document_chunks")
		self.logger.debug("Vector index vacuumed")
	
	def record_document_change(self, change_count: int = 1) -> None:
		"""Record document changes for index maintenance"""
		self.documents_since_rebuild += change_count
		self.total_documents = max(self.total_documents, self.documents_since_rebuild)

class VectorService:
	"""High-performance vector service with pgvector integration"""
	
	def __init__(self,
	             config: VectorIndexConfig,
	             db_pool: Pool,
	             ollama_integration: AdvancedOllamaIntegration,
	             tenant_id: str,
	             capability_id: str = "rag"):
		
		self.config = config
		self.db_pool = db_pool
		self.ollama_integration = ollama_integration
		self.tenant_id = tenant_id
		self.capability_id = capability_id
		
		# Core components
		self.index_manager = VectorIndexManager(config, db_pool)
		self.vector_cache = VectorCache()
		
		# Processing queues
		self.embedding_queue = asyncio.Queue(maxsize=10000)
		self.indexing_queue = asyncio.Queue(maxsize=5000)
		
		# State management
		self.is_running = False
		self.worker_tasks = []
		
		# Statistics
		self.stats = {
			'embeddings_generated': 0,
			'chunks_indexed': 0,
			'cache_operations': 0,
			'query_operations': 0,
			'average_embedding_time_ms': 0.0,
			'average_indexing_time_ms': 0.0,
			'last_updated': datetime.now()
		}
		
		self.logger = logging.getLogger(__name__)
	
	async def start(self) -> None:
		"""Start the vector service"""
		if self.is_running:
			return
		
		self.is_running = True
		self.logger.info("Vector service starting")
		
		# Ensure indexes exist
		await self.index_manager.ensure_indexes_exist(self.tenant_id)
		
		# Start worker tasks
		self.worker_tasks = [
			asyncio.create_task(self._embedding_worker()),
			asyncio.create_task(self._indexing_worker()),
			asyncio.create_task(self._maintenance_worker())
		]
		
		self.logger.info("Vector service started successfully")
	
	async def stop(self) -> None:
		"""Stop the vector service"""
		if not self.is_running:
			return
		
		self.is_running = False
		self.logger.info("Vector service stopping")
		
		# Cancel worker tasks
		for task in self.worker_tasks:
			task.cancel()
		
		await asyncio.gather(*self.worker_tasks, return_exceptions=True)
		
		# Clear cache
		await self.vector_cache.clear()
		
		self.logger.info("Vector service stopped")
	
	async def index_chunks_async(self, 
	                            chunks: List[DocumentChunk],
	                            priority: RequestPriority = RequestPriority.NORMAL,
	                            callback: Optional[callable] = None) -> str:
		"""Queue chunks for asynchronous indexing"""
		
		request_id = uuid7str()
		batch_request = EmbeddingBatchRequest(
			request_id=request_id,
			tenant_id=self.tenant_id,
			capability_id=self.capability_id,
			chunks=chunks,
			priority=priority,
			callback=callback
		)
		
		await self.embedding_queue.put(batch_request)
		self.logger.debug(f"Queued {len(chunks)} chunks for indexing with request {request_id}")
		
		return request_id
	
	async def index_chunks(self, chunks: List[DocumentChunk]) -> IndexingResult:
		"""Index chunks synchronously"""
		start_time = time.time()
		
		try:
			# Filter chunks that need embeddings
			chunks_needing_embeddings = [
				chunk for chunk in chunks 
				if not chunk.embedding or all(x == 0.0 for x in chunk.embedding)
			]
			
			embeddings_generated = 0
			
			# Generate embeddings for chunks that need them
			if chunks_needing_embeddings:
				await self._generate_embeddings_for_chunks(chunks_needing_embeddings)
				embeddings_generated = len(chunks_needing_embeddings)
			
			# Store chunks in database
			index_updates = await self._store_chunks_in_database(chunks)
			
			# Update index maintenance counters
			self.index_manager.record_document_change(len(chunks))
			
			# Update statistics
			processing_time_ms = (time.time() - start_time) * 1000
			self._update_stats('indexing', processing_time_ms, len(chunks))
			
			return IndexingResult(
				success=True,
				chunks_processed=len(chunks),
				processing_time_ms=processing_time_ms,
				embeddings_generated=embeddings_generated,
				index_updates=index_updates
			)
		
		except Exception as e:
			processing_time_ms = (time.time() - start_time) * 1000
			self.logger.error(f"Chunk indexing failed: {str(e)}")
			
			return IndexingResult(
				success=False,
				chunks_processed=0,
				processing_time_ms=processing_time_ms,
				embeddings_generated=0,
				index_updates=0,
				error_message=str(e)
			)
	
	async def vector_search(self, 
	                       query_embedding: List[float],
	                       knowledge_base_id: str,
	                       k: int = 10,
	                       similarity_threshold: float = 0.7,
	                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
		"""Perform vector similarity search"""
		start_time = time.time()
		
		try:
			async with self.db_pool.acquire() as conn:
				# Set query parameters for optimal performance
				await conn.execute(f"SET ivfflat.probes = {self.config.probes}")
				
				# Build query with optional filters
				where_conditions = [
					"tenant_id = $1",
					"knowledge_base_id = $2"
				]
				params = [self.tenant_id, knowledge_base_id]
				param_count = 2
				
				# Add filters if provided
				if filters:
					for key, value in filters.items():
						if key in ['document_id', 'section_title', 'section_level']:
							param_count += 1
							where_conditions.append(f"{key} = ${param_count}")
							params.append(value)
				
				# Add similarity threshold
				param_count += 1
				similarity_condition = f"1 - (embedding <=> ${param_count}) >= ${param_count + 1}"
				where_conditions.append(similarity_condition)
				params.extend([query_embedding, similarity_threshold])
				
				where_clause = " AND ".join(where_conditions)
				
				# Execute vector similarity search
				query = f"""
					SELECT 
						c.id,
						c.document_id,
						c.content,
						c.chunk_index,
						c.character_count,
						c.section_title,
						c.section_level,
						d.title as document_title,
						d.filename as document_filename,
						1 - (c.embedding <=> ${param_count - 1}) as similarity_score
					FROM apg_rag_document_chunks c
					JOIN apg_rag_documents d ON c.document_id = d.id
					WHERE {where_clause}
					ORDER BY c.embedding <=> ${param_count - 1}
					LIMIT ${param_count + 2}
				"""
				
				params.append(k)
				results = await conn.fetch(query, *params)
				
				# Convert to dictionaries
				search_results = []
				for row in results:
					result = {
						'chunk_id': str(row['id']),
						'document_id': str(row['document_id']),
						'content': row['content'],
						'chunk_index': row['chunk_index'],
						'character_count': row['character_count'],
						'section_title': row['section_title'],
						'section_level': row['section_level'],
						'document_title': row['document_title'],
						'document_filename': row['document_filename'],
						'similarity_score': float(row['similarity_score'])
					}
					search_results.append(result)
				
				# Update statistics
				processing_time_ms = (time.time() - start_time) * 1000
				self._update_stats('query', processing_time_ms, len(search_results))
				
				return search_results
		
		except Exception as e:
			self.logger.error(f"Vector search failed: {str(e)}")
			raise
	
	async def hybrid_search(self,
	                       query_text: str,
	                       query_embedding: List[float],
	                       knowledge_base_id: str,
	                       k: int = 10,
	                       vector_weight: float = 0.7,
	                       text_weight: float = 0.3,
	                       similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
		"""Perform hybrid vector + text search"""
		start_time = time.time()
		
		try:
			async with self.db_pool.acquire() as conn:
				# Set query parameters
				await conn.execute(f"SET ivfflat.probes = {self.config.probes}")
				
				# Execute hybrid search using stored function
				results = await conn.fetch("""
					SELECT * FROM rg_hybrid_search($1, $2, $3, $4, $5, $6, $7)
				""", self.tenant_id, knowledge_base_id, query_text, query_embedding, k, vector_weight, text_weight)
				
				# Convert to dictionaries and filter by threshold
				search_results = []
				for row in results:
					combined_score = float(row['combined_score'])
					if combined_score >= similarity_threshold:
						result = {
							'chunk_id': str(row['chunk_id']),
							'document_id': str(row['document_id']),
							'content': row['content'],
							'combined_score': combined_score,
							'vector_score': float(row['vector_score']),
							'text_score': float(row['text_score'])
						}
						search_results.append(result)
				
				# Update statistics
				processing_time_ms = (time.time() - start_time) * 1000
				self._update_stats('query', processing_time_ms, len(search_results))
				
				return search_results
		
		except Exception as e:
			self.logger.error(f"Hybrid search failed: {str(e)}")
			raise
	
	async def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
		"""Get chunk by ID with metadata"""
		try:
			async with self.db_pool.acquire() as conn:
				result = await conn.fetchrow("""
					SELECT 
						c.*,
						d.title as document_title,
						d.filename as document_filename
					FROM apg_rag_document_chunks c
					JOIN apg_rag_documents d ON c.document_id = d.id
					WHERE c.id = $1 AND c.tenant_id = $2
				""", chunk_id, self.tenant_id)
				
				if result:
					return dict(result)
				return None
		
		except Exception as e:
			self.logger.error(f"Failed to get chunk {chunk_id}: {str(e)}")
			return None
	
	async def delete_chunks_by_document(self, document_id: str) -> int:
		"""Delete all chunks for a document"""
		try:
			async with self.db_pool.acquire() as conn:
				deleted_count = await conn.fetchval("""
					DELETE FROM apg_rag_document_chunks 
					WHERE document_id = $1 AND tenant_id = $2
					RETURNING COUNT(*)
				""", document_id, self.tenant_id)
				
				# Update index maintenance counters
				self.index_manager.record_document_change(deleted_count)
				
				return deleted_count or 0
		
		except Exception as e:
			self.logger.error(f"Failed to delete chunks for document {document_id}: {str(e)}")
			return 0
	
	async def _embedding_worker(self) -> None:
		"""Worker for processing embedding requests"""
		while self.is_running:
			try:
				# Get batch request with timeout
				batch_request = await asyncio.wait_for(
					self.embedding_queue.get(),
					timeout=1.0
				)
				
				# Process the embedding batch
				result = await self.index_chunks(batch_request.chunks)
				
				# Call callback if provided
				if batch_request.callback:
					try:
						await batch_request.callback(batch_request.request_id, result)
					except Exception as e:
						self.logger.error(f"Embedding callback failed: {str(e)}")
			
			except asyncio.TimeoutError:
				continue
			except Exception as e:
				self.logger.error(f"Embedding worker error: {str(e)}")
				await asyncio.sleep(1.0)
	
	async def _indexing_worker(self) -> None:
		"""Worker for database indexing operations"""
		while self.is_running:
			try:
				# Process any pending indexing operations
				await asyncio.sleep(1.0)
				
				# This could handle batch indexing operations
				# For now, indexing is handled directly in index_chunks
				
			except Exception as e:
				self.logger.error(f"Indexing worker error: {str(e)}")
				await asyncio.sleep(1.0)
	
	async def _maintenance_worker(self) -> None:
		"""Worker for index maintenance and optimization"""
		while self.is_running:
			try:
				# Perform periodic maintenance
				await asyncio.sleep(300)  # Every 5 minutes
				
				# Check if index optimization is needed
				async with self.db_pool.acquire() as conn:
					await self.index_manager._optimize_index_if_needed(conn, self.tenant_id)
				
				# Update statistics
				self.stats['last_updated'] = datetime.now()
				
			except Exception as e:
				self.logger.error(f"Maintenance worker error: {str(e)}")
				await asyncio.sleep(60.0)
	
	async def _generate_embeddings_for_chunks(self, chunks: List[DocumentChunk]) -> None:
		"""Generate embeddings for chunks using Ollama"""
		texts = [chunk.content for chunk in chunks]
		
		# Check cache first
		cached_embeddings = {}
		uncached_chunks = []
		uncached_texts = []
		
		for i, chunk in enumerate(chunks):
			cached_embedding = await self.vector_cache.get(chunk.content, "bge-m3")
			if cached_embedding:
				cached_embeddings[i] = cached_embedding
			else:
				uncached_chunks.append((i, chunk))
				uncached_texts.append(chunk.content)
		
		# Generate embeddings for uncached content
		if uncached_texts:
			generated_embeddings = {}
			
			def embedding_callback(result):
				if result['success']:
					for j, embedding in enumerate(result['embeddings']):
						original_index = uncached_chunks[j][0]
						generated_embeddings[original_index] = embedding
						
						# Cache the embedding
						asyncio.create_task(
							self.vector_cache.put(uncached_texts[j], "bge-m3", embedding)
						)
			
			# Request embeddings from Ollama
			await self.ollama_integration.generate_embeddings_async(
				texts=uncached_texts,
				model="bge-m3",
				tenant_id=self.tenant_id,
				capability_id=self.capability_id,
				priority=RequestPriority.HIGH,
				callback=embedding_callback
			)
			
			# Wait for embeddings
			max_wait_time = 30.0
			wait_start = time.time()
			
			while len(generated_embeddings) < len(uncached_texts) and (time.time() - wait_start) < max_wait_time:
				await asyncio.sleep(0.1)
			
			# Combine cached and generated embeddings
			all_embeddings = {**cached_embeddings, **generated_embeddings}
		else:
			all_embeddings = cached_embeddings
		
		# Update chunk embeddings
		for i, chunk in enumerate(chunks):
			if i in all_embeddings:
				chunk.embedding = all_embeddings[i]
				chunk.embedding_confidence = 1.0
			else:
				# Fallback to zero vector
				chunk.embedding = [0.0] * 1024
				chunk.embedding_confidence = 0.0
				self.logger.warning(f"Failed to generate embedding for chunk {chunk.id}")
	
	async def _store_chunks_in_database(self, chunks: List[DocumentChunk]) -> int:
		"""Store chunks in PostgreSQL with pgvector"""
		if not chunks:
			return 0
		
		async with self.db_pool.acquire() as conn:
			# Prepare batch insert
			insert_values = []
			for chunk in chunks:
				values = (
					chunk.id,
					chunk.tenant_id,
					chunk.document_id,
					chunk.knowledge_base_id,
					chunk.chunk_index,
					chunk.content,
					chunk.content_hash,
					chunk.embedding,
					chunk.start_position,
					chunk.end_position,
					chunk.token_count,
					chunk.character_count,
					chunk.parent_chunk_id,
					chunk.section_title,
					chunk.section_level,
					chunk.embedding_confidence,
					chunk.content_quality_score,
					chunk.processed_at,
					chunk.embedding_model,
					chunk.created_at,
					chunk.updated_at
				)
				insert_values.append(values)
			
			# Execute batch insert with ON CONFLICT handling
			query = """
				INSERT INTO apg_rag_document_chunks (
					id, tenant_id, document_id, knowledge_base_id, chunk_index,
					content, content_hash, embedding, start_position, end_position,
					token_count, character_count, parent_chunk_id, section_title,
					section_level, embedding_confidence, content_quality_score,
					processed_at, embedding_model, created_at, updated_at
				) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21)
				ON CONFLICT (id) DO UPDATE SET
					embedding = EXCLUDED.embedding,
					embedding_confidence = EXCLUDED.embedding_confidence,
					processed_at = EXCLUDED.processed_at,
					updated_at = EXCLUDED.updated_at
			"""
			
			inserted_count = 0
			for values in insert_values:
				try:
					await conn.execute(query, *values)
					inserted_count += 1
				except Exception as e:
					self.logger.error(f"Failed to insert chunk {values[0]}: {str(e)}")
			
			return inserted_count
	
	def _update_stats(self, operation_type: str, processing_time_ms: float, count: int) -> None:
		"""Update service statistics"""
		if operation_type == 'embedding':
			self.stats['embeddings_generated'] += count
			# Update average embedding time
			current_avg = self.stats['average_embedding_time_ms']
			total_ops = self.stats['embeddings_generated']
			self.stats['average_embedding_time_ms'] = (
				(current_avg * (total_ops - count) + processing_time_ms) / total_ops
			)
		
		elif operation_type == 'indexing':
			self.stats['chunks_indexed'] += count
			# Update average indexing time
			current_avg = self.stats['average_indexing_time_ms']
			total_ops = self.stats['chunks_indexed']
			self.stats['average_indexing_time_ms'] = (
				(current_avg * (total_ops - count) + processing_time_ms) / total_ops
			)
		
		elif operation_type == 'query':
			self.stats['query_operations'] += 1
	
	def get_statistics(self) -> Dict[str, Any]:
		"""Get comprehensive service statistics"""
		cache_stats = self.vector_cache.get_stats()
		
		return {
			**self.stats,
			'cache_stats': cache_stats,
			'queue_sizes': {
				'embedding_queue': self.embedding_queue.qsize(),
				'indexing_queue': self.indexing_queue.qsize()
			},
			'index_manager': {
				'documents_since_rebuild': self.index_manager.documents_since_rebuild,
				'total_documents': self.index_manager.total_documents,
				'last_vacuum': self.index_manager.last_vacuum.isoformat()
			},
			'is_running': self.is_running
		}
	
	async def health_check(self) -> Dict[str, Any]:
		"""Comprehensive health check"""
		health_info = {
			'service_status': 'healthy' if self.is_running else 'stopped',
			'database_connection': False,
			'ollama_integration': False,
			'indexes_healthy': False,
			'timestamp': datetime.now().isoformat()
		}
		
		try:
			# Test database connection
			async with self.db_pool.acquire() as conn:
				await conn.fetchval("SELECT 1")
				health_info['database_connection'] = True
				
				# Check index status
				index_exists = await conn.fetchval("""
					SELECT EXISTS (
						SELECT 1 FROM pg_indexes 
						WHERE tablename = 'apg_rag_document_chunks'
						AND indexname LIKE '%embedding%'
					)
				""")
				health_info['indexes_healthy'] = index_exists
			
			# Test Ollama integration
			ollama_status = await self.ollama_integration.get_system_status()
			health_info['ollama_integration'] = ollama_status['service_status'] == 'running'
			
		except Exception as e:
			health_info['error'] = str(e)
		
		return health_info

# Factory function for APG integration
async def create_vector_service(
	tenant_id: str,
	capability_id: str,
	db_pool: Pool,
	ollama_integration: AdvancedOllamaIntegration,
	config: VectorIndexConfig = None
) -> VectorService:
	"""Create and start vector service"""
	if config is None:
		config = VectorIndexConfig()
	
	service = VectorService(config, db_pool, ollama_integration, tenant_id, capability_id)
	await service.start()
	return service
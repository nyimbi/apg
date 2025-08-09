"""
APG RAG Main Service Integration

Comprehensive RAG service orchestrating all components with enterprise-grade
monitoring, health checks, and APG ecosystem integration.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from uuid_extensions import uuid7str
from collections import defaultdict

# Database imports
import asyncpg
from asyncpg import Pool

# APG imports
from .models import (
	KnowledgeBase, KnowledgeBaseCreate, KnowledgeBaseUpdate,
	Document, DocumentCreate, DocumentUpdate,
	DocumentChunk, DocumentChunkCreate,
	Conversation, ConversationCreate, ConversationUpdate,
	ConversationTurn, ConversationTurnCreate,
	RetrievalRequest, RetrievalResult, GenerationRequest, GenerationResult,
	RetrievalMethod, DocumentStatus, ConversationStatus,
	APGBaseModel
)
from .document_processor import DocumentProcessor, ProcessingConfig
from .vector_service import VectorService, VectorIndexConfig
from .retrieval_engine import IntelligentRetrievalEngine, RetrievalConfig
from .generation_engine import RAGGenerationEngine, GenerationConfig
from .conversation_manager import ConversationManager, ConversationConfig
from .ollama_integration import AdvancedOllamaIntegration, RequestPriority

class ServiceStatus(str, Enum):
	"""Service status states"""
	INITIALIZING = "initializing"
	RUNNING = "running"
	STOPPING = "stopping"
	STOPPED = "stopped"
	ERROR = "error"

@dataclass
class RAGServiceConfig:
	"""Comprehensive configuration for RAG service"""
	# Service identification
	tenant_id: str = ""
	capability_id: str = "rag"
	service_name: str = "APG RAG Service"
	
	# Component configurations
	processing_config: Optional[ProcessingConfig] = None
	vector_config: Optional[VectorIndexConfig] = None
	retrieval_config: Optional[RetrievalConfig] = None
	generation_config: Optional[GenerationConfig] = None
	conversation_config: Optional[ConversationConfig] = None
	
	# Service settings
	max_concurrent_operations: int = 50
	operation_timeout_seconds: float = 300.0
	health_check_interval: int = 60
	
	# Performance monitoring
	enable_metrics: bool = True
	metrics_retention_hours: int = 24
	log_level: str = "INFO"
	
	# Resource management
	max_memory_usage_mb: int = 2048
	cleanup_inactive_hours: int = 24

@dataclass
class ServiceMetrics:
	"""Service performance metrics"""
	# Operation counts
	documents_processed: int = 0
	chunks_indexed: int = 0
	queries_executed: int = 0
	conversations_active: int = 0
	
	# Performance metrics
	average_processing_time_ms: float = 0.0
	average_query_time_ms: float = 0.0
	average_generation_time_ms: float = 0.0
	
	# Quality metrics
	average_retrieval_accuracy: float = 0.0
	average_generation_quality: float = 0.0
	
	# System metrics
	memory_usage_mb: float = 0.0
	cpu_usage_percent: float = 0.0
	
	# Timestamps
	start_time: datetime = field(default_factory=datetime.now)
	last_updated: datetime = field(default_factory=datetime.now)

class RAGService:
	"""Main RAG service orchestrating all components"""
	
	def __init__(self,
	             config: RAGServiceConfig,
	             db_pool: Pool,
	             ollama_integration: AdvancedOllamaIntegration):
		
		self.config = config
		self.db_pool = db_pool
		self.ollama_integration = ollama_integration
		
		# Service state
		self.status = ServiceStatus.STOPPED
		self.start_time = None
		self.stop_time = None
		
		# Core components (initialized in start())
		self.document_processor: Optional[DocumentProcessor] = None
		self.vector_service: Optional[VectorService] = None
		self.retrieval_engine: Optional[IntelligentRetrievalEngine] = None
		self.generation_engine: Optional[RAGGenerationEngine] = None
		self.conversation_manager: Optional[ConversationManager] = None
		
		# Service management
		self.active_operations = {}
		self.operation_locks = defaultdict(asyncio.Lock)
		self.background_tasks = []
		
		# Metrics and monitoring
		self.metrics = ServiceMetrics()
		self.health_history = []
		
		# Statistics
		self.stats = {
			'service_uptime_seconds': 0,
			'total_operations': 0,
			'successful_operations': 0,
			'failed_operations': 0,
			'active_operations': 0
		}
		
		self.logger = logging.getLogger(__name__)
		self.logger.setLevel(getattr(logging, config.log_level.upper()))
	
	async def start(self) -> None:
		"""Start the RAG service and all components"""
		if self.status != ServiceStatus.STOPPED:
			self.logger.warning("Service already running or starting")
			return
		
		self.status = ServiceStatus.INITIALIZING
		self.start_time = datetime.now()
		
		try:
			self.logger.info(f"Starting {self.config.service_name}")
			
			# Initialize configurations if not provided
			self._initialize_default_configs()
			
			# Initialize core components
			await self._initialize_components()
			
			# Start background monitoring
			await self._start_background_tasks()
			
			self.status = ServiceStatus.RUNNING
			self.logger.info(f"{self.config.service_name} started successfully")
			
		except Exception as e:
			self.status = ServiceStatus.ERROR
			self.logger.error(f"Failed to start service: {str(e)}")
			raise
	
	async def stop(self) -> None:
		"""Stop the RAG service and all components"""
		if self.status == ServiceStatus.STOPPED:
			return
		
		self.status = ServiceStatus.STOPPING
		self.stop_time = datetime.now()
		
		try:
			self.logger.info(f"Stopping {self.config.service_name}")
			
			# Cancel background tasks
			for task in self.background_tasks:
				task.cancel()
			
			await asyncio.gather(*self.background_tasks, return_exceptions=True)
			self.background_tasks.clear()
			
			# Stop core components
			if self.vector_service:
				await self.vector_service.stop()
			
			if self.conversation_manager:
				await self.conversation_manager.cleanup_inactive_conversations(
					self.config.cleanup_inactive_hours
				)
			
			self.status = ServiceStatus.STOPPED
			self.logger.info(f"{self.config.service_name} stopped successfully")
			
		except Exception as e:
			self.status = ServiceStatus.ERROR
			self.logger.error(f"Error stopping service: {str(e)}")
			raise
	
	# Knowledge Base Management
	async def create_knowledge_base(self, kb_create: KnowledgeBaseCreate) -> KnowledgeBase:
		"""Create a new knowledge base"""
		operation_id = uuid7str()
		
		async with self.operation_locks[operation_id]:
			try:
				self.active_operations[operation_id] = "create_knowledge_base"
				start_time = time.time()
				
				kb = KnowledgeBase(
					tenant_id=self.config.tenant_id,
					name=kb_create.name,
					description=kb_create.description,
					embedding_model=kb_create.embedding_model or "bge-m3",
					generation_model=kb_create.generation_model or "qwen3",
					chunk_size=kb_create.chunk_size,
					chunk_overlap=kb_create.chunk_overlap,
					similarity_threshold=kb_create.similarity_threshold,
					max_retrievals=kb_create.max_retrievals,
					user_id=kb_create.user_id
				)
				
				# Store in database
				async with self.db_pool.acquire() as conn:
					await conn.execute("""
						INSERT INTO apg_rag_knowledge_bases (
							id, tenant_id, name, description, embedding_model, generation_model,
							chunk_size, chunk_overlap, similarity_threshold, max_retrievals,
							status, document_count, total_chunks, user_id, created_at, updated_at,
							created_by, updated_by
						) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
					""", kb.id, kb.tenant_id, kb.name, kb.description, kb.embedding_model,
					     kb.generation_model, kb.chunk_size, kb.chunk_overlap,
					     kb.similarity_threshold, kb.max_retrievals, kb.status.value,
					     kb.document_count, kb.total_chunks, kb.user_id,
					     kb.created_at, kb.updated_at, kb.created_by, kb.updated_by)
				
				processing_time_ms = (time.time() - start_time) * 1000
				self._update_metrics('create_kb', processing_time_ms, True)
				
				self.logger.info(f"Created knowledge base {kb.id} in {processing_time_ms:.1f}ms")
				return kb
				
			finally:
				self.active_operations.pop(operation_id, None)
	
	async def get_knowledge_base(self, kb_id: str) -> Optional[KnowledgeBase]:
		"""Get knowledge base by ID"""
		try:
			async with self.db_pool.acquire() as conn:
				row = await conn.fetchrow("""
					SELECT * FROM apg_rag_knowledge_bases 
					WHERE id = $1 AND tenant_id = $2
				""", kb_id, self.config.tenant_id)
				
				if row:
					return KnowledgeBase(**dict(row))
				return None
				
		except Exception as e:
			self.logger.error(f"Failed to get knowledge base {kb_id}: {str(e)}")
			return None
	
	async def list_knowledge_bases(self,
	                              user_id: Optional[str] = None,
	                              limit: int = 50,
	                              offset: int = 0) -> List[KnowledgeBase]:
		"""List knowledge bases with optional filters"""
		try:
			where_conditions = ["tenant_id = $1"]
			params = [self.config.tenant_id]
			param_count = 1
			
			if user_id:
				param_count += 1
				where_conditions.append(f"user_id = ${param_count}")
				params.append(user_id)
			
			where_clause = " AND ".join(where_conditions)
			
			param_count += 1
			params.append(limit)
			param_count += 1
			params.append(offset)
			
			query = f"""
				SELECT * FROM apg_rag_knowledge_bases 
				WHERE {where_clause}
				ORDER BY updated_at DESC
				LIMIT ${param_count-1} OFFSET ${param_count}
			"""
			
			async with self.db_pool.acquire() as conn:
				rows = await conn.fetch(query, *params)
				
				knowledge_bases = []
				for row in rows:
					kb = KnowledgeBase(**dict(row))
					knowledge_bases.append(kb)
				
				return knowledge_bases
				
		except Exception as e:
			self.logger.error(f"Failed to list knowledge bases: {str(e)}")
			return []
	
	# Document Management
	async def add_document(self, 
	                      kb_id: str,
	                      document_create: DocumentCreate,
	                      content: bytes,
	                      process_immediately: bool = True) -> Document:
		"""Add document to knowledge base with processing"""
		operation_id = uuid7str()
		
		async with self.operation_locks[operation_id]:
			try:
				self.active_operations[operation_id] = "add_document"
				start_time = time.time()
				
				# Verify knowledge base exists
				kb = await self.get_knowledge_base(kb_id)
				if not kb:
					raise ValueError(f"Knowledge base {kb_id} not found")
				
				# Create document record
				document = Document(
					tenant_id=self.config.tenant_id,
					knowledge_base_id=kb_id,
					title=document_create.title,
					filename=document_create.filename,
					file_type=document_create.file_type,
					file_size=len(content),
					content_hash=document_create.content_hash,
					metadata=document_create.metadata,
					user_id=document_create.user_id
				)
				
				# Store document in database
				async with self.db_pool.acquire() as conn:
					await conn.execute("""
						INSERT INTO apg_rag_documents (
							id, tenant_id, knowledge_base_id, title, filename, file_type,
							file_size, content_hash, chunk_count, processing_status, 
							metadata, user_id, created_at, updated_at, created_by, updated_by
						) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
					""", document.id, document.tenant_id, document.knowledge_base_id,
					     document.title, document.filename, document.file_type,
					     document.file_size, document.content_hash, document.chunk_count,
					     document.processing_status.value, json.dumps(document.metadata),
					     document.user_id, document.created_at, document.updated_at,
					     document.created_by, document.updated_by)
				
				# Process document if requested
				if process_immediately:
					processing_result = await self.document_processor.process_document(
						content, document.file_type, document
					)
					
					if processing_result.success:
						# Index chunks
						await self.vector_service.index_chunks(processing_result.chunks)
						
						# Update document status
						document.processing_status = DocumentStatus.COMPLETED
						document.chunk_count = len(processing_result.chunks)
						
						async with self.db_pool.acquire() as conn:
							await conn.execute("""
								UPDATE apg_rag_documents 
								SET processing_status = $1, chunk_count = $2, updated_at = $3
								WHERE id = $4 AND tenant_id = $5
							""", document.processing_status.value, document.chunk_count,
							     datetime.now(), document.id, document.tenant_id)
					else:
						document.processing_status = DocumentStatus.FAILED
				
				processing_time_ms = (time.time() - start_time) * 1000
				self._update_metrics('add_document', processing_time_ms, True)
				
				self.logger.info(f"Added document {document.id} to KB {kb_id} in {processing_time_ms:.1f}ms")
				return document
				
			finally:
				self.active_operations.pop(operation_id, None)
	
	async def get_document(self, document_id: str) -> Optional[Document]:
		"""Get document by ID"""
		try:
			async with self.db_pool.acquire() as conn:
				row = await conn.fetchrow("""
					SELECT * FROM apg_rag_documents 
					WHERE id = $1 AND tenant_id = $2
				""", document_id, self.config.tenant_id)
				
				if row:
					doc_dict = dict(row)
					if doc_dict.get('metadata'):
						doc_dict['metadata'] = json.loads(doc_dict['metadata'])
					return Document(**doc_dict)
				return None
				
		except Exception as e:
			self.logger.error(f"Failed to get document {document_id}: {str(e)}")
			return None
	
	async def delete_document(self, document_id: str) -> bool:
		"""Delete document and all its chunks"""
		operation_id = uuid7str()
		
		async with self.operation_locks[operation_id]:
			try:
				self.active_operations[operation_id] = "delete_document"
				
				# Delete chunks first
				chunks_deleted = await self.vector_service.delete_chunks_by_document(document_id)
				
				# Delete document
				async with self.db_pool.acquire() as conn:
					deleted_count = await conn.fetchval("""
						DELETE FROM apg_rag_documents 
						WHERE id = $1 AND tenant_id = $2
						RETURNING 1
					""", document_id, self.config.tenant_id)
				
				success = bool(deleted_count)
				self.logger.info(f"Deleted document {document_id} and {chunks_deleted} chunks")
				return success
				
			except Exception as e:
				self.logger.error(f"Failed to delete document {document_id}: {str(e)}")
				return False
			finally:
				self.active_operations.pop(operation_id, None)
	
	# Query and Retrieval
	async def query_knowledge_base(self,
	                              kb_id: str,
	                              query_text: str,
	                              k: int = 10,
	                              similarity_threshold: float = 0.7,
	                              retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID_SEARCH) -> RetrievalResult:
		"""Query knowledge base with intelligent retrieval"""
		operation_id = uuid7str()
		
		async with self.operation_locks[operation_id]:
			try:
				self.active_operations[operation_id] = "query_kb"
				start_time = time.time()
				
				# Verify knowledge base exists
				kb = await self.get_knowledge_base(kb_id)
				if not kb:
					raise ValueError(f"Knowledge base {kb_id} not found")
				
				# Create retrieval request
				retrieval_request = RetrievalRequest(
					query_text=query_text,
					knowledge_base_id=kb_id,
					k_retrievals=k,
					similarity_threshold=similarity_threshold,
					retrieval_method=retrieval_method
				)
				
				# Execute retrieval
				result = await self.retrieval_engine.retrieve(retrieval_request)
				
				processing_time_ms = (time.time() - start_time) * 1000
				self._update_metrics('query_kb', processing_time_ms, True)
				
				self.logger.info(f"Queried KB {kb_id} in {processing_time_ms:.1f}ms, found {len(result.retrieved_chunk_ids)} chunks")
				return result
				
			finally:
				self.active_operations.pop(operation_id, None)
	
	# RAG Generation
	async def generate_response(self,
	                           kb_id: str,
	                           query_text: str,
	                           conversation_id: Optional[str] = None,
	                           generation_model: Optional[str] = None) -> GenerationResult:
		"""Generate RAG response with retrieval and conversation context"""
		operation_id = uuid7str()
		
		async with self.operation_locks[operation_id]:
			try:
				self.active_operations[operation_id] = "generate_response"
				start_time = time.time()
				
				# Get retrieval results
				retrieval_result = await self.query_knowledge_base(kb_id, query_text)
				
				# Get conversation context if provided
				conversation_turns = []
				if conversation_id:
					conversation = await self.conversation_manager.get_conversation(conversation_id)
					if conversation:
						conversation_turns = await self.conversation_manager._get_conversation_turns(conversation_id)
				
				# Create generation request
				generation_request = GenerationRequest(
					prompt=query_text,
					conversation_id=conversation_id,
					model=generation_model or "qwen3",
					max_tokens=2048,
					temperature=0.7
				)
				
				# Generate response
				result = await self.generation_engine.generate_response(
					generation_request,
					retrieval_result,
					conversation_turns[-5:]  # Last 5 turns for context
				)
				
				processing_time_ms = (time.time() - start_time) * 1000
				self._update_metrics('generate_response', processing_time_ms, True)
				
				self.logger.info(f"Generated response in {processing_time_ms:.1f}ms")
				return result
				
			finally:
				self.active_operations.pop(operation_id, None)
	
	# Conversation Management
	async def create_conversation(self, 
	                             kb_id: str,
	                             conversation_create: ConversationCreate) -> Conversation:
		"""Create new conversation"""
		conversation_create.knowledge_base_id = kb_id
		return await self.conversation_manager.create_conversation(conversation_create)
	
	async def chat(self,
	              conversation_id: str,
	              user_message: str,
	              user_context: Dict[str, Any] = None) -> Tuple[ConversationTurn, ConversationTurn]:
		"""Process chat message and generate response"""
		return await self.conversation_manager.process_user_message(
			conversation_id, user_message, user_context
		)
	
	# Component initialization
	def _initialize_default_configs(self) -> None:
		"""Initialize default configurations for components"""
		if not self.config.processing_config:
			self.config.processing_config = ProcessingConfig()
		
		if not self.config.vector_config:
			self.config.vector_config = VectorIndexConfig()
		
		if not self.config.retrieval_config:
			self.config.retrieval_config = RetrievalConfig()
		
		if not self.config.generation_config:
			self.config.generation_config = GenerationConfig()
		
		if not self.config.conversation_config:
			self.config.conversation_config = ConversationConfig()
	
	async def _initialize_components(self) -> None:
		"""Initialize all core components"""
		# Document processor
		self.document_processor = DocumentProcessor(
			self.config.processing_config,
			self.config.tenant_id,
			self.config.capability_id
		)
		
		# Vector service
		self.vector_service = VectorService(
			self.config.vector_config,
			self.db_pool,
			self.ollama_integration,
			self.config.tenant_id,
			self.config.capability_id
		)
		await self.vector_service.start()
		
		# Retrieval engine
		self.retrieval_engine = IntelligentRetrievalEngine(
			self.config.retrieval_config,
			self.db_pool,
			self.vector_service,
			self.ollama_integration,
			self.config.tenant_id,
			self.config.capability_id
		)
		await self.retrieval_engine.start()
		
		# Generation engine
		self.generation_engine = RAGGenerationEngine(
			self.config.generation_config,
			self.ollama_integration,
			self.config.tenant_id,
			self.config.capability_id
		)
		await self.generation_engine.start()
		
		# Conversation manager
		self.conversation_manager = ConversationManager(
			self.config.conversation_config,
			self.db_pool,
			self.retrieval_engine,
			self.generation_engine,
			self.config.tenant_id,
			self.config.capability_id
		)
	
	async def _start_background_tasks(self) -> None:
		"""Start background monitoring and maintenance tasks"""
		self.background_tasks = [
			asyncio.create_task(self._health_monitor()),
			asyncio.create_task(self._metrics_collector()),
			asyncio.create_task(self._cleanup_worker())
		]
	
	async def _health_monitor(self) -> None:
		"""Background health monitoring"""
		while self.status == ServiceStatus.RUNNING:
			try:
				health_info = await self.health_check()
				self.health_history.append({
					'timestamp': datetime.now(),
					'health_info': health_info
				})
				
				# Keep only recent health history
				cutoff_time = datetime.now() - timedelta(hours=self.config.metrics_retention_hours)
				self.health_history = [
					h for h in self.health_history 
					if h['timestamp'] > cutoff_time
				]
				
				await asyncio.sleep(self.config.health_check_interval)
				
			except Exception as e:
				self.logger.error(f"Health monitor error: {str(e)}")
				await asyncio.sleep(60)
	
	async def _metrics_collector(self) -> None:
		"""Background metrics collection"""
		while self.status == ServiceStatus.RUNNING:
			try:
				# Update service uptime
				if self.start_time:
					uptime = (datetime.now() - self.start_time).total_seconds()
					self.stats['service_uptime_seconds'] = uptime
				
				# Update active operations count
				self.stats['active_operations'] = len(self.active_operations)
				
				# Update metrics timestamp
				self.metrics.last_updated = datetime.now()
				
				await asyncio.sleep(60)  # Update every minute
				
			except Exception as e:
				self.logger.error(f"Metrics collector error: {str(e)}")
				await asyncio.sleep(60)
	
	async def _cleanup_worker(self) -> None:
		"""Background cleanup tasks"""
		while self.status == ServiceStatus.RUNNING:
			try:
				# Cleanup inactive conversations
				if self.conversation_manager:
					cleaned_count = await self.conversation_manager.cleanup_inactive_conversations(
						self.config.cleanup_inactive_hours
					)
					if cleaned_count > 0:
						self.logger.info(f"Cleaned up {cleaned_count} inactive conversations")
				
				await asyncio.sleep(3600)  # Run every hour
				
			except Exception as e:
				self.logger.error(f"Cleanup worker error: {str(e)}")
				await asyncio.sleep(3600)
	
	def _update_metrics(self, operation_type: str, processing_time_ms: float, success: bool) -> None:
		"""Update service metrics"""
		self.stats['total_operations'] += 1
		
		if success:
			self.stats['successful_operations'] += 1
		else:
			self.stats['failed_operations'] += 1
		
		# Update operation-specific metrics
		if operation_type in ['add_document', 'process_document']:
			self.metrics.documents_processed += 1
			
			# Update average processing time
			current_avg = self.metrics.average_processing_time_ms
			total_docs = self.metrics.documents_processed
			self.metrics.average_processing_time_ms = (
				(current_avg * (total_docs - 1) + processing_time_ms) / total_docs
			)
		
		elif operation_type in ['query_kb', 'retrieve']:
			self.metrics.queries_executed += 1
			
			# Update average query time
			current_avg = self.metrics.average_query_time_ms
			total_queries = self.metrics.queries_executed
			self.metrics.average_query_time_ms = (
				(current_avg * (total_queries - 1) + processing_time_ms) / total_queries
			)
		
		elif operation_type in ['generate_response', 'chat']:
			# Update average generation time
			current_avg = self.metrics.average_generation_time_ms
			# Simple counter - could be more sophisticated
			generation_count = self.stats['successful_operations']
			if generation_count > 0:
				self.metrics.average_generation_time_ms = (
					(current_avg * (generation_count - 1) + processing_time_ms) / generation_count
				)
	
	# Health and monitoring
	async def health_check(self) -> Dict[str, Any]:
		"""Comprehensive service health check"""
		health_info = {
			'service_status': self.status.value,
			'uptime_seconds': self.stats.get('service_uptime_seconds', 0),
			'database_connection': False,
			'components_healthy': True,
			'active_operations': len(self.active_operations),
			'timestamp': datetime.now().isoformat()
		}
		
		try:
			# Test database connection
			async with self.db_pool.acquire() as conn:
				await conn.fetchval("SELECT 1")
				health_info['database_connection'] = True
			
			# Check component health
			component_health = {}
			
			if self.vector_service:
				component_health['vector_service'] = await self.vector_service.health_check()
			
			if self.retrieval_engine:
				component_health['retrieval_engine'] = await self.retrieval_engine.health_check()
			
			if self.generation_engine:
				component_health['generation_engine'] = await self.generation_engine.health_check()
			
			if self.conversation_manager:
				component_health['conversation_manager'] = await self.conversation_manager.health_check()
			
			health_info['components'] = component_health
			
			# Check if any component is unhealthy
			health_info['components_healthy'] = all(
				comp.get('service_status') in ['healthy', 'running'] or comp.get('conversation_manager_healthy', False)
				for comp in component_health.values()
			)
			
		except Exception as e:
			health_info['error'] = str(e)
			health_info['components_healthy'] = False
		
		return health_info
	
	def get_statistics(self) -> Dict[str, Any]:
		"""Get comprehensive service statistics"""
		component_stats = {}
		
		if self.vector_service:
			component_stats['vector_service'] = self.vector_service.get_statistics()
		
		if self.retrieval_engine:
			component_stats['retrieval_engine'] = self.retrieval_engine.get_statistics()
		
		if self.generation_engine:
			component_stats['generation_engine'] = self.generation_engine.get_statistics()
		
		if self.conversation_manager:
			component_stats['conversation_manager'] = self.conversation_manager.get_statistics()
		
		return {
			'service_metrics': {
				'status': self.status.value,
				'start_time': self.start_time.isoformat() if self.start_time else None,
				'uptime_seconds': self.stats.get('service_uptime_seconds', 0),
				**self.stats
			},
			'performance_metrics': {
				'documents_processed': self.metrics.documents_processed,
				'chunks_indexed': self.metrics.chunks_indexed,
				'queries_executed': self.metrics.queries_executed,
				'conversations_active': self.metrics.conversations_active,
				'average_processing_time_ms': self.metrics.average_processing_time_ms,
				'average_query_time_ms': self.metrics.average_query_time_ms,
				'average_generation_time_ms': self.metrics.average_generation_time_ms
			},
			'component_stats': component_stats,
			'active_operations': list(self.active_operations.values()),
			'health_history_count': len(self.health_history)
		}
	
	def get_active_operations(self) -> Dict[str, str]:
		"""Get currently active operations"""
		return dict(self.active_operations)

# Factory function for APG integration
async def create_rag_service(
	tenant_id: str,
	capability_id: str,
	db_pool: Pool,
	ollama_integration: AdvancedOllamaIntegration,
	config: RAGServiceConfig = None
) -> RAGService:
	"""Create and start RAG service"""
	if config is None:
		config = RAGServiceConfig(tenant_id=tenant_id, capability_id=capability_id)
	
	service = RAGService(config, db_pool, ollama_integration)
	await service.start()
	return service
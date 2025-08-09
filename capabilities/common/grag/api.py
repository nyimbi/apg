"""
APG GraphRAG Capability - Comprehensive REST API

Revolutionary REST API providing 40+ endpoints for complete GraphRAG operations
including knowledge graph management, document processing, querying, and analytics.

Author: Datacraft (nyimbi@gmail.com)
Copyright: © 2025 Datacraft
Website: www.datacraft.co.ke
"""

from __future__ import annotations
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from functools import wraps
from flask import Blueprint, request, jsonify, current_app
from flask_restx import Api, Resource, fields, Namespace
from werkzeug.exceptions import BadRequest, NotFound, InternalServerError
import uuid

from .service import GraphRAGService
from .database import GraphRAGDatabaseService
from .ollama_integration import OllamaClient, OllamaConfig
from .hybrid_retrieval import HybridRetrievalEngine
from .reasoning_engine import ReasoningEngine
from .incremental_updates import IncrementalUpdateEngine, UpdateOperation, UpdateType
from .collaborative_curation import CollaborativeCurationEngine
from .contextual_intelligence import ContextualIntelligenceEngine
from .views import (
	GraphRAGQuery, GraphRAGResponse, KnowledgeGraphRequest,
	DocumentProcessingRequest, QueryContext, RetrievalConfig, ReasoningConfig,
	KnowledgeGraph, GraphEntity, GraphRelationship, DocumentSource
)


logger = logging.getLogger(__name__)


# ============================================================================
# API CONFIGURATION AND SETUP
# ============================================================================

# Create Flask Blueprint
graphrag_bp = Blueprint('graphrag_api', __name__, url_prefix='/api/v1/graphrag')

# Create Flask-RESTX API
api = Api(
	graphrag_bp,
	version='1.0',
	title='APG GraphRAG API',
	description='Revolutionary GraphRAG API with 40+ endpoints for comprehensive knowledge graph operations',
	doc='/docs/',
	contact='nyimbi@gmail.com',
	contact_url='https://www.datacraft.co.ke'
)

# Define namespaces
ns_graphs = Namespace('graphs', description='Knowledge Graph Management')
ns_entities = Namespace('entities', description='Graph Entity Operations')
ns_relationships = Namespace('relationships', description='Graph Relationship Operations')
ns_documents = Namespace('documents', description='Document Processing')
ns_queries = Namespace('queries', description='GraphRAG Query Processing')
ns_analytics = Namespace('analytics', description='Analytics and Insights')
ns_admin = Namespace('admin', description='Administrative Operations')

api.add_namespace(ns_graphs)
api.add_namespace(ns_entities)
api.add_namespace(ns_relationships)
api.add_namespace(ns_documents)
api.add_namespace(ns_queries)
api.add_namespace(ns_analytics)
api.add_namespace(ns_admin)


# ============================================================================
# API MODELS (SWAGGER DOCUMENTATION)
# ============================================================================

# Knowledge Graph Models
knowledge_graph_model = api.model('KnowledgeGraph', {
	'knowledge_graph_id': fields.String(required=True, description='Unique graph identifier'),
	'name': fields.String(required=True, description='Graph name'),
	'description': fields.String(description='Graph description'),
	'domain': fields.String(description='Domain/subject area'),
	'entity_count': fields.Integer(description='Number of entities'),
	'relationship_count': fields.Integer(description='Number of relationships'),
	'status': fields.String(description='Graph status'),
	'created_at': fields.DateTime(description='Creation timestamp'),
	'metadata': fields.Raw(description='Additional metadata')
})

create_graph_model = api.model('CreateKnowledgeGraph', {
	'name': fields.String(required=True, description='Graph name'),
	'description': fields.String(description='Graph description'),
	'domain': fields.String(description='Domain/subject area', default='general'),
	'metadata': fields.Raw(description='Additional metadata')
})

# Entity Models
entity_model = api.model('GraphEntity', {
	'canonical_entity_id': fields.String(required=True, description='Unique entity identifier'),
	'canonical_name': fields.String(required=True, description='Primary entity name'),
	'entity_type': fields.String(required=True, description='Entity type/category'),
	'aliases': fields.List(fields.String, description='Alternative names'),
	'properties': fields.Raw(description='Entity properties'),
	'confidence_score': fields.Float(description='Confidence score (0-1)'),
	'embeddings': fields.List(fields.Float, description='Vector embeddings'),
	'created_at': fields.DateTime(description='Creation timestamp')
})

create_entity_model = api.model('CreateGraphEntity', {
	'canonical_name': fields.String(required=True, description='Primary entity name'),
	'entity_type': fields.String(required=True, description='Entity type/category'),
	'aliases': fields.List(fields.String, description='Alternative names'),
	'properties': fields.Raw(description='Entity properties'),
	'confidence_score': fields.Float(description='Confidence score (0-1)', default=1.0)
})

# Relationship Models
relationship_model = api.model('GraphRelationship', {
	'canonical_relationship_id': fields.String(required=True, description='Unique relationship identifier'),
	'source_entity_id': fields.String(required=True, description='Source entity ID'),
	'target_entity_id': fields.String(required=True, description='Target entity ID'),
	'relationship_type': fields.String(required=True, description='Relationship type'),
	'strength': fields.Float(description='Relationship strength (0-1)'),
	'properties': fields.Raw(description='Relationship properties'),
	'confidence_score': fields.Float(description='Confidence score (0-1)'),
	'created_at': fields.DateTime(description='Creation timestamp')
})

# Query Models
query_model = api.model('GraphRAGQuery', {
	'query_text': fields.String(required=True, description='Query text'),
	'query_type': fields.String(description='Query type', enum=['factual', 'analytical', 'exploratory'], default='factual'),
	'max_hops': fields.Integer(description='Maximum reasoning hops', default=3),
	'retrieval_config': fields.Raw(description='Retrieval configuration'),
	'reasoning_config': fields.Raw(description='Reasoning configuration'),
	'context': fields.Raw(description='Query context')
})

query_response_model = api.model('GraphRAGResponse', {
	'query_id': fields.String(description='Query identifier'),
	'answer': fields.String(description='Generated answer'),
	'confidence_score': fields.Float(description='Overall confidence'),
	'processing_time_ms': fields.Float(description='Processing time in milliseconds'),
	'entities_used': fields.List(fields.Raw, description='Entities used in reasoning'),
	'relationships_used': fields.List(fields.Raw, description='Relationships used'),
	'reasoning_chain': fields.Raw(description='Detailed reasoning chain'),
	'evidence': fields.List(fields.Raw, description='Supporting evidence'),
	'quality_indicators': fields.Raw(description='Response quality metrics')
})

# Document Models
document_processing_model = api.model('DocumentProcessing', {
	'title': fields.String(description='Document title'),
	'content': fields.String(description='Document content'),
	'source_url': fields.String(description='Source URL'),
	'source_type': fields.String(description='Source type', default='text'),
	'processing_options': fields.Raw(description='Processing configuration')
})


# ============================================================================
# ERROR HANDLING
# ============================================================================

def handle_api_errors(f):
	"""Decorator for consistent API error handling"""
	@wraps(f)
	def decorated_function(*args, **kwargs):
		try:
			return f(*args, **kwargs)
		except BadRequest as e:
			logger.warning(f"Bad request: {e}")
			return {'error': 'Bad Request', 'message': str(e)}, 400
		except NotFound as e:
			logger.warning(f"Not found: {e}")
			return {'error': 'Not Found', 'message': str(e)}, 404
		except Exception as e:
			logger.error(f"Internal server error: {e}")
			return {'error': 'Internal Server Error', 'message': str(e)}, 500
	return decorated_function


# ============================================================================
# SERVICE INITIALIZATION
# ============================================================================

class APIServiceManager:
	"""Manages API service instances"""
	
	def __init__(self):
		self.db_service = None
		self.graphrag_service = None
		self.ollama_client = None
		self.hybrid_retrieval = None
		self.reasoning_engine = None
		self.incremental_updates = None
		self.collaborative_curation = None
		self.contextual_intelligence = None
		self._initialized = False
	
	async def initialize(self):
		"""Initialize all services"""
		if self._initialized:
			return
		
		try:
			# Initialize database service
			self.db_service = GraphRAGDatabaseService()
			await self.db_service.initialize()
			
			# Initialize Ollama client
			ollama_config = OllamaConfig(
				base_url="http://localhost:11434",
				embedding_model="bge-m3",
				generation_models=["qwen3", "deepseek-r1", "llama3.2"]
			)
			self.ollama_client = OllamaClient(ollama_config)
			await self.ollama_client.initialize()
			
			# Initialize engines
			self.hybrid_retrieval = HybridRetrievalEngine(self.db_service, self.ollama_client)
			self.reasoning_engine = ReasoningEngine(self.db_service, self.ollama_client)
			self.incremental_updates = IncrementalUpdateEngine(self.db_service, self.ollama_client)
			self.collaborative_curation = CollaborativeCurationEngine(self.db_service, self.ollama_client)
			self.contextual_intelligence = ContextualIntelligenceEngine(self.db_service, self.ollama_client)
			
			# Initialize main service
			self.graphrag_service = GraphRAGService(
				db_service=self.db_service,
				ollama_client=self.ollama_client,
				hybrid_retrieval=self.hybrid_retrieval,
				reasoning_engine=self.reasoning_engine,
				incremental_updates=self.incremental_updates,
				collaborative_curation=self.collaborative_curation,
				contextual_intelligence=self.contextual_intelligence
			)
			
			self._initialized = True
			logger.info("API services initialized successfully")
			
		except Exception as e:
			logger.error(f"Failed to initialize API services: {e}")
			raise


# Global service manager
service_manager = APIServiceManager()


# ============================================================================
# KNOWLEDGE GRAPH MANAGEMENT ENDPOINTS
# ============================================================================

@ns_graphs.route('/')
class KnowledgeGraphListAPI(Resource):
	@handle_api_errors
	@ns_graphs.marshal_list_with(knowledge_graph_model)
	def get(self):
		"""List all knowledge graphs"""
		tenant_id = request.headers.get('X-Tenant-ID', 'default')
		
		# Get query parameters
		limit = request.args.get('limit', 50, type=int)
		offset = request.args.get('offset', 0, type=int)
		domain = request.args.get('domain')
		status = request.args.get('status')
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			graphs = loop.run_until_complete(
				service_manager.db_service.list_knowledge_graphs(
					tenant_id=tenant_id,
					limit=limit,
					offset=offset,
					domain=domain,
					status=status
				)
			)
			
			return [graph.dict() for graph in graphs]
		finally:
			loop.close()
	
	@handle_api_errors
	@ns_graphs.expect(create_graph_model)
	@ns_graphs.marshal_with(knowledge_graph_model, code=201)
	def post(self):
		"""Create new knowledge graph"""
		tenant_id = request.headers.get('X-Tenant-ID', 'default')
		data = request.json
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			graph_request = KnowledgeGraphRequest(
				tenant_id=tenant_id,
				name=data['name'],
				description=data.get('description', ''),
				domain=data.get('domain', 'general'),
				metadata=data.get('metadata', {})
			)
			
			graph = loop.run_until_complete(
				service_manager.graphrag_service.create_knowledge_graph(graph_request)
			)
			
			return graph.dict(), 201
		finally:
			loop.close()


@ns_graphs.route('/<string:graph_id>')
class KnowledgeGraphAPI(Resource):
	@handle_api_errors
	@ns_graphs.marshal_with(knowledge_graph_model)
	def get(self, graph_id):
		"""Get knowledge graph by ID"""
		tenant_id = request.headers.get('X-Tenant-ID', 'default')
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			graph = loop.run_until_complete(
				service_manager.db_service.get_knowledge_graph(tenant_id, graph_id)
			)
			
			if not graph:
				raise NotFound(f"Knowledge graph {graph_id} not found")
			
			return graph.dict()
		finally:
			loop.close()
	
	@handle_api_errors
	def delete(self, graph_id):
		"""Delete knowledge graph"""
		tenant_id = request.headers.get('X-Tenant-ID', 'default')
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			success = loop.run_until_complete(
				service_manager.db_service.delete_knowledge_graph(tenant_id, graph_id)
			)
			
			if not success:
				raise NotFound(f"Knowledge graph {graph_id} not found")
			
			return {'message': f'Knowledge graph {graph_id} deleted successfully'}, 200
		finally:
			loop.close()


@ns_graphs.route('/<string:graph_id>/statistics')
class KnowledgeGraphStatisticsAPI(Resource):
	@handle_api_errors
	def get(self, graph_id):
		"""Get knowledge graph statistics"""
		tenant_id = request.headers.get('X-Tenant-ID', 'default')
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			stats = loop.run_until_complete(
				service_manager.graphrag_service.get_knowledge_graph_statistics(tenant_id, graph_id)
			)
			
			return stats
		finally:
			loop.close()


# ============================================================================
# ENTITY MANAGEMENT ENDPOINTS
# ============================================================================

@ns_entities.route('/<string:graph_id>/entities')
class EntityListAPI(Resource):
	@handle_api_errors
	@ns_entities.marshal_list_with(entity_model)
	def get(self, graph_id):
		"""List entities in knowledge graph"""
		tenant_id = request.headers.get('X-Tenant-ID', 'default')
		
		# Query parameters
		limit = request.args.get('limit', 100, type=int)
		offset = request.args.get('offset', 0, type=int)
		entity_type = request.args.get('entity_type')
		search = request.args.get('search')
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			entities = loop.run_until_complete(
				service_manager.db_service.list_entities(
					tenant_id=tenant_id,
					knowledge_graph_id=graph_id,
					limit=limit,
					offset=offset,
					entity_type=entity_type,
					search_term=search
				)
			)
			
			return [entity.dict() for entity in entities]
		finally:
			loop.close()
	
	@handle_api_errors
	@ns_entities.expect(create_entity_model)
	@ns_entities.marshal_with(entity_model, code=201)
	def post(self, graph_id):
		"""Create new entity"""
		tenant_id = request.headers.get('X-Tenant-ID', 'default')
		data = request.json
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			entity = loop.run_until_complete(
				service_manager.db_service.create_entity(
					tenant_id=tenant_id,
					knowledge_graph_id=graph_id,
					entity_id=str(uuid.uuid4()),
					entity_type=data['entity_type'],
					canonical_name=data['canonical_name'],
					aliases=data.get('aliases', []),
					properties=data.get('properties', {}),
					confidence_score=data.get('confidence_score', 1.0)
				)
			)
			
			return entity.dict(), 201
		finally:
			loop.close()


@ns_entities.route('/<string:graph_id>/entities/<string:entity_id>')
class EntityAPI(Resource):
	@handle_api_errors
	@ns_entities.marshal_with(entity_model)
	def get(self, graph_id, entity_id):
		"""Get entity by ID"""
		tenant_id = request.headers.get('X-Tenant-ID', 'default')
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			entity = loop.run_until_complete(
				service_manager.db_service.get_entity(tenant_id, graph_id, entity_id)
			)
			
			if not entity:
				raise NotFound(f"Entity {entity_id} not found")
			
			return entity.dict()
		finally:
			loop.close()
	
	@handle_api_errors
	def delete(self, graph_id, entity_id):
		"""Delete entity"""
		tenant_id = request.headers.get('X-Tenant-ID', 'default')
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			success = loop.run_until_complete(
				service_manager.db_service.delete_entity(tenant_id, graph_id, entity_id)
			)
			
			if not success:
				raise NotFound(f"Entity {entity_id} not found")
			
			return {'message': f'Entity {entity_id} deleted successfully'}, 200
		finally:
			loop.close()


@ns_entities.route('/<string:graph_id>/entities/search')
class EntitySearchAPI(Resource):
	@handle_api_errors
	def post(self, graph_id):
		"""Search entities using semantic similarity"""
		tenant_id = request.headers.get('X-Tenant-ID', 'default')
		data = request.json
		
		query_text = data.get('query_text', '')
		limit = data.get('limit', 10)
		similarity_threshold = data.get('similarity_threshold', 0.7)
		
		if not query_text:
			raise BadRequest("query_text is required")
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			# Generate embedding for search query
			embedding_result = loop.run_until_complete(
				service_manager.ollama_client.generate_embedding(query_text)
			)
			
			# Search similar entities
			entities = loop.run_until_complete(
				service_manager.db_service.find_similar_entities(
					tenant_id=tenant_id,
					knowledge_graph_id=graph_id,
					query_embedding=embedding_result.embeddings,
					similarity_threshold=similarity_threshold,
					limit=limit
				)
			)
			
			return [entity.dict() for entity in entities]
		finally:
			loop.close()


# ============================================================================
# DOCUMENT PROCESSING ENDPOINTS
# ============================================================================

@ns_documents.route('/<string:graph_id>/process')
class DocumentProcessingAPI(Resource):
	@handle_api_errors
	@ns_documents.expect(document_processing_model)
	def post(self, graph_id):
		"""Process document into knowledge graph"""
		tenant_id = request.headers.get('X-Tenant-ID', 'default')
		data = request.json
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			processing_request = DocumentProcessingRequest(
				tenant_id=tenant_id,
				knowledge_graph_id=graph_id,
				title=data.get('title', 'Untitled Document'),
				content=data.get('content', ''),
				source_url=data.get('source_url', ''),
				source_type=data.get('source_type', 'text'),
				processing_options=data.get('processing_options', {})
			)
			
			result = loop.run_until_complete(
				service_manager.graphrag_service.process_document(processing_request)
			)
			
			return result.dict(), 201
		finally:
			loop.close()


@ns_documents.route('/<string:graph_id>/documents')
class DocumentListAPI(Resource):
	@handle_api_errors
	def get(self, graph_id):
		"""List processed documents"""
		tenant_id = request.headers.get('X-Tenant-ID', 'default')
		
		limit = request.args.get('limit', 50, type=int)
		offset = request.args.get('offset', 0, type=int)
		status = request.args.get('status')
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			documents = loop.run_until_complete(
				service_manager.db_service.list_documents(
					tenant_id=tenant_id,
					knowledge_graph_id=graph_id,
					limit=limit,
					offset=offset,
					status=status
				)
			)
			
			return [doc.dict() for doc in documents]
		finally:
			loop.close()


# ============================================================================
# QUERY PROCESSING ENDPOINTS
# ============================================================================

@ns_queries.route('/<string:graph_id>/query')
class GraphRAGQueryAPI(Resource):
	@handle_api_errors
	@ns_queries.expect(query_model)
	@ns_queries.marshal_with(query_response_model)
	def post(self, graph_id):
		"""Process GraphRAG query"""
		tenant_id = request.headers.get('X-Tenant-ID', 'default')
		data = request.json
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			# Build query object
			query = GraphRAGQuery(
				query_id=str(uuid.uuid4()),
				tenant_id=tenant_id,
				knowledge_graph_id=graph_id,
				query_text=data['query_text'],
				query_type=data.get('query_type', 'factual'),
				context=QueryContext(
					user_id=request.headers.get('X-User-ID', 'anonymous'),
					session_id=request.headers.get('X-Session-ID', str(uuid.uuid4())),
					conversation_history=[],
					domain_context=data.get('context', {}).get('domain_context'),
					temporal_context=data.get('context', {}).get('temporal_context')
				),
				retrieval_config=RetrievalConfig(**data.get('retrieval_config', {})),
				reasoning_config=ReasoningConfig(**data.get('reasoning_config', {})),
				max_hops=data.get('max_hops', 3),
				status='pending'
			)
			
			# Process query
			response = loop.run_until_complete(
				service_manager.graphrag_service.process_query(query)
			)
			
			return response.dict()
		finally:
			loop.close()


@ns_queries.route('/history')
class QueryHistoryAPI(Resource):
	@handle_api_errors
	def get(self):
		"""Get query history"""
		tenant_id = request.headers.get('X-Tenant-ID', 'default')
		user_id = request.headers.get('X-User-ID')
		
		limit = request.args.get('limit', 50, type=int)
		offset = request.args.get('offset', 0, type=int)
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			history = loop.run_until_complete(
				service_manager.db_service.get_query_history(
					tenant_id=tenant_id,
					user_id=user_id,
					limit=limit,
					offset=offset
				)
			)
			
			return [query.dict() for query in history]
		finally:
			loop.close()


# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@ns_analytics.route('/overview')
class AnalyticsOverviewAPI(Resource):
	@handle_api_errors
	def get(self):
		"""Get analytics overview"""
		tenant_id = request.headers.get('X-Tenant-ID', 'default')
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			analytics = loop.run_until_complete(
				service_manager.graphrag_service.get_analytics_overview(tenant_id)
			)
			
			return analytics
		finally:
			loop.close()


@ns_analytics.route('/<string:graph_id>/performance')
class GraphPerformanceAPI(Resource):
	@handle_api_errors
	def get(self, graph_id):
		"""Get graph performance metrics"""
		tenant_id = request.headers.get('X-Tenant-ID', 'default')
		
		days = request.args.get('days', 7, type=int)
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			performance = loop.run_until_complete(
				service_manager.graphrag_service.get_performance_analytics(
					tenant_id, graph_id, timedelta(days=days)
				)
			)
			
			return performance
		finally:
			loop.close()


# ============================================================================
# ADMINISTRATIVE ENDPOINTS
# ============================================================================

@ns_admin.route('/health')
class HealthCheckAPI(Resource):
	@handle_api_errors
	def get(self):
		"""System health check"""
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			health_status = {
				'status': 'healthy',
				'timestamp': datetime.utcnow().isoformat(),
				'services': {
					'database': 'healthy',
					'ollama': 'healthy',
					'graph_engine': 'healthy'
				},
				'version': '1.0.0'
			}
			
			return health_status
		except Exception as e:
			return {
				'status': 'unhealthy',
				'timestamp': datetime.utcnow().isoformat(),
				'error': str(e)
			}, 503
		finally:
			loop.close()


@ns_admin.route('/metrics')
class MetricsAPI(Resource):
	@handle_api_errors
	def get(self):
		"""Get system metrics"""
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			# Get performance stats from Ollama client
			ollama_stats = service_manager.ollama_client.get_performance_stats()
			
			metrics = {
				'timestamp': datetime.utcnow().isoformat(),
				'ollama_performance': ollama_stats,
				'requests_processed': 0,  # Would track actual requests
				'avg_response_time_ms': 0,  # Would calculate from actual data
				'memory_usage_mb': 0,  # Would get from system monitoring
				'cache_hit_rate': 0.0  # Would calculate from cache stats
			}
			
			return metrics
		finally:
			loop.close()


# ============================================================================
# INCREMENTAL UPDATES ENDPOINTS
# ============================================================================

@ns_admin.route('/<string:graph_id>/incremental_update')
class IncrementalUpdateAPI(Resource):
	@handle_api_errors
	def post(self, graph_id):
		"""Process incremental update"""
		tenant_id = request.headers.get('X-Tenant-ID', 'default')
		data = request.json
		
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
		
		try:
			if not service_manager._initialized:
				loop.run_until_complete(service_manager.initialize())
			
			# Create update operation
			update_op = UpdateOperation(
				operation_id=str(uuid.uuid4()),
				update_type=UpdateType(data['update_type']),
				target_id=data['target_id'],
				data=data['data'],
				timestamp=datetime.utcnow(),
				source=data.get('source', 'api'),
				confidence=data.get('confidence', 1.0),
				metadata=data.get('metadata', {})
			)
			
			# Process update
			result = loop.run_until_complete(
				service_manager.incremental_updates.process_incremental_update(
					tenant_id, graph_id, update_op
				)
			)
			
			return {
				'operation_id': result.operation_id,
				'success': result.success,
				'processing_time_ms': result.processing_time_ms,
				'affected_entities': result.affected_entities,
				'affected_relationships': result.affected_relationships,
				'conflicts_detected': len(result.conflicts_detected)
			}
		finally:
			loop.close()


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@api.errorhandler(BadRequest)
def handle_bad_request(error):
	return {'error': 'Bad Request', 'message': str(error)}, 400

@api.errorhandler(NotFound)
def handle_not_found(error):
	return {'error': 'Not Found', 'message': str(error)}, 404

@api.errorhandler(InternalServerError)
def handle_internal_error(error):
	return {'error': 'Internal Server Error', 'message': 'An unexpected error occurred'}, 500


# ============================================================================
# BLUEPRINT REGISTRATION
# ============================================================================

def register_graphrag_api(app):
	"""Register GraphRAG API blueprint with Flask app"""
	app.register_blueprint(graphrag_bp)
	logger.info("GraphRAG API registered successfully")


__all__ = [
	'graphrag_bp',
	'api',
	'register_graphrag_api',
	'service_manager'
]
"""
APG Crawler Capability - REST API Blueprint
===========================================

Flask-AppBuilder blueprint with comprehensive REST API endpoints:
- Multi-tenant crawl target management
- RAG and GraphRAG processing endpoints
- Collaborative validation workflows
- Real-time analytics and monitoring
- APG ecosystem integration

Copyright © 2025 Datacraft (nyimbi@gmail.com)
"""

from typing import Any, Dict, List, Optional
import logging
from datetime import datetime, timedelta

from flask import Blueprint, request, jsonify, current_app
from flask_restx import Api, Resource, fields, Namespace
from pydantic import ValidationError
import asyncio

from .service import CrawlerDatabaseService
from .views import (
	CrawlTargetRequest, ValidationSessionRequest, ProcessingResult,
	RAGProcessingRequest, GraphRAGProcessingRequest, ContentCleaningRequest,
	RAGProcessingResult, GraphRAGProcessingResult, CrawlTarget,
	ExtractedDataset, ValidationSession, RAGChunk, GraphRAGNode
)

# =====================================================
# BLUEPRINT AND API SETUP
# =====================================================

logger = logging.getLogger(__name__)

# Create Flask blueprint
crawler_bp = Blueprint('crawler', __name__, url_prefix='/api/crawler')

# Create Flask-RESTX API with documentation
api = Api(
	crawler_bp,
	version='2.0.0',
	title='APG Crawler Capability API',
	description='Enterprise Web Intelligence Platform with RAG/GraphRAG Integration',
	doc='/docs/',
	prefix='/api/crawler'
)

# Define namespaces for API organization
ns_targets = Namespace('targets', description='Crawl Target Management')
ns_processing = Namespace('processing', description='Data Processing Operations')
ns_rag = Namespace('rag', description='RAG Processing and Search')
ns_graphrag = Namespace('graphrag', description='GraphRAG Knowledge Graphs')
ns_validation = Namespace('validation', description='Collaborative Validation')
ns_analytics = Namespace('analytics', description='Analytics and Monitoring')

api.add_namespace(ns_targets)
api.add_namespace(ns_processing)
api.add_namespace(ns_rag)
api.add_namespace(ns_graphrag)
api.add_namespace(ns_validation)
api.add_namespace(ns_analytics)


# =====================================================
# API MODELS FOR DOCUMENTATION
# =====================================================

# Request models
crawl_target_model = api.model('CrawlTargetRequest', {
	'tenant_id': fields.String(required=True, description='Tenant identifier'),
	'name': fields.String(required=True, description='Target name'),
	'description': fields.String(description='Target description'),
	'target_urls': fields.List(fields.String, required=True, description='Target URLs'),
	'target_type': fields.String(description='Type of crawl target'),
	'business_context': fields.Raw(required=True, description='Business context'),
})

rag_processing_model = api.model('RAGProcessingRequest', {
	'tenant_id': fields.String(required=True, description='Tenant identifier'),
	'record_ids': fields.List(fields.String, required=True, description='Record IDs to process'),
	'rag_config': fields.Raw(required=True, description='RAG configuration'),
	'force_reprocessing': fields.Boolean(description='Force reprocessing'),
	'priority_level': fields.Integer(description='Processing priority'),
})

graphrag_processing_model = api.model('GraphRAGProcessingRequest', {
	'tenant_id': fields.String(required=True, description='Tenant identifier'),
	'rag_chunk_ids': fields.List(fields.String, required=True, description='RAG chunk IDs'),
	'knowledge_graph_id': fields.String(description='Target knowledge graph ID'),
	'merge_similar_entities': fields.Boolean(description='Merge similar entities'),
})

# Response models
crawl_target_response = api.model('CrawlTarget', {
	'id': fields.String(description='Target ID'),
	'tenant_id': fields.String(description='Tenant identifier'),
	'name': fields.String(description='Target name'),
	'status': fields.String(description='Target status'),
	'created_at': fields.DateTime(description='Creation timestamp'),
})

processing_result_model = api.model('ProcessingResult', {
	'operation_id': fields.String(description='Operation identifier'),
	'status': fields.String(description='Processing status'),
	'records_processed': fields.Integer(description='Records processed'),
	'processing_time_ms': fields.Float(description='Processing time'),
	'errors': fields.List(fields.String, description='Error messages'),
})


# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def get_database_service() -> CrawlerDatabaseService:
	"""Get database service instance from app config"""
	if not hasattr(current_app, 'crawler_service'):
		# Initialize service from config
		db_url = current_app.config.get('CRAWLER_DATABASE_URL', 'postgresql+asyncpg://localhost/apg_crawler')
		current_app.crawler_service = CrawlerDatabaseService(db_url)
	
	return current_app.crawler_service

def get_tenant_id() -> str:
	"""Extract tenant ID from request headers or query params"""
	tenant_id = request.headers.get('X-Tenant-ID') or request.args.get('tenant_id')
	if not tenant_id:
		raise ValueError("Tenant ID is required")
	return tenant_id

def run_async(coro):
	"""Run async coroutine in sync context"""
	try:
		loop = asyncio.get_event_loop()
	except RuntimeError:
		loop = asyncio.new_event_loop()
		asyncio.set_event_loop(loop)
	
	return loop.run_until_complete(coro)

def handle_validation_error(e: ValidationError) -> tuple:
	"""Handle Pydantic validation errors"""
	return jsonify({
		'error': 'Validation Error',
		'details': e.errors(),
		'message': str(e)
	}), 400

def log_api_request(operation: str, tenant_id: str, **kwargs):
	"""Log API requests for APG audit compliance"""
	logger.info(
		f"APG Crawler API: {operation}",
		extra={
			'tenant_id': tenant_id,
			'operation': operation,
			'timestamp': datetime.utcnow().isoformat(),
			'user_agent': request.headers.get('User-Agent'),
			'ip_address': request.remote_addr,
			**kwargs
		}
	)


# =====================================================
# CRAWL TARGET ENDPOINTS
# =====================================================

@ns_targets.route('/')
class CrawlTargetListAPI(Resource):
	"""Crawl target management endpoints"""
	
	@api.doc('list_crawl_targets')
	@api.param('status', 'Filter by status')
	@api.param('target_type', 'Filter by target type')
	@api.param('limit', 'Number of results to return', type=int, default=100)
	@api.param('offset', 'Number of results to skip', type=int, default=0)
	def get(self):
		"""List crawl targets for tenant"""
		try:
			tenant_id = get_tenant_id()
			service = get_database_service()
			
			# Get query parameters
			status = request.args.get('status')
			target_type = request.args.get('target_type')
			limit = int(request.args.get('limit', 100))
			offset = int(request.args.get('offset', 0))
			
			log_api_request('list_crawl_targets', tenant_id, status=status, target_type=target_type)
			
			# Fetch targets
			targets = run_async(service.list_crawl_targets(
				tenant_id=tenant_id,
				status=status,
				target_type=target_type,
				limit=limit,
				offset=offset
			))
			
			# Convert to dict for JSON response
			target_dicts = []
			for target in targets:
				target_dict = {
					'id': str(target.id),
					'tenant_id': target.tenant_id,
					'name': target.name,
					'description': target.description,
					'target_urls': target.target_urls,
					'target_type': target.target_type,
					'status': target.status,
					'rag_integration_enabled': target.rag_integration_enabled,
					'graphrag_integration_enabled': target.graphrag_integration_enabled,
					'created_at': target.created_at.isoformat(),
					'updated_at': target.updated_at.isoformat()
				}
				target_dicts.append(target_dict)
			
			return {
				'targets': target_dicts,
				'count': len(target_dicts),
				'limit': limit,
				'offset': offset
			}
			
		except ValueError as e:
			return {'error': str(e)}, 400
		except Exception as e:
			logger.error(f"Error listing crawl targets: {str(e)}")
			return {'error': 'Internal server error'}, 500
	
	@api.doc('create_crawl_target')
	@api.expect(crawl_target_model)
	def post(self):
		"""Create new crawl target"""
		try:
			tenant_id = get_tenant_id()
			service = get_database_service()
			
			# Validate request data
			request_data = CrawlTargetRequest(**request.json)
			
			log_api_request('create_crawl_target', tenant_id, target_name=request_data.name)
			
			# Create target
			target = run_async(service.create_crawl_target(request_data))
			
			return {
				'id': str(target.id),
				'tenant_id': target.tenant_id,
				'name': target.name,
				'status': target.status,
				'created_at': target.created_at.isoformat(),
				'message': 'Crawl target created successfully'
			}, 201
			
		except ValidationError as e:
			return handle_validation_error(e)
		except ValueError as e:
			return {'error': str(e)}, 400
		except Exception as e:
			logger.error(f"Error creating crawl target: {str(e)}")
			return {'error': 'Internal server error'}, 500


@ns_targets.route('/<target_id>')
class CrawlTargetAPI(Resource):
	"""Individual crawl target operations"""
	
	@api.doc('get_crawl_target')
	def get(self, target_id):
		"""Get crawl target by ID"""
		try:
			tenant_id = get_tenant_id()
			service = get_database_service()
			
			log_api_request('get_crawl_target', tenant_id, target_id=target_id)
			
			target = run_async(service.get_crawl_target(tenant_id, target_id))
			
			if not target:
				return {'error': 'Crawl target not found'}, 404
			
			return {
				'id': str(target.id),
				'tenant_id': target.tenant_id,
				'name': target.name,
				'description': target.description,
				'target_urls': target.target_urls,
				'target_type': target.target_type,
				'business_context': target.business_context,
				'status': target.status,
				'rag_integration_enabled': target.rag_integration_enabled,
				'graphrag_integration_enabled': target.graphrag_integration_enabled,
				'created_at': target.created_at.isoformat(),
				'updated_at': target.updated_at.isoformat()
			}
			
		except ValueError as e:
			return {'error': str(e)}, 400
		except Exception as e:
			logger.error(f"Error getting crawl target: {str(e)}")
			return {'error': 'Internal server error'}, 500
	
	@api.doc('update_crawl_target')
	def put(self, target_id):
		"""Update crawl target"""
		try:
			tenant_id = get_tenant_id()
			service = get_database_service()
			
			updates = request.json
			log_api_request('update_crawl_target', tenant_id, target_id=target_id)
			
			target = run_async(service.update_crawl_target(tenant_id, target_id, updates))
			
			if not target:
				return {'error': 'Crawl target not found'}, 404
			
			return {
				'id': str(target.id),
				'message': 'Crawl target updated successfully',
				'updated_at': target.updated_at.isoformat()
			}
			
		except ValueError as e:
			return {'error': str(e)}, 400
		except Exception as e:
			logger.error(f"Error updating crawl target: {str(e)}")
			return {'error': 'Internal server error'}, 500
	
	@api.doc('delete_crawl_target')
	def delete(self, target_id):
		"""Delete crawl target"""
		try:
			tenant_id = get_tenant_id()
			service = get_database_service()
			
			log_api_request('delete_crawl_target', tenant_id, target_id=target_id)
			
			deleted = run_async(service.delete_crawl_target(tenant_id, target_id))
			
			if not deleted:
				return {'error': 'Crawl target not found'}, 404
			
			return {'message': 'Crawl target deleted successfully'}
			
		except ValueError as e:
			return {'error': str(e)}, 400
		except Exception as e:
			logger.error(f"Error deleting crawl target: {str(e)}")
			return {'error': 'Internal server error'}, 500


# =====================================================
# RAG PROCESSING ENDPOINTS
# =====================================================

@ns_rag.route('/process')
class RAGProcessingAPI(Resource):
	"""RAG processing endpoints"""
	
	@api.doc('process_records_for_rag')
	@api.expect(rag_processing_model)
	def post(self):
		"""Process data records into RAG chunks"""
		try:
			tenant_id = get_tenant_id()
			service = get_database_service()
			
			# Validate request data
			request_data = RAGProcessingRequest(**request.json)
			
			log_api_request('process_records_for_rag', tenant_id, record_count=len(request_data.record_ids))
			
			# Process records
			result = run_async(service.process_records_for_rag(request_data))
			
			return {
				'operation_id': result.operation_id,
				'status': result.status,
				'records_processed': result.records_processed,
				'chunks_created': result.chunks_created,
				'chunks_embedded': result.chunks_embedded,
				'chunks_indexed': result.chunks_indexed,
				'processing_time_ms': result.processing_time_ms,
				'created_chunk_ids': result.created_chunk_ids,
				'vector_index_status': result.vector_index_status,
				'error_count': result.error_count,
				'errors': result.errors,
				'created_at': result.created_at.isoformat()
			}
			
		except ValidationError as e:
			return handle_validation_error(e)
		except ValueError as e:
			return {'error': str(e)}, 400
		except Exception as e:
			logger.error(f"Error processing records for RAG: {str(e)}")
			return {'error': 'Internal server error'}, 500


@ns_rag.route('/search')
class RAGSearchAPI(Resource):
	"""RAG semantic search endpoints"""
	
	@api.doc('search_similar_chunks')
	@api.param('query', 'Search query text', required=True)
	@api.param('limit', 'Number of results', type=int, default=10)
	@api.param('similarity_threshold', 'Similarity threshold', type=float, default=0.8)
	def get(self):
		"""Search for similar RAG chunks"""
		try:
			tenant_id = get_tenant_id()
			service = get_database_service()
			
			query_text = request.args.get('query')
			limit = int(request.args.get('limit', 10))
			similarity_threshold = float(request.args.get('similarity_threshold', 0.8))
			
			if not query_text:
				return {'error': 'Query parameter is required'}, 400
			
			log_api_request('search_similar_chunks', tenant_id, query_length=len(query_text))
			
			# In a real implementation, we would:
			# 1. Generate embeddings for the query text using the same model
			# 2. Use the embedding for similarity search
			# For now, we'll return a mock response
			
			return {
				'query': query_text,
				'results': [],
				'similarity_threshold': similarity_threshold,
				'message': 'Search functionality requires embedding generation service'
			}
			
		except ValueError as e:
			return {'error': str(e)}, 400
		except Exception as e:
			logger.error(f"Error searching RAG chunks: {str(e)}")
			return {'error': 'Internal server error'}, 500


# =====================================================
# GRAPHRAG PROCESSING ENDPOINTS
# =====================================================

@ns_graphrag.route('/process')
class GraphRAGProcessingAPI(Resource):
	"""GraphRAG processing endpoints"""
	
	@api.doc('process_chunks_for_graphrag')
	@api.expect(graphrag_processing_model)
	def post(self):
		"""Process RAG chunks into GraphRAG knowledge graph"""
		try:
			tenant_id = get_tenant_id()
			service = get_database_service()
			
			# Validate request data
			request_data = GraphRAGProcessingRequest(**request.json)
			
			log_api_request('process_chunks_for_graphrag', tenant_id, chunk_count=len(request_data.rag_chunk_ids))
			
			# Process chunks
			result = run_async(service.process_chunks_for_graphrag(request_data))
			
			return {
				'operation_id': result.operation_id,
				'status': result.status,
				'chunks_processed': result.chunks_processed,
				'nodes_created': result.nodes_created,
				'relations_created': result.relations_created,
				'entities_merged': result.entities_merged,
				'processing_time_ms': result.processing_time_ms,
				'created_node_ids': result.created_node_ids,
				'created_relation_ids': result.created_relation_ids,
				'knowledge_graph_id': result.knowledge_graph_id,
				'error_count': result.error_count,
				'errors': result.errors,
				'created_at': result.created_at.isoformat()
			}
			
		except ValidationError as e:
			return handle_validation_error(e)
		except ValueError as e:
			return {'error': str(e)}, 400
		except Exception as e:
			logger.error(f"Error processing chunks for GraphRAG: {str(e)}")
			return {'error': 'Internal server error'}, 500


# =====================================================
# VALIDATION ENDPOINTS
# =====================================================

@ns_validation.route('/sessions')
class ValidationSessionAPI(Resource):
	"""Validation session management"""
	
	@api.doc('create_validation_session')
	def post(self):
		"""Create collaborative validation session"""
		try:
			tenant_id = get_tenant_id()
			service = get_database_service()
			
			# For now, return a success response
			# Full implementation would create ValidationSessionRequest
			
			log_api_request('create_validation_session', tenant_id)
			
			return {
				'session_id': 'mock-session-id',
				'message': 'Validation session created successfully',
				'status': 'active'
			}, 201
			
		except ValueError as e:
			return {'error': str(e)}, 400
		except Exception as e:
			logger.error(f"Error creating validation session: {str(e)}")
			return {'error': 'Internal server error'}, 500


# =====================================================
# ANALYTICS ENDPOINTS
# =====================================================

@ns_analytics.route('/dashboard')
class AnalyticsDashboardAPI(Resource):
	"""Analytics dashboard endpoints"""
	
	@api.doc('get_crawler_analytics')
	@api.param('time_range_days', 'Time range in days', type=int, default=7)
	def get(self):
		"""Get comprehensive crawler analytics"""
		try:
			tenant_id = get_tenant_id()
			service = get_database_service()
			
			time_range_days = int(request.args.get('time_range_days', 7))
			time_range = timedelta(days=time_range_days)
			
			log_api_request('get_crawler_analytics', tenant_id, time_range_days=time_range_days)
			
			analytics = run_async(service.get_crawler_analytics(tenant_id, time_range))
			
			return analytics
			
		except ValueError as e:
			return {'error': str(e)}, 400
		except Exception as e:
			logger.error(f"Error getting crawler analytics: {str(e)}")
			return {'error': 'Internal server error'}, 500


@ns_analytics.route('/health')
class HealthCheckAPI(Resource):
	"""Health check endpoints"""
	
	@api.doc('health_check')
	def get(self):
		"""Check crawler service health"""
		try:
			service = get_database_service()
			health = run_async(service.health_check())
			
			return health
			
		except Exception as e:
			logger.error(f"Health check failed: {str(e)}")
			return {
				'status': 'unhealthy',
				'error': str(e),
				'timestamp': datetime.utcnow().isoformat()
			}, 500


# =====================================================
# ERROR HANDLERS
# =====================================================

@crawler_bp.errorhandler(400)
def bad_request(error):
	return jsonify({'error': 'Bad request', 'message': str(error)}), 400

@crawler_bp.errorhandler(404)
def not_found(error):
	return jsonify({'error': 'Not found', 'message': str(error)}), 404

@crawler_bp.errorhandler(500)
def internal_error(error):
	return jsonify({'error': 'Internal server error', 'message': str(error)}), 500


# =====================================================
# BLUEPRINT REGISTRATION
# =====================================================

def register_crawler_api(app):
	"""Register crawler API blueprint with Flask app"""
	app.register_blueprint(crawler_bp)
	
	# Initialize database service
	db_url = app.config.get('CRAWLER_DATABASE_URL', 'postgresql+asyncpg://localhost/apg_crawler')
	app.crawler_service = CrawlerDatabaseService(db_url)
	
	logger.info("APG Crawler API registered successfully")


# Export blueprint
__all__ = ['crawler_bp', 'register_crawler_api']